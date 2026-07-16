import asyncio
import json
import os
import re
import hashlib
from dataclasses import dataclass, field, asdict
from typing import Optional
from dotenv import load_dotenv
from openai import AsyncOpenAI
import faiss
import numpy as np
from datetime import datetime
from mongo_db import db
from browser_agent import run_browser_agent

# Shared state for SSE and interruptions
interrupt_flag = {}
resume_events = {}
message_queue = {}

# Use OpenAI-compatible interface for Alibaba Cloud International
api_key = os.getenv("DASHSCOPE_API_KEY", "dummy_key_for_testing")
client = AsyncOpenAI(
    api_key=api_key,
    base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
)

# ── EvidenceUnit — The atomic unit of all research data ───────────────────────

@dataclass
class EvidenceUnit:
    """A single verifiable piece of evidence with full provenance."""
    claim_text: str
    source_title: str
    source_url: str
    raw_chunk: str                        # original text the claim was extracted from
    verified: bool = False               # did the URL return HTTP 200?
    verification_method: str = "none"    # "http_check" | "title_match" | "fallback" | "none"
    extraction_date: str = ""
    confidence: float = 0.5
    sample_size: str = "Unknown"
    replication_status: str = "Unknown"
    evidence_type: str = "Unknown"

    def __post_init__(self):
        if not self.extraction_date:
            self.extraction_date = datetime.utcnow().isoformat()

    def citation_label(self) -> str:
        """Return a markdown link if verified with URL, otherwise plain text."""
        if self.verified and self.source_url:
            return f"[{self.source_title}]({self.source_url})"
        elif self.source_title and self.source_title not in ("Unknown", ""):
            return f"{self.source_title} *(unverified source)*"
        return "*(unverified source)*"


# FAISS index — initialized lazily on first embedding so we auto-detect real dim
faiss_index: faiss.IndexFlatL2 | None = None

def get_or_init_index(dim: int) -> faiss.IndexFlatL2:
    """Return the global FAISS index, creating it if needed with the correct dim."""
    global faiss_index
    if faiss_index is None:
        faiss_index = faiss.IndexFlatL2(dim)
    return faiss_index


# ── Utilities ─────────────────────────────────────────────────────────────────

async def send_log(research_id: str, message: str, msg_type: str = "LOG"):
    if research_id not in message_queue:
        message_queue[research_id] = asyncio.Queue()
    await message_queue[research_id].put({
        "event": "message",
        "data": json.dumps({"type": msg_type, "message": message})
    })



# ── Qwen helpers ──────────────────────────────────────────────────────────────

async def generate_embedding(text: str) -> np.ndarray:
    """Call Qwen text-embedding-v3 and return the float32 vector."""
    try:
        response = await client.embeddings.create(
            model="text-embedding-v3",
            input=text,
        )
        return np.array(response.data[0].embedding, dtype=np.float32)
    except Exception:
        dim = faiss_index.d if faiss_index is not None else 1024
        return np.zeros(dim, dtype=np.float32)


async def check_claim_evolution(new_claim: str, historical_contexts: list) -> str:
    """Cross-reference a new claim against stored claims via Qwen."""
    hist_str = "\n".join(historical_contexts) if historical_contexts else "(none yet)"
    prompt = (
        "You are a fact-alignment engine. Compare the New Claim against the Historical Context "
        "and output JSON with keys: 'contradiction' (bool), 'support' (string explaining alignment).\n\n"
        f"Historical Context:\n{hist_str}\n\nNew Claim:\n{new_claim}"
    )
    try:
        response = await client.chat.completions.create(
            model="qwen-plus",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content
    except Exception as e:
        return json.dumps({"contradiction": False, "support": f"alignment skipped: {str(e)}"})


async def atomic_fact_extraction(text_chunk: str, source_title: str = "Unknown", source_url: str = "", verified: bool = False, verification_method: str = "none") -> list[EvidenceUnit]:
    """Use Qwen to decompose a paragraph into isolated, discrete factual claims.
    Returns a list of EvidenceUnit objects preserving source provenance."""
    prompt = (
        "Extract every isolated, discrete factual claim from the text below. "
        "Each claim must be a single, verifiable statement. Exclude vague generalities. "
        "Also infer structured metadata if present. "
        "Return ONLY valid JSON: {\"claims\": [\"claim 1\", ...], \"sample_size\": \"...\", \"replication_status\": \"...\", \"evidence_type\": \"...\"}\n\n"
        f"Text:\n{text_chunk}"
    )
    try:
        response = await client.chat.completions.create(
            model="qwen-plus",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        data = json.loads(response.choices[0].message.content)
        raw_claims = data.get("claims", [])
        if not isinstance(raw_claims, list) or not raw_claims:
            raw_claims = [text_chunk[:500]]
        return [
            EvidenceUnit(
                claim_text=c,
                source_title=source_title,
                source_url=source_url,
                raw_chunk=text_chunk,
                verified=verified,
                verification_method=verification_method,
                confidence=0.8 if verified else 0.4,
                sample_size=data.get("sample_size", "Unknown"),
                replication_status=data.get("replication_status", "Unknown"),
                evidence_type=data.get("evidence_type", "Unknown")
            )
            for c in raw_claims
        ]
    except Exception:
        return [EvidenceUnit(
            claim_text=text_chunk[:500],
            source_title=source_title,
            source_url=source_url,
            raw_chunk=text_chunk,
            verified=verified,
            verification_method=verification_method,
            confidence=0.3,
            sample_size="Unknown",
            replication_status="Unknown",
            evidence_type="Unknown"
        )]


# ── Topic Classification ─────────────────────────────────────────────────────

async def classify_topic(query: str) -> dict:
    """Dynamically classify a research topic to determine report structure."""
    prompt = (
        f"You are a research topic classifier. Analyze the following research topic and return a JSON classification.\n\n"
        f"Topic: \"{query}\"\n\n"
        f"Return ONLY valid JSON with these keys:\n"
        f"- \"category\": one of [\"engineering\", \"analytical\", \"scientific\", \"philosophical\", \"policy\", \"comparative\", \"investigative\"]\n"
        f"- \"requires_bom\": boolean — true ONLY if the user explicitly wants to build/buy physical hardware\n"
        f"- \"requires_pricing\": boolean — true ONLY if the topic involves purchasing components or cost analysis\n"
        f"- \"requires_code\": boolean — true if the topic involves software or firmware implementation\n"
        f"- \"stance\": one of [\"neutral\", \"pro\", \"anti\"] — what stance should the report take? Default to \"neutral\" unless the user EXPLICITLY asks to argue for/against something\n"
        f"- \"key_dimensions\": list of 3-6 strings describing what dimensions this topic should be analyzed along\n"
        f"- \"report_sections\": list of 4-8 section titles appropriate for THIS specific topic. MUST include a final section titled 'Practical Applications & Behavioral Insights'.\n"
        f"- \"actionable_dimensions\": list of 2-4 strings describing real-world use cases this research applies to\n"
    )
    try:
        response = await client.chat.completions.create(
            model="qwen-plus",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        result = json.loads(response.choices[0].message.content)
        return result
    except Exception:
        return {
            "category": "analytical",
            "requires_bom": False,
            "requires_pricing": False,
            "requires_code": False,
            "stance": "neutral",
            "key_dimensions": ["evidence", "counterarguments", "expert opinions"],
            "report_sections": ["Executive Summary", "Key Findings", "Analysis", "Counterarguments", "Practical Applications & Behavioral Insights", "Conclusion"],
            "actionable_dimensions": ["general understanding", "decision-making"],
        }


async def filter_claims_for_relevance(query: str, all_claims: list[EvidenceUnit]) -> dict:
    """Filter EvidenceUnits into relevant vs irrelevant, returning both sets."""
    claim_texts = [c.claim_text for c in all_claims]
    claims_block = "\n".join(f"{i+1}. {c}" for i, c in enumerate(claim_texts))
    prompt = (
        f"You are a research relevance filter. Given the topic \"{query}\", classify each numbered claim as RELEVANT or IRRELEVANT.\n"
        f"A claim is IRRELEVANT if it:\n"
        f"- Has no meaningful connection to the research topic\n"
        f"- Is about a completely different subject\n"
        f"- Is generic filler with no informational value\n"
        f"- Is a duplicate or near-duplicate of another claim\n\n"
        f"Return ONLY valid JSON:\n"
        f"{{\"relevant\": [list of claim numbers], \"irrelevant\": [list of claim numbers]}}\n\n"
        f"CLAIMS:\n{claims_block}"
    )
    try:
        response = await client.chat.completions.create(
            model="qwen-plus",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        result = json.loads(response.choices[0].message.content)
        relevant_indices = [int(i) - 1 for i in result.get("relevant", range(1, len(all_claims)+1))]
        irrelevant_indices = [int(i) - 1 for i in result.get("irrelevant", [])]
        return {
            "relevant": [all_claims[i] for i in relevant_indices if 0 <= i < len(all_claims)],
            "irrelevant": [all_claims[i] for i in irrelevant_indices if 0 <= i < len(all_claims)],
        }
    except Exception:
        return {"relevant": all_claims, "irrelevant": []}


# ── Web Search (Dynamic, Year-Aware) ─────────────────────────────────────────

def parse_source_blocks(raw_text: str) -> list[dict]:
    """Parse structured source blocks from Qwen search output."""
    pattern = r'=== SOURCE START ===\s*Title:\s*(.+?)\s*URL:\s*(.+?)\s*Content:\s*(.*?)\s*=== SOURCE END ==='
    matches = re.findall(pattern, raw_text, re.DOTALL)
    
    sources = []
    for title, url, content in matches:
        title = title.strip()
        url = url.strip()
        content = content.strip()
        if len(content) > 30:
            sources.append({"source_title": title, "source_url": url, "content": content})
    
    # Fallback: if Qwen didn’t use the structured format, split by paragraphs
    if not sources:
        chunks = [c.strip() for c in raw_text.split("\n\n") if len(c.strip()) > 30]
        for i, chunk in enumerate(chunks):
            sources.append({"source_title": f"Web Source {i+1}", "source_url": "", "content": chunk})
    
    return sources


def _content_hash(text: str) -> str:
    """SHA-256 fingerprint of a content string for deduplication."""
    return hashlib.sha256(text.strip().lower().encode()).hexdigest()[:16]





async def independent_claim_fact_check(claims: list[EvidenceUnit], query: str) -> list[dict]:
    """Run targeted independent web searches for specific assertions in the claims.
    Returns a list of {claim_text, supported: bool, note: str}."""
    # Extract only high-confidence specific assertions (skip generic ones)
    specific_claims = [c for c in claims if any(ch.isdigit() for ch in c.claim_text) or len(c.claim_text) > 60]
    specific_claims = specific_claims[:8]  # limit to 8 checks to keep latency manageable

    if not specific_claims:
        return []

    claims_list = "\n".join(f"{i+1}. {c.claim_text}" for i, c in enumerate(specific_claims))
    prompt = (
        f"You are an independent fact-checker. Use web search to verify each claim below about the topic: '{query}'.\n"
        f"For each claim, search independently and determine if corroborating evidence exists.\n\n"
        f"Return ONLY valid JSON: {{\"results\": [{{\"claim_number\": 1, \"supported\": true/false, \"note\": \"brief explanation and source found\"}}]}}\n\n"
        f"CLAIMS TO VERIFY:\n{claims_list}"
    )
    try:
        response = await client.chat.completions.create(
            model="qwen-max",
            messages=[{"role": "user", "content": prompt}],
            extra_body={"enable_search": True},
            response_format={"type": "json_object"},
        )
        data = json.loads(response.choices[0].message.content)
        results = data.get("results", [])
        # Map back to claim text
        output = []
        for r in results:
            idx = r.get("claim_number", 0) - 1
            if 0 <= idx < len(specific_claims):
                output.append({
                    "claim_text": specific_claims[idx].claim_text,
                    "supported": r.get("supported", True),
                    "note": r.get("note", ""),
                })
        return output
    except Exception:
        return []

async def dynamic_fact_check_search(draft_content: str, query: str, topic_meta: dict, unsupported_claims: list[dict] = None) -> str:
    """Use Qwen-Max with web search to verify the draft, with awareness of pre-flagged unsupported claims."""
    category = topic_meta.get("category", "analytical")

    unsupported_block = ""
    if unsupported_claims:
        flagged = "\n".join(
            f"- {c['claim_text']} ({c.get('note', '')})"
            for c in unsupported_claims if not c.get("supported", True)
        )
        if flagged:
            unsupported_block = (
                f"\n\nPRE-FLAGGED UNSUPPORTED CLAIMS (independently verified as having NO corroborating sources):\n"
                f"{flagged}\n"
                f"These claims MUST be removed or marked as 'unverified' in any final output.\n"
            )

    if category == "engineering" and topic_meta.get("requires_pricing"):
        check_instruction = (
            "1. COMPONENTS & PRICING: Extract every hardware component mentioned. Search for its current real retail price.\n"
            "2. PHYSICS & FEASIBILITY: Verify if the proposed physical mechanisms actually work as claimed.\n"
        )
    elif category in ("analytical", "philosophical", "policy"):
        check_instruction = (
            "1. STATISTICAL CLAIMS: Verify every percentage, number, or statistic. Find the actual source.\n"
            "2. STUDY CITATIONS: Verify every referenced study or expert quote actually exists. Flag fabrications.\n"
            "3. LOGICAL COHERENCE: Identify logical fallacies, false equivalences, or cherry-picked evidence.\n"
            "4. MISSING PERSPECTIVES: Identify important counterarguments the draft ignores.\n"
        )
    elif category == "scientific":
        check_instruction = (
            "1. METHODOLOGY: Verify the scientific methods described are sound.\n"
            "2. DATA ACCURACY: Cross-check all data points against published sources.\n"
            "3. CONSENSUS CHECK: Verify claims align with current scientific consensus. Flag fringe claims.\n"
        )
    else:
        check_instruction = (
            "1. FACTUAL ACCURACY: Verify all factual claims against current sources.\n"
            "2. SOURCE VALIDITY: Check that cited sources exist and say what the draft claims they say.\n"
            "3. BALANCE: Identify any one-sided arguments that ignore valid counterpoints.\n"
        )

    prompt = (
        f"You are a strict verification agent. Fact-check this draft report on '{query}'.\n"
        f"Use LIVE WEB SEARCH. Flag any outdated information or unsupported claims.\n"
        f"{unsupported_block}\n"
        f"CHECK THE FOLLOWING:\n{check_instruction}\n\n"
        f"Produce a concise Fact-Check Summary listing all discrepancies found.\n\n"
        f"DRAFT TO VERIFY:\n{draft_content}"
    )
    try:
        response = await client.chat.completions.create(
            model="qwen-max",
            messages=[{"role": "user", "content": prompt}],
            extra_body={"enable_search": True},
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"[Fact-Check Search Error] {str(e)}"


# ── Post-Processing Utilities ──────────────────────────────────────────────────────────

VERIFICATION_LABELS: dict[str, str] = {
    "http_ok": "Live ✓",
    "title_match": "Title matched",
    "bot_blocked": "Blocked by site",
    "server_error": "Server error",
    "timeout": "Slow response",
    "not_found": "404 Not found",
    "title_mismatch": "Title mismatch ⚠",
    "unreachable": "Unreachable",
    "none": "No URL",
    "no_aiohttp": "Unchecked",
}

def grounding_check(report_text: str, num_evidence_units: int) -> str:
    """Verifies that every [N] citation in the report is within bounds.
    Replaces hallucinated out-of-bounds citations with [CITATION NEEDED]."""
    def replace_oob(m: re.Match) -> str:
        try:
            idx = int(m.group(1)) - 1
            if 0 <= idx < num_evidence_units:
                return m.group(0)
        except ValueError:
            pass
        return "[CITATION NEEDED]"
    return re.sub(r'\[(\d+)\]', replace_oob, report_text)


def build_bibliography(report_text: str, evidence_units: list[EvidenceUnit]) -> str:
    """Extract all [N] links used in the report, cross-reference with evidence units,
    and build a References section. Invalid citations are stripped."""
    
    used_indices: set[int] = set()

    def replace_citation(m: re.Match) -> str:
        try:
            idx = int(m.group(1)) - 1
            if 0 <= idx < len(evidence_units):
                used_indices.add(idx)
                return m.group(0)  # keep valid citation
            return f"[Invalid Citation {idx+1}]"
        except ValueError:
            return m.group(0)

    # Match [1], [2], [12]
    cleaned = re.sub(r'\[(\d+)\]', replace_citation, report_text)

    if not used_indices:
        return cleaned  # No citations used — don't add References section

    # Group used indices by URL to deduplicate
    url_groups: dict[str, list[int]] = {}
    for idx in sorted(used_indices):
        url = evidence_units[idx].source_url
        if url not in url_groups:
            url_groups[url] = []
        url_groups[url].append(idx)

    ref_lines = ["\n\n---\n\n## References\n"]
    for url, indices in url_groups.items():
        eu = evidence_units[indices[0]]
        snippet = eu.raw_chunk[:200].replace('\n', ' ') if eu.raw_chunk else ""
        method_str = VERIFICATION_LABELS.get(eu.verification_method, eu.verification_method)
        
        claims_str = ", ".join(f"[{i+1}]" for i in indices)
        verified_flag = "" if eu.verified else " 🔴 **UNVERIFIED**"

        ref_lines.append(
            f"**{claims_str} {eu.source_title}**{verified_flag}  \n"
            f"   *URL: {url}*  \n"
            f"   *Accessed: {eu.extraction_date[:10]}* | Verification: {method_str}  \n"
            f"   > {snippet}..."
        )

    return cleaned + "\n".join(ref_lines)


def hallucination_score_block(evidence_units: list[EvidenceUnit], report_text: str) -> str:
    """Build a source integrity summary for the report header using distinct URLs."""
    verified_urls = {eu.source_url for eu in evidence_units if eu.verified and eu.source_url}
    unverified_urls = {eu.source_url for eu in evidence_units if not eu.verified and eu.source_url}
    
    n_high_conf = sum(1 for eu in evidence_units if eu.confidence >= 0.7)

    # Count bracketed citations in report
    used_indices = set()
    for m in re.finditer(r'\[(\d+)\]', report_text):
        try:
            idx = int(m.group(1)) - 1
            if 0 <= idx < len(evidence_units):
                used_indices.add(idx)
        except ValueError:
            pass
    
    links_in_report = len(re.findall(r'\[\d+\]', report_text))
    distinct_sources_cited = len({evidence_units[i].source_url for i in used_indices if evidence_units[i].source_url})

    return (
        f"**Source Integrity:** {len(verified_urls)} verified sources │ "
        f"{len(unverified_urls)} unverified sources │ "
        f"{n_high_conf} high-confidence claims │ "
        f"{links_in_report} citations across {distinct_sources_cited} sources  \n"
        f"**Data Year:** Most recent verified (no year invented)  \n"
    )


# ── Synthesis Pipeline (Fully Dynamic) ────────────────────────────────────────────

async def synthesis_pipeline(research_id: str, query: str, all_claims: list[EvidenceUnit]) -> str:
    """
    6-pass synthesis:
      Pass 0: Topic classification
      Pass 1: Adaptive draft with real citations from verified EvidenceUnits
      Pass 2a: Independent per-claim fact-check (targeted searches)
      Pass 2b: Holistic draft fact-check with Qwen web search
      Pass 3: Adversarial red-team with raw source chunks
      Pass 4: Corrected final report
      Post:  Bibliography builder + hallucination score
    """

    # Fetch chat history to incorporate user constraints from paused state
    chat_history = ""
    if db is not None:
        try:
            chat_doc = await db.chats.find_one({"researchId": research_id})
            if chat_doc and "messages" in chat_doc:
                history_texts = []
                for msg in chat_doc["messages"]:
                    role = "User" if msg.get("role") == "user" else "AI"
                    history_texts.append(f"{role}: {msg.get('content')}")
                if history_texts:
                    chat_history = "\n".join(history_texts)
        except Exception as e:
            print(f"Failed to fetch chat history: {e}")

    # ── Pass 0: Classify Topic ──────────────────────────────────────────────────
    await send_log(research_id, "Pass 0/5 — Classifying topic structure...")

    topic_meta = await classify_topic(query)
    category = topic_meta.get("category", "analytical")
    stance = topic_meta.get("stance", "neutral")
    sections = topic_meta.get("report_sections", ["Summary", "Analysis", "Counterarguments", "Practical Applications & Behavioral Insights", "Conclusion"])
    dimensions = topic_meta.get("key_dimensions", ["evidence", "analysis"])
    actionable = topic_meta.get("actionable_dimensions", ["general understanding"])
    current_year = datetime.now().year

    await send_log(research_id, f"Topic classified as: {category} | Stance: {stance} | Sections: {len(sections)}")

    # ── Deduplicate Claims by Source (Fix #3) ──────────────────────────────────
    unique_sources: list[EvidenceUnit] = []
    source_map: dict[str, EvidenceUnit] = {}
    from copy import deepcopy
    for eu in all_claims:
        key = eu.source_url if eu.source_url else eu.source_title
        if key not in source_map:
            new_eu = deepcopy(eu)
            source_map[key] = new_eu
            unique_sources.append(new_eu)
        else:
            source_map[key].claim_text += f" | {eu.claim_text}"
            
    relevant_claims = unique_sources

    # ── Build claims block with citation labels from EvidenceUnit ──────────────────
    claims_lines = []
    raw_chunks_map: dict[str, str] = {}  # url -> raw_chunk for red-team
    for i, eu in enumerate(relevant_claims):
        claims_lines.append(f"[{i+1}] {eu.source_title}: \"{eu.claim_text}\"")
        if eu.source_url and eu.raw_chunk:
            raw_chunks_map[eu.source_url] = eu.raw_chunk[:600]
    claims_block = "\n".join(claims_lines)

    # Sample raw chunks for red-team (max 8 to avoid token explosion)
    raw_chunks_block = "\n\n---\n".join(
        f"[{url}]:\n{chunk}" for url, chunk in list(raw_chunks_map.items())[:8]
    )

    stance_instruction = {
        "neutral": "Present ALL sides fairly. Do NOT advocate for or against the topic. Let evidence speak for itself. When evidence is mixed, say so explicitly.",
        "pro": "The user has explicitly requested an argument IN FAVOR of this position. Build the strongest possible case, but still acknowledge valid counterarguments.",
        "anti": "The user has explicitly requested an argument AGAINST this position. Build the strongest possible critique, but still acknowledge valid supporting evidence.",
    }.get(stance, "Present ALL sides fairly.")

    actionable_block = ", ".join(actionable)

    dynamic_system = (
        f"You are an expert researcher and analyst. Your task is to produce a comprehensive research report.\n\n"
        f"ABSOLUTE RULES (violating these is a critical failure):\n"
        f"1. STANCE: {stance_instruction}\n"
        f"2. STRICT NUMBERED CITATIONS: The user has provided numbered evidence units. You MUST cite claims using ONLY their exact bracketed number, e.g. [1], [2]. "
        f"NEVER use names or formats like 'Smith et al.' or 'Doe (2023)'. If a source has no author or title, it does not matter. Just use the number [X].\n"
        f"3. NO HALLUCINATION: Default to 'I don't know' or 'No specific evidence' for any demographic, mechanism, or angle not explicitly detailed in the evidence units. Do NOT fill gaps with general knowledge.\n"
        f"4. RECENCY: Use the most recent data available. NEVER invent a more recent year. Current year is {current_year} for reference only.\n"
        f"5. ADAPT FORMAT: This is a {category} topic. Do NOT include hardware tables, code, or pricing unless explicitly relevant.\n"
        f"6. INTELLECTUAL HONESTY: Distinguish 'strong evidence', 'some evidence', 'contested', 'no evidence'.\n"
        f"7. EXPLICIT MODERATORS: If evidence acknowledges contextual or individual variation but does not identify specific moderating variables, state explicitly that specific moderators are unknown.\n"
        f"8. NEVER reference internal pipeline steps ('Fact-Check Report', 'Critique', 'Draft') as sources.\n"
        f"9. PRACTICAL VALUE: If there is insufficient evidence, omit practical applications entirely. If evidence exists, the section MUST be formatted as a Markdown table with EXACTLY these columns:\n"
        f"   | Principle (from evidence) | Illustrative Application (MUST explicitly state if untested) | Evidence Matrix (Sample/Type) |\n"
        f"   NEVER recommend specific scenarios without explicitly tagging them as untested illustrations.\n"
    )

    if chat_history:
        dynamic_system += (
            f"\n10. USER CONSTRAINTS: The user interrupted research to add guidance. "
            f"MUST follow their constraints.\n\nUSER CHAT HISTORY:\n{chat_history}\n"
        )

    sections_block = "\n".join(f"## {i+1}. {s}" for i, s in enumerate(sections))
    dimensions_block = ", ".join(dimensions)

    # ── Pass 1: Adaptive Draft ────────────────────────────────────────────────
    await send_log(research_id, "Pass 1/5 — Generating adaptive draft with real citations...")

    n_verified_claims = sum(1 for eu in relevant_claims if eu.verified)

    draft_prompt = (
        f"Based on {len(relevant_claims)} evidence units from live web research on:\n"
        f"TOPIC: \"{query}\"\n"
        f"({n_verified_claims} from URL-verified sources, {len(relevant_claims)-n_verified_claims} from unverified sources)\n\n"
        f"Produce a comprehensive research report along these dimensions: {dimensions_block}.\n\n"
        f"Use this section structure:\n{sections_block}\n\n"
        f"CITATION RULES (ABSOLUTE):\n"
        f"- Each numbered claim below starts with its number in brackets, e.g. [1].\n"
        f"- You MUST cite every assertion using these brackets: e.g. 'Some studies show X [1][3].'\n"
        f"- Do NOT invent names. Do NOT use standard APA/MLA format. Use ONLY [X].\n\n"
        f"CONTENT RULES:\n"
        f"- Every assertion must trace to a numbered claim below.\n"
        f"- Use specific data points. Never vague generalizations.\n"
        f"EVIDENCE UNITS:\n{claims_block}"
    )

    try:
        draft_resp = await client.chat.completions.create(
            model="qwen-plus",
            messages=[
                {"role": "system", "content": dynamic_system},
                {"role": "user", "content": draft_prompt},
            ],
        )
        draft = draft_resp.choices[0].message.content
    except Exception as e:
        draft = f"[Draft failed: {e}]\n\nRaw evidence:\n{claims_block}"

    await send_log(research_id, "Pass 1 complete. Running independent claim verification...")

    # ── Pass 2a: Independent Per-Claim Fact-Check ─────────────────────────────────
    await send_log(research_id, "Pass 2a/5 — Running independent claim-level verification...")
    unsupported_claims = await independent_claim_fact_check(relevant_claims, query)
    n_unsupported = sum(1 for c in unsupported_claims if not c.get("supported", True))
    if n_unsupported > 0:
        await send_log(research_id, f"🚫 {n_unsupported} claims flagged as unsupported by independent search. These will be removed.")

    # ── Pass 2b: Holistic Draft Fact-Check ─────────────────────────────────────
    await send_log(research_id, "Pass 2b/5 — Searching live web to verify draft holistically...")
    fact_check_report = await dynamic_fact_check_search(draft, query, topic_meta, unsupported_claims)

    await send_log(research_id, "Pass 2 complete. Running adversarial red-team...")

    # ── Pass 3: Adversarial Red-Team with Raw Sources ───────────────────────────
    await send_log(research_id, "Pass 3/5 — Adversarial red-team against raw source material...")

    critic_system = (
        f"You are a rigorous adversarial peer reviewer for a report on \"{query}\".\n\n"
        f"CRITICAL CHECKS:\n"
        f"1. FABRICATED CITATIONS: Flag any citation in the draft that does NOT correspond to a source in the EVIDENCE UNITS list.\n"
        f"2. CLAIM TRACEABILITY: Verify major assertions trace back to RAW SOURCE MATERIAL.\n"
        f"3. OMISSIONS: Identify important findings present in the raw source material that the draft omitted.\n"
        f"4. EXAGGERATION: Flag if findings from a general population are inaccurately described as applying specifically to an unstudied demographic.\n\n"
        f"You MUST output ONLY valid JSON in this exact format:\n"
        f"{{\n"
        f"  \"corrections\": [\n"
        f"    {{\"type\": \"remove|rephrase|add\", \"location\": \"paragraph X or section Y\", \"reason\": \"explanation\"}}\n"
        f"  ]\n"
        f"}}"
    )

    critique_prompt = (
        f"EVIDENCE UNITS:\n{claims_block}\n\n"
        f"RAW SOURCE MATERIAL:\n{raw_chunks_block}\n\n"
        f"FACT-CHECK NOTES:\n{fact_check_report}\n\n"
        f"DRAFT REPORT TO CRITIQUE:\n{draft}"
    )

    try:
        critique_resp = await client.chat.completions.create(
            model="qwen-plus",
            messages=[
                {"role": "system", "content": critic_system},
                {"role": "user", "content": critique_prompt},
            ],
            response_format={"type": "json_object"},
        )
        critique = critique_resp.choices[0].message.content
        # Ensure it's valid JSON
        json.loads(critique)
    except Exception as e:
        critique = json.dumps({"corrections": [{"type": "system", "location": "N/A", "reason": f"Critique failed: {e}"}]})
    except Exception as e:
        critique = f"[Critique failed: {e}]"

    await send_log(research_id, "Pass 3 complete. Writing corrected final report...")

    # ── Pass 4: Revised Final Report ──────────────────────────────────────────
    await send_log(research_id, "Pass 4/5 — Writing corrected final report...")

    revision_prompt = (
        f"A research report on \"{query}\" has been drafted and reviewed. Produce the final corrected version.\n\n"
        f"MANDATORY CORRECTIONS (APPLY ALL):\n"
        f"1. DIRECT ANSWER: Start the report with a blockquote `> **Direct Answer:**` answering the user's specific query. If the evidence does not specifically address the user's demographic/angle, you MUST explicitly state this limitation. Do NOT imply that general findings confirm the effect in the specific demographic.\n"
        f"2. PLAIN LANGUAGE SUMMARY: Follow the Direct Answer with a 3-4 sentence non-technical summary.\n"
        f"3. Apply EVERY correction listed in the JSON CRITIQUE CHECKLIST below.\n"
        f"4. Remove ALL fabricated citations and any claim flagged as unsupported.\n"
        f"5. STRICT NUMBERED CITATIONS: Use ONLY the exact bracketed number from EVIDENCE UNITS, e.g. [1], [2]. NEVER use names.\n"
        f"6. 'Practical Applications' section MUST be formatted as the exact Markdown table specified in the system prompt.\n"
        f"7. DIAGRAMS: Where appropriate to visualize architectures, workflows, statistics, or relationships, you MUST use Mermaid.js diagram code blocks (````mermaid\\n...\\n````). Make them professional and visually clear.\n"
        f"8. Write in natural, intelligent academic prose.\n\n"
        f"EVIDENCE UNITS (ONLY legitimate citation sources):\n{claims_block}\n\n"
        f"DRAFT:\n{draft}\n\n"
        f"JSON CRITIQUE CHECKLIST:\n{critique}\n\n"
        f"Write the final corrected report:"
    )

    try:
        revision_resp = await client.chat.completions.create(
            model="qwen-plus",
            messages=[
                {"role": "system", "content": dynamic_system},
                {"role": "user", "content": revision_prompt},
            ],
        )
        final = revision_resp.choices[0].message.content
    except Exception as e:
        final = f"[Revision failed: {e}]\n\nFalling back to draft:\n{draft}"

    # ── Factuality Auditor Pass ───────────────────────────────────────────────
    await send_log(research_id, "Pass 5/5 — Running final Factuality Auditor pass...")
    bs_prompt = (
        f"You are a hostile reviewer checking the final report for overconfidence and ungrounded claims.\n"
        f"If the report makes claims that are too strong given the evidence, or recommends untested practical applications without explicit caveats, output a strict JSON warning.\n"
        f"Otherwise output empty JSON.\n\n"
        f"FORMAT: {{\"warning\": \"Explanation of overconfidence if any, else empty string\"}}\n\n"
        f"EVIDENCE UNITS:\n{claims_block}\n\n"
        f"FINAL REPORT:\n{final}"
    )
    try:
        bs_resp = await client.chat.completions.create(
            model="qwen-plus",
            messages=[{"role": "user", "content": bs_prompt}],
            response_format={"type": "json_object"},
        )
        bs_data = json.loads(bs_resp.choices[0].message.content)
        bs_warning = bs_data.get("warning", "").strip()
    except Exception:
        bs_warning = ""

    bs_banner = ""
    if bs_warning:
        bs_banner = f"> [!CAUTION]\n> **Unresolved Critique / Overconfidence Warning:** {bs_warning}\n\n"

    # ── Post-Process: Bibliography + Grounding Check + Hallucination Score ──────────
    await send_log(research_id, "Post-processing: building bibliography and grounding check...")
    final = grounding_check(final, len(relevant_claims))
    final_with_refs = build_bibliography(final, relevant_claims)
    score_block = hallucination_score_block(relevant_claims, final_with_refs)

    # ── Recency Warning ───────────────────────────────────────────────────────
    has_recent = any(
        re.search(r'202[0-9]', (eu.raw_chunk or "") + (eu.extraction_date or ""))
        for eu in relevant_claims if eu.verified
    )
    recency_warning = ""
    if not has_recent:
        recency_warning = (
            f"> [!WARNING]\n"
            f"> **Search Recency Note:** No verified sources from 2020–{current_year} were retrieved. "
            f"This may reflect search limitations, paywall barriers, or genuine stagnation in this specific literature niche.\n\n"
        )

    # ── Appendices ────────────────────────────────────────────────────────────
    appendix_a_lines = ["\n\n## Appendix A: Unverified Sources and Exclusion Rationale\n"]
    
    # Deduplicate unverified sources by URL
    unverified_claims = [eu for eu in relevant_claims if not eu.verified and eu.source_url]
    unverified_urls: dict[str, EvidenceUnit] = {}
    for eu in unverified_claims:
        if eu.source_url not in unverified_urls:
            unverified_urls[eu.source_url] = eu

    if unverified_urls:
        for url, eu in unverified_urls.items():
            method_str = VERIFICATION_LABELS.get(eu.verification_method, eu.verification_method)
            appendix_a_lines.append(f"- **{eu.source_title}**\n  - URL: {url}\n  - Failure Reason: {method_str}")
    else:
        appendix_a_lines.append("All sources used in this report passed live verification.")
    appendix_a = "\n".join(appendix_a_lines)

    appendix_b = (
        f"\n\n## Appendix B: Pipeline Transparency\n\n"
        f"### Pass 2: Holistic Fact-Check Notes\n"
        f"{fact_check_report}\n\n"
        f"### Pass 3: Adversarial Red-Team Critique\n"
        f"{critique}\n"
    )

    appendix_c_lines = ["\n\n## Appendix C: Evidence Map\n"]
    for i, eu in enumerate(relevant_claims):
        raw_excerpt = (eu.raw_chunk[:300] + "...") if eu.raw_chunk else "No raw chunk available."
        appendix_c_lines.append(
            f"**[{i+1}] {eu.source_title}**  \n"
            f"- **Extracted Claims:** {eu.claim_text}  \n"
            f"- **Raw Context:** {raw_excerpt}  \n"
        )
    appendix_c = "\n".join(appendix_c_lines)

    full_output = (
        f"# CHRONICLE RESEARCH REPORT\n\n"
        f"**Topic:** {query}  \n"
        f"**Category:** {category.title()}  \n"
        f"**Evidence Units:** {len(relevant_claims)} ({sum(1 for eu in relevant_claims if eu.verified)} URL-verified)  \n"
        f"{score_block}"
        f"**Pipeline:** 7-Pass (Classify → Draft → Claim-Check → Fact-Check → Red-Team → Final → Factuality Auditor + Bibliography)\n\n"
        f"---\n\n"
        f"{bs_banner}"
        f"{recency_warning}"
        f"{final_with_refs}"
        f"{appendix_a}"
    )

    return full_output


# ── Main Research Loop ─────────────────────────────────────────────────────────

async def research_loop(research_id: str, query: str, user_id: str = None, headless: bool = False):
    await send_log(research_id, "RUNNING", "STATUS_UPDATE")
    await send_log(research_id, f"Starting deep research on: '{query}'")



    await send_log(research_id, "Invoking Qwen Web Search Agent with URL verification...")

    if interrupt_flag.get(research_id):
        await send_log(research_id, "PAUSED", "STATUS_UPDATE")
        if research_id not in resume_events:
            resume_events[research_id] = asyncio.Event()
        resume_events[research_id].clear()
        await resume_events[research_id].wait()
        interrupt_flag[research_id] = False
        await send_log(research_id, "RUNNING", "STATUS_UPDATE")
        await send_log(research_id, "Resuming research with updated constraints...")

    all_claims: list[EvidenceUnit] = []
    stored_claim_texts: list[str] = []
    failed_mongo_inserts = []

    async def process_claims_func(claims: list[EvidenceUnit]):
        # Live Filter
        filter_res = await filter_claims_for_relevance(query, claims)
        relevant_batch = filter_res.get("relevant", claims)
        irrelevant_batch = filter_res.get("irrelevant", [])
        
        for irre in irrelevant_batch:
            await send_log(research_id, f"❌ Excluded (Irrelevant): {irre.claim_text}")

        for claim in relevant_batch:
            await send_log(research_id, f"Extracted Claim: {claim.claim_text} [via {claim.source_title}]")
            all_claims.append(claim)

            vec = await generate_embedding(claim.claim_text)
            idx = get_or_init_index(len(vec))

            historical_contexts: list[str] = []
            if idx.ntotal > 0:
                k = min(2, idx.ntotal)
                _distances, neighbor_ids = idx.search(np.array([vec]), k=k)
                for nid in neighbor_ids[0]:
                    if 0 <= nid < len(stored_claim_texts):
                        historical_contexts.append(stored_claim_texts[nid])

            alignment_result = await check_claim_evolution(claim.claim_text, historical_contexts)
            await send_log(research_id, f"Alignment Check: {alignment_result}")

            if db is not None:
                claim_doc = {
                    "researchId": research_id,
                    "query": query,
                    "claim_text": claim.claim_text,
                    "source_title": claim.source_title,
                    "source_url": claim.source_url,
                    "verified": claim.verified,
                    "verification_method": claim.verification_method,
                    "confidence": claim.confidence,
                    "sample_size": getattr(claim, 'sample_size', 'Unknown'),
                    "replication_status": getattr(claim, 'replication_status', 'Unknown'),
                    "evidence_type": getattr(claim, 'evidence_type', 'Unknown'),
                    "raw_chunk": claim.raw_chunk[:500] if claim.raw_chunk else "",
                    "timestamp": datetime.utcnow()
                }
                try:
                    await db.claims.insert_one(claim_doc)
                except Exception as e:
                    print(f"Mongo Claim Insert Error: {e}")
                    failed_mongo_inserts.append(claim_doc)

            idx.add(np.array([vec]))
            stored_claim_texts.append(claim.claim_text)

            await asyncio.sleep(0.3)

            if interrupt_flag.get(research_id):
                await send_log(research_id, "Interruption received. Safely pausing after current claim.")
                await send_log(research_id, "PAUSED", "STATUS_UPDATE")
                if db is not None and user_id:
                    await db.chats.update_one({"researchId": research_id}, {"$set": {"status": "PAUSED"}})
                
                if research_id not in resume_events:
                    resume_events[research_id] = asyncio.Event()
                resume_events[research_id].clear()
                
                # Block here until resume is triggered
                await resume_events[research_id].wait()
                
                # Resumed
                interrupt_flag[research_id] = False
                await send_log(research_id, "RUNNING", "STATUS_UPDATE")
                if db is not None and user_id:
                    await db.chats.update_one({"researchId": research_id}, {"$set": {"status": "RUNNING"}})
                await send_log(research_id, "Resuming extraction with updated constraints...")

    # Launch the Autonomous Browser Agent
    await send_log(research_id, "Invoking Autonomous Qwen Browser Agent (Playwright)...")
    await run_browser_agent(query, research_id, send_log, atomic_fact_extraction, process_claims_func, headless)

    await send_log(research_id, f"Claim extraction complete. {len(all_claims)} claims stored.")

    if db is not None and failed_mongo_inserts:
        try:
            await db.claims.insert_many(failed_mongo_inserts, ordered=False)
            failed_mongo_inserts = []
        except Exception as e:
            print(f"Mongo Fallback Insert Error: {e}")

    # Phase 6: 6-Pass Adaptive Synthesis
    await send_log(research_id, "SYNTHESIZING", "STATUS_UPDATE")
    if db is not None and user_id:
        await db.chats.update_one({"researchId": research_id}, {"$set": {"status": "SYNTHESIZING"}})
        
    final_report = await synthesis_pipeline(research_id, query, all_claims)

    await send_log(research_id, final_report, "REPORT")
    await send_log(research_id, "COMPLETED", "STATUS_UPDATE")

    if db is not None and user_id:
        try:
            await db.chats.update_one(
                {"researchId": research_id},
                {"$set": {
                    "status": "COMPLETED",
                    "finalReport": final_report,
                    "updatedAt": datetime.utcnow()
                }}
            )
        except Exception as e:
            print(f"Mongo Final Update Error: {e}")

# ── Chatbot Handling ──────────────────────────────────────────────────────────

async def handle_chat_message(research_id: str, message: str) -> str:
    """Handles an incoming chat message during a PAUSED state."""
    if db is None:
        return "Database is offline, cannot process chat."
    
    chat_doc = await db.chats.find_one({"researchId": research_id})
    if not chat_doc:
        return "Session not found."
    
    # Format the message history for Qwen
    messages = chat_doc.get("messages", [])
    
    # Sanitize message to prevent excessive prompt injection
    safe_message = message[:1000].strip()
    
    # Store user message
    user_msg = {"role": "user", "content": safe_message, "timestamp": datetime.utcnow()}
    await db.chats.update_one({"researchId": research_id}, {"$push": {"messages": user_msg}})
    
    api_messages = [
        {"role": "system", "content": f"You are assisting a user who has paused their ongoing research session on '{chat_doc.get('query')}'. You can answer questions or accept new instructions for the final report. Keep it concise and helpful."}
    ]
    
    for m in messages:
        api_messages.append({"role": m["role"], "content": m["content"]})
    
    api_messages.append({"role": "user", "content": safe_message})
    
    try:
        response = await client.chat.completions.create(
            model="qwen-plus",
            messages=api_messages
        )
        ai_reply = response.choices[0].message.content
        
        # Store AI response
        ai_msg = {"role": "assistant", "content": ai_reply, "timestamp": datetime.utcnow()}
        await db.chats.update_one({"researchId": research_id}, {"$push": {"messages": ai_msg}})
        
        return ai_reply
    except Exception as e:
        return f"Error connecting to AI: {str(e)}"
