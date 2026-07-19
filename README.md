# Chronicle AI — Deep Research Engine (Backend)

**Autonomous AI Research Pipeline | 7-Pass Fact-Checked Synthesis | Live Web Extraction**

The Chronicle AI engine is the core backend powering the Chronicle AI autonomous deep research platform. It orchestrates an end-to-end research pipeline — from live web browsing and evidence extraction to multi-pass adversarial synthesis — producing comprehensive, fact-checked research reports with full source provenance.

---

## Why This Matters

The internet is drowning in information, but reliable synthesis is scarce. Traditional AI chatbots generate fluent but ungrounded text — fabricating citations, inventing statistics, and presenting speculation as fact. Academic researchers spend weeks on literature reviews. Analysts spend days compiling reports from scattered sources.

Chronicle AI eliminates this gap. It conducts **autonomous live web research** using a browser-controlling AI agent, extracts **atomic evidence units** with full provenance, cross-references claims against a **vector knowledge base**, runs **independent fact-checking** against live search, and produces reports through a **7-pass adversarial synthesis pipeline** — ensuring every claim traces to a verified source and every conclusion survives adversarial scrutiny.

This is not a wrapper around a chatbot. This is a research engine.

---

## Core Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────┐
│   Autonomous Browser Agent       │  ← Qwen-Max + Playwright + DuckDuckGo
│   (Search → Navigate → Extract)  │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│   Atomic Fact Extraction         │  ← Qwen-Plus decomposes text into claims
│   + Relevance Filtering          │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│   FAISS Vector Index             │  ← Qwen text-embedding-v3 (1024-dim)
│   + Claim Alignment Check        │     Contradiction detection
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│   7-Pass Synthesis Pipeline      │
│   ├─ Pass 0: Topic Classification│
│   ├─ Pass 1: Adaptive Draft      │  ← Qwen-Plus with strict citation rules
│   ├─ Pass 2a: Per-Claim Verify   │  ← Qwen-Max with live web search
│   ├─ Pass 2b: Holistic Verify    │  ← Qwen-Max with live web search
│   ├─ Pass 3: Adversarial Review  │  ← Red-team against raw source material
│   ├─ Pass 4: Corrected Final     │  ← Apply all corrections
│   └─ Pass 5: Factuality Auditor  │  ← Flag overconfidence
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│   Post-Processing                │
│   ├─ Grounding Check             │  ← Strip hallucinated citations
│   ├─ Bibliography Builder        │  ← Cross-reference [N] tags with sources
│   └─ Hallucination Score         │  ← Source integrity metrics
└─────────────────────────────────┘
```

---

## Features in Detail

### 🤖 Autonomous Browser Agent
A Qwen-Max powered AI agent that **controls a live Chromium browser** via Playwright. It autonomously decides what to search, which pages to visit, and what text to extract — mimicking how a human researcher browses the web, but at machine speed.

**How it works:** The agent operates in a tool-calling loop with four tools: `search_google` (DuckDuckGo API), `navigate_and_read` (Playwright page extraction), `extract_evidence` (save relevant text), and `finish_research` (end session). Qwen-Max decides which tool to call at each step based on accumulated context, running up to 15 autonomous turns per session.

**How it was achieved:** We use the OpenAI-compatible chat completions API with Alibaba Cloud's DashScope endpoint. The agent receives structured tool definitions and returns tool calls in standard format. DuckDuckGo's API handles search without browser-based scraping (avoiding CAPTCHAs), while Playwright handles page reading with BeautifulSoup for clean text extraction.

### 🔬 Atomic Fact Extraction
Every piece of text the browser agent collects is decomposed into **isolated, discrete, verifiable factual claims** — the atomic unit of all research data in the system.

**How it works:** Raw text chunks are sent to Qwen-Plus with a structured extraction prompt. The model returns individual claims along with metadata (sample size, replication status, evidence type). Each claim is wrapped in an `EvidenceUnit` dataclass that preserves full provenance: source title, URL, raw chunk, verification status, confidence score, and extraction timestamp.

**How it was achieved:** The `atomic_fact_extraction` function enforces JSON-mode output from Qwen-Plus, parses the response into structured data, and assigns confidence scores based on verification status (0.8 for verified sources, 0.4 for unverified, 0.3 for extraction failures).

### 🧠 FAISS Vector Knowledge Base
Every extracted claim is embedded and indexed in a **FAISS vector database**, enabling semantic similarity search and contradiction detection across the growing knowledge base.

**How it works:** Claims are embedded using Qwen's `text-embedding-v3` model (1024-dimensional vectors). When a new claim arrives, its nearest neighbors in the FAISS index are retrieved and passed to Qwen-Plus for **alignment checking** — detecting whether the new claim contradicts, supports, or extends existing knowledge.

**How it was achieved:** The FAISS index is lazily initialized on the first embedding to auto-detect vector dimensions. The `check_claim_evolution` function formats historical context and new claims into a structured prompt, with Qwen-Plus returning JSON with contradiction flags and support explanations.

### 📋 Dynamic Topic Classification
Before synthesis begins, the engine **classifies the research topic** to determine the optimal report structure, stance, and verification strategy.

**How it works:** Qwen-Plus analyzes the query and returns a structured classification: category (engineering, analytical, scientific, philosophical, policy, comparative, investigative), whether it requires hardware BOMs, pricing, or code, the appropriate stance (neutral/pro/anti), key analysis dimensions, recommended report sections, and actionable use-case dimensions.

**How it was achieved:** The classification drives everything downstream — from which fact-checking instructions are used to how the report sections are structured. An engineering topic about building hardware gets component pricing verification; an analytical topic about social policy gets statistical claim verification and logical coherence checks.

### 🛡️ 7-Pass Adversarial Synthesis Pipeline
The heart of Chronicle AI's accuracy guarantee. Every report passes through **seven distinct processing passes**, each designed to catch a different category of error.

| Pass | Name | Model | Purpose |
|------|------|-------|---------|
| 0 | Topic Classification | Qwen-Plus | Determine report structure and verification strategy |
| 1 | Adaptive Draft | Qwen-Plus | Generate initial report with strict numbered citations |
| 2a | Per-Claim Verification | Qwen-Max + Web Search | Independent web search for specific assertions |
| 2b | Holistic Verification | Qwen-Max + Web Search | Full-draft fact-check against live web |
| 3 | Adversarial Red-Team | Qwen-Plus | Peer review against raw source material |
| 4 | Corrected Final | Qwen-Plus | Apply all corrections, add direct answer |
| 5 | Factuality Auditor | Qwen-Plus | Flag overconfidence and ungrounded recommendations |

**Post-processing** then runs grounding checks (replacing out-of-bounds citations with `[CITATION NEEDED]`), builds a cross-referenced bibliography, and computes a source integrity score.

### 📊 Relevance Filtering
Not every claim extracted from the web is relevant. The engine runs an AI-powered **relevance filter** that classifies each claim as relevant or irrelevant to the research topic, removing duplicates, off-topic noise, and generic filler.

**How it works:** All extracted claims are sent to Qwen-Plus with the research topic. The model classifies each numbered claim, and irrelevant claims are excluded from synthesis while being logged for transparency.

### ⏸️ Interactive Pause & Resume with Chat
Research sessions can be **paused mid-extraction**. While paused, users can chat with the AI to refine the research direction. When resumed, the chat history is injected into the synthesis pipeline's system prompt as user constraints.

**How it works:** An interrupt flag is checked after every claim extraction. When set, the engine awaits an asyncio Event. The `/api/chat` endpoint handles conversations during paused state using Qwen-Plus with session context. On resume, chat messages become hard constraints in the final report generation.

### 📧 Email Report Delivery
Completed reports can be emailed directly from the platform. Markdown is converted to styled HTML and sent via the Resend email API.

---

## How We Use Qwen and Why

Chronicle AI is built entirely on the **Qwen family of models** from Alibaba Cloud, accessed through the DashScope International API with OpenAI-compatible endpoints. We use three distinct models, each chosen for specific strengths:

| Model | Use Case | Why This Model |
|-------|----------|----------------|
| **Qwen-Max** | Browser agent reasoning, fact-checking with live web search | Strongest reasoning capability, supports `enable_search` for grounded verification |
| **Qwen-Plus** | Fact extraction, topic classification, draft generation, adversarial review, chat | Best balance of speed and quality for high-volume structured output tasks |
| **text-embedding-v3** | Claim vectorization for FAISS similarity search | High-quality 1024-dim embeddings optimized for semantic similarity |

### Why Qwen over alternatives:
- **Native web search integration** — `enable_search: true` gives Qwen-Max access to live search results during inference, critical for fact-checking
- **JSON mode** — `response_format: {"type": "json_object"}` ensures structured outputs for parsing (claims, classifications, critiques)
- **OpenAI-compatible API** — Drop-in compatibility via `AsyncOpenAI` client with custom `base_url`, minimizing integration complexity
- **Cost efficiency** — DashScope pricing enables running a 7-pass pipeline with multiple model calls per research session at viable cost

---

## How We Manage Accuracy

Accuracy is not a feature — it is the architecture. Every layer of the system is designed to prevent, detect, and correct errors:

### Prevention
- **Atomic decomposition** — Breaking text into isolated claims prevents context contamination
- **Source provenance** — Every claim carries its URL, title, raw chunk, and extraction timestamp
- **Strict citation rules** — The synthesis prompt explicitly forbids invented citations and enforces `[N]` numbered references only
- **No hallucination instruction** — The system prompt mandates "default to 'I don't know'" for any demographic, mechanism, or angle not in the evidence

### Detection
- **Independent per-claim verification** (Pass 2a) — Specific assertions are independently searched on the web
- **Holistic draft verification** (Pass 2b) — The complete draft is fact-checked against live web search, with category-specific verification strategies
- **Adversarial red-team** (Pass 3) — A hostile reviewer checks for fabricated citations, unverifiable claims, omissions, and exaggeration against raw source material
- **Factuality auditor** (Pass 5) — A final pass flags overconfident conclusions and ungrounded practical recommendations

### Correction
- **Grounding check** — Regex-based post-processing replaces out-of-bounds `[N]` citations with `[CITATION NEEDED]`
- **Bibliography cross-reference** — Only citations that map to real evidence units survive into the final bibliography
- **Unsupported claim removal** — Claims flagged by independent verification are explicitly marked for removal in the correction pass
- **Confidence scoring** — Evidence units carry confidence scores (0.3–0.8) based on verification status, surfaced in the report header

### Transparency
- **Appendix A** — Lists all unverified sources with exclusion rationale and verification failure reasons
- **Appendix B** — Exposes the raw fact-check notes and adversarial critique for full pipeline transparency
- **Appendix C** — Maps every evidence unit to its raw source chunk, enabling manual verification
- **Source Integrity Score** — Report header shows verified/unverified counts, citation coverage, and data recency warnings

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Health check |
| `POST` | `/api/research/explore` | Generate sub-topics from a broad query |
| `POST` | `/api/research/start` | Start a new research session |
| `POST` | `/api/research/interrupt` | Pause an active session |
| `POST` | `/api/research/resume` | Resume a paused session |
| `GET` | `/api/research/stream?id=` | SSE stream for real-time pipeline logs |
| `POST` | `/api/research/share` | Email a report via Resend |
| `GET` | `/api/chats/{user_id}` | Fetch user's research history |
| `DELETE` | `/api/chats/{research_id}` | Delete a research session |
| `GET` | `/api/chat/{research_id}` | Get chat history for a session |
| `POST` | `/api/chat` | Send a chat message during pause |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Framework | FastAPI (async, Python 3.11) |
| LLMs | Qwen-Max, Qwen-Plus (via DashScope) |
| Embeddings | Qwen text-embedding-v3 |
| Vector Search | FAISS (Facebook AI Similarity Search) |
| Browser Automation | Playwright (headless Chromium) |
| Web Search | DuckDuckGo API (ddgs) |
| HTML Parsing | BeautifulSoup 4 |
| Database | MongoDB (Motor async driver) |
| Email | Resend API |
| Streaming | SSE (sse-starlette) |
| Deployment | Docker → Render |

---

## Getting Started

### Prerequisites
- Python 3.11+
- MongoDB instance (Atlas or local)
- DashScope API key (Alibaba Cloud)

### Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Install Playwright browsers
python -m playwright install chromium

# Set environment variables
cp .env.example .env
# Edit .env with your API keys

# Run the server
uvicorn main:app --host 0.0.0.0 --port 10000
```

### Docker

```bash
docker build -t chronicle-engine .
docker run -p 10000:10000 --env-file .env chronicle-engine
```

---

## Environment Variables

| Variable | Description |
|---|---|
| `DASHSCOPE_API_KEY` | Alibaba Cloud DashScope API key for Qwen models |
| `MONGODB_URI` | MongoDB connection string |
| `RESEND_API_KEY` | Resend API key for email delivery |

---

## License

This project is part of the Chronicle AI platform.
