import asyncio
import json
import re
from typing import Callable, Awaitable
from playwright.async_api import async_playwright, Page, BrowserContext
from openai import AsyncOpenAI
import os
from bs4 import BeautifulSoup

# Use the same client setup as hibernation_engine
api_key = os.getenv("DASHSCOPE_API_KEY", "dummy_key_for_testing")
client = AsyncOpenAI(
    api_key=api_key,
    base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_google",
            "description": "Searches the web for the given query and returns a summary of results with URLs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query."}
                },
                "required": ["query"],
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "navigate_and_read",
            "description": "Navigates to a URL and returns the text content of the page.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The URL to visit."}
                },
                "required": ["url"],
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "extract_evidence",
            "description": "Extracts discrete factual claims from a specific chunk of text found on the page. Use this when you find useful information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The URL of the source."},
                    "title": {"type": "string", "description": "The title of the source page."},
                    "relevant_text": {"type": "string", "description": "The exact relevant text chunk (max 1000 chars) to save."}
                },
                "required": ["url", "title", "relevant_text"],
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "finish_research",
            "description": "Ends the browsing session. Call this when you have extracted enough evidence.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        }
    }
]

async def extract_page_text(page: Page) -> str:
    """Extract readable text from page using BeautifulSoup."""
    try:
        html = await page.content()
        soup = BeautifulSoup(html, "html.parser")
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer", "header", "aside"]):
            script.decompose()
        text = soup.get_text(separator=' ', strip=True)
        # Collapse multiple spaces
        text = re.sub(r'\s+', ' ', text)
        return text[:15000] # Limit to avoid context window explosion
    except Exception as e:
        return f"Error extracting text: {e}"


async def search_with_playwright(page: Page, search_query: str) -> list[dict]:
    """
    Robust search function that tries multiple search engines.
    Returns a list of dicts with 'title' and 'url' keys.
    """
    results = []
    
    # Strategy 1: Google search (most reliable)
    try:
        encoded_query = search_query.replace(' ', '+')
        await page.goto(
            f"https://www.google.com/search?q={encoded_query}",
            wait_until="domcontentloaded",
            timeout=20000
        )
        await asyncio.sleep(2)
        
        # Extract links from Google results using multiple selector strategies
        search_results = await page.evaluate("""
            () => {
                const results = [];
                // Strategy A: Standard Google result links
                document.querySelectorAll('div.g a[href^="http"]').forEach(a => {
                    const title = a.querySelector('h3');
                    if (title && a.href && !a.href.includes('google.com')) {
                        results.push({ title: title.innerText, url: a.href });
                    }
                });
                // Strategy B: Fallback - any h3 with parent link
                if (results.length === 0) {
                    document.querySelectorAll('a h3').forEach(h3 => {
                        const a = h3.closest('a');
                        if (a && a.href && !a.href.includes('google.com')) {
                            results.push({ title: h3.innerText, url: a.href });
                        }
                    });
                }
                // Strategy C: Last resort - grab all external links with reasonable text
                if (results.length === 0) {
                    document.querySelectorAll('a[href^="http"]').forEach(a => {
                        if (a.innerText.length > 10 && !a.href.includes('google.com') && !a.href.includes('accounts.google')) {
                            results.push({ title: a.innerText.substring(0, 100), url: a.href });
                        }
                    });
                }
                return results.slice(0, 8);
            }
        """)
        if search_results:
            results = search_results
            print(f"[Search] Google returned {len(results)} results for: {search_query}")
            return results
    except Exception as e:
        print(f"[Search] Google search failed: {e}")
    
    # Strategy 2: Fallback to Bing
    try:
        encoded_query = search_query.replace(' ', '+')
        await page.goto(
            f"https://www.bing.com/search?q={encoded_query}",
            wait_until="domcontentloaded",
            timeout=20000
        )
        await asyncio.sleep(2)
        
        search_results = await page.evaluate("""
            () => {
                const results = [];
                document.querySelectorAll('li.b_algo a').forEach(a => {
                    if (a.href && a.innerText.length > 5) {
                        results.push({ title: a.innerText.substring(0, 100), url: a.href });
                    }
                });
                if (results.length === 0) {
                    document.querySelectorAll('a[href^="http"]').forEach(a => {
                        if (a.innerText.length > 10 && !a.href.includes('bing.com') && !a.href.includes('microsoft.com')) {
                            results.push({ title: a.innerText.substring(0, 100), url: a.href });
                        }
                    });
                }
                return results.slice(0, 8);
            }
        """)
        if search_results:
            results = search_results
            print(f"[Search] Bing returned {len(results)} results for: {search_query}")
            return results
    except Exception as e:
        print(f"[Search] Bing search failed: {e}")
    
    # Strategy 3: Last resort - DuckDuckGo lite (minimal JS, less bot detection)
    try:
        encoded_query = search_query.replace(' ', '+')
        await page.goto(
            f"https://lite.duckduckgo.com/lite/?q={encoded_query}",
            wait_until="domcontentloaded",
            timeout=20000
        )
        await asyncio.sleep(2)
        
        search_results = await page.evaluate("""
            () => {
                const results = [];
                document.querySelectorAll('a.result-link').forEach(a => {
                    if (a.href) {
                        results.push({ title: a.innerText.substring(0, 100), url: a.href });
                    }
                });
                if (results.length === 0) {
                    document.querySelectorAll('a[href^="http"]').forEach(a => {
                        if (a.innerText.length > 10 && !a.href.includes('duckduckgo.com')) {
                            results.push({ title: a.innerText.substring(0, 100), url: a.href });
                        }
                    });
                }
                return results.slice(0, 8);
            }
        """)
        if search_results:
            results = search_results
            print(f"[Search] DuckDuckGo Lite returned {len(results)} results for: {search_query}")
    except Exception as e:
        print(f"[Search] DuckDuckGo Lite search failed: {e}")
    
    return results


async def run_browser_agent(query: str, research_id: str, send_log: Callable[[str, str, str], Awaitable[None]], extract_fact_func, process_claims_func, headless: bool = True) -> list:
    """Runs an autonomous Qwen agent controlling a live Playwright browser."""
    
    extracted_units = []
    
    await send_log(research_id, "🚀 Initializing Live Browser Agent...", "LOG")
    
    async with async_playwright() as p:
        # Launch browser in headless mode
        browser = await p.chromium.launch(
            headless=True,
            args=['--no-sandbox', '--disable-dev-shm-usage', '--disable-gpu']
        )
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
            viewport={"width": 1280, "height": 720},
            locale="en-US"
        )
        page = await context.new_page()
        
        system_prompt = (
            f"You are an autonomous research agent controlling a live web browser.\n"
            f"Your objective is to thoroughly research the topic: '{query}'.\n"
            f"1. Use search_google to find relevant URLs.\n"
            f"2. Use navigate_and_read to read the contents of those URLs.\n"
            f"3. Use extract_evidence to formally save relevant facts you find.\n"
            f"4. Once you have enough evidence (at least 3-5 high-quality distinct sources), call finish_research.\n"
            f"If a page blocks you or requires a captcha, just ignore it and search for something else."
        )
        
        messages = [{"role": "system", "content": system_prompt}]
        
        max_turns = 15
        turn = 0
        
        while turn < max_turns:
            turn += 1
            await send_log(research_id, f"Agent is thinking (Turn {turn}/{max_turns})...", "LOG")
            
            try:
                response = await client.chat.completions.create(
                    model="qwen-max",
                    messages=messages,
                    tools=TOOLS,
                    tool_choice="auto"
                )
            except Exception as e:
                await send_log(research_id, f"LLM Error: {e}", "LOG")
                break
                
            msg = response.choices[0].message
            messages.append(msg)
            
            if getattr(msg, "tool_calls", None) is None:
                # If it didn't call a tool, maybe it just spoke. Force it to keep going or finish.
                if msg.content and "finish_research" in msg.content:
                    break
                messages.append({"role": "user", "content": "Please use a tool to continue researching or finish."})
                continue

            finished = False
            for tool_call in msg.tool_calls:
                fn_name = tool_call.function.name
                args = json.loads(tool_call.function.arguments)
                
                await send_log(research_id, f"🤖 Agent Action: {fn_name}({str(args)[:100]}...)", "LOG")
                
                if fn_name == "search_google":
                    search_query = args.get("query", query)
                    search_results = await search_with_playwright(page, search_query)
                    
                    results_text = []
                    for r in search_results[:5]:
                        results_text.append(f"Title: {r['title']}\nURL: {r['url']}")
                    
                    tool_res = "\n".join(results_text) if results_text else "No results found. Try a different search query."
                    messages.append({"role": "tool", "tool_call_id": tool_call.id, "name": fn_name, "content": tool_res})

                elif fn_name == "navigate_and_read":
                    url = args.get("url", "")
                    try:
                        await page.goto(url, wait_until="domcontentloaded", timeout=20000)
                        await asyncio.sleep(2)
                        page_text = await extract_page_text(page)
                        messages.append({"role": "tool", "tool_call_id": tool_call.id, "name": fn_name, "content": page_text})
                    except Exception as e:
                        messages.append({"role": "tool", "tool_call_id": tool_call.id, "name": fn_name, "content": f"Failed to load: {e}"})

                elif fn_name == "extract_evidence":
                    await send_log(research_id, f"📝 Extracting facts from {args.get('url')}...", "LOG")
                    # Delegate to the robust hibernation_engine extraction logic
                    new_units = await extract_fact_func(
                        text_chunk=args.get("relevant_text", ""),
                        source_title=args.get("title", ""),
                        source_url=args.get("url", ""),
                        verified=True, 
                        verification_method="browser_live"
                    )
                    await process_claims_func(new_units)
                    extracted_units.extend(new_units)
                    messages.append({"role": "tool", "tool_call_id": tool_call.id, "name": fn_name, "content": f"Successfully saved {len(new_units)} facts."})
                
                elif fn_name == "finish_research":
                    messages.append({"role": "tool", "tool_call_id": tool_call.id, "name": fn_name, "content": "Research finished."})
                    finished = True
            
            if finished:
                break
                
        await browser.close()
        
    await send_log(research_id, f"✅ Browser session complete. Extracted {len(extracted_units)} raw units.", "LOG")
    return extracted_units

async def explore_topics(query: str, headless: bool = True) -> list[dict]:
    """Exploratory phase to generate sub-topics using Playwright and Qwen."""
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=['--no-sandbox', '--disable-dev-shm-usage', '--disable-gpu']
        )
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
            viewport={"width": 1280, "height": 720},
            locale="en-US"
        )
        page = await context.new_page()
        
        try:
            # Use our robust multi-engine search
            search_results = await search_with_playwright(page, f"{query} research trends future")
            
            # Now visit top results and extract snippets
            snippets = []
            for result in search_results[:5]:
                try:
                    await page.goto(result["url"], wait_until="domcontentloaded", timeout=15000)
                    await asyncio.sleep(1)
                    text = await extract_page_text(page)
                    snippets.append(f"Title: {result['title']}\nURL: {result['url']}\nContent: {text[:500]}")
                except Exception:
                    snippets.append(f"Title: {result['title']}\nURL: {result['url']}\nContent: (Could not load)")
            
            search_text = "\n\n".join(snippets) if snippets else f"No search results found for '{query}'. Generate subtopics based on your knowledge."
            
            prompt = (
                f"You are a Research Strategy AI. Based on the following recent search results for '{query}', "
                f"identify 3 to 5 highly significant, trending, and authentic sub-topics that warrant deep research.\n"
                f"Return ONLY valid JSON in this exact format:\n"
                f"{{\n"
                f"  \"topics\": [\n"
                f"    {{\"title\": \"Sub-topic Name\", \"score\": 95, \"reasoning\": \"Why this is important and trending\"}}\n"
                f"  ]\n"
                f"}}\n\n"
                f"Search Results:\n{search_text}"
            )
            
            response = await client.chat.completions.create(
                model="qwen-max",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            data = json.loads(content)
            return data.get("topics", [])
            
        except Exception as e:
            print(f"Explore topics error: {e}")
            # fallback
            return [
                {"title": f"{query} overview", "score": 80, "reasoning": "Fallback general topic due to search timeout"},
                {"title": f"Recent advancements in {query}", "score": 90, "reasoning": "Fallback subtopic"}
            ]
        finally:
            await browser.close()
