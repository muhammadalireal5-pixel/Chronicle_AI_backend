import os
import subprocess
import sys
import asyncio
from datetime import datetime
from typing import Dict, Any

from fastapi import FastAPI, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse

# Playwright Chromium is pre-installed via the Dockerfile
from mongo_db import db
from hibernation_engine import research_loop, interrupt_flag, resume_events, message_queue, handle_chat_message
from browser_agent import explore_topics
import resend
import markdown
import os
from dotenv import load_dotenv

load_dotenv()
resend.api_key = os.getenv("RESEND_API_KEY")

app = FastAPI(title="Chronicle Deep Research Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Next.js frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def on_startup():
    # MongoDB initializes globally in mongo_db.py
    pass

@app.post("/api/research/explore")
async def explore_research_topics(payload: Dict[str, Any]):
    """
    Given a broad topic, use Playwright to search and generate specific subtopics.
    """
    query = payload.get("query")
    headless = payload.get("headless", False)
    if not query:
        return {"error": "Query required", "topics": []}
    
    try:
        topics = await explore_topics(query, headless)
        return {"topics": topics}
    except Exception as e:
        return {"error": str(e), "topics": []}

@app.post("/api/research/share")
async def share_via_email(payload: Dict[str, Any]):
    """
    Converts markdown text to HTML and emails it using Resend.
    """
    report_text = payload.get("report")
    title = payload.get("title", "Chronicle AI Research Report")
    emails = payload.get("emails", [])
    
    if not report_text or not emails:
        return {"error": "Report text and emails required", "success": False}
    
    try:
        # Convert Markdown to HTML
        html_content = markdown.markdown(report_text, extensions=['tables', 'fenced_code'])
        
        # Add basic styling to HTML
        html_with_style = f"""
        <html>
          <body style="font-family: sans-serif; line-height: 1.6; color: #333; max-width: 800px; margin: 0 auto; padding: 20px;">
            {html_content}
          </body>
        </html>
        """
        
        # Send Email via Resend
        # For Resend free tier, 'from' must be onboarding@resend.dev unless a domain is verified.
        params = {
            "from": "Chronicle AI <onboarding@resend.dev>",
            "to": emails,
            "subject": title,
            "html": html_with_style
        }
        
        resend.Emails.send(params)
        
        return {"success": True}
    except Exception as e:
        return {"error": str(e), "success": False}

@app.post("/api/research/start")
async def start_research(payload: Dict[str, Any], background_tasks: BackgroundTasks):
    """
    Start the hibernation engine loop for a given query.
    """
    query = payload.get("query")
    research_id = payload.get("id", "default_research")
    user_id = payload.get("userId") # Passed from Clerk frontend
    headless = payload.get("headless", False)
    
    # reset interrupt flag if any
    interrupt_flag[research_id] = False
    
    # Save initial chat state to Mongo synchronously to avoid frontend race condition
    if db is not None and user_id:
        try:
            await db.chats.update_one(
                {"researchId": research_id},
                {"$set": {
                    "userId": user_id,
                    "query": query,
                    "status": "RUNNING",
                    "createdAt": datetime.utcnow()
                }},
                upsert=True
            )
        except Exception as e:
            print(f"Mongo Start Insert Error: {e}")

    # run background loop
    background_tasks.add_task(research_loop, research_id, query, user_id, headless)
    return {"status": "started", "research_id": research_id}

@app.post("/api/research/interrupt")
async def interrupt_research(payload: Dict[str, Any]):
    """
    Safely pause/interrupt the research loop.
    """
    research_id = payload.get("id", "default_research")
    interrupt_flag[research_id] = True
    return {"status": "interrupted", "research_id": research_id}

@app.get("/api/research/stream")
async def research_stream(request: Request, id: str = "default_research"):
    """
    SSE endpoint to stream logs to Next.js UI.
    """
    async def event_generator():
        while True:
            if await request.is_disconnected():
                break
            
            if id in message_queue and not message_queue[id].empty():
                msg = await message_queue[id].get()
                yield msg
            else:
                await asyncio.sleep(0.1)
                
    return EventSourceResponse(event_generator())

@app.get("/api/chats/{user_id}")
async def get_user_chats(user_id: str):
    """Fetch previous research sessions for the logged-in user."""
    if db is None:
        return {"chats": [], "error": "MongoDB not configured yet"}
    
    try:
        cursor = db.chats.find({"userId": user_id}).sort("createdAt", -1).limit(50)
        chats = []
        async for document in cursor:
            # Convert ObjectId to string
            document["_id"] = str(document["_id"])
            chats.append(document)
        return {"chats": chats}
    except Exception as e:
        return {"chats": [], "error": str(e)}

@app.delete("/api/chats/{research_id}")
async def delete_user_chat(research_id: str):
    """Delete a research session and all its associated claims."""
    if db is None:
        return {"success": False, "error": "MongoDB not configured"}
    
    try:
        await db.chats.delete_one({"researchId": research_id})
        await db.claims.delete_many({"researchId": research_id})
        return {"success": True}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/api/research/resume")
async def resume_research(payload: Dict[str, Any]):
    """
    Resume a paused research loop.
    """
    research_id = payload.get("id")
    if research_id in resume_events:
        resume_events[research_id].set()
        return {"status": "resumed", "research_id": research_id}
    return {"status": "error", "message": "Research ID not found or not paused"}

@app.get("/api/chat/{research_id}")
async def get_chat_history(research_id: str):
    """Fetch the chat history for a specific research session."""
    if db is None:
        return {"messages": []}
    
    try:
        chat_doc = await db.chats.find_one({"researchId": research_id})
        if chat_doc and "messages" in chat_doc:
            # Drop the internal object ID or format dates if needed, but dict is fine for JSON
            return {"messages": chat_doc["messages"]}
        return {"messages": []}
    except Exception as e:
        return {"messages": [], "error": str(e)}

@app.post("/api/chat")
async def send_chat_message(payload: Dict[str, Any]):
    """Send a message to the AI while the research is paused."""
    research_id = payload.get("id")
    message = payload.get("message")
    
    if not research_id or not message:
        return {"error": "Missing id or message"}
    
    reply = await handle_chat_message(research_id, message)
    return {"reply": reply}


