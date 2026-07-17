import os
import subprocess
import sys

# ──────────────────────────────────────────────
# Phase 1: Install Playwright Chromium at boot
# ──────────────────────────────────────────────
print("Installing Playwright Chromium...")
try:
    subprocess.run(
        [sys.executable, "-m", "playwright", "install", "chromium"],
        check=True
    )
    print("Playwright Chromium installed successfully!")
except Exception as e:
    print(f"Warning: Could not install Playwright Chromium: {e}")

# ──────────────────────────────────────────────
# Phase 2: Import FastAPI backend + Gradio
# ──────────────────────────────────────────────
import gradio as gr
from main import app as fastapi_app

# ──────────────────────────────────────────────
# Phase 3: Build a minimal Gradio UI
# ──────────────────────────────────────────────
with gr.Blocks(title="Chronicle AI Backend") as demo:
    gr.Markdown(
        """
        # 🧠 Chronicle AI Backend
        **Status:** Running ✅  
        API endpoints are live at `/api/`
        """
    )

# ──────────────────────────────────────────────
# Phase 4: Mount Gradio onto FastAPI + Launch
# ──────────────────────────────────────────────
# Mount the Gradio UI onto our existing FastAPI app at "/"
app = gr.mount_gradio_app(fastapi_app, demo, path="/")

# HF Spaces runs `python app.py` — we need to keep the process alive
# by starting uvicorn ourselves on port 7860.
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 7860)))
