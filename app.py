import subprocess
import sys
import os

# ──────────────────────────────────────────────
# Phase 1: Install Playwright Chromium at boot
# ──────────────────────────────────────────────
print("Installing Playwright Chromium...")
try:
    subprocess.run(
        [sys.executable, "-m", "playwright", "install", "chromium"],
        check=True,
    )
    print("Playwright Chromium installed successfully!")
except Exception as e:
    print(f"Warning – could not install Playwright Chromium: {e}")

# ──────────────────────────────────────────────
# Phase 2: Import FastAPI backend + Gradio
# ──────────────────────────────────────────────
import gradio as gr
from main import app as fastapi_app

# ──────────────────────────────────────────────
# Phase 3: Build a minimal Gradio status page
# (Named _blocks so HF does NOT auto-launch it)
# ──────────────────────────────────────────────
_blocks = gr.Blocks(title="Chronicle AI Backend")
with _blocks:
    gr.Markdown(
        """
        # 🧠 Chronicle AI Backend
        **Status:** Running ✅
        API endpoints are live at `/api/`
        """
    )

# ──────────────────────────────────────────────
# Phase 4: Mount Gradio UI onto FastAPI + Launch
# ──────────────────────────────────────────────
app = gr.mount_gradio_app(fastapi_app, _blocks, path="/")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 7860)))
