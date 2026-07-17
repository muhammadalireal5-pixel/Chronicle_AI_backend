import os
import subprocess
import sys
import uvicorn
import gradio as gr

# 1. Install Playwright browser on startup to the local non-root cache
print("Installing Playwright Chromium...")
try:
    subprocess.run([sys.executable, "-m", "playwright", "install", "chromium"], check=True)
    print("Playwright Chromium installed successfully!")
except Exception as e:
    print(f"Error installing Playwright Chromium: {e}")

# 2. Import our existing FastAPI backend
from main import app as fastapi_app

# 3. Create a dummy Gradio interface to satisfy the Hugging Face Gradio SDK
with gr.Blocks() as demo:
    gr.Markdown("# 🧠 Chronicle AI Backend")
    gr.Markdown("The FastAPI engine is running successfully. API is available at `/api/`.")

# 4. Mount the Gradio interface onto the FastAPI app
app = gr.mount_gradio_app(fastapi_app, demo, path="/")

# 5. Run the Uvicorn server if this script is executed directly
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
