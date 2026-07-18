# Use an official Python base image
FROM python:3.11-slim

# Install ALL system dependencies required by Chromium (as root)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libnss3 libatk1.0-0 libatk-bridge2.0-0 libcups2 libdrm2 libxkbcommon0 \
    libxcomposite1 libxdamage1 libxfixes3 libxrandr2 libgbm1 libasound2 \
    libcairo2 libpango-1.0-0 libpangocairo-1.0-0 libgdk-pixbuf-2.0-0 \
    libgtk-3-0 libx11-6 libx11-xcb1 libxcb1 libxext6 libxi6 \
    libxrender1 libxtst6 fonts-liberation xdg-utils wget \
    && rm -rf /var/lib/apt/lists/*

# Set up non-root user required by Hugging Face Spaces
RUN useradd -m -u 1000 appuser
USER appuser
WORKDIR /home/appuser/app

# Redirecting Playwright cache to a writable location for the non-root user
ENV PLAYWRIGHT_BROWSERS_PATH=/home/appuser/ms-playwright
ENV PIP_NO_CACHE_DIR=1
ENV PATH="/home/appuser/.local/bin:$PATH"

# Install Python dependencies
COPY --chown=appuser requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Playwright browsers (as the non-root user)
RUN python -m playwright install chromium

# Copy application code
COPY --chown=appuser . .

# Expose port (Render defaults to 10000)
EXPOSE 10000

# Command to run FastAPI (using shell form so $PORT gets evaluated by Render)
CMD python -m uvicorn main:app --host 0.0.0.0 --port ${PORT:-10000}
