# Use an official Python base image
FROM python:3.11-slim

# Install system dependencies for Playwright
RUN apt-get update && apt-get install -y --no-install-recommends \
    libnss3 libatk1.0-0 libatk-bridge2.0-0 libcups2 libdrm2 libxkbcommon0 \
    libxcomposite1 libxdamage1 libxfixes3 libxrandr2 libgbm1 libasound2 \
    && rm -rf /var/lib/apt/lists/*

# Set up non-root user required by Hugging Face Spaces
RUN useradd -m -u 1000 appuser
USER appuser
WORKDIR /home/appuser/app

# Redirecting Playwright cache to a writable location for the non-root user
ENV PLAYWRIGHT_BROWSERS_PATH=/home/appuser/ms-playwright
ENV PIP_NO_CACHE_DIR=1

# Install Python dependencies
COPY --chown=appuser requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Playwright browsers (as the non-root user)
RUN playwright install chromium --with-deps

# Copy application code
COPY --chown=appuser . .

# Expose port (default for HF Spaces is 7860)
EXPOSE 7860

# Command to run FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
