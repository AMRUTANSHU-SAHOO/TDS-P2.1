# Use official Playwright image (has Python + Chromium pre-installed)
FROM mcr.microsoft.com/playwright/python:v1.54.0-jammy

WORKDIR /app

# Copy requirements and install
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your app
COPY . .

# Create non-root user (optional, but recommended for Render)
RUN useradd --create-home --shell /bin/bash app \
    && chown -R app:app /app
USER app

# Command for Render
CMD python tools/scrape_website.py && uvicorn app:app --host 0.0.0.0 --port $PORT
