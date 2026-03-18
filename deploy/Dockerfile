FROM python:3.12-slim

# System deps for building native extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libffi-dev libssl-dev git curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/aura

# Install Python deps first (cached layer)
COPY requirements.txt .

# Filter out Windows-only packages before installing
RUN grep -v -E '(comtypes|pyttsx3|pycaw|winotify|screen-brightness-control|sounddevice|mss|pyperclip)' \
    requirements.txt > requirements-server.txt \
    && pip install --no-cache-dir -r requirements-server.txt

# Copy application code
COPY . .

# Create data directories
RUN mkdir -p data aura_data logs

# Create non-root user
RUN useradd -r -s /bin/false -d /opt/aura aura \
    && chown -R aura:aura /opt/aura

USER aura

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -sf http://localhost:8000/api/status || exit 1

CMD ["python", "run_web.py", "--host", "0.0.0.0", "--port", "8000", "--prod"]
