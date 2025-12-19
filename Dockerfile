# Dockerfile for LiveKit Cloud Agent Deployment
# This Dockerfile is optimized for LiveKit Cloud's build service
FROM python:3.11-slim

# Install system dependencies required for audio processing and ML models
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    python3-dev \
    curl \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/app" \
    --shell "/sbin/nologin" \
    --uid "${UID}" \
    appuser

WORKDIR /app

# Install build tools needed to build livekit-agents from source
RUN pip install --no-cache-dir --upgrade pip build

# Copy the modified livekit-agents source code
# Assuming the livekit-agents directory is in the build context
# If it's in a different location, adjust the COPY path accordingly
COPY livekit-agents/ ./livekit-agents/

# Copy requirements.txt first to check if livekit-agents is listed
COPY requirements.txt .

# Install dependencies, but uninstall livekit-agents if it was installed from PyPI
# so we can install our modified version
RUN pip install --no-cache-dir -r requirements.txt && \
    pip uninstall -y livekit-agents 2>/dev/null || true

# Build and install the modified livekit-agents package from source
# This will override any PyPI version
WORKDIR /app/livekit-agents
RUN python -m build --wheel && \
    pip install --no-cache-dir dist/*.whl

# Return to app directory
WORKDIR /app

# Copy application code
COPY voice_agent_entrypoint.py .
COPY app/ ./app/

# Copy workflow parser
COPY workflow_parser/ ./workflow_parser/

# Set proper permissions
RUN chown -R appuser:appuser /app
USER appuser

# Environment variables (will be overridden by LiveKit Cloud secrets)
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Health check for container readiness
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"

# LiveKit Cloud will inject environment variables via secrets
# The entrypoint is defined in livekit.toml, but we provide a default
CMD ["python", "voice_agent_entrypoint.py", "start"]

