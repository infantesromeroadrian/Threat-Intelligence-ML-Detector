# =============================================================================
# Threat Intelligence Aggregator - Production Dockerfile
# Multi-stage build for minimal image size and security
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Base Python image with system dependencies
# -----------------------------------------------------------------------------
FROM python:3.10-slim AS base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# -----------------------------------------------------------------------------
# Stage 2: Python dependencies installer
# -----------------------------------------------------------------------------
FROM base AS dependencies

# Copy requirements first (better layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy model
RUN python -m spacy download en_core_web_sm

# -----------------------------------------------------------------------------
# Stage 3: Application code
# -----------------------------------------------------------------------------
FROM dependencies AS app

# Copy application source code (entire tool)
COPY domain/ ./domain/
COPY infrastructure/ ./infrastructure/
COPY application/ ./application/
COPY models/ ./models/
COPY __init__.py ./
COPY pyproject.toml ./
COPY README.md ./

# Install application in editable mode (for proper imports)
RUN pip install --no-cache-dir -e .

# Create non-root user for security
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

# Create directories for runtime data
RUN mkdir -p /app/data /app/models /app/threat_intel_cache && \
    chown -R appuser:appuser /app/data /app/models /app/threat_intel_cache

# Switch to non-root user
USER appuser

# Copy environment template (will be overridden by volume mount or env vars)
COPY --chown=appuser:appuser .env.example .env

# Expose ports
# 8000: FastAPI REST API
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command: Start FastAPI server
CMD ["uvicorn", "infrastructure.api.main:app", \
     "--host", "0.0.0.0", "--port", "8000"]
