# =============================================================================
# Dockerfile
# =============================================================================
# PURPOSE: Packages the entire application into a Docker container image.
#
# WHAT IS DOCKER?
#   Docker lets you package your app + all its dependencies into a single
#   "container image" that runs identically everywhere:
#   your laptop, a colleague's machine, Railway's servers.
#   No more "it works on my machine" — it works in the container, period.
#
# HOW RAILWAY USES THIS:
#   Railway detects the Dockerfile, builds the image, and runs it.
#   It injects environment variables (OPENAI_API_KEY etc.) separately.
#
# BUILD STAGES OVERVIEW:
#   1. Start from Python 3.11 slim (small base image)
#   2. Install system dependencies (build tools for some Python packages)
#   3. Install Python dependencies from requirements.txt
#   4. Pre-download the sentence-transformers model (80MB, avoids cold start)
#   5. Copy application code
#   6. Set the startup command
# =============================================================================

# ── Base Image ────────────────────────────────────────────────────────────────
# python:3.11-slim = Python 3.11 on Debian slim (smaller than full Debian)
# Using 3.11 specifically because some ML packages have 3.11 wheels (faster install)
FROM python:3.11-slim

# ── Metadata ──────────────────────────────────────────────────────────────────
LABEL maintainer="SaaSify Engineering"
LABEL description="SaaSify Sales Assistant Agent API"
LABEL version="1.0.0"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # Make the app package importable during runtime and Alembic startup
    PYTHONPATH=/app \
    # Limit memory arena count to lower memory usage on glibc
    MALLOC_ARENA_MAX=2


# ── System Dependencies ───────────────────────────────────────────────────────
# Some Python packages (like chromadb's C++ components) need build tools.
# We install them, use them to build Python packages, then remove them
# to keep the final image size small.
RUN apt-get update && apt-get install -y --no-install-recommends \
    # C/C++ compiler needed for some native Python extensions
    build-essential \
    # SSL certificates for HTTPS connections (API calls to OpenAI)
    ca-certificates \
    # Curl for health checks (optional but useful)
    curl \
    && rm -rf /var/lib/apt/lists/*   # Clean up apt cache to reduce image size

# ── Working Directory ─────────────────────────────────────────────────────────
# All subsequent commands run from /app inside the container
WORKDIR /app

# ── Install Python Dependencies ───────────────────────────────────────────────
# Copy requirements.txt FIRST (before the rest of the code).
# WHY? Docker caches each layer. If we copy all code first, then requirements.txt,
# any code change would invalidate the cache and re-download all packages.
# By copying requirements.txt first, packages are only re-installed when
# requirements.txt changes — not when we change app code.
COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Pre-download Embedding Model ──────────────────────────────────────────────
# We now use OpenAI API embeddings (text-embedding-3-small) instead of a local model.
# This saves ~1GB of RAM, prevents out-of-memory crashes on 512MB free tiers,
# and reduces build times.


# ── Copy Application Code ─────────────────────────────────────────────────────
# Copy everything else (code changes don't invalidate the pip cache layer)
COPY . .

# ── Create directories for runtime data ───────────────────────────────────────
# These directories are created at build time so they exist with correct permissions
RUN mkdir -p /app/chroma_db /app/.cache

# ── Expose Port ───────────────────────────────────────────────────────────────
# Document that the container listens on ports commonly used by Railway.
# The actual port is controlled by the PORT environment variable.
EXPOSE 8000 8080

# ── Optional Docker healthcheck ────────────────────────────────────────────────
# Helps container orchestrators detect when the app is ready to receive traffic.
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s CMD \
    curl -fsS http://127.0.0.1:${PORT:-8000}/health || exit 1

# ── Startup Command ───────────────────────────────────────────────────────────
# This runs when the container starts.
# 1. alembic upgrade head: applies any pending DB migrations (safe to run on every start)
# 2. uvicorn: starts the FastAPI server
#
# ${PORT:-8000} = use the PORT env var if set, else default to 8000
# Railway sets PORT automatically; local Docker uses 8000
# PYTHONPATH=/app ensures alembic can import the app module
CMD sh -c "PYTHONPATH=/app alembic upgrade head && uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"
