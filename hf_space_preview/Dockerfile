# ═══════════════════════════════════════════════════════════════
# TIDAL — Hugging Face Spaces Dockerfile
# ═══════════════════════════════════════════════════════════════
# Deploy: Create a new Space (SDK: Docker), copy this + backend/
# Hardware: CPU Basic (free) or T4 GPU (recommended)
# ═══════════════════════════════════════════════════════════════

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEVICE=cpu \
    MODEL_DIR=/app/models \
    MODEL_VERSION=vR.P.30.1 \
    CORS_ORIGINS=*

RUN apt-get update && \
    apt-get install -y --no-install-recommends libgl1 libglib2.0-0 curl && \
    rm -rf /var/lib/apt/lists/*

# Non-root user (HF Spaces requirement)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Install dependencies
COPY --chown=user requirements-prod.txt .
RUN pip install --no-cache-dir --user -r requirements-prod.txt

# Copy application
COPY --chown=user backend/ ./backend/
COPY --chown=user models/ ./models/

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
