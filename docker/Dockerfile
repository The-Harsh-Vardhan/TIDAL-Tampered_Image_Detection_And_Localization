FROM nvidia/cuda:12.1.1-devel-ubuntu22.04 AS builder
ENV DEBIAN_FRONTEND=noninteractive PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends python3.11 python3.11-venv python3-pip build-essential && rm -rf /var/lib/apt/lists/*
RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY requirements-prod.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r /tmp/requirements.txt

FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04 AS runtime
ENV DEBIAN_FRONTEND=noninteractive PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PATH="/opt/venv/bin:$PATH"
RUN apt-get update && apt-get install -y --no-install-recommends python3.11 libgl1-mesa-glx libglib2.0-0 curl && rm -rf /var/lib/apt/lists/*
COPY --from=builder /opt/venv /opt/venv
RUN groupadd -r tidal && useradd -r -g tidal -d /app tidal
WORKDIR /app
COPY backend/ ./backend/
RUN mkdir -p /app/models /app/logs && chown -R tidal:tidal /app
USER tidal
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 CMD curl -f http://localhost:8000/health || exit 1
EXPOSE 8000
CMD ["uvicorn","backend.app:app","--host","0.0.0.0","--port","8000","--workers","1"]
