# ── Build stage ───────────────────────────────────────────────────────────────
FROM python:3.11-slim

# System dependencies (ChromaDB ke liye build tools chahiye)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies pehle copy karo (Docker cache optimization)
# Code badalne pe ye layer rebuild nahi hogi
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code copy karo
COPY semantic_cache.py .
COPY rag.py .
COPY ingest.py .
COPY app.py .

# Port expose karo
EXPOSE 8000

# App start karo
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

