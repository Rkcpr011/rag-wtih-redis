"""
app.py  ─  FastAPI RAG Server with Valkey Cache + BetterDB Observability
─────────────────────────────────────────────────────────────────────────

Endpoints:
  POST /ask            ← RAG query karo
  GET  /cache/stats    ← Valkey cache stats
  DELETE /cache/clear  ← Cache flush (testing ke liye)
  GET  /health         ← Health check (Docker healthcheck bhi yahi use karta hai)

Observability:
  BetterDB (localhost:3001) directly Valkey se connect karke
  saara data pull karta hai — app mein koi Prometheus code nahi chahiye.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import time
import logging

from rag import ask
from semantic_cache import SemanticCache
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="RAG API with Valkey Cache",
    description="Production-grade RAG with semantic caching — BetterDB se monitor karo",
    version="1.0.0"
)

VALKEY_URL = os.getenv("VALKEY_URL", "redis://localhost:6379")


# ── Models ────────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str
    class Config:
        json_schema_extra = {"example": {"query": "What is RAG?"}}


class QueryResponse(BaseModel):
    response: str
    source: str        # "cache" ya "llm"
    latency_ms: float


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """
    Docker healthcheck yahi use karta hai.
    Valkey connection bhi verify hota hai.
    """
    try:
        cache = SemanticCache(valkey_url=VALKEY_URL)
        cache.client.ping()
        valkey_ok = True
    except Exception:
        valkey_ok = False

    return {
        "status": "ok" if valkey_ok else "degraded",
        "valkey": valkey_ok
    }


@app.post("/ask", response_model=QueryResponse)
def query_rag(req: QueryRequest):
    """
    RAG pipeline entry point.
    - Cache hit   → instant response (~20ms)
    - Cache miss  → vector search + LLM call (~1500ms)
    BetterDB pe jaake dekho ki kitne commands Valkey ko gaye.
    """
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query empty nahi honi chahiye")

    result = ask(req.query)
    logger.info("source=%s latency=%.1fms", result["source"], result["latency_ms"])
    return QueryResponse(**result)


@app.get("/cache/stats")
def cache_stats():
    """Cache ki current state — entries count, memory usage."""
    cache = SemanticCache(valkey_url=VALKEY_URL)
    return cache.stats()


@app.delete("/cache/clear")
def cache_clear():
    """Saare cache entries delete karo. Testing ke liye useful."""
    cache = SemanticCache(valkey_url=VALKEY_URL)
    deleted = cache.clear()
    return {"deleted_entries": deleted, "status": "cleared"}
