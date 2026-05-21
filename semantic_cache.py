"""
semantic_cache.py
─────────────────
Valkey-backed Semantic Cache for RAG systems.

What this does:
  1. User query aata hai
  2. Uska embedding nikalo (sentence-transformers)
  3. Valkey mein stored embeddings ke saath cosine similarity check karo
  4. Similarity > threshold → cached response return karo  (CACHE HIT)
  5. Warna → None return karo, caller LLM call kare      (CACHE MISS)
  6. LLM response aane ke baad → cache mein store karo

Data structure in Valkey:
  Key   : "semcache:<sha256_of_query>"
  Value : JSON {
    "query":     original query string,
    "response":  LLM response string,
    "embedding": list[float],
    "ts":        unix timestamp
  }
  TTL   : configurable (default 24 hours)
"""

import json
import time
import hashlib
import logging
from typing import Optional

import numpy as np
import redis

logger = logging.getLogger(__name__)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    va = np.array(a, dtype=np.float32)
    vb = np.array(b, dtype=np.float32)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    if denom == 0:
        return 0.0
    return float(np.dot(va, vb) / denom)


def make_cache_key(query: str) -> str:
    digest = hashlib.sha256(query.strip().lower().encode()).hexdigest()[:16]
    return f"semcache:{digest}"


class SemanticCache:
    def __init__(
        self,
        valkey_url: str = "redis://localhost:6379",
        similarity_threshold: float = 0.90,
        ttl_seconds: int = 86400,
        max_scan_keys: int = 500,
    ):
        self.threshold = similarity_threshold
        self.ttl = ttl_seconds
        self.max_scan_keys = max_scan_keys

        self.client = redis.from_url(valkey_url, decode_responses=True)

        try:
            self.client.ping()
            logger.info("✅ Valkey connected at %s", valkey_url)
        except redis.ConnectionError as e:
            logger.error("❌ Valkey connection failed: %s", e)
            raise

    def lookup(self, query: str, query_embedding: list[float]) -> Optional[str]:
        try:
            best_similarity = 0.0
            best_response = None

            keys = list(self.client.scan_iter("semcache:*", count=self.max_scan_keys))

            for key in keys:
                raw = self.client.get(key)
                if not raw:
                    continue
                entry = json.loads(raw)
                sim = cosine_similarity(query_embedding, entry["embedding"])

                if sim > best_similarity:
                    best_similarity = sim
                    best_response = entry["response"] if sim >= self.threshold else None

            if best_response is not None:
                logger.info("Cache HIT  | similarity=%.4f | query='%s...'", best_similarity, query[:60])
                return best_response
            else:
                logger.info("Cache MISS | best_sim=%.4f | query='%s...'", best_similarity, query[:60])
                return None

        except Exception as e:
            logger.error("Cache lookup error: %s", e)
            return None

    def store(self, query: str, query_embedding: list[float], response: str) -> None:
        key = make_cache_key(query)
        payload = json.dumps({
            "query": query,
            "response": response,
            "embedding": query_embedding,
            "ts": time.time(),
        })
        try:
            self.client.setex(key, self.ttl, payload)
            logger.info("Cache STORE | key=%s | ttl=%ds", key, self.ttl)
        except Exception as e:
            logger.error("Cache store error: %s", e)

    def clear(self) -> int:
        keys = list(self.client.scan_iter("semcache:*"))
        if keys:
            self.client.delete(*keys)
        logger.info("Cache cleared: %d entries deleted", len(keys))
        return len(keys)

    def stats(self) -> dict:
        keys = list(self.client.scan_iter("semcache:*"))
        info = self.client.info("memory")
        return {
            "total_entries": len(keys),
            "used_memory_human": info.get("used_memory_human"),
            "similarity_threshold": self.threshold,
            "ttl_seconds": self.ttl,
        }
