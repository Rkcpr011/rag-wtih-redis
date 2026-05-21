"""
rag.py  ─  RAG Pipeline with Valkey Semantic Cache
───────────────────────────────────────────────────

Flow:
  User Query
      │
      ▼
  [Embed Query]  ← sentence-transformers (local, free)
      │
      ▼
  [SemanticCache.lookup()]  ← Valkey check
      │
  ┌───┴──────────────────────────────┐
  │ HIT                              │ MISS
  │ return cached response           │ continue
  │ (no LLM call, no vector search)  │
  └──────────────────────────────────┘
                                     │
                                     ▼
                              [Vector Search]  ← ChromaDB (local)
                                     │
                                     ▼
                              [Build Prompt]
                                     │
                                     ▼
                              [LLM Call]  ← AzureOpenAI / local Ollama
                                     │
                                     ▼
                              [SemanticCache.store()]  ← Valkey mein save
                                     │
                                     ▼
                              Return Response

Note: Ye file simple demo hai. Apne existing rag.py mein
      SemanticCache import karke same pattern follow karo.
"""

import os
import time
import logging
from typing import Optional

from sentence_transformers import SentenceTransformer
# from openai import OpenAI
# Ye daalo
from openai import OpenAI
import chromadb

from semantic_cache import SemanticCache

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

VALKEY_URL       = os.getenv("VALKEY_URL", "redis://localhost:6379")

CHROMA_PATH      = os.getenv("CHROMA_PATH", "./chroma_db")
COLLECTION_NAME  = os.getenv("COLLECTION_NAME", "rag_docs")
EMBED_MODEL      = "all-MiniLM-L6-v2"   # fast, 384-dim, free local model
SIMILARITY_THRESHOLD = float(os.getenv("CACHE_THRESHOLD", "0.90"))

endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT")
api_key = os.getenv("AZURE_OPENAI_API_KEY")
# ── Globals (lazy init) ───────────────────────────────────────────────────────

_embedder: Optional[SentenceTransformer] = None
_cache: Optional[SemanticCache] = None
_chroma_collection = None
_openai_client: Optional[OpenAI] = None


def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        logger.info("Loading embedding model: %s", EMBED_MODEL)
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder


def get_cache() -> SemanticCache:
    global _cache
    if _cache is None:
        _cache = SemanticCache(
            valkey_url=VALKEY_URL,
            similarity_threshold=SIMILARITY_THRESHOLD,
            ttl_seconds=86400,
        )
    return _cache


def get_chroma():
    global _chroma_collection
    if _chroma_collection is None: 
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        _chroma_collection = client.get_or_create_collection(COLLECTION_NAME)
    return _chroma_collection



def get_openai() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client =OpenAI(base_url=endpoint,api_key=api_key)
    return _openai_client


# ── Core Functions ────────────────────────────────────────────────────────────

def embed_query(query: str) -> list[float]:
    """Query string → embedding vector."""
    vec = get_embedder().encode(query, normalize_embeddings=True)
    return vec.tolist()


def retrieve_context(query: str, query_embedding: list[float], top_k: int = 3) -> str:
    """ChromaDB se relevant chunks retrieve karo."""
    collection = get_chroma()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "distances"]
    )
    docs = results.get("documents", [[]])[0]
    context = "\n\n---\n\n".join(docs)
    logger.info("Retrieved %d chunks from vector DB", len(docs))
    return context


def call_llm(query: str, context: str) -> str:
    """OpenAI se response lo with retrieved context."""
    system_prompt = (
        "You are a helpful assistant. "
        "Answer the user's question ONLY using the provided context. "
        "If the context doesn't contain the answer, say so honestly."
    )
    user_message = f"""Context: {context} Question: {query}"""
    response = get_openai().chat.completions.create(
         model=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini"),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message},
        ],
        temperature=0.2,
        max_completion_tokens=500
    )
    return response.choices[0].message.content.strip()


# ── Main RAG Function ─────────────────────────────────────────────────────────

def ask(query: str) -> dict:
    """
    RAG pipeline ka entry point.

    Returns:
        {
          "response": str,
          "source":   "cache" | "llm",
          "latency_ms": float
        }
    """
    t0 = time.perf_counter()

    # Step 1: Query embed karo (ye toh karna hi hoga — cache ke liye bhi chahiye)
    query_embedding = embed_query(query)

    # Step 2: Semantic Cache check karo
    cache = get_cache()
    cached_response = cache.lookup(query, query_embedding)

    if cached_response is not None:
        latency = (time.perf_counter() - t0) * 1000
        return {
            "response": cached_response,
            "source": "cache",
            "latency_ms": round(latency, 2)
        }

    # Step 3: Cache miss → Vector DB se context lao
    context = retrieve_context(query, query_embedding)

    # Step 4: LLM call karo
    response = call_llm(query, context)

    # Step 5: Response ko cache mein store karo for next time
    cache.store(query, query_embedding, response)

    latency = (time.perf_counter() - t0) * 1000
    return {
        "response": response,
        "source": "llm",
        "latency_ms": round(latency, 2)
    }


# ── Demo ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n🔥 RAG + Valkey Semantic Cache Demo\n")

    # Demo queries — note karo ye semantically similar hain
    queries = [
        "What is retrieval augmented generation?",          # 1st call → LLM
        "Can you explain what RAG means in AI?",            # Should hit cache!
        "How does RAG work in language models?",            # Should hit cache!
        "What are the benefits of using RAG?",              # Alag topic → LLM
    ]

    for q in queries:
        print(f"\n📥 Query: {q}")
        result = ask(q)
        icon = "⚡ CACHE" if result["source"] == "cache" else "🤖 LLM"
        print(f"{icon} | {result['latency_ms']}ms")
        print(f"📤 {result['response'][:150]}...")
        print("─" * 60)

    # Cache stats
    stats = get_cache().stats()
    print(f"\n📊 Cache Stats: {stats}")
