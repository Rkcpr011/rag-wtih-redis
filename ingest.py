"""
ingest.py  ─  PDF → Chunks → Embeddings → ChromaDB
───────────────────────────────────────────────────
Ye file do tarah se use ho sakti hai:

  1. FastAPI route se (app.py mein /ingest)
     POST /ingest  →  PDF file upload karo

  2. Direct command line se
     python ingest.py my_document.pdf

Flow:
  PDF file
      ↓
  Text extract karo (pypdf)
      ↓
  Chunks banao (overlap ke saath)
      ↓
  Har chunk embed karo (sentence-transformers)
      ↓
  ChromaDB mein store karo
"""

import os
import logging
from pathlib import Path

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

CHROMA_PATH     = os.getenv("CHROMA_PATH", "./chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "rag_docs")
EMBED_MODEL     = "all-MiniLM-L6-v2"
CHUNK_SIZE      = 500    # characters per chunk
CHUNK_OVERLAP   = 50     # overlap taaki context na toote

# ── Globals (lazy init) ───────────────────────────────────────────────────────

_embedder = None
_chroma_collection = None


def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        logger.info("Loading embedding model: %s", EMBED_MODEL)
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder


def get_collection():
    global _chroma_collection
    if _chroma_collection is None:
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        _chroma_collection = client.get_or_create_collection(COLLECTION_NAME)
    return _chroma_collection


# ── Core Functions ────────────────────────────────────────────────────────────

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """PDF bytes se plain text nikalo."""
    import io
    reader = PdfReader(io.BytesIO(pdf_bytes))
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text and text.strip():
            pages.append(text.strip())
    full_text = "\n\n".join(pages)
    logger.info("Extracted %d chars from %d pages", len(full_text), len(pages))
    return full_text


def make_chunks(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Text ko fixed-size overlapping chunks mein todo.

    Overlap kyun?
    Agar ek sentence chunk boundary pe cut ho jaaye,
    overlap ensure karta hai ki context dono chunks mein ho.

    Example (chunk_size=20, overlap=5):
      "The quick brown fox jumps over the lazy dog"
       ├─ "The quick brown fox" (0-20)
       └─ "n fox jumps over the" (15-35)  ← 5 char overlap
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    logger.info("Created %d chunks from %d chars", len(chunks), len(text))
    return chunks


def ingest_pdf(pdf_bytes: bytes, filename: str) -> dict:
    """
    PDF bytes lo, process karo, ChromaDB mein store karo.

    Returns:
        {"filename": str, "chunks": int, "status": "ok"}
    """
    # Step 1: Text extract karo
    text = extract_text_from_pdf(pdf_bytes)
    if not text.strip():
        raise ValueError(f"No extractable text found in {filename}")

    # Step 2: Chunks banao
    chunks = make_chunks(text)

    # Step 3: Embed karo
    embedder = get_embedder()
    embeddings = embedder.encode(chunks, normalize_embeddings=True).tolist()

    # Step 4: ChromaDB mein store karo
    collection = get_collection()

    # Unique IDs — filename + chunk index
    ids = [f"{filename}__chunk_{i}" for i in range(len(chunks))]

    # Agar ye file pehle se ingest hai toh pehle delete karo
    try:
        existing = collection.get(where={"source": filename})
        if existing["ids"]:
            collection.delete(where={"source": filename})
            logger.info("Deleted %d existing chunks for %s", len(existing["ids"]), filename)
    except Exception:
        pass  # pehli baar hai, delete ki zaroorat nahi

    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=[{"source": filename, "chunk_index": i} for i in range(len(chunks))]
    )

    logger.info("Ingested %d chunks from '%s' into ChromaDB", len(chunks), filename)
    return {"filename": filename, "chunks": len(chunks), "status": "ok"}


# ── CLI usage ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python ingest.py <path_to_pdf>")
        sys.exit(1)

    pdf_path = Path(sys.argv[1])
    if not pdf_path.exists():
        print(f"File not found: {pdf_path}")
        sys.exit(1)

    pdf_bytes = pdf_path.read_bytes()
    result = ingest_pdf(pdf_bytes, pdf_path.name)
    print(f"Done: {result}")
