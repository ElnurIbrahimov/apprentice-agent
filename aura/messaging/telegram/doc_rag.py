"""Per-user document RAG for the Telegram bot.

Chunks an uploaded document, embeds each chunk via Ollama's nomic-embed-text,
stores everything in ``TelegramStore.doc_chunks``, and exposes a search API
that ranks chunks by cosine similarity. Also handles LLM-based summarization
(TL;DR + key facts + questions the doc answers) for the Mini App DocumentCard.

This is deliberately a lightweight mini-index — one user, one document, no
cross-doc fusion, no re-embedding on reload. It replaces the old "prepend 30KB
of flat text to the next message" flow in media.py.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import re
from concurrent.futures import as_completed

from aura.pools import bg_pool
from typing import Any, Optional

try:
    import ollama  # type: ignore
    _OLLAMA_AVAILABLE = True
except Exception:
    _OLLAMA_AVAILABLE = False

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "nomic-embed-text"
DEFAULT_CHUNK_CHARS = 2000
DEFAULT_CHUNK_OVERLAP = 200
_EMBED_POOL_WORKERS = 4


# ---------------------------------------------------------------------------
# Chunking — sentence-aware, paragraph-first, character-based
# ---------------------------------------------------------------------------

def chunk_text(
    text: str,
    max_chars: int = DEFAULT_CHUNK_CHARS,
    overlap_chars: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    """Split a document into overlapping chunks at paragraph / sentence
    boundaries. Adapted from ``aura/tools/local_rag.py::_chunk_text`` but
    standalone so the Telegram bot doesn't pull the full local_rag module.
    """
    if not text:
        return []
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + max_chars, n)
        if end < n:
            para_break = text.rfind("\n\n", start, end)
            if para_break > start + max_chars // 2:
                end = para_break + 2
            else:
                for sep in (". ", ".\n", "! ", "? "):
                    sent_break = text.rfind(sep, start, end)
                    if sent_break > start + max_chars // 2:
                        end = sent_break + len(sep)
                        break
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        start = max(0, end - overlap_chars)
    return chunks


# ---------------------------------------------------------------------------
# Embedding — Ollama nomic-embed-text, parallel
# ---------------------------------------------------------------------------

def embed_chunks(chunks: list[str]) -> list[Optional[list[float]]]:
    """Return an embedding for each chunk, or None on failure. Uses a small
    thread pool so a 40-chunk doc embeds in ~3-4s rather than ~40s."""
    if not chunks:
        return []
    if not _OLLAMA_AVAILABLE:
        logger.warning("[DocRAG] ollama package missing — skipping embeddings")
        return [None] * len(chunks)

    client = ollama.Client()

    def _embed_one(idx: int, chunk: str) -> tuple[int, Optional[list[float]]]:
        try:
            resp = client.embeddings(model=EMBEDDING_MODEL, prompt=chunk)
            emb = getattr(resp, "embedding", None)
            if emb is None and isinstance(resp, dict):
                emb = resp.get("embedding")
            return idx, list(emb) if emb else None
        except Exception as exc:
            logger.debug(f"[DocRAG] embed chunk {idx} failed: {exc}")
            return idx, None

    out: list[Optional[list[float]]] = [None] * len(chunks)
    pool = bg_pool()
    futures = [pool.submit(_embed_one, i, c) for i, c in enumerate(chunks)]
    for fut in as_completed(futures):
        try:
            idx, emb = fut.result(timeout=60)
            out[idx] = emb
        except Exception as exc:
            logger.debug(f"[DocRAG] embed future error: {exc}")
    return out


def _embed_query(text: str) -> Optional[list[float]]:
    """Embed a single query string. Returns None if Ollama is unavailable."""
    if not text or not _OLLAMA_AVAILABLE:
        return None
    try:
        client = ollama.Client()
        resp = client.embeddings(model=EMBEDDING_MODEL, prompt=text)
        emb = getattr(resp, "embedding", None)
        if emb is None and isinstance(resp, dict):
            emb = resp.get("embedding")
        return list(emb) if emb else None
    except Exception as exc:
        logger.debug(f"[DocRAG] embed_query error: {exc}")
        return None


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


# ---------------------------------------------------------------------------
# DocumentIndex — the public API used by media.py
# ---------------------------------------------------------------------------

def make_doc_id(user_id: str, filename: str, size: int) -> str:
    """Stable per-user document id — same file re-upload hits the same id."""
    payload = f"{user_id}:{filename}:{size}".encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:16]


class DocumentIndex:
    """Per-user mini-index over uploaded documents, backed by TelegramStore."""

    def __init__(self, store: Any) -> None:
        self.store = store

    def index_document(
        self,
        user_id: str,
        doc_id: str,
        filename: str,
        full_text: str,
    ) -> dict:
        """Chunk + embed a document and persist it. Blocking; run in a thread."""
        chunks = chunk_text(full_text)
        if not chunks:
            return {"doc_id": doc_id, "filename": filename, "chunks_count": 0}

        embeddings = embed_chunks(chunks)
        payload = [
            {
                "chunk_idx": i,
                "chunk_text": c,
                "embedding": embeddings[i],
            }
            for i, c in enumerate(chunks)
        ]
        # Replace any prior index of this doc for this user (re-upload flow)
        try:
            self.store.delete_user_doc(user_id, doc_id)
        except Exception:
            pass
        self.store.insert_doc_chunks(user_id, doc_id, filename, payload)
        return {
            "doc_id": doc_id,
            "filename": filename,
            "chunks_count": len(chunks),
            "size_chars": len(full_text),
        }

    def search(
        self,
        user_id: str,
        query: str,
        *,
        doc_id: Optional[str] = None,
        k: int = 5,
    ) -> list[dict]:
        """Return the top-k chunks for ``query``, scored by cosine similarity.

        Falls back to substring match if embeddings are unavailable.
        """
        chunks = self.store.get_doc_chunks(user_id, doc_id)
        if not chunks:
            return []

        q_emb = _embed_query(query)
        if q_emb:
            scored = []
            for c in chunks:
                emb = c.get("embedding")
                if not emb:
                    continue
                score = _cosine(q_emb, emb)
                scored.append((score, c))
            scored.sort(key=lambda x: x[0], reverse=True)
            return [
                {
                    "chunk_idx": c["chunk_idx"],
                    "chunk_text": c["chunk_text"],
                    "doc_id": c["doc_id"],
                    "filename": c["filename"],
                    "score": float(s),
                }
                for s, c in scored[:k]
            ]

        # Fallback: case-insensitive substring scoring
        q_lower = query.lower()
        scored_fallback = []
        for c in chunks:
            text_l = c["chunk_text"].lower()
            if q_lower in text_l:
                scored_fallback.append((1.0, c))
            else:
                # weak fallback: count overlapping query words
                q_words = set(re.findall(r"\w+", q_lower))
                t_words = set(re.findall(r"\w+", text_l))
                if q_words and t_words:
                    overlap = len(q_words & t_words) / len(q_words)
                    if overlap > 0:
                        scored_fallback.append((overlap, c))
        scored_fallback.sort(key=lambda x: x[0], reverse=True)
        return [
            {
                "chunk_idx": c["chunk_idx"],
                "chunk_text": c["chunk_text"],
                "doc_id": c["doc_id"],
                "filename": c["filename"],
                "score": float(s),
            }
            for s, c in scored_fallback[:k]
        ]

    def list_user_docs(self, user_id: str) -> list[dict]:
        try:
            return self.store.list_user_docs(user_id)
        except Exception as exc:
            logger.debug(f"[DocRAG] list_user_docs error: {exc}")
            return []

    def delete_document(self, user_id: str, doc_id: str) -> int:
        try:
            return self.store.delete_user_doc(user_id, doc_id)
        except Exception as exc:
            logger.debug(f"[DocRAG] delete_document error: {exc}")
            return 0

    def get_summary(self, user_id: str, doc_id: str) -> Optional[dict]:
        raw = self.store.get_doc_summary(user_id, doc_id)
        if not raw:
            return None
        try:
            return json.loads(raw)
        except Exception:
            return None

    def set_summary(
        self,
        user_id: str,
        doc_id: str,
        filename: str,
        summary: dict,
    ) -> None:
        try:
            self.store.set_doc_summary(user_id, doc_id, filename, json.dumps(summary))
        except Exception as exc:
            logger.debug(f"[DocRAG] set_summary error: {exc}")


# ---------------------------------------------------------------------------
# Summarization — LLM-backed TL;DR + facts + questions
# ---------------------------------------------------------------------------

_SUMMARY_PROMPT = """You are given a document. Produce a concise structured summary.

Return ONLY valid JSON matching this exact shape (no prose before or after):
{
  "summary": "2-3 sentence TL;DR",
  "facts": ["fact 1", "fact 2", "fact 3", "fact 4", "fact 5"],
  "questions": ["question 1", "question 2", "question 3", "question 4", "question 5"]
}

- facts: the 5 most important concrete claims in the document
- questions: 5 questions this document directly answers (useful as follow-ups)

Document:
{text}
"""


def summarize_document_sync(brain: Any, text: str, max_chars: int = 12000) -> dict:
    """Run the summary prompt through the brain. Blocks — call from a thread.

    Returns ``{summary, facts, questions}``. Falls back to a minimal shape if
    the brain response can't be parsed as JSON.
    """
    snippet = text[:max_chars]
    prompt = _SUMMARY_PROMPT.replace("{text}", snippet)
    try:
        raw = brain.think(prompt) if hasattr(brain, "think") else ""
    except Exception as exc:
        logger.debug(f"[DocRAG] brain.think summary error: {exc}")
        raw = ""

    if not raw:
        return {"summary": "", "facts": [], "questions": []}

    # Try direct JSON parse; fall back to extracting the first {...} block
    candidate = raw.strip()
    try:
        parsed = json.loads(candidate)
    except Exception:
        match = re.search(r"\{.*\}", candidate, re.DOTALL)
        parsed = None
        if match:
            try:
                parsed = json.loads(match.group(0))
            except Exception:
                parsed = None

    if not isinstance(parsed, dict):
        # Last resort: treat the entire response as the summary
        return {
            "summary": candidate[:500],
            "facts": [],
            "questions": [],
        }

    summary = str(parsed.get("summary") or "")[:1000]
    facts = [str(f)[:300] for f in (parsed.get("facts") or [])[:10]]
    questions = [str(q)[:300] for q in (parsed.get("questions") or [])[:10]]
    return {"summary": summary, "facts": facts, "questions": questions}


__all__ = [
    "chunk_text",
    "embed_chunks",
    "make_doc_id",
    "DocumentIndex",
    "summarize_document_sync",
    "EMBEDDING_MODEL",
]
