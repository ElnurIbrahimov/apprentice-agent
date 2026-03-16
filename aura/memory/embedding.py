"""Shared embedding utility for AURA memory systems.

Single implementation of the Ollama nomic-embed-text embedding call,
used by retrieval.py, unified_memory.py, and memory_system.py.
"""

import logging
from typing import Optional, List

logger = logging.getLogger(__name__)


def get_embedding(text: str, timeout: float = 3.0) -> Optional[List[float]]:
    """Get embedding vector for text via Ollama nomic-embed-text.

    Args:
        text: Input text (truncated to first 1000 chars internally).
        timeout: HTTP request timeout in seconds.

    Returns:
        List of floats (embedding vector), or None on failure.
    """
    try:
        import requests
        from aura.config import Config
        url = getattr(Config, 'OLLAMA_HOST', 'http://localhost:11434') + '/api/embeddings'
        r = requests.post(
            url,
            json={"model": "nomic-embed-text:latest", "prompt": text[:1000]},
            timeout=timeout,
        )
        if r.status_code == 200:
            emb = r.json().get("embedding")
            if emb:
                return emb
    except Exception as e:
        logger.debug("[Embedding] Failed: %s", e)
    return None
