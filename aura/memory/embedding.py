"""Shared embedding utility for AURA memory systems.

Single implementation of the Ollama nomic-embed-text embedding call,
used by retrieval.py and unified_memory.py.
"""

import collections
import hashlib
import logging
import threading
from typing import List, Optional

logger = logging.getLogger(__name__)

_CACHE_MAX = 256
_embedding_cache: collections.OrderedDict[str, List[float]] = collections.OrderedDict()
_embedding_lock = threading.Lock()


def get_embedding(text: str, timeout: float = 3.0) -> Optional[List[float]]:
    """Get embedding vector for text via Ollama nomic-embed-text.

    Args:
        text: Input text (truncated to first 8000 chars internally).
        timeout: HTTP request timeout in seconds.

    Returns:
        List of floats (embedding vector), or None on failure.
    """
    key = hashlib.sha256(text[:8000].encode()).hexdigest()
    with _embedding_lock:
        if key in _embedding_cache:
            _embedding_cache.move_to_end(key)  # LRU: mark as recently used
            return _embedding_cache[key]

    try:
        import requests

        from aura.config import Config
        url = getattr(Config, 'OLLAMA_HOST', 'http://localhost:11434') + '/api/embeddings'
        r = requests.post(
            url,
            json={"model": "nomic-embed-text:latest", "prompt": text[:8000]},
            timeout=timeout,
        )
        if r.status_code == 200:
            emb = r.json().get("embedding")
            if emb:
                with _embedding_lock:
                    _embedding_cache[key] = emb
                    if len(_embedding_cache) > _CACHE_MAX:
                        _embedding_cache.popitem(last=False)  # Evict oldest
                return emb
    except Exception as e:
        logger.debug("[Embedding] Failed: %s", e)
    return None
