"""Remote model catalog — auto-discover new models.

Mirrors Hermes Agent's model_catalog config:
  model_catalog:
    enabled: true
    url: https://hermes-agent.nousresearch.com/docs/api/model-catalog.json
    ttl_hours: 24

The catalog is cached locally and refreshed after TTL expires.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


_CACHE_PATH = Path.home() / ".aura" / "model_catalog_cache.json"


def get_catalog_config() -> dict:
    """Get the model catalog config."""
    try:
        from aura.config_loader import get_config_value
        return get_config_value("model_catalog", {}) or {}
    except ImportError:
        return {}


def is_catalog_enabled() -> bool:
    """Check if remote model catalog is enabled."""
    return get_catalog_config().get("enabled", False)


def get_catalog_url() -> str:
    """Get the catalog URL."""
    return get_catalog_config().get(
        "url",
        "https://hermes-agent.nousresearch.com/docs/api/model-catalog.json"
    )


def get_ttl_seconds() -> float:
    """Get the cache TTL in seconds."""
    hours = get_catalog_config().get("ttl_hours", 24)
    return hours * 3600


def fetch_catalog(force: bool = False) -> Optional[dict]:
    """Fetch the model catalog from remote, with local cache.

    Returns None if catalog is disabled or fetch fails.
    """
    if not is_catalog_enabled():
        return None

    # Check cache
    if not force and _CACHE_PATH.exists():
        try:
            data = json.loads(_CACHE_PATH.read_text(encoding="utf-8"))
            fetched_at = data.get("fetched_at", 0)
            if time.time() - fetched_at < get_ttl_seconds():
                return data
        except (json.JSONDecodeError, OSError):
            pass

    # Fetch from remote
    try:
        import urllib.request
        req = urllib.request.Request(get_catalog_url(), headers={"User-Agent": "Aura/4.7"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            data["fetched_at"] = time.time()
            _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            _CACHE_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
            return data
    except Exception as e:
        logger.debug(f"Catalog fetch failed: {e}")
        return None


def list_catalog_models() -> list[dict]:
    """List models from the remote catalog."""
    catalog = fetch_catalog()
    if not catalog:
        return []
    return catalog.get("models", [])


def search_catalog(query: str) -> list[dict]:
    """Search the catalog by model name or provider."""
    query = query.lower()
    return [
        m for m in list_catalog_models()
        if query in m.get("id", "").lower()
        or query in m.get("provider", "").lower()
    ]
