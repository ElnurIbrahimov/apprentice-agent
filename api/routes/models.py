"""Model configuration API — list available models, get/set per-role routing."""

import os
import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/models", tags=["models"])

# Feature → chain mapping (for display in the UI)
FEATURE_CHAIN_MAP = {
    "fast":    ["Chat (simple)", "Search", "Translate", "Grammar", "Browser Agent"],
    "reason":  ["Chat (complex)", "Write", "Ask Quick-Action"],
    "code":    ["Code tasks", "Browser Agent (planning)"],
    "vision":  ["OCR (cloud)", "Image understanding"],
    "think":   ["Deep reasoning", "Multi-step analysis"],
    "longctx": ["PDF Chat", "Long document Q&A"],
}

ROLE_LABELS = {
    "fast":    "Fast / Chat",
    "reason":  "Reasoning / Write",
    "code":    "Code",
    "vision":  "Vision",
    "think":   "Deep Thinking",
    "longctx": "Long Context / PDF",
}


def _ollama_models() -> list[dict]:
    """Fetch model list from Ollama."""
    try:
        import requests
        host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        r = requests.get(f"{host}/api/tags", timeout=5)
        if r.status_code == 200:
            raw = r.json().get("models", [])
            return [
                {
                    "name": m["name"],
                    "size": m.get("size", 0),
                    "is_cloud": m["name"].endswith(":cloud"),
                }
                for m in raw
            ]
    except Exception as e:
        logger.warning(f"[Models] Could not fetch Ollama models: {e}")
    return []


@router.get("/available")
async def list_available_models():
    """Return all models available in Ollama, split into cloud and local."""
    models = _ollama_models()
    cloud = [m for m in models if m["is_cloud"]]
    local = [m for m in models if not m["is_cloud"]]
    return {"cloud": cloud, "local": local, "total": len(models)}


@router.get("/config")
async def get_model_config():
    """Return current model assignment for each role + chain options."""
    from aura.config import Config
    current = Config.get_all_models()
    chains = {
        "fast":    Config.MODEL_FAST_CHAIN,
        "reason":  Config.MODEL_REASON_CHAIN,
        "code":    Config.MODEL_CODE_CHAIN,
        "vision":  Config.MODEL_VISION_CHAIN,
        "think":   Config.MODEL_THINK_CHAIN,
        "longctx": Config.MODEL_LONGCTX_CHAIN,
    }
    return {
        "current": current,
        "chains": chains,
        "feature_map": FEATURE_CHAIN_MAP,
        "role_labels": ROLE_LABELS,
    }


class ModelPatch(BaseModel):
    role: str
    model: str


@router.patch("/config")
async def set_model_config(body: ModelPatch):
    """Set the active model for a given role."""
    from aura.config import Config
    valid_roles = ["fast", "reason", "code", "vision", "think", "longctx"]
    if body.role not in valid_roles:
        raise HTTPException(400, f"Invalid role. Must be one of: {valid_roles}")
    ok = Config.set_model(body.role, body.model)
    if not ok:
        raise HTTPException(500, "Failed to set model")
    return {"role": body.role, "model": body.model, "ok": True}


class ModelsBulkPatch(BaseModel):
    models: dict  # role -> model


@router.patch("/config/bulk")
async def set_models_bulk(body: ModelsBulkPatch):
    """Set multiple role→model assignments at once."""
    from aura.config import Config
    results = {}
    for role, model in body.models.items():
        results[role] = Config.set_model(role, model)
    return {"results": results, "ok": all(results.values())}
