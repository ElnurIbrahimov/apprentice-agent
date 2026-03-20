"""Model configuration API — list available models, get/set per-role routing."""

import logging
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field

from api.auth import require_api_key

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/models", tags=["models"], dependencies=[Depends(require_api_key)])

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


def _get_verified_models() -> dict:
    """Return verified model lists from config — no Ollama API query.

    Cloud models come from VERIFIED_CLOUD_MODELS (hardcoded, trusted).
    Local models come from VERIFIED_LOCAL_MODELS (utility only).
    ChatGPT models come from chatgpt_client or hardcoded fallback.
    """
    from aura.config import VERIFIED_CLOUD_MODELS, VERIFIED_LOCAL_MODELS

    cloud = [{"name": m, "size": 0, "is_cloud": True} for m in sorted(VERIFIED_CLOUD_MODELS)]
    local = [{"name": m, "size": 0, "is_cloud": False} for m in sorted(VERIFIED_LOCAL_MODELS)]

    # ChatGPT models
    try:
        from aura.auth.chatgpt_client import ALL_CHATGPT_MODELS
        chatgpt_names = list(ALL_CHATGPT_MODELS)
    except ImportError:
        chatgpt_names = [
            "chatgpt:gpt-5.4", "chatgpt:gpt-5.4-thinking", "chatgpt:gpt-5.4-pro",
            "chatgpt:gpt-5.3", "chatgpt:gpt-5.3-codex", "chatgpt:gpt-5.3-codex-spark",
            "chatgpt:gpt-5.2", "chatgpt:gpt-5.2-codex",
            "chatgpt:gpt-5.1", "chatgpt:gpt-5.1-codex", "chatgpt:gpt-5.1-codex-mini", "chatgpt:gpt-5.1-codex-max",
        ]
    chatgpt = [{"name": m, "size": 0, "is_cloud": True} for m in sorted(chatgpt_names)]

    # Direct API provider models (from configured providers)
    direct_api = []
    try:
        from aura.providers import list_all_provider_models
        for model_name, provider_display in list_all_provider_models():
            direct_api.append({"name": model_name, "size": 0, "is_cloud": True, "provider": provider_display})
    except Exception:
        pass

    return {
        "cloud": cloud, "local": local, "chatgpt": chatgpt, "direct_api": direct_api,
        "total": len(cloud) + len(local) + len(chatgpt) + len(direct_api),
    }


@router.get("/available")
async def list_available_models():
    """Return all verified models (cloud + local + ChatGPT), split by provider."""
    return _get_verified_models()


# Local models that are utility-only (not useful for chat)
_NON_CHAT_KEYWORDS = {"embed", "nomic-embed", "bge-", "e5-", "gte-", "ocr"}


@router.get("")
async def list_models_web():
    """Web UI compatible endpoint — returns flat model name lists.

    Filters out embedding/OCR local models that aren't useful for chat.
    """
    result = _get_verified_models()
    local_chat = [
        m["name"] for m in result["local"]
        if not any(kw in m["name"].lower() for kw in _NON_CHAT_KEYWORDS)
    ]
    direct_api = [m["name"] for m in result.get("direct_api", [])]
    return {
        "chatgpt_models": [m["name"] for m in result["chatgpt"]],
        "cloud_models": [m["name"] for m in result["cloud"]],
        "local_models": local_chat,
        "direct_api_models": direct_api,
        "total": len(result["chatgpt"]) + len(result["cloud"]) + len(local_chat) + len(direct_api),
    }


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
    role: str = Field(..., max_length=64)
    model: str = Field(..., max_length=200)


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
    valid_roles = ["fast", "reason", "code", "vision", "think", "longctx"]
    results = {}
    skipped = []
    for role, model in body.models.items():
        if role not in valid_roles:
            skipped.append(role)
            continue
        results[role] = Config.set_model(role, model)
    return {"results": results, "skipped_invalid_roles": skipped, "ok": all(results.values()) if results else False}
