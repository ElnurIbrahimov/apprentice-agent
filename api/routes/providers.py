"""Direct API provider management — list providers, set/remove API keys, list models."""

import logging
import os
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from api.auth import require_api_key

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/providers", tags=["providers"], dependencies=[Depends(require_api_key)])


class ApiKeyBody(BaseModel):
    key: str = Field(..., min_length=1, max_length=500)


@router.get("")
async def list_providers():
    """List all providers with configuration status and model counts."""
    from aura.providers import list_configured_providers
    providers = list_configured_providers()
    configured_count = sum(1 for p in providers if p["configured"])
    return {
        "providers": providers,
        "total": len(providers),
        "configured": configured_count,
    }


@router.get("/{name}/models")
async def list_provider_models(name: str):
    """List models for a specific provider."""
    import re
    if not re.match(r'^[a-z_]{1,32}$', name):
        raise HTTPException(400, "Invalid provider name")
    from aura.providers import get_provider
    provider = get_provider(name)
    if not provider:
        raise HTTPException(404, f"Unknown provider: {name}")
    return {
        "provider": name,
        "display_name": provider.display_name,
        "configured": provider.is_configured(),
        "models": provider.list_models(),
    }


def _resolve_env_var(name: str) -> str:
    """Resolve provider name to env var. Uses PROVIDER_CONFIGS if available,
    otherwise derives from convention: name -> NAME_API_KEY."""
    from aura.providers.registry import PROVIDER_CONFIGS
    cfg = PROVIDER_CONFIGS.get(name)
    if cfg:
        return cfg["env_var"]
    # Convention for non-text providers (image, video, audio, search)
    return f"{name.upper()}_API_KEY"


@router.post("/{name}/key")
async def set_provider_key(name: str, body: ApiKeyBody):
    """Set API key for a provider (saves to .env file and env vars)."""
    import re
    if not re.match(r'^[a-z_]{1,32}$', name):
        raise HTTPException(400, "Invalid provider name")
    env_var = _resolve_env_var(name)

    # Set in current process environment
    os.environ[env_var] = body.key

    # Persist to .env file
    _update_env_file(env_var, body.key)

    logger.info(f"[PROVIDERS] API key set for {name} ({env_var})")
    return {"ok": True, "provider": name, "env_var": env_var}


@router.delete("/{name}/key")
async def remove_provider_key(name: str):
    """Remove API key for a provider."""
    import re
    if not re.match(r"^[a-zA-Z0-9_-]{1,50}$", name):
        raise HTTPException(400, "Invalid provider name")
    env_var = _resolve_env_var(name)

    # Remove from current process environment
    os.environ.pop(env_var, None)

    # Remove from .env file
    _update_env_file(env_var, None)

    logger.info(f"[PROVIDERS] API key removed for {name} ({env_var})")
    return {"ok": True, "provider": name, "env_var": env_var}


def _update_env_file(env_var: str, value: str | None):
    """Add, update, or remove an env var in the project .env file."""
    # SECURITY: Reject newlines to prevent .env injection
    if value is not None and ('\n' in value or '\r' in value):
        raise HTTPException(400, "API key must not contain newline characters")
    # Strip any embedded quotes to prevent .env parsing issues
    if value is not None:
        value = value.strip().strip('"').strip("'")

    env_path = Path(__file__).resolve().parent.parent.parent / ".env"

    lines = []
    if env_path.exists():
        lines = env_path.read_text(encoding="utf-8").splitlines()

    # Find and update/remove existing line
    found = False
    new_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(f"{env_var}=") or stripped.startswith(f"{env_var} ="):
            found = True
            if value is not None:
                new_lines.append(f'{env_var}="{value}"')
            # else: skip line (remove)
        else:
            new_lines.append(line)

    # Append if not found and value is set
    if not found and value is not None:
        new_lines.append(f'{env_var}="{value}"')

    env_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
