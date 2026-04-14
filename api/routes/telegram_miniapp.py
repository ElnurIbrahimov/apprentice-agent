"""Telegram Mini App — initData validation endpoint.

Validates the cryptographic hash Telegram includes in initData so the
backend can verify a request genuinely comes from the Mini App.

See: https://core.telegram.org/bots/webapps#validating-data-received-via-the-mini-app
"""

import hashlib
import hmac
import json
import logging
import os
import time
from urllib.parse import parse_qs

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# This router is intentionally NOT gated by require_api_key: the Mini App runs
# in a Telegram WebView that has no way to carry a pre-shared X-API-Key without
# leaking it to the client. Authentication for these endpoints is the HMAC-SHA256
# signature Telegram puts in initData — verified below via _validate_init_data.
# Must also be listed in api/middleware.py PUBLIC_PATHS for the middleware to
# skip it.
router = APIRouter(
    prefix="/api/telegram",
    tags=["telegram-miniapp"],
)


class InitDataRequest(BaseModel):
    init_data: str  # raw window.Telegram.WebApp.initData string


class InitDataResponse(BaseModel):
    valid: bool
    user_id: int | None = None
    first_name: str | None = None
    username: str | None = None
    auth_date: int | None = None


class ProactiveActionRequest(BaseModel):
    init_data: str                     # Telegram Mini App initData for HMAC auth
    action_id: str                     # "ack" | "more" | "snooze_3600" | ...
    hand_name: str


def _validate_init_data(init_data: str, bot_token: str) -> dict | None:
    """Validate Telegram Mini App initData using HMAC-SHA256.

    Returns parsed data dict if valid, None if invalid.
    """
    if not init_data or not bot_token:
        return None

    try:
        parsed = parse_qs(init_data, keep_blank_values=True)
        # Extract hash
        received_hash = parsed.get("hash", [None])[0]
        if not received_hash:
            return None

        # Build data-check-string: sorted key=value pairs, excluding "hash"
        data_pairs = []
        for key, values in parsed.items():
            if key == "hash":
                continue
            data_pairs.append(f"{key}={values[0]}")
        data_pairs.sort()
        data_check_string = "\n".join(data_pairs)

        # HMAC-SHA256: secret_key = HMAC-SHA256("WebAppData", bot_token)
        secret_key = hmac.new(
            b"WebAppData", bot_token.encode(), hashlib.sha256
        ).digest()

        # Compute hash
        computed_hash = hmac.new(
            secret_key, data_check_string.encode(), hashlib.sha256
        ).hexdigest()

        if not hmac.compare_digest(computed_hash, received_hash):
            return None

        # Check auth_date freshness (reject if older than 1 hour)
        auth_date = int(parsed.get("auth_date", [0])[0])
        if time.time() - auth_date > 3600:
            logger.warning(f"[MiniApp] initData too old: auth_date={auth_date}")
            return None

        # Parse user object
        user_raw = parsed.get("user", [None])[0]
        user = json.loads(user_raw) if user_raw else {}

        return {
            "user_id": user.get("id"),
            "first_name": user.get("first_name"),
            "username": user.get("username"),
            "auth_date": auth_date,
        }

    except Exception as e:
        logger.error(f"[MiniApp] initData validation error: {e}")
        return None


@router.post("/validate-init", response_model=InitDataResponse)
async def validate_init_data(request: InitDataRequest):
    """Validate Telegram Mini App initData.

    The Mini App should call this on startup to prove it's running
    inside Telegram and get the authenticated user info.
    """
    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    if not bot_token:
        raise HTTPException(status_code=500, detail="Bot token not configured")

    result = _validate_init_data(request.init_data, bot_token)
    if result is None:
        return InitDataResponse(valid=False)

    return InitDataResponse(
        valid=True,
        user_id=result.get("user_id"),
        first_name=result.get("first_name"),
        username=result.get("username"),
        auth_date=result.get("auth_date"),
    )


@router.post("/proactive/action")
async def proactive_action(request: ProactiveActionRequest):
    """Apply a proactive-card action from the Mini App (ack / more / snooze).

    Authenticated via Telegram Mini App initData HMAC — no API key required
    because the Telegram WebView can't carry one. The HMAC proves the request
    originated from a legitimate Mini App session.
    """
    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    if not bot_token:
        raise HTTPException(status_code=500, detail="Bot token not configured")

    if _validate_init_data(request.init_data, bot_token) is None:
        raise HTTPException(status_code=401, detail="Invalid or expired initData")

    action = request.action_id
    hand_name = request.hand_name
    if not hand_name:
        raise HTTPException(status_code=400, detail="hand_name is required")

    try:
        from aura.hands.manager import get_hand_manager
        manager = get_hand_manager()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Hand manager unavailable: {exc}") from exc

    if action == "ack":
        return {"status": "acknowledged", "hand": hand_name}

    if action == "more":
        # Queue a follow-up run via push-trigger; results flow back through
        # the normal notify_hand_result path (Telegram + Mini App card).
        triggered = await manager.trigger_hand_async(
            hand_name,
            context={"source": "miniapp:more"},
        )
        if not triggered:
            raise HTTPException(
                status_code=409,
                detail=f"Could not trigger '{hand_name}' (unknown or snoozed)",
            )
        return {"status": "queued", "hand": hand_name}

    if action.startswith("snooze_"):
        try:
            seconds = int(action.split("_", 1)[1])
        except (IndexError, ValueError) as exc:
            raise HTTPException(status_code=400, detail="Invalid snooze action") from exc
        if seconds <= 0 or seconds > 7 * 24 * 3600:
            raise HTTPException(status_code=400, detail="Snooze out of range")
        if not manager.snooze(hand_name, seconds):
            raise HTTPException(status_code=404, detail=f"Unknown hand: {hand_name}")
        return {"status": "snoozed", "hand": hand_name, "seconds": seconds}

    raise HTTPException(status_code=400, detail=f"Unknown action: {action}")


# ============================================================================
# Memory Browser — Mini App public proxy
# ============================================================================
#
# These endpoints sit on the public router (no X-API-Key required) because the
# Telegram WebView can't carry a pre-shared header. Each handler validates the
# request's initData HMAC before delegating to UnifiedMemory.
#
# They mirror the authenticated endpoints in api/routes/memory.py but as POSTs
# carrying init_data in the body, so no query params leak to access logs.


class MemoryBrowseBody(BaseModel):
    init_data: str
    offset: int = 0
    limit: int = 50
    source: str | None = None


class MemoryItemRefBody(BaseModel):
    init_data: str
    memory_id: str


class MemoryPatchItemBody(BaseModel):
    init_data: str
    memory_id: str
    content: str | None = None
    tags: list[str] | None = None
    importance: float | None = None


class MemoryPinBody(BaseModel):
    init_data: str
    memory_id: str
    pinned: bool


class InitDataOnlyBody(BaseModel):
    init_data: str


class MemoryKgTopBody(BaseModel):
    init_data: str
    limit: int = 20


def _require_init_data(init_data: str) -> None:
    """Validate Telegram initData HMAC or raise 401."""
    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    if not bot_token:
        raise HTTPException(status_code=500, detail="Bot token not configured")
    if _validate_init_data(init_data, bot_token) is None:
        raise HTTPException(status_code=401, detail="Invalid or expired initData")


def _get_um():
    from aura.memory.unified_memory import get_unified_memory
    return get_unified_memory()


@router.post("/memory/browse")
async def miniapp_memory_browse(body: MemoryBrowseBody):
    """Paginated memory timeline for the Brain tab."""
    _require_init_data(body.init_data)
    if body.limit <= 0 or body.limit > 200:
        raise HTTPException(status_code=400, detail="limit out of range")
    if body.offset < 0:
        raise HTTPException(status_code=400, detail="offset must be >= 0")
    um = _get_um()
    items = um.list_recent(
        offset=body.offset,
        limit=body.limit,
        source_filter=body.source,
    )
    total = um.count_memories(source_filter=body.source)
    return {
        "items": items,
        "total": total,
        "offset": body.offset,
        "limit": body.limit,
        "source": body.source,
    }


@router.post("/memory/item/get")
async def miniapp_memory_get(body: MemoryItemRefBody):
    """Fetch a single memory by id."""
    _require_init_data(body.init_data)
    row = _get_um().get_memory(body.memory_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Memory not found")
    return row


@router.post("/memory/item/patch")
async def miniapp_memory_patch(body: MemoryPatchItemBody):
    """Update content / tags / importance on a memory."""
    _require_init_data(body.init_data)
    fields: dict = {}
    if body.content is not None:
        fields["content"] = body.content[:50000]
    if body.tags is not None:
        fields["tags"] = [str(t)[:100] for t in body.tags[:50]]
    if body.importance is not None:
        fields["importance"] = max(0.0, min(1.0, float(body.importance)))
    if not fields:
        raise HTTPException(status_code=400, detail="No fields to update")
    ok = _get_um().update_memory(body.memory_id, **fields)
    if not ok:
        raise HTTPException(status_code=404, detail="Memory not found")
    return {"id": body.memory_id, "status": "updated"}


@router.post("/memory/item/delete")
async def miniapp_memory_delete(body: MemoryItemRefBody):
    """Hard-delete a memory by id."""
    _require_init_data(body.init_data)
    ok = _get_um().delete_memory(body.memory_id)
    return {"deleted": bool(ok), "id": body.memory_id}


@router.post("/memory/item/pin")
async def miniapp_memory_pin(body: MemoryPinBody):
    """Pin or unpin a memory (stored as a tag for v1)."""
    _require_init_data(body.init_data)
    ok = _get_um().set_pinned(body.memory_id, bool(body.pinned))
    if not ok:
        raise HTTPException(status_code=404, detail="Memory not found")
    return {"id": body.memory_id, "pinned": bool(body.pinned)}


@router.post("/memory/stats")
async def miniapp_memory_stats(body: InitDataOnlyBody):
    """Memory tab header stats."""
    _require_init_data(body.init_data)
    um = _get_um()
    stats = um.get_stats() or {}
    stats["sources"] = um.list_sources()
    stats["total_count"] = um.count_memories()
    return stats


@router.post("/memory/kg/top")
async def miniapp_memory_kg_top(body: MemoryKgTopBody):
    """Top knowledge-graph entities for the Brain tab."""
    _require_init_data(body.init_data)
    limit = max(1, min(100, int(body.limit)))
    try:
        from api.services.agent_service import agent_service
        agent = getattr(agent_service, "agent", None)
        if not agent:
            return {"nodes": [], "total_entities": 0, "total_relationships": 0}
        kg = getattr(agent, "kg_brain", None)
        stats = {}
        if kg is not None and hasattr(kg, "get_statistics"):
            try:
                stats = kg.get_statistics() or {}
            except Exception:
                stats = {}
        nodes: list[dict] = []
        kg_tool = (getattr(agent, "tools", None) or {}).get("knowledge_graph")
        if kg_tool is not None:
            graph = getattr(kg_tool, "_graph", None) or getattr(kg_tool, "graph", None)
            raw_nodes = getattr(graph, "nodes", None) if graph else None
            if raw_nodes:
                items = list(raw_nodes.items() if hasattr(raw_nodes, "items") else raw_nodes)
                for entry in items[: limit * 3]:
                    if isinstance(entry, tuple) and len(entry) == 2:
                        nid, ndata = entry
                    else:
                        nid, ndata = entry, {}
                    if not isinstance(ndata, dict):
                        ndata = {"label": str(ndata)}
                    nodes.append({
                        "id": str(nid),
                        "label": str(ndata.get("label") or nid),
                        "type": str(ndata.get("type") or "entity"),
                        "confidence": float(ndata.get("confidence", 0.0) or 0.0),
                        "access_count": int(ndata.get("access_count", 0) or 0),
                    })
                nodes.sort(key=lambda n: (n["access_count"], n["confidence"]), reverse=True)
                nodes = nodes[:limit]
        return {
            "nodes": nodes,
            "total_entities": int(stats.get("total_entities", 0) or 0),
            "total_relationships": int(stats.get("total_relationships", 0) or 0),
            "type_distribution": stats.get("entity_type_distribution", {}) or {},
        }
    except Exception as exc:
        logger.debug(f"[MiniApp] kg/top error: {exc}")
        return {"nodes": [], "total_entities": 0, "total_relationships": 0}
