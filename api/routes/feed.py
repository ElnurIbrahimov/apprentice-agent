"""Extension Feed endpoints.

Receives captured website data from the Chrome extension and stores it
in ``aura_data/extension_feed/`` so the CLI agent loop can consume it
as design context for code generation.

Endpoints:
    POST   /api/feed/save       - save a capture from the extension
    GET    /api/feed/list       - list recent feed items
    GET    /api/feed/{item_id}  - get a specific feed item
    DELETE /api/feed/{item_id}  - delete a feed item
"""

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from api.auth import require_api_key

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/feed",
    tags=["feed"],
    dependencies=[Depends(require_api_key)],
)

# ---------------------------------------------------------------------------
# Feed directory
# ---------------------------------------------------------------------------
_AURA_ROOT = Path(__file__).resolve().parent.parent.parent  # D:\Aura
FEED_DIR = _AURA_ROOT / "aura_data" / "extension_feed"
FEED_DIR.mkdir(parents=True, exist_ok=True)

# Max items kept on disk (auto-prune oldest beyond this)
MAX_FEED_ITEMS = 200


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
class PageMetadata(BaseModel):
    title: str = ""
    description: str = ""
    og_image: str = ""
    og_title: str = ""
    og_description: str = ""
    og_type: str = ""
    og_site_name: str = ""
    favicon: str = ""


class CapturePayload(BaseModel):
    type: str = Field(..., description="'component' or 'page'")
    html: str = ""
    css: Optional[str] = ""
    css_map: Optional[dict] = None
    screenshot_b64: str = ""
    colors: List[str] = Field(default_factory=list)
    fonts: List[str] = Field(default_factory=list)
    metadata: Optional[PageMetadata] = None
    source_url: str = ""
    timestamp: Optional[float] = None
    # Component-specific fields
    tag_name: str = ""
    class_name: str = ""
    dimensions: Optional[dict] = None
    text_content: str = ""
    # Full-page-specific fields
    viewport: Optional[dict] = None
    asset_urls: Optional[dict] = None
    responsive_info: Optional[dict] = None
    element_count: int = 0


class FeedItemSummary(BaseModel):
    id: str
    type: str
    source_url: str
    title: str
    timestamp: float
    has_screenshot: bool
    element_count: int = 0
    size_bytes: int = 0


class FeedListResponse(BaseModel):
    items: List[FeedItemSummary]
    total: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _prune_old_items() -> None:
    """Remove oldest items if we exceed MAX_FEED_ITEMS."""
    try:
        files = sorted(FEED_DIR.glob("capture_*.json"), key=lambda f: f.stat().st_mtime)
        if len(files) > MAX_FEED_ITEMS:
            for f in files[: len(files) - MAX_FEED_ITEMS]:
                f.unlink(missing_ok=True)
    except Exception as e:
        logger.warning(f"[Feed] Prune error: {e}")


def _item_id_from_path(p: Path) -> str:
    return p.stem  # e.g. "capture_1710000000000"


def _read_item(p: Path) -> dict:
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@router.post("/save")
async def save_capture(payload: CapturePayload):
    """Save a capture from the extension to the feed directory."""
    ts = payload.timestamp or time.time()
    item_id = f"capture_{int(ts * 1000)}"
    filepath = FEED_DIR / f"{item_id}.json"

    data = payload.model_dump()
    data["timestamp"] = ts
    data["id"] = item_id

    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
    except Exception as e:
        logger.error(f"[Feed] Save error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save capture: {e}")

    _prune_old_items()

    size = filepath.stat().st_size
    logger.info(f"[Feed] Saved {item_id} ({payload.type}, {size:,} bytes) from {payload.source_url}")

    return {
        "ok": True,
        "id": item_id,
        "size_bytes": size,
        "path": str(filepath),
    }


@router.get("/list", response_model=FeedListResponse)
async def list_feed(limit: int = 50, offset: int = 0):
    """List recent feed items, newest first."""
    files = sorted(FEED_DIR.glob("capture_*.json"), key=lambda f: f.stat().st_mtime, reverse=True)
    total = len(files)
    page = files[offset : offset + limit]

    items: List[FeedItemSummary] = []
    for p in page:
        try:
            data = _read_item(p)
            items.append(FeedItemSummary(
                id=data.get("id", _item_id_from_path(p)),
                type=data.get("type", "unknown"),
                source_url=data.get("source_url", ""),
                title=(data.get("metadata") or {}).get("title", "") or data.get("source_url", ""),
                timestamp=data.get("timestamp", 0),
                has_screenshot=bool(data.get("screenshot_b64")),
                element_count=data.get("element_count", 0),
                size_bytes=p.stat().st_size,
            ))
        except Exception as e:
            logger.warning(f"[Feed] Error reading {p.name}: {e}")

    return FeedListResponse(items=items, total=total)


def _validate_feed_id(item_id: str) -> None:
    """Reject item_ids that could cause path traversal."""
    if not re.match(r'^[a-zA-Z0-9_-]+$', item_id):
        raise HTTPException(status_code=400, detail="Invalid feed item ID")


@router.get("/{item_id}")
async def get_feed_item(item_id: str):
    """Get a specific feed item by ID."""
    _validate_feed_id(item_id)
    filepath = FEED_DIR / f"{item_id}.json"
    if not filepath.exists():
        raise HTTPException(status_code=404, detail=f"Feed item '{item_id}' not found")
    try:
        data = _read_item(filepath)
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading feed item: {e}")


@router.delete("/{item_id}")
async def delete_feed_item(item_id: str):
    """Delete a specific feed item."""
    _validate_feed_id(item_id)
    filepath = FEED_DIR / f"{item_id}.json"
    if not filepath.exists():
        raise HTTPException(status_code=404, detail=f"Feed item '{item_id}' not found")
    try:
        filepath.unlink()
        logger.info(f"[Feed] Deleted {item_id}")
        return {"ok": True, "id": item_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting feed item: {e}")
