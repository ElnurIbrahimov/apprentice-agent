"""Extension Feed Tool — reads captured website data from the extension.

The Chrome extension captures full-page or component designs from any website
and saves them to ``aura_data/extension_feed/``.  This tool lets the CLI agent
consume those captures as context for code generation.

Usage in agent loop:
    from aura.tools.extension_feed import ExtensionFeedTool, get_feed_tool

    feed = get_feed_tool()
    items = feed.list_feed()
    if items:
        latest = feed.get_feed(items[0]["id"])
        # Use latest["html"], latest["css"], latest["screenshot_b64"], etc.
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_AURA_ROOT = Path(__file__).resolve().parent.parent.parent  # D:\Aura
FEED_DIR = _AURA_ROOT / "aura_data" / "extension_feed"
FEED_DIR.mkdir(parents=True, exist_ok=True)

# Singleton
_instance: Optional["ExtensionFeedTool"] = None


def get_feed_tool() -> "ExtensionFeedTool":
    global _instance
    if _instance is None:
        _instance = ExtensionFeedTool()
    return _instance


class ExtensionFeedTool:
    """Reads and manages captured website data from the extension feed."""

    name = "extension_feed"
    description = "Access captured website designs from the browser extension"

    def __init__(self, feed_dir: Optional[Path] = None):
        self.feed_dir = feed_dir or FEED_DIR
        self.feed_dir.mkdir(parents=True, exist_ok=True)
        self._last_seen_ts: float = 0.0

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def list_feed(self, limit: int = 20) -> List[Dict[str, Any]]:
        """List recent feed items, newest first.

        Returns a list of summary dicts with keys:
            id, type, source_url, title, timestamp, has_screenshot, size_bytes
        """
        files = sorted(
            self.feed_dir.glob("capture_*.json"),
            key=lambda f: f.stat().st_mtime,
            reverse=True,
        )[:limit]

        items: List[Dict[str, Any]] = []
        for p in files:
            try:
                data = self._read(p)
                meta = data.get("metadata") or {}
                items.append({
                    "id": data.get("id", p.stem),
                    "type": data.get("type", "unknown"),
                    "source_url": data.get("source_url", ""),
                    "title": meta.get("title", "") or data.get("source_url", ""),
                    "timestamp": data.get("timestamp", 0),
                    "has_screenshot": bool(data.get("screenshot_b64")),
                    "element_count": data.get("element_count", 0),
                    "size_bytes": p.stat().st_size,
                })
            except Exception as e:
                logger.warning(f"[ExtFeed] Error reading {p.name}: {e}")
        return items

    def get_feed(self, item_id: str, include_screenshot: bool = True) -> Optional[Dict[str, Any]]:
        """Read a specific feed item by ID.

        Set include_screenshot=False to skip the base64 image (saves memory).
        """
        filepath = self.feed_dir / f"{item_id}.json"
        if not filepath.exists():
            return None
        try:
            data = self._read(filepath)
            if not include_screenshot:
                data.pop("screenshot_b64", None)
            return data
        except Exception as e:
            logger.error(f"[ExtFeed] Error reading {item_id}: {e}")
            return None

    def get_latest(self, include_screenshot: bool = True) -> Optional[Dict[str, Any]]:
        """Get the most recent feed item."""
        items = self.list_feed(limit=1)
        if not items:
            return None
        return self.get_feed(items[0]["id"], include_screenshot=include_screenshot)

    def clear_feed(self, older_than_hours: float = 0) -> int:
        """Clear feed items. If older_than_hours > 0, only clear items older than that."""
        cutoff = time.time() - (older_than_hours * 3600) if older_than_hours > 0 else float("inf")
        count = 0
        for p in self.feed_dir.glob("capture_*.json"):
            try:
                if p.stat().st_mtime < cutoff or older_than_hours == 0:
                    p.unlink()
                    count += 1
            except Exception as e:
                logger.warning(f"[ExtFeed] Error deleting {p.name}: {e}")
        logger.info(f"[ExtFeed] Cleared {count} items")
        return count

    def watch_feed(self, callback=None, poll_interval: float = 2.0, timeout: float = 60.0) -> Optional[Dict[str, Any]]:
        """Watch for new feed items. Blocks until a new item appears or timeout.

        Args:
            callback: Optional function called with each new item dict.
            poll_interval: Seconds between polls.
            timeout: Max seconds to wait.

        Returns:
            The first new item found, or None on timeout.
        """
        start = time.time()
        # Record current newest
        existing = {p.name for p in self.feed_dir.glob("capture_*.json")}

        while time.time() - start < timeout:
            time.sleep(poll_interval)
            current = set()
            for p in self.feed_dir.glob("capture_*.json"):
                current.add(p.name)
                if p.name not in existing:
                    try:
                        data = self._read(p)
                        if callback:
                            callback(data)
                        return data
                    except Exception as e:
                        logger.warning(f"[ExtFeed] Error reading new item {p.name}: {e}")
            existing = current

        return None

    def get_new_items(self) -> List[Dict[str, Any]]:
        """Get items newer than the last call to this method."""
        items = []
        for p in sorted(self.feed_dir.glob("capture_*.json"), key=lambda f: f.stat().st_mtime):
            ts = p.stat().st_mtime
            if ts > self._last_seen_ts:
                try:
                    data = self._read(p)
                    items.append(data)
                    self._last_seen_ts = ts
                except Exception:
                    pass
        return items

    # ------------------------------------------------------------------
    # Context builder for agent prompts
    # ------------------------------------------------------------------

    def build_context(self, item_id: str, max_html_chars: int = 15000, max_css_chars: int = 8000) -> str:
        """Build a context string from a feed item suitable for an LLM prompt.

        Returns a formatted string the agent can use to understand the captured design.
        """
        data = self.get_feed(item_id, include_screenshot=False)
        if not data:
            return ""

        capture_type = data.get("type", "unknown")
        source_url = data.get("source_url", "unknown")
        meta = data.get("metadata") or {}
        colors = data.get("colors", [])
        fonts = data.get("fonts", [])
        viewport = data.get("viewport") or {}

        parts = [
            f"=== Captured {capture_type.upper()} from {source_url} ===",
            "",
        ]

        if meta.get("title"):
            parts.append(f"Page title: {meta['title']}")
        if meta.get("description"):
            parts.append(f"Description: {meta['description']}")

        if viewport:
            parts.append(f"Viewport: {viewport.get('width', '?')}x{viewport.get('height', '?')}")

        if colors:
            parts.append(f"Color palette: {', '.join(colors[:20])}")

        if fonts:
            parts.append(f"Fonts: {', '.join(fonts[:10])}")

        parts.append("")

        # HTML
        html = data.get("html", "")
        if html:
            truncated = html[:max_html_chars]
            if len(html) > max_html_chars:
                truncated += f"\n... (truncated, {len(html):,} total chars)"
            parts.append("--- HTML Structure ---")
            parts.append(truncated)
            parts.append("")

        # CSS
        css = data.get("css", "")
        css_map = data.get("css_map")
        if css_map and isinstance(css_map, dict):
            css_lines = []
            for selector, props in css_map.items():
                if not isinstance(props, dict):
                    continue
                css_lines.append(f"{selector} {{")
                for prop, val in props.items():
                    css_lines.append(f"  {prop}: {val};")
                css_lines.append("}")
                css_lines.append("")
            css = "\n".join(css_lines)

        if css:
            truncated_css = css[:max_css_chars]
            if len(css) > max_css_chars:
                truncated_css += f"\n... (truncated, {len(css):,} total chars)"
            parts.append("--- Computed Styles ---")
            parts.append(truncated_css)
            parts.append("")

        # Asset URLs
        assets = data.get("asset_urls") or {}
        if assets:
            parts.append("--- Assets ---")
            for kind, urls in assets.items():
                if urls:
                    parts.append(f"  {kind}: {', '.join(urls[:5])}")
            parts.append("")

        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _read(path: Path) -> Dict[str, Any]:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
