"""
Screenpipe Client - Interface to Screenpipe screen & audio capture.

Screenpipe runs locally, capturing screen (OCR) and audio (transcription)
24/7. This client communicates with its REST API at localhost:3030.

Install Screenpipe: https://screenpi.pe/
Docs: https://docs.screenpi.pe/

Author: Aura Development Team
Created: 2026-02-07
"""

import hashlib
import logging
import re
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# Perceptual hashing for visual change detection
try:
    import imagehash
    from PIL import Image
    IMAGEHASH_AVAILABLE = True
except ImportError:
    IMAGEHASH_AVAILABLE = False
    logger.debug("[Screenpipe] imagehash not installed, visual change detection disabled")

# Try httpx first (async-capable), fall back to requests
try:
    import httpx
    _HTTP_CLIENT = "httpx"
except ImportError:
    try:
        import requests
        _HTTP_CLIENT = "requests"
    except ImportError:
        _HTTP_CLIENT = None
        logger.warning("Neither httpx nor requests installed. Screenpipe client unavailable.")


class ScreenpipeClient:
    """
    Client for the Screenpipe REST API.

    Screenpipe captures screen content (via OCR) and audio (via transcription)
    continuously. This client queries the local API to retrieve that data.

    Usage:
        client = ScreenpipeClient()
        if client.health_check().get("healthy"):
            results = client.search(query="error", content_type="ocr", limit=5)
    """

    DEFAULT_HOST = "http://localhost:3030"
    REQUEST_TIMEOUT = 5  # seconds

    # Privacy: apps/windows to never capture content from
    DEFAULT_PRIVACY_APPS = {
        "1password", "lastpass", "bitwarden", "keepass", "keychain",
        "authenticator", "authy",
    }
    DEFAULT_PRIVACY_WINDOW_PATTERNS = [
        r"password", r"private\s*browsing", r"incognito",
        r"secret", r"credential", r"vault", r"keychain",
        r"bank", r"credit\s*card",
    ]

    def __init__(self, host: Optional[str] = None):
        """
        Initialize Screenpipe client.

        Args:
            host: Screenpipe API base URL (default: http://localhost:3030)
        """
        self.host = (host or self.DEFAULT_HOST).rstrip("/")
        self._healthy: Optional[bool] = None
        self._last_health_check: float = 0
        self._health_cache_ttl = 30  # Re-check health every 30s

        # Delta detection (Phase 5A): track content hashes to detect changes
        self._last_content_hash: str = ""
        self._last_app_name: str = ""
        self._delta_threshold: float = 0.3  # 30% content change = significant

        # Perceptual hashing for visual change detection
        self._last_perceptual_hash = None  # imagehash.ImageHash or None

        # Privacy filtering (Phase 5A)
        self._privacy_apps: Set[str] = set(self.DEFAULT_PRIVACY_APPS)
        self._privacy_window_patterns: List[str] = list(self.DEFAULT_PRIVACY_WINDOW_PATTERNS)

        if _HTTP_CLIENT is None:
            raise ImportError("Install httpx or requests for Screenpipe support")

    def _get(self, path: str, params: Optional[Dict] = None) -> Dict[str, Any]:
        """Make a GET request to the Screenpipe API."""
        url = f"{self.host}{path}"
        if params:
            # Remove None values
            params = {k: v for k, v in params.items() if v is not None}

        try:
            if _HTTP_CLIENT == "httpx":
                with httpx.Client(timeout=self.REQUEST_TIMEOUT) as client:
                    response = client.get(url, params=params)
                    response.raise_for_status()
                    return response.json()
            else:
                import requests
                response = requests.get(url, params=params, timeout=self.REQUEST_TIMEOUT)
                response.raise_for_status()
                return response.json()
        except Exception as e:
            logger.debug(f"[Screenpipe] GET {path} failed: {e}")
            raise

    # =========================================================================
    # HEALTH & STATUS
    # =========================================================================

    def health_check(self) -> Dict[str, Any]:
        """
        Check if Screenpipe is running and healthy.

        Returns:
            Dict with "healthy" bool and status info.
            Caches result for 30 seconds.
        """
        now = time.time()
        if self._healthy is not None and (now - self._last_health_check) < self._health_cache_ttl:
            return {"healthy": self._healthy, "cached": True}

        try:
            data = self._get("/health")
            self._healthy = True
            self._last_health_check = now
            return {
                "healthy": True,
                "status": data.get("status", "ok"),
                "frame_status": data.get("frame_status"),
                "audio_status": data.get("audio_status"),
                "raw": data,
            }
        except Exception as e:
            self._healthy = False
            self._last_health_check = now
            return {"healthy": False, "error": str(e)}

    def is_available(self) -> bool:
        """Quick check if Screenpipe is reachable."""
        return self.health_check().get("healthy", False)

    # =========================================================================
    # SEARCH
    # =========================================================================

    def search(
        self,
        query: Optional[str] = None,
        content_type: str = "all",
        limit: int = 10,
        offset: int = 0,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        app_name: Optional[str] = None,
        window_name: Optional[str] = None,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search Screenpipe captured data.

        Args:
            query: Text to search for (optional, returns recent if None)
            content_type: "ocr", "audio", or "all"
            limit: Max results to return
            offset: Pagination offset
            start_time: ISO timestamp for start of time range
            end_time: ISO timestamp for end of time range
            app_name: Filter by application name
            window_name: Filter by window title
            min_length: Minimum text length
            max_length: Maximum text length

        Returns:
            List of result dicts with normalized keys:
            - type: "ocr" or "audio"
            - text: The captured text content
            - app_name: Application name
            - window_name: Window title
            - timestamp: ISO timestamp
            - file_path: Path to screenshot/audio file
        """
        params = {
            "q": query,
            "content_type": content_type,
            "limit": limit,
            "offset": offset,
            "start_time": start_time,
            "end_time": end_time,
            "app_name": app_name,
            "window_name": window_name,
            "min_length": min_length,
            "max_length": max_length,
        }

        try:
            data = self._get("/search", params)
        except Exception:
            return []

        # Normalize response format
        results = []
        for item in data.get("data", []):
            normalized = self._normalize_result(item)
            if normalized:
                results.append(normalized)

        return results

    def search_recent(
        self,
        minutes: int = 5,
        content_type: str = "ocr",
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search recent screen/audio content.

        Convenience method for getting recent captures.

        Args:
            minutes: How many minutes back to search
            content_type: "ocr", "audio", or "all"
            limit: Max results
        """
        start = (datetime.now() - timedelta(minutes=minutes)).isoformat()
        return self.search(
            content_type=content_type,
            limit=limit,
            start_time=start
        )

    def get_current_screen_text(self) -> Optional[str]:
        """
        Get the most recent OCR text from the screen.

        Returns:
            The latest screen text, or None if unavailable.
        """
        results = self.search(content_type="ocr", limit=1)
        if results:
            return results[0].get("text")
        return None

    def get_current_app(self) -> Optional[Dict[str, str]]:
        """
        Get the currently active application info.

        Returns:
            Dict with app_name and window_name, or None.
        """
        results = self.search(content_type="ocr", limit=1)
        if results:
            return {
                "app_name": results[0].get("app_name", "Unknown"),
                "window_name": results[0].get("window_name", ""),
            }
        return None

    # =========================================================================
    # AUDIO
    # =========================================================================

    def search_audio(
        self,
        query: Optional[str] = None,
        minutes: int = 5,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search recent audio transcriptions.

        Args:
            query: Text to search in transcriptions
            minutes: How many minutes back
            limit: Max results
        """
        start = (datetime.now() - timedelta(minutes=minutes)).isoformat()
        return self.search(
            query=query,
            content_type="audio",
            limit=limit,
            start_time=start
        )

    # =========================================================================
    # CONTEXT BUILDING
    # =========================================================================

    def get_screen_context(
        self,
        minutes: int = 2,
        max_chars: int = 2000
    ) -> Dict[str, Any]:
        """
        Get a structured context summary of recent screen activity.

        Useful for injecting into LLM prompts for screen awareness.

        Args:
            minutes: How many minutes of history
            max_chars: Max total characters of text content

        Returns:
            Dict with:
            - current_app: Current app name
            - current_window: Current window title
            - recent_text: Recent OCR text (truncated)
            - apps_used: List of recent apps
            - has_errors: Whether error-like text was detected
        """
        results = self.search_recent(minutes=minutes, content_type="ocr", limit=20)

        if not results:
            return {
                "available": False,
                "current_app": None,
                "current_window": None,
                "recent_text": "",
                "apps_used": [],
                "has_errors": False,
            }

        # Current state (most recent)
        current = results[0]
        current_app = current.get("app_name", "Unknown")
        current_window = current.get("window_name", "")

        # Aggregate recent text
        texts = []
        total_chars = 0
        apps_seen = set()
        has_errors = False

        error_indicators = [
            "error", "exception", "traceback", "failed", "crash",
            "denied", "404", "500", "fatal", "critical", "ENOENT",
            "segfault", "panic", "undefined", "null reference",
        ]

        for r in results:
            app = r.get("app_name", "")
            if app:
                apps_seen.add(app)

            text = r.get("text", "")
            if text and total_chars < max_chars:
                texts.append(text[:500])
                total_chars += len(text[:500])

                # Check for errors
                text_lower = text.lower()
                if any(err in text_lower for err in error_indicators):
                    has_errors = True

        return {
            "available": True,
            "current_app": current_app,
            "current_window": current_window,
            "recent_text": "\n---\n".join(texts[:5]),
            "apps_used": list(apps_seen),
            "has_errors": has_errors,
            "result_count": len(results),
        }

    # =========================================================================
    # DELTA DETECTION (Phase 5A)
    # =========================================================================

    def _compute_perceptual_hash(self, file_path: str):
        """Compute dHash perceptual hash for an image file.

        Uses dHash (difference hash) for fast visual similarity detection.
        Hash size 16 gives 256-bit hash for good granularity.

        Returns:
            imagehash.ImageHash object, or None on failure.
        """
        if not IMAGEHASH_AVAILABLE or not file_path:
            return None
        try:
            img = Image.open(file_path)
            return imagehash.dhash(img, hash_size=16)
        except Exception as e:
            logger.debug(f"[Screenpipe] Perceptual hash failed for {file_path}: {e}")
            return None

    def has_significant_change(self, minutes: int = 1) -> Dict[str, Any]:
        """
        Check if screen content has changed significantly since last check.

        Uses content hashing to detect meaningful changes, filtering out
        minor cursor blinks, clock updates, etc.

        Returns:
            Dict with:
            - changed: bool - whether significant change detected
            - change_type: "app_switch", "content_change", "none"
            - current_app: current app name
            - previous_app: previous app name
        """
        results = self.search_recent(minutes=minutes, content_type="ocr", limit=3)
        if not results:
            return {"changed": False, "change_type": "none", "current_app": None}

        current = results[0]
        current_app = current.get("app_name", "")
        current_text = current.get("text", "")

        # Privacy filter
        if self._is_private(current_app, current.get("window_name", "")):
            return {"changed": False, "change_type": "private", "current_app": current_app}

        # Compute content hash (first 500 chars, normalized)
        normalized = re.sub(r'\s+', ' ', current_text[:500].lower().strip())
        content_hash = hashlib.md5(normalized.encode()).hexdigest()

        # Check for app switch
        if current_app != self._last_app_name and self._last_app_name:
            prev_app = self._last_app_name
            self._last_app_name = current_app
            self._last_content_hash = content_hash
            return {
                "changed": True,
                "change_type": "app_switch",
                "current_app": current_app,
                "previous_app": prev_app,
            }

        # Check for text content change
        text_changed = content_hash != self._last_content_hash and self._last_content_hash != ""
        self._last_app_name = current_app
        self._last_content_hash = content_hash

        # Perceptual hash for visual change detection
        visual_changed = False
        visual_distance = 0
        file_path = current.get("file_path", "")
        if file_path:
            new_hash = self._compute_perceptual_hash(file_path)
            if new_hash is not None and self._last_perceptual_hash is not None:
                visual_distance = new_hash - self._last_perceptual_hash
                from aura.config import Config
                visual_changed = visual_distance > Config.PHASH_CHANGE_THRESHOLD
            self._last_perceptual_hash = new_hash

        # Combine text and visual signals
        if text_changed and visual_changed:
            change_type = "major_change"
        elif visual_changed:
            change_type = "visual_change"
        elif text_changed:
            change_type = "content_change"
        else:
            change_type = "none"

        changed = text_changed or visual_changed

        return {
            "changed": changed,
            "change_type": change_type,
            "current_app": current_app,
            "previous_app": current_app,
            "visual_distance": visual_distance,
        }

    # =========================================================================
    # PRIVACY FILTERING (Phase 5A)
    # =========================================================================

    def add_privacy_app(self, app_name: str):
        """Add an app to the privacy filter list."""
        self._privacy_apps.add(app_name.lower())

    def add_privacy_window_pattern(self, pattern: str):
        """Add a window title regex pattern to the privacy filter."""
        self._privacy_window_patterns.append(pattern)

    def _is_private(self, app_name: str, window_name: str) -> bool:
        """
        Check if the given app/window should be privacy-filtered.

        Args:
            app_name: Application name
            window_name: Window title

        Returns:
            True if this content should be filtered out for privacy.
        """
        app_lower = app_name.lower()
        window_lower = window_name.lower()

        # Check app name
        if app_lower in self._privacy_apps:
            return True

        # Check window title patterns
        for pattern in self._privacy_window_patterns:
            if re.search(pattern, window_lower, re.IGNORECASE):
                return True

        return False

    def get_screen_context_filtered(
        self,
        minutes: int = 2,
        max_chars: int = 2000,
        only_if_changed: bool = False,
    ) -> Dict[str, Any]:
        """
        Get screen context with privacy filtering and optional delta detection.

        Enhanced version of get_screen_context that:
        - Filters out private apps/windows
        - Optionally only returns data if content has significantly changed

        Args:
            minutes: How many minutes of history
            max_chars: Max total characters
            only_if_changed: Only return context if significant change detected
        """
        # Check delta first if requested
        if only_if_changed:
            delta = self.has_significant_change()
            if not delta.get("changed"):
                return {
                    "available": True,
                    "changed": False,
                    "current_app": delta.get("current_app"),
                    "reason": "no_significant_change",
                }

        results = self.search_recent(minutes=minutes, content_type="ocr", limit=20)
        if not results:
            return {
                "available": False,
                "current_app": None,
                "current_window": None,
                "recent_text": "",
                "apps_used": [],
                "has_errors": False,
            }

        # Filter out private results
        filtered = []
        for r in results:
            if not self._is_private(r.get("app_name", ""), r.get("window_name", "")):
                filtered.append(r)

        if not filtered:
            return {
                "available": True,
                "current_app": "[private]",
                "current_window": "",
                "recent_text": "",
                "apps_used": [],
                "has_errors": False,
                "privacy_filtered": True,
            }

        # Current state
        current = filtered[0]
        current_app = current.get("app_name", "Unknown")
        current_window = current.get("window_name", "")

        # Aggregate
        texts = []
        total_chars = 0
        apps_seen = set()
        has_errors = False

        error_indicators = [
            "error", "exception", "traceback", "failed", "crash",
            "denied", "404", "500", "fatal", "critical", "ENOENT",
            "segfault", "panic", "undefined", "null reference",
        ]

        for r in filtered:
            app = r.get("app_name", "")
            if app:
                apps_seen.add(app)

            text = r.get("text", "")
            if text and total_chars < max_chars:
                texts.append(text[:500])
                total_chars += len(text[:500])
                text_lower = text.lower()
                if any(err in text_lower for err in error_indicators):
                    has_errors = True

        return {
            "available": True,
            "changed": True,
            "current_app": current_app,
            "current_window": current_window,
            "recent_text": "\n---\n".join(texts[:5]),
            "apps_used": list(apps_seen),
            "has_errors": has_errors,
            "result_count": len(filtered),
            "privacy_filtered_count": len(results) - len(filtered),
        }

    # =========================================================================
    # INTERNAL
    # =========================================================================

    def _normalize_result(self, item: Dict) -> Optional[Dict[str, Any]]:
        """
        Normalize a Screenpipe search result to a consistent format.

        Screenpipe returns different shapes for OCR vs Audio results.
        """
        item_type = item.get("type", "").lower()
        content = item.get("content", {})

        if item_type == "ocr":
            return {
                "type": "ocr",
                "text": content.get("text", ""),
                "app_name": content.get("app_name", "Unknown"),
                "window_name": content.get("window_name", ""),
                "timestamp": content.get("timestamp", ""),
                "file_path": content.get("file_path", ""),
                "frame_id": content.get("frame_id"),
            }
        elif item_type == "audio":
            return {
                "type": "audio",
                "text": content.get("transcription", ""),
                "app_name": content.get("device_name", "microphone"),
                "window_name": "",
                "timestamp": content.get("timestamp", ""),
                "file_path": content.get("file_path", ""),
                "duration": content.get("duration_secs"),
                "speaker_id": content.get("speaker_id"),
            }
        else:
            # Unknown type, try best effort
            return {
                "type": item_type or "unknown",
                "text": content.get("text", content.get("transcription", "")),
                "app_name": content.get("app_name", ""),
                "window_name": content.get("window_name", ""),
                "timestamp": content.get("timestamp", ""),
                "file_path": content.get("file_path", ""),
            }


# Singleton
_client_instance: Optional[ScreenpipeClient] = None


def get_screenpipe_client(host: Optional[str] = None) -> ScreenpipeClient:
    """Get or create the global Screenpipe client."""
    global _client_instance
    if _client_instance is None:
        _client_instance = ScreenpipeClient(host=host)
    return _client_instance


# Export
__all__ = [
    "ScreenpipeClient",
    "get_screenpipe_client",
]
