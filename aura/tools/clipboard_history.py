"""Clipboard History tool — extended clipboard with history, search, pinning, and categories."""

import json
import logging
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)

MAX_HISTORY = 200
HISTORY_FILE = Path(__file__).parent.parent.parent / "data" / "clipboard_history.json"


import threading as _threading

_file_lock = _threading.Lock()


class ClipboardHistoryTool:
    """Extended clipboard manager with history, search, pinning, and categories."""

    name = "clipboard_history"
    description = "Clipboard history with search, pinning, and categories"

    def __init__(self):
        self._ensure_file()

    def _ensure_file(self):
        HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        if not HISTORY_FILE.exists():
            self._save_history({"entries": [], "pinned": []})

    def _load_history(self) -> dict:
        with _file_lock:
            try:
                with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return {"entries": [], "pinned": []}

    def _save_history(self, data: dict) -> bool:
        with _file_lock:
            try:
                # Enforce max history
                if len(data.get("entries", [])) > MAX_HISTORY:
                    data["entries"] = data["entries"][-MAX_HISTORY:]
                with open(HISTORY_FILE, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
                return True
            except IOError:
                return False

    def _get_clipboard(self) -> Optional[str]:
        """Read current system clipboard."""
        try:
            import pyperclip
            return pyperclip.paste()
        except ImportError:
            pass

        # Fallback: platform-specific
        import sys
        if sys.platform == "win32":
            try:
                import ctypes
                CF_TEXT = 1
                CF_UNICODETEXT = 13
                user32 = ctypes.windll.user32
                kernel32 = ctypes.windll.kernel32

                user32.OpenClipboard(0)
                try:
                    handle = user32.GetClipboardData(CF_UNICODETEXT)
                    if handle:
                        kernel32.GlobalLock.restype = ctypes.c_wchar_p
                        data = kernel32.GlobalLock(handle)
                        kernel32.GlobalUnlock(handle)
                        return data
                finally:
                    user32.CloseClipboard()
            except Exception:
                pass

        return None

    def _set_clipboard(self, text: str) -> bool:
        """Write to system clipboard."""
        try:
            import pyperclip
            pyperclip.copy(text)
            return True
        except ImportError:
            pass

        import sys
        if sys.platform == "win32":
            try:
                import subprocess
                process = subprocess.Popen(
                    ["clip"], stdin=subprocess.PIPE, shell=False
                )
                process.communicate(text.encode("utf-16-le"))
                return True
            except Exception:
                pass

        return False

    def _detect_category(self, text: str) -> str:
        """Auto-detect content category."""
        if not text:
            return "empty"

        text_stripped = text.strip()

        # URL
        if re.match(r'^https?://', text_stripped):
            return "url"

        # Email
        if re.match(r'^[\w.+-]+@[\w-]+\.[\w.-]+$', text_stripped):
            return "email"

        # File path
        if re.match(r'^([A-Za-z]:\\|/)', text_stripped) or re.match(r'^~/', text_stripped):
            return "path"

        # Code
        code_indicators = ['def ', 'class ', 'import ', 'function ', 'const ', 'var ', 'let ',
                           'if (', 'for (', '=> ', '<?php', '<html', '#include']
        if any(ind in text_stripped for ind in code_indicators):
            return "code"

        # JSON
        if (text_stripped.startswith("{") and text_stripped.endswith("}")) or \
           (text_stripped.startswith("[") and text_stripped.endswith("]")):
            try:
                json.loads(text_stripped)
                return "json"
            except json.JSONDecodeError:
                pass

        # Number
        if re.match(r'^-?\d+\.?\d*$', text_stripped):
            return "number"

        # Multi-line text
        if "\n" in text_stripped:
            return "multiline"

        return "text"

    # -- Core operations ----------------------------------------------------

    def capture(self) -> dict:
        """Capture current clipboard content and add to history."""
        content = self._get_clipboard()
        if content is None:
            return {"success": False, "error": "Could not read clipboard"}
        if not content.strip():
            return {"success": True, "message": "Clipboard is empty", "response": "Clipboard is empty"}

        category = self._detect_category(content)
        entry = {
            "id": uuid.uuid4().hex[:8],
            "content": content,
            "category": category,
            "timestamp": datetime.now().isoformat(),
            "length": len(content),
        }

        data = self._load_history()
        # Avoid duplicate consecutive entries
        if data["entries"] and data["entries"][-1].get("content") == content:
            return {
                "success": True,
                "duplicate": True,
                "entry_id": data["entries"][-1]["id"],
                "response": "Clipboard unchanged (duplicate)"
            }

        data["entries"].append(entry)
        self._save_history(data)

        return {
            "success": True,
            "entry_id": entry["id"],
            "category": category,
            "length": len(content),
            "preview": content[:100],
            "response": f"Captured [{category}]: {content[:80]}..."
        }

    def list_history(self, limit: int = 20, category: str = None) -> dict:
        """List clipboard history."""
        data = self._load_history()
        entries = data.get("entries", [])

        if category:
            entries = [e for e in entries if e.get("category") == category.lower()]

        entries = entries[-limit:]
        entries.reverse()

        formatted = []
        for e in entries:
            preview = e.get("content", "")[:60].replace("\n", " ")
            pinned = " [PIN]" if e.get("id") in [p.get("id") for p in data.get("pinned", [])] else ""
            formatted.append(f"[{e['id']}] [{e['category']}]{pinned} {preview}")

        return {
            "success": True,
            "count": len(entries),
            "entries": entries,
            "formatted": "\n".join(formatted) if formatted else "No history",
            "response": f"Clipboard history ({len(entries)} entries):\n" + "\n".join(formatted)
        }

    def get_entry(self, entry_id: str) -> dict:
        """Get full content of a history entry."""
        data = self._load_history()
        for e in data.get("entries", []) + data.get("pinned", []):
            if e.get("id") == entry_id:
                return {
                    "success": True,
                    "entry": e,
                    "response": f"[{e['category']}] ({e['length']} chars):\n{e['content'][:2000]}"
                }
        return {"success": False, "error": f"Entry not found: {entry_id}"}

    def restore(self, entry_id: str) -> dict:
        """Restore a history entry to the system clipboard."""
        data = self._load_history()
        for e in data.get("entries", []) + data.get("pinned", []):
            if e.get("id") == entry_id:
                success = self._set_clipboard(e["content"])
                if success:
                    return {"success": True, "response": f"Restored to clipboard: {e['content'][:60]}..."}
                return {"success": False, "error": "Could not write to clipboard"}
        return {"success": False, "error": f"Entry not found: {entry_id}"}

    def search(self, query: str) -> dict:
        """Search clipboard history."""
        if not query:
            return {"success": False, "error": "No search query"}

        data = self._load_history()
        q = query.lower()
        matching = [e for e in data.get("entries", []) + data.get("pinned", [])
                     if q in e.get("content", "").lower()]

        formatted = []
        for e in matching[-20:]:
            preview = e.get("content", "")[:60].replace("\n", " ")
            formatted.append(f"[{e['id']}] [{e['category']}] {preview}")

        return {
            "success": True,
            "count": len(matching),
            "results": matching[-20:],
            "response": f"Found {len(matching)} match(es):\n" + "\n".join(formatted)
        }

    def pin(self, entry_id: str) -> dict:
        """Pin an entry (persists even when history is pruned)."""
        data = self._load_history()

        # Find in entries
        entry = None
        for e in data.get("entries", []):
            if e.get("id") == entry_id:
                entry = e
                break

        if not entry:
            return {"success": False, "error": f"Entry not found: {entry_id}"}

        # Check if already pinned
        if any(p.get("id") == entry_id for p in data.get("pinned", [])):
            return {"success": True, "response": f"Entry {entry_id} already pinned"}

        data.setdefault("pinned", []).append(entry)
        self._save_history(data)
        return {"success": True, "response": f"Pinned entry {entry_id}"}

    def unpin(self, entry_id: str) -> dict:
        """Unpin an entry."""
        data = self._load_history()
        original = len(data.get("pinned", []))
        data["pinned"] = [p for p in data.get("pinned", []) if p.get("id") != entry_id]
        if len(data["pinned"]) == original:
            return {"success": False, "error": f"Entry not pinned: {entry_id}"}
        self._save_history(data)
        return {"success": True, "response": f"Unpinned entry {entry_id}"}

    def list_pinned(self) -> dict:
        """List pinned entries."""
        data = self._load_history()
        pinned = data.get("pinned", [])

        formatted = []
        for e in pinned:
            preview = e.get("content", "")[:60].replace("\n", " ")
            formatted.append(f"[{e['id']}] [{e['category']}] {preview}")

        return {
            "success": True,
            "count": len(pinned),
            "pinned": pinned,
            "response": f"{len(pinned)} pinned entry(ies):\n" + "\n".join(formatted) if pinned else "No pinned entries"
        }

    def clear(self) -> dict:
        """Clear all non-pinned history."""
        data = self._load_history()
        count = len(data.get("entries", []))
        data["entries"] = []
        self._save_history(data)
        return {"success": True, "cleared": count, "response": f"Cleared {count} entries (pinned preserved)"}

    def stats(self) -> dict:
        """Clipboard usage statistics."""
        data = self._load_history()
        entries = data.get("entries", [])
        pinned = data.get("pinned", [])

        categories = {}
        for e in entries:
            cat = e.get("category", "unknown")
            categories[cat] = categories.get(cat, 0) + 1

        return {
            "success": True,
            "total_entries": len(entries),
            "pinned_count": len(pinned),
            "categories": categories,
            "response": f"History: {len(entries)} entries, {len(pinned)} pinned\n"
                        f"Categories: {', '.join(f'{k}:{v}' for k, v in sorted(categories.items()))}"
        }

    # -- Dispatch -----------------------------------------------------------

    def execute(self, action: str, **kwargs) -> dict:
        action_lower = action.lower().strip()

        # Capture
        if action_lower in ("capture", "grab", "save", "snap"):
            return self.capture()

        # List history
        if action_lower.startswith("list") or action_lower.startswith("history") or action_lower.startswith("show"):
            limit = kwargs.get("limit", 20)
            category = kwargs.get("category")
            return self.list_history(limit=limit, category=category)

        # Get entry
        if action_lower.startswith("get"):
            entry_id = kwargs.get("entry_id")
            if not entry_id:
                m = re.search(r'\b([a-f0-9]{8})\b', action)
                entry_id = m.group(1) if m else None
            if entry_id:
                return self.get_entry(entry_id)
            return {"success": False, "error": "No entry ID specified"}

        # Restore
        if action_lower.startswith("restore") or action_lower.startswith("copy"):
            entry_id = kwargs.get("entry_id")
            if not entry_id:
                m = re.search(r'\b([a-f0-9]{8})\b', action)
                entry_id = m.group(1) if m else None
            if entry_id:
                return self.restore(entry_id)
            return {"success": False, "error": "No entry ID specified"}

        # Search
        if action_lower.startswith("search") or action_lower.startswith("find"):
            query = kwargs.get("query") or (action.split(None, 1)[-1] if len(action.split()) > 1 else "")
            return self.search(query)

        # Pin
        if action_lower.startswith("pin") and "unpin" not in action_lower:
            entry_id = kwargs.get("entry_id")
            if not entry_id:
                m = re.search(r'\b([a-f0-9]{8})\b', action)
                entry_id = m.group(1) if m else None
            if entry_id:
                return self.pin(entry_id)
            return {"success": False, "error": "No entry ID specified"}

        # Unpin
        if action_lower.startswith("unpin"):
            entry_id = kwargs.get("entry_id")
            if not entry_id:
                m = re.search(r'\b([a-f0-9]{8})\b', action)
                entry_id = m.group(1) if m else None
            if entry_id:
                return self.unpin(entry_id)
            return {"success": False, "error": "No entry ID specified"}

        # List pinned
        if action_lower in ("pinned", "list_pinned", "pins"):
            return self.list_pinned()

        # Clear
        if action_lower in ("clear", "clear_history"):
            return self.clear()

        # Stats
        if action_lower in ("stats", "statistics", "info"):
            return self.stats()

        # Default: capture current clipboard
        return self.capture()


# Singleton
clipboard_history_tool = ClipboardHistoryTool()
