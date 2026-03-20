"""Clipboard tool — read, write, analyze, and history tracking.

Consolidated from clipboard.py, clipboard_history.py, and clipboard_memory.py.
"""

import json
import logging
import re
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict

import pyperclip

logger = logging.getLogger(__name__)

MAX_HISTORY = 200
HISTORY_FILE = Path(__file__).parent.parent.parent / "data" / "clipboard_history.json"
_file_lock = threading.Lock()


class ClipboardTool:
    """Clipboard tool with read/write, content analysis, and history tracking."""

    name = "clipboard"
    description = "Read, write, analyze clipboard content; search and browse clipboard history"

    def __init__(self):
        self._ensure_history_file()

    # ------------------------------------------------------------------
    #  History file management
    # ------------------------------------------------------------------

    def _ensure_history_file(self):
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
                if len(data.get("entries", [])) > MAX_HISTORY:
                    data["entries"] = data["entries"][-MAX_HISTORY:]
                with open(HISTORY_FILE, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
                return True
            except IOError:
                return False

    # ------------------------------------------------------------------
    #  Core clipboard operations
    # ------------------------------------------------------------------

    def read(self) -> dict:
        """Get current clipboard content."""
        try:
            content = pyperclip.paste()
            if content:
                return {
                    "success": True,
                    "content": content,
                    "length": len(content),
                    "lines": content.count('\n') + 1,
                }
            return {"success": True, "content": "", "message": "Clipboard is empty"}
        except Exception as e:
            return {"success": False, "error": f"Failed to read clipboard: {e}"}

    def write(self, text: str) -> dict:
        """Copy text to clipboard."""
        if not text:
            return {"success": False, "error": "No text provided to copy"}
        try:
            pyperclip.copy(text)
            return {
                "success": True,
                "message": f"Copied {len(text)} characters to clipboard",
                "preview": text[:100] + "..." if len(text) > 100 else text,
            }
        except Exception as e:
            return {"success": False, "error": f"Failed to write to clipboard: {e}"}

    # ------------------------------------------------------------------
    #  Content type detection / analysis
    # ------------------------------------------------------------------

    def analyze(self) -> dict:
        """Analyze clipboard content and detect its type."""
        read_result = self.read()
        if not read_result.get("success"):
            return read_result

        content = read_result.get("content", "")
        if not content:
            return {"success": True, "content_type": "empty", "message": "Clipboard is empty"}

        content_type = self._detect_content_type(content)
        result = {
            "success": True,
            "content_type": content_type,
            "length": len(content),
            "lines": content.count('\n') + 1,
            "preview": content[:200] + "..." if len(content) > 200 else content,
        }

        if content_type == "python_error":
            result["error_info"] = self._parse_python_error(content)
        elif content_type == "url":
            result["url"] = content.strip()
        elif content_type == "json":
            result["json_valid"] = self._is_valid_json(content)

        return result

    def _detect_content_type(self, content: str) -> str:
        s = content.strip()
        if self._is_python_error(content):
            return "python_error"
        if self._is_url(s):
            return "url"
        if self._is_valid_json(s):
            return "json"
        if self._is_code(content):
            return "code"
        if self._is_file_path(s):
            return "file_path"
        if re.match(r'^[\w.+-]+@[\w-]+\.[\w.-]+$', s):
            return "email"
        if re.match(r'^-?\d+\.?\d*$', s):
            return "number"
        if "\n" in s:
            return "multiline"
        return "text"

    def _is_python_error(self, content: str) -> bool:
        indicators = [
            'Traceback (most recent call last):', 'File "', 'Error:', 'Exception:',
            'raise ', 'SyntaxError:', 'TypeError:', 'ValueError:', 'KeyError:',
            'IndexError:', 'AttributeError:', 'ImportError:', 'ModuleNotFoundError:',
            'NameError:', 'ZeroDivisionError:', 'FileNotFoundError:', 'RuntimeError:',
        ]
        return any(ind in content for ind in indicators)

    def _is_url(self, content: str) -> bool:
        return bool(re.match(r'^https?://[^\s]+$', content.strip()))

    def _is_valid_json(self, content: str) -> bool:
        try:
            json.loads(content)
            return True
        except (json.JSONDecodeError, ValueError):
            return False

    def _is_code(self, content: str) -> bool:
        indicators = [
            'def ', 'class ', 'import ', 'from ', 'return ',
            'if ', 'for ', 'while ', 'try:', 'except:',
            'function ', 'const ', 'let ', 'var ',
            '#!/', '#include', 'public class', 'private ',
            '=>', '$(', '<?php', '<html', '<!DOCTYPE',
        ]
        count = sum(1 for ind in indicators if ind in content)
        has_braces = '{' in content and '}' in content
        has_semicolons = content.count(';') > 2
        return count >= 2 or (count >= 1 and (has_braces or has_semicolons))

    def _is_file_path(self, content: str) -> bool:
        if re.match(r'^[A-Za-z]:[/\\]', content):
            return True
        if content.startswith('/') and '/' in content[1:]:
            return True
        if content.startswith('~/'):
            return True
        return False

    def _parse_python_error(self, content: str) -> dict:
        info = {}
        error_match = re.search(r'(\w+Error|\w+Exception):', content)
        if error_match:
            info["error_type"] = error_match.group(1)
        lines = content.strip().split('\n')
        if lines:
            last_line = lines[-1].strip()
            if ':' in last_line:
                parts = last_line.split(':', 1)
                info["error_message"] = parts[1].strip() if len(parts) > 1 else last_line
        file_matches = re.findall(r'File "([^"]+)", line (\d+)', content)
        if file_matches:
            last_file, last_line_no = file_matches[-1]
            info["file"] = last_file
            info["line"] = int(last_line_no)
        return info

    # ------------------------------------------------------------------
    #  History operations (merged from clipboard_history.py)
    # ------------------------------------------------------------------

    def capture(self) -> dict:
        """Capture current clipboard content and add to history."""
        try:
            content = pyperclip.paste()
        except Exception:
            return {"success": False, "error": "Could not read clipboard"}
        if not content or not content.strip():
            return {"success": True, "message": "Clipboard is empty", "response": "Clipboard is empty"}

        category = self._detect_content_type(content)
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
                "response": "Clipboard unchanged (duplicate)",
            }

        data["entries"].append(entry)
        self._save_history(data)

        return {
            "success": True,
            "entry_id": entry["id"],
            "category": category,
            "length": len(content),
            "preview": content[:100],
            "response": f"Captured [{category}]: {content[:80]}...",
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
        pinned_ids = [p.get("id") for p in data.get("pinned", [])]
        for e in entries:
            preview = e.get("content", "")[:60].replace("\n", " ")
            pin_tag = " [PIN]" if e.get("id") in pinned_ids else ""
            formatted.append(f"[{e['id']}] [{e['category']}]{pin_tag} {preview}")

        return {
            "success": True,
            "count": len(entries),
            "entries": entries,
            "formatted": "\n".join(formatted) if formatted else "No history",
            "response": f"Clipboard history ({len(entries)} entries):\n" + "\n".join(formatted),
        }

    def list_recent(self, limit: int = 10) -> dict:
        """List most recent clipboard entries (compatible API for scheduled tasks)."""
        return self.list_history(limit=limit)

    def get_entry(self, entry_id: str) -> dict:
        """Get full content of a history entry."""
        data = self._load_history()
        for e in data.get("entries", []) + data.get("pinned", []):
            if e.get("id") == entry_id:
                return {
                    "success": True,
                    "entry": e,
                    "response": f"[{e['category']}] ({e['length']} chars):\n{e['content'][:2000]}",
                }
        return {"success": False, "error": f"Entry not found: {entry_id}"}

    def restore(self, entry_id: str) -> dict:
        """Restore a history entry to the system clipboard."""
        data = self._load_history()
        for e in data.get("entries", []) + data.get("pinned", []):
            if e.get("id") == entry_id:
                try:
                    pyperclip.copy(e["content"])
                    return {"success": True, "response": f"Restored to clipboard: {e['content'][:60]}..."}
                except Exception:
                    return {"success": False, "error": "Could not write to clipboard"}
        return {"success": False, "error": f"Entry not found: {entry_id}"}

    def search(self, query: str) -> dict:
        """Search clipboard history by keyword."""
        if not query:
            return {"success": False, "error": "No search query"}

        data = self._load_history()
        q = query.lower()
        matching = [
            e for e in data.get("entries", []) + data.get("pinned", [])
            if q in e.get("content", "").lower()
        ]

        formatted = []
        for e in matching[-20:]:
            preview = e.get("content", "")[:60].replace("\n", " ")
            formatted.append(f"[{e['id']}] [{e['category']}] {preview}")

        return {
            "success": True,
            "count": len(matching),
            "results": matching[-20:],
            "response": f"Found {len(matching)} match(es):\n" + "\n".join(formatted),
        }

    def pin(self, entry_id: str) -> dict:
        """Pin an entry (persists even when history is pruned)."""
        data = self._load_history()
        entry = None
        for e in data.get("entries", []):
            if e.get("id") == entry_id:
                entry = e
                break
        if not entry:
            return {"success": False, "error": f"Entry not found: {entry_id}"}
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
            "response": f"{len(pinned)} pinned entry(ies):\n" + "\n".join(formatted) if pinned else "No pinned entries",
        }

    def clear_history(self) -> dict:
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
        categories: Dict[str, int] = {}
        for e in entries:
            cat = e.get("category", "unknown")
            categories[cat] = categories.get(cat, 0) + 1
        return {
            "success": True,
            "total_entries": len(entries),
            "pinned_count": len(pinned),
            "categories": categories,
            "response": (
                f"History: {len(entries)} entries, {len(pinned)} pinned\n"
                f"Categories: {', '.join(f'{k}:{v}' for k, v in sorted(categories.items()))}"
            ),
        }

    # ------------------------------------------------------------------
    #  Dispatch
    # ------------------------------------------------------------------

    def execute(self, action: str, **kwargs) -> dict:
        """Execute a clipboard action.

        Actions: read, write/copy, analyze, capture, list/history, get,
                 restore, search/find, pin, unpin, pinned, clear, stats
        """
        a = action.lower().strip()

        # Analyze
        if "analyze" in a or "detect" in a or "type" in a:
            return self.analyze()

        # Write / copy
        if ("write" in a or "copy" in a) and "history" not in a:
            text = kwargs.get("text")
            if not text:
                text = self._extract_text_to_copy(action)
            if text:
                return self.write(text)
            return {"success": False, "error": "No text specified to copy"}

        # Capture to history
        if a in ("capture", "grab", "save", "snap"):
            return self.capture()

        # List history
        if a.startswith("list") or a.startswith("history") or a.startswith("show") or "recent" in a:
            limit = kwargs.get("limit", 20)
            category = kwargs.get("category")
            return self.list_history(limit=limit, category=category)

        # Get entry
        if a.startswith("get"):
            entry_id = kwargs.get("entry_id") or self._extract_entry_id(action)
            if entry_id:
                return self.get_entry(entry_id)
            return {"success": False, "error": "No entry ID specified"}

        # Restore
        if a.startswith("restore"):
            entry_id = kwargs.get("entry_id") or self._extract_entry_id(action)
            if entry_id:
                return self.restore(entry_id)
            return {"success": False, "error": "No entry ID specified"}

        # Search
        if a.startswith("search") or a.startswith("find") or "recall" in a or "remember" in a:
            query = kwargs.get("query") or kwargs.get("text") or (action.split(None, 1)[-1] if len(action.split()) > 1 else "")
            return self.search(query)

        # Pin
        if a.startswith("pin") and "unpin" not in a:
            entry_id = kwargs.get("entry_id") or self._extract_entry_id(action)
            if entry_id:
                return self.pin(entry_id)
            return {"success": False, "error": "No entry ID specified"}

        # Unpin
        if a.startswith("unpin"):
            entry_id = kwargs.get("entry_id") or self._extract_entry_id(action)
            if entry_id:
                return self.unpin(entry_id)
            return {"success": False, "error": "No entry ID specified"}

        # List pinned
        if a in ("pinned", "list_pinned", "pins"):
            return self.list_pinned()

        # Clear
        if a in ("clear", "clear_history"):
            return self.clear_history()

        # Stats
        if a in ("stats", "statistics", "info"):
            return self.stats()

        # Default: read clipboard
        return self.read()

    def _extract_text_to_copy(self, action: str) -> Optional[str]:
        quoted = re.findall(r'["\']([^"\']+)["\']', action)
        if quoted:
            return quoted[0]
        for pattern in [r'copy[:\s]+(.+)', r'write[:\s]+(.+)', r'clipboard[:\s]+(.+)']:
            match = re.search(pattern, action, re.IGNORECASE)
            if match:
                return match.group(1).strip().strip('"\'')
        return None

    def _extract_entry_id(self, action: str) -> Optional[str]:
        m = re.search(r'\b([a-f0-9]{8})\b', action)
        return m.group(1) if m else None


# Singleton
clipboard_tool = ClipboardTool()
