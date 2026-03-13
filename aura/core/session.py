"""Persist full agentic sessions (including tool_calls) to JSON on disk.

Unlike Brain's conversation store which only saves {role, content},
this preserves the complete message history including tool_calls and
tool results — enabling full session resume.
"""

import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class AgenticSession:
    """Persist full agentic sessions to JSON on disk."""

    def __init__(self, sessions_dir: str = None):
        self.sessions_dir = Path(sessions_dir or "data/agentic_sessions")
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.session_id: str = ""
        self.messages: list[dict] = []
        self.metadata: dict = {}
        self._dirty = False
        self._save_counter = 0

    def new(self, project_root: str = "", model: str = "") -> str:
        """Create new session. Returns session_id."""
        ts = int(time.time())
        short_id = uuid.uuid4().hex[:8]
        self.session_id = f"ses_{ts}_{short_id}"
        self.messages = []
        self.metadata = {
            "id": self.session_id,
            "project_root": project_root,
            "created_at": ts,
            "updated_at": ts,
            "title": "",
            "model": model,
            "stats": {"iterations": 0, "tool_calls": 0},
        }
        self._dirty = True
        self._save_counter = 0
        logger.debug(f"[Session] Created {self.session_id}")
        return self.session_id

    def save(self) -> None:
        """Atomic write to sessions_dir/session_id/session.json."""
        if not self.session_id:
            return
        if not self._dirty and self._save_counter > 0:
            return

        session_dir = self.sessions_dir / self.session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        target = session_dir / "session.json"
        tmp = session_dir / "session.tmp"

        self.metadata["updated_at"] = int(time.time())
        self.metadata["stats"]["message_count"] = len(self.messages)

        data = {**self.metadata, "messages": self.messages}

        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str, ensure_ascii=False)
            # Atomic rename (works on same filesystem)
            if target.exists():
                target.unlink()
            tmp.rename(target)
            self._dirty = False
            self._save_counter += 1
        except Exception as e:
            logger.error(f"[Session] Save failed: {e}")
            if tmp.exists():
                tmp.unlink()

    def load(self, session_id: str) -> list[dict]:
        """Load session from disk. Returns messages list."""
        session_file = self.sessions_dir / session_id / "session.json"
        if not session_file.exists():
            logger.warning(f"[Session] Not found: {session_id}")
            return []

        try:
            with open(session_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.error(f"[Session] Load failed: {e}")
            return []

        self.session_id = session_id
        self.messages = data.get("messages", [])
        self.metadata = {k: v for k, v in data.items() if k != "messages"}
        self._dirty = False
        self._save_counter = 0
        return self.messages

    def list_sessions(self, limit: int = 20) -> list[dict]:
        """Return session summaries sorted by updated desc."""
        sessions = []
        if not self.sessions_dir.exists():
            return sessions

        for d in self.sessions_dir.iterdir():
            if not d.is_dir():
                continue
            session_file = d / "session.json"
            if not session_file.exists():
                continue
            try:
                with open(session_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                sessions.append({
                    "id": data.get("id", d.name),
                    "title": data.get("title", "Untitled"),
                    "created_at": data.get("created_at", 0),
                    "updated_at": data.get("updated_at", 0),
                    "message_count": len(data.get("messages", [])),
                    "project": data.get("project_root", ""),
                    "model": data.get("model", ""),
                })
            except Exception:
                continue

        sessions.sort(key=lambda s: s["updated_at"], reverse=True)
        return sessions[:limit]

    def delete(self, session_id: str) -> bool:
        """Delete a session directory."""
        session_dir = self.sessions_dir / session_id
        if not session_dir.exists():
            return False
        try:
            import shutil
            shutil.rmtree(session_dir)
            if self.session_id == session_id:
                self.session_id = ""
                self.messages = []
                self.metadata = {}
            return True
        except Exception as e:
            logger.error(f"[Session] Delete failed: {e}")
            return False

    def append(self, message: dict) -> None:
        """Append message, normalize Pydantic objects to dicts, auto-save every 5 messages."""
        serialized = self._serialize_message(message)
        self.messages.append(serialized)
        self._dirty = True

        # Auto-title from first user message
        if not self.metadata.get("title") and message.get("role") == "user":
            self.metadata["title"] = self._auto_title(message.get("content", ""))

        # Auto-save every 5 messages
        if len(self.messages) % 5 == 0:
            self.save()

    def update_stats(self, iterations: int = 0, tool_calls: int = 0) -> None:
        """Update session stats."""
        stats = self.metadata.setdefault("stats", {})
        stats["iterations"] = iterations
        stats["tool_calls"] = tool_calls
        self._dirty = True

    def _serialize_message(self, msg: dict) -> dict:
        """Convert Pydantic ToolCall objects to plain dicts for JSON serialization."""
        result = {}
        for key, value in msg.items():
            if key == "tool_calls" and value is not None:
                serialized_calls = []
                for tc in value:
                    if isinstance(tc, dict):
                        serialized_calls.append(tc)
                    else:
                        # Pydantic ToolCall object
                        func = getattr(tc, "function", None)
                        if func:
                            args = getattr(func, "arguments", {})
                            if isinstance(args, str):
                                try:
                                    args = json.loads(args)
                                except (json.JSONDecodeError, TypeError):
                                    pass
                            serialized_calls.append({
                                "function": {
                                    "name": getattr(func, "name", ""),
                                    "arguments": args,
                                }
                            })
                result["tool_calls"] = serialized_calls
            else:
                result[key] = value
        return result

    def _auto_title(self, content: str) -> str:
        """Title from first user message (first 60 chars)."""
        if not content:
            return "Untitled"
        title = content.strip().split("\n")[0][:60]
        if len(content.strip().split("\n")[0]) > 60:
            title += "..."
        return title
