"""Session state manager for code execution — persists variables across runs."""

import copy
import json
import keyword
import threading
import time
from collections import OrderedDict


class CodeSessionManager:
    """Stores serialized variable state per session_id for multi-cell execution.

    - DataFrame persistence: stored as CSV string, re-created with pd.read_csv
    - TTL: 1 hour, max 50 sessions, LRU eviction
    - Thread-safe with threading.Lock
    """

    MAX_SESSIONS = 50
    TTL_SECONDS = 3600  # 1 hour

    def __init__(self):
        self._lock = threading.Lock()
        # OrderedDict for LRU: most-recently-used at the end
        self._sessions: OrderedDict[str, dict] = OrderedDict()
        # {session_id: {"variables": {name: {value, type_name, is_dataframe}}, "ts": float}}

    def _evict(self):
        """Remove expired sessions and enforce max count (caller holds lock)."""
        now = time.time()
        # Remove expired
        expired = [k for k, v in self._sessions.items() if now - v["ts"] > self.TTL_SECONDS]
        for k in expired:
            del self._sessions[k]
        # Enforce max count (LRU — pop oldest)
        while len(self._sessions) > self.MAX_SESSIONS:
            self._sessions.popitem(last=False)

    def get_preamble(self, session_id: str) -> str:
        """Return Python code that re-declares previous session variables."""
        with self._lock:
            entry = self._sessions.get(session_id)
            if not entry or not entry["variables"]:
                return ""
            # Touch for LRU
            self._sessions.move_to_end(session_id)
            # Deep copy under lock to avoid race with concurrent save_state
            variables = copy.deepcopy(entry["variables"])

        lines = ["# --- Session state restored ---"]
        for name, info in variables.items():
            # Security: only emit valid Python identifiers (prevents code injection via crafted names)
            if not name.isidentifier() or keyword.iskeyword(name) or name.startswith('_'):
                continue
            if info.get("is_dataframe"):
                csv_str = info["value"]
                # Escape for triple-quoted string
                escaped = csv_str.replace("\\", "\\\\").replace('"""', '\\"\\"\\"')
                lines.append('import pandas as _pd, io as _io')
                lines.append(f'{name} = _pd.read_csv(_io.StringIO("""{escaped}"""))')
            else:
                # JSON-safe value — use json.loads for safety
                json_val = json.dumps(info["value"])
                lines.append(f'{name} = __import__("json").loads({json_val!r})')
        lines.append("# --- End session state ---\n")
        return "\n".join(lines)

    def save_state(self, session_id: str, state_json: str):
        """Parse __AURA_STATE__ JSON and store variables for this session."""
        try:
            raw = json.loads(state_json)
        except (json.JSONDecodeError, TypeError):
            return

        if not isinstance(raw, dict):
            return

        variables = {}
        for name, info in raw.items():
            if not isinstance(info, dict):
                continue
            # Security: only store valid Python identifiers
            if not name.isidentifier() or keyword.iskeyword(name) or name.startswith('_'):
                continue
            variables[name] = {
                "value": info.get("value"),
                "type_name": info.get("type_name", "unknown"),
                "is_dataframe": info.get("is_dataframe", False),
            }

        with self._lock:
            self._sessions[session_id] = {"variables": variables, "ts": time.time()}
            self._sessions.move_to_end(session_id)
            self._evict()

    def reset(self, session_id: str):
        """Clear session state."""
        with self._lock:
            self._sessions.pop(session_id, None)

    def has_state(self, session_id: str) -> bool:
        with self._lock:
            return session_id in self._sessions and bool(self._sessions[session_id].get("variables"))


# Module-level singleton
_instance: CodeSessionManager | None = None
_instance_lock = threading.Lock()


def get_session_manager() -> CodeSessionManager:
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = CodeSessionManager()
    return _instance
