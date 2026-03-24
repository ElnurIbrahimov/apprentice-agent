"""
ConversationManager — Cross-Surface Conversation Sync (Phase 2)

Central singleton that all surfaces (Web, CLI, Extension, Telegram) use
to read/write conversations. Handles:
- Surface tagging on messages (which surface sent each message)
- Real-time broadcast to connected surfaces
- Session binding (e.g., Telegram user → conversation ID)
- Unified conversation listing with surface activity info
"""

import json
import logging
import threading
import time
import asyncio
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("aura.conversation_manager")

# Surface identifiers
SURFACES = ("web", "cli", "extension", "telegram", "whatsapp", "discord", "api")


@dataclass
class SurfaceMessage:
    """A message with surface attribution."""
    index: int              # Index in the brain's conversation_history
    surface: str            # Which surface sent this message ("web", "telegram", etc.)
    surface_user: str       # Surface-specific user ID (e.g., telegram user_id)
    timestamp: float        # Unix timestamp
    role: str               # "user" or "assistant"
    preview: str = ""       # First 100 chars of content


@dataclass
class ConversationEvent:
    """Event broadcast to all listeners when something changes."""
    event_type: str         # "message_added", "conversation_switched", "conversation_created"
    conversation_id: str
    surface: str            # Which surface triggered this
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class ConversationManager:
    """
    Central manager for cross-surface conversation sync.

    Usage:
        manager = get_conversation_manager()
        manager.initialize(brain)

        # Any surface adds a message:
        manager.on_message_added(conv_id, role="user", content="hello",
                                  surface="telegram", surface_user="12345")

        # Telegram binds to a conversation:
        manager.bind_surface("telegram:12345", conv_id)

        # Get which conversation a surface is bound to:
        conv_id = manager.get_bound_conversation("telegram:12345")

        # Register for real-time updates:
        manager.register_listener(my_callback)
    """

    _instance: Optional["ConversationManager"] = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._brain = None
        self._data_dir: Optional[Path] = None
        self._bindings_file: Optional[Path] = None
        self._surface_bindings: Dict[str, str] = {}   # "telegram:12345" → conv_id
        self._listeners: List[Callable] = []
        self._async_listeners: List[Callable] = []
        self._rw_lock = threading.RLock()
        self._initialized = True
        logger.info("[ConvManager] Initialized")

    def initialize(self, brain) -> None:
        """Connect to the brain's conversation system. Call once on startup."""
        self._brain = brain
        self._data_dir = Path(brain._conversations_dir)
        self._bindings_file = self._data_dir / "surface_bindings.json"
        self._load_bindings()
        logger.info(f"[ConvManager] Connected to brain, {len(self._surface_bindings)} surface bindings loaded")

    @property
    def brain(self):
        if self._brain is None:
            raise RuntimeError("ConversationManager not initialized — call initialize(brain) first")
        return self._brain

    # ─── Surface Bindings ──────────────────────────────────────────────

    def bind_surface(self, surface_key: str, conversation_id: str) -> None:
        """Bind a surface to a conversation.
        Args:
            surface_key: e.g. "telegram:12345", "extension:default"
            conversation_id: e.g. "conv_1772149257_3c839b"
        """
        with self._rw_lock:
            self._surface_bindings[surface_key] = conversation_id
            self._save_bindings()
        logger.info(f"[ConvManager] Bound {surface_key} → {conversation_id}")

    def unbind_surface(self, surface_key: str) -> None:
        """Remove a surface binding."""
        with self._rw_lock:
            self._surface_bindings.pop(surface_key, None)
            self._save_bindings()

    def get_bound_conversation(self, surface_key: str) -> Optional[str]:
        """Get the conversation ID bound to a surface key."""
        return self._surface_bindings.get(surface_key)

    def get_surfaces_for_conversation(self, conversation_id: str) -> List[str]:
        """Get all surface keys bound to a conversation."""
        return [k for k, v in self._surface_bindings.items() if v == conversation_id]

    def _load_bindings(self) -> None:
        if self._bindings_file and self._bindings_file.exists():
            try:
                self._surface_bindings = json.loads(self._bindings_file.read_text(encoding="utf-8"))
            except Exception as e:
                logger.error(f"[ConvManager] Failed to load bindings: {e}")
                self._surface_bindings = {}

    def _save_bindings(self) -> None:
        if self._bindings_file:
            try:
                self._bindings_file.write_text(
                    json.dumps(self._surface_bindings, indent=2),
                    encoding="utf-8",
                )
            except Exception as e:
                logger.error(f"[ConvManager] Failed to save bindings: {e}")

    # ─── Surface Log (per-conversation message attribution) ────────────

    def _surface_log_path(self, conversation_id: str) -> Path:
        return self._data_dir / conversation_id / "surface_log.json"

    def _load_surface_log(self, conversation_id: str) -> List[dict]:
        path = self._surface_log_path(conversation_id)
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                return []
        return []

    def _save_surface_log(self, conversation_id: str, log: List[dict]) -> None:
        path = self._surface_log_path(conversation_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            path.write_text(json.dumps(log, indent=1), encoding="utf-8")
        except Exception as e:
            logger.error(f"[ConvManager] Failed to save surface log: {e}")

    # ─── Conversation Operations (wrapping brain) ──────────────────────

    def create_conversation(self, title: Optional[str] = None, surface: str = "web") -> str:
        """Create a new conversation, tagged with the creating surface."""
        conv_id = self.brain.create_conversation(title)
        # Log creation event
        self._broadcast(ConversationEvent(
            event_type="conversation_created",
            conversation_id=conv_id,
            surface=surface,
            data={"title": title or "New Chat"},
        ))
        return conv_id

    def list_conversations(self) -> List[dict]:
        """List conversations with surface activity info."""
        convs = self.brain.list_conversations()
        for conv in convs:
            conv_id = conv["id"]
            # Add which surfaces are bound to this conversation
            conv["bound_surfaces"] = self.get_surfaces_for_conversation(conv_id)
            # Add surface log summary (last surface used)
            log = self._load_surface_log(conv_id)
            if log:
                conv["last_surface"] = log[-1].get("surface", "unknown")
                # Count messages per surface
                surface_counts = {}
                for entry in log:
                    s = entry.get("surface", "unknown")
                    surface_counts[s] = surface_counts.get(s, 0) + 1
                conv["surface_activity"] = surface_counts
            else:
                conv["last_surface"] = None
                conv["surface_activity"] = {}
        return convs

    def switch_conversation(self, conversation_id: str, surface: str = "web") -> bool:
        """Switch to a conversation from a specific surface."""
        success = self.brain.switch_conversation(conversation_id)
        if success:
            self._broadcast(ConversationEvent(
                event_type="conversation_switched",
                conversation_id=conversation_id,
                surface=surface,
            ))
        return success

    def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation."""
        # Clean up surface bindings pointing to this conversation
        with self._rw_lock:
            keys_to_remove = [k for k, v in self._surface_bindings.items() if v == conversation_id]
            for k in keys_to_remove:
                del self._surface_bindings[k]
            if keys_to_remove:
                self._save_bindings()

        return self.brain.delete_conversation(conversation_id)

    def rename_conversation(self, conversation_id: str, title: str) -> bool:
        return self.brain.rename_conversation(conversation_id, title)

    def get_current_conversation_id(self) -> Optional[str]:
        return self.brain.get_current_conversation_id()

    def get_conversation_messages(self, conversation_id: str) -> List[dict]:
        """Get messages for a conversation with surface attribution."""
        messages = self.brain.get_conversation_messages(conversation_id)
        surface_log = self._load_surface_log(conversation_id)

        # Build index lookup
        surface_by_index = {entry["index"]: entry for entry in surface_log}

        # Annotate messages with surface info
        for i, msg in enumerate(messages):
            if i in surface_by_index:
                entry = surface_by_index[i]
                msg["surface"] = entry.get("surface", "unknown")
                msg["surface_user"] = entry.get("surface_user", "")
                msg["surface_timestamp"] = entry.get("timestamp", 0)
            else:
                msg["surface"] = "unknown"
                msg["surface_user"] = ""
                msg["surface_timestamp"] = 0

        return messages

    # ─── Message Tracking ──────────────────────────────────────────────

    def on_message_added(
        self,
        conversation_id: str,
        role: str,
        content: str,
        surface: str,
        surface_user: str = "",
    ) -> None:
        """
        Call this AFTER a message has been added to the brain's history.
        Records surface attribution and broadcasts to all listeners.
        """
        # Get current message count to determine index
        messages = self.brain.get_conversation_messages(conversation_id)
        index = len(messages) - 1  # Last message is the one just added

        # Record in surface log
        log = self._load_surface_log(conversation_id)
        log.append({
            "index": index,
            "surface": surface,
            "surface_user": surface_user,
            "timestamp": time.time(),
            "role": role,
            "preview": content[:100] if content else "",
        })
        self._save_surface_log(conversation_id, log)

        # Broadcast to all listeners
        self._broadcast(ConversationEvent(
            event_type="message_added",
            conversation_id=conversation_id,
            surface=surface,
            data={
                "role": role,
                "preview": content[:200] if content else "",
                "surface_user": surface_user,
                "message_index": index,
            },
        ))

    # ─── Broadcast / Listeners ─────────────────────────────────────────

    def register_listener(self, callback: Callable[[ConversationEvent], None]) -> None:
        """Register a sync listener for conversation events."""
        self._listeners.append(callback)

    def register_async_listener(self, callback: Callable) -> None:
        """Register an async listener for conversation events."""
        self._async_listeners.append(callback)

    def unregister_listener(self, callback: Callable) -> None:
        """Remove a listener."""
        self._listeners = [l for l in self._listeners if l is not callback]
        self._async_listeners = [l for l in self._async_listeners if l is not callback]

    def _broadcast(self, event: ConversationEvent) -> None:
        """Notify all registered listeners of an event."""
        # Sync listeners
        for listener in self._listeners:
            try:
                listener(event)
            except Exception as e:
                logger.error(f"[ConvManager] Listener error: {e}")

        # Async listeners — fire and forget
        for listener in self._async_listeners:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.ensure_future(listener(event))
                else:
                    loop.run_until_complete(listener(event))
            except RuntimeError:
                # No event loop — skip async listeners
                pass
            except Exception as e:
                logger.error(f"[ConvManager] Async listener error: {e}")

    # ─── Session Management (for Telegram / messaging surfaces) ────────

    def get_or_create_session(
        self, surface: str, surface_user: str, default_title: Optional[str] = None
    ) -> str:
        """
        Get the conversation bound to a surface user, or create one.
        Used by messaging platforms (Telegram, Discord, etc.) to auto-bind.
        """
        surface_key = f"{surface}:{surface_user}"
        conv_id = self.get_bound_conversation(surface_key)

        if conv_id:
            # Verify the conversation still exists
            convs = self.brain.list_conversations()
            if any(c["id"] == conv_id for c in convs):
                return conv_id
            # Conversation was deleted — unbind and create new
            self.unbind_surface(surface_key)

        # Create new conversation for this surface user
        title = default_title or f"{surface.title()} Chat"
        conv_id = self.create_conversation(title, surface=surface)
        self.bind_surface(surface_key, conv_id)
        return conv_id

    def switch_session(self, surface: str, surface_user: str, conversation_id: str) -> bool:
        """Switch a surface user to a different conversation."""
        surface_key = f"{surface}:{surface_user}"
        success = self.switch_conversation(conversation_id, surface=surface)
        if success:
            self.bind_surface(surface_key, conversation_id)
        return success

    def new_session(self, surface: str, surface_user: str, title: Optional[str] = None) -> str:
        """Create a new conversation and bind this surface user to it."""
        surface_key = f"{surface}:{surface_user}"
        conv_id = self.create_conversation(title or f"{surface.title()} Chat", surface=surface)
        self.bind_surface(surface_key, conv_id)
        return conv_id

    def list_sessions(self, surface: str, surface_user: str) -> List[dict]:
        """List conversations accessible to a surface user, marking the active one."""
        surface_key = f"{surface}:{surface_user}"
        active_id = self.get_bound_conversation(surface_key)
        convs = self.list_conversations()
        for conv in convs:
            conv["is_bound"] = conv["id"] == active_id
        return convs

    # ─── Status ────────────────────────────────────────────────────────

    def get_status(self) -> dict:
        status: dict = {
            "initialized": self._brain is not None,
            "total_bindings": len(self._surface_bindings),
            "bindings": dict(self._surface_bindings),
            "listener_count": len(self._listeners) + len(self._async_listeners),
            "current_conversation": None,
        }
        if self._brain is not None:
            try:
                status["current_conversation"] = self.get_current_conversation_id()
            except Exception:
                pass
        return status


# ─── Module-level singleton accessor ──────────────────────────────────

_manager: Optional[ConversationManager] = None

def get_conversation_manager() -> ConversationManager:
    """Get the global ConversationManager singleton."""
    global _manager
    if _manager is None:
        _manager = ConversationManager()
    return _manager
