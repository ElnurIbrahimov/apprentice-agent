"""
MultiUserManager - Session orchestration for multi-user AURA (ADV-04).

Central coordinator that manages:
- User identification from various channels (telegram, web, api, cli)
- Session lifecycle with idle timeout
- LRU cache of UserMindModel instances
- Context switching with callbacks
- Cross-user learning via KnowledgeAbstractor
"""

import hashlib
import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, TYPE_CHECKING

from .schemas import UserSession
from .identity_core import IdentityCore
from .knowledge_abstractor import AbstractInsight, KnowledgeAbstractor
from .privacy_guard import PrivacyGuard

if TYPE_CHECKING:
    from .user_mind_model import UserMindModel

logger = logging.getLogger(__name__)


# ============================================================================
# User Identification
# ============================================================================

class UserIdentifier:
    """Identifies users from various interaction channels."""

    DEFAULT_USER = "default_user"

    def __init__(self, mode: str = "auto"):
        self.mode = mode
        self._api_key_map: Dict[str, str] = {}
        self._platform_map: Dict[str, str] = {}

    def identify(self, context: Dict[str, Any]) -> str:
        """Identify user from interaction context.

        Args:
            context: Dict with platform, telegram_user_id, session_token, etc.

        Returns:
            Stable user_id string.
        """
        if self.mode == "single":
            return self.DEFAULT_USER

        platform = context.get("platform", "unknown")

        if platform == "telegram":
            tg_id = context.get("telegram_user_id")
            if tg_id:
                return f"tg_{tg_id}"

        if platform == "web":
            session = context.get("session_token")
            if session:
                return self._platform_map.get(
                    f"web:{session}", self.DEFAULT_USER
                )

        if platform == "api":
            api_key = context.get("api_key")
            if api_key and api_key in self._api_key_map:
                return self._api_key_map[api_key]

        if platform == "cli":
            return self.DEFAULT_USER

        return self.DEFAULT_USER

    def register_api_key(self, api_key: str, user_id: str) -> None:
        """Register an API key -> user_id mapping."""
        self._api_key_map[api_key] = user_id

    def register_platform_user(
        self, platform: str, platform_id: str, user_id: str,
    ) -> None:
        """Register a platform-specific user mapping."""
        self._platform_map[f"{platform}:{platform_id}"] = user_id


# ============================================================================
# MultiUserManager
# ============================================================================

class MultiUserManager:
    """Central coordinator for multi-user AURA."""

    SESSION_TIMEOUT_MINUTES = 30
    MAX_CONCURRENT_SESSIONS = 10
    MAX_CACHED_MODELS = 20

    def __init__(
        self,
        data_dir: Optional[Path] = None,
        mode: str = "auto",
        identity_core: Optional[IdentityCore] = None,
        knowledge_abstractor: Optional[KnowledgeAbstractor] = None,
    ):
        self._data_dir = data_dir or Path("data/multi_user")
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

        self.identifier = UserIdentifier(mode=mode)
        self.identity_core = identity_core
        self.knowledge_abstractor = knowledge_abstractor

        # Active sessions and cached models
        self._sessions: Dict[str, UserSession] = {}
        self._models: Dict[str, 'UserMindModel'] = {}
        self._model_access_times: Dict[str, float] = {}
        self._active_user_id: Optional[str] = None

        # Event callbacks
        self._on_user_switch: List[Callable] = []
        self._on_session_start: List[Callable] = []
        self._on_session_end: List[Callable] = []

        # Known users registry
        self._known_users: Set[str] = set()
        self._load_known_users()

        logger.info(
            f"[MultiUserManager] Initialized in '{mode}' mode, "
            f"{len(self._known_users)} known users"
        )

    # ====================================================================
    # Message Processing
    # ====================================================================

    def process_message(
        self, message: str, context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Process an incoming message with user identification.

        Identifies the user, gets/creates their session and model,
        observes the message, and returns enriched context.
        """
        with self._lock:
            user_id = self.identifier.identify(context)
            session = self._get_or_create_session(user_id, context)
            session.touch()

            model = self.get_user_model(user_id)
            model.observe_message(message, role="user")

            if user_id != self._active_user_id:
                self._switch_context(user_id)

            enriched = {
                **context,
                "user_id": user_id,
                "session_id": session.session_id,
                "user_model_context": model.get_context_for_prompt(),
                "identity_prompt": (
                    self.identity_core.get_identity_prompt(user_id)
                    if self.identity_core else ""
                ),
                "trust_level": model.relationship.trust_level.value,
                "session_duration_min": session.duration_minutes,
                "session_message_count": session.message_count,
                "cross_user_insights": (
                    [i.recommendation for i in
                     self.knowledge_abstractor.get_applicable_insights(model)]
                    if self.knowledge_abstractor else []
                ),
            }
            return enriched

    # ====================================================================
    # User Model Access
    # ====================================================================

    def get_user_model(self, user_id: str) -> 'UserMindModel':
        """Get or create a user's mental model (lazy-loaded with LRU cache)."""
        if user_id not in self._models:
            from .user_mind_model import UserMindModel

            model = UserMindModel(
                user_id=user_id,
                data_dir=self._data_dir / "models" / user_id,
            )
            self._models[user_id] = model
            self._known_users.add(user_id)
            self._save_known_users()

            # Evict oldest model if cache is full
            if len(self._models) > self.MAX_CACHED_MODELS:
                self._evict_oldest_model()

        self._model_access_times[user_id] = time.time()
        return self._models[user_id]

    def get_active_user_model(self) -> Optional['UserMindModel']:
        """Get the currently active user's model."""
        if self._active_user_id and self._active_user_id in self._models:
            return self._models[self._active_user_id]
        return None

    def get_active_user_id(self) -> Optional[str]:
        """Get the currently active user's ID."""
        return self._active_user_id

    # ====================================================================
    # Session Management
    # ====================================================================

    def _get_or_create_session(
        self, user_id: str, context: Dict[str, Any],
    ) -> UserSession:
        """Get existing session or create a new one."""
        if user_id in self._sessions:
            session = self._sessions[user_id]
            if session.idle_minutes < self.SESSION_TIMEOUT_MINUTES:
                return session
            else:
                self._end_session(user_id)

        # Create new session
        import secrets
        session_id = secrets.token_hex(12)
        session = UserSession(
            user_id=user_id, session_id=session_id,
            platform=context.get("platform", "unknown"),
            channel_id=context.get("channel_id", ""),
        )
        self._sessions[user_id] = session

        # Update model with session info
        model = self.get_user_model(user_id)
        model.relationship.total_sessions += 1
        model.current_session_start = time.time()

        # Fire callbacks
        for callback in self._on_session_start:
            try:
                callback(user_id, session_id)
            except Exception:
                pass
        logger.info(
            f"[MultiUserManager] New session for {user_id}: {session_id}"
        )
        return session

    def _end_session(self, user_id: str) -> None:
        """End a user's session and save their model."""
        session = self._sessions.pop(user_id, None)
        if session:
            session.is_active = False
            model = self._models.get(user_id)
            if model:
                model.save()
            for callback in self._on_session_end:
                try:
                    callback(user_id, session.session_id)
                except Exception:
                    pass

    def _switch_context(self, user_id: str) -> None:
        """Switch active user context and fire callbacks."""
        prev_user = self._active_user_id
        self._active_user_id = user_id
        for callback in self._on_user_switch:
            try:
                callback(prev_user, user_id)
            except Exception:
                pass
        logger.debug(
            f"[MultiUserManager] Context switch: {prev_user} -> {user_id}"
        )

    def _evict_oldest_model(self) -> None:
        """Evict the least recently used model from the cache."""
        if not self._model_access_times:
            return
        oldest_user = min(
            self._model_access_times, key=self._model_access_times.get
        )
        # Don't evict the active user
        if oldest_user == self._active_user_id:
            return
        model = self._models.pop(oldest_user, None)
        if model:
            model.save()
        self._model_access_times.pop(oldest_user, None)

    # ====================================================================
    # Event Callbacks
    # ====================================================================

    def on_user_switch(self, callback: Callable) -> None:
        """Register a callback for user context switches."""
        self._on_user_switch.append(callback)

    def on_session_start(self, callback: Callable) -> None:
        """Register a callback for new session starts."""
        self._on_session_start.append(callback)

    def on_session_end(self, callback: Callable) -> None:
        """Register a callback for session endings."""
        self._on_session_end.append(callback)

    # ====================================================================
    # Administration
    # ====================================================================

    def get_all_user_summaries(self) -> List[Dict[str, Any]]:
        """Get summaries of all known users."""
        summaries = []
        for user_id in self._known_users:
            model = self.get_user_model(user_id)
            summary = model.to_summary()
            session = self._sessions.get(user_id)
            summary["is_active"] = session is not None and session.is_active
            summaries.append(summary)
        return summaries

    def cleanup_expired_sessions(self) -> int:
        """End sessions that have been idle past the timeout. Returns count."""
        expired = [
            uid for uid, session in self._sessions.items()
            if session.idle_minutes >= self.SESSION_TIMEOUT_MINUTES
        ]
        for uid in expired:
            self._end_session(uid)
        return len(expired)

    def trigger_consolidation(self) -> List[AbstractInsight]:
        """Trigger cross-user learning via KnowledgeAbstractor."""
        if not self.knowledge_abstractor:
            return []
        return self.knowledge_abstractor.analyze_patterns(self._models)

    # ====================================================================
    # Known Users Registry
    # ====================================================================

    def _load_known_users(self) -> None:
        """Load the set of known user IDs."""
        path = self._data_dir / "known_users.json"
        if path.exists():
            try:
                self._known_users = set(
                    json.loads(path.read_text(encoding="utf-8"))
                )
            except Exception:
                self._known_users = set()

    def _save_known_users(self) -> None:
        """Persist the set of known user IDs."""
        try:
            path = self._data_dir / "known_users.json"
            path.write_text(
                json.dumps(list(self._known_users), indent=2), encoding="utf-8"
            )
        except Exception as e:
            logger.warning(f"[MultiUserManager] Failed to save known users: {e}")


# ============================================================================
# Singleton (double-checked locking, same pattern as global_workspace.py)
# ============================================================================

_manager_instance: Optional[MultiUserManager] = None
_manager_lock = threading.Lock()


def get_multi_user_manager() -> MultiUserManager:
    """Get or create the MultiUserManager singleton."""
    global _manager_instance
    if _manager_instance is None:
        with _manager_lock:
            if _manager_instance is None:
                from aura.config import Config

                identity_core = IdentityCore()
                privacy_guard = PrivacyGuard()
                knowledge_abstractor = KnowledgeAbstractor(
                    privacy_guard=privacy_guard
                )

                _manager_instance = MultiUserManager(
                    mode="auto",
                    identity_core=identity_core,
                    knowledge_abstractor=knowledge_abstractor,
                )

                # Apply config overrides
                _manager_instance.SESSION_TIMEOUT_MINUTES = (
                    Config.MULTI_USER_SESSION_TIMEOUT
                )
    return _manager_instance
