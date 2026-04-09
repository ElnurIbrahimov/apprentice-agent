"""Conversation management mixin extracted from brain.py.

Provides multi-conversation CRUD (create, list, switch, delete, rename),
legacy history migration, and conversation-to-memory persistence.

All methods reference ``self.*`` attributes initialised by
``OllamaBrain.__init__`` — this module is only useful as a mixin base
for that class.
"""

import json
import logging
import shutil
import time
import uuid
from pathlib import Path
from typing import Optional

from aura.pools import bg_submit as _bg_submit

logger = logging.getLogger(__name__)


class ConversationMixin:
    """Multi-conversation management methods for OllamaBrain."""

    # =========================================================================
    # Multi-Conversation Management
    # =========================================================================

    def _migrate_legacy_history(self) -> None:
        """Migrate existing single-conversation history into multi-conversation system."""
        index = self._load_conversations_index()
        if index:
            # Already have conversations, check if we need to set current
            if not self._current_conversation_id:
                # Find most recently updated conversation
                sorted_convs = sorted(index, key=lambda c: c.get("updated_at", 0), reverse=True)
                if sorted_convs:
                    self._current_conversation_id = sorted_convs[0]["id"]
                    # Point history file to current conversation
                    conv_dir = self._conversations_dir / self._current_conversation_id
                    if conv_dir.exists():
                        self._history_file = conv_dir / "history.json"
            return

        # No conversations index yet — migrate legacy history if it exists
        legacy_file = self._history_dir / "history.json"
        if legacy_file.exists():
            try:
                data = json.loads(legacy_file.read_text(encoding="utf-8"))
                history = data.get("history", [])
                if history:
                    # Create first conversation from legacy data
                    conv_id = self._generate_conversation_id()
                    conv_dir = self._conversations_dir / conv_id
                    conv_dir.mkdir(parents=True, exist_ok=True)

                    # Copy history to new location
                    new_history_file = conv_dir / "history.json"
                    new_history_file.write_text(
                        json.dumps(data, indent=2, ensure_ascii=False),
                        encoding="utf-8"
                    )

                    # Generate title from first user message
                    title = self._auto_title(history)

                    # Get preview from last message
                    preview = ""
                    if history:
                        last_msg = history[-1].get("content", "")
                        preview = last_msg[:100]

                    # Create index with migrated conversation
                    index_entry = {
                        "id": conv_id,
                        "title": title,
                        "created_at": int(time.time()),
                        "updated_at": int(time.time()),
                        "message_count": len(history),
                        "preview": preview,
                    }
                    self._save_conversations_index([index_entry])
                    self._current_conversation_id = conv_id
                    self._history_file = new_history_file
                    logger.info(f"[BRAIN] Migrated legacy history to conversation: {conv_id} ({title})")
                    return
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"[BRAIN] Could not migrate legacy history: {e}")

        # No legacy history either — create a default conversation
        conv_id = self._create_conversation_dir("New Chat")
        self._current_conversation_id = conv_id
        conv_dir = self._conversations_dir / conv_id
        self._history_file = conv_dir / "history.json"

    def _generate_conversation_id(self) -> str:
        """Generate a unique conversation ID."""
        return f"conv_{int(time.time())}_{uuid.uuid4().hex[:8]}"

    def _auto_title(self, messages: list) -> str:
        """Generate a title from the first user message."""
        for msg in messages:
            if msg.get("role") == "user":
                content = msg.get("content", "").strip()
                # Strip file attachment context markers
                if "[FILE_ATTACHMENT_CONTEXT]" in content:
                    # Try to find the user request after the context
                    parts = content.split("User request:")
                    if len(parts) > 1:
                        content = parts[-1].strip()
                    else:
                        content = content.split("\n")[0].strip()
                # Truncate to 50 chars
                if len(content) > 50:
                    content = content[:47] + "..."
                return content or "New Chat"
        return "New Chat"

    def _create_conversation_dir(self, title: str) -> str:
        """Create a new conversation directory and add to index."""
        conv_id = self._generate_conversation_id()
        conv_dir = self._conversations_dir / conv_id
        conv_dir.mkdir(parents=True, exist_ok=True)

        # Write empty history
        empty_data = {"history": [], "query_count": 0, "total_query_count": 0}
        (conv_dir / "history.json").write_text(
            json.dumps(empty_data, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )

        # Add to index
        index = self._load_conversations_index()
        index.append({
            "id": conv_id,
            "title": title,
            "created_at": int(time.time()),
            "updated_at": int(time.time()),
            "message_count": 0,
            "preview": "",
        })
        self._save_conversations_index(index)
        return conv_id

    def _load_conversations_index(self) -> list:
        """Load the conversations index (thread-safe)."""
        with self._conversations_index_lock:
            if self._conversations_index_cache is not None:
                return list(self._conversations_index_cache)
            try:
                if self._conversations_index_file.exists():
                    data = json.loads(self._conversations_index_file.read_text(encoding="utf-8"))
                    self._conversations_index_cache = data
                    return list(data)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"[BRAIN] Could not load conversations index: {e}")
            self._conversations_index_cache = []
            return []

    def _invalidate_conversation_cache(self) -> None:
        """Invalidate the in-memory conversations index cache.

        Must be called by any method that mutates conversations (create, delete,
        rename, switch) so the next _load_conversations_index() re-reads from disk.
        """
        with self._conversations_index_lock:
            self._conversations_index_cache = None

    def _save_conversations_index(self, index: list) -> None:
        """Save the conversations index (thread-safe)."""
        with self._conversations_index_lock:
            try:
                self._conversations_index_file.write_text(
                    json.dumps(index, indent=2, ensure_ascii=False),
                    encoding="utf-8"
                )
                self._conversations_index_cache = index
            except IOError as e:
                logger.warning(f"[BRAIN] Could not save conversations index: {e}")

    def _update_conversation_index_entry(
        self,
        snap_message_count: int | None = None,
        snap_last_content: str | None = None,
        snap_history: list | None = None,
    ) -> None:
        """Update the current conversation's index entry with latest metadata.

        When called from _save_history_unlocked, snapshot values are passed so
        the index stays consistent with the serialised data even if the live
        history list changes between serialisation and this call.
        """
        if not self._current_conversation_id:
            return
        msg_count = snap_message_count if snap_message_count is not None else len(self.conversation_history)
        last_content = snap_last_content if snap_last_content is not None else (
            self.conversation_history[-1].get("content", "") if self.conversation_history else None
        )
        history_for_title = snap_history if snap_history is not None else self.conversation_history
        # Update the in-memory cache under the lock (fast, no I/O),
        # then flush to disk outside the lock via _BG_EXECUTOR.
        # Lock ordering invariant: _history_lock -> _conversations_index_lock
        with self._conversations_index_lock:
            index = self._conversations_index_cache
            if index is None:
                # Cold start: need disk read (rare — only if cache was invalidated)
                try:
                    if self._conversations_index_file.exists():
                        index = json.loads(
                            self._conversations_index_file.read_text(encoding="utf-8")
                        )
                    else:
                        index = []
                except Exception as e:
                    logger.debug(f"[BRAIN] Conversation index parse failed: {e}")
                    index = []
            for entry in index:
                if entry["id"] == self._current_conversation_id:
                    entry["updated_at"] = int(time.time())
                    entry["message_count"] = msg_count
                    if last_content is not None:
                        entry["preview"] = last_content[:100]
                    # Update title if still "New Chat" and we have messages
                    if entry["title"] == "New Chat" and msg_count > 0:
                        entry["title"] = self._auto_title(history_for_title)
                    break
            # Store a copy so concurrent mutations don't affect the bg write
            self._conversations_index_cache = list(index)

        # Disk write outside lock, in background — uses the snapshot (immutable string)
        _index_snapshot = json.dumps(index, indent=2, ensure_ascii=False)
        def _bg_write_index(path, data):
            try:
                path.write_text(data, encoding="utf-8")
            except IOError as e:
                logger.warning(f"[BRAIN] Could not save conversations index: {e}")
        _bg_submit(_bg_write_index, self._conversations_index_file, _index_snapshot)

    def create_conversation(self, title: Optional[str] = None) -> str:
        """Create a new conversation.

        Args:
            title: Optional title, defaults to "New Chat"

        Returns:
            The new conversation's ID
        """
        # Save current conversation first
        self._save_history()
        self._update_conversation_index_entry()
        self._invalidate_conversation_cache()

        effective_title = title or "New Chat"
        conv_id = self._create_conversation_dir(effective_title)

        # Switch to the new conversation
        self._current_conversation_id = conv_id
        conv_dir = self._conversations_dir / conv_id
        self._history_file = conv_dir / "history.json"
        self.conversation_history = []
        self._query_count = 0
        logger.info(f"[BRAIN] Created new conversation: {conv_id} ({effective_title})")
        return conv_id

    def list_conversations(self) -> list:
        """List all conversations.

        Returns:
            List of conversation summaries sorted by updated_at descending
        """
        index = self._load_conversations_index()
        # Sort by updated_at descending (most recent first)
        index.sort(key=lambda c: c.get("updated_at", 0), reverse=True)
        # Return copies with is_active flag to avoid mutating cached index
        result = []
        for entry in index:
            entry_copy = dict(entry)
            entry_copy["is_active"] = entry["id"] == self._current_conversation_id
            result.append(entry_copy)
        return result

    def _validate_conversation_path(self, conversation_id: str) -> Path | None:
        """Validate that a conversation_id resolves to a safe path inside _conversations_dir.

        Returns the resolved path if safe, or None if the ID is invalid or escapes.
        """
        try:
            conv_dir = (self._conversations_dir / conversation_id).resolve()
            conv_dir.relative_to(self._conversations_dir.resolve())
        except (ValueError, OSError):
            logger.warning(f"[BRAIN] Blocked path traversal attempt: {conversation_id!r}")
            return None
        return conv_dir

    def switch_conversation(self, conversation_id: str) -> bool:
        """Switch to a different conversation.

        Args:
            conversation_id: ID of conversation to switch to

        Returns:
            True if switched successfully
        """
        if conversation_id == self._current_conversation_id:
            return True

        conv_dir = self._validate_conversation_path(conversation_id)
        if conv_dir is None or not conv_dir.exists():
            logger.warning(f"[BRAIN] Conversation not found: {conversation_id}")
            return False

        with self._history_lock:
            # Save current conversation (unlocked — we already hold the lock)
            self._save_history_unlocked()
            self._update_conversation_index_entry()
            self._invalidate_conversation_cache()

            # Load new conversation
            self._current_conversation_id = conversation_id
            self._history_file = conv_dir / "history.json"
            self._load_history()
        logger.info(f"[BRAIN] Switched to conversation: {conversation_id} ({len(self.conversation_history)} messages)")
        return True

    def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation.

        Args:
            conversation_id: ID of conversation to delete

        Returns:
            True if deleted successfully
        """
        conv_dir = self._validate_conversation_path(conversation_id)
        if conv_dir is None or not conv_dir.exists():
            return False

        # Safety: only delete directories that contain a history.json (our canary)
        if not (conv_dir / "history.json").exists():
            logger.warning(f"[BRAIN] Refusing to delete non-conversation dir: {conv_dir}")
            return False

        # Remove directory
        try:
            shutil.rmtree(conv_dir)
        except OSError as e:
            logger.error(f"[BRAIN] Failed to delete conversation dir: {e}")
            return False

        # Invalidate cache before re-reading index
        self._invalidate_conversation_cache()

        # Remove from index
        index = self._load_conversations_index()
        index = [c for c in index if c["id"] != conversation_id]
        self._save_conversations_index(index)

        # If we deleted the active conversation, switch to another or create new
        if conversation_id == self._current_conversation_id:
            if index:
                # Sort by most recently updated so we switch to the latest conversation
                index.sort(key=lambda c: c.get("updated_at", 0), reverse=True)
                target_id = index[0]["id"]
                with self._history_lock:
                    target_dir = self._conversations_dir / target_id
                    self._current_conversation_id = target_id
                    self._history_file = target_dir / "history.json"
                    self._load_history()
            else:
                conv_id = self._create_conversation_dir("New Chat")
                with self._history_lock:
                    self._current_conversation_id = conv_id
                    self._history_file = self._conversations_dir / conv_id / "history.json"
                    self.conversation_history = []
                    self._query_count = 0

        logger.info(f"[BRAIN] Deleted conversation: {conversation_id}")
        return True

    def rename_conversation(self, conversation_id: str, title: str) -> bool:
        """Rename a conversation.

        Args:
            conversation_id: ID of conversation to rename
            title: New title

        Returns:
            True if renamed successfully
        """
        if self._validate_conversation_path(conversation_id) is None:
            return False
        self._invalidate_conversation_cache()
        index = self._load_conversations_index()
        for entry in index:
            if entry["id"] == conversation_id:
                entry["title"] = title
                self._save_conversations_index(index)
                logger.info(f"[BRAIN] Renamed conversation {conversation_id}: {title}")
                return True
        return False

    def get_current_conversation_id(self) -> Optional[str]:
        """Get the current conversation ID."""
        return self._current_conversation_id

    def get_conversation_messages(self, conversation_id: str) -> list:
        """Get messages for a specific conversation without switching.

        Args:
            conversation_id: ID of conversation

        Returns:
            List of messages
        """
        if conversation_id == self._current_conversation_id:
            return list(self.conversation_history)

        conv_dir = self._validate_conversation_path(conversation_id)
        if conv_dir is None:
            return []
        history_file = conv_dir / "history.json"
        if history_file.exists():
            try:
                data = json.loads(history_file.read_text(encoding="utf-8"))
                return data.get("history", [])
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"[Brain] Failed to load conversation history from {history_file}: {e}")
        return []

    def save_conversation_to_memory(self, conversation_id: Optional[str] = None) -> dict:
        """Save a conversation's content to AURA's long-term memory (UnifiedMemory).

        Args:
            conversation_id: ID of conversation to save, or None for current

        Returns:
            Dict with success status and details
        """
        conv_id = conversation_id or self._current_conversation_id
        if not conv_id:
            return {"success": False, "error": "No active conversation"}

        messages = self.get_conversation_messages(conv_id)
        if not messages:
            return {"success": False, "error": "Conversation is empty"}

        # Get conversation title
        index = self._load_conversations_index()
        title = "Unknown"
        for entry in index:
            if entry["id"] == conv_id:
                title = entry["title"]
                break

        # Build a summary of the conversation
        user_messages = [m["content"] for m in messages if m.get("role") == "user"]
        assistant_messages = [m["content"] for m in messages if m.get("role") == "assistant"]

        # Create a condensed version for memory
        conversation_text = ""
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            # Skip file attachment context markers for cleaner memory
            if "[FILE_ATTACHMENT_CONTEXT]" in content:
                parts = content.split("User request:")
                content = parts[-1].strip() if len(parts) > 1 else content[:200]
            conversation_text += f"{role.upper()}: {content[:300]}\n"

        # Truncate if too long (keep under 2000 chars for memory)
        if len(conversation_text) > 2000:
            conversation_text = conversation_text[:1900] + "\n...(truncated)"

        memory_content = f"Conversation: {title}\n\n{conversation_text}"

        # Save to UnifiedMemory (primary — A-MEM removed 2026-03-22)
        try:
            from aura.memory.unified_memory import get_unified_memory
            umem = get_unified_memory()
            result = umem.store(
                content=memory_content,
                source="conversation_save",
                importance=0.6,
                tags=["conversation", "chat_history"],
            )
            logger.info(f"[BRAIN] Saved conversation {conv_id} to UnifiedMemory")
            return {
                "success": True,
                "note_id": result.get("store", ""),
                "message_count": len(messages),
                "title": title,
            }
        except Exception as e:
            logger.error(f"[BRAIN] Failed to save conversation: {e}")
            return {"success": False, "error": f"Memory save failed: {e}"}
