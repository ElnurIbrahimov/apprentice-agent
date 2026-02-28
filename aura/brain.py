"""Ollama API integration as the agent's reasoning engine."""

import os
import re
import json
import logging
import threading
import time
import shutil
import concurrent.futures
import atexit
from enum import Enum
from pathlib import Path
from typing import Optional, Callable, Any
import ollama

from .config import Config
from .identity import get_identity_prompt

logger = logging.getLogger(__name__)

# ALMA Emotional Intelligence System
try:
    from .emotion.integration import (
        get_emotional_tone_modifier,
        process_user_message,
        process_response_outcome,
        bridge_evoemo_detection,
        get_mood_emoji,
    )
    from .emotion.alma_engine import alma_engine, trigger_emotion
    ALMA_AVAILABLE = True
except ImportError:
    ALMA_AVAILABLE = False
    logger.warning("[BRAIN] ALMA emotional system not available")

# Default timeouts (in seconds)
LLM_TIMEOUT = 60  # 60 seconds for LLM calls
WARMUP_TIMEOUT = 10  # 10 seconds for warmup

# Neuromodulator bounds for safety (multipliers on default values)
NEURO_MIN_MULTIPLIER = 0.7   # Never reduce below 70% of default
NEURO_MAX_MULTIPLIER = 1.4   # Never increase above 140% of default


def _run_world_model_extraction(conversation_id, messages):
    """Background thread target for world model extraction (ADV-02 Phase 2)."""
    try:
        from aura.consciousness.world_model import get_world_model
        wm = get_world_model()
        wm.process_conversation(conversation_id, messages)
    except Exception as e:
        logger.debug(f"[BRAIN] World model extraction failed: {e}")

    # ADV-02 Phase 3: Quick proactive awareness analysis after extraction
    try:
        from aura.config import Config
        if getattr(Config, "PROACTIVE_AWARENESS_QUICK_AFTER_CHAT", True):
            from aura.consciousness.proactive_awareness import get_proactive_awareness_engine
            engine = get_proactive_awareness_engine()
            engine.run_quick_analysis()
    except Exception as e:
        logger.debug(f"[BRAIN] Proactive awareness quick analysis failed: {e}")


def _get_neuromodulator_levels() -> dict:
    """Get current neuromodulator levels from ALMA, with safe defaults.

    Returns dict with dopamine, serotonin, norepinephrine, oxytocin (all 0-1).
    Returns 0.5 for all if ALMA is unavailable.
    During sleep, applies NeuroDream oscillation-based neuromodulator offsets.
    """
    try:
        from aura.emotion.alma_engine import alma_engine
        state = alma_engine.get_emotional_state()
        if state and "neuromodulators" in state:
            base = state["neuromodulators"]
            # Apply sleep neuromodulator influence from NeuroDream
            try:
                from aura.tools.neurodream import get_neurodream
                nd = get_neurodream()
                if nd.current_phase.value != "awake":
                    influence = nd.get_sleep_neuromodulator_influence()
                    return {k: max(0.0, min(1.0, base[k] + influence.get(k, 0.0))) for k in base}
            except Exception:
                pass
            return base
    except Exception:
        pass
    return {"dopamine": 0.5, "serotonin": 0.5, "norepinephrine": 0.5, "oxytocin": 0.5}


def _neuro_scale(base_value: float, neuro_level: float, sensitivity: float = 0.5) -> float:
    """Scale a base value by a neuromodulator level.

    neuro_level=0.5 -> no change (returns base_value)
    neuro_level=1.0 -> increase by sensitivity * (NEURO_MAX_MULTIPLIER - 1)
    neuro_level=0.0 -> decrease by sensitivity * (1 - NEURO_MIN_MULTIPLIER)

    Safety: result is always clamped to [base * NEURO_MIN_MULTIPLIER, base * NEURO_MAX_MULTIPLIER]
    """
    # Map neuro_level 0-1 to multiplier centered at 1.0
    offset = (neuro_level - 0.5) * 2 * sensitivity  # -sensitivity to +sensitivity
    multiplier = 1.0 + offset

    # Clamp to safety bounds
    multiplier = max(NEURO_MIN_MULTIPLIER, min(NEURO_MAX_MULTIPLIER, multiplier))
    return base_value * multiplier

# Shared thread pool to prevent thread leaks (max 3 concurrent LLM calls)
_SHARED_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=3, thread_name_prefix="llm_worker")

def _cleanup_executor():
    """Cleanup shared executor on exit."""
    _SHARED_EXECUTOR.shutdown(wait=False, cancel_futures=True)

atexit.register(_cleanup_executor)


def call_with_timeout(func: Callable, timeout: int = LLM_TIMEOUT, default: Any = None) -> Any:
    """Execute a function with timeout protection using shared thread pool.

    Args:
        func: Function to execute (should be a lambda or callable with no args)
        timeout: Timeout in seconds
        default: Value to return on timeout

    Returns:
        Function result or default on timeout
    """
    try:
        future = _SHARED_EXECUTOR.submit(func)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            logger.warning(f"LLM call timed out after {timeout}s")
            future.cancel()  # Try to cancel the pending task
            return default
        except concurrent.futures.CancelledError:
            logger.warning("LLM call was cancelled")
            return default
        except (ConnectionError, OSError) as e:
            logger.error(f"LLM connection error: {e}")
            return default
        except ValueError as e:
            logger.error(f"LLM value error (bad response?): {e}")
            return default
        except Exception as e:
            # Log unexpected errors with full context for debugging
            logger.exception(f"Unexpected LLM error: {type(e).__name__}: {e}")
            return default
    except RuntimeError as e:
        # Executor might be shut down, create a one-off
        logger.warning(f"Shared executor unavailable ({e}), using fallback")
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(func)
            try:
                return future.result(timeout=timeout)
            except concurrent.futures.TimeoutError:
                logger.warning("Fallback LLM call timed out")
                return default
            except (ConnectionError, OSError, ValueError) as e:
                logger.error(f"Fallback LLM error: {e}")
                return default


class TaskType(Enum):
    """Types of tasks for model routing."""
    SIMPLE = "simple"       # Greetings, short answers, basic queries
    REASONING = "reasoning" # Planning, evaluation, complex decisions
    CODE = "code"           # Code generation, calculations
    VISION = "vision"       # Image analysis


class OllamaBrain:
    """Handles all interactions with Ollama API for reasoning and decision-making."""

    # Limit conversation history to prevent unbounded memory growth
    MAX_HISTORY_LENGTH = 20  # Keep last 20 messages (10 exchanges)

    # Auto-reset context after this many queries to prevent slowdown
    AUTO_RESET_INTERVAL = 15  # Reset every 15 queries

    # Ollama cloud configuration
    OLLAMA_CLOUD_HOST = "https://api.ollama.com"

    def __init__(self, warmup: bool = True):
        # Local Ollama client (for local models)
        self.client = ollama.Client(host=Config.OLLAMA_HOST)

        # Cloud Ollama client (for cloud models like deepseek-v3.1:671b-cloud)
        self._cloud_client = None
        api_key = os.getenv("OLLAMA_API_KEY")
        if api_key:
            self._cloud_client = ollama.Client(
                host=self.OLLAMA_CLOUD_HOST,
                headers={"Authorization": f"Bearer {api_key}"}
            )
            logger.debug(f"[BRAIN] Ollama cloud client initialized")
        else:
            logger.debug(f"[BRAIN] Warning: OLLAMA_API_KEY not set, cloud models unavailable")

        self.model = Config.MODEL_NAME
        self.conversation_history: list[dict] = []
        self._history_lock = threading.Lock()
        self._last_model_used: str = self.model  # Track for metacognition
        self._query_count: int = 0  # Track queries for auto-reset (resets every 15)
        self._total_query_count: int = 0  # Total queries (never resets)
        self._model_override: Optional[str] = None  # Manual model override (bypasses auto-selection)

        # Setup persistent history storage (legacy single-conversation path)
        self._history_dir = Config.CHROMADB_PATH.parent / "conversation"
        self._history_dir.mkdir(parents=True, exist_ok=True)
        self._history_file = self._history_dir / "history.json"

        # Multi-conversation support
        self._conversations_dir = Config.CHROMADB_PATH.parent / "conversations"
        self._conversations_dir.mkdir(parents=True, exist_ok=True)
        self._conversations_index_file = self._conversations_dir / "index.json"
        self._current_conversation_id: Optional[str] = None

        # Migrate legacy history and initialize conversations
        self._migrate_legacy_history()
        self._load_history()

        # ALMA Emotional Intelligence
        self._alma_enabled = ALMA_AVAILABLE
        self._auto_emotional_tone = True  # Automatically add emotional tone to responses
        if self._alma_enabled:
            logger.info(f"[BRAIN] ALMA emotional system enabled {get_mood_emoji()}")

        if warmup:
            self._warmup_models()

    def _get_client_for_model(self, model: str) -> tuple[ollama.Client, str]:
        """Get the appropriate client (local or cloud) based on model name.

        Cloud models end with '-cloud' or ':cloud' suffix and require the cloud client.

        Returns:
            Tuple of (client, actual_model_name) - model name may be modified for fallback
        """
        if model.endswith(("-cloud", ":cloud")):
            if self._cloud_client:
                logger.debug(f"[BRAIN] Using cloud client for model: {model}")
                return self._cloud_client, model
            else:
                # No api.ollama.com key — route through local Ollama bridge which
                # proxies cloud models (deepseek-v3.2:cloud, qwen3.5:397b-cloud, etc.)
                logger.debug(f"[BRAIN] No OLLAMA_API_KEY — routing {model} via local Ollama bridge")
                return self.client, model
        return self.client, model

    def _warmup_models(self) -> None:
        """Warm up local Ollama models with a keep-alive ping. Skipped for cloud models."""
        models_to_warm = [
            m for m in [Config.MODEL_FAST, Config.MODEL_REASON, Config.MODEL_CODE]
            if not m.endswith(("-cloud", ":cloud"))
        ]
        if not models_to_warm:
            logger.info("[BRAIN] All models are cloud-hosted — skipping warmup")
            return
        for model in models_to_warm:
            try:
                call_with_timeout(
                    lambda m=model: self.client.generate(model=m, prompt="", keep_alive="30m"),
                    timeout=WARMUP_TIMEOUT,
                    default=None
                )
                logger.info(f"[BRAIN] Warmed up local model: {model}")
            except Exception as e:
                logger.warning(f"[BRAIN] Warmup failed for {model}: {e}")

    def _load_history(self) -> None:
        """Load conversation history from disk."""
        try:
            if self._history_file.exists():
                data = json.loads(self._history_file.read_text(encoding="utf-8"))
                self.conversation_history = data.get("history", [])
                self._query_count = data.get("query_count", 0)
                self._total_query_count = data.get("total_query_count", 0)
                logger.info(f"[BRAIN] Loaded {len(self.conversation_history)} messages from history (total queries: {self._total_query_count})")
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"[BRAIN] Could not load history: {e}")
            self.conversation_history = []

    def _save_history(self) -> None:
        """Save conversation history to disk."""
        try:
            data = {
                "history": self.conversation_history,
                "query_count": self._query_count,
                "total_query_count": self._total_query_count
            }
            self._history_file.write_text(
                json.dumps(data, indent=2, ensure_ascii=False),
                encoding="utf-8"
            )
            # Update conversation index metadata
            self._update_conversation_index_entry()
        except IOError as e:
            logger.warning(f"[BRAIN] Could not save history: {e}")

    def _save_history_snapshot(self, history: list, query_count: int, total_query_count: int) -> None:
        """Save a pre-copied history list to disk (called OUTSIDE _history_lock).

        This avoids holding the lock during disk I/O, which would serialize
        all concurrent requests behind a slow write.
        """
        try:
            data = {
                "history": history,
                "query_count": query_count,
                "total_query_count": total_query_count
            }
            self._history_file.write_text(
                json.dumps(data, indent=2, ensure_ascii=False),
                encoding="utf-8"
            )
            # Update conversation index metadata
            self._update_conversation_index_entry()
        except IOError as e:
            logger.warning(f"[BRAIN] Could not save history snapshot: {e}")

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
        import uuid
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
        """Load the conversations index."""
        try:
            if self._conversations_index_file.exists():
                return json.loads(self._conversations_index_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"[BRAIN] Could not load conversations index: {e}")
        return []

    def _save_conversations_index(self, index: list) -> None:
        """Save the conversations index."""
        try:
            self._conversations_index_file.write_text(
                json.dumps(index, indent=2, ensure_ascii=False),
                encoding="utf-8"
            )
        except IOError as e:
            logger.warning(f"[BRAIN] Could not save conversations index: {e}")

    def _update_conversation_index_entry(self) -> None:
        """Update the current conversation's index entry with latest metadata."""
        if not self._current_conversation_id:
            return
        index = self._load_conversations_index()
        for entry in index:
            if entry["id"] == self._current_conversation_id:
                entry["updated_at"] = int(time.time())
                entry["message_count"] = len(self.conversation_history)
                if self.conversation_history:
                    last_msg = self.conversation_history[-1].get("content", "")
                    entry["preview"] = last_msg[:100]
                # Update title if still "New Chat" and we have messages
                if entry["title"] == "New Chat" and self.conversation_history:
                    entry["title"] = self._auto_title(self.conversation_history)
                break
        self._save_conversations_index(index)

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
        # Add is_active flag
        for entry in index:
            entry["is_active"] = entry["id"] == self._current_conversation_id
        return index

    def switch_conversation(self, conversation_id: str) -> bool:
        """Switch to a different conversation.

        Args:
            conversation_id: ID of conversation to switch to

        Returns:
            True if switched successfully
        """
        if conversation_id == self._current_conversation_id:
            return True

        conv_dir = self._conversations_dir / conversation_id
        if not conv_dir.exists():
            logger.warning(f"[BRAIN] Conversation not found: {conversation_id}")
            return False

        # Save current conversation
        self._save_history()
        self._update_conversation_index_entry()

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
        conv_dir = self._conversations_dir / conversation_id
        if not conv_dir.exists():
            return False

        # Remove directory
        try:
            shutil.rmtree(conv_dir)
        except OSError as e:
            logger.error(f"[BRAIN] Failed to delete conversation dir: {e}")
            return False

        # Remove from index
        index = self._load_conversations_index()
        index = [c for c in index if c["id"] != conversation_id]
        self._save_conversations_index(index)

        # If we deleted the active conversation, switch to another or create new
        if conversation_id == self._current_conversation_id:
            if index:
                self.switch_conversation(index[0]["id"])
            else:
                conv_id = self._create_conversation_dir("New Chat")
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

        conv_dir = self._conversations_dir / conversation_id
        history_file = conv_dir / "history.json"
        if history_file.exists():
            try:
                data = json.loads(history_file.read_text(encoding="utf-8"))
                return data.get("history", [])
            except (json.JSONDecodeError, IOError):
                pass
        return []

    def save_conversation_to_memory(self, conversation_id: Optional[str] = None) -> dict:
        """Save a conversation's content to AURA's long-term memory (A-MEM).

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

        # Try to save to A-MEM
        try:
            from aura.tools.amem import get_amem
            amem = get_amem()
            note = amem.add(
                content=memory_content,
                tags=["conversation", "chat_history", title.lower().replace(" ", "_")[:30]],
                category="conversation",
                source="conversation_save",
                importance=0.6,
            )
            logger.info(f"[BRAIN] Saved conversation {conv_id} to A-MEM (note: {note.id})")
            return {
                "success": True,
                "note_id": note.id,
                "message_count": len(messages),
                "title": title,
            }
        except Exception as e:
            logger.error(f"[BRAIN] Failed to save conversation to memory: {e}")

        # Fallback: try hybrid memory
        try:
            from aura.tools.hybrid_amem import get_hybrid_memory
            hybrid = get_hybrid_memory()
            result = hybrid.remember(
                content=memory_content,
                memory_type="episodic",
                tags=["conversation", "chat_history"],
                importance=0.6,
                source="conversation_save",
            )
            logger.info(f"[BRAIN] Saved conversation {conv_id} to hybrid memory")
            return {
                "success": True,
                "note_id": result.get("note_id"),
                "message_count": len(messages),
                "title": title,
            }
        except Exception as e2:
            logger.error(f"[BRAIN] Hybrid memory fallback also failed: {e2}")
            return {"success": False, "error": f"Memory save failed: {e}; {e2}"}

    def clear_history(self):
        """Clear conversation history to free memory."""
        self.conversation_history.clear()
        self._query_count = 0
        self._save_history()
        logger.info("[BRAIN] Conversation history cleared")

    def reset_context(self, model: Optional[str] = None):
        """Reset Ollama's context for a model to prevent slowdown.

        Args:
            model: Model to reset, or None for current model
        """
        target_model = model or self._last_model_used
        try:
            # Unload and reload the model to clear its context
            self.client.generate(
                model=target_model,
                prompt="",
                keep_alive="0"  # Unload immediately
            )
            logger.info(f"[BRAIN] Reset context for {target_model}")
        except Exception as e:
            logger.debug(f"[BRAIN] Context reset failed (ok if model not loaded): {e}")

    def full_reset(self):
        """Full reset: clear history and reset Ollama context."""
        self.clear_history()
        self.reset_context()
        logger.info("[BRAIN] Full reset completed")

    def _quick_generate(self, prompt: str) -> str:
        """Use MODEL_FAST for cheap/fast generation (summarization, planning).

        No history, no system prompt injection — just prompt -> response.

        Args:
            prompt: The prompt to send

        Returns:
            Generated response string
        """
        from .config import Config
        fast_model = Config.MODEL_FAST
        try:
            client, actual_model = self._get_client_for_model(fast_model)
            response = client.chat(
                model=actual_model,
                messages=[{"role": "user", "content": prompt}]
            )
            return response["message"]["content"]
        except Exception as e:
            logger.error(f"[BRAIN] Quick generate failed: {e}")
            return ""

    def compact_history(self, focus: str = None) -> str:
        """Compact conversation history by summarizing older messages.

        Takes the oldest 2/3 of conversation_history, asks LLM to summarize
        them in 2-4 sentences, then replaces history with:
        [summary as system message] + recent 1/3.

        Args:
            focus: Optional topic to focus the summary on

        Returns:
            The summary text, or empty string if nothing to compact
        """
        history = self.conversation_history
        if len(history) < 6:
            return ""

        # Split: oldest 2/3 to summarize, keep recent 1/3
        split_point = (len(history) * 2) // 3
        old_messages = [m for m in history[:split_point] if m.get("role") != "system"]
        recent_messages = history[split_point:]

        # Build summary prompt
        conversation_text = "\n".join(
            f"{msg['role'].upper()}: {msg['content'][:300]}"
            for msg in old_messages
        )

        focus_instruction = ""
        if focus:
            focus_instruction = f" Focus especially on topics related to: {focus}."

        summary_prompt = (
            f"Summarize this conversation in 2-4 concise sentences. "
            f"Capture the key topics, decisions, and any important context.{focus_instruction}\n\n"
            f"{conversation_text}"
        )

        summary = self._quick_generate(summary_prompt)
        if not summary:
            return ""

        # Replace history: summary as system message + recent messages
        self.conversation_history = [
            {"role": "system", "content": f"[Conversation summary] {summary}"}
        ] + recent_messages
        self._save_history()

        logger.info(f"[BRAIN] Compacted {len(old_messages)} messages into summary, kept {len(recent_messages)} recent")
        return summary

    def _check_auto_reset(self):
        """Check if auto-reset is needed and perform it.

        Instead of just resetting the counter, compacts history to preserve
        context. Falls back to simple reset if compaction fails.
        """
        self._query_count += 1
        self._total_query_count += 1  # Total count never resets
        if self._query_count >= self.AUTO_RESET_INTERVAL:
            logger.info(f"[BRAIN] Auto-compact after {self._query_count} queries (total: {self._total_query_count})")
            self._query_count = 0
            # Try to compact instead of just saving
            try:
                summary = self.compact_history()
                if summary:
                    logger.info(f"[BRAIN] Auto-compacted history: {summary[:100]}...")
                else:
                    self._save_history()
            except Exception as e:
                logger.warning(f"[BRAIN] Auto-compact failed, saving history: {e}")
                self._save_history()

    def think(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        use_history: bool = True,
        task_type: Optional[TaskType] = None,
        tone_modifier: Optional[str] = None
    ) -> str:
        """Generate a response using Ollama for reasoning tasks.

        Args:
            prompt: The prompt to send to the model
            system_prompt: Optional system prompt
            use_history: Whether to include conversation history
            task_type: Type of task for model routing (auto-detected if None)
            tone_modifier: Optional emotional tone modifier from EvoEmo/ALMA
        """
        # Check if auto-reset is needed to prevent slowdown
        self._check_auto_reset()

        # ALMA: Process user message for emotional triggers
        if self._alma_enabled and use_history:
            try:
                process_user_message(prompt)
            except Exception as e:
                logger.debug(f"[BRAIN] ALMA message processing failed: {e}")

        # Select model based on task type
        model = self._select_model(prompt, task_type)
        self._last_model_used = model

        # Prepend identity to system prompt
        identity_prompt = get_identity_prompt()
        if system_prompt:
            full_system_prompt = f"{identity_prompt}\n\n{system_prompt}"
        else:
            full_system_prompt = identity_prompt

        # === LEARNED CONTEXT INJECTION (Phase 4D: Letta-style) ===
        try:
            from aura.tools.neurodream import get_neurodream
            nd = get_neurodream()
            learned_ctx = nd.get_learned_context_prompt()
            if learned_ctx:
                full_system_prompt = f"{full_system_prompt}\n\n{learned_ctx}"
        except Exception:
            pass  # NeuroDream not available

        # === CALENDAR CONTEXT INJECTION (Phase 5D) ===
        try:
            from aura.proactive.monitors.calendar_monitor import get_calendar_monitor
            cm = get_calendar_monitor()
            cal_ctx = cm.get_context_for_prompt()
            if cal_ctx:
                full_system_prompt = f"{full_system_prompt}\n\n{cal_ctx}"
        except Exception:
            pass  # Calendar monitor not available

        # === SELF-MODEL INJECTION (Phase 6B: Metacognitive Self-Improvement) ===
        try:
            from aura.consciousness.metacognition import get_metacognitive_engine
            mc = get_metacognitive_engine()
            self_model_ctx = mc.get_self_model_prompt()
            if self_model_ctx:
                full_system_prompt = f"{full_system_prompt}\n\n{self_model_ctx}"
        except Exception:
            pass  # Metacognition not available

        # === USER MODEL INJECTION (Phase 6C / ADV-04: Theory of Mind) ===
        try:
            from aura.config import Config
            if Config.MULTI_USER_ENABLED:
                from aura.multi_user import get_multi_user_manager
                manager = get_multi_user_manager()
                user_model = manager.get_active_user_model()
                if user_model:
                    user_model.observe_message(prompt, role="user")
                    user_model_ctx = user_model.get_context_for_prompt()
                    if user_model_ctx:
                        full_system_prompt = f"{full_system_prompt}\n\n{user_model_ctx}"
            else:
                from aura.proactive.theory_of_mind import get_theory_of_mind
                tom = get_theory_of_mind()
                tom.observe_message(prompt, role="user")
                user_model_ctx = tom.get_context_for_prompt()
                if user_model_ctx:
                    full_system_prompt = f"{full_system_prompt}\n\n{user_model_ctx}"
        except Exception:
            pass  # Theory of Mind / Multi-User not available

        # === MOTIVATION INJECTION (Phase 6E: Intrinsic Motivation) ===
        try:
            from aura.consciousness.intrinsic_motivation import get_intrinsic_motivation
            im = get_intrinsic_motivation()
            im.record_interaction()  # Satisfies social drive
            motivation_ctx = im.get_context_for_prompt()
            if motivation_ctx:
                full_system_prompt = f"{full_system_prompt}\n\n{motivation_ctx}"
        except Exception:
            pass  # Intrinsic motivation not available

        # === CONSCIOUS FOCUS INJECTION (Phase 7: Global Workspace Theory) ===
        try:
            from aura.consciousness.global_workspace import get_global_workspace
            conscious_ctx = get_global_workspace().get_conscious_state().to_prompt_context()
            if conscious_ctx:
                full_system_prompt = f"{full_system_prompt}\n\n{conscious_ctx}"
        except Exception:
            pass  # Global Workspace not available

        # === WORLD STATE INJECTION (ADV-02: Persistent World Model) ===
        try:
            from aura.consciousness.world_model import get_world_model
            wm = get_world_model()
            world_ctx = wm.get_context_summary()
            if world_ctx:
                full_system_prompt = f"{full_system_prompt}\n\n{world_ctx}"
        except Exception:
            pass  # World Model not available

        # Apply emotional tone modifier - auto-generate from ALMA if not provided
        if tone_modifier:
            full_system_prompt = f"{full_system_prompt}\n\n{tone_modifier}"
        elif self._alma_enabled and self._auto_emotional_tone:
            try:
                alma_tone = get_emotional_tone_modifier()
                if alma_tone:
                    full_system_prompt = f"{full_system_prompt}\n\n{alma_tone}"
            except Exception as e:
                logger.debug(f"[BRAIN] ALMA tone generation failed: {e}")

        # ALMA response modulation: verbosity, formality, exploration hints
        if self._alma_enabled:
            try:
                from aura.emotion.alma_engine import get_response_modulation
                mod = get_response_modulation()
                mod_parts = []
                if mod.get("verbosity", 0.5) < 0.35:
                    mod_parts.append("Keep response concise.")
                elif mod.get("verbosity", 0.5) > 0.65:
                    mod_parts.append("Feel free to elaborate.")
                if mod.get("formality", 0.4) > 0.6:
                    mod_parts.append("Use a formal tone.")
                elif mod.get("formality", 0.4) < 0.25:
                    mod_parts.append("Use a casual, conversational tone.")
                if mod.get("enthusiasm", 0.5) > 0.7:
                    mod_parts.append("Try creative or novel approaches.")
                elif mod.get("enthusiasm", 0.5) < 0.3:
                    mod_parts.append("Prefer proven, reliable approaches.")
                if mod_parts:
                    full_system_prompt = f"{full_system_prompt}\n\n[Style guidance: {' '.join(mod_parts)}]"
            except Exception:
                pass

        messages = []
        if full_system_prompt:
            messages.append({"role": "system", "content": full_system_prompt})
        if use_history:
            messages.extend(self.conversation_history[-self.MAX_HISTORY_LENGTH:])
        messages.append({"role": "user", "content": prompt})

        # Neuromodulator: Serotonin modulates patience (timeout)
        # High serotonin = more patience = longer timeout; low = impatient = shorter
        neuro = _get_neuromodulator_levels()
        adjusted_timeout = int(_neuro_scale(LLM_TIMEOUT, neuro["serotonin"], sensitivity=0.3))
        logger.debug(f"[BRAIN] Calling {model} with timeout={adjusted_timeout}s (serotonin={neuro['serotonin']:.2f})")

        # Get appropriate client (local or cloud) - may return fallback model
        client, actual_model = self._get_client_for_model(model)
        if actual_model != model:
            logger.info(f"[BRAIN] Model fallback: {model} -> {actual_model}")

        # Update last model used to reflect ACTUAL model, not requested
        self._last_model_used = actual_model

        # === PHASE 1: Record real thinking — LLM inference starting ===
        try:
            from api.routes.thinking import get_manager as get_thinking_manager
            tm = get_thinking_manager()
            tm.record_real_thought("formulating", f"reasoning with {actual_model}...", intensity=0.7, source="brain")
        except Exception:
            pass

        # Neuromodulator: Dopamine modulates temperature (creativity/exploration)
        # High dopamine = slightly higher temp = more creative; low = more conservative
        base_temp = 0.7
        adjusted_temp = round(_neuro_scale(base_temp, neuro["dopamine"], sensitivity=0.25), 2)

        # Neuromodulator: Serotonin modulates num_predict (response thoroughness)
        # High serotonin = patience = longer responses allowed; low = terse
        base_num_predict = 1024
        adjusted_num_predict = int(_neuro_scale(base_num_predict, neuro["serotonin"], sensitivity=0.3))

        # Neuromodulator: Norepinephrine modulates top_p (focus vs exploration)
        # High norepinephrine = alert/focused = lower top_p (more deterministic)
        # Low norepinephrine = relaxed = higher top_p (more varied responses)
        base_top_p = 0.9
        adjusted_top_p = round(base_top_p - (neuro["norepinephrine"] - 0.5) * 0.15, 2)
        adjusted_top_p = max(0.7, min(0.95, adjusted_top_p))

        # Neuromodulator: Acetylcholine modulates repeat_penalty (attention precision)
        # High acetylcholine = focused attention = higher repeat penalty (less repetitive)
        base_repeat_penalty = 1.1
        ach = neuro.get("acetylcholine", 0.5)
        adjusted_repeat_penalty = round(_neuro_scale(base_repeat_penalty, ach, sensitivity=0.15), 2)

        llm_options = {
            "temperature": adjusted_temp,
            "num_predict": adjusted_num_predict,
            "top_p": adjusted_top_p,
            "repeat_penalty": adjusted_repeat_penalty,
        }

        logger.debug(
            f"[BRAIN] Neuro-modulated LLM: temp={adjusted_temp} "
            f"(DA={neuro['dopamine']:.2f}), "
            f"num_predict={adjusted_num_predict} "
            f"(5HT={neuro['serotonin']:.2f}), "
            f"top_p={adjusted_top_p} "
            f"(NE={neuro['norepinephrine']:.2f})"
        )

        # Record neuromodulator influence on thinking panel
        try:
            from api.routes.thinking import record_thought
            neuro_effects = []
            if abs(neuro["dopamine"] - 0.5) > 0.1:
                neuro_effects.append(f"DA={'high' if neuro['dopamine']>0.5 else 'low'}")
            if abs(neuro["serotonin"] - 0.5) > 0.1:
                neuro_effects.append(f"5HT={'high' if neuro['serotonin']>0.5 else 'low'}")
            if abs(neuro["norepinephrine"] - 0.5) > 0.1:
                neuro_effects.append(f"NE={'high' if neuro['norepinephrine']>0.5 else 'low'}")
            if neuro_effects:
                record_thought(
                    "observing",
                    f"neuromodulators influencing response: {', '.join(neuro_effects)}",
                    0.4, "emotion"
                )
        except Exception:
            pass

        # Call with timeout protection (serotonin-modulated)
        response = call_with_timeout(
            lambda: client.chat(model=actual_model, messages=messages, options=llm_options),
            timeout=adjusted_timeout,
            default=None
        )

        if response is None:
            logger.warning(f"[BRAIN] LLM call timed out or failed, returning fallback")
            return "I'm having trouble processing that right now. Please try again."

        assistant_message = response["message"]["content"]

        if use_history:
            with self._history_lock:
                self.conversation_history.append({"role": "user", "content": prompt})
                self.conversation_history.append({"role": "assistant", "content": assistant_message})
                recent = list(self.conversation_history[-6:])
                _history_snapshot = list(self.conversation_history)
                _qc = self._query_count
                _tqc = self._total_query_count
            # Disk I/O outside the lock to avoid serializing concurrent requests
            self._save_history_snapshot(_history_snapshot, _qc, _tqc)
        else:
            recent = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": assistant_message},
            ]

        # === SELF-IMPROVEMENT: Record interaction outcome ===
        try:
            from aura.consciousness.self_improvement import get_self_improvement_engine
            _SHARED_EXECUTOR.submit(
                get_self_improvement_engine().record_chat_outcome,
                prompt, assistant_message, actual_model
            )
        except Exception:
            pass

        # === WORLD MODEL EXTRACTION (ADV-02 Phase 2) ===
        try:
            from aura.consciousness.world_model import get_world_model
            wm = get_world_model()
            if wm.enabled:
                conv_id = self.get_current_conversation_id()
                try:
                    _SHARED_EXECUTOR.submit(_run_world_model_extraction, conv_id, list(recent))
                except Exception:
                    threading.Thread(
                        target=_run_world_model_extraction,
                        args=(conv_id, list(recent)),
                        daemon=True,
                        name=f"wm-extract-{conv_id[:8]}",
                    ).start()
        except Exception:
            pass

        return assistant_message

    def think_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        use_history: bool = True,
        task_type: Optional[TaskType] = None,
        tone_modifier: Optional[str] = None
    ):
        """Generate a streaming response using Ollama for reasoning tasks.

        This is the streaming version of think() that yields chunks as they arrive.

        Args:
            prompt: The prompt to send to the model
            system_prompt: Optional system prompt
            use_history: Whether to include conversation history
            task_type: Type of task for model routing (auto-detected if None)
            tone_modifier: Optional emotional tone modifier from EvoEmo/ALMA

        Yields:
            str: Response chunks as they are generated
        """
        # Check if auto-reset is needed to prevent slowdown
        self._check_auto_reset()

        # ALMA: Process user message for emotional triggers
        if self._alma_enabled and use_history:
            try:
                process_user_message(prompt)
            except Exception as e:
                logger.debug(f"[BRAIN] ALMA message processing failed: {e}")

        # Select model based on task type
        model = self._select_model(prompt, task_type)
        self._last_model_used = model

        # Prepend identity to system prompt
        identity_prompt = get_identity_prompt()
        if system_prompt:
            full_system_prompt = f"{identity_prompt}\n\n{system_prompt}"
        else:
            full_system_prompt = identity_prompt

        # === LEARNED CONTEXT INJECTION (Phase 4D: Letta-style) ===
        try:
            from aura.tools.neurodream import get_neurodream
            nd = get_neurodream()
            learned_ctx = nd.get_learned_context_prompt()
            if learned_ctx:
                full_system_prompt = f"{full_system_prompt}\n\n{learned_ctx}"
        except Exception:
            pass  # NeuroDream not available

        # === CALENDAR CONTEXT INJECTION (Phase 5D) ===
        try:
            from aura.proactive.monitors.calendar_monitor import get_calendar_monitor
            cm = get_calendar_monitor()
            cal_ctx = cm.get_context_for_prompt()
            if cal_ctx:
                full_system_prompt = f"{full_system_prompt}\n\n{cal_ctx}"
        except Exception:
            pass  # Calendar monitor not available

        # === SELF-MODEL INJECTION (Phase 6B: Metacognitive Self-Improvement) ===
        try:
            from aura.consciousness.metacognition import get_metacognitive_engine
            mc = get_metacognitive_engine()
            self_model_ctx = mc.get_self_model_prompt()
            if self_model_ctx:
                full_system_prompt = f"{full_system_prompt}\n\n{self_model_ctx}"
        except Exception:
            pass  # Metacognition not available

        # === USER MODEL INJECTION (Phase 6C / ADV-04: Theory of Mind) ===
        try:
            from aura.config import Config
            if Config.MULTI_USER_ENABLED:
                from aura.multi_user import get_multi_user_manager
                manager = get_multi_user_manager()
                user_model = manager.get_active_user_model()
                if user_model:
                    user_model.observe_message(prompt, role="user")
                    user_model_ctx = user_model.get_context_for_prompt()
                    if user_model_ctx:
                        full_system_prompt = f"{full_system_prompt}\n\n{user_model_ctx}"
            else:
                from aura.proactive.theory_of_mind import get_theory_of_mind
                tom = get_theory_of_mind()
                tom.observe_message(prompt, role="user")
                user_model_ctx = tom.get_context_for_prompt()
                if user_model_ctx:
                    full_system_prompt = f"{full_system_prompt}\n\n{user_model_ctx}"
        except Exception:
            pass  # Theory of Mind / Multi-User not available

        # === MOTIVATION INJECTION (Phase 6E: Intrinsic Motivation) ===
        try:
            from aura.consciousness.intrinsic_motivation import get_intrinsic_motivation
            im = get_intrinsic_motivation()
            im.record_interaction()  # Satisfies social drive
            motivation_ctx = im.get_context_for_prompt()
            if motivation_ctx:
                full_system_prompt = f"{full_system_prompt}\n\n{motivation_ctx}"
        except Exception:
            pass  # Intrinsic motivation not available

        # === CONSCIOUS FOCUS INJECTION (Phase 7: Global Workspace Theory) ===
        try:
            from aura.consciousness.global_workspace import get_global_workspace
            conscious_ctx = get_global_workspace().get_conscious_state().to_prompt_context()
            if conscious_ctx:
                full_system_prompt = f"{full_system_prompt}\n\n{conscious_ctx}"
        except Exception:
            pass  # Global Workspace not available

        # === WORLD STATE INJECTION (ADV-02: Persistent World Model) ===
        try:
            from aura.consciousness.world_model import get_world_model
            wm = get_world_model()
            world_ctx = wm.get_context_summary()
            if world_ctx:
                full_system_prompt = f"{full_system_prompt}\n\n{world_ctx}"
        except Exception:
            pass  # World Model not available

        # Apply emotional tone modifier - auto-generate from ALMA if not provided
        if tone_modifier:
            full_system_prompt = f"{full_system_prompt}\n\n{tone_modifier}"
        elif self._alma_enabled and self._auto_emotional_tone:
            try:
                alma_tone = get_emotional_tone_modifier()
                if alma_tone:
                    full_system_prompt = f"{full_system_prompt}\n\n{alma_tone}"
            except Exception as e:
                logger.debug(f"[BRAIN] ALMA tone generation failed: {e}")

        # ALMA response modulation: verbosity, formality, exploration hints
        if self._alma_enabled:
            try:
                from aura.emotion.alma_engine import get_response_modulation
                mod = get_response_modulation()
                mod_parts = []
                if mod.get("verbosity", 0.5) < 0.35:
                    mod_parts.append("Keep response concise.")
                elif mod.get("verbosity", 0.5) > 0.65:
                    mod_parts.append("Feel free to elaborate.")
                if mod.get("formality", 0.4) > 0.6:
                    mod_parts.append("Use a formal tone.")
                elif mod.get("formality", 0.4) < 0.25:
                    mod_parts.append("Use a casual, conversational tone.")
                if mod.get("enthusiasm", 0.5) > 0.7:
                    mod_parts.append("Try creative or novel approaches.")
                elif mod.get("enthusiasm", 0.5) < 0.3:
                    mod_parts.append("Prefer proven, reliable approaches.")
                if mod_parts:
                    full_system_prompt = f"{full_system_prompt}\n\n[Style guidance: {' '.join(mod_parts)}]"
            except Exception:
                pass

        messages = []
        if full_system_prompt:
            messages.append({"role": "system", "content": full_system_prompt})
        if use_history:
            messages.extend(self.conversation_history[-self.MAX_HISTORY_LENGTH:])
        messages.append({"role": "user", "content": prompt})

        logger.debug(f"[BRAIN] Streaming call to {model}")

        # Get appropriate client (local or cloud) - may return fallback model
        client, actual_model = self._get_client_for_model(model)
        if actual_model != model:
            logger.info(f"[BRAIN] Model fallback: {model} -> {actual_model}")

        # Update last model used to reflect ACTUAL model, not requested
        self._last_model_used = actual_model

        # === PHASE 1: Record real thinking — streaming inference starting ===
        try:
            from api.routes.thinking import get_manager as get_thinking_manager
            tm = get_thinking_manager()
            tm.record_real_thought("formulating", f"streaming response with {actual_model}...", intensity=0.7, source="brain")
        except Exception:
            pass

        full_response = ""
        try:
            # Use Ollama's streaming API
            stream = client.chat(model=actual_model, messages=messages, stream=True)

            for chunk in stream:
                if chunk and "message" in chunk and "content" in chunk["message"]:
                    content = chunk["message"]["content"]
                    full_response += content
                    yield content

        except Exception as e:
            logger.error(f"[BRAIN] Streaming error: {e}")
            fallback = "I'm having trouble processing that right now. Please try again."
            yield fallback
            full_response = fallback

        # Update history after streaming completes
        if use_history and full_response:
            with self._history_lock:
                self.conversation_history.append({"role": "user", "content": prompt})
                self.conversation_history.append({"role": "assistant", "content": full_response})
                # Enforce history limit
                if len(self.conversation_history) > self.MAX_HISTORY_LENGTH:
                    self.conversation_history = self.conversation_history[-self.MAX_HISTORY_LENGTH:]
                recent = list(self.conversation_history[-6:])
                _history_snapshot = list(self.conversation_history)
                _qc = self._query_count
                _tqc = self._total_query_count
            # Disk I/O outside the lock to avoid serializing concurrent requests
            self._save_history_snapshot(_history_snapshot, _qc, _tqc)
        else:
            recent = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": full_response},
            ]

        # === SELF-IMPROVEMENT: Record interaction outcome ===
        try:
            from aura.consciousness.self_improvement import get_self_improvement_engine
            _SHARED_EXECUTOR.submit(
                get_self_improvement_engine().record_chat_outcome,
                prompt, full_response, actual_model
            )
        except Exception:
            pass

        # === WORLD MODEL EXTRACTION (ADV-02 Phase 2) ===
        try:
            from aura.consciousness.world_model import get_world_model
            wm = get_world_model()
            if wm.enabled:
                conv_id = self.get_current_conversation_id()
                try:
                    _SHARED_EXECUTOR.submit(_run_world_model_extraction, conv_id, list(recent))
                except Exception:
                    threading.Thread(
                        target=_run_world_model_extraction,
                        args=(conv_id, list(recent)),
                        daemon=True,
                        name=f"wm-extract-{conv_id[:8]}",
                    ).start()
        except Exception:
            pass

    def _is_complex_query(self, prompt: str) -> bool:
        """Detect if a query is complex and needs cloud model.

        Complex queries include:
        - Research/analysis requests
        - Multi-step reasoning
        - Comparisons requiring deep knowledge
        - Long-form content generation
        """
        prompt_lower = prompt.lower()
        words = prompt.split()

        # Long prompts are likely complex
        if len(words) > 50:
            return True

        # Complex task indicators — must be explicit task requests, not conversational references
        # Bad: 'research', 'review', 'tell me about' — match casual questions like
        #      "what do you think about this research?" → wrongly triggers 397B model
        complex_patterns = [
            'write an essay', 'write a report', 'write a detailed',
            'comprehensive analysis', 'in-depth analysis', 'thorough analysis',
            'deep dive into', 'deep search', 'investigate in detail',
            'pros and cons of', 'advantages and disadvantages',
            'step by step guide', 'detailed explanation of',
            'compare and contrast',
        ]

        if any(pattern in prompt_lower for pattern in complex_patterns):
            return True

        return False

    def set_model_override(self, model: Optional[str]) -> None:
        """Set a manual model override that bypasses auto-selection.

        Args:
            model: Model name to force, or None to return to auto-selection
        """
        self._model_override = model
        if model:
            logger.info(f"[BRAIN] Model override set: {model}")
        else:
            logger.info("[BRAIN] Model override cleared, returning to auto-selection")

    def _get_domain_confidence(self, prompt: str) -> tuple:
        """Get domain and confidence score from metacognition for a prompt.

        Returns:
            (domain_name: str, confidence: float) tuple.
            Falls back to (None, 0.5) if metacognition unavailable.
        """
        try:
            from aura.consciousness.metacognition import get_metacognitive_engine
            engine = get_metacognitive_engine()
            domain = engine.get_domain_for_query(prompt)
            if domain is None:
                return (None, 0.5)
            caps = engine.assess_capabilities()
            cap = caps.get(domain.value)
            if cap and cap.confidence > 0.1:
                return (domain.value, cap.score)
            return (domain.value, 0.5)
        except Exception:
            return (None, 0.5)

    def _should_escalate_to_system2(self, prompt: str, task_type: Optional[TaskType] = None) -> tuple:
        """Decide whether to use System 2 (deliberative) over System 1 (fast).

        Implements Kahneman-inspired dual-process routing:
        - Direct System 2 triggers for known complex patterns
        - Confidence-based escalation via metacognition
        - Neuromodulator tie-breaking for mid-range confidence

        Returns:
            (use_system2: bool, domain: str, confidence: float, reason: str)
        """
        # Direct System 2 triggers
        if self._is_complex_query(prompt):
            return (True, None, 0.0, "complex_query_heuristic")
        if task_type == TaskType.REASONING:
            return (True, None, 0.0, "explicit_reasoning_task")

        # Confidence-based escalation
        domain, confidence = self._get_domain_confidence(prompt)

        if confidence < Config.S2_CONFIDENCE_THRESHOLD:
            return (True, domain, confidence, "low_confidence")
        if confidence > Config.S1_CONFIDENCE_THRESHOLD:
            return (False, domain, confidence, "high_confidence")

        # Mid-range confidence: use neuromodulator state as tie-breaker
        neuro = _get_neuromodulator_levels()
        if neuro["norepinephrine"] > 0.6:
            return (True, domain, confidence, "high_norepinephrine")
        if neuro["dopamine"] > 0.7:
            return (False, domain, confidence, "high_dopamine")

        return (False, domain, confidence, "default_fast")

    def _select_model(self, prompt: str, task_type: Optional[TaskType] = None) -> str:
        """Select the appropriate model based on task type and complexity.

        SYSTEM 1/SYSTEM 2 HYBRID ROUTING (Kahneman dual-process):
        - System 1 (fast): Simple queries, high confidence → MODEL_FAST
        - System 2 (deliberative): Complex queries, low confidence → MODEL_REASON
        - Specialized: Vision/Code tasks use dedicated model chains
        - Cloud: Complex queries that need cloud-scale models

        Args:
            prompt: The prompt to analyze
            task_type: Explicit task type, or None for auto-detection

        Returns:
            Model name to use
        """
        # Check for manual override first
        if self._model_override:
            logger.info(f"[BRAIN] Using manual model override: {self._model_override}")
            return self._model_override

        use_cloud = self._is_complex_query(prompt)
        prompt_lower = prompt.lower()

        # Specialized task routing (Vision/Code have dedicated models)
        if task_type == TaskType.VISION or any(kw in prompt_lower for kw in ['image', 'picture', 'screenshot', 'photo', 'analyze image']):
            return Config.get_model("vision")

        if task_type == TaskType.CODE:
            return Config.get_model("code")

        # Code detection from prompt keywords
        code_patterns = [
            'calculate', 'compute', 'factorial', 'fibonacci', 'prime',
            'print(', 'import ', 'def ', 'python',
            'code', 'script', 'function', 'algorithm',
            'debug', 'fix this', 'fix the', 'write a script', 'implement',
            'refactor', 'class ', 'method', 'variable', 'loop',
            'error', 'exception', 'traceback', 'bug', 'syntax'
        ]
        if any(pattern in prompt_lower for pattern in code_patterns):
            return Config.get_model("code")

        # Identity questions always use reasoning model
        identity_patterns = [
            'what is your name', 'who are you', 'your name', 'are you called',
            'what should i call you', 'introduce yourself', 'tell me about yourself',
            'what are you', 'are you an ai', 'are you a bot', 'what model are you'
        ]
        if any(pattern in prompt_lower for pattern in identity_patterns):
            return Config.get_model("reason")

        # System 1/System 2 decision for all other queries
        use_s2, domain, confidence, reason = self._should_escalate_to_system2(prompt, task_type)

        # Apply explicit thinking-mode override + cognitive load
        try:
            from aura.thinking_mode import get_thinking_mode_manager
            tmm = get_thinking_mode_manager()
            use_s2, reason = tmm.get_effective_decision(use_s2)
            # Track this query in cognitive load window
            was_complex = use_cloud or (task_type == TaskType.CODE)
            tmm.cognitive_load.record_query(confidence, was_complex, use_s2)
        except Exception:
            pass  # Graceful fallback if thinking_mode not available

        if use_s2:
            model = Config.get_model("reason")
            logger.info(f"[BRAIN] System 2 (deliberative): domain={domain}, confidence={confidence:.2f}, reason={reason}, model={model}")
            return model
        else:
            model = Config.MODEL_FAST
            logger.info(f"[BRAIN] System 1 (fast): domain={domain}, confidence={confidence:.2f}, reason={reason}, model={model}")
            return model

    def get_last_model_used(self) -> str:
        """Get the model used in the last think() call."""
        return self._last_model_used

    def observe(self, context: dict) -> str:
        """Process observations about the current state."""
        prompt = f"""Context:
{self._format_context(context)}

List 3-5 key observations. Be brief."""

        return self.think(prompt, system_prompt=self._observer_prompt())

    def plan(self, goal: str, observations: str, available_tools: list[str]) -> str:
        """Create a plan to achieve the goal based on observations."""
        # === PHASE 1: Record real thinking — planning ===
        try:
            from api.routes.thinking import get_manager as get_thinking_manager
            tm = get_thinking_manager()
            tm.record_real_thought("analyzing", f"planning approach for: {goal[:60]}", intensity=0.7, source="brain")
        except Exception:
            pass

        tool_descriptions = self._get_tool_descriptions(available_tools)

        # Detect task type from goal
        goal_lower = goal.lower()

        # Vision/image analysis keywords
        vision_keywords = [
            'analyze', 'describe', 'what do you see', 'look at this image',
            'what\'s in this image', 'what is in this image', 'examine image',
            'read this image', 'interpret', 'identify', 'recognize'
        ]
        is_vision_task = any(kw in goal_lower for kw in vision_keywords)

        # Screenshot keywords
        screenshot_keywords = [
            'screenshot', 'screen shot', 'capture screen', 'screen capture',
            'take a picture of screen', 'grab screen', 'what\'s on my screen',
            'what is on my screen', 'capture my screen', 'print screen'
        ]
        is_screenshot_task = any(kw in goal_lower for kw in screenshot_keywords)

        # Combined screenshot + vision task (e.g., "take a screenshot and describe it")
        is_screenshot_and_vision = is_screenshot_task and (
            is_vision_task or
            'and describe' in goal_lower or
            'and analyze' in goal_lower or
            'and tell me' in goal_lower or
            'then describe' in goal_lower or
            'then analyze' in goal_lower
        )

        # Search/web keywords
        search_keywords = [
            'search', 'find', 'look up', 'lookup', 'google', 'web', 'internet',
            'online', 'news', 'latest', 'current', 'today', 'price', 'weather',
            'stock', 'bitcoin', 'crypto', 'what is the', 'who is', 'where is',
            'when did', 'how much', 'trending', 'recent', 'update'
        ]
        is_search_task = any(kw in goal_lower for kw in search_keywords) and not is_screenshot_task and not is_vision_task

        # Code/calculation keywords
        code_keywords = [
            'python', 'calculate', 'compute', 'factorial', 'code', 'program',
            'script', 'generate', 'write code', 'run', 'execute', 'math',
            'sum', 'average', 'sort', 'algorithm', 'function', 'check',
            'prime', 'number', 'verify', 'test', 'fibonacci', 'loop',
            'print', 'multiply', 'divide', 'add', 'subtract', 'power',
            'square', 'root', 'modulo', 'remainder', 'even', 'odd'
        ]
        is_code_task = any(kw in goal_lower for kw in code_keywords) and not is_search_task and not is_screenshot_task and not is_vision_task

        # PDF keywords
        pdf_keywords = [
            'pdf', '.pdf', 'document', 'read pdf', 'extract pdf', 'summarize pdf',
            'pdf file', 'open pdf', 'pdf content', 'pdf text', 'pdf pages',
            'search pdf', 'find in pdf', 'pdf info', 'pdf metadata'
        ]
        is_pdf_task = any(kw in goal_lower for kw in pdf_keywords)

        # Clipboard keywords - check for explicit clipboard mentions
        clipboard_keywords = [
            'clipboard', 'paste', 'copied', 'what\'s in my clipboard',
            'what is in my clipboard', 'read clipboard', 'write clipboard',
            'copy to clipboard', 'analyze clipboard', 'clipboard content'
        ]
        # Only mark as PURE clipboard task if clipboard is mentioned WITHOUT other tool keywords
        # This allows multi-tool tasks like "read clipboard then search web"
        has_clipboard = 'clipboard' in goal_lower
        has_other_tools = is_search_task or is_code_task or is_screenshot_task or is_vision_task or is_pdf_task
        is_clipboard_task = has_clipboard and not has_other_tools

        # System control keywords
        system_control_keywords = [
            'system info', 'system information', 'cpu usage', 'cpu', 'ram usage',
            'ram', 'memory usage', 'gpu', 'gpu usage', 'disk usage', 'disk space',
            'get volume', 'set volume', 'volume level', 'get brightness',
            'set brightness', 'brightness level', 'open app', 'launch app',
            'open notepad', 'open calculator', 'open browser', 'open chrome',
            'open firefox', 'open vscode', 'open terminal', 'lock screen',
            'show me system', 'what is my cpu', 'what is my ram', 'how much ram',
            'how much memory', 'computer info', 'pc info', 'machine info'
        ]
        is_system_control_task = any(kw in goal_lower for kw in system_control_keywords) and not is_code_task

        # Notification keywords
        notification_keywords = [
            'remind', 'reminder', 'notify', 'notification', 'alert me',
            'schedule', 'every day', 'every morning', 'every evening',
            'daily at', 'weekly', 'weekdays', 'in 5 minutes', 'in 10 minutes',
            'in 30 minutes', 'in an hour', 'in 2 hours', 'set reminder',
            'set alarm', 'remind me', 'alert when', 'notify when',
            'list reminders', 'show reminders', 'cancel reminder', 'clear reminders'
        ]
        is_notification_task = any(kw in goal_lower for kw in notification_keywords)

        # Tool builder keywords
        tool_builder_keywords = [
            'create tool', 'make tool', 'build tool', 'new tool', 'i need a tool',
            'custom tool', 'generate tool', 'tool builder', 'list custom tools',
            'test tool', 'enable tool', 'disable tool', 'delete tool', 'remove tool',
            'rollback tool', 'show custom tools'
        ]
        is_tool_builder_task = any(kw in goal_lower for kw in tool_builder_keywords)

        # Store for use in decide_action and _generate_default_code
        self._current_goal_is_code = is_code_task
        self._current_goal_is_search = is_search_task
        self._current_goal_is_screenshot = is_screenshot_task
        self._current_goal_is_vision = is_vision_task
        self._current_goal_is_screenshot_and_vision = is_screenshot_and_vision
        self._current_goal_is_pdf = is_pdf_task
        self._current_goal_is_clipboard = is_clipboard_task
        self._current_goal_is_system_control = is_system_control_task
        self._current_goal_is_notification = is_notification_task
        self._current_goal_is_tool_builder = is_tool_builder_task
        self._current_goal = goal

        if is_clipboard_task:
            prompt = f"""Goal: {goal}

This is a CLIPBOARD task. Use clipboard tool to read, write, or analyze clipboard content.

Available tools:
{tool_descriptions}

Create a 1-step plan:
1. Use clipboard to read/write/analyze the clipboard"""
        elif is_screenshot_and_vision:
            prompt = f"""Goal: {goal}

This is a SCREENSHOT + VISION task. First capture the screen, then analyze it.

Available tools:
{tool_descriptions}

Create a 2-step plan:
1. Use screenshot to capture the screen
2. Use vision to analyze/describe the captured screenshot"""
        elif is_vision_task and not is_screenshot_task:
            prompt = f"""Goal: {goal}

This is a VISION/IMAGE ANALYSIS task. Use vision tool to analyze an image.

Available tools:
{tool_descriptions}

Create a 1-step plan:
1. Use vision to analyze the image"""
        elif is_screenshot_task:
            prompt = f"""Goal: {goal}

This is a SCREENSHOT task. Use screenshot tool to capture the screen.

Available tools:
{tool_descriptions}

Create a 1-step plan:
1. Use screenshot to capture the screen"""
        elif is_search_task:
            prompt = f"""Goal: {goal}

This is a WEB SEARCH task. Use web_search to find information online.
DO NOT use code_executor for web searches - it cannot access the internet.

Available tools:
{tool_descriptions}

Create a 2-3 step plan:
1. Use web_search with a clear search query
2. Summarize the results if needed"""
        elif is_code_task:
            prompt = f"""Goal: {goal}

This is a CODE/CALCULATION task. Use code_executor to run Python code directly.
DO NOT search the web. Just write and run the Python code.

Available tools:
{tool_descriptions}

Create a 1-2 step plan:
1. Use code_executor with the actual Python code to solve this
2. (Optional) Summarize if needed"""
        elif is_pdf_task:
            prompt = f"""Goal: {goal}

This is a PDF task. Use pdf_reader tool to read, extract, or search PDF content.

Available tools:
{tool_descriptions}

Create a 1-2 step plan:
1. Use pdf_reader to read/extract/search the PDF
2. (Optional) Summarize the content if needed"""
        elif is_system_control_task:
            prompt = f"""Goal: {goal}

This is a SYSTEM CONTROL task. Use system_control tool to get system info, control volume/brightness, or launch apps.

Available tools:
{tool_descriptions}

Create a 1-step plan:
1. Use system_control to execute the system command"""
        elif is_notification_task:
            prompt = f"""Goal: {goal}

This is a NOTIFICATION task. Use notifications tool to set reminders, schedule notifications, or create conditional alerts.

Available tools:
{tool_descriptions}

Create a 1-step plan:
1. Use notifications to add/list/remove reminder or scheduled task"""
        elif is_tool_builder_task:
            prompt = f"""Goal: {goal}

This is a TOOL BUILDER task. Use tool_builder to create, test, enable, disable, or list custom tools.

Available tools:
{tool_descriptions}

Create a 1-step plan:
1. Use tool_builder to list/test/enable/disable/create custom tools"""
        else:
            prompt = f"""Goal: {goal}

Observations: {observations[:500]}

Available tools:
{tool_descriptions}

Create a short 3-5 step plan. Be specific about which tool to use for each step."""

        return self.think(prompt, system_prompt=self._planner_prompt())

    def decide_action(self, plan: str, available_tools: list[str]) -> dict:
        """Decide the next action to take based on the plan."""
        # === PHASE 1: Record real thinking — deciding action ===
        try:
            from api.routes.thinking import get_manager as get_thinking_manager
            tm = get_thinking_manager()
            tools_short = ", ".join(available_tools[:4])
            tm.record_real_thought("connecting", f"selecting tool from: {tools_short}", intensity=0.6, source="brain")
        except Exception:
            pass

        tool_descriptions = self._get_tool_descriptions(available_tools)

        # Check task type
        is_screenshot = getattr(self, '_current_goal_is_screenshot', False)
        is_search = getattr(self, '_current_goal_is_search', False)
        is_vision = getattr(self, '_current_goal_is_vision', False)
        is_screenshot_and_vision = getattr(self, '_current_goal_is_screenshot_and_vision', False)
        is_clipboard = getattr(self, '_current_goal_is_clipboard', False)
        screenshot_path = getattr(self, '_last_screenshot_path', None)

        # Check clipboard FIRST (before vision which also has "analyze")
        if is_clipboard:
            prompt = f"""Plan: {plan[:500]}

This is a CLIPBOARD task. Use clipboard tool to read, write, or analyze clipboard content.

Pick ONE action. Reply ONLY in this format:

TOOL: clipboard
ACTION: <read/write/analyze> [text to copy]
REASONING: <why>

Examples:

TOOL: clipboard
ACTION: read
REASONING: get current clipboard content

TOOL: clipboard
ACTION: analyze
REASONING: detect clipboard content type

TOOL: clipboard
ACTION: write "hello world"
REASONING: copy text to clipboard"""
        # Check combined screenshot+vision (before screenshot alone)
        elif is_screenshot_and_vision:
            if screenshot_path:
                # Screenshot already taken, now use vision
                prompt = f"""Plan: {plan[:500]}

This is a SCREENSHOT + VISION task. Screenshot was already taken at: {screenshot_path}

NOW use vision tool to analyze the captured screenshot.

Reply ONLY in this format:

TOOL: vision
ACTION: analyze {screenshot_path}
REASONING: analyze the captured screenshot"""
            else:
                # Need to take screenshot first
                prompt = f"""Plan: {plan[:500]}

This is a SCREENSHOT + VISION task. Take screenshot first, then analyze it.

Screenshot has NOT been taken yet. Use screenshot tool first.

Reply ONLY in this format:

TOOL: screenshot
ACTION: capture
REASONING: need to capture screen first"""
        elif is_screenshot:
            prompt = f"""Plan: {plan[:500]}

This is a SCREENSHOT task. You MUST use screenshot tool.

Pick ONE action. Reply ONLY in this format:

TOOL: screenshot
ACTION: capture
REASONING: take screenshot of the screen

Example:

TOOL: screenshot
ACTION: capture full screen
REASONING: capture current screen"""
        elif is_search:
            prompt = f"""Plan: {plan[:500]}

Available tools:
{tool_descriptions}

This is a WEB SEARCH task. You MUST use web_search tool.
DO NOT use code_executor - it cannot access the internet!

Pick ONE action. Reply ONLY in this format:

TOOL: web_search
ACTION: <your search query>
REASONING: <why>

Examples:

TOOL: web_search
ACTION: Bitcoin price today USD
REASONING: find current Bitcoin price

TOOL: web_search
ACTION: latest AI news 2024
REASONING: search for recent AI news

TOOL: web_search
ACTION: weather New York today
REASONING: get current weather"""
        elif is_vision:
            # Vision-only task (not combined with screenshot)
            prompt = f"""Plan: {plan[:500]}

This is a VISION/IMAGE ANALYSIS task. Use vision tool to analyze an image.

Pick ONE action. Reply ONLY in this format:

TOOL: vision
ACTION: <analyze/describe/read> <image_path>
REASONING: <why>

Examples:

TOOL: vision
ACTION: analyze screenshots/screenshot_20260115_152610.png
REASONING: analyze what is in the image

TOOL: vision
ACTION: describe screen screenshots/latest.png
REASONING: describe what is on screen

TOOL: vision
ACTION: read text document.png
REASONING: extract text from image"""
        elif getattr(self, '_current_goal_is_pdf', False):
            # PDF task
            prompt = f"""Plan: {plan[:500]}

This is a PDF task. Use pdf_reader tool to read, extract, or search PDF content.

Pick ONE action. Reply ONLY in this format:

TOOL: pdf_reader
ACTION: <read/extract/search/info> <pdf_path> [pages/query]
REASONING: <why>

Examples:

TOOL: pdf_reader
ACTION: read C:/Documents/report.pdf
REASONING: read entire PDF content

TOOL: pdf_reader
ACTION: read C:/Documents/report.pdf pages 1-5
REASONING: read first 5 pages

TOOL: pdf_reader
ACTION: search C:/Documents/report.pdf "revenue"
REASONING: find pages mentioning revenue

TOOL: pdf_reader
ACTION: info C:/Documents/report.pdf
REASONING: get PDF metadata and page count"""
        elif getattr(self, '_current_goal_is_system_control', False):
            # System control task
            prompt = f"""Plan: {plan[:500]}

This is a SYSTEM CONTROL task. Use system_control tool.

Pick ONE action. Reply ONLY in this format:

TOOL: system_control
ACTION: <get_system_info/get_volume/set_volume/get_brightness/set_brightness/open_app/lock_screen> [args]
REASONING: <why>

Examples:

TOOL: system_control
ACTION: get_system_info
REASONING: get CPU, RAM, GPU, and disk usage

TOOL: system_control
ACTION: set_volume 50
REASONING: set volume to 50%

TOOL: system_control
ACTION: open_app notepad
REASONING: launch notepad application

TOOL: system_control
ACTION: get_brightness
REASONING: get current screen brightness"""
        elif getattr(self, '_current_goal_is_notification', False):
            # Notification task
            prompt = f"""Plan: {plan[:500]}

This is a NOTIFICATION task. Use notifications tool.

Pick ONE action. Reply ONLY in this format:

TOOL: notifications
ACTION: <add_reminder/add_scheduled/add_condition/list/remove/clear> [args]
REASONING: <why>

Examples:

TOOL: notifications
ACTION: add_reminder "take a break" in 30 minutes
REASONING: set a reminder for 30 minutes

TOOL: notifications
ACTION: add_scheduled "standup meeting" 9:00 AM daily
REASONING: schedule daily notification at 9 AM

TOOL: notifications
ACTION: add_condition "high CPU alert" cpu 80
REASONING: alert when CPU exceeds 80%

TOOL: notifications
ACTION: list
REASONING: show all scheduled tasks"""
        elif getattr(self, '_current_goal_is_tool_builder', False):
            # Tool builder task
            prompt = f"""Plan: {plan[:500]}

This is a TOOL BUILDER task. Use tool_builder tool.

Pick ONE action. Reply ONLY in this format:

TOOL: tool_builder
ACTION: <list/test/enable/disable/rollback> [tool_name]
REASONING: <why>

Examples:

TOOL: tool_builder
ACTION: list
REASONING: list all custom tools

TOOL: tool_builder
ACTION: test my_tool
REASONING: run tests for my_tool

TOOL: tool_builder
ACTION: enable my_tool
REASONING: activate the tool after testing

TOOL: tool_builder
ACTION: disable my_tool
REASONING: temporarily disable the tool"""
        else:
            prompt = f"""Plan: {plan[:500]}

Available tools:
{tool_descriptions}

Pick ONE action. Reply ONLY in this format:

TOOL: <tool_name>
ACTION: <actual code, path, or query>
REASONING: <why>

RULES:
- For calculations/math/Python -> use code_executor with ACTUAL Python code
- For local files -> use filesystem
- For internet/online info -> use web_search (NOT code_executor!)
- code_executor CANNOT access the internet - use web_search instead!

Examples:

TOOL: code_executor
ACTION: import math; print(math.factorial(50))
REASONING: calculate factorial

TOOL: web_search
ACTION: Bitcoin price today
REASONING: search internet for price

TOOL: filesystem
ACTION: list C:/Users/project
REASONING: see directory"""

        response = self.think(prompt, system_prompt=self._actor_prompt())
        return self._parse_action_response(response)

    def evaluate(self, action: str, result: str, goal: str) -> dict:
        """Evaluate the result of an action."""
        # === PHASE 1: Record real thinking — evaluating result ===
        try:
            from api.routes.thinking import get_manager as get_thinking_manager
            tm = get_thinking_manager()
            tm.record_real_thought("observing", f"evaluating result of: {action[:50]}", intensity=0.5, source="brain")
        except Exception:
            pass

        # Truncate result to avoid overwhelming the model
        result_truncated = result[:1000] if len(result) > 1000 else result

        # Check for multi-step tasks (screenshot + vision)
        goal_lower = goal.lower()
        is_screenshot_and_vision = getattr(self, '_current_goal_is_screenshot_and_vision', False)

        # If this is a combined task and we just did screenshot, continue to vision
        if is_screenshot_and_vision and 'screenshot' in action.lower() and 'success' in result_truncated.lower():
            extra_instruction = """
IMPORTANT: This is a 2-step task (screenshot + describe/analyze).
If only screenshot was done, say NEXT: continue (still need to analyze the image).
Only say NEXT: complete if BOTH screenshot AND vision/description are done."""
        else:
            extra_instruction = ""

        prompt = f"""Goal: {goal}
Action: {action}
Result: {result_truncated}
{extra_instruction}
Reply ONLY in this format:

SUCCESS: yes OR no
CONFIDENCE: 0-100 (how confident are you the goal is fully achieved)
PROGRESS: one sentence about progress
NEXT: continue OR complete OR retry

If the goal is achieved, say NEXT: complete"""

        response = self.think(prompt, system_prompt=self._evaluator_prompt())
        return self._parse_evaluation_response(response)

    def summarize_for_memory(self, episode: dict) -> str:
        """Create a memory-worthy summary of an episode."""
        prompt = f"""Summarize in 2-3 sentences:
Goal: {episode.get('goal', 'N/A')}
Actions: {episode.get('actions', [])}
Outcome: {episode.get('outcome', 'N/A')}"""

        return self.think(prompt, system_prompt=self._memory_prompt(), use_history=False)

    def unload_model(self, model: str = None) -> bool:
        """Unload a model from Ollama to free VRAM.

        Args:
            model: Model name to unload. If None, unloads the last used model.

        Returns:
            True if successful, False otherwise.
        """
        model_to_unload = model or self._last_model_used
        try:
            # Send empty generate with keep_alive=0 to unload (with timeout)
            result = call_with_timeout(
                lambda: self.client.generate(
                    model=model_to_unload,
                    prompt="",
                    keep_alive="0s"
                ),
                timeout=10,
                default=None
            )
            return result is not None
        except Exception:
            return False

    def unload_all_models(self) -> dict:
        """Unload all commonly used models to free VRAM.

        Returns:
            Dict with unload status for each model.
        """
        models = [Config.MODEL_FAST, Config.MODEL_REASON, Config.MODEL_CODE, Config.MODEL_VISION]
        results = {}
        for model in models:
            results[model] = self.unload_model(model)
        return results

    def _format_context(self, context: dict) -> str:
        """Format context dictionary for prompts."""
        return "\n".join(f"- {k}: {v}" for k, v in context.items())

    def summarize(self, content: str, goal: str) -> str:
        """Summarize content in relation to a goal."""
        prompt = f"""Goal: {goal}

Content to summarize:
{content[:2000]}

Write a clear, concise summary (3-5 sentences) of the key points relevant to the goal."""

        return self.think(prompt, system_prompt="You summarize information clearly and concisely.", use_history=False)

    def _get_tool_descriptions(self, available_tools: list[str]) -> str:
        """Get clear descriptions for available tools."""
        descriptions = {
            "filesystem": "filesystem - list or read LOCAL files on this computer. ACTION: 'list <path>' or 'read <path>'",
            "web_search": "web_search - search the INTERNET for information. ACTION: the search query",
            "code_executor": "code_executor - run Python code and get the output. Use for calculations, data processing. ACTION: the Python code",
            "screenshot": "screenshot - capture a screenshot of the screen. ACTION: 'capture' or 'capture region x y width height'",
            "vision": "vision - analyze images using AI vision model. ACTION: 'analyze <image_path>' or 'describe screen <path>' or 'read text <path>'",
            "pdf_reader": "pdf_reader - read, extract text, or search PDF files. ACTION: 'read <path>' or 'extract <path> pages 1-5' or 'search <path> query' or 'info <path>'",
            "clipboard": "clipboard - read, write, or analyze clipboard content. ACTION: 'read' or 'write <text>' or 'analyze'",
            "system_control": "system_control - get system info (CPU, RAM, GPU, disk), control volume/brightness, open apps, lock screen. ACTION: 'get_system_info' or 'get_volume' or 'set_volume <level>' or 'open_app <name>'",
            "notifications": "notifications - set reminders, schedule notifications, create conditional alerts. ACTION: 'add_reminder <msg> in <time>' or 'add_scheduled <msg> <time> <repeat>' or 'list' or 'remove <id>'",
            "tool_builder": "tool_builder - create, test, enable, disable, or list custom tools. ACTION: 'list' or 'test <name>' or 'enable <name>' or 'disable <name>' or 'rollback <name>'",
            "summarize": "summarize - summarize gathered information. ACTION: 'results'",
            "calendar": "calendar - manage events, appointments, schedules. ACTION: 'add <title> on <date> at <time>' or 'today' or 'upcoming' or 'list <date>' or 'remove <id>' or 'search <query>'",
            "shell_executor": "shell_executor - execute shell/terminal commands with persistent sessions. ACTION: the command to run (e.g. 'ls -la', 'git status', 'python script.py')",
            "screen_reader": "screen_reader - read text from screen via OCR, detect active window, monitor for changes. ACTION: 'read' or 'read_region x y w h' or 'active_window' or 'watch <keyword>'",
            "email": "email - read and send emails. ACTION: 'fetch' or 'fetch unread' or 'read <id>' or 'send to:<addr> subject:<subj> body:<text>' or 'search <query>' or 'setup'",
            "spaced_repetition": "spaced_repetition - flashcard learning with spaced repetition. ACTION: 'review' or 'add front:<q> back:<a>' or 'answer <id> <quality 0-5>' or 'due' or 'stats' or 'auto_generate <text>'",
            "task_manager": "task_manager - manage tasks, projects, kanban boards. ACTION: 'add <title>' or 'list' or 'board' or 'update <id> status:<status>' or 'projects' or 'overdue' or 'search <query>'",
            "api_tester": "api_tester - test HTTP APIs and REST endpoints. ACTION: 'GET <url>' or 'POST <url> body:<json>' or 'PUT <url> body:<json>' or 'DELETE <url>' or 'history' or 'inspect <id>'",
            "database": "database - query SQLite databases, inspect schemas, import/export CSV. ACTION: SQL query like 'SELECT * FROM table' or 'schema' or 'tables' or 'import <csv> <table>' or 'export <table>'",
            "audio_transcriber": "audio_transcriber - transcribe audio/video files to text using Whisper. ACTION: 'transcribe <path>' or 'translate <path>' or 'detect <path>' or 'list' or 'status'",
            "clipboard_history": "clipboard_history - clipboard history with search, pinning, categories. ACTION: 'capture' or 'list' or 'search <query>' or 'pin <id>' or 'restore <id>' or 'stats'",
            "research": "research - save, search, and organize research notes and findings. ACTION: 'save title:<title> content:<text> category:<cat>' or 'search <query>' or 'list' or 'list <category>' or 'read <filename>' or 'stats' or 'skills' or 'tag <tagname>'"
        }
        return "\n".join(descriptions.get(t, t) for t in available_tools)

    def _parse_action_response(self, response: str) -> dict:
        """Parse the action decision response with better extraction for local models."""
        result = {"tool": None, "action": None, "reasoning": None, "raw": response}

        # Check if this is a code task - if so, force code_executor
        is_code_task = getattr(self, '_current_goal_is_code', False)

        # Try to find TOOL, ACTION, REASONING in the response
        for line in response.split("\n"):
            line = line.strip()
            if line.upper().startswith("TOOL:"):
                tool = line[5:].strip().lower()
                # Clean up common variations
                tool = tool.replace("**", "").replace("`", "").strip()
                if "code" in tool or "execute" in tool or "python" in tool or "run" in tool:
                    tool = "code_executor"
                elif "summar" in tool:
                    tool = "summarize"
                elif "web" in tool or "search" in tool:
                    tool = "web_search"
                elif "file" in tool or "fs" in tool:
                    tool = "filesystem"
                elif "screenshot" in tool or "screen" in tool or "capture" in tool:
                    tool = "screenshot"
                elif "vision" in tool or "llava" in tool or "image" in tool or "analyze" in tool:
                    tool = "vision"
                elif "pdf" in tool or "document" in tool:
                    tool = "pdf_reader"
                elif "clipboard" in tool or "copy" in tool or "paste" in tool:
                    tool = "clipboard"
                elif "system" in tool or "control" in tool:
                    tool = "system_control"
                elif "notif" in tool or "remind" in tool or "schedule" in tool or "alert" in tool:
                    tool = "notifications"
                elif "tool_builder" in tool or "builder" in tool or "create tool" in tool or "custom tool" in tool:
                    tool = "tool_builder"
                elif "calendar" in tool or "event" in tool or "schedule" in tool or "agenda" in tool:
                    tool = "calendar"
                elif "shell" in tool or "terminal" in tool or "command" in tool or "bash" in tool:
                    tool = "shell_executor"
                elif "screen_reader" in tool or "ocr" in tool or "monitor" in tool or "active window" in tool:
                    tool = "screen_reader"
                elif "email" in tool or "mail" in tool or "inbox" in tool or "send email" in tool:
                    tool = "email"
                elif "flashcard" in tool or "spaced" in tool or "repetition" in tool or "review card" in tool:
                    tool = "spaced_repetition"
                elif "task_manager" in tool or "task" in tool or "kanban" in tool or "todo" in tool:
                    tool = "task_manager"
                elif "api_tester" in tool or "api test" in tool or "http" in tool or "rest" in tool:
                    tool = "api_tester"
                elif "database" in tool or "sqlite" in tool or "sql" in tool or "db" in tool:
                    tool = "database"
                elif "audio" in tool or "transcrib" in tool or "whisper" in tool or "speech" in tool:
                    tool = "audio_transcriber"
                elif "clipboard_history" in tool or "clip hist" in tool:
                    tool = "clipboard_history"
                elif "research" in tool or "save research" in tool or "notes" in tool:
                    tool = "research"
                result["tool"] = tool
            elif line.upper().startswith("ACTION:"):
                action = line[7:].strip()
                # Clean up the action - remove common prefixes local models add
                action = self._clean_action(action)
                result["action"] = action
            elif line.upper().startswith("REASONING:"):
                result["reasoning"] = line[10:].strip()

        # Fallback: try to extract from less structured responses
        if not result["tool"]:
            response_lower = response.lower()
            # Check for code executor indicators
            if "code_executor" in response_lower or "python" in response_lower or "calculate" in response_lower or "factorial" in response_lower or "print(" in response:
                result["tool"] = "code_executor"
            # Check for filesystem indicators
            elif "filesystem" in response_lower or "list " in response_lower or "read " in response_lower or "directory" in response_lower:
                result["tool"] = "filesystem"
            elif "summarize" in response_lower or "summary" in response_lower:
                result["tool"] = "summarize"
            elif "web_search" in response_lower or "internet" in response_lower or "online" in response_lower:
                result["tool"] = "web_search"
            elif "screenshot" in response_lower or "capture screen" in response_lower:
                result["tool"] = "screenshot"
            elif "vision" in response_lower or "analyze image" in response_lower or "describe image" in response_lower:
                result["tool"] = "vision"
            elif "pdf_reader" in response_lower or "read pdf" in response_lower or "extract pdf" in response_lower:
                result["tool"] = "pdf_reader"
            elif "clipboard" in response_lower or "paste" in response_lower or "copy to" in response_lower:
                result["tool"] = "clipboard"
            elif "system_control" in response_lower or "system info" in response_lower or "cpu" in response_lower or "ram" in response_lower or "volume" in response_lower or "brightness" in response_lower:
                result["tool"] = "system_control"
            elif "notification" in response_lower or "reminder" in response_lower or "remind" in response_lower or "schedule" in response_lower or "alert" in response_lower:
                result["tool"] = "notifications"
            elif "tool_builder" in response_lower or "create tool" in response_lower or "custom tool" in response_lower or "list tools" in response_lower:
                result["tool"] = "tool_builder"
            elif "calendar" in response_lower or "event" in response_lower or "agenda" in response_lower or "appointment" in response_lower:
                result["tool"] = "calendar"
            elif "shell" in response_lower or "terminal" in response_lower or "command line" in response_lower or "run command" in response_lower:
                result["tool"] = "shell_executor"
            elif "screen reader" in response_lower or "ocr" in response_lower or "read screen" in response_lower or "active window" in response_lower:
                result["tool"] = "screen_reader"
            elif "email" in response_lower or "inbox" in response_lower or "send mail" in response_lower or "check mail" in response_lower:
                result["tool"] = "email"
            elif "flashcard" in response_lower or "spaced repetition" in response_lower or "review card" in response_lower:
                result["tool"] = "spaced_repetition"
            elif "task_manager" in response_lower or "kanban" in response_lower or "todo list" in response_lower or "project board" in response_lower:
                result["tool"] = "task_manager"
            elif "api_tester" in response_lower or "test api" in response_lower or "http request" in response_lower or "rest api" in response_lower:
                result["tool"] = "api_tester"
            elif "database" in response_lower or "sql query" in response_lower or "sqlite" in response_lower or "run query" in response_lower:
                result["tool"] = "database"
            elif "audio_transcriber" in response_lower or "transcribe" in response_lower or "speech to text" in response_lower or "whisper" in response_lower:
                result["tool"] = "audio_transcriber"
            elif "clipboard_history" in response_lower or "clipboard history" in response_lower or "clip history" in response_lower:
                result["tool"] = "clipboard_history"
            elif "save research" in response_lower or "research note" in response_lower or "save finding" in response_lower:
                result["tool"] = "research"

        if not result["action"] and result["tool"] == "web_search":
            # Try to extract a search query from the response
            result["action"] = self._extract_search_query(response)

        # FORCE code_executor for code tasks - override any wrong tool selection
        if is_code_task and result["tool"] != "code_executor":
            result["tool"] = "code_executor"
            # Try to extract code from the action or response
            if result["action"]:
                # Clean up action to be valid Python code
                action = result["action"]
                # If it looks like a description, try to extract actual code
                if not any(ind in action for ind in ['print(', '=', 'import ', 'def ', 'for ', 'if ']):
                    # Extract any Python-like code from the full response
                    code = self._extract_code_from_response(response)
                    if code:
                        result["action"] = code
                    else:
                        # Generate default code based on the original goal
                        result["action"] = self._generate_default_code()

        # Also force code_executor if action looks like code
        if result["tool"] != "code_executor" and result["action"]:
            if any(ind in result["action"] for ind in ['print(', 'import ', 'def ', 'for i in']):
                result["tool"] = "code_executor"

        # FORCE clipboard for clipboard tasks - override tool AND action
        is_clipboard_task = getattr(self, '_current_goal_is_clipboard', False)
        if is_clipboard_task:
            result["tool"] = "clipboard"
            goal = getattr(self, '_current_goal', '').lower()
            current_action = (result.get("action") or "").lower()

            # Determine correct action based on goal keywords
            if 'analyze' in goal or 'type' in goal or 'detect' in goal:
                result["action"] = "analyze"
            elif 'copy' in goal or 'write' in goal:
                # Extract text to copy if present in action
                if '"' in current_action:
                    # Keep action with quoted text
                    pass
                elif 'write' in current_action and len(current_action) > 10:
                    # Keep action if it has text after "write"
                    pass
                else:
                    result["action"] = "write"
            else:
                # Default to read for "what's in clipboard", "paste", etc.
                result["action"] = "read"

        # FORCE system_control for system control tasks
        is_system_control_task = getattr(self, '_current_goal_is_system_control', False)
        if is_system_control_task:
            result["tool"] = "system_control"
            goal = getattr(self, '_current_goal', '').lower()
            current_action = (result.get("action") or "").lower()

            # Determine correct action based on goal keywords
            if 'volume' in goal:
                if 'set' in goal or any(c.isdigit() for c in goal):
                    result["action"] = current_action if 'volume' in current_action else "set_volume"
                else:
                    result["action"] = "get_volume"
            elif 'brightness' in goal:
                if 'set' in goal or any(c.isdigit() for c in goal):
                    result["action"] = current_action if 'brightness' in current_action else "set_brightness"
                else:
                    result["action"] = "get_brightness"
            elif 'open' in goal or 'launch' in goal:
                result["action"] = current_action if 'open' in current_action else "open_app"
            elif 'lock' in goal:
                result["action"] = "lock_screen"
            else:
                # Default to get_system_info for "system info", "cpu", "ram", etc.
                result["action"] = "get_system_info"

        # FORCE notifications for notification tasks
        is_notification_task = getattr(self, '_current_goal_is_notification', False)
        if is_notification_task:
            result["tool"] = "notifications"
            goal = getattr(self, '_current_goal', '').lower()
            current_action = (result.get("action") or "").lower()

            # Determine correct action based on goal keywords
            if 'list' in goal or 'show' in goal or 'all' in goal:
                result["action"] = "list"
            elif 'remove' in goal or 'delete' in goal or 'cancel' in goal:
                result["action"] = current_action if 'remove' in current_action else "remove"
            elif 'clear' in goal:
                result["action"] = "clear"
            elif 'condition' in goal or 'alert when' in goal or 'notify when' in goal or ('cpu' in goal and ('above' in goal or 'exceed' in goal or '%' in goal)):
                result["action"] = current_action if 'condition' in current_action else "add_condition"
            elif 'schedule' in goal or 'every day' in goal or 'daily' in goal or 'weekday' in goal or 'weekly' in goal or 'every morning' in goal or 'every evening' in goal:
                result["action"] = current_action if 'scheduled' in current_action else "add_scheduled"
            else:
                # Default to add_reminder for "remind me", "in 30 minutes", etc.
                result["action"] = current_action if 'reminder' in current_action else "add_reminder"

        # FORCE tool_builder for tool builder tasks
        is_tool_builder_task = getattr(self, '_current_goal_is_tool_builder', False)
        if is_tool_builder_task:
            result["tool"] = "tool_builder"
            goal = getattr(self, '_current_goal', '').lower()
            current_action = (result.get("action") or "").lower()

            # Determine correct action based on goal keywords
            if 'list' in goal or 'show' in goal:
                result["action"] = "list"
            elif 'test' in goal:
                result["action"] = current_action if 'test' in current_action else "test"
            elif 'enable' in goal or 'activate' in goal:
                result["action"] = current_action if 'enable' in current_action else "enable"
            elif 'disable' in goal or 'deactivate' in goal:
                result["action"] = current_action if 'disable' in current_action else "disable"
            elif 'rollback' in goal or 'delete' in goal or 'remove' in goal:
                result["action"] = current_action if 'rollback' in current_action else "rollback"
            else:
                # Default to list for general tool builder queries
                result["action"] = current_action if current_action else "list"

        return result

    def _generate_default_code(self) -> str:
        """Generate default Python code based on the current goal."""
        goal = getattr(self, '_current_goal', '').lower()

        # Prime number check
        if 'prime' in goal:
            # Extract number from goal
            numbers = re.findall(r'\d+', goal)
            if numbers:
                n = numbers[0]
                return f"n = {n}; is_prime = n > 1 and all(n % i != 0 for i in range(2, int(n**0.5) + 1)); print(str(n) + ' is ' + ('' if is_prime else 'not ') + 'a prime number')"

        # Factorial
        if 'factorial' in goal:
            numbers = re.findall(r'\d+', goal)
            if numbers:
                n = numbers[0]
                return f"import math; print(f'factorial({n}) = {{math.factorial({n})}}')"

        # Fibonacci
        if 'fibonacci' in goal or 'fib' in goal:
            numbers = re.findall(r'\d+', goal)
            if numbers:
                n = numbers[0]
                return f"def fib(n): return n if n <= 1 else fib(n-1) + fib(n-2); print(f'fibonacci({n}) = {{fib({n})}}')"

        # Default: just print hello
        return "print('Code executed successfully')"

    def _clean_action(self, action: str) -> str:
        """Clean up action string from verbose local model outputs."""
        # Remove common prefixes that local models add
        prefixes_to_remove = [
            "use web_search tool to search for",
            "use web_search to search for",
            "search for",
            "search the web for",
            "search:",
            "query:",
            "use filesystem to",
            "use the",
        ]

        action_lower = action.lower()
        for prefix in prefixes_to_remove:
            if action_lower.startswith(prefix):
                action = action[len(prefix):].strip()
                action_lower = action.lower()

        # Remove quotes if present
        action = action.strip('"\'')

        # Remove markdown formatting
        action = action.replace("**", "").replace("`", "")

        return action.strip()

    def _extract_search_query(self, response: str) -> str:
        """Extract a search query from a verbose response."""
        # Look for quoted text
        quoted = re.findall(r'["\']([^"\']+)["\']', response)
        if quoted:
            return quoted[0]

        # Look for text after common patterns
        patterns = [
            r'search (?:for |query[: ]+)?["\']?([^"\'\n]+)',
            r'query[: ]+([^\n]+)',
        ]
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                return match.group(1).strip().strip('"\'')

        # Fallback: return a default query based on context
        return "latest news"

    def _extract_code_from_response(self, response: str) -> str:
        """Extract Python code from a verbose LLM response."""
        # Look for code blocks
        code_block = re.search(r'```(?:python)?\s*(.*?)```', response, re.DOTALL)
        if code_block:
            return code_block.group(1).strip()

        # Look for lines that look like Python code
        code_indicators = ['print(', 'import ', 'def ', 'for ', 'while ', 'if ', '=']
        for line in response.split('\n'):
            line = line.strip()
            if any(ind in line for ind in code_indicators):
                # This looks like code
                return line

        # Look for code after "ACTION:" anywhere in response
        action_match = re.search(r'ACTION:\s*(.+)', response, re.IGNORECASE)
        if action_match:
            return action_match.group(1).strip()

        return None

    def _parse_evaluation_response(self, response: str) -> dict:
        """Parse the evaluation response."""
        result = {"success": False, "confidence": 0, "progress": None, "next": None, "raw": response}

        response_lower = response.lower()

        for line in response.split("\n"):
            line_stripped = line.strip()
            line_lower = line_stripped.lower()

            if line_lower.startswith("success:"):
                result["success"] = "yes" in line_lower or "true" in line_lower
            elif line_lower.startswith("confidence:"):
                # Extract numeric confidence value
                conf_str = line_stripped[11:].strip()
                # Extract first number found
                conf_match = re.search(r'\d+', conf_str)
                if conf_match:
                    result["confidence"] = min(100, max(0, int(conf_match.group())))
            elif line_lower.startswith("progress:"):
                result["progress"] = line_stripped[9:].strip()
            elif line_lower.startswith("next:"):
                next_val = line_stripped[5:].strip().lower()
                result["next"] = next_val
                # Check if goal is complete
                if "complete" in next_val or "done" in next_val or "achieved" in next_val:
                    result["success"] = True

        # Fallback detection
        if result["progress"] is None:
            if "success" in response_lower or "found" in response_lower:
                result["progress"] = "Made progress"

        return result

    def _default_system_prompt(self) -> str:
        return "You are a helpful AI assistant. Be concise and direct."

    def _observer_prompt(self) -> str:
        return "You analyze situations. List only key observations. Be very brief."

    def _planner_prompt(self) -> str:
        return """You create simple action plans. Be brief.
CRITICAL: For ANY calculation, math, Python, or code task -> use code_executor FIRST. Do NOT search the web for how to do it. Just write and execute the code directly.
For local files -> filesystem.
For internet info -> web_search."""

    def _actor_prompt(self) -> str:
        return """You select actions. Output ONLY: TOOL, ACTION, REASONING lines.
For code_executor: ACTION must be actual Python code with print() to show results.
Do NOT describe code - write the actual code!"""

    def _evaluator_prompt(self) -> str:
        return """You evaluate results. Follow the format exactly.
Say 'NEXT: complete' when the goal is achieved."""

    def _memory_prompt(self) -> str:
        return "Summarize in 2-3 short sentences. Focus on what happened and what was learned."

    # =========================================================================
    # ALMA Emotional Intelligence Methods
    # =========================================================================

    def set_emotional_tone(self, enabled: bool = True):
        """Enable or disable automatic emotional tone in responses.

        Args:
            enabled: Whether to automatically add emotional context to prompts
        """
        self._auto_emotional_tone = enabled
        logger.info(f"[BRAIN] Automatic emotional tone: {'enabled' if enabled else 'disabled'}")

    def trigger_emotional_response(self, emotion: str, intensity: float = 0.7, reason: str = "manual"):
        """Trigger an emotional response in AURA.

        Args:
            emotion: Name of emotion (joy, curious, excited, etc.)
            intensity: Strength of emotion (0.0 to 1.0)
            reason: Why this emotion was triggered
        """
        if self._alma_enabled:
            try:
                trigger_emotion(emotion, intensity, reason)
                logger.debug(f"[BRAIN] Triggered emotion: {emotion} ({intensity})")
            except Exception as e:
                logger.warning(f"[BRAIN] Failed to trigger emotion: {e}")

    def get_emotional_state(self) -> Optional[dict]:
        """Get AURA's current emotional state.

        Returns:
            Dictionary with emotional state info, or None if ALMA not available
        """
        if not self._alma_enabled:
            return None
        try:
            return alma_engine.get_emotional_state()
        except Exception as e:
            logger.warning(f"[BRAIN] Failed to get emotional state: {e}")
            return None

    def get_mood_emoji(self) -> str:
        """Get emoji representing AURA's current mood.

        Returns:
            Mood emoji string
        """
        if self._alma_enabled:
            try:
                return get_mood_emoji()
            except Exception:
                pass
        return "🤖"

    def update_emotional_state(self, success: bool = True, user_satisfied: bool = True):
        """Update emotional state after an interaction.

        Args:
            success: Whether the response was successful
            user_satisfied: Whether the user seemed satisfied
        """
        if self._alma_enabled:
            try:
                process_response_outcome(success, user_satisfied)
            except Exception as e:
                logger.debug(f"[BRAIN] Failed to update emotional state: {e}")
