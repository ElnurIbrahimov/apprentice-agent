"""
Context Builder - The Heart of AURA

This builds the rich context that makes AURA feel alive.
Every response includes: SOUL + USER + MEMORIES + HISTORY
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

from aura.memory.daily_log import DailyLogManager
from aura.memory.memory_store import MemoryStore
from aura.memory.vector_search import VectorSearch

logger = logging.getLogger(__name__)


class ContextBuilder:
    """
    Builds rich context for every AURA response.

    This is what makes AURA remember and feel alive.
    """

    def __init__(self, data_dir: str = "aura/data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.daily_log = DailyLogManager(str(self.data_dir / "memory"))
        self.memory_store = MemoryStore(str(self.data_dir))
        self.vector_search = VectorSearch(str(self.data_dir / "memory"))

        # Load static files
        self.soul = self._load_file("SOUL.md")
        self.heartbeat = self._load_file("HEARTBEAT.md")

        # Conversation histories (in-memory, per chat)
        self.conversations: Dict[str, List[Dict]] = {}

        logger.info("ContextBuilder initialized")

    def _load_file(self, filename: str) -> str:
        """Load a file from data directory."""
        filepath = self.data_dir / filename
        if filepath.exists():
            return filepath.read_text(encoding='utf-8')
        return ""

    def build_system_prompt(
        self,
        user_message: str,
        chat_id: str = "default",
        include_memories: bool = True
    ) -> str:
        """
        Build the complete system prompt with all context.

        This is injected into every LLM call.
        """

        parts = []

        # 1. SOUL (personality)
        parts.append("# WHO YOU ARE\n")
        parts.append(self.soul)
        parts.append("\n")

        # 2. USER profile
        user_profile = self.memory_store.get_user_profile()
        if user_profile:
            # Only include learned parts (not placeholder text)
            learned_parts = []
            for line in user_profile.split('\n'):
                if line.strip() and line.startswith('- ') and '[To be learned]' not in line:
                    learned_parts.append(line)

            if learned_parts:
                parts.append("\n# ABOUT THE PERSON YOU'RE TALKING TO\n")
                for line in learned_parts:
                    parts.append(line + "\n")

        # 3. Recent memories (today + yesterday logs)
        today_log = self.daily_log.get_today_log()
        yesterday_log = self.daily_log.get_yesterday_log()

        if today_log or yesterday_log:
            parts.append("\n# RECENT CONTEXT\n")

            if today_log:
                # Only include last few interactions to save context
                sections = today_log.split("## [")[-5:]  # Last 5 interactions
                if sections and any(s.strip() for s in sections):
                    parts.append("\n## Earlier Today:\n")
                    for section in sections:
                        if section.strip():
                            # Truncate long sections
                            truncated = section[:500]
                            if len(section) > 500:
                                truncated += "..."
                            parts.append(f"[{truncated}]\n")

            if yesterday_log:
                sections = yesterday_log.split("## [")[-3:]  # Last 3 from yesterday
                if sections and any(s.strip() for s in sections):
                    parts.append("\n## Yesterday:\n")
                    for section in sections:
                        if section.strip():
                            truncated = section[:300]
                            if len(section) > 300:
                                truncated += "..."
                            parts.append(f"[{truncated}]\n")

        # 4. Relevant memories (semantic search)
        if include_memories and user_message:
            relevant = self.vector_search.search(user_message, top_k=3)
            if relevant:
                has_relevant = False
                relevant_parts = []
                for source, content, score in relevant:
                    if score > 0.1:  # Only include if somewhat relevant
                        has_relevant = True
                        truncated = content[:200]
                        if len(content) > 200:
                            truncated += "..."
                        relevant_parts.append(f"- [{source}] {truncated}")

                if has_relevant:
                    parts.append("\n# RELEVANT MEMORIES\n")
                    parts.extend([p + "\n" for p in relevant_parts])

            # Also check memory store for keyword matches
            memory_matches = self.memory_store.get_relevant_memories(user_message)
            if memory_matches:
                parts.append("\n# STORED FACTS\n")
                for memory in memory_matches[:5]:
                    parts.append(f"{memory}\n")

        # 5. Current time context
        now = datetime.now()
        parts.append(f"\n# CURRENT TIME\n")
        parts.append(f"- Date: {now.strftime('%A, %B %d, %Y')}\n")
        parts.append(f"- Time: {now.strftime('%I:%M %p')}\n")

        return "".join(parts)

    def get_conversation_history(self, chat_id: str, limit: int = 10) -> List[Dict]:
        """Get recent conversation history for a chat."""

        if chat_id not in self.conversations:
            self.conversations[chat_id] = []

        return self.conversations[chat_id][-limit:]

    def add_to_history(self, chat_id: str, role: str, content: str):
        """Add a message to conversation history."""

        if chat_id not in self.conversations:
            self.conversations[chat_id] = []

        self.conversations[chat_id].append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })

        # Keep only last 20 messages in memory
        if len(self.conversations[chat_id]) > 20:
            self.conversations[chat_id] = self.conversations[chat_id][-20:]

    def save_interaction(
        self,
        chat_id: str,
        user_message: str,
        aura_response: str,
        metadata: dict = None
    ):
        """Save interaction to daily log and update history."""

        # Add to conversation history
        self.add_to_history(chat_id, "user", user_message)
        self.add_to_history(chat_id, "assistant", aura_response)

        # Save to daily log
        self.daily_log.append_interaction(
            user_message=user_message,
            aura_response=aura_response,
            chat_id=chat_id,
            metadata=metadata
        )

        # Mark vector index as dirty
        self.vector_search.index_dirty = True
