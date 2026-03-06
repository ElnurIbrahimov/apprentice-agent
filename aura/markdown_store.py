"""
MarkdownStore - Clawdbot-Style Memory System for AURA v3.0

Human-readable markdown files for persistent memory storage.
Each memory type gets its own .md file that can be edited by
both AURA and the user.

Memory Files:
- user_profile.md: Who the user is, preferences, context
- conversations.md: Key conversation highlights
- learned_facts.md: Things AURA has learned about the world
- emotional_state.md: Current mood and emotional context
- patterns.md: Recognized behavioral patterns
"""

import os
import re
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class MemoryEntry:
    """A single memory entry."""
    content: str
    timestamp: str
    tags: List[str] = field(default_factory=list)
    importance: float = 0.5  # 0.0 to 1.0

    def to_markdown(self) -> str:
        """Convert to markdown format."""
        tags_str = ", ".join(self.tags) if self.tags else "general"
        return f"- **[{self.timestamp}]** {self.content} `[{tags_str}]` (importance: {self.importance:.1f})"


class MarkdownStore:
    """
    Markdown-based memory storage system.

    Stores memories in human-readable .md files that can be:
    - Viewed and edited by users
    - Easily backed up
    - Version controlled with git
    """

    # Memory file definitions
    MEMORY_FILES = {
        "user_profile": {
            "filename": "user_profile.md",
            "header": "# User Profile\n\nInformation about the user.\n\n",
            "sections": ["Basic Info", "Preferences", "Goals", "Context"]
        },
        "conversations": {
            "filename": "conversations.md",
            "header": "# Conversation Highlights\n\nKey moments and insights from our conversations.\n\n",
            "sections": ["Recent", "Important", "Unresolved"]
        },
        "learned_facts": {
            "filename": "learned_facts.md",
            "header": "# Learned Facts\n\nThings I've learned and should remember.\n\n",
            "sections": ["User-Specific", "Technical", "General"]
        },
        "emotional_state": {
            "filename": "emotional_state.md",
            "header": "# Emotional State\n\nCurrent mood and emotional context.\n\n",
            "sections": ["Current Mood", "Mood History", "Triggers"]
        },
        "patterns": {
            "filename": "patterns.md",
            "header": "# Recognized Patterns\n\nBehavioral patterns I've noticed.\n\n",
            "sections": ["User Patterns", "Conversation Patterns", "Temporal Patterns"]
        }
    }

    def __init__(self, data_dir: Optional[str] = None):
        """
        Initialize the markdown store.

        Args:
            data_dir: Directory for memory files (default: data/memory at project root)
        """
        if data_dir is None:
            data_dir = Path(__file__).parent.parent / "data" / "memory"

        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Initialize all memory files
        self._init_memory_files()

        logger.info(f"MarkdownStore initialized at {self.data_dir}")

    def _init_memory_files(self) -> None:
        """Create memory files if they don't exist."""
        for mem_type, config in self.MEMORY_FILES.items():
            filepath = self.data_dir / config["filename"]
            if not filepath.exists():
                content = config["header"]
                for section in config["sections"]:
                    content += f"## {section}\n\n"
                filepath.write_text(content, encoding="utf-8")
                logger.info(f"Created memory file: {filepath}")

    def _get_file_path(self, memory_type: str) -> Path:
        """Get the path for a memory type."""
        if memory_type not in self.MEMORY_FILES:
            raise ValueError(f"Unknown memory type: {memory_type}")
        return self.data_dir / self.MEMORY_FILES[memory_type]["filename"]

    def read(self, memory_type: str) -> str:
        """
        Read entire memory file.

        Args:
            memory_type: Type of memory to read

        Returns:
            Full markdown content
        """
        filepath = self._get_file_path(memory_type)
        try:
            return filepath.read_text(encoding="utf-8")
        except IOError as e:
            logger.error(f"Error reading {memory_type}: {e}")
            return ""

    def read_section(self, memory_type: str, section: str) -> str:
        """
        Read a specific section from a memory file.

        Args:
            memory_type: Type of memory
            section: Section header to extract

        Returns:
            Content of that section
        """
        content = self.read(memory_type)

        # Find section using regex
        pattern = rf"## {re.escape(section)}\s*\n(.*?)(?=\n## |\Z)"
        match = re.search(pattern, content, re.DOTALL)

        if match:
            return match.group(1).strip()
        return ""

    def add_entry(
        self,
        memory_type: str,
        section: str,
        content: str,
        tags: Optional[List[str]] = None,
        importance: float = 0.5
    ) -> bool:
        """
        Add an entry to a memory section.

        Args:
            memory_type: Type of memory
            section: Section to add to
            content: The memory content
            tags: Optional tags for categorization
            importance: How important (0.0-1.0)

        Returns:
            True if successful
        """
        filepath = self._get_file_path(memory_type)

        try:
            full_content = filepath.read_text(encoding="utf-8")

            # Create entry
            entry = MemoryEntry(
                content=content,
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M"),
                tags=tags or [],
                importance=max(0.0, min(1.0, importance))
            )

            # Find section and append
            section_pattern = rf"(## {re.escape(section)}\s*\n)"
            match = re.search(section_pattern, full_content)

            if match:
                insert_pos = match.end()
                new_content = (
                    full_content[:insert_pos] +
                    entry.to_markdown() + "\n" +
                    full_content[insert_pos:]
                )
                filepath.write_text(new_content, encoding="utf-8")
                logger.info(f"Added entry to {memory_type}/{section}")
                return True
            else:
                logger.warning(f"Section '{section}' not found in {memory_type}")
                return False

        except IOError as e:
            logger.error(f"Error adding entry: {e}")
            return False

    def update_section(
        self,
        memory_type: str,
        section: str,
        new_content: str
    ) -> bool:
        """
        Replace entire section content.

        Args:
            memory_type: Type of memory
            section: Section to update
            new_content: New content for the section

        Returns:
            True if successful
        """
        filepath = self._get_file_path(memory_type)

        try:
            full_content = filepath.read_text(encoding="utf-8")

            # Replace section content
            pattern = rf"(## {re.escape(section)}\s*\n).*?(?=\n## |\Z)"
            replacement = rf"\1{new_content}\n\n"

            new_full = re.sub(pattern, replacement, full_content, flags=re.DOTALL)

            if new_full != full_content:
                filepath.write_text(new_full, encoding="utf-8")
                logger.info(f"Updated section {memory_type}/{section}")
                return True
            return False

        except IOError as e:
            logger.error(f"Error updating section: {e}")
            return False

    def search(
        self,
        query: str,
        memory_types: Optional[List[str]] = None,
        max_results: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search across memory files.

        Args:
            query: Search query (simple text matching)
            memory_types: Types to search (default: all)
            max_results: Maximum results to return

        Returns:
            List of matching entries with context
        """
        if memory_types is None:
            memory_types = list(self.MEMORY_FILES.keys())

        results = []
        query_lower = query.lower()
        query_words = set(query_lower.split())

        for mem_type in memory_types:
            content = self.read(mem_type)
            lines = content.split("\n")

            for i, line in enumerate(lines):
                line_lower = line.lower()

                # Check for word overlap
                line_words = set(line_lower.split())
                overlap = len(query_words & line_words)

                if overlap > 0 or query_lower in line_lower:
                    # Get context (surrounding lines)
                    start = max(0, i - 1)
                    end = min(len(lines), i + 2)
                    context = "\n".join(lines[start:end])

                    results.append({
                        "memory_type": mem_type,
                        "line_number": i + 1,
                        "content": line.strip(),
                        "context": context,
                        "relevance": overlap + (1 if query_lower in line_lower else 0)
                    })

        # Sort by relevance
        results.sort(key=lambda x: x["relevance"], reverse=True)
        return results[:max_results]

    def get_recent(
        self,
        memory_type: str,
        limit: int = 5
    ) -> List[str]:
        """
        Get most recent entries from a memory type.

        Args:
            memory_type: Type of memory
            limit: Maximum entries to return

        Returns:
            List of recent entries
        """
        content = self.read(memory_type)

        # Find timestamped entries
        pattern = r"- \*\*\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2})\]\*\* (.+)"
        matches = re.findall(pattern, content)

        # Sort by timestamp (most recent first)
        matches.sort(key=lambda x: x[0], reverse=True)

        return [match[1] for match in matches[:limit]]

    def get_context_for_llm(self, max_tokens: int = 500) -> str:
        """
        Build a context string for LLM prompts.

        Args:
            max_tokens: Approximate max characters (rough token estimate)

        Returns:
            Formatted context string
        """
        context_parts = []

        # User profile basics
        profile = self.read_section("user_profile", "Basic Info")
        if profile:
            context_parts.append(f"User: {profile[:150]}")

        # Current mood
        mood = self.read_section("emotional_state", "Current Mood")
        if mood:
            context_parts.append(f"Mood: {mood[:100]}")

        # Recent conversations
        recent = self.get_recent("conversations", limit=3)
        if recent:
            context_parts.append("Recent topics: " + "; ".join(r[:50] for r in recent))

        # Key facts
        facts = self.get_recent("learned_facts", limit=3)
        if facts:
            context_parts.append("Remember: " + "; ".join(f[:50] for f in facts))

        result = "\n".join(context_parts)
        return result[:max_tokens]

    def export_all(self) -> Dict[str, str]:
        """Export all memories as a dictionary."""
        return {
            mem_type: self.read(mem_type)
            for mem_type in self.MEMORY_FILES
        }

    def clear_memory(self, memory_type: str, confirm: bool = False) -> bool:
        """
        Clear a memory file (reset to default).

        Args:
            memory_type: Type to clear
            confirm: Must be True to actually clear

        Returns:
            True if cleared
        """
        if not confirm:
            logger.warning("Clear not confirmed")
            return False

        config = self.MEMORY_FILES.get(memory_type)
        if not config:
            return False

        filepath = self._get_file_path(memory_type)
        content = config["header"]
        for section in config["sections"]:
            content += f"## {section}\n\n"

        filepath.write_text(content, encoding="utf-8")
        logger.info(f"Cleared memory: {memory_type}")
        return True

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about stored memories."""
        stats = {}
        for mem_type in self.MEMORY_FILES:
            content = self.read(mem_type)
            entry_count = len(re.findall(r"- \*\*\[", content))
            stats[mem_type] = {
                "entries": entry_count,
                "size_bytes": len(content.encode("utf-8"))
            }
        return stats

    def sync_emotional_state(self, state: Dict[str, Any]) -> bool:
        """
        Sync emotional state from JSON to markdown.

        Args:
            state: Emotional state dict with mood, energy, warmth, etc.

        Returns:
            True if synced successfully
        """
        mood = state.get('mood', 'neutral')
        energy = state.get('energy', 0.5)
        warmth = state.get('warmth', 0.5)
        engagement = state.get('engagement', 0.5)
        curiosity = state.get('curiosity', 0.5)
        reason = state.get('mood_reason', 'Baseline state')
        updated = state.get('updated_at', 'Unknown')

        content = f"""**Current State** (auto-synced)
- Mood: {mood}
- Energy: {energy:.0%}
- Warmth: {warmth:.0%}
- Engagement: {engagement:.0%}
- Curiosity: {curiosity:.0%}
- Reason: {reason}
- Last updated: {updated}
"""
        return self.update_section("emotional_state", "Current Mood", content)

    def sync_patterns(self, patterns: Dict[str, Any]) -> bool:
        """
        Sync patterns from JSON to markdown.

        Args:
            patterns: Dict of pattern name -> pattern data

        Returns:
            True if synced successfully
        """
        # Organize patterns by type
        user_patterns = []
        conversation_patterns = []
        temporal_patterns = []

        for name, pattern in patterns.items():
            ptype = pattern.get('pattern_type', 'unknown')
            desc = pattern.get('description', name)
            confidence = pattern.get('confidence', 0)
            occurrences = pattern.get('occurrences', 0)

            entry = f"- {desc} (confidence: {confidence:.0%}, seen {occurrences}x)"

            if ptype == 'temporal':
                temporal_patterns.append(entry)
            elif ptype == 'sequence':
                conversation_patterns.append(entry)
            elif ptype in ['cluster', 'behavioral']:
                user_patterns.append(entry)

        # Update each section
        success = True

        if user_patterns:
            content = "**Auto-synced patterns:**\n" + "\n".join(user_patterns[:10])
            success = success and self.update_section("patterns", "User Patterns", content)

        if conversation_patterns:
            content = "**Auto-synced patterns:**\n" + "\n".join(conversation_patterns[:10])
            success = success and self.update_section("patterns", "Conversation Patterns", content)

        if temporal_patterns:
            content = "**Auto-synced patterns:**\n" + "\n".join(temporal_patterns[:10])
            success = success and self.update_section("patterns", "Temporal Patterns", content)

        return success

    def sync_from_files(self, emotional_state_file: Path, patterns_file: Path) -> bool:
        """
        Sync markdown from JSON files.

        Args:
            emotional_state_file: Path to emotional_state.json
            patterns_file: Path to patterns.json

        Returns:
            True if all syncs successful
        """
        import json
        success = True

        # Sync emotional state
        if emotional_state_file.exists():
            try:
                data = json.loads(emotional_state_file.read_text(encoding="utf-8"))
                success = success and self.sync_emotional_state(data)
                logger.info("Synced emotional state to markdown")
            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"Error syncing emotional state: {e}")
                success = False

        # Sync patterns
        if patterns_file.exists():
            try:
                data = json.loads(patterns_file.read_text(encoding="utf-8"))
                success = success and self.sync_patterns(data)
                logger.info("Synced patterns to markdown")
            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"Error syncing patterns: {e}")
                success = False

        return success

    def extract_and_store_profile(self, message: str) -> Optional[str]:
        """
        Extract profile information from a user message and store it.

        Args:
            message: User's message to analyze

        Returns:
            Section name where data was stored, or None if no profile info found
        """
        message_lower = message.lower()
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')

        # Profile extraction patterns
        # Note: Patterns are designed to avoid false positives from emotional statements
        profile_patterns = {
            "Basic Info": [
                (r"my name is (\w+)", "Name: {}"),
                (r"i(?:'m| am) a(?:n)? (developer|engineer|designer|student|teacher|manager|writer|artist|programmer|scientist|researcher|consultant|freelancer|professional)\b", "Role: {}"),
                (r"i work (?:as|at|for) (.+?)(?:\.|,|$)", "Work: {}"),
                (r"i live in (.+?)(?:\.|,|$)", "Location: {}"),
                (r"i(?:'m| am) (\d+)(?: years old)?", "Age: {}"),
            ],
            "Preferences": [
                (r"i (?:really )?(?:like|love|enjoy) (.+?)(?:\.|,|$)", "Likes: {}"),
                (r"i (?:prefer|always use) (.+?)(?:\.|,|$)", "Prefers: {}"),
                (r"i (?:hate|dislike|can't stand) (.+?)(?:\.|,|$)", "Dislikes: {}"),
                (r"my favorite (\w+) is (.+?)(?:\.|,|$)", "Favorite {}: {}"),
            ],
            "Goals": [
                (r"i(?:'m| am) trying to (.+?)(?:\.|,|$)", "Trying to: {}"),
                (r"i want to (.+?)(?:\.|,|$)", "Wants to: {}"),
                (r"my goal is (?:to )?(.+?)(?:\.|,|$)", "Goal: {}"),
                (r"i need to (.+?)(?:\.|,|$)", "Needs to: {}"),
                (r"i(?:'m| am) learning (.+?)(?:\.|,|$)", "Learning: {}"),
            ],
            "Context": [
                (r"i(?:'m| am) (?:currently )?working on (.+?)(?:\.|,|$)", "Working on: {}"),
                (r"(?:today|this week|recently) i (.+?)(?:\.|,|$)", "Recent: {}"),
                (r"i(?:'m| am) using (.+?)(?:\.|,|$)", "Uses: {}"),
            ],
        }

        stored_section = None

        for section, patterns in profile_patterns.items():
            for pattern_tuple in patterns:
                if len(pattern_tuple) == 2:
                    pattern, template = pattern_tuple
                else:
                    continue

                if template is None:
                    continue

                match = re.search(pattern, message_lower)
                if match:
                    groups = match.groups()
                    if len(groups) == 1:
                        extracted = template.format(groups[0].strip())
                    elif len(groups) == 2:
                        extracted = template.format(groups[0].strip(), groups[1].strip())
                    else:
                        continue

                    # Store in the appropriate section (timestamp added by add_entry)
                    self.add_entry(
                        "user_profile",
                        section,
                        extracted,
                        importance=0.7
                    )
                    stored_section = section
                    logger.info(f"Extracted profile info to {section}: {extracted}")

        return stored_section


if __name__ == "__main__":
    print("=" * 60)
    print("MarkdownStore - Memory System Test")
    print("=" * 60)

    store = MarkdownStore()

    # Test adding entries
    print("\n--- Adding test entries ---")
    store.add_entry(
        "user_profile",
        "Basic Info",
        "User prefers dark mode and concise answers",
        tags=["preference", "ui"],
        importance=0.7
    )

    store.add_entry(
        "conversations",
        "Recent",
        "Discussed AURA v3.0 architecture improvements",
        tags=["development", "aura"],
        importance=0.8
    )

    store.add_entry(
        "learned_facts",
        "Technical",
        "User's system has 8GB VRAM (RTX 4060)",
        tags=["hardware", "constraints"],
        importance=0.9
    )

    # Test reading
    print("\n--- Reading user profile ---")
    print(store.read_section("user_profile", "Basic Info"))

    # Test search
    print("\n--- Searching for 'AURA' ---")
    results = store.search("AURA")
    for r in results:
        print(f"  [{r['memory_type']}] {r['content'][:50]}...")

    # Test context for LLM
    print("\n--- LLM Context ---")
    print(store.get_context_for_llm())

    # Stats
    print("\n--- Statistics ---")
    stats = store.get_stats()
    for mem_type, data in stats.items():
        print(f"  {mem_type}: {data['entries']} entries, {data['size_bytes']} bytes")

    print("\n" + "=" * 60)
    print("Test complete!")
