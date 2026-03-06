"""
Memory Retriever - Makes AURA Actually Remember

This retrieves relevant memories and injects them into every response.
The key to making AURA feel like it knows you.
"""

import logging
import re
from pathlib import Path
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class MemoryRetriever:
    """
    Retrieves relevant memories for context injection.

    Uses hybrid retrieval: vector search via Qdrant + nomic-embed-text
    with graceful fallback to keyword matching.
    """

    # Common stop words to filter out
    STOP_WORDS = {
        "i", "you", "the", "a", "an", "is", "are", "was", "were",
        "it", "this", "that", "what", "how", "why", "when", "where",
        "do", "does", "did", "have", "has", "had", "be", "been",
        "will", "would", "could", "should", "can", "may", "might",
        "to", "for", "of", "in", "on", "at", "by", "with", "about",
        "just", "really", "very", "so", "and", "but", "or", "if",
        "my", "me", "your", "our", "their", "its", "am", "im",
        "hey", "hi", "hello", "thanks", "thank", "please", "yeah",
        "yes", "no", "ok", "okay", "sure", "well", "like", "know"
    }

    def __init__(self, memory_store=None, data_dir: Optional[Path] = None):
        """
        Initialize the memory retriever.

        Args:
            memory_store: MarkdownStore instance
            data_dir: Data directory path (uses default if not provided)
        """
        self.memory = memory_store
        if data_dir:
            self.data_dir = Path(data_dir)
        else:
            self.data_dir = Path(__file__).parent.parent / "data" / "memory"

        self.data_dir.mkdir(parents=True, exist_ok=True)

        # User profile storage
        self.user_profile_path = self.data_dir / "user_profile.md"
        self.user_profile = self._load_user_profile()

        logger.info(f"MemoryRetriever initialized at {self.data_dir}")
        logger.info(f"[MEMORY] Loaded profile: {len(self.user_profile)} facts")

    def _load_user_profile(self) -> Dict[str, str]:
        """Load user profile from disk."""
        profile = {}
        if self.user_profile_path.exists():
            try:
                content = self.user_profile_path.read_text(encoding='utf-8')
                for line in content.split('\n'):
                    line = line.strip()
                    if ':' in line and not line.startswith('#'):
                        key, value = line.split(':', 1)
                        key = key.strip().lower().replace('**', '')
                        value = value.strip()
                        if key and value:
                            profile[key] = value
            except Exception as e:
                logger.warning(f"Failed to load profile: {e}")
        return profile

    def _save_user_profile(self):
        """Save user profile to disk."""
        try:
            with open(self.user_profile_path, 'w', encoding='utf-8') as f:
                f.write("# User Profile\n\n")
                f.write(f"*Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n\n")
                for key, value in self.user_profile.items():
                    f.write(f"**{key.title()}**: {value}\n")
            logger.info(f"[MEMORY] Saved profile: {len(self.user_profile)} facts")
        except Exception as e:
            logger.error(f"Failed to save profile: {e}")

    MAX_PROFILE_FACTS = 200

    def store_fact(self, key: str, value: str):
        """Store a user fact (name, preference, etc.)."""
        self.user_profile[key.lower()] = value
        # Evict oldest entries if over limit (dict preserves insertion order in 3.7+)
        if len(self.user_profile) > self.MAX_PROFILE_FACTS:
            excess = len(self.user_profile) - self.MAX_PROFILE_FACTS
            for old_key in list(self.user_profile.keys())[:excess]:
                del self.user_profile[old_key]
        self._save_user_profile()
        logger.info(f"[MEMORY] Stored fact: {key}={value}")

    def get_fact(self, key: str) -> str:
        """Get a user fact."""
        return self.user_profile.get(key.lower(), "")

    def get_relevant_memories(
        self,
        user_message: str,
        limit: int = 10
    ) -> List[str]:
        """
        Get memories relevant to the current message using hybrid retrieval:
        1. Always inject user profile facts (name, preferences etc.)
        2. Vector search episodic Qdrant store via nomic-embed-text
        3. Fallback to keyword search if vector search fails
        """
        memories = []

        # 1. Always inject user profile facts
        profile_facts = self._get_profile_facts()
        memories.extend(profile_facts)

        # 2. Try vector search
        vector_memories = self._vector_search(user_message, limit=limit)
        if vector_memories:
            memories.extend(vector_memories)
        else:
            # Fallback: keyword search on flat files
            memories.extend(self._keyword_search(user_message, limit=limit))

        # Deduplicate
        seen = set()
        unique = []
        for m in memories:
            m_clean = m.strip()
            if m_clean and m_clean not in seen:
                seen.add(m_clean)
                unique.append(m_clean)

        return unique[:limit]

    def _get_profile_facts(self) -> List[str]:
        """Always-inject: user profile and recent learned facts."""
        facts = []
        if self.user_profile:
            for key, val in list(self.user_profile.items())[:5]:
                facts.append(f"{key.title()}: {val}")
        # Also read learned_facts.md last 5 entries
        facts_file = self.data_dir / "learned_facts.md"
        if facts_file.exists():
            try:
                lines = facts_file.read_text(encoding='utf-8').strip().split('\n')
                # Get last 5 non-empty, non-header lines
                recent = [l.strip() for l in lines if l.strip() and not l.startswith('#')][-5:]
                facts.extend(recent)
            except Exception:
                pass
        return facts

    def _embed_text(self, text: str) -> Optional[List[float]]:
        """Get embedding from Ollama nomic-embed-text."""
        try:
            import requests
            resp = requests.post(
                "http://localhost:11434/api/embeddings",
                json={"model": "nomic-embed-text:latest", "prompt": text},
                timeout=3,
            )
            if resp.status_code == 200:
                return resp.json().get("embedding")
        except Exception:
            pass
        return None

    def _vector_search(self, query: str, limit: int = 5) -> List[str]:
        """Search episodic memory via UnifiedMemory (fixes the raw QdrantClient singleton conflict)."""
        try:
            from aura.memory.unified_memory import get_unified_memory
            results = get_unified_memory().query(query, k=limit, sources=["episodic", "amem"])
            memories = [r.content[:200] for r in results if r.content and len(r.content) > 10]
            return memories
        except Exception as e:
            logger.debug(f"[MemoryRetriever] Vector search failed: {e}")
            return []

    def _keyword_search(self, user_message: str, limit: int = 8) -> List[str]:
        """Original keyword-based search (fallback)."""
        memories = []
        keywords = self._extract_keywords(user_message)
        if not keywords:
            return self._get_recent_memories(days=3)[:limit]

        memory_files = ["learned_facts.md", "conversations.md"]
        for filename in memory_files:
            filepath = self.data_dir / filename
            if filepath.exists():
                try:
                    content = filepath.read_text(encoding='utf-8')
                    file_memories = self._search_content(content, keywords)
                    memories.extend(file_memories)
                except Exception:
                    pass

        daily_memories = self._get_recent_memories(days=7)
        memories.extend(daily_memories)
        return memories[:limit]

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract meaningful keywords from text."""

        # Clean and split
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())

        # Filter out stop words and short words
        keywords = [
            w for w in words
            if w not in self.STOP_WORDS and len(w) > 2
        ]

        # Add bigrams for better matching
        for i in range(len(words) - 1):
            bigram = f"{words[i]} {words[i+1]}"
            if words[i] not in self.STOP_WORDS or words[i+1] not in self.STOP_WORDS:
                keywords.append(bigram)

        return keywords

    def _search_content(
        self,
        content: str,
        keywords: List[str]
    ) -> List[str]:
        """Search content for keyword matches."""

        matches = []
        content_lower = content.lower()

        for line in content.split('\n'):
            line = line.strip()

            # Skip headers and empty lines
            if not line or line.startswith('#'):
                continue

            # Skip metadata lines
            if line.startswith('-') and '**[' in line:
                # This is a timestamped entry - extract just the content
                match = re.search(r'\*\*\[.*?\]\*\*\s*(.+?)(?:\s*`\[|$)', line)
                if match:
                    line = match.group(1).strip()

            line_lower = line.lower()

            # Check for keyword matches
            score = 0
            for kw in keywords:
                if kw in line_lower:
                    score += 2 if len(kw) > 5 else 1

            if score > 0:
                # Clean up the line
                clean_line = re.sub(r'^\s*[-*]\s*', '', line)
                clean_line = re.sub(r'\s*`\[.*?\]`.*$', '', clean_line)
                clean_line = re.sub(r'\s*\(importance:.*\)$', '', clean_line)

                if clean_line and len(clean_line) > 5:
                    matches.append((score, clean_line.strip()))

        # Sort by score and return just the text
        matches.sort(key=lambda x: x[0], reverse=True)
        return [m[1] for m in matches[:10]]

    def _get_recent_memories(self, days: int = 7) -> List[str]:
        """Get memories from recent conversations."""

        memories = []

        # Check daily notes directory
        daily_dir = self.data_dir / "daily"
        if daily_dir.exists():
            cutoff = datetime.now() - timedelta(days=days)

            for filepath in daily_dir.glob("*.md"):
                try:
                    # Parse date from filename (YYYY-MM-DD.md)
                    date_str = filepath.stem
                    file_date = datetime.strptime(date_str, "%Y-%m-%d")

                    if file_date >= cutoff:
                        content = filepath.read_text(encoding='utf-8')
                        # Get key points (non-empty, non-header lines)
                        for line in content.split('\n'):
                            line = line.strip()
                            if line and not line.startswith('#'):
                                # Extract just the content part
                                if ']' in line:
                                    line = line.split(']', 1)[-1].strip()
                                if line and len(line) > 10:
                                    memories.append(f"[{date_str}] {line[:100]}")
                except (ValueError, OSError, UnicodeDecodeError):
                    continue

        return memories[:5]  # Limit recent memories

    def get_user_profile(self) -> Dict[str, Any]:
        """Get user profile information for context."""

        profile = {}

        profile_file = self.data_dir / "user_profile.md"
        if profile_file.exists():
            try:
                content = profile_file.read_text(encoding='utf-8')

                # Extract name
                name_match = re.search(r'Name:\s*(.+?)(?:\n|$)', content)
                if name_match:
                    profile["name"] = name_match.group(1).strip()

                # Extract work/role
                work_match = re.search(r'(?:Work|Role):\s*(.+?)(?:\n|$)', content)
                if work_match:
                    profile["work"] = work_match.group(1).strip()

                # Extract interests/likes
                likes = re.findall(r'Likes?:\s*(.+?)(?:\n|$)', content)
                if likes:
                    profile["interests"] = [l.strip() for l in likes[:5]]

                # Extract recent context
                context_match = re.search(r'(?:Working on|Recent):\s*(.+?)(?:\n|$)', content)
                if context_match:
                    profile["context"] = context_match.group(1).strip()

            except Exception as e:
                logger.warning(f"Could not read user profile: {e}")

        return profile

    def store_interaction(
        self,
        user_message: str,
        aura_response: str,
        chat_id: Optional[str] = None
    ) -> bool:
        """
        Store this interaction for future memory.

        Args:
            user_message: What the user said
            aura_response: What AURA responded
            chat_id: Optional chat identifier

        Returns:
            True if stored successfully
        """

        today = datetime.now().strftime("%Y-%m-%d")
        daily_dir = self.data_dir / "daily"
        daily_dir.mkdir(parents=True, exist_ok=True)

        daily_file = daily_dir / f"{today}.md"

        # Append to daily file
        timestamp = datetime.now().strftime("%H:%M")
        entry = f"\n[{timestamp}] User: {user_message[:100]}\n[{timestamp}] AURA: {aura_response[:100]}\n"

        try:
            with open(daily_file, "a", encoding='utf-8') as f:
                f.write(entry)

            # Also extract and store any facts
            self._extract_facts(user_message)

            return True

        except Exception as e:
            logger.error(f"Failed to store interaction: {e}")
            return False

    def _extract_facts(self, message: str) -> None:
        """Extract facts to remember from message."""

        msg_lower = message.lower()

        # Patterns that indicate facts to store with value extraction
        fact_patterns = [
            (r"my name is (\w+)", "name"),
            (r"i'm (\w+)", "name"),
            (r"i am (\w+)", "name"),
            (r"call me (\w+)", "name"),
            (r"i work (?:at|for) (.+?)(?:\.|,|$)", "work"),
            (r"i(?:'m| am) a(?:n)? (developer|engineer|designer|student|programmer|writer|artist)", "role"),
            (r"my dog(?:'s| is) name(?:d| is) (\w+)", "pet_dog"),
            (r"my cat(?:'s| is) name(?:d| is) (\w+)", "pet_cat"),
            (r"i live in (.+?)(?:\.|,|$)", "location"),
            (r"i'm from (.+?)(?:\.|,|$)", "location"),
            (r"i like (.+?)(?:\.|,|$)", "likes"),
            (r"i love (.+?)(?:\.|,|$)", "loves"),
            (r"you(?:'re| are) my (\w+)", "relationship"),
            (r"i(?:'m| am) your (\w+)", "relationship"),
        ]

        # Also store to learned_facts.md for history
        facts_file = self.data_dir / "learned_facts.md"

        for pattern, category in fact_patterns:
            match = re.search(pattern, msg_lower)
            if match:
                value = match.group(1).strip().title()
                try:
                    # Store to user profile (persisted)
                    self.store_fact(category, value)

                    # Also log to facts file for history
                    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
                    entry = f"\n- **[{timestamp}]** [{category}] {value} `[auto-extracted]`\n"
                    with open(facts_file, "a", encoding='utf-8') as f:
                        f.write(entry)
                    logger.info(f"Extracted fact: [{category}] = {value}")
                except Exception as e:
                    logger.warning(f"Failed to store fact: {e}")
                break  # Only store one fact per message


if __name__ == "__main__":
    print("=" * 60)
    print("Memory Retriever - Test")
    print("=" * 60)

    retriever = MemoryRetriever()

    # Test keyword extraction
    print("\n--- Keyword extraction ---")
    keywords = retriever._extract_keywords("How did my interview at Google go?")
    print(f"Keywords: {keywords}")

    # Test memory retrieval
    print("\n--- Memory retrieval ---")
    memories = retriever.get_relevant_memories("interview")
    print(f"Found {len(memories)} relevant memories:")
    for m in memories[:5]:
        print(f"  - {m[:80]}...")

    # Test user profile
    print("\n--- User profile ---")
    profile = retriever.get_user_profile()
    print(f"Profile: {profile}")

    # Test interaction storage
    print("\n--- Store interaction ---")
    success = retriever.store_interaction(
        "My cat's name is Whiskers",
        "That's a cute name! How old is Whiskers?"
    )
    print(f"Stored: {success}")

    print("\n" + "=" * 60)
    print("Test complete!")
