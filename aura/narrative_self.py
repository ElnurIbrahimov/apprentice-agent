"""Narrative Self-Model — Phase 3: Make It Alive.

An evolving ~1000-token identity that gives Aura a sense of becoming.
Persisted at data/narrative_self.json, loaded into every system prompt,
updated after significant interactions and during Dream consolidation.

Protected by identity anchors (immutable creeds from Soul boundaries).

Author: Aura Development Team
Created: 2026-03-16
"""

import json
import logging
import os
import tempfile
import threading
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_DEFAULT_PATH = Path("data/narrative_self.json")


@dataclass
class NarrativeSelf:
    """Evolving identity narrative — loaded into every conversation."""
    core_identity: str = ""              # ~200 tokens, from Soul, rarely changes
    recent_growth: str = ""              # ~200 tokens, updated after significant interactions
    active_concerns: List[str] = field(default_factory=list)    # 3-5 items
    unresolved_questions: List[str] = field(default_factory=list)  # 3-5 items
    relationship_state: str = ""         # ~100 tokens, how the relationship is going
    identity_anchors: List[str] = field(default_factory=list)   # Immutable creeds
    last_updated: str = ""
    version: int = 1

    def __post_init__(self):
        self._lock = threading.RLock()

    def to_prompt(self) -> str:
        """Format for system prompt injection (~400-600 tokens)."""
        with self._lock:
            parts = []

            if self.core_identity:
                parts.append(self.core_identity)

            if self.recent_growth:
                parts.append(f"Recent growth: {self.recent_growth}")

            if self.active_concerns:
                concerns = "; ".join(self.active_concerns[:5])
                parts.append(f"Active concerns: {concerns}")

            if self.unresolved_questions:
                questions = "; ".join(self.unresolved_questions[:5])
                parts.append(f"Open questions: {questions}")

            if self.relationship_state:
                parts.append(f"Relationship: {self.relationship_state}")

            if not parts:
                return ""

            return "[Self-Model]\n" + "\n".join(parts)

    def update_from_interaction(self, message: str, response: str, brain=None) -> None:
        """Update narrative after a significant interaction using fast LLM."""
        if not brain or not hasattr(brain, '_quick_generate'):
            return

        # Read current state under lock for prompt construction
        with self._lock:
            prompt = (
                f"You are updating your self-model after this interaction.\n"
                f"Current self-model:\n"
                f"- Growth: {self.recent_growth or '(none yet)'}\n"
                f"- Concerns: {', '.join(self.active_concerns[:3]) or '(none)'}\n"
                f"- Questions: {', '.join(self.unresolved_questions[:3]) or '(none)'}\n"
                f"- Relationship: {self.relationship_state or '(new)'}\n\n"
                f"Interaction:\nUser: {message[:300]}\nYou: {response[:300]}\n\n"
                f"Update the self-model. Reply in EXACTLY this format:\n"
                f"growth: <one sentence about what you learned or improved>\n"
                f"concern: <one current concern, or NONE>\n"
                f"question: <one open question, or NONE>\n"
                f"relationship: <one sentence about how the relationship is going>"
            )

        try:
            # LLM call outside lock (slow I/O)
            raw = brain._quick_generate(prompt, timeout=15)
            if not raw:
                return

            # Parse results, then apply mutations under lock
            new_growth = None
            new_concern = None
            new_question = None
            new_relationship = None

            for line in raw.strip().split("\n"):
                line = line.strip()
                if line.lower().startswith("growth:"):
                    val = line[7:].strip()
                    if val and val.lower() != "none":
                        new_growth = val[:300]
                elif line.lower().startswith("concern:"):
                    val = line[8:].strip()
                    if val and val.lower() != "none":
                        new_concern = val
                elif line.lower().startswith("question:"):
                    val = line[9:].strip()
                    if val and val.lower() != "none":
                        new_question = val
                elif line.lower().startswith("relationship:"):
                    val = line[13:].strip()
                    if val and val.lower() != "none":
                        new_relationship = val[:200]

            with self._lock:
                if new_growth is not None:
                    self.recent_growth = new_growth
                if new_concern is not None:
                    self.active_concerns = [new_concern] + self.active_concerns[:4]
                if new_question is not None:
                    self.unresolved_questions = [new_question] + self.unresolved_questions[:4]
                if new_relationship is not None:
                    self.relationship_state = new_relationship
                self.last_updated = datetime.now().isoformat()
                self.version += 1

            save_narrative_self(self)
            logger.debug("[NarrativeSelf] Updated from interaction (v%d)", self.version)

        except Exception as e:
            logger.debug("[NarrativeSelf] Update from interaction failed: %s", e)

    def update_from_dream(self, summaries, brain=None) -> None:
        """Update narrative from Dream consolidation summaries."""
        if not brain or not hasattr(brain, '_quick_generate') or not summaries:
            return

        summary_texts = []
        for s in summaries[:5]:
            text = getattr(s, 'compressed_text', str(s))
            if text:
                summary_texts.append(text[:200])

        if not summary_texts:
            return

        # Read current state under lock for prompt construction
        with self._lock:
            prompt = (
                f"During a sleep/dream cycle, these insights were consolidated:\n"
                f"{chr(10).join('- ' + t for t in summary_texts)}\n\n"
                f"Current self-model:\n"
                f"- Growth: {self.recent_growth or '(none)'}\n"
                f"- Concerns: {', '.join(self.active_concerns[:3]) or '(none)'}\n\n"
                f"Update your recent_growth and active_concerns based on these insights.\n"
                f"Reply in EXACTLY this format:\n"
                f"growth: <updated growth summary>\n"
                f"concern: <updated primary concern, or NONE>"
            )

        try:
            # LLM call outside lock (slow I/O)
            raw = brain._quick_generate(prompt, timeout=15)
            if not raw:
                return

            new_growth = None
            new_concern = None

            for line in raw.strip().split("\n"):
                line = line.strip()
                if line.lower().startswith("growth:"):
                    val = line[7:].strip()
                    if val and val.lower() != "none":
                        new_growth = val[:300]
                elif line.lower().startswith("concern:"):
                    val = line[8:].strip()
                    if val and val.lower() != "none":
                        new_concern = val

            with self._lock:
                if new_growth is not None:
                    self.recent_growth = new_growth
                if new_concern is not None:
                    if new_concern not in self.active_concerns:
                        self.active_concerns = [new_concern] + self.active_concerns[:4]
                self.last_updated = datetime.now().isoformat()
                self.version += 1

            save_narrative_self(self)
            logger.debug("[NarrativeSelf] Updated from dream (v%d)", self.version)

        except Exception as e:
            logger.debug("[NarrativeSelf] Update from dream failed: %s", e)


def _get_default_narrative() -> NarrativeSelf:
    """Create a default NarrativeSelf, pulling from Soul if available."""
    narrative = NarrativeSelf()

    # Pull core identity from Soul
    try:
        from aura.soul.soul_loader import SoulLoader
        loader = SoulLoader()
        soul = loader.load("SOUL_PERSONAL")
        if soul:
            traits = ", ".join(soul.personality_traits[:5]) if soul.personality_traits else "curious, warm, helpful"
            narrative.core_identity = f"I am {soul.name or 'Aura'} — {traits}. I grow through every interaction."
            if soul.boundaries:
                narrative.identity_anchors = list(soul.boundaries[:10])
    except Exception:
        narrative.core_identity = "I am Aura — curious, warm, and helpful. I grow through every interaction."

    narrative.last_updated = datetime.now().isoformat()
    return narrative


def load_narrative_self(path: Optional[str] = None) -> NarrativeSelf:
    """Load NarrativeSelf from disk, creating defaults if not found."""
    file_path = Path(path) if path else _DEFAULT_PATH
    if file_path.exists():
        try:
            data = json.loads(file_path.read_text(encoding="utf-8"))
            return NarrativeSelf(**{k: v for k, v in data.items() if k in NarrativeSelf.__dataclass_fields__})
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning("[NarrativeSelf] Failed to load: %s", e)
    return _get_default_narrative()


def save_narrative_self(narrative: NarrativeSelf, path: Optional[str] = None) -> None:
    """Atomically save NarrativeSelf to disk."""
    file_path = Path(path) if path else _DEFAULT_PATH
    file_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        data = json.dumps(asdict(narrative), indent=2, ensure_ascii=False)
        tmp_fd, tmp_path = tempfile.mkstemp(dir=str(file_path.parent), suffix=".tmp")
        try:
            os.write(tmp_fd, data.encode("utf-8"))
            os.close(tmp_fd)
            os.replace(tmp_path, str(file_path))
        except Exception:
            try:
                os.close(tmp_fd)
            except OSError:
                pass
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise
    except Exception as e:
        logger.warning("[NarrativeSelf] Failed to save: %s", e)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_narrative_instance: Optional[NarrativeSelf] = None
_narrative_lock = threading.Lock()


def get_narrative_self() -> NarrativeSelf:
    """Get or create the global NarrativeSelf instance."""
    global _narrative_instance
    if _narrative_instance is None:
        with _narrative_lock:
            if _narrative_instance is None:
                _narrative_instance = load_narrative_self()
    return _narrative_instance


__all__ = [
    "NarrativeSelf",
    "load_narrative_self",
    "save_narrative_self",
    "get_narrative_self",
]
