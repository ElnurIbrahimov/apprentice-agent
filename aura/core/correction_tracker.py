"""Track and learn from user corrections to improve future responses."""
from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path

logger = logging.getLogger(__name__)

_CORRECTIONS_PATH = Path.home() / ".aura" / "corrections.json"

_CORRECTION_PATTERNS = [
    re.compile(r"^no[,.\s]", re.IGNORECASE),
    re.compile(r"^wrong[,.\s]", re.IGNORECASE),
    re.compile(r"^that'?s (?:not |in)?correct", re.IGNORECASE),
    re.compile(r"^actually[,.\s]", re.IGNORECASE),
    re.compile(r"^instead[,.\s]", re.IGNORECASE),
    re.compile(r"the correct (?:approach|way|answer)", re.IGNORECASE),
    re.compile(r"you should (?:have|use)", re.IGNORECASE),
    re.compile(r"don'?t (?:do|use) that", re.IGNORECASE),
    re.compile(r"^not like that", re.IGNORECASE),
    re.compile(r"^stop doing", re.IGNORECASE),
]


class CorrectionTracker:
    """Track user corrections and inject relevant ones into future prompts."""

    def __init__(self, path: Path = _CORRECTIONS_PATH):
        self._path = path
        self._corrections: list[dict] = []
        self._load()

    def _load(self) -> None:
        try:
            if self._path.exists():
                self._corrections = json.loads(self._path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            self._corrections = []

    def _save(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self._corrections, indent=1), encoding="utf-8")
            tmp.replace(self._path)
        except OSError:
            logger.debug("correction_save_failed", exc_info=True)

    def detect_correction(self, user_msg: str) -> bool:
        """Check if user_msg looks like a correction of the AI response."""
        text = user_msg.strip()
        if len(text) < 5:
            return False
        return any(p.search(text) for p in _CORRECTION_PATTERNS)

    def record(self, user_correction: str, original_response: str, context: str = "") -> None:
        """Record a correction."""
        self._corrections.append({
            "correction": user_correction[:500],
            "original": original_response[:500],
            "context": context[:200],
            "timestamp": time.time(),
        })
        # Keep last 100
        if len(self._corrections) > 100:
            self._corrections = self._corrections[-100:]
        self._save()

    def get_relevant_corrections(self, prompt: str, limit: int = 3) -> list[str]:
        """Find corrections relevant to the current prompt by keyword overlap."""
        if not self._corrections:
            return []

        prompt_words = set(prompt.lower().split())
        scored = []
        for c in self._corrections:
            correction_words = set(c["correction"].lower().split())
            context_words = set(c.get("context", "").lower().split())
            overlap = len(prompt_words & (correction_words | context_words))
            if overlap >= 2:
                scored.append((c["correction"], overlap))

        scored.sort(key=lambda x: -x[1])
        return [text for text, _ in scored[:limit]]

    def to_system_prompt_fragment(self, prompt: str) -> str:
        """Format relevant corrections for injection into system prompt."""
        relevant = self.get_relevant_corrections(prompt)
        if not relevant:
            return ""
        lines = ["## Previous user corrections (apply these lessons)"]
        for c in relevant:
            lines.append(f"- {c}")
        return "\n".join(lines)
