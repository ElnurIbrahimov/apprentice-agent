"""
State Extraction Pipeline for the World Model — ADV-02 Phase 2.

Uses LLM to analyze conversations and extract structured state changes.
Runs as a background task after every conversation turn in brain.py.

Extraction flow:
  1. Format recent conversation messages
  2. Inject current world state summary for context
  3. Call LLM with extraction prompt
  4. Parse JSON response (resilient to markdown fences, preamble)
  5. Return structured dict for WorldModel.process_conversation()
"""

import json
import logging
import threading
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ============================================================================
# Extraction Prompt
# ============================================================================

EXTRACTION_PROMPT = """You are the World Model Updater for AURA, an AI assistant.

Your job: analyze the conversation below and extract any changes to the user's world state.

## Current World State Summary:
{world_state_summary}

## Conversation to Analyze:
{conversation}

## Examples

Example 1 — Project mention + belief:
USER: "I'm working on the dashboard with React, Sarah is helping"
Extract: project "dashboard" (action: "new" or "mention", technologies: ["React"]),
person "Sarah" (role: "collaborator", context: "helping with dashboard"),
belief "User is building a dashboard with React" (category: "project_state").

Example 2 — Goal + environment:
USER: "I need to finish the API docs by Friday, switched to VS Code"
Extract: goal "Finish the API docs" (horizon: "short_term", evidence: "due by Friday"),
environment change (key: "editor", category: "tool", value: "VS Code").

## Extract the following (JSON format):

{{
  "projects": [
    {{
      "name": "string",
      "action": "new|update|mention",
      "status_change": null,
      "new_blockers": [],
      "resolved_blockers": [],
      "progress_notes": "string",
      "technologies_mentioned": []
    }}
  ],
  "goals": [
    {{
      "description": "string",
      "action": "new|update|achieved",
      "horizon": "short_term|medium_term|long_term",
      "progress_delta": 0.0,
      "evidence": "string"
    }}
  ],
  "beliefs": [
    {{
      "statement": "string",
      "category": "user_intent|technical_constraint|preference|project_state|relationship|schedule|habit|environment",
      "confidence": 0.7,
      "contradicts_existing": null
    }}
  ],
  "people_mentioned": [
    {{
      "name": "string",
      "role": "string",
      "context": "string",
      "sentiment": "positive|neutral|negative"
    }}
  ],
  "environment_changes": [
    {{
      "key": "string",
      "category": "hardware|tool|habit|schedule|preference",
      "value": "string"
    }}
  ]
}}

Rules:
- Only extract information explicitly stated or strongly implied
- Set confidence based on how explicit the information is
- Flag contradictions with existing beliefs via contradicts_existing
- If nothing relevant, return empty arrays
- Do NOT hallucinate information not in the conversation"""


# ============================================================================
# StateExtractor
# ============================================================================

class StateExtractor:
    """
    Extracts world state changes from conversations using LLM.

    Uses a dedicated OllamaBrain(warmup=False) instance to avoid
    lock contention with the main brain. Includes debounce to prevent
    excessive extraction calls on rapid-fire messages.
    """

    # Debounce: minimum seconds between extractions
    MIN_EXTRACT_INTERVAL = 5.0

    # Conversation formatting limits
    MAX_MESSAGES = 6
    MAX_MESSAGE_LENGTH = 2000

    def __init__(self):
        self._brain = None
        self._brain_lock = threading.Lock()
        self._last_extract_time: float = 0.0
        self._extraction_count: int = 0
        self._stats = {"total": 0, "successes": 0, "parse_failures": 0, "empty_results": 0}

    def _get_brain(self):
        """Lazy-load a dedicated OllamaBrain (double-checked locking)."""
        if self._brain is None:
            with self._brain_lock:
                if self._brain is None:
                    try:
                        from aura.brain import OllamaBrain
                        self._brain = OllamaBrain(warmup=False)
                    except Exception as e:
                        logger.error(f"[StateExtractor] Failed to create brain: {e}")
                        raise
        return self._brain

    def extract(
        self,
        messages: List[Dict[str, str]],
        current_state_summary: str = "",
    ) -> Optional[Dict]:
        """
        Run the extraction pipeline on a conversation.

        Args:
            messages: Recent conversation messages [{"role": ..., "content": ...}]
            current_state_summary: WorldModel.get_context_summary() output

        Returns:
            Structured extraction dict, or None on failure.
        """
        if not messages:
            return None

        self._stats["total"] += 1

        # Format conversation
        conversation_text = self._format_conversation(messages)
        if not conversation_text.strip():
            return None

        # Build prompt
        prompt = EXTRACTION_PROMPT.format(
            world_state_summary=current_state_summary or "(empty — no prior state)",
            conversation=conversation_text,
        )

        # Call LLM — use _quick_generate (lightweight, timeout-protected) instead
        # of brain.think() which builds a full system prompt with 8+ module injections.
        try:
            brain = self._get_brain()
            response = brain._quick_generate(prompt, timeout=45)
        except Exception as e:
            logger.debug(f"[StateExtractor] LLM call failed: {e}")
            return None

        if not response:
            return None

        # Parse JSON response
        result = self._parse_json(response)
        if result is None:
            self._stats["parse_failures"] += 1
            return None

        self._stats["successes"] += 1
        self._extraction_count += 1
        self._last_extract_time = time.monotonic()

        # Check if all extracted arrays are empty
        all_empty = all(
            not result.get(key) for key in
            ("projects", "goals", "beliefs", "people_mentioned", "environment_changes")
        )
        if all_empty:
            self._stats["empty_results"] += 1

        return result

    def should_extract(self, messages: List[Dict[str, str]]) -> bool:
        """
        Decide whether extraction should run.

        Skips:
        - If no messages
        - If last user message is trivial (< 3 words)
        - If within debounce interval
        """
        if not messages:
            return False

        # Find last user message
        last_user = None
        for msg in reversed(messages):
            if msg.get("role") == "user":
                last_user = msg.get("content", "")
                break

        if last_user is None:
            return False

        # Skip trivial messages
        word_count = len(last_user.strip().split())
        if word_count < 3:
            return False

        # Debounce check
        from aura.config import Config
        min_interval = getattr(Config, "WORLD_MODEL_EXTRACTION_MIN_INTERVAL", self.MIN_EXTRACT_INTERVAL)
        elapsed = time.monotonic() - self._last_extract_time
        if elapsed < min_interval:
            return False

        return True

    def get_stats(self) -> Dict[str, int]:
        """Return extraction statistics."""
        return dict(self._stats)

    def _format_conversation(self, messages: List[Dict[str, str]]) -> str:
        """
        Format recent messages for the extraction prompt.

        Takes last MAX_MESSAGES, truncates each at MAX_MESSAGE_LENGTH.
        """
        recent = messages[-self.MAX_MESSAGES:]
        parts = []
        for msg in recent:
            role = msg.get("role", "unknown").upper()
            content = msg.get("content", "")
            if len(content) > self.MAX_MESSAGE_LENGTH:
                content = content[:self.MAX_MESSAGE_LENGTH] + "... [truncated]"
            parts.append(f"{role}: {content}")
        return "\n\n".join(parts)

    def _parse_json(self, text: str) -> Optional[Dict]:
        """Parse JSON from LLM response, handling markdown code blocks."""
        from aura.core.json_utils import parse_llm_json

        return parse_llm_json(text)


# ============================================================================
# Singleton
# ============================================================================

_state_extractor: Optional[StateExtractor] = None
_singleton_lock = threading.Lock()


def get_state_extractor() -> Optional[StateExtractor]:
    """Get or create the singleton StateExtractor (double-checked locking).

    Returns None if extraction is disabled in Config.
    """
    global _state_extractor
    from aura.config import Config
    if not getattr(Config, "WORLD_MODEL_EXTRACTION_ENABLED", True):
        return None

    if _state_extractor is None:
        with _singleton_lock:
            if _state_extractor is None:
                _state_extractor = StateExtractor()
    return _state_extractor
