"""Shared JSON parsing utilities for LLM responses.

Consolidates the repeated pattern of stripping markdown fences, finding
JSON objects/arrays in free text, and fixing common LLM formatting errors.
"""

import json
import logging
import re
from typing import Any, Optional

logger = logging.getLogger(__name__)


def parse_llm_json(text: str, *, default: Any = None) -> Any:
    """Parse JSON from an LLM response, handling markdown fences and junk text.

    Strategy:
    1. Strip markdown code fences (```json ... ```)
    2. Try json.loads on cleaned text
    3. Fix trailing commas, try again
    4. Extract {…} or [...] substring, try that
    5. Return *default* on total failure
    """
    if not text:
        return default

    cleaned = text.strip()

    # Strip markdown code fences
    if cleaned.startswith("```"):
        first_nl = cleaned.find("\n")
        if first_nl != -1:
            cleaned = cleaned[first_nl + 1:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

    # Direct parse
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Fix trailing commas
    fixed = re.sub(r",\s*}", "}", cleaned)
    fixed = re.sub(r",\s*]", "]", fixed)
    if fixed != cleaned:
        try:
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass

    # Extract first JSON object
    obj_start = cleaned.find("{")
    obj_end = cleaned.rfind("}")
    if obj_start >= 0 and obj_end > obj_start:
        try:
            return json.loads(cleaned[obj_start : obj_end + 1])
        except json.JSONDecodeError:
            pass

    # Extract first JSON array
    arr_start = cleaned.find("[")
    arr_end = cleaned.rfind("]")
    if arr_start >= 0 and arr_end > arr_start:
        try:
            return json.loads(cleaned[arr_start : arr_end + 1])
        except json.JSONDecodeError:
            pass

    logger.debug("Failed to parse JSON from LLM response: %.100s", text)
    return default


def safe_json_loads(s: Optional[str]) -> Any:
    """Safe json.loads — returns None on failure (no LLM fence stripping)."""
    if not s:
        return None
    try:
        return json.loads(s)
    except (json.JSONDecodeError, TypeError):
        return None
