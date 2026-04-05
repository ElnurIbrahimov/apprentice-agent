"""Extract hand configuration from natural language descriptions using the brain LLM."""

import json
import logging
import re
import asyncio
from typing import Any, Optional

logger = logging.getLogger(__name__)

SCHEMA_PROMPT = """Extract a hand configuration from the user's description. Return ONLY valid JSON.

Schema:
{
  "name": "lowercase_snake_case, 2-40 chars",
  "description": "1 sentence",
  "goal": "what the hand should accomplish",
  "search_queries": ["query1", "query2"],
  "interval_minutes": 240,
  "idle_only": true,
  "trigger_on_drive": null,
  "model_preference": "fast",
  "max_tokens": 20000,
  "max_cost_usd": 0.20
}

Example input: "Monitor Hacker News for AI papers daily"
Example output: {"name": "hn_ai_monitor", "description": "Monitors Hacker News for AI research papers", "goal": "Find latest AI research papers on Hacker News", "search_queries": ["hacker news AI papers today", "latest AI research papers"], "interval_minutes": 1440}
"""


async def extract_hand_config(description: str, brain: Any) -> Optional[dict]:
    """Extract a hand configuration from a natural language description.

    Args:
        description: Natural language description of what the hand should do.
        brain: OllamaBrain instance for LLM calls.

    Returns:
        Validated config dict with is_custom=True, or None on failure.
    """
    prompt = f"{SCHEMA_PROMPT}\n\nUser description: {description}\n\nReturn ONLY the JSON object, no markdown, no explanation."

    try:
        response = await asyncio.to_thread(
            lambda: brain.think(prompt, system_prompt="You are a JSON config extractor. Output only valid JSON.")
        )
    except Exception as e:
        logger.error(f"[HandConfigExtractor] Brain call failed: {e}")
        return None

    if not response:
        logger.warning("[HandConfigExtractor] Brain returned empty response")
        return None

    raw = str(response).strip()

    # Try direct parse first
    config = None
    try:
        config = json.loads(raw)
    except json.JSONDecodeError:
        # Fallback: extract first {...} block
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            try:
                config = json.loads(match.group(0))
            except json.JSONDecodeError:
                logger.error(f"[HandConfigExtractor] Failed to parse JSON from response: {raw[:200]}")
                return None
        else:
            logger.error(f"[HandConfigExtractor] No JSON object found in response: {raw[:200]}")
            return None

    # Validate required fields
    name = config.get("name", "")
    if not re.match(r'^[a-z0-9_]{2,40}$', str(name)):
        logger.error(f"[HandConfigExtractor] Invalid name '{name}' — must match ^[a-z0-9_]{{2,40}}$")
        return None

    interval = config.get("interval_minutes")
    if interval is not None:
        try:
            interval = int(interval)
        except (ValueError, TypeError):
            interval = 240
        if interval < 30:
            logger.warning(f"[HandConfigExtractor] interval_minutes {interval} < 30, clamping to 30")
            interval = 30
        config["interval_minutes"] = interval

    config["is_custom"] = True
    return config
