#!/usr/bin/env python3
"""Seed the SkillStore with distilled skills from Aura's existing signal.

GEPA (weekly auto-evolver) needs >= 3 skills in the library to do anything. If
the library is sparse, the whole self-improvement loop is dormant. This script
primes the pump by:

  1. Pulling recent successful interactions from Unified Memory (episodic channel).
  2. Asking a cloud LLM to distill recurring workflows into structured skills.
  3. Validating the output and writing skills via SkillStore.save().

Usage:
    python scripts/seed_skill_store.py --dry-run          # preview only
    python scripts/seed_skill_store.py --limit 5          # write up to 5 skills
    python scripts/seed_skill_store.py --model qwen3.5:397b-cloud

The script is idempotent-ish: it skips candidates whose trigger patterns
overlap with an existing skill (based on exact trigger-pattern string match).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "qwen3.5:397b-cloud"
_MAX_INTERACTIONS_TO_SAMPLE = 50

_DISTILL_SYSTEM_PROMPT = """You are a skill distiller for an AI agent's skill library.

Given a sample of past successful interactions between a user and the agent, your
job is to identify RECURRING WORKFLOWS — patterns the agent does often that
should be encoded as reusable skills.

Return ONLY a JSON array. Each element MUST have exactly these fields:
  - "name": short, imperative (e.g. "Summarize a research paper")
  - "description": one sentence explaining when to use this skill
  - "category": one of: coding, writing, research, automation, analysis, communication, learning, custom
  - "trigger_patterns": array of 2-5 short phrases that signal this skill applies (e.g. ["summarize this paper", "tl;dr of this pdf"])
  - "procedure": 3-8 numbered steps, as plain text (not markdown)

Rules:
  - Output ONLY a JSON array. No prose, no code fences.
  - Only include workflows that appear 2+ times in the sample, or that are clearly generalizable.
  - Skip one-off tasks.
  - Keep each skill focused on ONE thing.
"""


def _fetch_recent_interactions(limit: int) -> list[dict]:
    """Pull recent successful interactions from UnifiedMemory's episodic channel."""
    try:
        from aura.memory.unified_memory import get_unified_memory
        um = get_unified_memory()
        # query() returns a mix of memory types — we just want episodic signal here.
        results = um.query(query="", limit=limit, sources=["episodic"]) or []
        return [r for r in results if isinstance(r, dict) and r.get("content")]
    except Exception as e:
        logger.warning("UnifiedMemory unavailable (%s) — falling back to empty sample", e)
        return []


def _call_llm(model: str, system_prompt: str, user_prompt: str, base_url: Optional[str] = None) -> str:
    """Call Ollama chat endpoint. Raises on network error."""
    if base_url is None:
        base_url = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "options": {"temperature": 0.3, "num_predict": 2048},
    }
    req = urllib.request.Request(
        f"{base_url}/api/chat",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            body = json.loads(resp.read().decode())
            return body.get("message", {}).get("content", "")
    except urllib.error.URLError as e:
        raise RuntimeError(f"Ollama call failed ({model}): {e}")


def _parse_candidates(raw: str) -> list[dict]:
    """Extract the JSON array from an LLM response, tolerating code fences."""
    text = raw.strip()
    # Strip markdown fences if the model disobeyed.
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    text = text.strip()
    # Find the first [ ... ] — models sometimes prepend commentary
    start = text.find("[")
    end = text.rfind("]")
    if start < 0 or end < 0 or end <= start:
        return []
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError as e:
        logger.warning("Could not parse LLM output as JSON: %s", e)
        return []
    return [d for d in data if isinstance(d, dict)]


_REQUIRED_FIELDS = {"name", "description", "category", "trigger_patterns", "procedure"}
_VALID_CATEGORIES = {
    "coding", "writing", "research", "automation", "analysis",
    "communication", "learning", "custom",
}


def _validate_candidate(cand: dict) -> tuple[bool, str]:
    """Return (ok, reason)."""
    missing = _REQUIRED_FIELDS - set(cand.keys())
    if missing:
        return False, f"missing fields: {sorted(missing)}"
    if not isinstance(cand["name"], str) or len(cand["name"]) < 3:
        return False, "name must be a non-empty string"
    if cand["category"] not in _VALID_CATEGORIES:
        return False, f"invalid category: {cand['category']!r}"
    tp = cand["trigger_patterns"]
    if not isinstance(tp, list) or not tp or not all(isinstance(t, str) for t in tp):
        return False, "trigger_patterns must be a non-empty list of strings"
    if not isinstance(cand["procedure"], str) or len(cand["procedure"]) < 20:
        return False, "procedure must be a non-trivial string"
    return True, ""


def _existing_triggers(store) -> set[str]:
    """Return the set of every trigger pattern already in the store (lowercased)."""
    triggers: set[str] = set()
    for skill_id in list(store.index.keys()):
        skill = store.load(skill_id)
        if skill is None:
            continue
        for t in (skill.trigger_patterns or []):
            if isinstance(t, str):
                triggers.add(t.strip().lower())
    return triggers


def seed(
    model: str,
    limit: int,
    dry_run: bool,
    storage_path: Optional[str] = None,
) -> dict[str, Any]:
    """Run one seeding cycle. Returns a summary dict."""
    from aura_skill_library.skill import Skill, SkillCategory
    from aura_skill_library.skill_store import SkillStore

    if storage_path is None:
        storage_path = str(_PROJECT_ROOT / "aura_data" / "skill_library")
    store = SkillStore(storage_path=storage_path)

    interactions = _fetch_recent_interactions(_MAX_INTERACTIONS_TO_SAMPLE)
    if not interactions:
        return {
            "status": "no_signal",
            "reason": "UnifiedMemory returned zero episodic interactions — nothing to distill.",
            "skills_added": 0,
        }

    # Build a compact textual sample the LLM can reason over.
    sample_lines = []
    for entry in interactions[:30]:
        content = str(entry.get("content", "")).replace("\n", " ")[:400]
        if content:
            sample_lines.append(f"- {content}")
    sample_text = "\n".join(sample_lines)

    user_prompt = (
        f"Here are {len(sample_lines)} recent interactions with the agent. "
        f"Identify up to {limit} distinct recurring workflows and return them as a JSON array.\n\n"
        f"{sample_text}"
    )

    logger.info("Calling %s to distill skills from %d interactions...", model, len(sample_lines))
    raw = _call_llm(model, _DISTILL_SYSTEM_PROMPT, user_prompt)
    candidates = _parse_candidates(raw)
    if not candidates:
        return {
            "status": "no_candidates",
            "reason": "LLM returned no parseable skill candidates.",
            "raw_preview": raw[:500],
            "skills_added": 0,
        }

    existing = _existing_triggers(store)
    accepted: list[dict] = []
    rejected: list[dict] = []

    for cand in candidates[:limit]:
        ok, reason = _validate_candidate(cand)
        if not ok:
            rejected.append({"name": cand.get("name"), "reason": reason})
            continue
        overlap = [t for t in cand["trigger_patterns"] if t.strip().lower() in existing]
        if overlap:
            rejected.append({
                "name": cand["name"],
                "reason": f"trigger overlap with existing skill: {overlap[:3]}",
            })
            continue
        accepted.append(cand)

    if dry_run:
        return {
            "status": "dry_run",
            "accepted": accepted,
            "rejected": rejected,
            "skills_added": 0,
        }

    added = 0
    for cand in accepted:
        skill = Skill.create(
            name=cand["name"],
            description=cand["description"],
            category=SkillCategory(cand["category"]),
            trigger_patterns=cand["trigger_patterns"],
            procedure=cand["procedure"],
            tags=["seeded"],
        )
        store.save(skill)
        added += 1
        logger.info("Saved skill: %s (%s)", skill.name, skill.id)

    return {
        "status": "ok",
        "skills_added": added,
        "rejected": rejected,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing")
    parser.add_argument("--limit", type=int, default=5, help="Max skills to add per run")
    parser.add_argument("--model", default=_DEFAULT_MODEL, help="Ollama model for distillation")
    parser.add_argument("--storage-path", default=None, help="Override skill library path")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    result = seed(
        model=args.model,
        limit=args.limit,
        dry_run=args.dry_run,
        storage_path=args.storage_path,
    )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
