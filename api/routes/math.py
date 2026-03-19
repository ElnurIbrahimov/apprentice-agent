"""
Math Solver — step-by-step math problem solving with LaTeX output.
Uses Ollama directly for single-shot LLM calls (same pattern as agent_action.py).
"""

import json as _json
import logging
import os
import re

import httpx
from fastapi import APIRouter, HTTPException, Depends

from api.auth import require_api_key
from api.utils import safe_error_detail

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/math", tags=["math"], dependencies=[Depends(require_api_key)])

OLLAMA_BASE = os.getenv("OLLAMA_BASE_URL") or os.getenv("OLLAMA_HOST", "http://localhost:11434")

SYSTEM_PROMPT = (
    "You are a math expert. Solve problems step by step. "
    "Always respond with valid JSON matching exactly this structure: "
    "{solution, steps, latex, graph_data}. "
    "Use LaTeX notation for mathematical expressions."
)

SOLVE_INSTRUCTIONS = """Solve this math problem step by step.

Problem: {problem}

Respond with ONLY a JSON object — no markdown fences, no extra text — in exactly this format:
{{
  "solution": "the final answer as a plain string",
  "steps": ["Step 1: ...", "Step 2: ...", "Step 3: ..."],
  "latex": "LaTeX expression for the key formula or final answer, e.g. \\\\frac{{-b \\\\pm \\\\sqrt{{b^2-4ac}}}}{{2a}}",
  "graph_data": null
}}"""

EXPLAIN_INSTRUCTIONS = """Explain this math concept or problem clearly with examples.

Problem: {problem}

Respond with ONLY a JSON object — no markdown fences, no extra text — in exactly this format:
{{
  "solution": "a clear conceptual explanation",
  "steps": ["Key concept 1: ...", "Key concept 2: ...", "Example: ..."],
  "latex": "LaTeX expression illustrating the core formula if applicable, otherwise empty string",
  "graph_data": null
}}"""

GRAPH_INSTRUCTIONS = """Analyze this mathematical function and provide data points for graphing.

Function: {problem}

Respond with ONLY a JSON object — no markdown fences, no extra text — in exactly this format:
{{
  "solution": "description of the function and its key features",
  "steps": ["Step 1: identify the function", "Step 2: find key points", "Step 3: determine range"],
  "latex": "LaTeX representation of the function",
  "graph_data": {{
    "points": [[-10, 100], [-8, 64], [-6, 36], [-4, 16], [-2, 4], [0, 0], [2, 4], [4, 16], [6, 36], [8, 64], [10, 100]],
    "x_min": -10,
    "x_max": 10,
    "label": "y = x²"
  }}
}}

Generate at least 20 evenly spaced points across the domain. Compute the actual y values for the given function."""


def _build_prompt(problem: str, mode: str) -> str:
    if mode == "explain":
        template = EXPLAIN_INSTRUCTIONS
    elif mode == "graph_data":
        template = GRAPH_INSTRUCTIONS
    else:
        template = SOLVE_INSTRUCTIONS
    return SYSTEM_PROMPT + "\n\n" + template.format(problem=problem)


def _extract_json(text: str) -> dict:
    """Extract and parse the first JSON object from LLM response text."""
    # Try direct parse first
    text = text.strip()
    try:
        return _json.loads(text)
    except Exception:
        pass

    # Strip markdown code fences
    fenced = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    fenced = re.sub(r"\s*```$", "", fenced)
    try:
        return _json.loads(fenced.strip())
    except Exception:
        pass

    # Extract first {...} block (handles extra prose before/after)
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        try:
            return _json.loads(m.group())
        except Exception:
            pass

    raise ValueError(f"Could not extract JSON from LLM response: {text[:300]}")


@router.post("/solve")
async def solve_math(body: dict):
    """
    Solve a math problem using an LLM.

    Body: { "problem": str, "mode": "solve"|"explain"|"graph_data", "model": str|null }
    Returns: { "solution": str, "steps": [...], "latex": str, "graph_data": dict|null }
    """
    problem = body.get("problem", "").strip()
    if not problem:
        raise HTTPException(400, "problem is required")
    if len(problem) > 4000:
        raise HTTPException(400, "problem exceeds maximum length of 4000 characters")

    mode = body.get("mode", "solve")
    if mode not in ("solve", "explain", "graph_data"):
        mode = "solve"

    model = body.get("model") or os.getenv("AURA_MATH_MODEL", "nemotron-3-super:cloud")

    prompt = _build_prompt(problem, mode)

    try:
        async with httpx.AsyncClient(timeout=30) as c:
            r = await c.post(
                f"{OLLAMA_BASE}/api/generate",
                json={"model": model, "prompt": prompt, "stream": False},
            )
        r.raise_for_status()
        response_text = r.json().get("response", "")
    except Exception as e:
        logger.error("[MathSolver] LLM call failed: %s", e)
        raise HTTPException(500, safe_error_detail(e, "LLM call failed"))

    if not response_text:
        raise HTTPException(500, "Empty response from LLM")

    try:
        result = _extract_json(response_text)
    except ValueError as e:
        logger.warning("[MathSolver] JSON parse failed: %s", e)
        # Fallback: return the raw text as solution
        return {
            "solution": response_text[:2000],
            "steps": [],
            "latex": "",
            "graph_data": None,
        }

    # Normalise: ensure all expected keys exist
    return {
        "solution": str(result.get("solution", "")),
        "steps": list(result.get("steps") or []),
        "latex": str(result.get("latex", "")),
        "graph_data": result.get("graph_data") or None,
    }
