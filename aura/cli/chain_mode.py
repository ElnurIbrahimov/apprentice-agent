"""Prompt pipelines: chain multiple prompts where output feeds into the next."""

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

MAX_PREV_OUTPUT_CHARS = 4000


@dataclass
class ChainStep:
    prompt_template: str  # May contain {prev} placeholder for previous output
    model: Optional[str] = None  # Override model for this step


@dataclass
class Chain:
    name: str
    steps: list  # List[ChainStep]
    created_at: float = field(default_factory=time.time)


@dataclass
class ChainResult:
    chain_name: str
    step_results: list  # List[dict] with prompt, response, model, elapsed per step
    total_elapsed: float
    success: bool


CHAINS_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "chains"


def parse_chain(raw: str) -> list[ChainStep]:
    """Parse '/chain step1 -> step2 -> step3' syntax.

    Also supports per-step model override with @model suffix:
        'research X @kimi-k2.6:cloud -> summarize @nemotron-3-super:cloud'

    The @model marker must be at the END of a step (last non-space token
    prefixed with ``@``) so that addresses like ``bob@corp.com`` or
    Python decorators in the middle of a prompt don't get misread as model
    overrides.
    """
    import re as _re
    # Match exactly one trailing `@token` (no spaces inside the token, so
    # `@kimi-k2.6:cloud` matches but `bob@corp.com` in the middle does not).
    _MODEL_SUFFIX_RE = _re.compile(r"\s@([^\s@]+)\s*$")

    parts = [p.strip().strip('"').strip("'") for p in raw.split("->")]
    steps = []
    for p in parts:
        if not p:
            continue
        model = None
        m = _MODEL_SUFFIX_RE.search(p)
        if m:
            model = m.group(1)
            p = p[:m.start()].rstrip()
        steps.append(ChainStep(prompt_template=p, model=model))
    return steps


def _inject_context(step: ChainStep, prev_output: str) -> str:
    """Build the actual prompt for a step, injecting previous output."""
    prompt = step.prompt_template

    if not prev_output:
        return prompt

    # Truncate to last N chars if too long (tail is most relevant)
    if len(prev_output) > MAX_PREV_OUTPUT_CHARS:
        prev_output = "...\n" + prev_output[-MAX_PREV_OUTPUT_CHARS:]

    # Explicit placeholder — single replacement to avoid injection via prev_output
    if "{prev}" in prompt:
        idx = prompt.index("{prev}")
        return prompt[:idx] + prev_output + prompt[idx + 6:]

    # Auto-prepend context
    return f"Based on the following context:\n\n{prev_output}\n\n{prompt}"


def run_chain(brain, steps: list[ChainStep], on_step=None) -> ChainResult:
    """Execute a chain of prompts, feeding output forward.

    Args:
        brain: OllamaBrain instance with .think() method.
        steps: Ordered list of ChainStep to execute.
        on_step: Optional callback(step_num, total, result_dict) called after each step.

    Returns:
        ChainResult with all step outputs and timing.
    """
    results = []
    prev_output = ""
    start = time.time()

    for i, step in enumerate(steps):
        prompt = _inject_context(step, prev_output)

        step_start = time.time()
        try:
            response = brain.think(prompt, model_override=step.model, use_history=False)
            # brain.think() may return str or dict depending on code path
            if isinstance(response, dict):
                response = response.get("response", response.get("content", str(response)))
            if response is None:
                response = ""
        except Exception as e:
            logger.warning(f"Chain step {i + 1} failed: {e}")
            response = f"[Error: {e}]"

        elapsed = time.time() - step_start

        result = {
            "step": i + 1,
            "prompt": step.prompt_template,
            "response": response,
            "model": step.model or "auto",
            "elapsed": elapsed,
        }
        results.append(result)
        prev_output = response

        if on_step:
            on_step(i + 1, len(steps), result)

    return ChainResult(
        chain_name="",
        step_results=results,
        total_elapsed=time.time() - start,
        success=all(r["response"] and not r["response"].startswith("[Error:") for r in results),
    )


def save_chain(name: str, steps: list[ChainStep]) -> Path:
    """Save a named chain for reuse. Returns the file path."""
    CHAINS_DIR.mkdir(parents=True, exist_ok=True)
    path = CHAINS_DIR / f"{name}.json"
    data = {
        "name": name,
        "steps": [{"prompt_template": s.prompt_template, "model": s.model} for s in steps],
        "created_at": time.time(),
    }
    path.write_text(json.dumps(data, indent=2))
    return path


def load_chain(name: str) -> Optional[Chain]:
    """Load a saved chain by name."""
    path = CHAINS_DIR / f"{name}.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Failed to load chain '{name}': {e}")
        return None
    return Chain(
        name=data["name"],
        steps=[ChainStep(**s) for s in data["steps"]],
        created_at=data.get("created_at", 0),
    )


def list_chains() -> list[str]:
    """List all saved chain names."""
    if not CHAINS_DIR.exists():
        return []
    return sorted(p.stem for p in CHAINS_DIR.glob("*.json"))


def delete_chain(name: str) -> bool:
    """Delete a saved chain. Returns True if deleted."""
    path = CHAINS_DIR / f"{name}.json"
    if path.exists():
        path.unlink()
        return True
    return False
