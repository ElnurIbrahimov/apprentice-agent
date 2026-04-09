"""Multi-model debate: pit different LLMs against each other on a question."""
from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from dataclasses import dataclass, field
from typing import List, Optional

DEBATER_TIMEOUT = 60  # seconds per model call

from rich.console import Console
from rich.live import Live
from rich.panel import Panel

# ── Data classes ──────────────────────────────────────────────────────

@dataclass
class DebatePosition:
    model: str
    role: str  # "advocate", "critic", "analyst"
    argument: str = ""
    elapsed: float = 0.0
    done: bool = False
    error: str = ""


@dataclass
class DebateResult:
    question: str
    positions: List[DebatePosition] = field(default_factory=list)
    synthesis: str = ""
    judge_model: str = ""
    total_elapsed: float = 0.0


# ── Role configs ──────────────────────────────────────────────────────

ROLE_CONFIG = {
    "advocate": {
        "emoji": "\u2694\ufe0f",
        "label": "Advocate",
        "color": "green",
        "system": (
            "You are a debate advocate. Argue FOR the first option presented "
            "in the question. Be specific, give concrete reasons, real-world "
            "examples, and practical advantages. Keep it under 250 words."
        ),
    },
    "critic": {
        "emoji": "\U0001f6e1\ufe0f",
        "label": "Critic",
        "color": "red",
        "system": (
            "You are a debate critic. Argue AGAINST the first option presented "
            "in the question. Find weaknesses, risks, hidden costs, and better "
            "alternatives. Keep it under 250 words."
        ),
    },
    "analyst": {
        "emoji": "\U0001f50d",
        "label": "Analyst",
        "color": "blue",
        "system": (
            "You are a neutral analyst. Analyze the question objectively. "
            "Consider tradeoffs, context-dependent factors, and edge cases "
            "that both sides might miss. Keep it under 250 words."
        ),
    },
}

SYNTHESIS_SYSTEM = (
    "You are a debate judge. You have received three perspectives on a question: "
    "an Advocate (arguing for), a Critic (arguing against), and an Analyst "
    "(neutral analysis). Synthesize the strongest points from all three into a "
    "clear, actionable recommendation. Note where they agree and disagree. "
    "Keep it under 300 words."
)


# ── Default model selection ───────────────────────────────────────────

DEFAULT_MODELS = {
    "advocate": "kimi-k2.5:cloud",
    "critic": "deepseek-v3.2:cloud",
    "analyst": "chatgpt:gpt-5.4",
}

# Fallback when ChatGPT is not available (2 debaters only, saves a slot)
FALLBACK_MODELS = {
    "advocate": "kimi-k2.5:cloud",
    "critic": "deepseek-v3.2:cloud",
}


def _select_models(brain, user_models: Optional[str] = None) -> dict:
    """Pick debate models respecting the 3-slot Ollama limit.

    Rules:
    - Max 2 Ollama cloud models (leaves 1 slot free)
    - ChatGPT models don't count against Ollama slots
    - If user specified --models, parse and use those
    - If ChatGPT unavailable, drop to 2 debaters
    """
    if user_models:
        aliases = {
            "kimi": "kimi-k2.5:cloud",
            "qwen": "qwen3.5:397b-cloud",
            "deepseek": "deepseek-v3.2:cloud",
            "minimax": "minimax-m2.7:cloud",
            "glm": "glm-5:cloud",
            "nemotron": "nemotron-3-super:cloud",
            "chatgpt": "chatgpt:gpt-5.4",
            "chatgpt-codex": "chatgpt:gpt-5.3-codex",
        }
        parts = [p.strip() for p in user_models.split(",")]
        roles = ["advocate", "critic", "analyst"]
        models = {}
        for i, part in enumerate(parts[:3]):
            model = aliases.get(part, part)
            models[roles[i]] = model
        return models

    # Auto-select: check ChatGPT availability
    has_chatgpt = brain._chatgpt_client is not None
    if has_chatgpt:
        return dict(DEFAULT_MODELS)
    else:
        return dict(FALLBACK_MODELS)


# ── Display helpers ───────────────────────────────────────────────────

def _make_position_panel(pos: DebatePosition) -> Panel:
    """Build a Rich Panel for one debater's response."""
    cfg = ROLE_CONFIG[pos.role]
    title = f"{cfg['emoji']} {cfg['label']} ({pos.model})"

    if pos.error:
        content = f"[red]Error: {pos.error}[/red]"
    elif pos.done:
        elapsed_tag = f"[dim]{pos.elapsed:.1f}s[/dim]"
        content = f"{pos.argument}\n\n{elapsed_tag}"
    else:
        content = "[dim italic]Thinking...[/dim italic]"

    return Panel(
        content,
        title=f"[bold {cfg['color']}]{title}[/bold {cfg['color']}]",
        border_style=cfg["color"],
        expand=True,
        padding=(1, 2),
    )


def _make_synthesis_panel(synthesis: str, judge_model: str, elapsed: float) -> Panel:
    """Build the judge synthesis panel."""
    if not synthesis:
        content = "[dim italic]Synthesizing...[/dim italic]"
    else:
        content = f"{synthesis}\n\n[dim]Judge: {judge_model} \u00b7 {elapsed:.1f}s[/dim]"

    return Panel(
        content,
        title="[bold yellow]\u2696\ufe0f Synthesis[/bold yellow]",
        border_style="yellow",
        expand=True,
        padding=(1, 2),
    )


def _build_debate_display(result: DebateResult, phase: str = "debating") -> Panel:
    """Build the full debate display."""
    panels = [_make_position_panel(p) for p in result.positions]

    if phase in ("synthesizing", "done"):
        panels.append(_make_synthesis_panel(
            result.synthesis, result.judge_model, result.total_elapsed
        ))

    from rich.console import Group
    group = Group(*panels)

    border = "cyan" if phase != "done" else "green"
    status = {
        "debating": "[cyan]Models debating...[/cyan]",
        "synthesizing": "[yellow]Judge synthesizing...[/yellow]",
        "done": f"[green]Complete \u00b7 {result.total_elapsed:.1f}s total[/green]",
    }.get(phase, "")

    return Panel(
        group,
        title=f"[bold cyan]\U0001f3db\ufe0f  Debate: {result.question[:60]}[/bold cyan]",
        subtitle=status,
        border_style=border,
        padding=(0, 1),
    )


# ── Core debate runner ────────────────────────────────────────────────

def run_debate(brain, question: str, user_models: Optional[str] = None) -> DebateResult:
    """Run a multi-model debate and display results live.

    Args:
        brain: OllamaBrain instance
        question: The question to debate
        user_models: Optional comma-separated model list
    """
    console = Console()
    models = _select_models(brain, user_models)

    result = DebateResult(question=question)
    positions = []
    for role, model in models.items():
        pos = DebatePosition(model=model, role=role)
        positions.append(pos)
    result.positions = positions

    lock = threading.Lock()

    # Shared timeout pool for all debaters (avoids per-debater executor leak)
    timeout_pool = ThreadPoolExecutor(max_workers=len(positions))

    def _run_debater(pos: DebatePosition):
        cfg = ROLE_CONFIG[pos.role]
        prompt = f"Question: {question}"
        start = time.time()
        try:
            fut = timeout_pool.submit(
                brain.think,
                prompt,
                system_prompt=cfg["system"],
                use_history=False,
                model_override=pos.model,
            )
            try:
                response = fut.result(timeout=DEBATER_TIMEOUT)
            except FuturesTimeoutError:
                with lock:
                    pos.error = f"Timed out after {DEBATER_TIMEOUT}s"
                    pos.elapsed = time.time() - start
                    pos.done = True
                return
            # think() can return str or dict
            if isinstance(response, dict):
                response = response.get("response", response.get("content", str(response)))
            with lock:
                pos.argument = response or "(no response)"
                pos.elapsed = time.time() - start
                pos.done = True
        except Exception as e:
            with lock:
                pos.error = str(e)[:200]
                pos.elapsed = time.time() - start
                pos.done = True

    # Phase 1: Run all debaters in parallel
    console.print()
    with Live(
        _build_debate_display(result, "debating"),
        console=console,
        refresh_per_second=2,
        transient=False,
    ) as live:
        with ThreadPoolExecutor(max_workers=3) as pool:
            {pool.submit(_run_debater, p): p for p in positions}
            while not all(p.done for p in positions):
                time.sleep(0.5)
                with lock:
                    live.update(_build_debate_display(result, "debating"))
            # Final update with all positions done
            live.update(_build_debate_display(result, "synthesizing"))
        timeout_pool.shutdown(wait=False)

        # Phase 2: Judge synthesis
        from aura.config import Config
        judge_model = Config.MODEL_THINK

        arguments_text = ""
        for pos in positions:
            cfg = ROLE_CONFIG[pos.role]
            if pos.error:
                arguments_text += f"\n{cfg['label']} ({pos.model}): [ERROR: {pos.error}]\n"
            else:
                arguments_text += f"\n{cfg['label']} ({pos.model}):\n{pos.argument}\n"

        synthesis_prompt = (
            f"Question: {question}\n\n"
            f"Three perspectives:\n{arguments_text}\n\n"
            "Synthesize the best answer. Who made the strongest points? "
            "What is the recommended path forward?"
        )

        synth_start = time.time()
        try:
            synth_response = brain.think(
                synthesis_prompt,
                system_prompt=SYNTHESIS_SYSTEM,
                use_history=False,
                model_override=judge_model,
            )
            if isinstance(synth_response, dict):
                synth_response = synth_response.get(
                    "response", synth_response.get("content", str(synth_response))
                )
            result.synthesis = synth_response or "(no synthesis)"
        except Exception as e:
            result.synthesis = f"Synthesis failed: {e}"

        result.judge_model = judge_model
        result.total_elapsed = time.time() - synth_start + max(
            (p.elapsed for p in positions), default=0
        )

        live.update(_build_debate_display(result, "done"))

    console.print()
    return result


def parse_debate_args(arg: str) -> tuple:
    """Parse /debate command arguments.

    Supports:
        /debate should I use SQLite or Postgres?
        /debate --models kimi,deepseek,chatgpt should I use SQLite or Postgres?

    Returns:
        (question, user_models_or_None)
    """
    arg = arg.strip()
    if arg.startswith("--models"):
        # Split: --models kimi,deepseek,chatgpt <question>
        parts = arg.split(maxsplit=2)
        if len(parts) >= 3:
            return parts[2], parts[1]
        elif len(parts) == 2:
            return "", parts[1]
        return "", None
    return arg, None
