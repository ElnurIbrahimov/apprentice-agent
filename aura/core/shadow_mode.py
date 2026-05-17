"""Shadow mode — run 2 models in parallel, surface disagreement.

When the dispatcher returns low confidence OR the user explicitly requests it,
call the primary model and its best alternative concurrently. Ollama Pro's
2-slot sweet spot is exactly right for this. After both return, compare the
responses; if they meaningfully differ, render a Rich side-by-side panel and
let the user pick. If they agree (or one errors), return the primary silently.

This sits *above* brain.think() — it calls think() twice with model_override
pointing at each model.
"""
from __future__ import annotations

import difflib
import logging
from concurrent.futures import as_completed
from dataclasses import dataclass
from typing import Optional

from aura.pools import llm_pool

logger = logging.getLogger(__name__)


# Minimum similarity (0-1) below which we consider responses "disagreeing"
# and worth showing both to the user. Higher = stricter "same answer".
AGREEMENT_THRESHOLD = 0.72


@dataclass
class ShadowResult:
    primary_model: str
    primary_response: str
    shadow_model: str
    shadow_response: str
    agreement: float            # 0-1 text similarity
    primary_tokens: int = 0
    shadow_tokens: int = 0
    error: Optional[str] = None

    @property
    def disagrees(self) -> bool:
        return self.agreement < AGREEMENT_THRESHOLD

    def to_dict(self) -> dict:
        return {
            "primary_model": self.primary_model,
            "shadow_model": self.shadow_model,
            "agreement": round(self.agreement, 3),
            "disagrees": self.disagrees,
            "primary_tokens": self.primary_tokens,
            "shadow_tokens": self.shadow_tokens,
            "error": self.error,
        }


def _similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0 if (a or b) else 1.0
    # Use SequenceMatcher — cheap, roughly tracks "do they say the same thing"
    # for short-to-medium responses. For longer outputs we trim to first 2k chars
    # to bound cost.
    return difflib.SequenceMatcher(None, a[:2000], b[:2000]).ratio()


def run_shadow(
    brain,
    prompt: str,
    primary_model: str,
    shadow_model: str,
    *,
    system_prompt: Optional[str] = None,
    use_history: bool = False,
) -> ShadowResult:
    """Run two models on the same prompt in parallel via llm_pool.

    Uses `brain.think()` with `model_override` on each. Returns after both
    complete (or one errors — the other's response is still returned).
    """
    if primary_model == shadow_model:
        raise ValueError("shadow model must differ from primary")

    pool = llm_pool()

    def _call(model: str) -> tuple[str, str, int]:
        try:
            resp = brain.think(
                prompt,
                system_prompt=system_prompt,
                use_history=use_history,
                model_override=model,
            )
            return model, resp or "", len((resp or "").split())
        except Exception as e:
            return model, f"[error: {e}]", 0

    f_primary = pool.submit(_call, primary_model)
    f_shadow = pool.submit(_call, shadow_model)

    primary_result = None
    shadow_result = None
    error: Optional[str] = None

    for fut in as_completed([f_primary, f_shadow], timeout=180):
        try:
            model, resp, n = fut.result()
            if model == primary_model:
                primary_result = (resp, n)
            else:
                shadow_result = (resp, n)
        except Exception as e:
            error = str(e)

    if primary_result is None:
        primary_result = ("", 0)
    if shadow_result is None:
        shadow_result = ("", 0)

    agreement = _similarity(primary_result[0], shadow_result[0])

    return ShadowResult(
        primary_model=primary_model,
        primary_response=primary_result[0],
        shadow_model=shadow_model,
        shadow_response=shadow_result[0],
        agreement=agreement,
        primary_tokens=primary_result[1],
        shadow_tokens=shadow_result[1],
        error=error,
    )


def render_shadow_result(result: ShadowResult) -> None:
    """Pretty-print a ShadowResult. Called by /shadow and auto-triggered paths."""
    try:
        from aura.cli.display import console
    except ImportError:
        from rich.console import Console
        console = Console()

    from rich.columns import Columns
    from rich.panel import Panel

    agreement_color = "green" if result.agreement >= AGREEMENT_THRESHOLD else "yellow"
    console.print()
    console.print(
        f"[bold]Shadow mode[/]  agreement=[{agreement_color}]{result.agreement:.2f}[/]  "
        f"({result.primary_model} vs {result.shadow_model})"
    )
    if result.error:
        console.print(f"  [red]partial error:[/] {result.error}")

    panels = [
        Panel(
            result.primary_response or "[dim](empty)[/]",
            title=f"[cyan]{result.primary_model}[/] (primary)",
            border_style="cyan",
        ),
        Panel(
            result.shadow_response or "[dim](empty)[/]",
            title=f"[magenta]{result.shadow_model}[/] (shadow)",
            border_style="magenta",
        ),
    ]
    console.print(Columns(panels, equal=True, expand=True))

    if result.disagrees:
        console.print()
        console.print("[yellow]Models disagree meaningfully. Pick which to use:[/]")
        console.print("  [cyan]p[/]rimary   [magenta]s[/]hadow   [dim]b[/]oth   [dim]n[/]either")


__all__ = [
    "AGREEMENT_THRESHOLD",
    "ShadowResult",
    "render_shadow_result",
    "run_shadow",
]
