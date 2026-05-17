"""
/bench — multi-model benchmark. Run any prompt against several models
and get a comparison table with speed, tokens, cost, and response preview.
"""
from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class BenchResult:
    model: str
    response: str = ""
    elapsed: float = 0.0
    tokens: int = 0
    tok_per_sec: float = 0.0
    cost: float = 0.0
    error: str = ""


# Default model sets for /bench
BENCH_TIERS: dict[str, list[str]] = {
    "fast": [
        "qwen3.5:8b", "deepseek-r1:14b", "llama3.2:3b",
        "gemma3:12b", "phi4:14b",
    ],
    "balanced": [
        "qwen3.5:397b-cloud", "kimi-k2.6:cloud", "deepseek-v3.2:cloud",
        "nemotron-3-super:cloud", "glm-5:cloud",
    ],
    "reasoning": [
        "kimi-k2.6:cloud", "deepseek-v3.2:cloud",
        "chatgpt:gpt-5.4", "qwen3.5:397b-cloud",
    ],
    "ultra": [
        "kimi-k2.6:cloud", "deepseek-v3.2:cloud",
        "chatgpt:gpt-5.4", "nemotron-3-super:cloud",
        "qwen3.5:397b-cloud", "glm-5:cloud",
        "minimax-m2.7:cloud",
    ],
}


def handle_bench(agent: Any, arg: str, context: dict) -> Optional[str]:
    from rich.console import Group
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    from ..display import console

    if not arg:
        console.print()
        console.print("  [bold]/bench[/bold] — run a prompt against multiple models and compare results.")
        console.print()
        console.print("  [dim]Usage:[/dim]")
        console.print("    /bench <prompt>                    — use default 'balanced' tier")
        console.print("    /bench fast <prompt>               — use local/fast models")
        console.print("    /bench balanced <prompt>           — use balanced cloud models")
        console.print("    /bench reasoning <prompt>          — use best reasoning models")
        console.print("    /bench ultra <prompt>              — max coverage, all cloud models")
        console.print("    /bench model1,model2 <prompt>      — specify models manually")
        console.print()
        console.print(f"  [dim]Available tiers: {', '.join(BENCH_TIERS.keys())}[/dim]")
        console.print()
        return None

    # Parse arguments
    tier, prompt, custom_models = _parse_bench_args(arg)

    if not prompt:
        console.print("  [red]No prompt provided.[/red]")
        return None

    models = custom_models if custom_models else BENCH_TIERS.get(tier, BENCH_TIERS["balanced"])

    console.print()
    console.print(f"  [bold cyan]Benchmarking[/bold cyan] [dim]\\u00b7[/dim] {len(models)} models [dim]\\u00b7[/dim] tier: {tier}")
    console.print(f"  [dim]Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}[/dim]")
    console.print()

    # Run all models in parallel
    results = _run_benchmark(agent, prompt, models)

    # Sort by speed
    results.sort(key=lambda r: r.elapsed)

    # Build table
    table = Table(box=None, padding=(0, 1), show_header=True, header_style="bold")
    table.add_column("#", width=3, justify="right", style="dim")
    table.add_column("Model", style="bold", width=28)
    table.add_column("Time", width=7, justify="right")
    table.add_column("Tokens", width=7, justify="right")
    table.add_column("T/s", width=7, justify="right")
    table.add_column("Cost", width=8, justify="right")
    table.add_column("Preview", min_width=40, style="dim")

    for i, r in enumerate(results, 1):
        if r.error:
            preview = f"[red]ERROR: {r.error[:50]}[/red]"
            time_str = "[red]FAIL[/red]"
        else:
            preview = r.response[:80].replace("\n", " ") + ("..." if len(r.response) > 80 else "")
            time_str = f"{r.elapsed:.1f}s"

        tok_str = f"{r.tokens}" if r.tokens > 0 else "-"
        tps_str = f"{r.tok_per_sec:.0f}" if r.tok_per_sec > 0 else "-"
        cost_str = f"${r.cost:.4f}" if r.cost > 0.0001 else "-"

        # Highlight fastest row
        if i == 1 and not r.error:
            time_str = f"[green]{time_str}[/green]"

        table.add_row(str(i), r.model, time_str, tok_str, tps_str, cost_str, preview)

    # Summary stats
    successful = [r for r in results if not r.error]
    if successful:
        fastest = successful[0]
        cheapest = min(successful, key=lambda r: r.cost)
        most_tokens = max(successful, key=lambda r: r.tokens)
        summary = Text()
        summary.append("Fastest: ", style="dim")
        summary.append(f"{fastest.model}", style="bold green")
        summary.append(f" ({fastest.elapsed:.1f}s)", style="dim")
        if cheapest.cost > 0.0001:
            summary.append("  ·  ", style="dim")
            summary.append("Cheapest: ", style="dim")
            summary.append(f"{cheapest.model}", style="bold cyan")
            summary.append(f" (${cheapest.cost:.4f})", style="dim")
        summary.append("  ·  ", style="dim")
        summary.append("Most verbose: ", style="dim")
        summary.append(f"{most_tokens.model}", style="bold yellow")
        summary.append(f" ({most_tokens.tokens} tok)", style="dim")
    else:
        summary = Text("All models failed.", style="red")

    failed = [r for r in results if r.error]
    fail_note = ""
    if failed:
        fail_note = f"  [red]{len(failed)} model(s) failed[/red]"

    console.print(Panel(
        Group(table, Text(), summary),
        title="[bold cyan]⚡ /bench results[/bold cyan]",
        subtitle=fail_note or None,
        border_style="cyan",
        padding=(1, 2),
    ))
    console.print()

    return None


def _parse_bench_args(arg: str) -> tuple[str, str, Optional[list[str]]]:
    """Parse /bench arguments. Returns (tier, prompt, custom_models_or_none)."""
    parts = arg.strip().split(maxsplit=1)
    first = parts[0].lower()
    rest = parts[1] if len(parts) > 1 else ""

    # Check if first word is a tier name
    if first in BENCH_TIERS:
        return first, rest, None

    # Check if first word is a comma-separated model list
    if "," in first:
        models = [m.strip() for m in first.split(",") if m.strip()]
        return "custom", rest, models

    # No tier/model prefix — whole arg is the prompt
    return "balanced", arg, None


def _run_benchmark(agent: Any, prompt: str, models: list[str]) -> list[BenchResult]:
    """Run prompt against all models in parallel, collect results."""
    results: list[BenchResult] = []

    def _query_model(model: str) -> BenchResult:
        start = time.time()
        try:
            # Use the agent's brain to query with model override
            response = agent.brain.think(
                prompt,
                use_history=False,
                model_override=model,
            )
            elapsed = time.time() - start

            if isinstance(response, dict):
                text = response.get("response", response.get("content", str(response)))
                tokens = response.get("tokens", response.get("token_count", 0))
            else:
                text = str(response)
                tokens = 0

            # Estimate tokens from text length if not provided
            if tokens == 0 and text:
                tokens = int(len(text) / 3.5)

            tok_per_sec = tokens / elapsed if elapsed > 0 else 0.0

            # Estimate cost (rough, provider-agnostic)
            cost = _estimate_cost(model, tokens)

            return BenchResult(
                model=model,
                response=text,
                elapsed=elapsed,
                tokens=tokens,
                tok_per_sec=tok_per_sec,
                cost=cost,
            )
        except Exception as e:
            elapsed = time.time() - start
            return BenchResult(model=model, elapsed=elapsed, error=str(e)[:100])

    # Parallel execution with thread pool
    with ThreadPoolExecutor(max_workers=min(len(models), 6)) as pool:
        futures = {pool.submit(_query_model, m): m for m in models}
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                model = futures[future]
                results.append(BenchResult(model=model, error=str(e)[:100]))

    return results


def _estimate_cost(model: str, tokens: int) -> float:
    """Rough cost estimate per model. Prices are approximate per 1M output tokens."""
    PRICES: dict[str, float] = {
        "kimi": 0.40, "deepseek": 0.28, "qwen": 0.35,
        "nemotron": 0.30, "glm": 0.25, "minimax": 0.20,
        "chatgpt": 15.00,
    }
    for key, price in PRICES.items():
        if key in model.lower():
            return (tokens / 1_000_000) * price
    return (tokens / 1_000_000) * 0.50  # default estimate
