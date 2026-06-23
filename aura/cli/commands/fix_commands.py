"""
/fix — intelligent auto-fixer. Run tests → identify failures → generate fixes → verify.
Continuously loops until tests pass or max attempts exhausted.
"""
from __future__ import annotations

import logging
import os
import subprocess
import time
from dataclasses import dataclass
from typing import Any, Optional

from .common import TIER_BETA, command

logger = logging.getLogger(__name__)

MAX_FIX_ATTEMPTS = 5
MAX_FIX_TIME = 300  # 5 minutes total

@dataclass
class FixResult:
    attempts: int = 0
    fixed: bool = False
    total_elapsed: float = 0.0
    test_output: str = ""
    error: str = ""


@command("/fix",      "Auto-fix failing tests (run → fix → verify loop)",     tier=TIER_BETA)
def handle_fix(agent: Any, arg: str, context: dict) -> Optional[str]:

    from ..context import get_ctx
    from ..display import console

    ctx = get_ctx()
    if not ctx or not ctx.agentic_loop:
        console.print("  [dim]No active session.[/dim]")
        return None

    arg = (arg or "").strip()
    test_cmd = arg if arg else _get_configured_test_cmd()

    if not test_cmd:
        console.print("  [dim]No test command. Usage: /fix <test-command>[/dim]")
        console.print("  [dim]Example: /fix pytest -q  or  /fix npm test[/dim]")
        return None

    console.print()
    console.print("  [bold cyan]🔧 Auto-fix Mode[/bold cyan]")
    console.print(f"  [dim]Test: {test_cmd}[/dim]")
    console.print()

    result = FixResult()
    overall_start = time.time()

    for attempt in range(1, MAX_FIX_ATTEMPTS + 1):
        if time.time() - overall_start > MAX_FIX_TIME:
            console.print(f"  [yellow]Time limit ({MAX_FIX_TIME}s) reached. Stopping.[/yellow]")
            break

        console.print(f"  [bold]Attempt {attempt}/{MAX_FIX_ATTEMPTS}[/bold]")

        # Step 1: Run tests
        test_result = _run_tests(test_cmd)
        if test_result.success:
            result.fixed = True
            result.attempts = attempt
            result.test_output = test_result.output
            result.total_elapsed = time.time() - overall_start
            console.print("  [green]✓ All tests pass![/green]")
            break

        # Step 2: Show failures
        failures = test_result.failure_summary
        if not failures:
            console.print("  [yellow]Tests failed but couldn't extract failure details.[/yellow]")
            if attempt == MAX_FIX_ATTEMPTS:
                result.test_output = test_result.output
            break

        console.print(f"  [yellow]{len(failures.splitlines())} failure(s) detected[/yellow]")

        # Step 3: Ask agent to fix
        fix_prompt = _build_fix_prompt(failures, attempt)
        console.print("  [dim]Generating fix...[/dim]")

        try:
            loop = ctx.agentic_loop
            fix_result = loop.run(
                fix_prompt,
                steering_queue=ctx.steering,
            )
            if fix_result and fix_result.get("success"):
                console.print("  [dim]Fix applied. Re-running tests...[/dim]")
            else:
                console.print("  [red]Fix generation failed.[/red]")
                if attempt == MAX_FIX_ATTEMPTS:
                    result.test_output = test_result.output
                break
        except Exception as e:
            console.print(f"  [red]Fix attempt crashed: {e}[/red]")
            if attempt == MAX_FIX_ATTEMPTS:
                result.test_output = test_result.output
                result.error = str(e)
            break

        result.attempts = attempt

    # Final summary
    result.total_elapsed = time.time() - overall_start

    _render_fix_result(console, result, test_cmd)

    return None


def _get_configured_test_cmd() -> str:
    """Get the configured test command from AURA.md or environment."""
    try:
        from aura.core.context import get_aura_md_config
        cfg = get_aura_md_config(os.getcwd()) or {}
        return cfg.get("test_cmd", "")
    except Exception:
        pass
    return os.environ.get("AURA_TEST_CMD", "")


@dataclass
class TestRunResult:
    success: bool = False
    output: str = ""
    failure_summary: str = ""


def _run_tests(test_cmd: str) -> TestRunResult:
    """Run tests and extract failure information."""
    import shlex
    try:
        if os.name == "nt":
            cmd_args = shlex.split(test_cmd, posix=False)
        else:
            cmd_args = shlex.split(test_cmd)
        result = subprocess.run(
            cmd_args, capture_output=True, text=True,
            timeout=120, cwd=os.getcwd(),
        )
        output = result.stdout[-5000:] or result.stderr[-5000:]
        success = result.returncode == 0

        if success:
            return TestRunResult(success=True, output=output)

        # Extract failure summary from pytest-style output
        failures = _extract_pytest_failures(output)
        if not failures:
            failures = _extract_generic_failures(output)

        return TestRunResult(success=False, output=output, failure_summary=failures)
    except subprocess.TimeoutExpired:
        return TestRunResult(
            success=False,
            output="Test timed out after 120s",
            failure_summary="Test timeout",
        )
    except Exception as e:
        return TestRunResult(
            success=False,
            output=str(e),
            failure_summary=f"Test runner error: {e}",
        )


def _extract_pytest_failures(output: str) -> str:
    """Extract FAILED lines and summary from pytest output."""
    lines = output.split("\n")
    failures: list[str] = []

    # Find the failure section
    in_failures = False
    for line in lines:
        if "FAILURES" in line and "==" in line:
            in_failures = True
            continue
        if in_failures:
            if "short test summary" in line.lower():
                break
            failures.append(line)

    if failures:
        return "\n".join(failures[-100:])

    # Fallback: just get FAILED lines
    failed_lines = [l for l in lines if "FAILED" in l and "::" in l]
    if failed_lines:
        return "\n".join(failed_lines[:20])

    return ""


def _extract_generic_failures(output: str) -> str:
    """Extract error lines from non-pytest output."""
    lines = output.split("\n")
    error_lines = [
        l for l in lines
        if any(kw in l.lower() for kw in ("error", "fail", "assert", "traceback", "exception"))
    ]
    if error_lines:
        return "\n".join(error_lines[:30])
    # Just return last 40 lines
    return "\n".join(lines[-40:])


def _build_fix_prompt(failures: str, attempt: int) -> str:
    """Build the prompt for the agent to fix test failures."""
    urgency = ""
    if attempt == 1:
        urgency = "This is the first attempt. Analyze the failures carefully and fix the root cause, not just symptoms."
    elif attempt >= 3:
        urgency = f"This is attempt {attempt}. Previous attempts didn't fix it. Think differently — the root cause might be deeper. Consider if the tests themselves are wrong, or if there's a systemic issue."

    return f"""The following tests are failing. Fix the code so all tests pass.

{failures[:3000]}

{urgency}

Important:
- Fix the SOURCE code, not the tests (unless tests are clearly wrong)
- Think about WHY the tests fail, not just WHAT fails
- Run the tests after making changes to verify
- If a previous fix was applied and tests still fail, try a DIFFERENT approach
- Don't add unnecessary changes — minimum fix, maximum impact"""


def _render_fix_result(console: Any, result: FixResult, test_cmd: str) -> None:
    """Render the fix result panel."""
    from rich.panel import Panel
    from rich.text import Text

    body = Text()

    if result.fixed:
        body.append(f"  ✓ Tests pass after {result.attempts} attempt(s)", style="bold green")
        body.append(f"\n  Time: {result.total_elapsed:.1f}s", style="dim")
    else:
        body.append(f"  ✗ Could not fix after {result.attempts} attempt(s)", style="bold red")
        body.append(f"\n  Time: {result.total_elapsed:.1f}s", style="dim")
        if result.error:
            body.append(f"\n  Error: {result.error}", style="red")

        if result.test_output:
            body.append("\n\n  [bold]Last test output:[/bold]")
            # Show last 15 lines
            lines = result.test_output.split("\n")
            for line in lines[-15:]:
                body.append(f"\n  [dim]{line[:120]}[/dim]")

        body.append(f"\n\n  [dim]Try: /fix {test_cmd}  or fix manually and run /verify[/dim]")

    console.print()
    console.print(Panel(
        body,
        title="[bold cyan]🔧 /fix results[/bold cyan]",
        border_style="green" if result.fixed else "red",
        padding=(1, 2),
    ))
    console.print()
