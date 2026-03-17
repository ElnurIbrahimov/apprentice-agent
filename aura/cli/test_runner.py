# aura/cli/test_runner.py
"""Test runner integration — /test with formatted output and auto-fix loop."""
from __future__ import annotations
import subprocess
import re
import time
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from rich.console import Console
from rich.panel import Panel
from rich.text import Text


@dataclass
class TestResult:
    """Parsed test execution result."""
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    errors: int = 0
    duration: float = 0.0
    output: str = ""
    failures: List[Dict] = field(default_factory=list)

    @property
    def total(self) -> int:
        return self.passed + self.failed + self.skipped + self.errors

    @property
    def success(self) -> bool:
        return self.failed == 0 and self.errors == 0

    @property
    def summary(self) -> str:
        parts = []
        if self.passed: parts.append(f"[green]{self.passed} passed[/green]")
        if self.failed: parts.append(f"[red]{self.failed} failed[/red]")
        if self.skipped: parts.append(f"[yellow]{self.skipped} skipped[/yellow]")
        if self.errors: parts.append(f"[red]{self.errors} errors[/red]")
        return ", ".join(parts) if parts else "[dim]no tests[/dim]"


def run_tests(test_cmd: str, cwd: str = ".", timeout: int = 300) -> TestResult:
    """Run tests and return parsed results."""
    import shlex
    import os
    start = time.time()
    try:
        if os.name == "nt":
            cmd_args = shlex.split(test_cmd, posix=False)
        else:
            cmd_args = shlex.split(test_cmd)
        proc = subprocess.run(
            cmd_args,
            capture_output=True, text=True, timeout=timeout,
            cwd=cwd, shell=False,
        )
        elapsed = time.time() - start
        result = TestResult(output=proc.stdout + proc.stderr, duration=elapsed)
        _parse_output(result, proc.stdout + proc.stderr)
        return result
    except subprocess.TimeoutExpired:
        return TestResult(output=f"Tests timed out after {timeout}s", errors=1)
    except FileNotFoundError:
        return TestResult(output=f"Test command not found: {test_cmd}", errors=1)
    except Exception as e:
        return TestResult(output=f"Test error: {e}", errors=1)


def _parse_output(result: TestResult, output: str) -> None:
    """Parse test output (supports pytest, jest, go test, cargo test patterns)."""
    # pytest: "5 passed, 2 failed, 1 skipped in 3.2s"
    m = re.search(r'(\d+) passed', output)
    if m: result.passed = int(m.group(1))
    m = re.search(r'(\d+) failed', output)
    if m: result.failed = int(m.group(1))
    m = re.search(r'(\d+) skipped', output)
    if m: result.skipped = int(m.group(1))
    m = re.search(r'(\d+) error', output)
    if m: result.errors = int(m.group(1))

    # jest: "Tests: 2 failed, 5 passed, 7 total"
    m = re.search(r'Tests:\s*(\d+)\s+failed.*?(\d+)\s+passed', output)
    if m and result.passed == 0:
        result.failed = int(m.group(1))
        result.passed = int(m.group(2))

    # Extract failure details
    failure_blocks = re.findall(r'FAILED\s+(.+?)(?:\n|$)', output)
    for fb in failure_blocks[:10]:
        result.failures.append({"test": fb.strip()})


def render_test_results(console: Console, result: TestResult) -> None:
    """Render test results with color coding."""
    icon = "[green]\u2713[/green]" if result.success else "[red]\u2717[/red]"
    header = f"{icon} Tests: {result.summary} ({result.duration:.1f}s)"

    text = Text()
    for line in result.output.splitlines()[-20:]:  # last 20 lines
        if "PASSED" in line or "passed" in line or "\u2713" in line:
            text.append(line + "\n", style="green")
        elif "FAILED" in line or "failed" in line or "\u2717" in line or "ERROR" in line:
            text.append(line + "\n", style="red")
        elif "skip" in line.lower() or "SKIP" in line:
            text.append(line + "\n", style="yellow")
        else:
            text.append(line + "\n", style="dim")

    border = "green" if result.success else "red"
    console.print(Panel(text, title=f"[bold]{header}[/bold]", border_style=border, padding=(0, 1)))

    if result.failures:
        console.print(f"\n[red]Failed tests:[/red]")
        for f in result.failures[:5]:
            console.print(f"  [red]\u2717[/red] {f['test']}")


@dataclass
class TestHistory:
    """Track test results over a session."""
    runs: List[TestResult] = field(default_factory=list)

    def add(self, result: TestResult) -> None:
        self.runs.append(result)
        if len(self.runs) > 50:
            self.runs = self.runs[-50:]

    @property
    def pass_rate(self) -> float:
        if not self.runs:
            return 0.0
        successes = sum(1 for r in self.runs if r.success)
        return successes / len(self.runs)

    @property
    def total_runs(self) -> int:
        return len(self.runs)

    def summary(self) -> str:
        if not self.runs:
            return "No test runs yet"
        rate = self.pass_rate * 100
        return f"{self.total_runs} runs, {rate:.0f}% pass rate"
