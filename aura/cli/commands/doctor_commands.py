"""
/doctor — comprehensive system diagnostic for Aura CLI.
Checks every dependency and subsystem in one shot.
"""
from __future__ import annotations

import logging
import os
import subprocess
import sys
from typing import Any, Optional

logger = logging.getLogger(__name__)


def handle_doctor(agent: Any, arg: str, context: dict) -> Optional[str]:
    """Run a full system diagnostic and display results as a Rich panel."""
    from rich.console import Group
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    from ..context import get_ctx
    from ..display import console

    ctx = get_ctx()
    results: list[tuple[str, str, str, str]] = []  # (check, status, detail, hint)

    # ── Python Environment ──
    _python_check(results)

    # ── Ollama ──
    _ollama_check(results)

    # ── Cloud Providers ──
    _cloud_providers_check(results)

    # ── Git ──
    _git_check(results)

    # ── Disk ──
    _disk_check(results)

    # ── Session State ──
    if ctx and ctx.agentic_loop:
        _session_check(results, ctx)

    # ── Tools ──
    if ctx and ctx.agentic_loop:
        _tools_check(results, ctx)

    # ── Memory ──
    _memory_check(results)

    # ── Hooks ──
    if ctx and ctx.hook_manager:
        _hooks_check(results, ctx)

    # ── Activity Log ──
    if ctx:
        _activity_log_check(results, ctx)

    # ── Build display ──
    passed = sum(1 for _, s, _, _ in results if s == "PASS")
    warned = sum(1 for _, s, _, _ in results if s == "WARN")
    failed = sum(1 for _, s, _, _ in results if s == "FAIL")
    total = len(results)

    status_text = f"{passed}P/{warned}W/{failed}F ({total} checks)"

    table = Table(box=None, padding=(0, 1), show_header=True, header_style="bold")
    table.add_column("Check", style="bold", width=18)
    table.add_column("Status", width=6, justify="center")
    table.add_column("Detail", min_width=40)
    table.add_column("Hint", style="dim", min_width=20)

    STATUS_STYLES = {
        "PASS": ("✓", "green"),
        "WARN": ("△", "yellow"),
        "FAIL": ("✗", "red"),
        "SKIP": ("−", "dim"),  # noqa: RUF001 — intentional minus sign icon
    }

    for check, status, detail, hint in results:
        icon, color = STATUS_STYLES.get(status, ("?", "dim"))
        table.add_row(check, f"[{color}]{icon}[/{color}]", detail, hint or "")

    console.print()
    console.print(Panel(
        Group(
            Text(f"Aura System Diagnostic  ·  {status_text}", style="bold"),
            Text(),
            table,
        ),
        title="[bold cyan]🔬 /doctor[/bold cyan]",
        border_style="cyan",
        padding=(1, 2),
    ))
    console.print()

    return None


# ── Check helpers ──────────────────────────────────────────────────────────


def _python_check(results: list) -> None:
    detail = f"Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    hint = sys.executable
    results.append(("Python", "PASS", detail, hint))


def _ollama_check(results: list) -> None:
    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    try:
        import urllib.request
        req = urllib.request.Request(host, method="HEAD")
        urllib.request.urlopen(req, timeout=3)
        results.append(("Ollama (local)", "PASS", "Responding", host))
    except Exception as e:
        results.append(("Ollama (local)", "FAIL", "Unreachable", f"ollama serve  ({str(e)[:40]})"))

    cloud_key = os.environ.get("OLLAMA_API_KEY")
    if cloud_key:
        results.append(("Ollama Cloud", "PASS", "OLLAMA_API_KEY set", f"key: ...{cloud_key[-6:]}" if len(cloud_key) > 6 else "key set"))
    else:
        results.append(("Ollama Cloud", "WARN", "OLLAMA_API_KEY not set", "Set in .env or AURA.md"))


def _cloud_providers_check(results: list) -> None:
    try:
        import urllib.request
        req = urllib.request.Request("https://api.openai.com/v1/models", method="HEAD",
                                     headers={"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', '')}"})
        urllib.request.urlopen(req, timeout=5)
        results.append(("OpenAI", "PASS", "API reachable", ""))
    except Exception:
        key = os.environ.get("OPENAI_API_KEY")
        if key:
            results.append(("OpenAI", "WARN", "API key set but unreachable", "Check network or billing"))
        else:
            results.append(("OpenAI", "SKIP", "No OPENAI_API_KEY", "Set to enable"))

    try:
        import urllib.request
        req = urllib.request.Request("https://generativelanguage.googleapis.com/v1/models", method="HEAD")
        urllib.request.urlopen(req, timeout=5)
        results.append(("Gemini", "PASS", "API reachable", ""))
    except Exception:
        key = os.environ.get("GEMINI_API_KEY")
        if key:
            results.append(("Gemini", "WARN", "API key set but unreachable", ""))
        else:
            results.append(("Gemini", "SKIP", "No GEMINI_API_KEY", ""))


def _git_check(results: list) -> None:
    try:
        r = subprocess.run(["git", "--version"], capture_output=True, text=True, timeout=3)
        if r.returncode == 0:
            version = r.stdout.strip()
            results.append(("Git", "PASS", version, ""))
        else:
            results.append(("Git", "WARN", "git not found", "Install git"))
    except FileNotFoundError:
        results.append(("Git", "WARN", "git not installed", "Install git"))
    except Exception:
        results.append(("Git", "WARN", "git check failed", ""))

    try:
        r = subprocess.run(["git", "rev-parse", "--is-inside-work-tree"],
                           capture_output=True, text=True, timeout=3, cwd=os.getcwd())
        if r.returncode == 0 and r.stdout.strip() == "true":
            branch = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True, text=True, timeout=3, cwd=os.getcwd(),
            ).stdout.strip()
            status = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True, text=True, timeout=5, cwd=os.getcwd(),
            ).stdout
            dirty = len(status.splitlines())
            dirty_str = "" if dirty == 0 else f" ({dirty} dirty files)"
            results.append(("Git Repo", "PASS", f"Branch: {branch}{dirty_str}", os.getcwd()))
        else:
            results.append(("Git Repo", "SKIP", "Not in a git repo", ""))
    except Exception:
        results.append(("Git Repo", "SKIP", "Could not check", ""))


def _disk_check(results: list) -> None:
    try:
        import shutil
        usage = shutil.disk_usage(os.getcwd())
        free_gb = usage.free / (1024 ** 3)
        total_gb = usage.total / (1024 ** 3)
        pct = (1 - usage.free / usage.total) * 100
        if free_gb < 1:
            status, _color = "FAIL", "red"
        elif free_gb < 10:
            status, _color = "WARN", "yellow"
        else:
            status, _color = "PASS", "green"
        results.append(("Disk", status, f"{free_gb:.1f} GB free / {total_gb:.0f} GB ({pct:.0f}% used)", os.getcwd()))
    except Exception:
        results.append(("Disk", "SKIP", "Could not check disk usage", ""))


def _session_check(results: list, ctx: Any) -> None:
    loop = ctx.agentic_loop
    hist_len = len(getattr(loop, "_conversation_history", []) or [])
    session_id = ""
    if ctx.session:
        session_id = getattr(ctx.session, "session_id", "") or ""
    detail = f"{hist_len} messages in session"
    hint = session_id[:16] if session_id else "no session"
    results.append(("Session", "PASS", detail, hint))


def _tools_check(results: list, ctx: Any) -> None:
    try:
        tool_count = len(ctx.agent.tools)
        if tool_count > 0:
            tool_names = ", ".join(sorted(ctx.agent.tools.keys())[:8])
            results.append(("Tools", "PASS", f"{tool_count} loaded", tool_names))
        else:
            results.append(("Tools", "WARN", "No tools loaded", "Check install"))
    except Exception:
        results.append(("Tools", "SKIP", "Could not enumerate", ""))


def _memory_check(results: list) -> None:
    try:
        from aura.core.agentic_loop_support import _recall_memories
        recall = _recall_memories("test")
        count = getattr(recall, "count", 0)
        if count > 0:
            results.append(("Memory", "PASS", f"{count} memories stored", ""))
        else:
            results.append(("Memory", "SKIP", "No memories yet", "Will populate during use"))
    except Exception:
        results.append(("Memory", "SKIP", "Memory system unavailable", ""))


def _hooks_check(results: list, ctx: Any) -> None:
    try:
        hooks = ctx.hook_manager.list_hooks()
        if hooks:
            active = sum(1 for h in hooks if h.enabled)
            results.append(("Hooks", "PASS", f"{active} active / {len(hooks)} total", ""))
        else:
            results.append(("Hooks", "SKIP", "No hooks registered", "Configure in AURA.md"))
    except Exception:
        results.append(("Hooks", "SKIP", "Hook manager unavailable", ""))


def _activity_log_check(results: list, ctx: Any) -> None:
    try:
        log = getattr(ctx, "activity_log", None)
        if log:
            results.append(("Activity Log", "PASS", "Active", ""))
        else:
            results.append(("Activity Log", "SKIP", "Not initialized", ""))
    except Exception:
        results.append(("Activity Log", "SKIP", "Unavailable", ""))
