import argparse
import logging
from typing import Optional

logger = logging.getLogger(__name__)

"""Subcommand handlers for Aura Dev CLI.

Handles: aura init, aura doctor, aura config, aura models, aura commit, aura cost
"""

import json
import os
import subprocess
from pathlib import Path

from rich.panel import Panel
from rich.table import Table

try:
    from aura.cli.display import console
except ImportError:
    from rich.console import Console
    console = Console()


def _create_subcommand_permission_manager():
    """Create a permission manager for top-level subcommands.

    Subcommands don't have a live CLIContext, so they need a lightweight
    confirm callback to preserve the same policy model used in chat mode.
    """
    from aura.core.permissions import PermissionManager

    permissions = PermissionManager()
    permissions.set_mode("careful")

    def _confirm(tool_name: str, description: str) -> bool | str:
        console.print()
        console.print("  [bold yellow]Permission required:[/]")
        console.print(f"    {tool_name}")
        if description:
            for line in description.split("\n"):
                console.print(f"    [dim]{line}[/]")
        try:
            response = input("    Allow? [y/n/always]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return False
        if response == "always":
            return "always"
        return response in ("y", "yes")

    permissions.set_confirm_callback(_confirm)
    return permissions


def handle_subcommand(command: str, args: argparse.Namespace) -> int:
    """Dispatch subcommand. Returns exit code."""
    from aura.cli.commands.daemon_commands import cmd_start as _cmd_start
    from aura.cli.commands.daemon_commands import cmd_stop as _cmd_stop
    from aura.cli.commands.heatmap_commands import cmd_heatmap as _cmd_heatmap
    from aura.cli.commands.log_commands import cmd_log as _cmd_log
    from aura.cli.commands.worktree_commands import cmd_worktree as _cmd_worktree
    handlers = {
        "init": cmd_init,
        "setup": cmd_setup,
        "doctor": cmd_doctor,
        "config": cmd_config,
        "models": cmd_models,
        "commit": cmd_commit,
        "cost": cmd_cost,
        "ide": cmd_ide_setup,
        "status": cmd_status,
        "recall": cmd_recall,
        "why": cmd_why,
        "log": _cmd_log,
        "start": _cmd_start,
        "stop": _cmd_stop,
        "heatmap": _cmd_heatmap,
        "worktree": _cmd_worktree,
    }
    handler = handlers.get(command)
    if handler:
        try:
            return handler(args)
        except KeyboardInterrupt:
            return 130
        except Exception as e:
            console.print(f"[red]Error:[/] {e}")
            return 1
    console.print(f"[red]Unknown command:[/] {command}")
    return 1


def cmd_init(args: argparse.Namespace) -> int:
    """Create AURA.md in the current project."""
    from aura.tools.code_search import CodeSearchTool

    cwd = os.getcwd()
    aura_md = os.path.join(cwd, "AURA.md")

    if os.path.exists(aura_md):
        console.print(f"  AURA.md already exists at [cyan]{aura_md}[/]")
        return 0

    # Detect project type for template customization
    searcher = CodeSearchTool()
    project_info = searcher.detect_project_type(cwd)

    # Build template with frontmatter
    project_type = project_info.get("project_type", "unknown")
    frameworks = project_info.get("frameworks", [])
    stack = project_info.get("stack", [])

    # Detect test command
    test_cmd = _detect_test_cmd(cwd)

    frontmatter_lines = [
        "---",
        "tier: balanced",
        "# model: qwen3.5:397b-cloud",
    ]
    if test_cmd:
        frontmatter_lines.append(f"test_cmd: {test_cmd}")
        frontmatter_lines.append("auto_test: true")
    frontmatter_lines.extend([
        "# permissions:",
        "#   shell: auto",
        "#   edit_file: auto",
        "# max_iterations: 50",
        "# budget: 5.0",
        "---",
        "",
    ])

    body_lines = [
        f"# {Path(cwd).name}",
        "",
    ]
    if stack:
        body_lines.append(f"Stack: {', '.join(stack)}")
    if frameworks:
        body_lines.append(f"Frameworks: {', '.join(frameworks)}")
    body_lines.extend([
        "",
        "## Instructions",
        "",
        "<!-- Add project-specific instructions for Aura here -->",
        "",
    ])

    content = "\n".join(frontmatter_lines + body_lines)

    with open(aura_md, "w", encoding="utf-8") as f:
        f.write(content)

    console.print(f"  [green]Created[/] {aura_md}")
    if project_type != "unknown":
        console.print(f"  Detected: [cyan]{project_type}[/] project ({', '.join(stack)})")
    if test_cmd:
        console.print(f"  Test command: [cyan]{test_cmd}[/]")
    console.print("\n  Edit AURA.md to customize Aura's behavior for this project.")
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    """Check Ollama, models, dependencies."""
    console.print("\n[bold]Aura Doctor[/]\n")
    all_ok = True

    # 1. Check Ollama
    console.print("  [bold]Ollama:[/]")
    try:
        import ollama
        models = ollama.list()
        model_names = [m.get("name", m.get("model", "?")) for m in models.get("models", [])]
        console.print(f"    [green]Running[/], {len(model_names)} models loaded")
        for name in sorted(model_names)[:15]:
            console.print(f"      [dim]{name}[/]")
        if len(model_names) > 15:
            console.print(f"      [dim]... and {len(model_names) - 15} more[/]")
    except Exception as e:
        console.print(f"    [red]ERROR[/] Not reachable: {e}")
        console.print("    Run: [cyan]ollama serve[/]")
        all_ok = False

    # 2. Check key dependencies
    dep_table = Table(show_header=False, box=None, padding=(0, 2))
    dep_table.add_column("Package", style="bold")
    dep_table.add_column("Status")
    deps = [
        ("rich", "rich"),
        ("prompt_toolkit", "prompt_toolkit"),
        ("yaml", "PyYAML"),
        ("ollama", "ollama"),
    ]
    for module, pkg in deps:
        try:
            __import__(module)
            dep_table.add_row(pkg, "[green]OK[/]")
        except ImportError:
            dep_table.add_row(pkg, f"[red]MISSING[/] (pip install {pkg})")
            all_ok = False
    console.print("\n  [bold]Dependencies:[/]")
    console.print(dep_table)

    # 3. Check optional tools
    console.print("\n  [bold]Optional tools:[/]")
    optionals = [
        ("aura.tools.brave_search", "BraveSearchTool", "BRAVE_API_KEY"),
        ("aura.tools.tavily_tool", "TavilyTool", "TAVILY_API_KEY"),
    ]
    for module, cls, env_var in optionals:
        try:
            __import__(module)
            has_key = bool(os.environ.get(env_var))
            status = "[green]OK[/]" if has_key else f"[yellow]no {env_var}[/]"
            console.print(f"    {cls}: {status}")
        except ImportError:
            console.print(f"    {cls}: [dim]not installed[/]")

    # 4. Check AURA.md
    console.print("\n  [bold]Project:[/]")
    aura_md = os.path.join(os.getcwd(), "AURA.md")
    if os.path.exists(aura_md):
        console.print("    AURA.md: [green]found[/]")
    else:
        console.print("    AURA.md: [yellow]not found[/] (run: [cyan]aura init[/])")

    # 5. Check git
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            capture_output=True, text=True, timeout=5, cwd=os.getcwd(),
        )
        if result.returncode == 0:
            console.print("    Git repo: [green]yes[/]")
        else:
            console.print("    Git repo: [yellow]no[/]")
    except Exception:
        console.print("    Git: [dim]not available[/]")

    if all_ok:
        console.print("\n  [bold green]All checks passed![/]\n")
    else:
        console.print("\n  [bold yellow]Some issues found.[/]\n")
    return 0 if all_ok else 1


def cmd_config(args: argparse.Namespace) -> int:
    """Show current configuration including AURA.md overrides."""
    from aura.config import Config
    from aura.core.context import get_aura_md_config

    console.print("\n[bold]Aura Configuration[/]\n")

    # Global config
    tbl = Table(show_header=False, box=None, padding=(0, 2))
    tbl.add_column("Setting", style="bold")
    tbl.add_column("Value", style="cyan")
    tbl.add_row("Model (fast)", str(Config.MODEL_FAST))
    tbl.add_row("Model (reason)", str(Config.MODEL_REASON))
    tbl.add_row("Model (code)", str(Config.MODEL_CODE))
    tbl.add_row("Ollama host", str(getattr(Config, "OLLAMA_HOST", "http://localhost:11434")))
    console.print("  [bold]Global:[/]")
    console.print(tbl)

    # Model chains
    chains = {
        "Fast chain": getattr(Config, "MODEL_FAST_CHAIN", []),
        "Reason chain": getattr(Config, "MODEL_REASON_CHAIN", []),
        "Code chain": getattr(Config, "MODEL_CODE_CHAIN", []),
    }
    has_chains = any(chains.values())
    if has_chains:
        console.print("\n  [bold]Model chains:[/]")
        for label, chain in chains.items():
            if chain:
                console.print(f"    {label}: [cyan]{' -> '.join(chain)}[/]")

    # Project-level AURA.md overrides
    aura_config = get_aura_md_config(os.getcwd())
    if aura_config:
        proj_tbl = Table(show_header=False, box=None, padding=(0, 2))
        proj_tbl.add_column("Key", style="bold")
        proj_tbl.add_column("Value", style="cyan")
        for key in ["tier", "model", "test_cmd", "auto_test", "max_iterations", "budget"]:
            val = aura_config.get(key)
            if val is not None:
                proj_tbl.add_row(key, str(val))
        perms = aura_config.get("permissions")
        if perms:
            proj_tbl.add_row("permissions", str(perms))
        console.print("\n  [bold]Project (AURA.md):[/]")
        console.print(proj_tbl)
    else:
        console.print(f"\n  No AURA.md found in {os.getcwd()} (run: [cyan]aura init[/])")

    console.print()
    return 0


def cmd_models(args: argparse.Namespace) -> int:
    """List available models with routing roles."""
    from aura.core.router import ROUTING_TABLE

    console.print("\n[bold]Aura Model Routing[/]\n")

    tbl = Table(box=None, padding=(0, 2))
    tbl.add_column("Category", style="bold")
    tbl.add_column("local", style="dim")
    tbl.add_column("balanced", style="cyan")
    tbl.add_column("max", style="green")

    for category, tiers in ROUTING_TABLE.items():
        tbl.add_row(
            category,
            tiers.get("local", "-"),
            tiers.get("balanced", "-"),
            tiers.get("max", "-"),
        )
    console.print(tbl)

    # Show which models are actually available
    console.print()
    try:
        import ollama
        models = ollama.list()
        available = {m.get("name", m.get("model", "")) for m in models.get("models", [])}
        console.print(f"  [green]{len(available)}[/] models available locally")
    except Exception:
        console.print("  [dim](Could not check available models — is Ollama running?)[/]")

    console.print()
    return 0


def cmd_commit(args: argparse.Namespace) -> int:
    """Smart commit with AI-generated message."""
    from aura import ApprenticeAgent
    from aura.tools.git_tool import GitTool

    git = GitTool()
    cwd = os.getcwd()
    permissions = _create_subcommand_permission_manager()

    # Check for changes
    status = git.status(cwd)
    if not status.get("success"):
        console.print("[red]Not in a git repository or git error.[/]")
        return 1

    diff_result = git.diff(cwd)
    diff_text = diff_result.get("diff", "")

    if not diff_text and not status.get("dirty_count", 0):
        console.print("No changes to commit.")
        return 0

    # Stage all if --all flag
    if getattr(args, 'all', False):
        if not permissions.check("git", {"action": "add", "files": "."}):
            console.print("  Cancelled.")
            return 0
        add_result = git.add(cwd, files=".")
        if not add_result.get("success"):
            console.print(f"[red]Stage failed:[/] {add_result.get('error', 'unknown error')}")
            return 1

    # Get diff of staged changes
    try:
        subprocess.run(
            ["git", "diff", "--cached", "--stat"],
            capture_output=True, text=True, timeout=10, cwd=cwd,
        )
        staged_diff_proc = subprocess.run(
            ["git", "diff", "--cached"],
            capture_output=True, text=True, timeout=10, cwd=cwd,
        )
        diff_text = staged_diff_proc.stdout
    except (subprocess.TimeoutExpired, FileNotFoundError):
        diff_text = diff_result.get("diff", "")

    if not diff_text:
        console.print("No staged changes. Use [cyan]git add[/] first or pass [cyan]--all[/].")
        return 1

    # Generate commit message
    console.print("[dim]Generating commit message...[/]")
    try:
        agent = ApprenticeAgent()
        # Use more diff context for better messages
        max_diff = 8000
        truncated = f"\n... (truncated {len(diff_text) - max_diff} chars)" if len(diff_text) > max_diff else ""
        prompt = f"""Generate a concise git commit message for these changes.
Return ONLY the commit message (1-2 lines), no explanation.

Diff:
{diff_text[:max_diff]}{truncated}"""

        result = agent.brain.think(prompt, use_history=False)
        message = result.strip().strip('"').strip("'").strip("`")

        # Clean up common LLM artifacts
        for prefix in ["commit message:", "here's the commit message:", "here's",
                        "here is the commit message:", "here is", "message:"]:
            if message.lower().startswith(prefix):
                message = message[len(prefix):].strip().strip('"').strip("'")
                break

        if not message:
            console.print("[red]Error:[/] LLM returned empty commit message.")
            return 1

    except Exception as e:
        console.print(f"[red]Error generating message:[/] {e}")
        return 1

    console.print(f"\n  Commit message: [bold]{message}[/]\n")
    try:
        confirm = input("  Edit commit message before approval? [y/N]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return 1

    if confirm == "edit" or confirm == "e":
        try:
            message = input("  Enter message: ").strip()
        except (EOFError, KeyboardInterrupt):
            return 1
    elif confirm in ("y", "yes"):
        try:
            message = input("  Enter message: ").strip()
        except (EOFError, KeyboardInterrupt):
            return 1
    elif confirm not in ("n", "no", ""):
        console.print("  Cancelled.")
        return 0

    if not permissions.check("git", {"action": "commit", "message": message}):
        console.print("  Cancelled.")
        return 0

    result = git.commit(cwd, message=message)
    if result.get("success"):
        console.print(f"  [green]Committed:[/] {message}")
        return 0
    else:
        console.print(f"  [red]Commit failed:[/] {result.get('error', 'unknown error')}")
        return 1


def cmd_cost(args: argparse.Namespace) -> int:
    """Show session cost breakdown from activity log."""
    try:
        from aura.cli.activity_log import ActivityLog
        log = ActivityLog()
    except (ImportError, OSError) as e:
        console.print(f"\n[red]Could not read activity log:[/] {e}")
        console.print("Cost data is tracked during interactive sessions.\n")
        return 1

    session_id = getattr(args, "session", "") or ""
    by_model = bool(getattr(args, "by_model", False))
    by_provider = bool(getattr(args, "by_provider", False))

    try:
        stats = log.get_stats(session_id=session_id)
    except Exception as e:
        console.print(f"\n[red]Stats query failed:[/] {e}")
        return 1

    title = "Aura Cost Summary"
    if session_id:
        title += f" \u2014 session {session_id}"
    console.print(f"\n[bold]{title}[/]\n")

    total_cost = stats.get("total_cost", 0.0)
    total_interactions = stats.get("total_interactions", 0)
    tokens_in = stats.get("total_tokens_in", 0)
    tokens_out = stats.get("total_tokens_out", 0)
    total_tokens = tokens_in + tokens_out
    total_tool_calls = stats.get("total_tool_calls", 0)

    tbl = Table(show_header=False, box=None, padding=(0, 2))
    tbl.add_column("Metric", style="bold")
    tbl.add_column("Value", style="cyan")
    tbl.add_row("Total cost", f"${total_cost:.4f}")
    tbl.add_row("Interactions", str(total_interactions))
    tbl.add_row("Tokens", f"{total_tokens:,} (in: {tokens_in:,} / out: {tokens_out:,})")
    tbl.add_row("Tool calls", str(total_tool_calls))
    console.print(tbl)

    if by_model:
        rows = log.get_stats_by_model(session_id=session_id)
        if rows:
            console.print("\n[bold]By Model[/]")
            mt = Table(box=None, padding=(0, 2))
            mt.add_column("Model", style="cyan")
            mt.add_column("Calls", justify="right")
            mt.add_column("Tokens in", justify="right")
            mt.add_column("Tokens out", justify="right")
            mt.add_column("Cost", justify="right", style="yellow")
            for r in rows:
                mt.add_row(
                    r["model"], str(r["interactions"]),
                    f"{r['tokens_in']:,}", f"{r['tokens_out']:,}",
                    f"${r['cost']:.4f}",
                )
            console.print(mt)

    if by_provider:
        rows = log.get_stats_by_provider(session_id=session_id)
        if rows:
            console.print("\n[bold]By Provider[/]")
            pt = Table(box=None, padding=(0, 2))
            pt.add_column("Provider", style="cyan")
            pt.add_column("Models", justify="right")
            pt.add_column("Calls", justify="right")
            pt.add_column("Tokens", justify="right")
            pt.add_column("Cost", justify="right", style="yellow")
            for r in rows:
                pt.add_row(
                    r["provider"], str(r["models"]), str(r["interactions"]),
                    f"{(r['tokens_in'] + r['tokens_out']):,}",
                    f"${r['cost']:.4f}",
                )
            console.print(pt)

    try:
        from aura.reliability import all_rate_limit_snapshots
        snaps = all_rate_limit_snapshots()
        if snaps:
            console.print("\n[bold]Rate Limits (most recent observations)[/]")
            rt = Table(box=None, padding=(0, 2))
            rt.add_column("Provider", style="cyan")
            rt.add_column("RPM", justify="right")
            rt.add_column("TPM", justify="right")
            for provider, state in snaps.items():
                rpm = f"{state.requests_min.remaining}/{state.requests_min.limit}" if state.requests_min.limit else "-"
                tpm = f"{state.tokens_min.remaining}/{state.tokens_min.limit}" if state.tokens_min.limit else "-"
                rt.add_row(provider, rpm, tpm)
            console.print(rt)
    except Exception:
        pass

    console.print()
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    """Show a quick overview of what Aura is doing right now."""
    console.print("\n[bold]Aura Status[/]\n")

    console.print("  [bold]Ollama:[/]")
    try:
        import ollama
        ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        models = ollama.list()
        model_count = len(models.get("models", []))
        console.print(f"    [green]reachable[/] @ {ollama_host}  ({model_count} models)")
    except Exception as e:
        console.print(f"    [red]unreachable[/]: {e}")

    console.print("\n  [bold]Config:[/]")
    try:
        from aura.config import Config
        console.print(f"    tier default: [cyan]{getattr(Config, 'MODEL_REASON', '?')}[/]")
        console.print(f"    budget mode:  {getattr(Config, 'BUDGET_MODE', False)}")
        if getattr(Config, 'BUDGET_MODE', False):
            console.print(f"    budget cap:   ${getattr(Config, 'BUDGET_MAX_USD_PER_SESSION', 0):.2f}")
    except Exception as e:
        console.print(f"    [yellow]config load failed: {e}[/]")

    console.print("\n  [bold]Routing stats:[/]")
    try:
        from aura.reliability.routing_stats import get_routing_stats
        store = get_routing_stats()
        stats = getattr(store, "_stats", {})
        models_seen = sorted({m for (_, m) in stats.keys()})[:10]
        console.print(f"    {len(stats)} (category, model) pairs tracked")
        if models_seen:
            console.print(f"    top models: [dim]{', '.join(models_seen)}[/]")
    except Exception as e:
        console.print(f"    [yellow]stats unavailable: {e}[/]")

    console.print("\n  [bold]Strategy bandit:[/]")
    try:
        import sqlite3
        from pathlib import Path as _P
        data_dir = _P(os.getenv("AURA_DATA_DIR", "data"))
        db_path = data_dir / "aura_meta.db"
        if db_path.exists():
            conn = sqlite3.connect(str(db_path))
            rows = conn.execute(
                "SELECT strategy, COUNT(*), COALESCE(SUM(total_pulls), 0) FROM strategy_arms GROUP BY strategy"
            ).fetchall()
            conn.close()
            for strategy, arms, pulls in rows:
                console.print(f"    {strategy}: {arms} arms, {pulls} pulls")
        else:
            console.print("    [dim]no bandit DB yet[/]")
    except Exception as e:
        console.print(f"    [yellow]bandit unavailable: {e}[/]")

    console.print("\n  [bold]Daemon:[/]")
    try:
        from pathlib import Path as _P
        pid_file = _P(os.getenv("AURA_DATA_DIR", "data")) / "daemon.pid"
        if pid_file.exists():
            pid = pid_file.read_text().strip()
            console.print(f"    PID: [cyan]{pid}[/] (see {pid_file})")
        else:
            console.print("    [dim]not running[/]")
    except Exception:
        console.print("    [dim]unknown[/]")

    console.print()
    return 0


def cmd_why(args: argparse.Namespace) -> int:
    """Intent-to-Code Ledger query. Usage: aura why <file>[:line] [--limit N]"""
    target = (getattr(args, "why_target", "") or "").strip()
    if not target:
        console.print("[yellow]Usage: aura why <file>[:<line>][/]")
        return 1

    file_part = target
    line_part: Optional[int] = None
    if ":" in target:
        file_str, line_str = target.rsplit(":", 1)
        try:
            line_part = int(line_str)
            file_part = file_str
        except ValueError:
            file_part = target

    limit = int(getattr(args, "why_limit", 5) or 5)

    try:
        from aura import ledger
        entries = ledger.why(file_part, line=line_part, limit=limit)
    except Exception as e:
        console.print(f"[red]Ledger query failed:[/] {e}")
        return 1

    if not entries:
        console.print(f"  [dim]No ledger entries for {target}[/]")
        return 0

    import datetime as _dt
    table = Table(show_header=True, header_style="bold", box=None, padding=(0, 1))
    table.add_column("When", style="dim")
    table.add_column("Kind")
    table.add_column("Lines")
    table.add_column("Model")
    table.add_column("Intent")
    for e in entries:
        when = _dt.datetime.fromtimestamp(e.get("ts", 0)).strftime("%m-%d %H:%M")
        kind = e.get("kind", "edit")
        lines = e.get("lines_touched") or []
        lines_str = f"{lines[0]}-{lines[-1]}" if len(lines) >= 2 else ""
        model = (e.get("model") or "")[:28]
        intent = (e.get("intent") or "").replace("\n", " ")[:80]
        table.add_row(when, kind, lines_str, model, intent)
    console.print()
    console.print(f"  [bold]Why {target}?[/] (top {len(entries)})")
    console.print(table)
    console.print()
    return 0


def cmd_recall(args: argparse.Namespace) -> int:
    """Query UnifiedMemory from the shell. Usage: aura recall "topic" [--limit 5]"""
    query = " ".join(getattr(args, "recall_query", []) or []).strip()
    if not query:
        console.print("[yellow]Usage: aura recall \"your topic\"[/]")
        return 1
    limit = getattr(args, "recall_limit", 5) or 5

    try:
        from aura.memory.unified_memory import UnifiedMemory
        mem = UnifiedMemory()
    except Exception as e:
        console.print(f"[red]UnifiedMemory unavailable:[/] {e}")
        return 1

    try:
        results = mem.query(query, k=limit)
    except Exception as e:
        console.print(f"[red]Query failed:[/] {e}")
        return 1

    if not results:
        console.print(f"  [dim]No matches for '[cyan]{query}[/]'[/]")
        return 0

    console.print(f"\n  [bold]Top {len(results)} matches for '[cyan]{query}[/]'[/]\n")
    for i, r in enumerate(results, 1):
        content = getattr(r, "content", None) or (r.get("content") if isinstance(r, dict) else str(r))
        score = getattr(r, "score", None) or (r.get("score") if isinstance(r, dict) else None)
        source = getattr(r, "source", None) or ""
        snippet = (content or "")[:220].replace("\n", " ")
        score_str = f"[dim]({score:.3f})[/]" if isinstance(score, (int, float)) else ""
        src_str = f"[dim cyan][{source}][/]" if source else ""
        console.print(f"  {i}. {score_str} {src_str} {snippet}")
    console.print()
    return 0


def cmd_ide_setup(args: argparse.Namespace) -> int:
    """Dispatch `aura ide {setup|reset|validate}`."""
    action = (getattr(args, "action", None) or "setup").lower()
    if action == "reset":
        return _cmd_ide_reset()
    if action == "validate":
        return _cmd_ide_validate()
    return _cmd_ide_setup_apply()


def _cmd_ide_setup_apply() -> int:
    """Generate VS Code tasks.json and print MCP config snippet."""
    import sys as _sys

    cwd = os.getcwd()
    vscode_dir = os.path.join(cwd, ".vscode")
    tasks_path = os.path.join(vscode_dir, "tasks.json")

    # Resolve the current Python interpreter and absolute path to Aura's main.py.
    # `python -m main` would only work when cwd contains a top-level main module,
    # which is not the case for arbitrary user projects.
    main_py = str(Path(__file__).resolve().parents[2] / "main.py")
    aura_cmd = f'"{_sys.executable}" "{main_py}"'

    # Aura tasks for VS Code
    aura_tasks = [
        {
            "label": "Aura: Chat",
            "type": "shell",
            "command": aura_cmd,
            "presentation": {"reveal": "always", "panel": "dedicated"},
            "problemMatcher": [],
        },
        {
            "label": "Aura: Run Prompt",
            "type": "shell",
            "command": f'{aura_cmd} -p "${{input:auraPrompt}}"',
            "presentation": {"reveal": "always"},
            "problemMatcher": [],
        },
        {
            "label": "Aura: Init Project",
            "type": "shell",
            "command": f"{aura_cmd} init",
            "presentation": {"reveal": "always"},
            "problemMatcher": [],
        },
        {
            "label": "Aura: Smart Commit",
            "type": "shell",
            "command": f"{aura_cmd} commit --all",
            "presentation": {"reveal": "always"},
            "problemMatcher": [],
        },
    ]

    inputs = [
        {
            "id": "auraPrompt",
            "description": "What should Aura do?",
            "type": "promptString",
        },
    ]

    # Merge with existing tasks.json if present
    if os.path.exists(tasks_path):
        try:
            with open(tasks_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except (json.JSONDecodeError, OSError):
            existing = {"version": "2.0.0", "tasks": []}

        # Remove old Aura tasks
        existing_tasks = [
            t for t in existing.get("tasks", [])
            if not t.get("label", "").startswith("Aura:")
        ]
        existing_tasks.extend(aura_tasks)
        existing["tasks"] = existing_tasks

        # Add inputs if not present
        existing_inputs = existing.get("inputs", [])
        existing_input_ids = {i.get("id") for i in existing_inputs}
        for inp in inputs:
            if inp["id"] not in existing_input_ids:
                existing_inputs.append(inp)
        existing["inputs"] = existing_inputs

        tasks_data = existing
    else:
        tasks_data = {
            "version": "2.0.0",
            "tasks": aura_tasks,
            "inputs": inputs,
        }

    # Write tasks.json
    os.makedirs(vscode_dir, exist_ok=True)
    with open(tasks_path, "w", encoding="utf-8") as f:
        json.dump(tasks_data, f, indent=2)

    console.print(f"\n  [green]Created[/] {tasks_path}")
    console.print("    - Aura: Chat (interactive mode)")
    console.print("    - Aura: Run Prompt (one-shot)")
    console.print("    - Aura: Init Project")
    console.print("    - Aura: Smart Commit")

    # Print MCP config snippet
    # Point cwd at Aura's install dir so `python -m aura.core.mcp_server` can
    # resolve the package without requiring a site-packages install.
    aura_root = str(Path(__file__).resolve().parents[2])
    console.print("\n  [bold]MCP Server config for VS Code settings.json:[/]\n")
    mcp_config = {
        "mcp.servers": {
            "aura": {
                "command": _sys.executable,
                "args": ["-m", "aura.core.mcp_server"],
                "cwd": aura_root,
            }
        }
    }
    console.print(Panel(json.dumps(mcp_config, indent=4), border_style="dim"))
    console.print()
    return 0


def _cmd_ide_reset() -> int:
    """Remove Aura-authored entries from .vscode/tasks.json; leave user tasks intact.

    We identify Aura tasks by their "Aura: " label prefix (the convention used by
    the setup path). Non-Aura tasks, inputs that weren't authored by us, and any
    other keys in the tasks.json are preserved untouched.
    """
    cwd = os.getcwd()
    tasks_path = os.path.join(cwd, ".vscode", "tasks.json")

    if not os.path.exists(tasks_path):
        console.print(f"  [dim]No {tasks_path} to reset.[/dim]")
        return 0

    try:
        with open(tasks_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        console.print(f"  [red]Could not parse {tasks_path}: {e}[/red]")
        return 1

    original_tasks = data.get("tasks", [])
    kept_tasks = [t for t in original_tasks if not str(t.get("label", "")).startswith("Aura:")]
    removed = len(original_tasks) - len(kept_tasks)

    # Remove inputs we added (auraPrompt is the only one)
    inputs = data.get("inputs", [])
    kept_inputs = [i for i in inputs if i.get("id") != "auraPrompt"]
    inputs_removed = len(inputs) - len(kept_inputs)

    if removed == 0 and inputs_removed == 0:
        console.print("  [dim]No Aura-authored entries found; nothing to reset.[/dim]")
        return 0

    data["tasks"] = kept_tasks
    if kept_inputs or inputs_removed:
        data["inputs"] = kept_inputs

    # If nothing else is left in the file, offer to remove it entirely.
    is_empty = not kept_tasks and not kept_inputs and set(data.keys()) <= {"version", "tasks", "inputs"}
    if is_empty:
        try:
            os.remove(tasks_path)
            console.print(f"  [green]Removed[/] {tasks_path} (was only Aura entries).")
        except OSError as e:
            console.print(f"  [red]Could not remove {tasks_path}: {e}[/red]")
            return 1
    else:
        try:
            with open(tasks_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            console.print(f"  [green]Reset[/] {tasks_path} (removed {removed} task(s), {inputs_removed} input(s))")
        except OSError as e:
            console.print(f"  [red]Could not write {tasks_path}: {e}[/red]")
            return 1
    return 0


def _cmd_ide_validate() -> int:
    """Compare current .vscode/tasks.json against what cmd_ide_setup would generate.

    Reports any drift: missing Aura tasks, extra Aura tasks, or a command string
    that has changed (usually means Aura was moved or the Python interpreter
    changed). Exit code 0 = matches, 1 = drift, 2 = no integration installed.
    """
    import sys as _sys

    cwd = os.getcwd()
    tasks_path = os.path.join(cwd, ".vscode", "tasks.json")
    if not os.path.exists(tasks_path):
        console.print("  [yellow]No Aura IDE integration installed.[/] Run: aura ide setup")
        return 2

    try:
        with open(tasks_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        console.print(f"  [red]Could not parse {tasks_path}: {e}[/red]")
        return 1

    main_py = str(Path(__file__).resolve().parents[2] / "main.py")
    expected_cmd_prefix = f'"{_sys.executable}" "{main_py}"'
    expected_labels = {
        "Aura: Chat",
        "Aura: Run Prompt",
        "Aura: Init Project",
        "Aura: Smart Commit",
    }

    actual_aura_tasks = [t for t in data.get("tasks", []) if str(t.get("label", "")).startswith("Aura:")]
    actual_labels = {t.get("label") for t in actual_aura_tasks}

    missing = expected_labels - actual_labels
    extra = actual_labels - expected_labels
    stale_cmd: list[str] = []
    for t in actual_aura_tasks:
        cmd = str(t.get("command", ""))
        if cmd and expected_cmd_prefix not in cmd:
            stale_cmd.append(f"  - {t.get('label', '?')}: command path drifted")

    drift = bool(missing or extra or stale_cmd)
    if not drift:
        console.print(f"  [green]OK[/] — {tasks_path} matches current Aura integration.")
        return 0

    console.print(f"  [yellow]Drift detected in {tasks_path}:[/]")
    if missing:
        console.print(f"  Missing: {', '.join(sorted(missing))}")
    if extra:
        console.print(f"  Extra Aura tasks (not in current template): {', '.join(sorted(extra))}")
    for line in stale_cmd:
        console.print(line)
    console.print("\n  Fix with: [cyan]aura ide setup[/] (re-applies current template)")
    return 1


def _detect_test_cmd(project_root: str) -> str:
    """Try to detect the project's test command."""
    root = Path(project_root)

    # Python
    if (root / "pytest.ini").exists() or (root / "pyproject.toml").exists():
        return "pytest"
    if (root / "setup.py").exists():
        return "python -m pytest"

    # Node
    pkg_json = root / "package.json"
    if pkg_json.exists():
        try:
            pkg = json.loads(pkg_json.read_text())
            scripts = pkg.get("scripts", {})
            if "test" in scripts:
                test_script = scripts["test"]
                if "vitest" in test_script:
                    return "npx vitest run"
                elif "jest" in test_script:
                    return "npx jest"
                return "npm test"
        except Exception as e:
            logger.debug(f"[Commands] non-critical: {e}")
    # Rust
    if (root / "Cargo.toml").exists():
        return "cargo test"

    # Go
    if (root / "go.mod").exists():
        return "go test ./..."

    return ""


def cmd_setup(args: argparse.Namespace) -> int:
    """Interactive setup wizard for configuring Aura in a project."""
    cwd = os.getcwd()
    project_name = Path(cwd).name

    console.print(f"\n  [bold]Aura Setup Wizard[/] for [cyan]'{project_name}'[/]\n")

    # Step 1: Detect project type
    console.print("  [bold]Step 1:[/] Detecting project...")
    try:
        from aura.tools.code_search import CodeSearchTool
        searcher = CodeSearchTool()
        info = searcher.detect_project_type(cwd)
        project_type = info.get("project_type", "unknown")
        frameworks = info.get("frameworks", [])
        stack = info.get("stack", [])
        if project_type != "unknown":
            console.print(f"    Detected: [cyan]{project_type}[/] ({', '.join(stack)})")
        else:
            console.print("    [yellow]Could not auto-detect project type.[/]")
    except Exception:
        project_type = "unknown"
        frameworks = []
        stack = []
        console.print("    [yellow]Could not auto-detect (code_search unavailable).[/]")

    # Step 2: Choose tier
    console.print("\n  [bold]Step 2:[/] Choose model tier")
    console.print("    [cyan]fast[/]     — Quick responses, lower cost")
    console.print("    [cyan]balanced[/] — Good balance of speed and quality (recommended)")
    console.print("    [cyan]max[/]      — Best quality, higher cost")
    tier = _prompt("    Tier", "balanced")
    if tier not in ("fast", "balanced", "max"):
        console.print(f"    [yellow]Invalid tier '{tier}', using 'balanced'.[/]")
        tier = "balanced"

    # Step 3: Model
    model = _prompt("\n  Step 3: Model (or 'auto' for smart routing)", "auto")

    # Step 4: Test command
    detected_test = _detect_test_cmd(cwd)
    default_test = detected_test or "pytest"
    test_cmd = _prompt("\n  Step 4: Test command", default_test)
    auto_test_str = _prompt("    Auto-run tests after edits? [y/n]", "y")
    auto_test = auto_test_str.lower() in ("y", "yes")

    # Step 5: API keys
    console.print("\n  [bold]Step 5:[/] Checking API keys...")
    ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    is_ollama_cloud = "api.ollama.com" in ollama_host
    key_checks = []
    if is_ollama_cloud:
        key_checks.append(("OLLAMA_API_KEY", "Ollama Cloud authentication"))
    else:
        console.print(f"    [dim]Ollama host: {ollama_host} (local — no API key required)[/]")
    key_checks.extend([
        ("BRAVE_API_KEY", "web search"),
        ("TAVILY_API_KEY", "web search"),
    ])
    for key_name, purpose in key_checks:
        has = bool(os.environ.get(key_name))
        status = "[green]found[/]" if has else f"[yellow]not set[/] ({purpose})"
        console.print(f"    {key_name}: {status}")

    # Step 6: Generate AURA.md
    console.print("\n  [bold]Step 6:[/] Creating AURA.md...")
    aura_md_path = os.path.join(cwd, "AURA.md")

    if os.path.exists(aura_md_path):
        overwrite = _prompt("    AURA.md already exists. Overwrite? [y/n]", "n")
        if overwrite.lower() not in ("y", "yes"):
            console.print("    Kept existing AURA.md.")
            console.print("\n  [bold green]Setup complete![/] Run [cyan]aura[/] to start.\n")
            return 0

    # Build content
    lines = ["---", f"tier: {tier}"]
    if model and model != "auto":
        lines.append(f"model: {model}")
    if test_cmd:
        lines.append(f"test_cmd: {test_cmd}")
    if auto_test:
        lines.append("auto_test: true")
    lines.extend([
        "# permissions:",
        "#   shell: auto",
        "#   edit_file: auto",
        "---",
        "",
        f"# {project_name}",
        "",
    ])
    if stack:
        lines.append(f"Stack: {', '.join(stack)}")
    if frameworks:
        lines.append(f"Frameworks: {', '.join(frameworks)}")
    lines.extend([
        "",
        "## Instructions",
        "",
        "<!-- Add project-specific instructions for Aura here -->",
        "",
    ])

    content = "\n".join(lines)
    with open(aura_md_path, "w", encoding="utf-8") as f:
        f.write(content)

    console.print(f"    [green]Created[/] {aura_md_path}")
    console.print("\n  [bold green]Setup complete![/] Run [cyan]aura[/] to start.\n")
    return 0


def _prompt(text: str, default: str) -> str:
    """Prompt with a default value shown in brackets."""
    try:
        value = input(f"{text} [{default}]: ").strip()
        return value if value else default
    except (EOFError, KeyboardInterrupt):
        return default
