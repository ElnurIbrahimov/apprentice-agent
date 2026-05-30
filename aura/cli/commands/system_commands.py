import logging
from typing import Optional

from ..context import get_ctx
from ..display import console
from .common import command, TIER_BETA, TIER_EXPERIMENTAL, TIER_STABLE

logger = logging.getLogger(__name__)


@command("/hook",     "Manage hooks",                                     tier=TIER_BETA)
def handle_hook(agent, arg, context) -> Optional[str]:
    from ..display import console as _hook_console
    ctx = get_ctx()
    _hook_mgr = ctx.hook_manager if ctx else None
    if _hook_mgr is None:
        from ..hooks import HookManager
        _hook_mgr = HookManager()
        if ctx:
            ctx.hook_manager = _hook_mgr
    from ..hooks import render_hooks_table
    sub = arg.strip().split(None, 1)
    if not sub or sub[0] == "list":
        render_hooks_table(_hook_console, _hook_mgr.list_hooks())
    elif sub[0] == "add" and len(sub) > 1:
        parts_h = sub[1].split(None, 1)
        if len(parts_h) == 2:
            try:
                _hook_mgr.add(parts_h[0], parts_h[1])
                _hook_console.print(f"[green]Hook added: {parts_h[0]} -> {parts_h[1]}[/green]")
            except ValueError as e:
                from ..display import show_error as _hook_err
                _hook_err(str(e))
        else:
            _hook_console.print("[dim]Usage: /hook add <event> <command>[/dim]")
    elif sub[0] == "remove" and len(sub) > 1:
        if _hook_mgr.remove(sub[1]):
            _hook_console.print(f"[green]Hook removed: {sub[1]}[/green]")
        else:
            from ..display import show_error as _hook_err2
            _hook_err2(f"Hook not found: {sub[1]}")


@command("/mcp",      "Manage MCP server connections",                    tier=TIER_BETA)
def handle_mcp(agent, arg, context) -> Optional[str]:
    """Dispatch /mcp subcommands. Supports: list (default), status, reconnect <name>."""
    ctx = get_ctx()
    mgr = None
    if ctx and ctx.agentic_loop and hasattr(ctx.agentic_loop, '_mcp_client'):
        mgr = ctx.agentic_loop._mcp_client

    if mgr is None or not getattr(mgr, "connections", None):
        console.print(
            "  [dim]No MCP servers connected. Configure in AURA.md under mcp_servers:[/dim]"
        )
        return None

    parts = (arg or "").strip().split(None, 1)
    sub = parts[0].lower() if parts else "list"
    sub_arg = parts[1].strip() if len(parts) > 1 else ""

    if sub in ("", "list"):
        for name, conn in mgr.connections.items():
            console.print(f"  [cyan]{name}[/cyan]: {len(conn.tools)} tools")
            for t in conn.tools[:5]:
                console.print(
                    f"    [dim]- {t['name']}: {t.get('description', '')[:60]}[/dim]"
                )
            if len(conn.tools) > 5:
                console.print(f"    [dim]... and {len(conn.tools) - 5} more[/dim]")
        return None

    if sub == "status":
        for name, conn in mgr.connections.items():
            ok = bool(getattr(conn, "tools", None))
            dot = "[green]●[/green]" if ok else "[red]●[/red]"
            tool_count = len(conn.tools) if ok else 0
            console.print(
                f"  {dot} [cyan]{name}[/cyan]  {tool_count} tools"
            )
        return None

    if sub == "reconnect":
        if not sub_arg:
            console.print("  [dim]Usage: /mcp reconnect <server-name>[/dim]")
            return None
        if sub_arg not in mgr.connections:
            console.print(f"  [red]Unknown MCP server:[/red] {sub_arg}")
            return None
        try:
            # Reconnect pattern: call the manager's reconnect if present,
            # else fall back to disconnect+connect. Keep this best-effort —
            # MCP client API may vary across versions.
            if hasattr(mgr, "reconnect"):
                mgr.reconnect(sub_arg)
            elif hasattr(mgr, "disconnect") and hasattr(mgr, "connect"):
                mgr.disconnect(sub_arg)
                mgr.connect(sub_arg)
            else:
                console.print(
                    "  [yellow]Reconnect not supported by this MCP client.[/yellow]"
                )
                return None
            console.print(f"  [green]Reconnected:[/green] {sub_arg}")
        except Exception as e:
            console.print(f"  [red]Reconnect failed:[/red] {e}")
        return None

    console.print(
        "  [dim]Usage: /mcp [list|status|reconnect <name>][/dim]"
    )


@command("/audit",    "Inspect Merkle audit chain",                       tier=TIER_BETA)
def handle_audit(agent, arg, context) -> Optional[str]:
    _handle_audit_command(arg)


@command("/evolve",   "Evolve skills with GEPA",                          tier=TIER_EXPERIMENTAL)
def handle_evolve(agent, arg, context) -> Optional[str]:
    _handle_evolve_command(agent, arg)


def _handle_audit_command(arg: str):
    from aura.security.audit_chain import get_audit_chain

    chain = get_audit_chain()
    parts = arg.strip().split(maxsplit=1)
    subcmd = parts[0].lower() if parts else "tail"

    if subcmd == "verify":
        console.print("  Verifying audit chain integrity...")
        valid, count, error = chain.verify()
        if valid:
            console.print(f"  [green]VALID[/green] — {count} entries verified, chain intact.")
        else:
            console.print(f"  [red]TAMPERED[/red] — integrity failure at entry {count}: {error}")

    elif subcmd == "tail":
        n = 15
        try:
            n = int(parts[1]) if len(parts) > 1 else 15
        except ValueError:
            pass
        entries = chain.tail(n)
        if not entries:
            console.print("  [dim]Audit chain is empty.[/dim]")
            return
        console.print(f"\n  [bold]Last {len(entries)} audit entries:[/bold]")
        from datetime import datetime
        for e in entries:
            ts = datetime.fromtimestamp(e.timestamp).strftime("%H:%M:%S")
            data_preview = e.action_data[:60] + "..." if len(e.action_data) > 60 else e.action_data
            console.print(f"  [dim][{ts}][/dim] [cyan]{e.action_type:<16}[/cyan] {e.agent_id:<12} {data_preview}")
        console.print(f"\n  [dim]Total entries: {chain.count()} | Chain hash: {entries[-1].entry_hash[:16]}...[/dim]")

    elif subcmd == "count":
        console.print(f"  Audit chain: [bold]{chain.count()}[/bold] entries")

    else:
        console.print("[yellow]Usage: /audit <verify|tail|count> [args][/yellow]")
        console.print("  [dim]/audit verify    — Verify chain integrity (detect tampering)[/dim]")
        console.print("  [dim]/audit tail [n]  — Show last N entries (default 15)[/dim]")
        console.print("  [dim]/audit count     — Show total entry count[/dim]")


def _handle_evolve_command(agent, arg: str):
    try:
        from aura.evolution.runner import run_evolution
        from aura_skill_library.skill_store import SkillStore
    except ImportError as e:
        console.print(f"\n  [red][GEPA] Import error: {e}[/red]\n")
        return

    parts = arg.split() if arg else []
    skill_ids = None
    dry_run = "--dry-run" in parts
    max_iterations = 5

    for i, p in enumerate(parts):
        if p == "--skill-ids" and i + 1 < len(parts):
            skill_ids = [s.strip() for s in parts[i + 1].split(",") if s.strip()]
        if p == "--skill" and i + 1 < len(parts):
            skill_ids = [parts[i + 1]]
        if p == "--max-iterations" and i + 1 < len(parts):
            try:
                max_iterations = int(parts[i + 1])
            except ValueError:
                console.print(f"  [red][GEPA] Invalid --max-iterations value: {parts[i + 1]}[/red]")
                return

    before_procedures = {}
    try:
        from aura.paths import SKILL_LIBRARY_DIR
        store = SkillStore(storage_path=str(SKILL_LIBRARY_DIR))
        target_ids = skill_ids if skill_ids else list(store.index.keys())
        for sid in target_ids:
            skill = store.load(sid)
            if skill:
                before_procedures[sid] = skill.procedure
    except (OSError, KeyError, TypeError, AttributeError):
        logger.debug("gepa_skill_snapshot_failed", exc_info=True)

    if dry_run:
        console.print("\n  [yellow][GEPA] Dry run — previewing evolution plan...[/yellow]")
    else:
        console.print(f"\n  [cyan][GEPA] Starting skill evolution (max {max_iterations} iterations)...[/cyan]")
        if skill_ids:
            console.print(f"  Target skills: [cyan]{', '.join(skill_ids)}[/cyan]")
        else:
            console.print("  [dim]Target: all skills in library[/dim]")

    try:
        result = run_evolution(
            skill_ids=skill_ids,
            config_overrides={"max_iterations": max_iterations},
            dry_run=dry_run,
        )
    except Exception as e:
        console.print(f"  [red][GEPA] Failed: {e}[/red]\n")
        return

    if result.get("error"):
        console.print(f"  [red][GEPA] Error: {result['error']}[/red]")
    elif result.get("dry_run"):
        console.print("\n  [bold][GEPA] Dry-run results:[/bold]")
        console.print(f"  Skills to evolve: [bold]{len(result.get('skills', []))}[/bold]")
        for sid in result.get("skills", []):
            name = store.index.get(sid, {}).get("name", sid) if store else sid
            console.print(f"    [dim]- {name} ({sid})[/dim]")
        config = result.get("config", {})
        console.print(f"  Max iterations: {config.get('max_iterations', '?')}")
        console.print(f"  Reflection model: [cyan]{config.get('reflection_model', '?')}[/cyan]")
        console.print(f"  Eval model: [cyan]{config.get('eval_model', '?')}[/cyan]")
    else:
        improvement = result.get("improvement", 0)
        seed_score = result.get("seed_score", 0)
        best_score = result.get("best_score", 0)
        updated = result.get("skills_updated", 0)

        console.print("\n  [green][GEPA] Evolution complete![/green]")
        console.print(f"  Score: {seed_score:.3f} -> [green]{best_score:.3f}[/green] ([green]+{improvement:.3f}[/green])")
        console.print(f"  Skills updated: [bold]{updated}[/bold]")
        console.print(f"  Iterations: {result.get('iterations', 0)}, Evals: {result.get('total_evals', 0)}")
        console.print(f"  Time: {result.get('duration_seconds', 0):.1f}s")
        console.print(f"  Stop reason: [dim]{result.get('stop_reason', 'N/A')}[/dim]")
        console.print(f"  Run saved to: [cyan]{result.get('run_dir', 'N/A')}[/cyan]")

        if updated > 0 and before_procedures:
            console.print("\n  [bold]--- Procedure diffs ---[/bold]")
            import difflib
            try:
                from aura.paths import SKILL_LIBRARY_DIR
                store_after = SkillStore(storage_path=str(SKILL_LIBRARY_DIR))
                for sid, old_proc in before_procedures.items():
                    skill_after = store_after.load(sid)
                    if not skill_after:
                        continue
                    new_proc = skill_after.procedure
                    if old_proc == new_proc:
                        continue
                    name = store_after.index.get(sid, {}).get("name", sid)
                    console.print(f"\n  [bold]{name}[/bold] (v{skill_after.metadata.version}):")
                    old_lines = old_proc.splitlines()
                    new_lines = new_proc.splitlines()
                    diff = difflib.unified_diff(
                        old_lines, new_lines,
                        fromfile=f"{name} (before)",
                        tofile=f"{name} (after)",
                        lineterm="",
                    )
                    diff_lines = list(diff)
                    if diff_lines:
                        for line in diff_lines[:40]:
                            prefix = "  "
                            if line.startswith("+") and not line.startswith("+++"):
                                prefix = "  + "
                            elif line.startswith("-") and not line.startswith("---"):
                                prefix = "  - "
                            elif line.startswith("@@"):
                                prefix = "  "
                            console.print(f"  {prefix}{line}")
                        if len(diff_lines) > 40:
                            console.print(f"    [dim]... ({len(diff_lines) - 40} more lines)[/dim]")
                    else:
                        console.print("    [dim](no textual changes)[/dim]")
            except (OSError, KeyError, TypeError, AttributeError) as e:
                console.print(f"  [dim](Could not generate diff: {e})[/dim]")

    console.print()
