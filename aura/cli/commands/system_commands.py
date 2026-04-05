import logging
from typing import Optional

from ..context import get_ctx

logger = logging.getLogger(__name__)


def handle_hook(agent, arg, context) -> Optional[str]:
    from ..display import console as _hook_console
    ctx = get_ctx()
    _hook_mgr = ctx.hook_manager if ctx else None
    if _hook_mgr is None:
        from ..hooks import HookManager
        _hook_mgr = HookManager()
        if ctx:
            ctx.hook_manager = _hook_mgr
    from ..hooks import render_hooks_table, HookEvent
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


def handle_mcp(agent, arg, context) -> Optional[str]:
    ctx = get_ctx()
    if ctx and ctx.agentic_loop and hasattr(ctx.agentic_loop, '_mcp_client'):
        mgr = ctx.agentic_loop._mcp_client
        if not mgr.connections:
            print("  No MCP servers connected. Configure in AURA.md under mcp_servers:")
        else:
            for name, conn in mgr.connections.items():
                print(f"  {name}: {len(conn.tools)} tools")
                for t in conn.tools[:5]:
                    print(f"    - {t['name']}: {t.get('description', '')[:60]}")
                if len(conn.tools) > 5:
                    print(f"    ... and {len(conn.tools) - 5} more")
    else:
        print("  No MCP servers connected. Configure in AURA.md under mcp_servers:")


def handle_audit(agent, arg, context) -> Optional[str]:
    _handle_audit_command(arg)


def handle_evolve(agent, arg, context) -> Optional[str]:
    _handle_evolve_command(agent, arg)


def _handle_audit_command(arg: str):
    from aura.security.audit_chain import get_audit_chain

    chain = get_audit_chain()
    parts = arg.strip().split(maxsplit=1)
    subcmd = parts[0].lower() if parts else "tail"

    if subcmd == "verify":
        print("  Verifying audit chain integrity...")
        valid, count, error = chain.verify()
        if valid:
            print(f"  VALID — {count} entries verified, chain intact.")
        else:
            print(f"  TAMPERED — integrity failure at entry {count}: {error}")

    elif subcmd == "tail":
        n = 15
        try:
            n = int(parts[1]) if len(parts) > 1 else 15
        except ValueError:
            pass
        entries = chain.tail(n)
        if not entries:
            print("  Audit chain is empty.")
            return
        print(f"\n  Last {len(entries)} audit entries:")
        from datetime import datetime
        for e in entries:
            ts = datetime.fromtimestamp(e.timestamp).strftime("%H:%M:%S")
            data_preview = e.action_data[:60] + "..." if len(e.action_data) > 60 else e.action_data
            print(f"  [{ts}] {e.action_type:<16} {e.agent_id:<12} {data_preview}")
        print(f"\n  Total entries: {chain.count()} | Chain hash: {entries[-1].entry_hash[:16]}...")

    elif subcmd == "count":
        print(f"  Audit chain: {chain.count()} entries")

    else:
        print("Usage: /audit <verify|tail|count> [args]")
        print("  /audit verify    — Verify chain integrity (detect tampering)")
        print("  /audit tail [n]  — Show last N entries (default 15)")
        print("  /audit count     — Show total entry count")


def _handle_evolve_command(agent, arg: str):
    try:
        from aura.evolution.runner import run_evolution
        from aura_skill_library.skill_store import SkillStore
    except ImportError as e:
        print(f"\n  [GEPA] Import error: {e}\n")
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
                print(f"  [GEPA] Invalid --max-iterations value: {parts[i + 1]}")
                return

    before_procedures = {}
    try:
        store = SkillStore(storage_path="./aura_data/skill_library")
        target_ids = skill_ids if skill_ids else list(store.index.keys())
        for sid in target_ids:
            skill = store.load(sid)
            if skill:
                before_procedures[sid] = skill.procedure
    except (OSError, KeyError, TypeError, AttributeError):
        logger.debug("gepa_skill_snapshot_failed", exc_info=True)

    if dry_run:
        print(f"\n  [GEPA] Dry run — previewing evolution plan...")
    else:
        print(f"\n  [GEPA] Starting skill evolution (max {max_iterations} iterations)...")
        if skill_ids:
            print(f"  Target skills: {', '.join(skill_ids)}")
        else:
            print(f"  Target: all skills in library")

    try:
        result = run_evolution(
            skill_ids=skill_ids,
            config_overrides={"max_iterations": max_iterations},
            dry_run=dry_run,
        )
    except Exception as e:
        print(f"  [GEPA] Failed: {e}\n")
        return

    if result.get("error"):
        print(f"  [GEPA] Error: {result['error']}")
    elif result.get("dry_run"):
        print(f"\n  [GEPA] Dry-run results:")
        print(f"  Skills to evolve: {len(result.get('skills', []))}")
        for sid in result.get("skills", []):
            name = store.index.get(sid, {}).get("name", sid) if store else sid
            print(f"    - {name} ({sid})")
        config = result.get("config", {})
        print(f"  Max iterations: {config.get('max_iterations', '?')}")
        print(f"  Reflection model: {config.get('reflection_model', '?')}")
        print(f"  Eval model: {config.get('eval_model', '?')}")
    else:
        improvement = result.get("improvement", 0)
        seed_score = result.get("seed_score", 0)
        best_score = result.get("best_score", 0)
        updated = result.get("skills_updated", 0)

        print(f"\n  [GEPA] Evolution complete!")
        print(f"  Score: {seed_score:.3f} -> {best_score:.3f} (+{improvement:.3f})")
        print(f"  Skills updated: {updated}")
        print(f"  Iterations: {result.get('iterations', 0)}, Evals: {result.get('total_evals', 0)}")
        print(f"  Time: {result.get('duration_seconds', 0):.1f}s")
        print(f"  Stop reason: {result.get('stop_reason', 'N/A')}")
        print(f"  Run saved to: {result.get('run_dir', 'N/A')}")

        if updated > 0 and before_procedures:
            print(f"\n  --- Procedure diffs ---")
            try:
                store_after = SkillStore(storage_path="./aura_data/skill_library")
                for sid, old_proc in before_procedures.items():
                    skill_after = store_after.load(sid)
                    if not skill_after:
                        continue
                    new_proc = skill_after.procedure
                    if old_proc == new_proc:
                        continue
                    name = store_after.index.get(sid, {}).get("name", sid)
                    print(f"\n  [{name}] (v{skill_after.metadata.version}):")
                    old_lines = old_proc.splitlines()
                    new_lines = new_proc.splitlines()
                    import difflib
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
                            print(f"  {prefix}{line}")
                        if len(diff_lines) > 40:
                            print(f"    ... ({len(diff_lines) - 40} more lines)")
                    else:
                        print(f"    (no textual changes)")
            except (OSError, KeyError, TypeError, AttributeError) as e:
                print(f"  (Could not generate diff: {e})")

    print()
