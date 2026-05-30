import logging
import os
import subprocess
from typing import Optional

from ..context import get_ctx
from .common import confirm_action
from .common import command, TIER_BETA, TIER_EXPERIMENTAL, TIER_STABLE

logger = logging.getLogger(__name__)


class ErrorTracker:
    """Track repeated tool errors for deduplication."""

    def __init__(self, max_recent: int = 5):
        self._recent: list[str] = []
        self._max = max_recent

    def record(self, error: str) -> str:
        """Record an error. Returns augmented message if repeated."""
        normalized = error.strip()[:200]
        count = self._recent.count(normalized)
        self._recent.append(normalized)
        if len(self._recent) > self._max:
            self._recent.pop(0)
        if count > 0:
            return f"{error}\n(This error has occurred {count + 1} times recently)"
        return error

    def clear(self):
        self._recent.clear()


def handle_export_session(agent, arg, context) -> "str | None":
    """Export the current session to a markdown or JSON file."""
    from ..context import get_ctx
    from ..display import show_error, show_info

    ctx = get_ctx()
    session_id = ""
    if ctx and ctx.session:
        session_id = getattr(ctx.session, 'session_id', '')

    if not session_id:
        show_error("No active session to export.")
        return None

    try:
        from ..activity_log import ActivityLog
        log = ActivityLog()
    except Exception as e:
        show_error(f"Activity log unavailable: {e}")
        return None

    fmt = "markdown"
    if arg and "--json" in arg:
        fmt = "json"

    content = log.export_session(session_id, format=fmt)
    if not content:
        show_error("No interactions found for this session.")
        return None

    import time
    from pathlib import Path
    ext = "json" if fmt == "json" else "md"
    filename = f"session_{session_id[:12]}_{int(time.time())}.{ext}"
    outpath = Path.cwd() / filename
    outpath.write_text(content, encoding="utf-8")
    show_info(f"Exported to {outpath.name} ({len(content)} chars)")
    return None


def _summarize_trace_event(event: dict) -> str:
    event_type = str(event.get("type", "event"))
    payload = event.get("payload", {}) if isinstance(event.get("payload"), dict) else {}

    if event_type == "tool_start":
        tool_name = payload.get("tool_name", "?")
        tool_args = payload.get("tool_args", {})
        desc = ""
        if isinstance(tool_args, dict):
            desc = (
                tool_args.get("path")
                or tool_args.get("command")
                or tool_args.get("query")
                or tool_args.get("pattern")
                or ""
            )
        desc_text = f" {str(desc)[:60]}" if desc else ""
        return f"tool start: {tool_name}{desc_text}"

    if event_type == "tool_result":
        tool_name = payload.get("tool_name", "?")
        result = payload.get("tool_result")
        result_text = str(result)
        if isinstance(result, str) and '"error"' in result.lower():
            return f"tool result: {tool_name} error"
        if "Permission denied by user" in result_text:
            return f"tool result: {tool_name} denied"
        return f"tool result: {tool_name} ok"

    if event_type == "response":
        text = str(payload.get("text", "")).strip().replace("\n", " ")
        return f"response: {text[:80]}" if text else "response"

    if event_type == "run_finished":
        status = payload.get("status", "unknown")
        model = payload.get("model", "")
        suffix = f" ({model})" if model else ""
        return f"run finished: {status}{suffix}"

    return event_type.replace("_", " ")


def _split_trace_runs(events: list[dict]) -> list[list[dict]]:
    runs: list[list[dict]] = []
    grouped_by_id: dict[str, list[dict]] = {}
    ordered_ids: list[str] = []
    for event in events:
        run_id = str(event.get("run_id", "")).strip()
        if run_id:
            if run_id not in grouped_by_id:
                grouped_by_id[run_id] = []
                ordered_ids.append(run_id)
            grouped_by_id[run_id].append(event)
    if ordered_ids:
        return [grouped_by_id[run_id] for run_id in ordered_ids]

    current: list[dict] = []
    for event in events:
        current.append(event)
        if event.get("type") == "run_finished":
            runs.append(current)
            current = []
    if current:
        runs.append(current)
    return runs


def _render_trace_events(console, session_id: str, events: list[dict], total_events: int, label: str) -> None:
    console.print(f"  Trace for {session_id} {label} ({len(events)}/{total_events} events)")
    for event in events:
        iteration = event.get("iteration", "-")
        summary = _summarize_trace_event(event)
        console.print(f"    [{iteration}] {summary}")


def _render_trace_run_summaries(console, session_id: str, runs: list[list[dict]], label: str) -> None:
    console.print(f"  Trace for {session_id} {label} ({len(runs)} runs)")
    for idx, run in enumerate(runs, start=1):
        finish = run[-1] if run and run[-1].get("type") == "run_finished" else {}
        run_id = str((finish or run[0]).get("run_id", "")).strip() if run else ""
        payload = finish.get("payload", {}) if isinstance(finish.get("payload"), dict) else {}
        status = payload.get("status", "in_progress")
        model = payload.get("model", "")
        tool_calls = payload.get("tool_calls", 0)
        response = str(payload.get("response", "")).strip().replace("\n", " ")
        suffix = f" ({model})" if model else ""
        run_tag = f" [{run_id}]" if run_id else ""
        summary = f"run {idx}{run_tag}: {status}{suffix}, {tool_calls} tool calls"
        if response:
            summary += f" -> {response[:70]}"
        console.print(f"    {summary}")


@command("/trace",    "Show structured session trace and run summaries", tier=TIER_BETA)


def handle_trace(agent, arg, context) -> Optional[str]:
    from ..display import console, show_error

    ctx = get_ctx()
    session = None
    if ctx and ctx.session:
        session = ctx.session
    elif ctx and ctx.agentic_loop and getattr(ctx.agentic_loop, "session", None):
        session = ctx.agentic_loop.session

    if not session:
        show_error("No active session trace available.")
        return

    events = list(getattr(session, "events", []) or [])
    if not events:
        show_error("No trace events recorded for this session yet.")
        return

    session_id = getattr(session, "session_id", "") or "session"
    arg = (arg or "").strip()

    if not arg:
        recent = events[-12:]
        _render_trace_events(console, session_id, recent, len(events), "recent")
        return

    if arg == "last":
        runs = _split_trace_runs(events)
        last_run = runs[-1] if runs else []
        if not last_run:
            show_error("No trace run available yet.")
            return
        _render_trace_events(console, session_id, last_run, len(events), "last run")
        return

    if arg == "runs":
        runs = _split_trace_runs(events)
        recent_runs = runs[-5:]
        _render_trace_run_summaries(console, session_id, recent_runs, "recent runs")
        return

    if arg == "failures":
        runs = _split_trace_runs(events)
        failed_runs = [
            run
            for run in runs
            if run
            and run[-1].get("type") == "run_finished"
            and str(run[-1].get("payload", {}).get("status", "completed")) != "completed"
        ]
        if not failed_runs:
            show_error("No failed runs in this session trace.")
            return
        _render_trace_run_summaries(console, session_id, failed_runs[-5:], "failed runs")
        return

    try:
        limit = max(1, min(50, int(arg)))
    except ValueError:
        show_error("Usage: /trace [count|last|runs|failures]")
        return

    recent = events[-limit:]
    _render_trace_events(console, session_id, recent, len(events), "recent")
    return


@command("/sessions", "Manage sessions",                               tier=TIER_STABLE)


def handle_sessions(agent, arg, context) -> Optional[str]:
    from aura.core.session import AgenticSession
    session_mgr = AgenticSession()
    agentic_sessions = session_mgr.list_sessions()
    brain_conversations = agent.brain.list_conversations()

    parts_arg = arg.split(maxsplit=1) if arg else []
    subcmd = parts_arg[0].lower() if parts_arg else ""

    if subcmd == "list":
        if not agentic_sessions and not brain_conversations:
            from ..display import console
            console.print("  No sessions found.")
            return
        from ..display import console
        from ..session_picker import _format_session_line
        ctx = get_ctx()
        current_sid = ""
        if ctx and ctx.session:
            current_sid = getattr(ctx.session, 'session_id', "") or ""
        all_sessions = agentic_sessions + brain_conversations
        console.print(f"  {'Session':<35} {'Msgs':>5}  {'Model':<15} {'Last Active':>8}")
        console.print(f"  {'-' * 72}")
        for s in all_sessions:
            is_current = s.get("id", "") == current_sid
            line = _format_session_line(s, is_current=is_current)
            marker = " *" if is_current else "  "
            console.print(f" {marker}{line}")
        console.print(f"\n  {len(all_sessions)} session(s) total.")
        return
    elif subcmd == "delete" and len(parts_arg) > 1:
        target = parts_arg[1]
        from ..display import console
        if session_mgr.delete(target):
            console.print(f"  Deleted session: {target}")
        else:
            console.print(f"  Session not found: {target}")
    elif subcmd == "export":
        return handle_export_session(agent, " ".join(parts_arg[1:]) if len(parts_arg) > 1 else "", context)
    elif subcmd == "new":
        ctx = get_ctx()
        if ctx and ctx.session:
            ctx.session.save()
        new_ses = AgenticSession()
        new_ses.new(project_root=os.getcwd())
        if ctx and ctx.agentic_loop:
            ctx.agentic_loop.session = new_ses
            ctx.agentic_loop.clear_history()
        if ctx:
            ctx.session = new_ses
        from ..display import console
        console.print("  Started new session.")
    else:
        if not agentic_sessions and not brain_conversations:
            from ..display import console
            console.print("  No sessions found.")
            return
        from ..display import console as _sessions_console
        from ..session_picker import pick_session
        ctx = get_ctx()
        current_sid = ""
        if ctx and ctx.session:
            current_sid = getattr(ctx.session, 'session_id', "") or ""
        all_sessions = agentic_sessions + brain_conversations
        result = pick_session(_sessions_console, all_sessions, current_sid)
        if result and "__action__" not in result:
            sid = result.get("id", "")
            if sid and ctx and ctx.agentic_loop:
                if ctx.agentic_loop.load_session(sid):
                    _sessions_console.print(f"  Switched to session: {result.get('title', 'Untitled')}")
                else:
                    _sessions_console.print(f"  Failed to load session: {sid}")
        elif result and result.get("__action__") == "delete":
            target_session = result.get("session", {})
            target_id = target_session.get("id", "")
            if target_id and session_mgr.delete(target_id):
                _sessions_console.print(f"  Deleted session: {target_session.get('title', target_id)}")
            else:
                _sessions_console.print("  Failed to delete session.")


@command("/clear",    "Clear conversation history",                      tier=TIER_STABLE)


def handle_clear(agent, arg, context) -> Optional[str]:
    if arg.strip() != "--force":
        from ..display import console
        approved = confirm_action(
            agent,
            "clear_history",
            {"scope": "conversation_history"},
            fallback_prompt="  Clear conversation history? [y/N] ",
        )
        if not approved:
            console.print("  [dim]Cancelled.[/dim]")
            return
    agent.brain.clear_history()
    ctx = get_ctx()
    if ctx and ctx.agentic_loop:
        ctx.agentic_loop.clear_history()
    from ..display import console
    console.print("Conversation history cleared.")


@command("/compact",  "Compact conversation history",                    tier=TIER_STABLE)


def handle_compact(agent, arg, context) -> Optional[str]:
    from ..display import console
    focus = arg if arg else None
    console.print("Compacting conversation history...")
    summary = agent.brain.compact_history(focus=focus)
    if summary:
        console.print(f"Compacted. Summary: {summary[:200]}...")
    else:
        console.print("Nothing to compact (history too short).")


def handle_retry(agent, arg, context) -> Optional[str]:
    """Re-run the last prompt, with optional tier escalation if the last run errored."""
    from ..display import show_error, show_info, show_response

    ctx = get_ctx()
    if not ctx or not ctx.agentic_loop:
        show_error("Retry not available outside chat mode.")
        return

    loop = ctx.agentic_loop
    last_prompt = None
    for msg in reversed(getattr(loop, "_conversation_history", []) or []):
        if msg.get("role") == "user":
            last_prompt = msg.get("content", "")
            break

    if not last_prompt:
        show_error("Nothing to retry — no previous prompt.")
        return

    if getattr(loop, "_loop_error", False):
        escalation_map = {"fast": "balanced", "balanced": "max"}
        current_tier = getattr(loop, "_current_tier", None) or getattr(loop, "tier", "balanced")
        if current_tier in escalation_map:
            next_tier = escalation_map[current_tier]
            approved = confirm_action(
                agent,
                "retry_tier_escalation",
                {"from_tier": current_tier, "to_tier": next_tier, "prompt": last_prompt},
                fallback_prompt=f"  Retry with '{next_tier}' tier? (y/n): ",
            )
            if approved:
                router = getattr(loop, "router", None)
                if router is not None:
                    router.tier = next_tier
                show_info(f"Escalated to '{next_tier}' tier.")

    show_info(f"Retrying: {last_prompt[:60]}...")
    try:
        result = loop.run(last_prompt)
    except Exception as exc:
        show_error(f"Retry failed: {exc}")
        return

    if result:
        response = result.get("response", "")
        if response:
            show_response(response, model=result.get("model", ""), stream=False)


@command("/context",  "Show context window usage",                       tier=TIER_STABLE)


def handle_context(agent, arg, context) -> Optional[str]:
    ctx = get_ctx()
    if ctx and ctx.agentic_loop:
        from rich.panel import Panel

        from ..context_bar import (
            build_context_breakdown,
            estimate_messages_tokens,
            estimate_tokens,
            get_context_limit,
        )
        from ..display import console as _context_console
        agentic_loop = ctx.agentic_loop
        tokens_used = estimate_messages_tokens(agentic_loop._conversation_history)
        token_limit = get_context_limit(agent.brain._model_override or "default")
        system_tokens = 0
        try:
            system_tokens = estimate_tokens(agentic_loop._build_system_prompt(""))
        except (TypeError, AttributeError, ValueError):
            logger.debug("context_sys_tokens_estimate_failed", exc_info=True)
        _context_console.print(Panel(
            build_context_breakdown(system_tokens, tokens_used, 0, token_limit),
            title="[bold cyan]Context Window[/bold cyan]",
            border_style="cyan",
        ))
    else:
        from ..display import console
        console.print("  Context tracking not available.")


@command("/cost",     "Show session cost breakdown",                     tier=TIER_STABLE)


def handle_cost(agent, arg, context) -> Optional[str]:
    from ..display import console
    stats = agent.brain.get_session_stats()
    console.print("\n  Session Cost:")
    console.print(f"    Input tokens:  {stats['input_tokens']:,}")
    console.print(f"    Output tokens: {stats['output_tokens']:,}")
    console.print(f"    Total tokens:  {stats['total_tokens']:,}")
    console.print(f"    Estimated cost: ${stats['cost_usd']:.4f}")
    console.print(f"    Queries: {stats['queries']}")
    console.print()


@command("/rewind",   "Rewind file changes to a checkpoint",             tier=TIER_STABLE)


def handle_rewind(agent, arg, context) -> Optional[str]:
    ctx = get_ctx()
    if ctx and ctx.agentic_loop and hasattr(ctx.agentic_loop, '_checkpoint_mgr'):
        from ..chat_loop import _rewind_picker
        from ..display import console as _rw_console
        _rewind_picker(ctx.agentic_loop._checkpoint_mgr, _rw_console)
    else:
        from ..display import console
        console.print("  No checkpoint manager available.")


# ── Conversation Forking Commands ────────────────────────────────────


def _get_tree(agent):
    """Get or create the ConversationTree attached to the agentic loop."""
    from ..context import get_ctx
    ctx = get_ctx()
    if not ctx or not ctx.agentic_loop:
        return None

    loop = ctx.agentic_loop
    if not hasattr(loop, '_conv_tree') or loop._conv_tree is None:
        from aura.core.conversation_fork import ConversationTree
        session_dir = None
        if loop.session and loop.session.session_id:
            session_dir = loop.session.sessions_dir / loop.session.session_id
        loop._conv_tree = ConversationTree(session_dir=session_dir)
        # Seed main branch with current history
        loop._conv_tree.branches["main"].history = list(loop._conversation_history)
    return loop._conv_tree


@command("/fork",    "Fork conversation into a new branch",             tier=TIER_EXPERIMENTAL)


def handle_fork(agent, arg, context) -> Optional[str]:
    """Fork current conversation into a new branch."""
    from ..display import console as _fork_console
    ctx = get_ctx()
    if not ctx or not ctx.agentic_loop:
        _fork_console.print("  Fork/checkout/merge requires an active chat session.")
        return

    tree = _get_tree(agent)
    if tree is None:
        _fork_console.print("  Fork not available outside chat mode.")
        return

    loop = ctx.agentic_loop
    # Sync current history into the tree before forking
    tree.sync_history(loop._conversation_history)

    name = arg.strip() if arg and arg.strip() else None
    parent_name = tree.get_current().name
    parent_msgs = len(tree.get_current().history)
    new_branch = tree.fork(name=name)

    # Point the agentic loop at the new branch's history
    loop._conversation_history = new_branch.history

    tree.save()
    _fork_console.print(
        f"  [green]Forked[/green] from [cyan]'{parent_name}'[/cyan] "
        f"at message {parent_msgs} -> [bold cyan]{new_branch.name}[/bold cyan] ({new_branch.id})"
    )


@command("/branches", "List conversation branches",                        tier=TIER_EXPERIMENTAL)


def handle_branches(agent, arg, context) -> Optional[str]:
    """List all conversation branches as a tree."""
    from ..display import console as _br_console

    tree = _get_tree(agent)
    if tree is None:
        _br_console.print("  Branches not available outside chat mode.")
        return

    # Sync current history
    ctx = get_ctx()
    loop = ctx.agentic_loop
    tree.sync_history(loop._conversation_history)

    branches = tree.list_branches()
    if not branches:
        _br_console.print("  No branches.")
        return

    # Render using git-log-graph ASCII style
    graph_lines = tree.render_tree_graph()
    _br_console.print()
    for line in graph_lines:
        _br_console.print(f"  {line}")
    _br_console.print()


@command("/checkout","Switch to a conversation branch",                 tier=TIER_EXPERIMENTAL)


def handle_checkout(agent, arg, context) -> Optional[str]:
    """Switch to a different conversation branch."""
    from ..display import console as _co_console
    ctx = get_ctx()
    if not ctx or not ctx.agentic_loop:
        _co_console.print("  Fork/checkout/merge requires an active chat session.")
        return

    tree = _get_tree(agent)
    if tree is None:
        _co_console.print("  Checkout not available outside chat mode.")
        return

    if not arg or not arg.strip():
        _co_console.print("  Usage: /checkout <branch-id|number>  (e.g. /checkout 1, /checkout main)")
        return

    loop = ctx.agentic_loop
    # Save current branch's history before switching
    tree.sync_history(loop._conversation_history)

    branch_id = arg.strip()
    try:
        branch = tree.switch(branch_id)
    except KeyError as e:
        _co_console.print(f"  {e}")
        return

    # Point the agentic loop at the target branch's history
    loop._conversation_history = branch.history

    tree.save()
    msg_count = len(branch.history)
    _co_console.print(
        f"  [green]Switched[/green] to [bold cyan]{branch.name}[/bold cyan] "
        f"({branch.id}, {msg_count} messages)"
    )


@command("/changes", "Show files modified in this session",              tier=TIER_BETA)


def handle_changes(agent, arg, context) -> Optional[str]:
    """Show files modified in the current session with diffs."""
    from ..context import get_ctx
    from ..display import console
    ctx = get_ctx()
    if not ctx or not ctx.agentic_loop:
        console.print("[dim]No active session.[/dim]")
        return

    hot_files = getattr(ctx.agentic_loop, '_hot_files', [])
    if not hot_files:
        console.print("[dim]No files modified in this session.[/dim]")
        return

    console.print(f"\n[bold]Files modified this session ({len(hot_files)}):[/bold]\n")
    for f in hot_files:
        basename = os.path.basename(f)
        # Guard against git argument injection: paths starting with - could be interpreted as flags
        safe_path = f if not f.startswith("-") else "./" + f
        try:
            result = subprocess.run(
                ["git", "diff", "HEAD~1", "--", safe_path],
                capture_output=True, text=True, timeout=5,
                cwd=os.path.dirname(f) or ".",
            )
            if result.stdout:
                adds = result.stdout.count('\n+') - result.stdout.count('\n+++')
                dels = result.stdout.count('\n-') - result.stdout.count('\n---')
                console.print(f"  [cyan]{basename}[/cyan] [green]+{adds}[/green] [red]-{dels}[/red]")
            else:
                console.print(f"  [cyan]{basename}[/cyan] [dim](no diff)[/dim]")
        except Exception:
            console.print(f"  [cyan]{basename}[/cyan]")
    console.print()


@command("/merge",   "Merge branch back to parent",                       tier=TIER_EXPERIMENTAL)


def handle_merge(agent, arg, context) -> Optional[str]:
    """Merge current branch back to parent."""
    from ..display import console as _mg_console
    ctx = get_ctx()
    if not ctx or not ctx.agentic_loop:
        _mg_console.print("  Fork/checkout/merge requires an active chat session.")
        return

    tree = _get_tree(agent)
    if tree is None:
        _mg_console.print("  Merge not available outside chat mode.")
        return

    loop = ctx.agentic_loop
    # Sync current history before merge
    tree.sync_history(loop._conversation_history)

    result = tree.merge_to_parent()

    if result.get("error"):
        _mg_console.print(f"  {result['error']}")
        return

    # Point the agentic loop at the parent's history (now current)
    parent_branch = tree.get_current()
    loop._conversation_history = parent_branch.history

    tree.save()
    merged_count = result["merged"]
    from_name = result["from"]
    target_name = result["target"]
    _mg_console.print(
        f"  [green]Merged[/green] {merged_count} messages from "
        f"[cyan]'{from_name}'[/cyan] into [bold cyan]'{target_name}'[/bold cyan]"
    )
