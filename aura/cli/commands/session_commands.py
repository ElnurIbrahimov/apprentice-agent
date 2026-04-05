import os
import logging
from typing import Optional

from ..context import get_ctx

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
    from ..display import console, show_info, show_error
    from ..context import get_ctx

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

    from pathlib import Path
    import time
    ext = "json" if fmt == "json" else "md"
    filename = f"session_{session_id[:12]}_{int(time.time())}.{ext}"
    outpath = Path.cwd() / filename
    outpath.write_text(content, encoding="utf-8")
    show_info(f"Exported to {outpath.name} ({len(content)} chars)")
    return None


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
        from ..session_picker import _format_session_line
        from ..display import console
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
        from ..session_picker import pick_session
        from ..display import console as _sessions_console
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
                _sessions_console.print(f"  Failed to delete session.")


def handle_clear(agent, arg, context) -> Optional[str]:
    if arg.strip() != "--force":
        from ..display import console
        response = console.input("  Clear conversation history? [y/N] ").strip().lower()
        if response not in ("y", "yes"):
            console.print("  [dim]Cancelled.[/dim]")
            return
    agent.brain.clear_history()
    ctx = get_ctx()
    if ctx and ctx.agentic_loop:
        ctx.agentic_loop.clear_history()
    from ..display import console
    console.print("Conversation history cleared.")


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
    """Re-run the last prompt with optional model tier escalation."""
    from ..context import get_ctx
    from ..display import show_info as _retry_info, show_error as _retry_error, show_response as _retry_response

    ctx = get_ctx()
    if not ctx or not ctx.agentic_loop:
        _retry_error("Retry not available outside chat mode.")
        return

    loop = ctx.agentic_loop
    # Find last user message in conversation history
    last_prompt = None
    for msg in reversed(loop._conversation_history):
        if msg.get("role") == "user":
            last_prompt = msg.get("content", "")
            break

    if not last_prompt:
        _retry_error("Nothing to retry — no previous prompt.")
        return

    # Check if last run errored and suggest tier escalation
    had_error = getattr(loop, '_loop_error', False)
    if had_error:
        escalation_map = {"fast": "balanced", "balanced": "max"}
        current_tier = getattr(loop, '_current_tier', None) or getattr(loop, 'tier', 'balanced')
        if current_tier in escalation_map:
            next_tier = escalation_map[current_tier]
            try:
                from ..display import show_warning as _retry_warn
                _retry_warn(f"Last attempt failed on '{current_tier}' tier.")
            except ImportError:
                _retry_info(f"Last attempt failed on '{current_tier}' tier.")

            try:
                choice = input(f"  Retry with '{next_tier}' tier? (y/n): ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                return

            if choice in ("y", "yes"):
                # Temporarily override the tier for this retry
                if hasattr(loop, 'router') and loop.router:
                    loop.router.tier = next_tier
                _retry_info(f"Escalated to '{next_tier}' tier.")

    _retry_info(f"Retrying: {last_prompt[:60]}...")
    try:
        result = loop.run(last_prompt)
    except Exception as exc:  # Catch-all: protect CLI from agentic loop crash on retry
        _retry_error(f"Retry failed: {exc}")
        return

    if result:
        response = result.get("response", "")
        if response:
            _retry_response(response, model=result.get("model", ""), stream=False)


def handle_context(agent, arg, context) -> Optional[str]:
    ctx = get_ctx()
    if ctx and ctx.agentic_loop:
        from ..display import console as _context_console
        from ..context_bar import estimate_messages_tokens, get_context_limit, build_context_breakdown, estimate_tokens
        from rich.panel import Panel
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


def handle_cost(agent, arg, context) -> Optional[str]:
    from ..display import console
    stats = agent.brain.get_session_stats()
    console.print(f"\n  Session Cost:")
    console.print(f"    Input tokens:  {stats['input_tokens']:,}")
    console.print(f"    Output tokens: {stats['output_tokens']:,}")
    console.print(f"    Total tokens:  {stats['total_tokens']:,}")
    console.print(f"    Estimated cost: ${stats['cost_usd']:.4f}")
    console.print(f"    Queries: {stats['queries']}")
    console.print()


def handle_rewind(agent, arg, context) -> Optional[str]:
    ctx = get_ctx()
    if ctx and ctx.agentic_loop and hasattr(ctx.agentic_loop, '_checkpoint_mgr'):
        from ..display import console as _rw_console
        from ..chat_loop import _rewind_picker
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
        import subprocess, os
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
