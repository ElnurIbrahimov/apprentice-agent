import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def handle_sessions(agent, arg, context) -> Optional[str]:
    from aura.core.session import AgenticSession
    session_mgr = AgenticSession()
    agentic_sessions = session_mgr.list_sessions()
    brain_conversations = agent.brain.list_conversations()

    parts_arg = arg.split(maxsplit=1) if arg else []
    subcmd = parts_arg[0].lower() if parts_arg else ""

    if subcmd == "list":
        if not agentic_sessions and not brain_conversations:
            print("  No sessions found.")
            return
        from ..session_picker import _format_session_line
        current_sid = ""
        if hasattr(agent, '_agentic_session') and agent._agentic_session:
            current_sid = getattr(agent._agentic_session, 'session_id', "") or ""
        all_sessions = agentic_sessions + brain_conversations
        print(f"  {'Session':<35} {'Msgs':>5}  {'Model':<15} {'Last Active':>8}")
        print(f"  {'-' * 72}")
        for s in all_sessions:
            is_current = s.get("id", "") == current_sid
            line = _format_session_line(s, is_current=is_current)
            marker = " *" if is_current else "  "
            print(f" {marker}{line}")
        print(f"\n  {len(all_sessions)} session(s) total.")
        return
    elif subcmd == "delete" and len(parts_arg) > 1:
        target = parts_arg[1]
        if session_mgr.delete(target):
            print(f"  Deleted session: {target}")
        else:
            print(f"  Session not found: {target}")
    elif subcmd == "new":
        if hasattr(agent, '_agentic_session'):
            agent._agentic_session.save()
        new_ses = AgenticSession()
        new_ses.new(project_root=os.getcwd())
        if hasattr(agent, '_agentic_loop'):
            agent._agentic_loop.session = new_ses
            agent._agentic_loop.clear_history()
        agent._agentic_session = new_ses
        print("  Started new session.")
    else:
        if not agentic_sessions and not brain_conversations:
            print("  No sessions found.")
            return
        from ..session_picker import pick_session
        from ..display import console as _sessions_console
        current_sid = ""
        if hasattr(agent, '_agentic_session') and agent._agentic_session:
            current_sid = getattr(agent._agentic_session, 'session_id', "") or ""
        all_sessions = agentic_sessions + brain_conversations
        result = pick_session(_sessions_console, all_sessions, current_sid)
        if result and "__action__" not in result:
            sid = result.get("id", "")
            if sid and hasattr(agent, '_agentic_loop'):
                if agent._agentic_loop.load_session(sid):
                    print(f"  Switched to session: {result.get('title', 'Untitled')}")
                else:
                    print(f"  Failed to load session: {sid}")
        elif result and result.get("__action__") == "delete":
            target_session = result.get("session", {})
            target_id = target_session.get("id", "")
            if target_id and session_mgr.delete(target_id):
                print(f"  Deleted session: {target_session.get('title', target_id)}")
            else:
                print(f"  Failed to delete session.")


def handle_clear(agent, arg, context) -> Optional[str]:
    if arg.strip() != "--force":
        from ..display import console
        response = console.input("  Clear conversation history? [y/N] ").strip().lower()
        if response not in ("y", "yes"):
            console.print("  [dim]Cancelled.[/dim]")
            return
    agent.brain.clear_history()
    if hasattr(agent, '_agentic_loop'):
        agent._agentic_loop.clear_history()
    print("Conversation history cleared.")


def handle_compact(agent, arg, context) -> Optional[str]:
    focus = arg if arg else None
    print("Compacting conversation history...")
    summary = agent.brain.compact_history(focus=focus)
    if summary:
        print(f"Compacted. Summary: {summary[:200]}...")
    else:
        print("Nothing to compact (history too short).")


def handle_retry(agent, arg, context) -> Optional[str]:
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
    if hasattr(agent, '_agentic_loop'):
        from ..display import console as _context_console
        from ..context_bar import estimate_messages_tokens, get_context_limit, build_context_breakdown, estimate_tokens
        from rich.panel import Panel
        agentic_loop = agent._agentic_loop
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
        print("  Context tracking not available.")


def handle_cost(agent, arg, context) -> Optional[str]:
    stats = agent.brain.get_session_stats()
    print(f"\n  Session Cost:")
    print(f"    Input tokens:  {stats['input_tokens']:,}")
    print(f"    Output tokens: {stats['output_tokens']:,}")
    print(f"    Total tokens:  {stats['total_tokens']:,}")
    print(f"    Estimated cost: ${stats['cost_usd']:.4f}")
    print(f"    Queries: {stats['queries']}")
    print()


def handle_rewind(agent, arg, context) -> Optional[str]:
    if hasattr(agent, '_agentic_loop') and hasattr(agent._agentic_loop, '_checkpoint_mgr'):
        from ..display import console as _rw_console
        from ..chat_loop import _rewind_picker
        _rewind_picker(agent._agentic_loop._checkpoint_mgr, _rw_console)
    else:
        print("  No checkpoint manager available.")


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
    if not hasattr(agent, '_agentic_loop') or not agent._agentic_loop:
        print("  Fork/checkout/merge requires an active chat session.")
        return

    from ..display import console as _fork_console

    tree = _get_tree(agent)
    if tree is None:
        print("  Fork not available outside chat mode.")
        return

    loop = agent._agentic_loop
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
        print("  Branches not available outside chat mode.")
        return

    # Sync current history
    loop = agent._agentic_loop
    tree.sync_history(loop._conversation_history)

    branches = tree.list_branches()
    if not branches:
        print("  No branches.")
        return

    _br_console.print()
    current_id = tree.current_branch

    def _render_branch(branch, prefix="", is_last=True):
        """Render a single branch line with tree connectors."""
        is_current = branch.id == current_id
        marker = "[bold green]*[/bold green] " if is_current else "  "
        current_label = " [bold green]<- current[/bold green]" if is_current else ""
        connector = "\\-- " if is_last else "|-- "
        msg_count = len(branch.history)

        if branch.id == "main":
            _br_console.print(f"  {marker}[bold]{branch.name}[/bold] ({msg_count} messages){current_label}")
        else:
            fork_new = msg_count - branch.fork_point
            name_display = f'"{branch.name}"' if branch.name != branch.id else ""
            _br_console.print(
                f"  {prefix}{connector}{marker}[cyan]{branch.id}[/cyan] "
                f"{name_display} ({msg_count} msgs, +{fork_new} since fork){current_label}"
            )

        children = tree.get_children(branch.id)
        for i, child in enumerate(children):
            child_is_last = (i == len(children) - 1)
            child_prefix = prefix + ("    " if is_last else "|   ")
            _render_branch(child, child_prefix, child_is_last)

    # Start from main
    main = tree.branches.get("main")
    if main:
        _render_branch(main)

    _br_console.print()


def handle_checkout(agent, arg, context) -> Optional[str]:
    """Switch to a different conversation branch."""
    if not hasattr(agent, '_agentic_loop') or not agent._agentic_loop:
        print("  Fork/checkout/merge requires an active chat session.")
        return

    from ..display import console as _co_console

    tree = _get_tree(agent)
    if tree is None:
        print("  Checkout not available outside chat mode.")
        return

    if not arg or not arg.strip():
        print("  Usage: /checkout <branch-id|number>  (e.g. /checkout 1, /checkout main)")
        return

    loop = agent._agentic_loop
    # Save current branch's history before switching
    tree.sync_history(loop._conversation_history)

    branch_id = arg.strip()
    try:
        branch = tree.switch(branch_id)
    except KeyError as e:
        print(f"  {e}")
        return

    # Point the agentic loop at the target branch's history
    loop._conversation_history = branch.history

    tree.save()
    msg_count = len(branch.history)
    _co_console.print(
        f"  [green]Switched[/green] to [bold cyan]{branch.name}[/bold cyan] "
        f"({branch.id}, {msg_count} messages)"
    )


def handle_merge(agent, arg, context) -> Optional[str]:
    """Merge current branch back to parent."""
    if not hasattr(agent, '_agentic_loop') or not agent._agentic_loop:
        print("  Fork/checkout/merge requires an active chat session.")
        return

    from ..display import console as _mg_console

    tree = _get_tree(agent)
    if tree is None:
        print("  Merge not available outside chat mode.")
        return

    loop = agent._agentic_loop
    # Sync current history before merge
    tree.sync_history(loop._conversation_history)

    result = tree.merge_to_parent()

    if result.get("error"):
        print(f"  {result['error']}")
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
