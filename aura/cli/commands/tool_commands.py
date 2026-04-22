import logging
import os
from pathlib import Path
from typing import Optional

from ._permissions import confirm_action

logger = logging.getLogger(__name__)


def handle_shell(agent, arg, context) -> Optional[str]:
    _handle_shell_command(agent, arg)


def handle_grep(agent, arg, context) -> Optional[str]:
    _handle_grep_command(agent, arg)


def handle_search(agent, arg, context) -> Optional[str]:
    _handle_search_command(agent, arg)


def handle_edit(agent, arg, context) -> Optional[str]:
    _handle_edit_command(agent, arg)


def handle_test(agent, arg, context) -> Optional[str]:
    from ..display import console as _test_console
    from ..display import show_response as _test_show
    from ..display import show_tool_call as _test_tool
    from ..test_runner import render_test_results, run_tests
    test_cmd = arg.strip()
    if not test_cmd:
        try:
            from aura.core.context import get_aura_md_config as _get_test_cfg
            _test_cfg = _get_test_cfg(os.getcwd())
            test_cmd = _test_cfg.get("test_cmd", "") if _test_cfg else ""
        except (ImportError, OSError, KeyError, TypeError):
            logger.debug("test_cmd_config_lookup_failed", exc_info=True)
            test_cmd = ""
    if not test_cmd:
        _test_console.print("[dim]No test command configured. Usage: /test <command> or set test_cmd in AURA.md[/dim]")
        return
    _test_console.print(f"[dim]Running: {test_cmd}[/dim]")
    result = run_tests(test_cmd)
    render_test_results(_test_console, result)
    if not result.success and result.failures:
        approved = confirm_action(
            agent,
            "auto_fix_tests",
            {"command": test_cmd, "failure_count": len(result.failures)},
            fallback_prompt="  Auto-fix failing tests? (y/n): ",
        )
        if approved:
            from aura.cli.context import get_ctx
            ctx = get_ctx()
            agentic_loop = ctx.agentic_loop if ctx else None
            if agentic_loop:
                fix_prompt = f"These tests failed:\n{result.output[-2000:]}\n\nFix the failing tests."
                def _test_on_tool_call(name, args_t, _result):
                    desc = args_t.get("path") or args_t.get("pattern") or args_t.get("query") or ""
                    if not desc and "command" in args_t:
                        desc = args_t["command"][:60]
                    _test_tool(name, str(desc))
                fix_result = agentic_loop.run(fix_prompt, on_tool_call=_test_on_tool_call)
                _test_show(fix_result.get("response", ""), agent.brain._model_override or "auto")


def handle_project(agent, arg, context) -> Optional[str]:
    _handle_project_command(agent, arg)


def handle_watch(agent, arg, context) -> Optional[str]:
    """`/watch [start|stop|status|clear]` — file watcher for AURA:/AI: comments.

    Subcommands:
      /watch start [path]   Begin monitoring (default path: cwd)
      /watch stop           Stop the active watcher
      /watch status         Show running state and unresolved hits
      /watch clear          Mark all current hits as resolved
      /watch                Shortcut — start if stopped, status if running
    """
    from ..context import get_ctx
    from ..display import console as _watch_console
    from ..watch_mode import FileWatcher

    ctx = get_ctx()
    watcher = ctx.file_watcher if ctx else None

    tokens = (arg or "").strip().split(maxsplit=1)
    sub = tokens[0].lower() if tokens else ""
    rest = tokens[1] if len(tokens) > 1 else ""

    def _ensure_watcher():
        nonlocal watcher
        if watcher is None:
            watcher = FileWatcher()
            if ctx:
                ctx.file_watcher = watcher
        return watcher

    def _do_start(path_arg: str):
        w = _ensure_watcher()
        if w.is_running:
            _watch_console.print("[dim]Watch already running. Use /watch stop first.[/dim]")
            return

        def _on_watch_hit(hit):
            _watch_console.print(
                f"[magenta]Detected:[/magenta] "
                f"[cyan]{Path(hit.file_path).name}:{hit.line_number}[/cyan] {hit.instruction}"
            )

        w.set_callback(_on_watch_hit)
        if path_arg:
            from ._util import resolve_user_path
            try:
                w.root = str(resolve_user_path(path_arg))  # FileWatcher supports attribute reassignment
            except AttributeError:
                pass
        hits = w.scan_all()
        if hits:
            _watch_console.print(f"[magenta]Found {len(hits)} AI comments:[/magenta]")
            for h in hits[:5]:
                _watch_console.print(f"  [cyan]{Path(h.file_path).name}:{h.line_number}[/cyan] {h.instruction}")
        w.start()
        _watch_console.print("[dim]Watch mode started. Monitoring for AURA: and AI: comments.[/dim]")

    def _do_stop():
        if watcher is None or not watcher.is_running:
            _watch_console.print("[dim]Watch mode is not running.[/dim]")
            return
        watcher.stop()
        _watch_console.print("[dim]Watch mode stopped.[/dim]")

    def _do_status():
        if watcher is None or not watcher.is_running:
            _watch_console.print("[dim]Watch mode stopped. Start with /watch start[/dim]")
            return
        hits = watcher.get_unresolved()
        if hits:
            _watch_console.print(f"[magenta]Watching -- {len(hits)} unresolved comments:[/magenta]")
            for h in hits[:10]:
                fname = Path(h.file_path).name
                _watch_console.print(f"  [cyan]{fname}:{h.line_number}[/cyan] {h.instruction}")
        else:
            _watch_console.print("[dim]Watching -- no AI comments detected.[/dim]")

    def _do_clear():
        if watcher is None:
            _watch_console.print("[dim]No watcher to clear.[/dim]")
            return
        # mark_resolved takes a specific hit; iterate all outstanding ones.
        for h in watcher.get_unresolved():
            try:
                watcher.mark_resolved(h)
            except Exception:
                logger.debug("watch_mark_resolved_failed", exc_info=True)
        _watch_console.print("[dim]Cleared unresolved hits.[/dim]")

    if sub == "start":
        _do_start(rest)
    elif sub == "stop":
        _do_stop()
    elif sub == "status":
        _do_status()
    elif sub == "clear":
        _do_clear()
    elif sub == "":
        # Bare /watch toggles: start if stopped, status if running
        if watcher is None or not watcher.is_running:
            _do_start("")
        else:
            _do_status()
    else:
        _watch_console.print(
            "[yellow]Usage: /watch [start|stop|status|clear] [path][/yellow]"
        )


def handle_undo(agent, arg, context) -> Optional[str]:
    from ..context import get_ctx
    from ..display import console
    ctx = get_ctx()
    if not (ctx and ctx.agentic_loop):
        console.print("  No active agentic loop.")
        return None

    tool_exec = ctx.agentic_loop.executor
    cp_mgr = getattr(tool_exec, "_checkpoint_mgr", None)

    # If user passed a checkpoint id (cp_...), restore that exact one
    if arg and arg.startswith("cp_") and cp_mgr:
        ok = cp_mgr.restore(arg)
        console.print(f"  Restored checkpoint {arg}" if ok else f"  Checkpoint {arg} not found")
        return None

    # If user passed a path, use .bak rollback (file-specific)
    if arg and not arg.startswith("cp_"):
        result = tool_exec.code_edit.rollback(arg)
        if result.get("success"):
            console.print(f"  Rolled back: {result.get('restored', arg)}")
            if cp_mgr:
                cp_mgr.snapshot(arg, label=f"post-undo {Path(arg).name}")
        else:
            console.print(f"  Error: {result.get('error', 'Unknown error')}")
        return None

    # No arg: prefer CheckpointManager (persistent) over _last_backups
    if cp_mgr:
        checkpoints = cp_mgr.list_checkpoints()
        if checkpoints:
            latest = checkpoints[0]
            # Snapshot current state to the redo stack before undoing
            _redo_stack = getattr(tool_exec, "_redo_stack", None)
            if _redo_stack is None:
                _redo_stack = []
                tool_exec._redo_stack = _redo_stack
            current_files = [f["original_path"] for f in latest["files"] if f.get("original_exists")]
            if current_files:
                redo_cp = cp_mgr.snapshot_multi(current_files, label=f"pre-redo-of {latest['id']}")
                _redo_stack.append(redo_cp)
            ok = cp_mgr.restore(latest["id"])
            if ok:
                files_str = ", ".join(Path(f["original_path"]).name for f in latest["files"][:3])
                suffix = f" (+{len(latest['files']) - 3} more)" if len(latest["files"]) > 3 else ""
                console.print(f"  Restored checkpoint {latest['id']}: {files_str}{suffix}")
            else:
                console.print(f"  Restore failed for {latest['id']}")
            return None

    # Fallback: legacy .bak
    backups = tool_exec.code_edit._last_backups
    if backups:
        last_path = list(backups.keys())[-1]
        result = tool_exec.code_edit.rollback(last_path)
        if result.get("success"):
            console.print(f"  Rolled back: {last_path}")
        else:
            console.print(f"  Error: {result.get('error', 'Unknown error')}")
    else:
        console.print("  No edits to undo.")


def handle_redo(agent, arg, context) -> Optional[str]:
    """Reverse the last /undo by restoring the snapshot taken before the undo."""
    from ..context import get_ctx
    from ..display import console
    ctx = get_ctx()
    if not (ctx and ctx.agentic_loop):
        console.print("  No active agentic loop.")
        return None

    tool_exec = ctx.agentic_loop.executor
    cp_mgr = getattr(tool_exec, "_checkpoint_mgr", None)
    if cp_mgr is None:
        console.print("  /redo requires the checkpoint manager.")
        return None

    _redo_stack = getattr(tool_exec, "_redo_stack", None)
    if not _redo_stack:
        console.print("  Nothing to redo.")
        return None

    cp_id = _redo_stack.pop()
    ok = cp_mgr.restore(cp_id)
    console.print(f"  Redid: {cp_id}" if ok else f"  Redo failed ({cp_id} missing)")


def _handle_shell_command(agent, arg: str):
    from ..display import console as _shell_console
    if not arg:
        _shell_console.print("Usage: /shell <command>")
        _shell_console.print("  /shell git status")
        _shell_console.print("  /run npm test")
        _shell_console.print("  /bash ls -la")
        return

    try:
        if not confirm_action(
            agent,
            "shell",
            {"command": arg, "cwd": os.getcwd()},
            fallback_prompt=f"  Execute shell command: {arg}\n  Confirm? (y/n): ",
        ):
            _shell_console.print("  Cancelled.")
            return
    except (AttributeError, EOFError, KeyboardInterrupt, OSError):
        logger.debug("shell_permission_check_skipped", exc_info=True)

    tool = agent.tools.get("shell_executor")
    if not tool:
        from aura.tools.shell_executor import ShellExecutorTool
        tool = ShellExecutorTool()
        agent.tools["shell_executor"] = tool

    def on_line(line):
        _shell_console.print(f"  {line}")

    result = tool.run_streaming(command=arg, on_output=on_line)

    if not result.get("success"):
        error = result.get("error", "")
        if error:
            _shell_console.print(f"\n  Error: {error}")
    _shell_console.print(f"\n  \\[exit {result.get('exit_code', '?')}] ({result.get('elapsed', '?')}s)")


def _handle_grep_command(agent, arg: str):
    from ..display import console as _grep_console
    if not arg:
        _grep_console.print("Usage: /grep <pattern> \\[path]")
        _grep_console.print("  /grep 'def my_func'")
        _grep_console.print("  /grep 'import os' ./src")
        _grep_console.print("  /grep 'TODO' --type py")
        return

    tool = agent.tools.get("code_search")
    if not tool:
        from aura.tools.code_search import CodeSearchTool
        tool = CodeSearchTool()
        agent.tools["code_search"] = tool

    import shlex as _shlex
    try:
        parts = _shlex.split(arg, posix=True)
    except ValueError:
        # Unbalanced quotes: fall back to whitespace split rather than
        # crashing on the user. They'll see the broken pattern and retry.
        parts = arg.split()
    if not parts:
        _grep_console.print("Usage: /grep <pattern> \\[path]")
        return
    pattern = parts[0]
    path = "."

    file_type = None
    case_insensitive = False
    context = 0
    i = 1
    while i < len(parts):
        if parts[i] == "--type" and i + 1 < len(parts):
            file_type = parts[i + 1]
            i += 2
        elif parts[i] == "-i":
            case_insensitive = True
            i += 1
        elif parts[i] == "-C" and i + 1 < len(parts):
            context = int(parts[i + 1])
            i += 2
        else:
            from ._util import resolve_user_path
            path = str(resolve_user_path(parts[i]))
            i += 1

    result = tool.grep(
        pattern=pattern, path=path, file_type=file_type,
        case_insensitive=case_insensitive, context_lines=context,
    )

    if not result.get("success"):
        _grep_console.print(f"  Error: {result.get('error')}")
        return

    matches = result.get("matches", [])
    total = result.get("total_matches", 0)
    _grep_console.print(f"\n  {total} matches in {result.get('files_searched', 0)} files:\n")
    for m in matches[:50]:
        _grep_console.print(f"  {m['file']}:{m['line']}\t{m['text']}")
        for ctx in m.get("before", []):
            _grep_console.print(f"    {ctx}")
        for ctx in m.get("after", []):
            _grep_console.print(f"    {ctx}")
    if total > 50:
        _grep_console.print(f"\n  ... and {total - 50} more matches")


def _handle_search_command(agent, arg: str):
    from ..display import console as _search_console
    if not arg:
        _search_console.print("Usage: /search <glob-pattern>  or  /find def <name>")
        _search_console.print("  /search '*.py'")
        _search_console.print("  /find def MyClass")
        _search_console.print("  /search structure")
        return

    tool = agent.tools.get("code_search")
    if not tool:
        from aura.tools.code_search import CodeSearchTool
        tool = CodeSearchTool()
        agent.tools["code_search"] = tool

    parts = arg.split(maxsplit=1)
    subcmd = parts[0].lower()

    if subcmd == "def" or subcmd == "definition":
        name = parts[1] if len(parts) > 1 else ""
        if not name:
            _search_console.print("Usage: /find def <name>")
            return
        result = tool.find_definition(name=name)
        if result.get("success"):
            defs = result.get("definitions", [])
            _search_console.print(f"\n  Found {len(defs)} definition(s) of '{name}':\n")
            for d in defs:
                _search_console.print(f"  {d['file']}:{d['line']} ({d['kind']})")
                _search_console.print(f"    {d['text']}")
        else:
            _search_console.print(f"  Error: {result.get('error')}")

    elif subcmd == "ref" or subcmd == "references":
        name = parts[1] if len(parts) > 1 else ""
        if not name:
            _search_console.print("Usage: /find ref <name>")
            return
        result = tool.find_references(name=name)
        if result.get("success"):
            refs = result.get("references", [])
            _search_console.print(f"\n  Found {len(refs)} reference(s) to '{name}':\n")
            for r in refs[:30]:
                _search_console.print(f"  {r['file']}:{r['line']}\t{r['text']}")
        else:
            _search_console.print(f"  Error: {result.get('error')}")

    elif subcmd == "structure" or subcmd == "tree":
        from ._util import resolve_user_path
        raw_path = parts[1] if len(parts) > 1 else "."
        path = str(resolve_user_path(raw_path))
        result = tool.project_structure(path=path)
        if result.get("success"):
            _search_console.print(f"\n{result['tree']}")
            s = result.get("stats", {})
            _search_console.print(f"\n  {s.get('files', 0)} files, {s.get('dirs', 0)} dirs")
        else:
            _search_console.print(f"  Error: {result.get('error')}")

    else:
        result = tool.glob(pattern=arg)
        if result.get("success"):
            files = result.get("files", [])
            _search_console.print(f"\n  Found {result.get('total', 0)} files:\n")
            for f in files[:50]:
                size = f.get("size", 0)
                size_str = f"{size // 1024}KB" if size > 1024 else f"{size}B"
                _search_console.print(f"  {f['path']}  ({size_str})")
            if result.get("truncated"):
                _search_console.print(f"\n  ... truncated ({result['total']} total)")
        else:
            _search_console.print(f"  Error: {result.get('error')}")


def _handle_edit_command(agent, arg: str):
    from ..display import console as _edit_console
    if not arg:
        _edit_console.print("Usage: /edit <file-path> \\[line-offset]")
        _edit_console.print("  /edit src/main.py")
        _edit_console.print("  /edit src/main.py 100")
        return

    tool = agent.tools.get("code_edit")
    if not tool:
        from aura.tools.code_edit import CodeEditTool
        tool = CodeEditTool()
        agent.tools["code_edit"] = tool

    from ._util import resolve_user_path
    parts = arg.split()
    path = str(resolve_user_path(parts[0]))
    offset = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
    limit = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 100

    result = tool.read_file(path=path, offset=offset, limit=limit)
    if result.get("success"):
        _edit_console.print(f"\n  {result['showing']}  ({result['path']})\n")
        _edit_console.print(result["content"])
    else:
        _edit_console.print(f"  Error: {result.get('error')}")


def _handle_project_command(agent, arg: str):
    from ..display import console as _proj_console
    tool = agent.tools.get("code_search")
    if not tool:
        from aura.tools.code_search import CodeSearchTool
        tool = CodeSearchTool()
        agent.tools["code_search"] = tool

    parts = arg.split(maxsplit=1) if arg else ["info"]
    subcmd = parts[0].lower()

    if subcmd == "init":
        from aura.tools.project_context import init_project
        path = parts[1] if len(parts) > 1 else "."
        _proj_console.print(init_project(path))

    elif subcmd == "detect" or subcmd == "info":
        path = parts[1] if len(parts) > 1 else "."
        result = tool.detect_project_type(path=path)
        if result.get("success"):
            _proj_console.print(f"\n  Project Type:     {result.get('project_type', 'unknown')}")
            _proj_console.print(f"  Language:         {result.get('language', 'N/A')}")
            _proj_console.print(f"  Stack:            {', '.join(result.get('stack', [])) or 'N/A'}")
            _proj_console.print(f"  Frameworks:       {', '.join(result.get('frameworks', [])) or 'N/A'}")
            _proj_console.print(f"  Package Manager:  {result.get('package_manager', 'N/A')}")
            _proj_console.print(f"  Key Files:        {', '.join(result.get('key_files', [])) or 'N/A'}")
        else:
            _proj_console.print(f"  Error: {result.get('error')}")

    elif subcmd == "context":
        from aura.tools.project_context import load_project_context
        path = parts[1] if len(parts) > 1 else None
        ctx = load_project_context(path)
        if ctx:
            _proj_console.print(f"\n{ctx}")
        else:
            _proj_console.print("  No AURA.md found. Create one with: /project init")

    elif subcmd == "index":
        path = parts[1] if len(parts) > 1 else "."
        from aura.tools.codebase_index import CodebaseIndex
        idx = CodebaseIndex(path)
        def on_progress(current, total, fpath):
            if current % 20 == 0 or current == total:
                _proj_console.print(f"  \\[{current}/{total}] {fpath}")
        _proj_console.print("  Indexing codebase...")
        result = idx.index(progress_callback=on_progress)
        _proj_console.print(f"\n  Done: {result['indexed']} files indexed, {result['total_chunks']} chunks, "
              f"{result['skipped']} unchanged, {result['elapsed']}s")
        idx.close()

    elif subcmd == "search":
        query = parts[1] if len(parts) > 1 else ""
        if not query:
            _proj_console.print("Usage: /project search <query>")
            return
        path = "."
        from aura.tools.codebase_index import CodebaseIndex
        idx = CodebaseIndex(path)
        if idx.stats()["total_chunks"] == 0:
            _proj_console.print("  No index found, indexing first...")
            idx.index()
        results = idx.search(query, top_k=10)
        if results:
            _proj_console.print(f"\n  Results for '{query}':\n")
            for r in results:
                score_pct = f"{r['score']:.0%}"
                _proj_console.print(f"  \\[{score_pct}] {r['file_path']}:{r['line_start']} ({r['kind']}) {r['name']}")
                snippet = (r.get('content') or '')[:100].replace('\n', ' ')
                _proj_console.print(f"        {snippet}")
        else:
            _proj_console.print("  No results found.")
        idx.close()

    else:
        _proj_console.print("Usage: /project \\[info|detect|init|context|index|search] \\[path|query]")
