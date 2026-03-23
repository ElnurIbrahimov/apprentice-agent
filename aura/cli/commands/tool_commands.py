import os
import logging
from pathlib import Path
from typing import Optional

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
    from ..test_runner import run_tests, render_test_results
    from ..display import console as _test_console, show_response as _test_show, show_info as _test_info, show_tool_call as _test_tool
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
        _test_console.print("[dim]Auto-fix failures? (y/n)[/dim]")
        try:
            choice = input("> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return
        if choice in ("y", "yes"):
            agentic_loop = getattr(agent, '_agentic_loop', None)
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
    from ..watch_mode import FileWatcher, create_watch_indicator, remove_ai_comment
    from ..display import console as _watch_console
    watcher = getattr(agent, '_file_watcher', None)
    if not watcher:
        watcher = FileWatcher()
        agent._file_watcher = watcher

    if arg.strip() == "stop":
        watcher.stop()
        _watch_console.print("[dim]Watch mode stopped.[/dim]")
        return

    if watcher.is_running:
        hits = watcher.get_unresolved()
        if hits:
            _watch_console.print(f"[magenta]Watching -- {len(hits)} unresolved comments:[/magenta]")
            for h in hits[:10]:
                fname = Path(h.file_path).name
                _watch_console.print(f"  [cyan]{fname}:{h.line_number}[/cyan] {h.instruction}")
        else:
            _watch_console.print("[dim]Watching -- no AI comments detected.[/dim]")
        return

    def _on_watch_hit(hit):
        _watch_console.print(f"[magenta]Detected:[/magenta] [cyan]{Path(hit.file_path).name}:{hit.line_number}[/cyan] {hit.instruction}")

    watcher.set_callback(_on_watch_hit)
    hits = watcher.scan_all()
    if hits:
        _watch_console.print(f"[magenta]Found {len(hits)} AI comments:[/magenta]")
        for h in hits[:5]:
            _watch_console.print(f"  [cyan]{Path(h.file_path).name}:{h.line_number}[/cyan] {h.instruction}")
    watcher.start()
    _watch_console.print("[dim]Watch mode started. Monitoring for AURA: and AI: comments.[/dim]")


def handle_undo(agent, arg, context) -> Optional[str]:
    if hasattr(agent, '_agentic_loop'):
        tool_exec = agent._agentic_loop.executor
        if arg:
            result = tool_exec.code_edit.rollback(arg)
            if result.get("success"):
                print(f"  Rolled back: {result.get('restored', arg)}")
            else:
                print(f"  Error: {result.get('error', 'Unknown error')}")
        else:
            backups = tool_exec.code_edit._last_backups
            if backups:
                last_path = list(backups.keys())[-1]
                result = tool_exec.code_edit.rollback(last_path)
                if result.get("success"):
                    print(f"  Rolled back: {last_path}")
                else:
                    print(f"  Error: {result.get('error', 'Unknown error')}")
            else:
                print("  No edits to undo (no .bak files)")
    else:
        print("  No active agentic loop.")


def _handle_shell_command(agent, arg: str):
    if not arg:
        print("Usage: /shell <command>")
        print("  /shell git status")
        print("  /run npm test")
        print("  /bash ls -la")
        return

    try:
        from aura.core.permissions import PermissionTier
        pm = getattr(agent, "permissions", None)
        if pm and pm.current_mode != PermissionTier.FULL_AUTO:
            confirm = input(f"  Execute shell command: {arg}\n  Confirm? (y/n): ").strip().lower()
            if confirm not in ("y", "yes"):
                print("  Cancelled.")
                return
    except (ImportError, AttributeError, EOFError, KeyboardInterrupt):
        logger.debug("shell_permission_check_skipped", exc_info=True)

    tool = agent.tools.get("shell_executor")
    if not tool:
        from aura.tools.shell_executor import ShellExecutorTool
        tool = ShellExecutorTool()
        agent.tools["shell_executor"] = tool

    def on_line(line):
        print(f"  {line}")

    result = tool.run_streaming(command=arg, on_output=on_line)

    if not result.get("success"):
        error = result.get("error", "")
        if error:
            print(f"\n  Error: {error}")
    print(f"\n  [exit {result.get('exit_code', '?')}] ({result.get('elapsed', '?')}s)")


def _handle_grep_command(agent, arg: str):
    if not arg:
        print("Usage: /grep <pattern> [path]")
        print("  /grep 'def my_func'")
        print("  /grep 'import os' ./src")
        print("  /grep 'TODO' --type py")
        return

    tool = agent.tools.get("code_search")
    if not tool:
        from aura.tools.code_search import CodeSearchTool
        tool = CodeSearchTool()
        agent.tools["code_search"] = tool

    parts = arg.split()
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
            path = parts[i]
            i += 1

    result = tool.grep(
        pattern=pattern, path=path, file_type=file_type,
        case_insensitive=case_insensitive, context_lines=context,
    )

    if not result.get("success"):
        print(f"  Error: {result.get('error')}")
        return

    matches = result.get("matches", [])
    total = result.get("total_matches", 0)
    print(f"\n  {total} matches in {result.get('files_searched', 0)} files:\n")
    for m in matches[:50]:
        print(f"  {m['file']}:{m['line']}\t{m['text']}")
        for ctx in m.get("before", []):
            print(f"    {ctx}")
        for ctx in m.get("after", []):
            print(f"    {ctx}")
    if total > 50:
        print(f"\n  ... and {total - 50} more matches")


def _handle_search_command(agent, arg: str):
    if not arg:
        print("Usage: /search <glob-pattern>  or  /find def <name>")
        print("  /search '*.py'")
        print("  /find def MyClass")
        print("  /search structure")
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
            print("Usage: /find def <name>")
            return
        result = tool.find_definition(name=name)
        if result.get("success"):
            defs = result.get("definitions", [])
            print(f"\n  Found {len(defs)} definition(s) of '{name}':\n")
            for d in defs:
                print(f"  {d['file']}:{d['line']} ({d['kind']})")
                print(f"    {d['text']}")
        else:
            print(f"  Error: {result.get('error')}")

    elif subcmd == "ref" or subcmd == "references":
        name = parts[1] if len(parts) > 1 else ""
        if not name:
            print("Usage: /find ref <name>")
            return
        result = tool.find_references(name=name)
        if result.get("success"):
            refs = result.get("references", [])
            print(f"\n  Found {len(refs)} reference(s) to '{name}':\n")
            for r in refs[:30]:
                print(f"  {r['file']}:{r['line']}\t{r['text']}")
        else:
            print(f"  Error: {result.get('error')}")

    elif subcmd == "structure" or subcmd == "tree":
        path = parts[1] if len(parts) > 1 else "."
        result = tool.project_structure(path=path)
        if result.get("success"):
            print(f"\n{result['tree']}")
            s = result.get("stats", {})
            print(f"\n  {s.get('files', 0)} files, {s.get('dirs', 0)} dirs")
        else:
            print(f"  Error: {result.get('error')}")

    else:
        result = tool.glob(pattern=arg)
        if result.get("success"):
            files = result.get("files", [])
            print(f"\n  Found {result.get('total', 0)} files:\n")
            for f in files[:50]:
                size = f.get("size", 0)
                size_str = f"{size // 1024}KB" if size > 1024 else f"{size}B"
                print(f"  {f['path']}  ({size_str})")
            if result.get("truncated"):
                print(f"\n  ... truncated ({result['total']} total)")
        else:
            print(f"  Error: {result.get('error')}")


def _handle_edit_command(agent, arg: str):
    if not arg:
        print("Usage: /edit <file-path> [line-offset]")
        print("  /edit src/main.py")
        print("  /edit src/main.py 100")
        return

    tool = agent.tools.get("code_edit")
    if not tool:
        from aura.tools.code_edit import CodeEditTool
        tool = CodeEditTool()
        agent.tools["code_edit"] = tool

    parts = arg.split()
    path = parts[0]
    offset = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
    limit = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 100

    result = tool.read_file(path=path, offset=offset, limit=limit)
    if result.get("success"):
        print(f"\n  {result['showing']}  ({result['path']})\n")
        print(result["content"])
    else:
        print(f"  Error: {result.get('error')}")


def _handle_project_command(agent, arg: str):
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
        print(init_project(path))

    elif subcmd == "detect" or subcmd == "info":
        path = parts[1] if len(parts) > 1 else "."
        result = tool.detect_project_type(path=path)
        if result.get("success"):
            print(f"\n  Project Type:     {result.get('project_type', 'unknown')}")
            print(f"  Language:         {result.get('language', 'N/A')}")
            print(f"  Stack:            {', '.join(result.get('stack', [])) or 'N/A'}")
            print(f"  Frameworks:       {', '.join(result.get('frameworks', [])) or 'N/A'}")
            print(f"  Package Manager:  {result.get('package_manager', 'N/A')}")
            print(f"  Key Files:        {', '.join(result.get('key_files', [])) or 'N/A'}")
        else:
            print(f"  Error: {result.get('error')}")

    elif subcmd == "context":
        from aura.tools.project_context import load_project_context
        path = parts[1] if len(parts) > 1 else None
        ctx = load_project_context(path)
        if ctx:
            print(f"\n{ctx}")
        else:
            print("  No AURA.md found. Create one with: /project init")

    elif subcmd == "index":
        path = parts[1] if len(parts) > 1 else "."
        from aura.tools.codebase_index import CodebaseIndex
        idx = CodebaseIndex(path)
        def on_progress(current, total, fpath):
            if current % 20 == 0 or current == total:
                print(f"  [{current}/{total}] {fpath}")
        print("  Indexing codebase...")
        result = idx.index(progress_callback=on_progress)
        print(f"\n  Done: {result['indexed']} files indexed, {result['total_chunks']} chunks, "
              f"{result['skipped']} unchanged, {result['elapsed']}s")
        idx.close()

    elif subcmd == "search":
        query = parts[1] if len(parts) > 1 else ""
        if not query:
            print("Usage: /project search <query>")
            return
        path = "."
        from aura.tools.codebase_index import CodebaseIndex
        idx = CodebaseIndex(path)
        if idx.stats()["total_chunks"] == 0:
            print("  No index found, indexing first...")
            idx.index()
        results = idx.search(query, top_k=10)
        if results:
            print(f"\n  Results for '{query}':\n")
            for r in results:
                score_pct = f"{r['score']:.0%}"
                print(f"  [{score_pct}] {r['file_path']}:{r['line_start']} ({r['kind']}) {r['name']}")
                snippet = (r.get('content') or '')[:100].replace('\n', ' ')
                print(f"        {snippet}")
        else:
            print("  No results found.")
        idx.close()

    else:
        print("Usage: /project [info|detect|init|context|index|search] [path|query]")
