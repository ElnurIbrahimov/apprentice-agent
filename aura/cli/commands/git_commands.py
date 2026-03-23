import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def handle_diff(agent, arg, context) -> Optional[str]:
    import subprocess as _sp
    from ..display import console as _diff_console, show_error as _diff_err
    from rich.syntax import Syntax as _DiffSyntax
    try:
        diff_args = ["git", "diff"]
        if arg:
            safe_tokens = []
            for token in arg.split():
                if token.startswith("-c") or token.startswith("--ext-diff"):
                    _diff_err(f"Blocked unsafe flag: {token}")
                    safe_tokens = None
                    break
                safe_tokens.append(token)
            if safe_tokens is None:
                return
            diff_args.extend(safe_tokens)
        result = _sp.run(diff_args, capture_output=True, text=True, cwd=os.getcwd(), timeout=10)
        if result.stdout:
            total = len(result.stdout)
            limit = 10000
            text = result.stdout[:limit]
            _diff_console.print(_DiffSyntax(text, "diff", theme="monokai"))
            if total > limit:
                _diff_console.print(f"[dim]... [truncated — {total} total chars, showing first {limit}][/dim]")
        else:
            _diff_console.print("  [dim]No changes.[/dim]")
    except (OSError, _sp.SubprocessError, ValueError) as e:
        _diff_err(str(e))


def handle_git(agent, arg, context) -> Optional[str]:
    if not arg:
        print("Usage: /git <command> (e.g., /git status, /git log, /git diff)")
    else:
        import subprocess as _sp
        GIT_SAFE_SUBCOMMANDS = frozenset({
            "status", "log", "diff", "branch", "show", "stash",
            "remote", "tag", "shortlog", "describe", "rev-parse",
            "ls-files", "ls-tree", "blame",
        })
        git_tokens = arg.split()
        git_subcmd = git_tokens[0].lower() if git_tokens else ""
        has_dangerous_flag = any(t == "-c" or t.startswith("-c=") or t.startswith("--exec") or t.startswith("--upload-pack") for t in git_tokens)
        if has_dangerous_flag:
            print("  Blocked: dangerous git flags (-c, --exec, --upload-pack) are not allowed")
        elif git_subcmd not in GIT_SAFE_SUBCOMMANDS:
            print(f"  Blocked: '/git {git_subcmd}' — only read-only git commands are allowed")
            print(f"  Allowed: {', '.join(sorted(GIT_SAFE_SUBCOMMANDS))}")
        else:
            from ..display import console as _git_console, show_error as _git_err
            try:
                result = _sp.run(
                    ["git"] + git_tokens,
                    capture_output=True, text=True, cwd=os.getcwd(), timeout=15,
                )
                output = result.stdout or result.stderr
                if output:
                    limit = 10000
                    total = len(output)
                    text = output[:limit]
                    if git_subcmd in ("diff", "show"):
                        from rich.syntax import Syntax as _GitSyntax
                        _git_console.print(_GitSyntax(text, "diff", theme="monokai"))
                    else:
                        _git_console.print(text, highlight=False)
                    if total > limit:
                        _git_console.print(f"[dim]... [truncated — {total} total chars, showing first {limit}][/dim]")
                else:
                    _git_console.print("  [dim](no output)[/dim]")
            except (OSError, _sp.SubprocessError, ValueError) as e:
                _git_err(str(e))


def handle_pr(agent, arg, context) -> Optional[str]:
    from ..git_tools import create_pr, get_staged_diff, get_recent_log, get_current_branch, PR_DESCRIPTION_PROMPT
    from ..display import console as _pr_console
    branch = get_current_branch()
    diff = get_staged_diff()
    log = get_recent_log()
    prompt = PR_DESCRIPTION_PROMPT.format(branch=branch, diff=diff, log=log)
    response = agent.brain.think(prompt)
    if isinstance(response, dict):
        response = response.get("response", response.get("content", str(response)))
    title = branch
    body = response or ""
    for line in (response or "").splitlines():
        if line.startswith("TITLE:"):
            title = line[6:].strip()[:70]
            break
    body_start = (response or "").find("BODY:")
    if body_start >= 0:
        body = response[body_start + 5:].strip()
    result = create_pr(title, body)
    if result["success"]:
        _pr_console.print(f"[green]PR created: {result['url']}[/green]")
    else:
        _pr_console.print(f"[red]{result['error']}[/red]")


def handle_branch(agent, arg, context) -> Optional[str]:
    from ..git_tools import create_branch
    from ..display import console as _branch_console
    name = arg.strip()
    if not name:
        _branch_console.print("[dim]Usage: /branch <name>[/dim]")
        return
    result = create_branch(name)
    if result["success"]:
        _branch_console.print(f"[green]Created branch: {result['branch']}[/green]")
    else:
        _branch_console.print(f"[red]{result['message']}[/red]")


def handle_stash(agent, arg, context) -> Optional[str]:
    from ..git_tools import smart_stash
    from ..display import console as _stash_console
    desc = arg.strip() or "Aura stash"
    result = smart_stash(desc)
    if result["success"]:
        _stash_console.print(f"[green]Stashed: {desc}[/green]")
    else:
        _stash_console.print(f"[red]{result.get('stderr', 'Stash failed')}[/red]")


def handle_blame(agent, arg, context) -> Optional[str]:
    from ..git_tools import get_blame
    from ..display import console as _blame_console
    parts_b = arg.strip().rsplit(":", 1)
    if len(parts_b) != 2 or not parts_b[1].isdigit():
        _blame_console.print("[dim]Usage: /blame <file>:<line>[/dim]")
        return
    result = get_blame(parts_b[0], int(parts_b[1]))
    if result.get("success"):
        _blame_console.print(f"  Author: [cyan]{result.get('author', '?')}[/cyan]")
        _blame_console.print(f"  Date:   {result.get('date', '?')}")
        _blame_console.print(f"  Commit: [dim]{result.get('commit_message', '?')}[/dim]")
        _blame_console.print(f"  Line:   {result.get('content', '?')}")
    else:
        _blame_console.print(f"[red]{result.get('error', result.get('stderr', 'Blame failed'))}[/red]")
