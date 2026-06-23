import logging
import os
from typing import Optional

from ..display import console, show_info
from .common import command, TIER_BETA, TIER_STABLE

logger = logging.getLogger(__name__)


@command("/diff",     "Show git diff with syntax highlighting",           tier=TIER_STABLE)
def handle_diff(agent, arg, context) -> Optional[str]:
    import subprocess as _sp

    from rich.syntax import Syntax as _DiffSyntax

    from ..display import show_error
    try:
        diff_args = ["git", "diff"]
        if arg:
            safe_tokens = []
            for token in arg.split():
                if token == "-c" or token.startswith("-c=") or token == "--ext-diff" or token.startswith("--ext-diff=") or token.startswith("--extcmd"):
                    show_error(f"Blocked unsafe flag: {token}")
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
            console.print(_DiffSyntax(text, "diff", theme="monokai"))
            if total > limit:
                show_info(f"... [truncated — {total} total chars, showing first {limit}]")
        else:
            show_info("No changes.")
    except (OSError, _sp.SubprocessError, ValueError) as e:
        show_error(str(e))


@command("/git",      "Run read-only git commands",                       tier=TIER_STABLE)
def handle_git(agent, arg, context) -> Optional[str]:
    if not arg:
        console.print("[yellow]Usage: /git <command> (e.g., /git status, /git log, /git diff)[/yellow]")
    else:
        import subprocess as _sp
        # NB: re-importing ``console`` inside the function (rather than
        # using the module-level binding) is intentional — it lets the
        # test_handlers_smoke tests mock ``aura.cli.display.console`` and
        # see the calls. The module-level ``console`` is bound once at
        # import time and won't reflect later patches.
        from ..display import console as _git_console
        from ..display import show_error as _git_err
        GIT_SAFE_SUBCOMMANDS = frozenset({
            "status", "log", "diff", "branch", "show", "stash",
            "remote", "tag", "shortlog", "describe", "rev-parse",
            "ls-files", "ls-tree", "blame",
        })
        git_tokens = arg.split()
        git_subcmd = git_tokens[0].lower() if git_tokens else ""
        has_dangerous_flag = any(t == "-c" or t.startswith("-c=") or t.startswith("--exec") or t.startswith("--upload-pack") for t in git_tokens)
        if has_dangerous_flag:
            console.print("  [red]Blocked: dangerous git flags (-c, --exec, --upload-pack) are not allowed[/red]")
        elif git_subcmd not in GIT_SAFE_SUBCOMMANDS:
            console.print(f"  [red]Blocked: '/git {git_subcmd}' — only read-only git commands are allowed[/red]")
            show_info(f"Allowed: {', '.join(sorted(GIT_SAFE_SUBCOMMANDS))}")
        else:
            try:
                result = _sp.run(
                    ["git", *git_tokens],
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
                        show_info(f"... [truncated — {total} total chars, showing first {limit}]")
                else:
                    show_info("(no output)")
            except (OSError, _sp.SubprocessError, ValueError) as e:
                _git_err(str(e))


@command("/pr",       "Create pull request",                              tier=TIER_BETA)
def handle_pr(agent, arg, context) -> Optional[str]:
    from ..display import show_success, show_error
    from ..git_tools import (
        PR_DESCRIPTION_PROMPT,
        create_pr,
        get_current_branch,
        get_recent_log,
        get_staged_diff,
    )
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
        show_success(f"PR created: {result['url']}")
    else:
        show_error(result['error'])


@command("/branch",   "Create git branch",                                tier=TIER_BETA)
def handle_branch(agent, arg, context) -> Optional[str]:
    from ..display import show_info, show_success, show_error
    from ..git_tools import create_branch
    name = arg.strip()
    if not name:
        show_info("Usage: /branch <name>")
        return
    result = create_branch(name)
    if result["success"]:
        show_success(f"Created branch: {result['branch']}")
    else:
        show_error(result['message'])


@command("/stash",    "Smart git stash",                                  tier=TIER_BETA)
def handle_stash(agent, arg, context) -> Optional[str]:
    from ..display import show_success, show_error
    from ..git_tools import smart_stash
    desc = arg.strip() or "Aura stash"
    result = smart_stash(desc)
    if result["success"]:
        show_success(f"Stashed: {desc}")
    else:
        show_error(result.get('stderr', 'Stash failed'))


@command("/blame",    "Git blame with context",                           tier=TIER_BETA)
def handle_blame(agent, arg, context) -> Optional[str]:
    from ..display import show_error
    from ..git_tools import get_blame
    parts_b = arg.strip().rsplit(":", 1)
    if len(parts_b) != 2 or not parts_b[1].isdigit():
        show_info("Usage: /blame <file>:<line>")
        return
    result = get_blame(parts_b[0], int(parts_b[1]))
    if result.get("success"):
        console.print(f"  Author: [cyan]{result.get('author', '?')}[/cyan]")
        console.print(f"  Date:   {result.get('date', '?')}")
        console.print(f"  Commit: [dim]{result.get('commit_message', '?')}[/dim]")
        console.print(f"  Line:   {result.get('content', '?')}")
    else:
        show_error(result.get('error', result.get('stderr', 'Blame failed')))
