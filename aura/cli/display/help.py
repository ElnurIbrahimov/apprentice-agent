"""The /? help screen — extracted from the old monolithic display.py."""
from __future__ import annotations

import re as _re

from aura.cli import display as _display


def _extract_cmd_name(key: str) -> str:
    """Pull the slash command name out of a help entry like '/model [name]'."""
    m = _re.match(r"(/[a-z]+)", key)
    return m.group(1) if m else ""


def show_help(show_experimental: bool = False) -> None:
    """Display help with clean aligned text, no heavy borders.

    Experimental commands (GEPA `/evolve`, `/hand`, `/debate`, conversation-fork
    family) are hidden unless *show_experimental* is True (invoked via
    `/help all`). This prevents new users from treating exploratory features
    as supported.
    """
    try:
        from aura.cli.commands import EXPERIMENTAL_COMMANDS
    except ImportError:
        EXPERIMENTAL_COMMANDS = set()

    colors = _display._get_theme_colors()

    _display.console.print()
    _display.console.print("  [bold]Commands & Shortcuts[/bold]")
    if not show_experimental and EXPERIMENTAL_COMMANDS:
        _display.console.print(
            "  [dim]type[/dim] [cyan]/help all[/cyan] [dim]to show experimental commands[/dim]"
        )
    _display.console.print()

    sections = [
        ("Keyboard", [
            ("Alt+M", "Model picker"),
            ("Ctrl+L", "Clear screen"),
            ("Ctrl+N", "New session"),
            ("Ctrl+K", "Command palette"),
            ("Ctrl+G", "Open editor for long prompt"),
            ("Shift+Tab", "Cycle permission mode"),
            ("Ctrl+Z", "Rewind to checkpoint"),
            ("Alt+Enter", "Insert newline"),
            ("Ctrl+C / Ctrl+D", "Exit"),
            ("?", "Show this help"),
        ]),
        ("Model & Session", [
            ("/model [name]", "Pick or set model"),
            ("/routing", "Show neural routing status"),
            ("/sessions", "Manage sessions"),
            ("/trace [count|last|runs|failures]", "Show session trace and run summaries"),
            ("/compact", "Compress conversation"),
            ("/clear", "Clear conversation"),
        ]),
        ("Code & Files", [
            ("/grep <pattern>", "Search code content"),
            ("/search <query>", "Search files and definitions"),
            ("/edit <file>", "Read file with line numbers"),
            ("/project [cmd]", "Project context, indexing, search"),
        ]),
        ("Execution", [
            ("/shell <cmd>", "Execute shell command"),
            ("/plan <task>", "Create and execute a plan"),
            ("/agent <name> <task>", "Run specialist agent"),
            ("/goal <objective>", "Run a goal"),
        ]),
        ("Parallel & Background", [
            ("/fleet <task>", "Run parallel sub-agents"),
            ("/chain step1 -> step2", "Run prompt pipeline"),
            ("& <prompt>", "Run as background task"),
            ("/tasks", "Show background tasks"),
        ]),
        ("Research", [
            ("/research <topic>", "Start research mode"),
            ("/sources", "Show collected sources"),
            ("/export research", "Export to Markdown"),
        ]),
        ("Git", [
            ("/pr", "Create pull request"),
            ("/branch <name>", "Create and switch branch"),
            ("/stash [desc]", "Smart stash"),
            ("/blame file:N", "Explain line history"),
            ("/diff [args]", "Show git diff"),
            ("/git <command>", "Run read-only git commands"),
        ]),
        ("Testing & Watch", [
            ("/test [cmd]", "Run tests"),
            ("/watch", "Monitor files for AI comments"),
        ]),
        ("Utilities", [
            ("/browse <url>", "Browse web pages"),
            ("/speak <text>", "Text-to-speech"),
            ("/recall <query>", "Search memories"),
            ("/context", "Show context usage"),
            ("/rewind", "Rewind to checkpoint"),
            ("/theme [name]", "Switch color theme"),
            ("/mood", "Show emotional state"),
            ("/hook [cmd]", "Manage automation hooks"),
        ]),
        ("Multi-agent", [
            ("/debate <topic>", "Multi-agent debate"),
            ("/fork [name]", "Fork conversation"),
            ("/branches", "List branches"),
            ("/checkout <branch>", "Switch branch"),
            ("/merge <branch>", "Merge branch"),
            ("/undo", "Undo last file edit"),
        ]),
        ("MCP & Audit", [
            ("/mcp [cmd]", "Manage MCP servers"),
            ("/audit [cmd]", "Inspect audit chain"),
        ]),
        ("Autonomous", [
            ("/hand [cmd]", "Manage autonomous Hands"),
            ("/evolve [...]", "Evolve skills with GEPA"),
        ]),
        ("Other", [
            ("/voice", "Start voice mode"),
            ("/retry", "Re-run last prompt"),
            ("/cost", "Show token usage and cost"),
            ("/trust", "Toggle trust mode"),
            ("/quit", "Exit AURA"),
        ]),
    ]

    accent = colors["accent"]
    for section_name, commands in sections:
        # Filter out experimental commands when not requested, and skip entire
        # sections that would become empty (e.g. the "Autonomous" block is all
        # experimental: /hand, /evolve).
        if not show_experimental:
            commands = [
                (k, a) for k, a in commands
                if _extract_cmd_name(k) not in EXPERIMENTAL_COMMANDS
            ]
        if not commands:
            continue
        _display.console.print(f"  [dim]{section_name}[/dim]")
        for key, action in commands:
            padded_key = key.ljust(28)
            cmd_name = _extract_cmd_name(key)
            tag = " [yellow dim][exp][/yellow dim]" if cmd_name in EXPERIMENTAL_COMMANDS else ""
            _display.console.print(
                f"    [{accent}]{padded_key}[/{accent}] [dim]{action}[/dim]{tag}"
            )
        _display.console.print()
