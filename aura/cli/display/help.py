"""The /? help screen — extracted from the old monolithic display.py."""
from __future__ import annotations

from aura.cli import display as _display


def show_help() -> None:
    """Display help with clean aligned text, no heavy borders."""
    colors = _display._get_theme_colors()

    _display.console.print()
    _display.console.print("  [bold]Commands & Shortcuts[/bold]")
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
        _display.console.print(f"  [dim]{section_name}[/dim]")
        for key, action in commands:
            padded_key = key.ljust(28)
            _display.console.print(f"    [{accent}]{padded_key}[/{accent}] [dim]{action}[/dim]")
        _display.console.print()
