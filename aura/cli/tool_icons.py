"""Tool icons and status indicators for AURA CLI."""

# Per-tool text labels (Claude Code style — readable at a glance)
TOOL_ICONS = {
    "read_file": "Read",
    "write_file": "Write",
    "edit_file": "Edit",
    "glob": "Find",
    "grep": "Search",
    "list_dir": "List",
    "shell": "Run",
    "run_command": "Run",
    "web_search": "Web",
    "search_web": "Web",
    "web_fetch": "Fetch",
    "browse": "Browse",
    "browse_url": "Browse",
    "code_search": "Search",
    "project_structure": "Tree",
    "git": "Git",
    "image_gen": "Image",
    "inner_monologue": "Think",
    "spawn_agent": "Agent",
    "tool_search": "Search",
    "load_skill": "Skill",
    "clipboard": "Clip",
    "notifications": "Notify",
    "calendar": "Calendar",
    "task_manager": "Tasks",
    "code_executor": "Exec",
}

# Status indicators
STATUS_ICONS = {
    "pending": ("\u25cb", "dim"),
    "running": ("\u25cf", "cyan"),
    "success": ("\u2713", "green"),
    "error": ("\u2717", "red"),
    "cancelled": ("\u2500", "yellow"),
}

# Braille spinner frames (80ms per frame)
BRAILLE_SPINNER = "\u280b\u2819\u2839\u2838\u283c\u2834\u2826\u2827\u2807\u280f"


def get_tool_icon(tool_name: str) -> str:
    """Get the text label for a tool (e.g. 'Read', 'Edit', 'Run')."""
    return TOOL_ICONS.get(tool_name, "Tool")
