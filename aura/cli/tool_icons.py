"""Tool icons and status indicators for AURA CLI."""

# Per-tool Unicode icons (inspired by OpenCode)
TOOL_ICONS = {
    "read_file": "\u2192",
    "write_file": "\u2190",
    "edit_file": "\u2190",
    "glob": "\u2731",
    "grep": "\u2731",
    "list_dir": "\u2192",
    "shell": "$",
    "run_command": "$",
    "web_search": "\u25c8",
    "search_web": "\u25c8",
    "web_fetch": "%",
    "browse": "%",
    "browse_url": "%",
    "code_search": "\u25c7",
    "project_structure": "\u25c7",
    "git": "\u238b",
    "image_gen": "\u25ce",
    "inner_monologue": "\u00b7",
    "spawn_agent": "\u2502",
    "tool_search": "\u25c7",
    "load_skill": "\u26a1",
    "clipboard": "\U0001f4cb",
    "notifications": "\u25c6",
    "calendar": "\u25c6",
    "task_manager": "\u25c6",
    "code_executor": "\u2699",
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
    """Get the icon for a tool."""
    return TOOL_ICONS.get(tool_name, "\u2699")
