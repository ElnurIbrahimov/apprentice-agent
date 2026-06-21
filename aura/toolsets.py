"""Toolset definitions — composable tool groups for per-platform control.

Inspired by Hermes Agent's toolset system. Tools are grouped into
named toolsets that can be enabled/disabled per platform (CLI, Telegram, API).

Config section (in ~/.aura/config.yaml):
    toolsets:
      enabled: [core, research, media]
      disabled: [cognitive, voice]
      platform:
        cli:      [core, research, media, system, deployment]
        telegram: [core, research, knowledge]
        api:      [core, research, media]

When a toolset is disabled, its tools are not registered in the agent's
tool dict and their keywords are removed from the fast-path matcher.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Set

logger = logging.getLogger(__name__)


# ── Toolset definitions ─────────────────────────────────────────────────

# Each toolset maps to a list of tool names (matching the keys in the tools dict).
# Tools not listed in any toolset are always loaded (uncategorized essentials).

TOOLSETS: Dict[str, Dict] = {
    "core": {
        "description": "Essential coding tools — search, edit, filesystem, git, code execution",
        "tools": [
            "code_search", "code_edit", "filesystem", "code_executor",
            "git", "brave_search", "web_search", "tool_search", "load_skill",
        ],
    },
    "research": {
        "description": "Research and information gathering",
        "tools": [
            "arxiv_search", "deep_research", "research_tool", "tavily_tool",
            "firecrawl_tool", "search_planner", "search_fallback",
        ],
    },
    "media": {
        "description": "Vision, images, PDFs, audio, documents",
        "tools": [
            "vision", "image_gen", "pdf_reader", "audio_transcriber",
            "document_generator", "visual_feedback", "screenshot",
            "screen_reader", "screenpipe",
        ],
    },
    "system": {
        "description": "System control and OS integration",
        "tools": [
            "system_control", "windows_control", "clipboard",
            "notifications",
        ],
    },
    "communication": {
        "description": "Email, calendar, messaging",
        "tools": [
            "email_tool", "calendar_tool",
        ],
    },
    "knowledge": {
        "description": "Knowledge graph, RAG, memory, skills",
        "tools": [
            "knowledge_graph", "local_rag", "obsidian_tool",
            "task_manager", "spaced_repetition",
        ],
    },
    "cognitive": {
        "description": "Inner monologue, dreams, emotions, reasoning trees",
        "tools": [
            "inner_monologue", "neurodream", "evoemo",
            "mcts_reasoning", "mcts_value_fn", "reasoning_tree_tool",
        ],
    },
    "productivity": {
        "description": "Task management, scheduling, predictions",
        "tools": [
            "task_scheduler", "predictive_tasks", "meeting_intel",
            "life_logger", "log_analyst",
        ],
    },
    "deployment": {
        "description": "Deploy, scaffold, GitHub, marketplace",
        "tools": [
            "deploy_tool", "scaffold", "github_tool", "marketplace",
            "component_registry", "tool_builder",
        ],
    },
    "browser": {
        "description": "Browser automation",
        "tools": ["browser"],
    },
    "voice": {
        "description": "Voice input/output and TTS",
        "tools": ["voice", "voice_synth", "voice_manager"],
    },
    "code": {
        "description": "Code analysis, type checking, sessions",
        "tools": [
            "codebase_index", "codebase_index_watcher", "coding_agent",
            "code_session_manager", "typecheck", "auto_verify",
            "auto_verify_hypothesis", "test_detection",
        ],
    },
    "database": {
        "description": "Database querying and management",
        "tools": ["database_tool"],
    },
    "crypto": {
        "description": "Crypto price tracking",
        "tools": ["crypto_price"],
    },
}


# Tools that are always loaded regardless of toolset config
ALWAYS_LOADED: Set[str] = {"code_search", "code_edit", "filesystem", "git", "tool_search"}


def get_enabled_toolsets(platform: str = "cli") -> List[str]:
    """Get the list of enabled toolset names for a platform.

    Reads from config.yaml, falling back to all toolsets enabled.

    Args:
        platform: 'cli', 'telegram', 'api', etc.
    """
    try:
        from aura.config_loader import get_toolsets_config
        cfg = get_toolsets_config()
    except ImportError:
        return list(TOOLSETS.keys())

    # Check platform-specific config first
    platform_cfg = cfg.get("platform", {})
    if platform and platform in platform_cfg:
        return platform_cfg[platform]

    # Fall back to enabled list
    enabled = cfg.get("enabled")
    if enabled is not None:
        return enabled

    # No config — all enabled
    return list(TOOLSETS.keys())


def get_disabled_toolsets() -> List[str]:
    """Get explicitly disabled toolsets."""
    try:
        from aura.config_loader import get_toolsets_config
        cfg = get_toolsets_config()
    except ImportError:
        return []

    return cfg.get("disabled", []) or []


def get_tools_for_platform(platform: str = "cli") -> Set[str]:
    """Get the set of tool names that should be loaded for a platform.

    Always includes ALWAYS_LOADED tools plus all tools from enabled toolsets.
    Excludes tools from disabled toolsets.
    """
    enabled = get_enabled_toolsets(platform)
    disabled = set(get_disabled_toolsets())
    active = [t for t in enabled if t not in disabled]

    tool_names: Set[str] = set(ALWAYS_LOADED)
    for ts_name in active:
        ts = TOOLSETS.get(ts_name)
        if ts:
            tool_names.update(ts["tools"])

    return tool_names


def should_load_tool(tool_name: str, platform: str = "cli") -> bool:
    """Check if a tool should be loaded for a platform."""
    if tool_name in ALWAYS_LOADED:
        return True
    return tool_name in get_tools_for_platform(platform)


def list_toolsets() -> List[dict]:
    """List all toolsets with their status and tool count."""
    enabled = set(get_enabled_toolsets())
    disabled = set(get_disabled_toolsets())

    result = []
    for name, info in sorted(TOOLSETS.items()):
        is_enabled = name in enabled and name not in disabled
        result.append({
            "name": name,
            "description": info["description"],
            "tool_count": len(info["tools"]),
            "tools": info["tools"],
            "enabled": is_enabled,
        })
    return result
