"""Auto-gather project context for the agentic loop system prompt.

Collects AURA.md content, git state, and project structure before the
first LLM call so the model has immediate project awareness.
"""

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


def gather_context(project_root: str) -> str:
    """Build context string for the agentic system prompt.

    Gathers:
    1. AGENTS.md content (cross-tool standard, loaded first)
    2. AURA.md content (project-specific overrides)
    3. Git status (branch, dirty file count)
    4. Project structure (depth 2, compact tree)
    5. Repo map with key symbols

    Returns a formatted string ready for injection into the system prompt.
    """
    parts = []

    # 1. AGENTS.md content (cross-tool standard, loaded first so AURA.md overrides)
    agents_md_content = _load_agents_md(project_root)
    if agents_md_content:
        parts.append(f"## Agent Instructions (AGENTS.md)\n{agents_md_content}")

    # 2. AURA.md content (project-specific, takes precedence)
    aura_md_content, _aura_md_config = _load_aura_md(project_root)
    if aura_md_content:
        parts.append(f"## Project Instructions (AURA.md)\n{aura_md_content}")

    # 3. Git state
    git_info = _get_git_info(project_root)
    if git_info:
        parts.append(f"## Git Status\n{git_info}")

    # 4. Project structure (compact)
    tree = _get_project_tree(project_root)
    if tree:
        parts.append(f"## Project Structure\n{tree}")

    # 5. Repo map with key symbols
    try:
        from .repo_map import build_repo_map
        repo_map = build_repo_map(project_root)
        if repo_map:
            parts.append(f"## Key Symbols (Repo Map)\n{repo_map}")
    except Exception as e:
        logger.debug(f"[Context] Repo map unavailable: {e}")

    if not parts:
        return f"Working directory: {project_root}"

    return "\n\n".join(parts)


def get_aura_md_config(project_root: str) -> dict:
    """Parse and return just the AURA.md frontmatter config, if any."""
    _, config = _load_aura_md(project_root)
    return config


def _load_agents_md(project_root: str) -> Optional[str]:
    """Load AGENTS.md or .agents.md from the project root.

    This is the cross-tool configuration standard used by Claude Code, Codex,
    Cursor, Goose, Copilot, etc. (https://agents.md/)

    Checks AGENTS.md first, then .agents.md (hidden variant).
    Capped at 32KB. Returns None if not found.
    """
    _MAX_AGENTS_MD_SIZE = 32 * 1024

    for filename in ("AGENTS.md", ".agents.md"):
        filepath = os.path.join(project_root, filename)
        if not os.path.isfile(filepath):
            continue

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
        except (OSError, PermissionError):
            logger.debug(f"[Context] Could not read {filename}")
            continue

        if len(content) > _MAX_AGENTS_MD_SIZE:
            logger.warning(
                f"[Context] {filename} exceeds {_MAX_AGENTS_MD_SIZE} bytes "
                f"({len(content)}), truncating"
            )
            content = content[:_MAX_AGENTS_MD_SIZE]

        logger.debug(f"[Context] Loaded {filename} ({len(content)} bytes)")
        return content.strip() or None

    return None


def _load_aura_md(project_root: str) -> tuple[Optional[str], dict]:
    """Load AURA.md, split into (markdown_body, frontmatter_config).

    Returns (None, {}) if no AURA.md found.
    """
    aura_md_path = os.path.join(project_root, "AURA.md")
    if not os.path.isfile(aura_md_path):
        return None, {}

    try:
        with open(aura_md_path, "r", encoding="utf-8") as f:
            content = f.read()
    except (OSError, PermissionError):
        return None, {}

    # Cap file size to 64KB to prevent yaml bomb / memory abuse
    _MAX_AURA_MD_SIZE = 64 * 1024
    if len(content) > _MAX_AURA_MD_SIZE:
        logger.warning(f"[Context] AURA.md exceeds {_MAX_AURA_MD_SIZE} bytes ({len(content)}), truncating")
        content = content[:_MAX_AURA_MD_SIZE]

    return parse_aura_md(content)


def parse_aura_md(content: str) -> tuple[str, dict]:
    """Parse AURA.md content into (markdown_body, frontmatter_config).

    Frontmatter is YAML between --- delimiters at the top of the file.

    Example AURA.md:
        ---
        tier: balanced
        model: qwen3.5:397b-cloud
        test_cmd: pytest
        auto_test: true
        permissions:
          shell: auto
        ---
        # My Project
        This is a Next.js app...

    Returns:
        (markdown_body, config_dict) — config is {} if no frontmatter
    """
    content = content.strip()
    if not content.startswith("---"):
        return content, {}

    # Find closing ---
    end = content.find("---", 3)
    if end == -1:
        return content, {}

    frontmatter_raw = content[3:end].strip()
    body = content[end + 3:].strip()

    # Parse YAML frontmatter
    config = {}
    try:
        import yaml
        config = yaml.safe_load(frontmatter_raw) or {}
    except Exception as e:
        logger.warning(f"[Context] Failed to parse AURA.md frontmatter: {e}")
        # Fallback: simple key: value parsing
        for line in frontmatter_raw.split("\n"):
            line = line.strip()
            if ":" in line and not line.startswith("#"):
                key, _, value = line.partition(":")
                val = value.strip()
                # Strip surrounding quotes (YAML style)
                if len(val) >= 2 and val[0] == val[-1] and val[0] in ('"', "'"):
                    val = val[1:-1]
                config[key.strip()] = val

    return body, config


def _get_git_info(project_root: str) -> Optional[str]:
    """Get compact git status info."""
    try:
        from aura.tools.git_tool import GitTool
        git = GitTool()
        status = git.status(project_root)
        if not status.get("success"):
            return None

        branch = status.get("branch", "unknown")
        dirty = status.get("dirty_count", 0)
        parts = [f"Branch: {branch}"]
        if dirty:
            parts.append(f"Dirty files: {dirty}")
            # Show changed files (compact)
            changed = status.get("changed_files", [])
            if changed:
                for f in changed[:10]:
                    parts.append(f"  {f}")
                if len(changed) > 10:
                    parts.append(f"  ... and {len(changed) - 10} more")
        else:
            parts.append("Clean working tree")

        # Add recent diff (first 2000 chars) for dirty files
        if dirty:
            try:
                import subprocess
                diff_result = subprocess.run(
                    ["git", "diff", "--stat", "--no-color"],
                    capture_output=True, text=True, timeout=5,
                    cwd=project_root,
                )
                if diff_result.stdout:
                    parts.append(f"\nRecent changes:\n{diff_result.stdout[:2000]}")
            except Exception:
                logger.debug("Failed to get git diff for context", exc_info=True)

        # Add last 3 commit messages for dev direction context
        try:
            import subprocess
            log_result = subprocess.run(
                ["git", "log", "--oneline", "-3", "--no-color"],
                capture_output=True, text=True, timeout=5,
                cwd=project_root,
            )
            if log_result.stdout:
                parts.append(f"\nRecent commits:\n{log_result.stdout.strip()}")
        except Exception:
            logger.debug("Failed to get git log for context", exc_info=True)

        return "\n".join(parts)
    except Exception as e:
        logger.debug(f"[Context] Git info unavailable: {e}")
        return None


def _get_project_tree(project_root: str) -> Optional[str]:
    """Get compact project tree (depth 2)."""
    try:
        from aura.tools.code_search import CodeSearchTool
        searcher = CodeSearchTool()
        result = searcher.project_structure(project_root, max_depth=2)
        if result.get("success"):
            tree = result.get("tree", "")
            # Cap at 60 lines for context budget
            lines = tree.split("\n")
            if len(lines) > 60:
                tree = "\n".join(lines[:60]) + f"\n... ({len(lines) - 60} more entries)"
            stats = result.get("stats", {})
            summary = f"{stats.get('files', 0)} files, {stats.get('dirs', 0)} dirs"
            langs = stats.get("languages", {})
            if langs:
                top_langs = sorted(langs.items(), key=lambda x: -x[1])[:5]
                summary += " | " + ", ".join(f"{l}({c})" for l, c in top_langs)
            return f"{tree}\n\n{summary}"
        return None
    except Exception as e:
        logger.debug(f"[Context] Project tree unavailable: {e}")
        return None
