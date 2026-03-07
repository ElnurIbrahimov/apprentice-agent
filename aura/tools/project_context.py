"""AURA.md Project Context — per-project persistent context injected into every response."""

import os
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

AURA_MD_FILENAME = "AURA.md"

AURA_MD_TEMPLATE = """# AURA Project Context
## Stack
<!-- Tech stack, frameworks, languages -->

## Key Files
<!-- Most important files and their purpose -->

## Conventions
<!-- Coding style, naming patterns, architecture decisions -->

## Current Focus
<!-- What you're actively working on right now -->

## Notes from AURA
<!-- Auto-updated by AURA after sessions -->
"""


def load_project_context(start_path: Optional[str] = None) -> Optional[str]:
    """Walk up from start_path looking for AURA.md. Returns content or None."""
    try:
        search_path = Path(start_path).resolve() if start_path else Path.cwd().resolve()
    except (OSError, TypeError):
        search_path = Path.cwd().resolve()

    # Walk up directory tree looking for AURA.md
    current = search_path
    for _ in range(8):  # Max 8 levels up
        candidate = current / AURA_MD_FILENAME
        if candidate.exists() and candidate.is_file():
            try:
                content = candidate.read_text(encoding="utf-8").strip()
                if content:
                    logger.debug(f"[ProjectContext] Loaded AURA.md from: {candidate}")
                    return content
            except (OSError, PermissionError) as e:
                logger.warning(f"[ProjectContext] Could not read {candidate}: {e}")
                return None
        parent = current.parent
        if parent == current:
            break
        current = parent

    return None


def init_project(path: str) -> str:
    """Create AURA.md in path with template. Return confirmation message."""
    try:
        dir_path = Path(path).resolve()
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)

        aura_md = dir_path / AURA_MD_FILENAME
        if aura_md.exists():
            return f"AURA.md already exists at: {aura_md}\nEdit it to update your project context."

        aura_md.write_text(AURA_MD_TEMPLATE, encoding="utf-8")
        return f"Created AURA.md at: {aura_md}\nEdit it to set your project context — AURA will read it automatically."
    except (OSError, PermissionError) as e:
        return f"Failed to create AURA.md: {e}"


def update_project_notes(path: str, note: str) -> bool:
    """Append note to '## Notes from AURA' section."""
    try:
        aura_md = Path(path).resolve() / AURA_MD_FILENAME
        if not aura_md.exists():
            return False

        content = aura_md.read_text(encoding="utf-8")
        section = "## Notes from AURA"

        if section in content:
            # Insert note after the section header
            idx = content.index(section) + len(section)
            # Skip any existing comment line
            rest = content[idx:]
            insert_text = f"\n- {note}"
            content = content[:idx] + insert_text + rest
        else:
            content += f"\n\n{section}\n- {note}"

        aura_md.write_text(content, encoding="utf-8")
        return True
    except (OSError, PermissionError):
        return False
