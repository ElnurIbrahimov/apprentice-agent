"""
/skill — interactive skill discovery. Search, view, and load skills.
Without arguments: interactive picker. With name: load directly.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Optional

from .common import TIER_BETA, command

logger = logging.getLogger(__name__)


def _find_skills() -> list[dict]:
    """Scan all skill directories for SKILL.md files."""
    skills: list[dict] = []
    search_paths = [
        os.path.expanduser("~/.aura/skills"),
        os.path.expanduser("~/.aura/aura_skill_library/skills"),
    ]
    for base in search_paths:
        base_path = Path(base)
        if not base_path.exists():
            continue
        for skill_file in base_path.rglob("SKILL.md"):
            try:
                meta = _parse_skill_metadata(skill_file)
                meta["path"] = str(skill_file)
                skills.append(meta)
            except Exception:
                logger.debug("skill_parse_failed path=%s", skill_file, exc_info=True)
    return skills


def _parse_skill_metadata(path: Path) -> dict:
    """Extract name and description from a SKILL.md frontmatter."""
    content = path.read_text(encoding="utf-8", errors="ignore")
    name = path.parent.name if path.parent.name != "skills" else path.stem
    description = ""
    in_frontmatter = False
    for line in content.split("\n"):
        line = line.strip()
        if line == "---":
            if not in_frontmatter:
                in_frontmatter = True
                continue
            else:
                break
        if in_frontmatter:
            if line.startswith("name:"):
                name = line.split(":", 1)[1].strip().strip('"').strip("'")
            elif line.startswith("description:"):
                description = line.split(":", 1)[1].strip().strip('"').strip("'")
    # Also check for markdown H1 title
    for line in content.split("\n"):
        if line.startswith("# "):
            title = line[2:].strip()
            if not name or name == path.parent.name:
                name = title
            if not description:
                # Use first paragraph after title as description
                pass
            break
    return {"name": name, "description": description[:80]}


@command("/skill",    "Browse and load skills",                             tier=TIER_BETA)
def handle_skill(agent: Any, arg: str, context: dict) -> Optional[str]:
    from rich.panel import Panel
    from rich.table import Table

    from ..display import console

    skills = _find_skills()

    arg = (arg or "").strip()

    # Direct load: /skill <name>
    if arg:
        matching = [s for s in skills if arg.lower() in s["name"].lower()]
        if not matching:
            console.print()
            console.print(f"  [yellow]No skill matching '{arg}' found.[/yellow]")
            console.print("  [dim]Use /skill to browse available skills.[/dim]")
            console.print()
            return None

        if len(matching) == 1:
            skill = matching[0]
        else:
            # Multiple matches — let user pick
            skill = _pick_skill(matching, console)

        if skill:
            try:
                from aura.tools.skill_loader import load_skill
                load_skill(skill["path"])
                console.print(f"  [green]✓ Loaded:[/green] [bold]{skill['name']}[/bold]")
                if skill.get("description"):
                    console.print(f"    [dim]{skill['description']}[/dim]")
            except ImportError:
                # Fallback: try loading SKILL.md content into context
                try:
                    content = Path(skill["path"]).read_text(encoding="utf-8")
                    agent.brain._inject_skill(skill["name"], content)
                    console.print(f"  [green]✓ Injected:[/green] [bold]{skill['name']}[/bold]")
                except Exception as e:
                    console.print(f"  [red]Failed to load skill: {e}[/red]")
            except Exception as e:
                console.print(f"  [red]Failed to load skill: {e}[/red]")
        return None

    # Browse mode: interactive table
    if not skills:
        console.print()
        console.print("  [dim]No skills found. Create skills in ~/.aura/skills/[/dim]")
        console.print()
        return None

    # Group by directory
    table = Table(box=None, padding=(0, 1), show_header=True, header_style="bold")
    table.add_column("#", width=3, justify="right", style="dim")
    table.add_column("Skill", style="bold", width=30)
    table.add_column("Description", min_width=40, style="dim")
    table.add_column("Source", width=20, style="dim")

    for i, skill in enumerate(skills, 1):
        source = _source_label(skill["path"])
        table.add_row(str(i), skill["name"], skill.get("description", ""), source)

    console.print()
    console.print(Panel(
        table,
        title="[bold cyan]📚 /skill[/bold cyan]",
        subtitle=f"[dim]{len(skills)} skills available · /skill <name> to load[/dim]",
        border_style="cyan",
        padding=(1, 2),
    ))
    console.print()

    # Interactive picker
    if skills:
        choice = _pick_skill(skills, console)
        if choice:
            # Recurse to load
            return handle_skill(agent, choice["name"], context)

    return None


def _source_label(path: str) -> str:
    """Shorten a skill path for display."""
    path = path.replace("\\", "/")
    if "/aura_skill_library/" in path:
        return "library"
    if "/skills/" in path:
        parts = path.split("/skills/", 1)
        if len(parts) > 1:
            return parts[1].split("/SKILL.md")[0].split("/")[0][:18]
    return path.split("/")[-2][:18]


def _pick_skill(skills: list[dict], console: Any) -> Optional[dict]:
    """Interactive numbered picker for skills."""
    try:
        choice = console.input("  [bold]Load skill # (Enter to skip): [/bold]").strip()
    except (EOFError, KeyboardInterrupt):
        return None
    if not choice:
        return None
    try:
        idx = int(choice) - 1
        if 0 <= idx < len(skills):
            return skills[idx]
    except ValueError:
        # Try matching by name
        for s in skills:
            if choice.lower() in s["name"].lower():
                return s
    return None
