"""/skills and aura skills — skill hub management.

Browse, search, install, and uninstall skills from the community
skill library. Mirrors Hermes Agent's `hermes skills` pattern.

Skill sources:
  ~/.aura/skills/                  — user-installed skills
  ~/.aura/aura_skill_library/skills/ — community library (657 skills)
  aura_skills/                     — built-in skill library
  skills/community/               — community-imported skills
"""
from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import Any, Optional

from ..display import console, show_error, show_info, show_success
from .common import command, TIER_STABLE

logger = logging.getLogger(__name__)


def _get_skill_sources() -> list[Path]:
    """Get all directories to scan for skills."""
    sources = [
        Path.home() / ".aura" / "skills",
        Path.home() / ".aura" / "aura_skill_library" / "skills",
    ]
    # Project-local skill directories
    project_skills = Path(os.getcwd()) / "skills"
    if project_skills.exists():
        sources.append(project_skills)
    # Built-in
    builtin = Path(__file__).resolve().parent.parent.parent.parent / "aura_skills"
    if builtin.exists():
        sources.append(builtin)
    return [s for s in sources if s.exists()]


def _find_all_skills() -> list[dict]:
    """Scan all skill directories for SKILL.md files."""
    skills: list[dict] = []
    seen_names: set[str] = set()
    for base in _get_skill_sources():
        for skill_file in base.rglob("SKILL.md"):
            try:
                meta = _parse_skill_metadata(skill_file)
                if meta["name"] not in seen_names:
                    seen_names.add(meta["name"])
                    meta["path"] = str(skill_file)
                    meta["source_dir"] = str(base)
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
    return {"name": name, "description": description[:100]}


@command("/skills", "Browse, search, and manage skills", tier=TIER_STABLE)
def handle_skills(agent: Any, arg: str, context: dict) -> Optional[str]:
    """Skill hub management.

    Usage:
        /skills              List all installed skills
        /skills search QUERY  Search skills by name/description
        /skills install URL   Install a skill from a GitHub URL
        /skills uninstall N   Uninstall a skill by name
    """
    parts = (arg or "").strip().split(None, 1)
    sub = parts[0].lower() if parts else "list"

    if sub == "list" or sub == "":
        _list_skills()
    elif sub == "search" and len(parts) >= 2:
        _search_skills(parts[1].strip())
    elif sub == "install" and len(parts) >= 2:
        _install_skill(parts[1].strip())
    elif sub == "uninstall" and len(parts) >= 2:
        _uninstall_skill(parts[1].strip())
    else:
        show_info("Usage: /skills [list|search QUERY|install URL|uninstall NAME]")

    return None


def _list_skills() -> None:
    """List all installed skills."""
    from rich.table import Table
    from rich.panel import Panel

    skills = _find_all_skills()
    if not skills:
        show_info("No skills found. Install with /skills install <url>")
        return

    table = Table(box=None, padding=(0, 1), show_header=True, header_style="bold")
    table.add_column("#", width=4, justify="right")
    table.add_column("Name", style="bold cyan", min_width=25)
    table.add_column("Description", min_width=50)

    for i, s in enumerate(sorted(skills, key=lambda x: x["name"]), 1):
        table.add_row(str(i), s["name"][:35], s["description"][:70])

    console.print()
    console.print(Panel(
        table,
        title=f"[bold cyan]Skills  ({len(skills)} installed)[/bold cyan]",
        border_style="cyan",
        padding=(1, 2),
    ))
    console.print("  [dim]Use /skills search <query> to filter, /skills install <url> to add.[/dim]")
    console.print()


def _search_skills(query: str) -> None:
    """Search skills by fuzzy match on name + description."""
    skills = _find_all_skills()
    if not skills:
        show_info("No skills found.")
        return

    q = query.lower()
    matches = [
        s for s in skills
        if q in s["name"].lower() or q in s.get("description", "").lower()
    ]

    if not matches:
        show_info(f"No skills matching '{query}'.")
        return

    console.print(f"\n  [bold]Skills matching '{query}'[/bold] ({len(matches)} found):\n")
    for i, s in enumerate(sorted(matches, key=lambda x: x["name"]), 1):
        console.print(f"  [cyan]{i:>3}[/cyan]. [bold]{s['name'][:35]}[/bold]  [dim]{s['description'][:60]}[/dim]")


def _install_skill(url: str) -> None:
    """Install a skill from a GitHub URL or copy from local path."""
    target_dir = Path.home() / ".aura" / "skills"
    target_dir.mkdir(parents=True, exist_ok=True)

    if url.startswith("https://github.com/") or url.startswith("https://raw.githubusercontent.com/"):
        # GitHub URL — try to fetch SKILL.md
        try:
            import urllib.request
            # If it's a repo URL, try common SKILL.md paths
            if "github.com" in url and "/blob/" not in url:
                # Repo URL — try raw SKILL.md
                raw_url = url.replace("github.com", "raw.githubusercontent.com")
                if not raw_url.endswith("/SKILL.md"):
                    raw_url = raw_url.rstrip("/") + "/main/SKILL.md"
            else:
                raw_url = url

            resp = urllib.request.urlopen(raw_url, timeout=15)
            content = resp.read().decode("utf-8")

            # Extract skill name from URL, sanitized against path traversal.
            # Reject anything that isn't a simple slug to prevent escaping target_dir.
            raw_name = raw_url.rstrip("/").split("/")[-2] if raw_url.endswith("/SKILL.md") else raw_url.split("/")[-1].replace(".md", "")
            import re as _re
            skill_name = _re.sub(r"[^A-Za-z0-9_.-]", "_", raw_name).strip("._")[:60]
            if not skill_name or skill_name in (".", ".."):
                show_error("Could not derive safe skill name from URL.")
                return

            # Verify the resolved path stays within target_dir
            skill_dir = (target_dir / skill_name).resolve()
            try:
                skill_dir.relative_to(target_dir.resolve())
            except ValueError:
                show_error("Refusing to install: path escapes skills directory.")
                return

            skill_dir.mkdir(parents=True, exist_ok=True)
            (skill_dir / "SKILL.md").write_text(content, encoding="utf-8")

            show_success(f"Installed skill '{skill_name}' to {skill_dir}")
        except Exception as e:
            console.print(f"[red]Failed to install from URL: {e}[/red]")
    elif Path(url).exists() and Path(url).is_dir():
        # Local directory — copy. Sanitize the name and reject symlinks.
        raw_name = Path(url).name
        import re as _re
        skill_name = _re.sub(r"[^A-Za-z0-9_.-]", "_", raw_name).strip("._")[:60]
        if not skill_name or skill_name in (".", ".."):
            console.print(f"[red]Invalid skill name: {raw_name}[/red]")
            return
        dest = (target_dir / skill_name).resolve()
        try:
            dest.relative_to(target_dir.resolve())
        except ValueError:
            show_error("Refusing to install: path escapes skills directory.")
            return
        if dest.exists():
            console.print(f"[red]Skill '{skill_name}' already installed.[/red]")
            return
        shutil.copytree(url, dest)
        show_success(f"Installed skill '{skill_name}' from {url}")
    else:
        console.print(f"[red]Invalid URL or path: {url}[/red]")
        show_info("Usage: /skills install https://github.com/user/repo")


def _uninstall_skill(name: str) -> None:
    """Remove a skill by name."""
    import re as _re
    target_dir = (Path.home() / ".aura" / "skills").resolve()
    safe_name = _re.sub(r"[^A-Za-z0-9_.-]", "_", name).strip("._")[:60]
    if not safe_name or safe_name in (".", ".."):
        console.print(f"[red]Invalid skill name: {name}[/red]")
        return
    skill_dir = (target_dir / safe_name).resolve()
    # Verify the resolved path stays within target_dir.
    try:
        skill_dir.relative_to(target_dir)
    except ValueError:
        show_error("Refusing to uninstall: path escapes skills directory.")
        return

    if not skill_dir.exists():
        console.print(f"[red]Skill '{name}' not found in {target_dir}[/red]")
        return

    try:
        shutil.rmtree(skill_dir)
        show_success(f"Uninstalled skill '{name}'.")
    except Exception as e:
        console.print(f"[red]Failed to uninstall: {e}[/red]")
