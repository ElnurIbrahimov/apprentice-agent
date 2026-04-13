"""Claude/OpenCode SKILL.md importer for Aura's SkillLibrary.

Claude Code and OpenCode use a minimal SKILL.md format:

    ---
    name: my-skill
    description: "What the skill does and when to use it."
    ---
    # Heading
    ...free-form markdown body...

Aura's native Skill format expects richer frontmatter (id, category,
trigger_patterns, version, etc). This module bridges the two formats so
Elnur's existing Claude Code skills drop-in to Aura's SkillLibrary.

Usage (Python):
    from aura_skill_library import SkillLibrary
    from aura_skill_library.skill_md_importer import import_skill_md_dir

    library = SkillLibrary(storage_path="D:/Aura/aura_skills")
    ids = import_skill_md_dir(
        source="C:/Users/asus/.claude/skills/ui-ux-pro-max",
        library=library,
        category_hint="coding",
    )

Usage (CLI):
    python -m aura_skill_library.skill_md_importer \
        C:/Users/asus/.claude/skills/ui-ux-pro-max coding

The importer is idempotent: re-running it updates existing entries rather
than creating duplicates, keyed by skill name.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)

# Category inference keywords — first match wins
_CATEGORY_KEYWORDS: List[tuple[str, List[str]]] = [
    ("coding", [
        "code", "coding", "develop", "implement", "refactor", "ui", "ux",
        "design", "frontend", "backend", "component", "react", "next",
        "tailwind", "shadcn", "typescript", "python", "tsx", "css", "html",
        "api", "database", "sql", "build", "debug", "test",
    ]),
    ("writing", [
        "write", "writing", "document", "documentation", "blog", "article",
        "markdown", "edit", "proofread", "translate", "draft",
    ]),
    ("research", [
        "research", "search", "paper", "arxiv", "literature", "review",
        "investigate", "analyze paper", "academic",
    ]),
    ("automation", [
        "automate", "automation", "schedule", "workflow", "pipeline",
        "cron", "hook", "trigger", "orchestrate",
    ]),
    ("analysis", [
        "analy", "inspect", "audit", "review", "evaluate", "assess",
        "metric", "benchmark",
    ]),
    ("communication", [
        "email", "message", "reply", "slack", "telegram", "respond",
        "notify", "report",
    ]),
    ("learning", [
        "learn", "study", "flashcard", "quiz", "spaced repetition",
    ]),
]

# Words that are too common to use as trigger patterns
_STOP_TOKENS = frozenset({
    "the", "a", "an", "and", "or", "but", "for", "to", "of", "in", "on",
    "at", "by", "with", "from", "as", "is", "are", "was", "were", "be",
    "been", "use", "used", "using", "this", "that", "when", "how", "what",
    "which", "why", "who", "it", "its", "any", "all", "some", "etc",
})


def _parse_frontmatter(content: str) -> tuple[Dict[str, Any], str]:
    """Extract YAML frontmatter and body from a SKILL.md file.

    Returns (frontmatter_dict, body_str). Frontmatter is {} if missing.
    """
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)$", content, re.DOTALL)
    if not match:
        return {}, content

    header = match.group(1)
    body = match.group(2)

    try:
        import yaml  # type: ignore

        data = yaml.safe_load(header) or {}
        if not isinstance(data, dict):
            data = {}
    except ImportError:
        # Minimal fallback parser — supports only flat `key: value` pairs.
        data = {}
        for line in header.splitlines():
            if ":" in line and not line.lstrip().startswith("#"):
                key, _, value = line.partition(":")
                data[key.strip()] = value.strip().strip('"').strip("'")

    return data, body


def _infer_category(description: str, path: Path) -> str:
    """Guess an Aura SkillCategory string from description + file path."""
    haystack = (description + " " + str(path)).lower()
    for category, keywords in _CATEGORY_KEYWORDS:
        for kw in keywords:
            if kw in haystack:
                return category
    return "custom"


def _infer_trigger_patterns(description: str, name: str) -> List[str]:
    """Produce 5–8 trigger phrases from a skill description.

    Splits on commas / semicolons / periods, filters stop-tokens, then
    seeds the list with the skill name itself so exact-name lookup works.
    """
    patterns: List[str] = [name.lower()]

    # Replace bullet punctuation with separators
    normalized = re.sub(r"[;·•]", ",", description)
    # Split on punctuation + line breaks
    chunks = re.split(r"[,\n.()]", normalized)

    for chunk in chunks:
        chunk = chunk.strip().lower()
        if not chunk or len(chunk) < 3:
            continue
        # Drop chunks that are just stop-tokens
        words = [w for w in re.findall(r"[a-z0-9]+", chunk) if w not in _STOP_TOKENS]
        if not words:
            continue
        phrase = " ".join(words)[:60]
        if phrase and phrase not in patterns:
            patterns.append(phrase)
        if len(patterns) >= 8:
            break

    return patterns


def _find_skill_md_files(source: Path) -> List[Path]:
    """Return SKILL.md files under `source`. Accepts a file, a skill dir, or a parent dir."""
    if source.is_file() and source.name.upper() == "SKILL.MD":
        return [source]

    if not source.exists():
        return []

    # A single skill dir: has SKILL.md at top level
    direct = source / "SKILL.md"
    if direct.is_file():
        return [direct]

    # A parent dir of many skill dirs: one level deep
    found: List[Path] = []
    for child in sorted(source.iterdir()):
        if child.is_dir():
            candidate = child / "SKILL.md"
            if candidate.is_file():
                found.append(candidate)
    return found


def _find_existing_skill_id(library: Any, name: str) -> Optional[str]:
    """Return existing skill ID for a given name, or None."""
    try:
        index = library.store.index  # type: ignore[attr-defined]
    except AttributeError:
        return None
    for skill_id, info in index.items():
        if info.get("name", "").strip().lower() == name.strip().lower():
            return skill_id
    return None


def import_skill_md_file(
    skill_md_path: Path,
    library: Any,
    category_hint: Optional[str] = None,
    tag_prefix: str = "imported",
) -> Optional[str]:
    """Import a single SKILL.md into Aura's SkillLibrary.

    Returns the skill ID on success, None on failure. Idempotent: an
    existing skill with the same name is overwritten.
    """
    try:
        content = skill_md_path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.error("[SKILL.md import] cannot read %s: %s", skill_md_path, exc)
        return None

    frontmatter, body = _parse_frontmatter(content)

    name = str(frontmatter.get("name") or skill_md_path.parent.name).strip()
    description = str(frontmatter.get("description") or "").strip()
    if not description:
        # Fall back to the first paragraph of the body
        first_para = next(
            (p.strip() for p in body.split("\n\n") if p.strip() and not p.startswith("#")),
            "",
        )
        description = first_para[:500]

    if not name:
        logger.warning("[SKILL.md import] skipping %s — no name", skill_md_path)
        return None

    category = (category_hint or _infer_category(description, skill_md_path)).lower()
    trigger_patterns = _infer_trigger_patterns(description, name)

    # Full body becomes the procedure. Keep a reference to the source path
    # so downstream tools (e.g. ui-ux-pro-max) can still shell-out to scripts.
    procedure_header = (
        f"_Source: `{skill_md_path}`_\n\n"
        f"This skill was imported from a Claude/OpenCode SKILL.md file. "
        f"To invoke its underlying CLI scripts (if any), look for a `scripts/` "
        f"sibling folder next to the source path above.\n\n"
    )
    procedure = procedure_header + body.strip()

    tags = [tag_prefix, category]
    existing_id = _find_existing_skill_id(library, name)
    if existing_id:
        # Update in place via the Skill object round-trip
        try:
            skill = library.get_skill(existing_id)
        except Exception:  # noqa: BLE001
            skill = None
        if skill is not None:
            skill.description = description
            skill.procedure = procedure
            skill.trigger_patterns = trigger_patterns
            skill.metadata.tags = sorted(set(skill.metadata.tags + tags))
            library.store.save(skill)
            logger.info("[SKILL.md import] updated '%s' (%s)", name, existing_id)
            return existing_id

    skill_id = library.create_skill(
        name=name,
        description=description or f"Imported skill: {name}",
        category=category,
        trigger_patterns=trigger_patterns,
        procedure=procedure,
        tags=tags,
    )
    logger.info("[SKILL.md import] created '%s' (%s)", name, skill_id)
    return skill_id


def import_skill_md_dir(
    source: str | Path,
    library: Any,
    category_hint: Optional[str] = None,
    tag_prefix: str = "imported",
) -> List[str]:
    """Import every SKILL.md found under `source` into the library.

    `source` may point at a single SKILL.md file, a skill folder containing
    one, or a parent folder containing many skill folders.
    """
    source_path = Path(source).expanduser()
    files = _find_skill_md_files(source_path)
    if not files:
        logger.warning("[SKILL.md import] no SKILL.md files under %s", source_path)
        return []

    imported: List[str] = []
    for skill_md in files:
        skill_id = import_skill_md_file(
            skill_md_path=skill_md,
            library=library,
            category_hint=category_hint,
            tag_prefix=tag_prefix,
        )
        if skill_id:
            imported.append(skill_id)
    return imported


def import_many(
    sources: Iterable[str | Path],
    library: Any,
    category_hint: Optional[str] = None,
) -> List[str]:
    """Import several source locations in one call."""
    all_ids: List[str] = []
    for src in sources:
        all_ids.extend(
            import_skill_md_dir(source=src, library=library, category_hint=category_hint)
        )
    return all_ids


def _cli(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m aura_skill_library.skill_md_importer",
        description="Import Claude/OpenCode SKILL.md files into Aura's SkillLibrary.",
    )
    parser.add_argument("source", help="SKILL.md file, skill dir, or parent dir")
    parser.add_argument(
        "category",
        nargs="?",
        default=None,
        help="Category hint (coding/writing/research/automation/analysis/communication/learning/custom). Auto-inferred if omitted.",
    )
    parser.add_argument(
        "--storage",
        default="D:/Aura/aura_skills",
        help="SkillLibrary storage path (default: D:/Aura/aura_skills)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    from aura_skill_library import SkillLibrary

    library = SkillLibrary(storage_path=args.storage)
    ids = import_skill_md_dir(
        source=args.source,
        library=library,
        category_hint=args.category,
    )
    print(f"imported {len(ids)} skill(s): {', '.join(ids) if ids else '<none>'}")
    library.shutdown()
    return 0 if ids else 1


if __name__ == "__main__":
    sys.exit(_cli())
