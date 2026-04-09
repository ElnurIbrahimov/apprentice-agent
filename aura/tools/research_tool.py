"""Research Tool — save, organize, search, and manage research files."""

import json
import logging
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

RESEARCH_DIR = Path(__file__).parent.parent.parent / "research"
SKILLS_DIR = Path(__file__).parent.parent.parent / "skills"
INDEX_FILE = RESEARCH_DIR / "_index.json"

# Categories map to subdirectories
CATEGORIES = {
    "architecture": "architecture",
    "ai-models": "ai-models",
    "ai_models": "ai-models",
    "models": "ai-models",
    "consciousness": "consciousness",
    "memory": "memory-systems",
    "memory-systems": "memory-systems",
    "tools": "tools",
    "papers": "papers",
    "apis": "apis",
    "integrations": "integrations",
    "patterns": "patterns",
    "troubleshooting": "troubleshooting",
    "claude-code": "claude-code",
    "claude": "claude-code",
}

# Skill categories
SKILL_CATEGORIES = {
    "patterns": "patterns",
    "prompts": "prompts",
    "workflows": "workflows",
    "troubleshooting": "troubleshooting",
}


class ResearchTool:
    """Save, organize, search, and manage research and skills files."""

    name = "research"
    description = "Save and search research notes, findings, and skills"

    def __init__(self):
        self._ensure_dirs()
        self._ensure_index()

    def _ensure_dirs(self):
        """Create all research and skills directories."""
        RESEARCH_DIR.mkdir(parents=True, exist_ok=True)
        for subdir in CATEGORIES.values():
            (RESEARCH_DIR / subdir).mkdir(parents=True, exist_ok=True)
        SKILLS_DIR.mkdir(parents=True, exist_ok=True)
        for subdir in SKILL_CATEGORIES.values():
            (SKILLS_DIR / subdir).mkdir(parents=True, exist_ok=True)

    def _ensure_index(self):
        """Create index file if it doesn't exist."""
        if not INDEX_FILE.exists():
            self._save_index({"entries": [], "tags": {}})

    def _load_index(self) -> dict:
        try:
            with open(INDEX_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {"entries": [], "tags": {}}

    def _save_index(self, data: dict) -> bool:
        try:
            with open(INDEX_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            return True
        except IOError:
            return False

    def _slugify(self, text: str) -> str:
        """Convert text to filename-safe slug."""
        slug = text.lower().strip()
        slug = re.sub(r'[^\w\s-]', '', slug)
        slug = re.sub(r'[\s_]+', '-', slug)
        slug = re.sub(r'-+', '-', slug)
        return slug[:80].strip('-')

    def _resolve_category(self, category: str) -> str:
        """Resolve category name to directory."""
        cat = category.lower().strip()
        resolved = CATEGORIES.get(cat)
        if resolved is None:
            raise ValueError(f"Unknown category: {category!r}. Allowed: {list(CATEGORIES.keys())}")
        return resolved

    # -- Core Operations ---------------------------------------------------

    def save(self, title: str, content: str, category: str = "tools",
             tags: Optional[List[str]] = None, sources: Optional[List[str]] = None) -> dict:
        """Save a research note to a file.

        Args:
            title: Research title (becomes filename)
            content: Research content (markdown)
            category: Category directory (architecture, ai-models, consciousness, memory, tools, papers, apis, integrations, patterns, troubleshooting)
            tags: Optional tags for indexing
            sources: Optional list of source URLs
        """
        if not title or not content:
            return {"success": False, "error": "Title and content are required"}

        cat_dir = self._resolve_category(category)
        target_dir = RESEARCH_DIR / cat_dir
        target_dir.mkdir(parents=True, exist_ok=True)

        slug = self._slugify(title)
        filename = f"{slug}.md"
        filepath = target_dir / filename

        # Build markdown content with metadata header
        header = f"# {title}\n\n"
        if tags:
            header += f"**Tags:** {', '.join(tags)}\n"
        if sources:
            header += f"**Sources:** {', '.join(sources)}\n"
        header += f"**Created:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
        header += f"**Category:** {category}\n\n---\n\n"

        full_content = header + content

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(full_content)
        except IOError as e:
            return {"success": False, "error": f"Could not save file: {e}"}

        # Update index
        entry_id = uuid.uuid4().hex[:8]
        index = self._load_index()
        entry = {
            "id": entry_id,
            "title": title,
            "filename": filename,
            "category": cat_dir,
            "path": str(filepath),
            "tags": tags or [],
            "sources": sources or [],
            "created_at": datetime.now().isoformat(),
            "word_count": len(content.split()),
        }
        index["entries"].append(entry)

        # Update tag index
        for tag in (tags or []):
            tag_lower = tag.lower()
            if tag_lower not in index["tags"]:
                index["tags"][tag_lower] = []
            index["tags"][tag_lower].append(entry_id)

        self._save_index(index)

        return {
            "success": True,
            "entry_id": entry_id,
            "path": str(filepath),
            "category": cat_dir,
            "word_count": len(content.split()),
            "response": f"Saved research: '{title}' -> {cat_dir}/{filename} ({len(content.split())} words)"
        }

    def save_skill(self, title: str, content: str, category: str = "patterns") -> dict:
        """Save a skill/pattern to the skills directory.

        Args:
            title: Skill title
            content: Skill content (markdown)
            category: patterns, prompts, workflows, troubleshooting
        """
        if not title or not content:
            return {"success": False, "error": "Title and content are required"}

        cat = SKILL_CATEGORIES.get(category.lower(), category.lower())
        target_dir = SKILLS_DIR / cat
        target_dir.mkdir(parents=True, exist_ok=True)

        slug = self._slugify(title)
        filename = f"{slug}.md"
        filepath = target_dir / filename

        header = f"# {title}\n\n"
        header += f"**Category:** {category}\n"
        header += f"**Created:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n---\n\n"

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(header + content)
        except IOError as e:
            return {"success": False, "error": f"Could not save: {e}"}

        return {
            "success": True,
            "path": str(filepath),
            "response": f"Saved skill: '{title}' -> skills/{cat}/{filename}"
        }

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def _semantic_search(self, query: str, search_dirs: List[Path], top_k: int = 3) -> List[Dict]:
        """Fallback semantic search using embeddings when substring search finds nothing."""
        try:
            from aura.memory.embedding import get_embedding
        except ImportError:
            return []

        query_emb = get_embedding(query)
        if not query_emb:
            return []

        scored = []
        seen_dirs = set()
        for d in search_dirs:
            d_str = str(d)
            if d_str in seen_dirs:
                continue
            seen_dirs.add(d_str)
            if not d.exists():
                continue
            for f in d.glob("*.md"):
                try:
                    text = f.read_text(encoding="utf-8")
                    # Use first 2000 chars for embedding to keep it fast
                    file_emb = get_embedding(text[:2000])
                    if not file_emb:
                        continue
                    sim = self._cosine_similarity(query_emb, file_emb)
                    # Extract first non-empty line for context
                    first_line = next((l.strip() for l in text.split("\n") if l.strip()), "")
                    scored.append((sim, {
                        "file": f.name,
                        "path": str(f),
                        "category": f.parent.name,
                        "context": first_line[:120],
                        "similarity": round(sim, 3),
                    }))
                except Exception:
                    continue

        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored[:top_k]]

    def search(self, query: str, category: Optional[str] = None) -> dict:
        """Search across all research files by content.

        Falls back to semantic search (embeddings) when substring search returns 0 results.

        Args:
            query: Search text
            category: Optional category filter
        """
        if not query:
            return {"success": False, "error": "No search query"}

        q = query.lower()
        results = []
        search_dirs = []

        if category:
            cat_dir = self._resolve_category(category)
            search_dirs.append(RESEARCH_DIR / cat_dir)
        else:
            # Search all research dirs
            for subdir in CATEGORIES.values():
                d = RESEARCH_DIR / subdir
                if d.exists():
                    search_dirs.append(d)
            # Also search skills
            for subdir in SKILL_CATEGORIES.values():
                d = SKILLS_DIR / subdir
                if d.exists():
                    search_dirs.append(d)

        seen_dirs = set()
        for d in search_dirs:
            d_str = str(d)
            if d_str in seen_dirs:
                continue
            seen_dirs.add(d_str)
            if not d.exists():
                continue
            for f in d.glob("*.md"):
                try:
                    text = f.read_text(encoding="utf-8")
                    if q in text.lower():
                        # Extract first matching line for context
                        match_lines = [line.strip() for line in text.split("\n")
                                       if q in line.lower() and line.strip()]
                        context = match_lines[0][:120] if match_lines else ""
                        results.append({
                            "file": f.name,
                            "path": str(f),
                            "category": f.parent.name,
                            "context": context,
                        })
                except Exception:
                    continue

        # Semantic fallback: if substring search returned nothing, try embeddings
        search_method = "substring"
        if not results:
            try:
                semantic_results = self._semantic_search(query, search_dirs, top_k=3)
                if semantic_results:
                    results = semantic_results
                    search_method = "semantic"
            except Exception as e:
                logger.debug(f"[Research] Semantic search fallback failed: {e}")

        formatted = []
        for r in results[:30]:
            sim_note = f" (sim: {r['similarity']})" if "similarity" in r else ""
            formatted.append(f"  [{r['category']}] {r['file']}: {r['context']}{sim_note}")

        return {
            "success": True,
            "count": len(results),
            "search_method": search_method,
            "results": results[:30],
            "response": f"Found {len(results)} match(es) for '{query}' ({search_method}):\n" +
                        "\n".join(formatted) if results else f"No results for '{query}'"
        }

    def search_by_tag(self, tag: str) -> dict:
        """Search research by tag."""
        index = self._load_index()
        tag_lower = tag.lower()
        entry_ids = index.get("tags", {}).get(tag_lower, [])

        if not entry_ids:
            return {"success": True, "count": 0, "results": [],
                    "response": f"No research tagged '{tag}'"}

        matching = [e for e in index["entries"] if e["id"] in entry_ids]
        formatted = [f"  [{e['category']}] {e['title']}" for e in matching]

        return {
            "success": True,
            "count": len(matching),
            "results": matching,
            "response": f"{len(matching)} research note(s) tagged '{tag}':\n" + "\n".join(formatted)
        }

    def list_research(self, category: Optional[str] = None) -> dict:
        """List all research files.

        Args:
            category: Optional category filter
        """
        if category:
            cat_dir = self._resolve_category(category)
            target = RESEARCH_DIR / cat_dir
            if not target.exists():
                return {"success": True, "count": 0, "files": [],
                        "response": f"No research in '{category}'"}
            files = sorted(target.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True)
            items = [{"name": f.name, "category": cat_dir, "path": str(f),
                       "size": f.stat().st_size} for f in files]
        else:
            items = []
            for subdir in sorted(set(CATEGORIES.values())):
                d = RESEARCH_DIR / subdir
                if d.exists():
                    for f in sorted(d.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True):
                        if f.name.startswith("_"):
                            continue
                        items.append({"name": f.name, "category": subdir,
                                       "path": str(f), "size": f.stat().st_size})

        formatted = []
        for item in items:
            formatted.append(f"  [{item['category']}] {item['name']} ({item['size']} bytes)")

        return {
            "success": True,
            "count": len(items),
            "files": items,
            "response": f"{len(items)} research file(s):\n" + "\n".join(formatted) if items else "No research files"
        }

    def list_skills(self) -> dict:
        """List all skill files."""
        items = []
        for subdir in sorted(SKILL_CATEGORIES.values()):
            d = SKILLS_DIR / subdir
            if d.exists():
                for f in sorted(d.glob("*.md")):
                    items.append({"name": f.name, "category": subdir, "path": str(f)})

        formatted = [f"  [{i['category']}] {i['name']}" for i in items]
        return {
            "success": True,
            "count": len(items),
            "files": items,
            "response": f"{len(items)} skill(s):\n" + "\n".join(formatted) if items else "No skills saved"
        }

    def read(self, filename: str, category: Optional[str] = None) -> dict:
        """Read a research file.

        Args:
            filename: File name (with or without .md extension)
            category: Optional category to narrow search
        """
        if not filename.endswith(".md"):
            filename += ".md"

        if category:
            cat_dir = self._resolve_category(category)
            filepath = RESEARCH_DIR / cat_dir / filename
            if filepath.exists():
                text = filepath.read_text(encoding="utf-8")
                return {"success": True, "content": text, "path": str(filepath),
                        "response": text[:3000]}
        else:
            # Search all directories
            for subdir in set(CATEGORIES.values()):
                filepath = RESEARCH_DIR / subdir / filename
                if filepath.exists():
                    text = filepath.read_text(encoding="utf-8")
                    return {"success": True, "content": text, "path": str(filepath),
                            "response": text[:3000]}
            # Check skills too
            for subdir in SKILL_CATEGORIES.values():
                filepath = SKILLS_DIR / subdir / filename
                if filepath.exists():
                    text = filepath.read_text(encoding="utf-8")
                    return {"success": True, "content": text, "path": str(filepath),
                            "response": text[:3000]}

        return {"success": False, "error": f"File not found: {filename}"}

    def delete(self, filename: str, category: Optional[str] = None) -> dict:
        """Delete a research file."""
        if not filename.endswith(".md"):
            filename += ".md"

        filepath = None
        if category:
            cat_dir = self._resolve_category(category)
            candidate = RESEARCH_DIR / cat_dir / filename
            if candidate.exists():
                filepath = candidate
        else:
            for subdir in set(CATEGORIES.values()):
                candidate = RESEARCH_DIR / subdir / filename
                if candidate.exists():
                    filepath = candidate
                    break

        if not filepath:
            return {"success": False, "error": f"File not found: {filename}"}

        try:
            filepath.unlink()
            # Remove from index
            index = self._load_index()
            index["entries"] = [e for e in index["entries"] if e.get("filename") != filename]
            self._save_index(index)
            return {"success": True, "response": f"Deleted: {filename}"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def stats(self) -> dict:
        """Research statistics."""
        total_files = 0
        total_size = 0
        by_category = {}

        for subdir in set(CATEGORIES.values()):
            d = RESEARCH_DIR / subdir
            if d.exists():
                files = list(d.glob("*.md"))
                count = len([f for f in files if not f.name.startswith("_")])
                size = sum(f.stat().st_size for f in files if not f.name.startswith("_"))
                if count > 0:
                    by_category[subdir] = {"count": count, "size": size}
                total_files += count
                total_size += size

        skill_count = 0
        for subdir in SKILL_CATEGORIES.values():
            d = SKILLS_DIR / subdir
            if d.exists():
                skill_count += len(list(d.glob("*.md")))

        index = self._load_index()
        tag_count = len(index.get("tags", {}))

        return {
            "success": True,
            "total_research_files": total_files,
            "total_size_bytes": total_size,
            "total_skills": skill_count,
            "total_tags": tag_count,
            "by_category": by_category,
            "response": f"Research: {total_files} files ({total_size // 1024}KB), "
                        f"{skill_count} skills, {tag_count} tags\n" +
                        "\n".join(f"  {cat}: {info['count']} files" for cat, info in sorted(by_category.items()))
        }

    # -- Dispatch -----------------------------------------------------------

    def execute(self, action: str, **kwargs) -> dict:
        action_lower = action.lower().strip()

        # Save research
        if action_lower.startswith("save") and "skill" not in action_lower:
            title = kwargs.get("title", "")
            content = kwargs.get("content", "")
            category = kwargs.get("category", "tools")
            tags = kwargs.get("tags")
            sources = kwargs.get("sources")
            if not title and not content:
                # Try to parse: "save <title>"
                parts = action.split(None, 1)
                if len(parts) > 1:
                    title = parts[1]
            return self.save(title=title, content=content, category=category,
                            tags=tags, sources=sources)

        # Save skill
        if action_lower.startswith("save_skill") or action_lower.startswith("save skill"):
            title = kwargs.get("title", "")
            content = kwargs.get("content", "")
            category = kwargs.get("category", "patterns")
            return self.save_skill(title=title, content=content, category=category)

        # Search
        if action_lower.startswith("search") or action_lower.startswith("find"):
            query = kwargs.get("query", "")
            if not query and len(action.split()) > 1:
                query = action.split(None, 1)[-1]
            category = kwargs.get("category")
            return self.search(query, category=category)

        # Search by tag
        if action_lower.startswith("tag"):
            tag = kwargs.get("tag", "")
            if not tag and len(action.split()) > 1:
                tag = action.split(None, 1)[-1]
            return self.search_by_tag(tag)

        # List research
        if action_lower.startswith("list") and "skill" not in action_lower:
            category = kwargs.get("category")
            if not category and len(action.split()) > 1:
                category = action.split(None, 1)[-1]
            return self.list_research(category=category)

        # List skills
        if action_lower in ("skills", "list_skills", "list skills"):
            return self.list_skills()

        # Read
        if action_lower.startswith("read") or action_lower.startswith("open"):
            filename = kwargs.get("filename", "")
            if not filename and len(action.split()) > 1:
                filename = action.split(None, 1)[-1]
            category = kwargs.get("category")
            return self.read(filename, category=category)

        # Delete
        if action_lower.startswith("delete") or action_lower.startswith("remove"):
            filename = kwargs.get("filename", "")
            if not filename and len(action.split()) > 1:
                filename = action.split(None, 1)[-1]
            category = kwargs.get("category")
            return self.delete(filename, category=category)

        # Stats
        if action_lower in ("stats", "statistics", "info", "status"):
            return self.stats()

        # Default: show stats
        return self.stats()


# Singleton
research_tool = ResearchTool()
