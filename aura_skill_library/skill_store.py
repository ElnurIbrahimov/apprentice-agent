"""
AURA Skill Library - Storage Backend

Disk-based skill storage with embedding index for semantic retrieval.
Zero VRAM - embeddings computed on CPU.
"""

import json
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timezone
import numpy as np

from .skill import Skill, SkillCategory, SkillMetadata

# Lazy-loaded to avoid ~5s startup penalty
SENTENCE_TRANSFORMERS_AVAILABLE = True
_ST_CLASS = None

def _get_st():
    global _ST_CLASS, SENTENCE_TRANSFORMERS_AVAILABLE
    if _ST_CLASS is not None:
        return _ST_CLASS
    try:
        from sentence_transformers import SentenceTransformer
        _ST_CLASS = SentenceTransformer
        return _ST_CLASS
    except ImportError:
        SENTENCE_TRANSFORMERS_AVAILABLE = False
        return None

SentenceTransformer = None  # placeholder

logger = logging.getLogger(__name__)


class SkillStore:
    """
    Persistent skill storage with semantic search.
    Zero VRAM - embeddings computed on CPU.
    """

    def __init__(
        self,
        storage_path: str = "./aura_skills",
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        """
        Initialize skill store.

        Args:
            storage_path: Path to skill storage directory
            embedding_model: Name of sentence-transformers model
        """
        self.storage_path = Path(storage_path)
        self.index_path = self.storage_path / "index.json"
        self.categories_path = self.storage_path / "categories"
        self.learned_path = self.storage_path / "learned"

        # Initialize storage structure
        self._init_storage()

        # Load embedding model (CPU only, ~100MB)
        self._embedder = None
        self._embedding_model_name = embedding_model

        # Load index
        self.index: Dict[str, Dict] = self._load_index()

        logger.info(f"SkillStore initialized at {storage_path} with {len(self.index)} skills")

    @property
    def embedder(self):
        """Lazy load embedding model."""
        if self._embedder is None:
            ST = _get_st()
            if ST is not None:
                logger.info(f"Loading embedding model: {self._embedding_model_name}")
                self._embedder = ST(self._embedding_model_name, device='cpu')
        return self._embedder

    def _init_storage(self):
        """Create storage directory structure."""
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.categories_path.mkdir(exist_ok=True)
        self.learned_path.mkdir(exist_ok=True)

        # Create category directories
        for category in SkillCategory:
            (self.categories_path / category.value).mkdir(exist_ok=True)

        # Initialize index if not exists
        if not self.index_path.exists():
            self._save_index({})

    def _load_index(self) -> Dict[str, Dict]:
        """Load skill index from disk."""
        if self.index_path.exists():
            try:
                with open(self.index_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load index: {e}")
                return {}
        return {}

    def _save_index(self, index: Dict[str, Dict]):
        """Save skill index to disk."""
        with open(self.index_path, 'w', encoding='utf-8') as f:
            json.dump(index, f, indent=2, default=str)

    def _get_skill_path(self, skill: Skill) -> Path:
        """Get the file path for a skill."""
        filename = f"{skill.name.lower().replace(' ', '-').replace('/', '-')}.skill.md"
        if skill.id.startswith("learned_"):
            return self.learned_path / filename
        return self.categories_path / skill.category.value / filename

    def _compute_embedding(self, skill: Skill) -> List[float]:
        """Compute embedding for skill retrieval."""
        if self.embedder is None:
            return []

        # Combine description + triggers for embedding
        text = f"{skill.name}. {skill.description}. " + " ".join(skill.trigger_patterns)
        embedding = self.embedder.encode(text, convert_to_numpy=True, show_progress_bar=False)
        return embedding.tolist()

    def save(self, skill: Skill) -> str:
        """
        Save a skill to disk and update index.

        Args:
            skill: Skill to save

        Returns:
            The skill ID
        """
        # Write skill file
        skill_path = self._get_skill_path(skill)
        skill_path.parent.mkdir(parents=True, exist_ok=True)

        with open(skill_path, 'w', encoding='utf-8') as f:
            f.write(skill.to_markdown())

        # Update index
        self.index[skill.id] = {
            "name": skill.name,
            "description": skill.description,
            "category": skill.category.value,
            "path": str(skill_path.relative_to(self.storage_path)),
            "trigger_patterns": skill.trigger_patterns,
            "embedding": self._compute_embedding(skill),
            "success_rate": skill.metadata.success_rate,
            "total_uses": skill.metadata.total_uses,
            "tags": skill.metadata.tags,
            "updated_at": datetime.now(timezone.utc).isoformat()
        }
        self._save_index(self.index)

        logger.debug(f"Saved skill: {skill.name} ({skill.id})")
        return skill.id

    def load(self, skill_id: str) -> Optional[Skill]:
        """
        Load a skill by ID.

        Args:
            skill_id: Skill ID

        Returns:
            Skill or None if not found
        """
        if skill_id not in self.index:
            return None

        skill_info = self.index[skill_id]
        skill_path = self.storage_path / skill_info["path"]

        if not skill_path.exists():
            logger.warning(f"Skill file not found: {skill_path}")
            return None

        try:
            with open(skill_path, 'r', encoding='utf-8') as f:
                return Skill.from_markdown(f.read())
        except Exception as e:
            logger.error(f"Failed to load skill {skill_id}: {e}")
            return None

    def delete(self, skill_id: str) -> bool:
        """
        Delete a skill.

        Args:
            skill_id: Skill ID to delete

        Returns:
            True if deleted, False if not found
        """
        if skill_id not in self.index:
            return False

        skill_info = self.index[skill_id]
        skill_path = self.storage_path / skill_info["path"]

        if skill_path.exists():
            skill_path.unlink()

        del self.index[skill_id]
        self._save_index(self.index)

        logger.debug(f"Deleted skill: {skill_id}")
        return True

    def search(
        self,
        query: str,
        limit: int = 5,
        category: Optional[SkillCategory] = None,
        min_success_rate: float = 0.0
    ) -> List[Tuple[str, float]]:
        """
        Semantic search for relevant skills.

        Args:
            query: Search query
            limit: Maximum results
            category: Optional category filter
            min_success_rate: Minimum success rate filter

        Returns:
            List of (skill_id, similarity_score)
        """
        if not self.index:
            return []

        if self.embedder is None:
            # Fall back to trigger matching if no embedder
            return self.search_by_trigger(query, threshold=0.0)[:limit]

        # Compute query embedding
        query_embedding = self.embedder.encode(query, convert_to_numpy=True, show_progress_bar=False)

        results = []
        for skill_id, info in self.index.items():
            # Apply filters
            if category and info["category"] != category.value:
                continue
            if info.get("success_rate", 0) < min_success_rate:
                continue

            # Compute similarity
            skill_embedding = info.get("embedding", [])
            if not skill_embedding:
                continue

            skill_embedding = np.array(skill_embedding)
            similarity = float(np.dot(query_embedding, skill_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(skill_embedding) + 1e-8
            ))

            results.append((skill_id, similarity))

        # Sort by similarity
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]

    def search_by_trigger(
        self,
        text: str,
        threshold: float = 0.7
    ) -> List[Tuple[str, float]]:
        """
        Find skills whose trigger patterns match the input text.

        Args:
            text: Input text to match
            threshold: Minimum similarity threshold

        Returns:
            List of (skill_id, score) tuples
        """
        text_lower = text.lower()
        matches = []

        for skill_id, info in self.index.items():
            best_score = 0.0

            # Check exact trigger matches
            for pattern in info.get("trigger_patterns", []):
                if pattern.lower() in text_lower:
                    best_score = 1.0
                    break

            # Fall back to semantic search if embedder available
            if best_score == 0 and self.embedder is not None:
                skill_embedding = info.get("embedding", [])
                if skill_embedding:
                    query_emb = self.embedder.encode(text, convert_to_numpy=True, show_progress_bar=False)
                    skill_emb = np.array(skill_embedding)
                    similarity = float(np.dot(query_emb, skill_emb) / (
                        np.linalg.norm(query_emb) * np.linalg.norm(skill_emb) + 1e-8
                    ))
                    best_score = similarity

            if best_score >= threshold:
                matches.append((skill_id, best_score))

        # Sort by score
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches

    def list_all(
        self,
        category: Optional[SkillCategory] = None,
        sort_by: str = "success_rate"
    ) -> List[Dict]:
        """
        List all skills with metadata.

        Args:
            category: Optional category filter
            sort_by: Sort field (success_rate, uses, name, updated)

        Returns:
            List of skill info dictionaries
        """
        skills = []
        for skill_id, info in self.index.items():
            if category and info["category"] != category.value:
                continue
            skills.append({
                "id": skill_id,
                **info
            })

        # Sort
        if sort_by == "success_rate":
            skills.sort(key=lambda x: x.get("success_rate", 0), reverse=True)
        elif sort_by == "uses":
            skills.sort(key=lambda x: x.get("total_uses", 0), reverse=True)
        elif sort_by == "name":
            skills.sort(key=lambda x: x["name"])
        elif sort_by == "updated":
            skills.sort(key=lambda x: x.get("updated_at", ""), reverse=True)

        return skills

    def get_stats(self) -> Dict[str, Any]:
        """Get library statistics."""
        total = len(self.index)
        by_category = {}
        total_uses = 0
        total_success = 0

        for info in self.index.values():
            cat = info["category"]
            by_category[cat] = by_category.get(cat, 0) + 1
            uses = info.get("total_uses", 0)
            total_uses += uses
            total_success += int(info.get("success_rate", 0) * uses)

        return {
            "total_skills": total,
            "by_category": by_category,
            "total_uses": total_uses,
            "overall_success_rate": total_success / total_uses if total_uses > 0 else 0,
            "learned_skills": len(list(self.learned_path.glob("*.skill.md")))
        }

    def rebuild_index(self):
        """Rebuild index from skill files on disk."""
        new_index = {}

        # Scan categories
        for category_dir in self.categories_path.iterdir():
            if category_dir.is_dir():
                for skill_file in category_dir.glob("*.skill.md"):
                    try:
                        with open(skill_file, 'r', encoding='utf-8') as f:
                            skill = Skill.from_markdown(f.read())
                            new_index[skill.id] = {
                                "name": skill.name,
                                "description": skill.description,
                                "category": skill.category.value,
                                "path": str(skill_file.relative_to(self.storage_path)),
                                "trigger_patterns": skill.trigger_patterns,
                                "embedding": self._compute_embedding(skill),
                                "success_rate": skill.metadata.success_rate,
                                "total_uses": skill.metadata.total_uses,
                                "tags": skill.metadata.tags,
                                "updated_at": skill.updated_at.isoformat()
                            }
                    except Exception as e:
                        logger.error(f"Failed to index {skill_file}: {e}")

        # Scan learned skills
        for skill_file in self.learned_path.glob("*.skill.md"):
            try:
                with open(skill_file, 'r', encoding='utf-8') as f:
                    skill = Skill.from_markdown(f.read())
                    new_index[skill.id] = {
                        "name": skill.name,
                        "description": skill.description,
                        "category": skill.category.value,
                        "path": str(skill_file.relative_to(self.storage_path)),
                        "trigger_patterns": skill.trigger_patterns,
                        "embedding": self._compute_embedding(skill),
                        "success_rate": skill.metadata.success_rate,
                        "total_uses": skill.metadata.total_uses,
                        "tags": skill.metadata.tags,
                        "updated_at": skill.updated_at.isoformat()
                    }
            except Exception as e:
                logger.error(f"Failed to index {skill_file}: {e}")

        self.index = new_index
        self._save_index(self.index)
        logger.info(f"Rebuilt index with {len(self.index)} skills")
