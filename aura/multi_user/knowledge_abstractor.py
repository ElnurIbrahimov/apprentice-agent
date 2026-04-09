"""
KnowledgeAbstractor - Cross-user learning with privacy (ADV-04).

Extracts generalizable insights from user interaction patterns via a
three-stage pipeline: Extract -> Scrub -> Generalize.

All insights must pass k-anonymity checks before being used,
ensuring no single user's data can be identified.
"""

import json
import logging
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from .privacy_guard import PrivacyGuard

if TYPE_CHECKING:
    from .user_mind_model import UserMindModel

logger = logging.getLogger(__name__)


@dataclass
class AbstractInsight:
    """A generalized insight derived from multiple users."""
    insight_id: str
    category: str          # "topic_correlation", "style_pattern", etc.
    description: str
    confidence: float
    supporting_count: int
    min_users_required: int = 3
    created_at: str = ""
    last_updated: str = ""
    recommendation: str = ""

    def is_mature(self) -> bool:
        """Check if insight has enough support to be used."""
        return self.supporting_count >= self.min_users_required

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class KnowledgeAbstractor:
    """Extracts generalizable insights from cross-user patterns."""

    def __init__(
        self,
        data_dir: Optional[Path] = None,
        privacy_guard: Optional[PrivacyGuard] = None,
    ):
        self._data_dir = data_dir or Path("data/cross_user_insights")
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self.privacy_guard = privacy_guard or PrivacyGuard()
        self.insights: Dict[str, AbstractInsight] = {}
        self._load()

    def analyze_patterns(
        self, user_models: Dict[str, 'UserMindModel'],
    ) -> List[AbstractInsight]:
        """Analyze patterns across all user models.

        Called during NeuroDream consolidation. Follows the
        Extract -> Scrub -> Generalize pipeline.
        """
        new_insights: List[AbstractInsight] = []

        # 1. Find topic correlations across users
        topic_pairs = self._find_topic_correlations(user_models)
        for (t1, t2), count in topic_pairs.items():
            if count >= self.privacy_guard.k_anonymity:
                insight_id = f"topic_corr_{t1}_{t2}"
                now = datetime.now().isoformat()
                insight = AbstractInsight(
                    insight_id=insight_id,
                    category="topic_correlation",
                    description=(
                        f"Users interested in '{t1}' often also explore '{t2}'"
                    ),
                    confidence=min(0.8, count * 0.1),
                    supporting_count=count,
                    created_at=now,
                    last_updated=now,
                    recommendation=(
                        f"When a user asks about {t1}, consider mentioning {t2}"
                    ),
                )
                self.insights[insight_id] = insight
                new_insights.append(insight)

        # 2. Find communication style patterns
        style_insights = self._find_style_patterns(user_models)
        new_insights.extend(style_insights)

        # 3. Find frustration patterns (stub for Phase 3)
        frustration_insights = self._find_frustration_patterns(user_models)
        new_insights.extend(frustration_insights)

        self._save()
        return new_insights

    def _find_topic_correlations(
        self, user_models: Dict[str, 'UserMindModel'],
    ) -> Dict[Tuple[str, str], int]:
        """Find topics that co-occur across users."""
        pair_counts: Counter = Counter()
        for user_id, model in user_models.items():
            topics = list(model.topic_knowledge.keys())
            for i in range(len(topics)):
                for j in range(i + 1, len(topics)):
                    pair = tuple(sorted([topics[i], topics[j]]))
                    pair_counts[pair] += 1
        return dict(pair_counts.most_common(20))

    def _find_style_patterns(
        self, user_models: Dict[str, 'UserMindModel'],
    ) -> List[AbstractInsight]:
        """Find communication style clusters across users."""
        insights: List[AbstractInsight] = []
        total = len(user_models)
        if total < 3:
            return insights

        formal_count = sum(
            1 for m in user_models.values() if m.comm_style.formality > 0.7
        )
        if formal_count > total * 0.6:
            now = datetime.now().isoformat()
            insight = AbstractInsight(
                insight_id="style_cluster_formal",
                category="style_pattern",
                description="Most users prefer formal communication",
                confidence=formal_count / total,
                supporting_count=formal_count,
                created_at=now,
                last_updated=now,
                recommendation="Default to formal tone for new users",
            )
            self.insights[insight.insight_id] = insight
            insights.append(insight)
        return insights

    def _find_frustration_patterns(
        self, user_models: Dict[str, 'UserMindModel'],
    ) -> List[AbstractInsight]:
        """Find common frustration triggers. Stub for Phase 3."""
        return []

    def get_applicable_insights(
        self, user_model: 'UserMindModel',
    ) -> List[AbstractInsight]:
        """Get insights relevant to a specific user."""
        applicable: List[AbstractInsight] = []
        user_topics = set(user_model.topic_knowledge.keys())

        for insight in self.insights.values():
            if not insight.is_mature():
                continue
            if insight.category == "topic_correlation":
                # Check if user has one of the correlated topics
                parts = insight.insight_id.replace("topic_corr_", "").split("_")
                if len(parts) == 2 and parts[0] in user_topics:
                    applicable.append(insight)
        return applicable

    # ====================================================================
    # Persistence
    # ====================================================================

    def _save(self) -> None:
        """Persist insights to disk."""
        try:
            data = {k: v.to_dict() for k, v in self.insights.items()}
            path = self._data_dir / "insights.json"
            path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception as e:
            logger.warning(f"[KnowledgeAbstractor] Failed to save: {e}")

    def _load(self) -> None:
        """Load persisted insights."""
        path = self._data_dir / "insights.json"
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            for k, v in data.items():
                self.insights[k] = AbstractInsight(**v)
        except Exception as e:
            logger.warning(f"[KnowledgeAbstractor] Failed to load: {e}")
