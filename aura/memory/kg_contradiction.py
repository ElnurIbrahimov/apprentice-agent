"""
Knowledge Graph Contradiction & Supersession Support — Phase 3.

Adds semantic contradiction detection and supersession tracking to the
NetworkX-based KG in aura/tools/knowledge_graph.py.

Works as an overlay — does NOT rewrite the KG implementation.
Instead it:
  1. Inspects the existing graph for potential contradictions when a new
     fact arrives.
  2. Adds special "CONTRADICTS" and "SUPERSEDES" edge types.
  3. Marks nodes with a `lifecycle_state` property.
  4. Surfaces unresolved contradictions for Dream Mode / reporting.

Author: Aura reliability upgrade (2026-03)
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Edge types for contradiction/supersession
# ---------------------------------------------------------------------------

CONTRADICTS_EDGE  = "CONTRADICTS"
SUPERSEDES_EDGE   = "SUPERSEDES"

# Node lifecycle states (stored in node's `lifecycle_state` property)
KG_NODE_ACTIVE     = "active"
KG_NODE_SUPERSEDED = "superseded"
KG_NODE_CONTESTED  = "contested"   # unresolved contradiction


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class ContradictionRecord:
    node_a_id: str
    node_b_id: str
    label_a: str
    label_b: str
    contradiction_type: str    # "direct_negation" | "value_conflict" | "temporal_override"
    confidence: float = 0.7
    resolved: bool = False
    resolution: str = ""       # "superseded_by_newer" | "user_confirmed" | ""
    detected_at: float = field(default_factory=time.time)
    contradiction_id: str = field(default_factory=lambda: str(uuid.uuid4())[:10])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "contradiction_id": self.contradiction_id,
            "node_a": self.label_a,
            "node_a_id": self.node_a_id,
            "node_b": self.label_b,
            "node_b_id": self.node_b_id,
            "type": self.contradiction_type,
            "confidence": round(self.confidence, 3),
            "resolved": self.resolved,
            "resolution": self.resolution,
        }


@dataclass
class SupersessionRecord:
    old_node_id: str
    new_node_id: str
    old_label: str
    new_label: str
    reason: str = ""
    superseded_at: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Contradiction detector
# ---------------------------------------------------------------------------

class KGContradictionDetector:
    """
    Detects contradictions and manages supersession in an existing KG.

    Usage:
        detector = KGContradictionDetector(kg)
        records = detector.check_for_contradictions(new_node_id)
        detector.supersede(old_id, new_id, reason="user_correction")
    """

    # Negation phrase patterns
    NEGATION_PAIRS = [
        ({"always", "every", "all"}, {"never", "none", "no"}),
        ({"prefers", "likes", "uses"}, {"hates", "dislikes", "avoids"}),
        ({"enabled", "on", "active"}, {"disabled", "off", "inactive"}),
        ({"increased", "more", "added"}, {"decreased", "less", "removed"}),
    ]

    def __init__(self, kg) -> None:
        """
        Args:
            kg: An instance of KnowledgeGraph from aura/tools/knowledge_graph.py
        """
        self._kg = kg
        self._contradictions: List[ContradictionRecord] = []
        self._enabled = True
        try:
            from aura.config import Config
            self._enabled = getattr(Config, "ENABLE_KG_CONTRADICTIONS", True)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check_for_contradictions(
        self,
        new_node_id: str,
        similarity_threshold: float = 0.6,
    ) -> List[ContradictionRecord]:
        """
        Check if a newly added node contradicts existing nodes.

        Args:
            new_node_id:           ID of the newly added KG node.
            similarity_threshold:  Minimum semantic overlap to consider for contradiction.

        Returns:
            List of ContradictionRecord (may be empty).
        """
        if not self._enabled:
            return []

        try:
            new_node = self._get_node(new_node_id)
        except Exception:
            return []

        if not new_node:
            return []

        new_label = new_node.get("label", "")
        new_props = new_node.get("properties", {})
        new_type  = new_node.get("type", "")

        found: List[ContradictionRecord] = []

        # Search existing nodes of same type for potential conflicts
        try:
            similar_ids = self._find_similar_nodes(new_node_id, new_label, new_type)
        except Exception as e:
            logger.debug("[KGContradiction] find_similar error: %s", e)
            return []

        for candidate_id in similar_ids:
            if candidate_id == new_node_id:
                continue
            try:
                candidate = self._get_node(candidate_id)
                if not candidate:
                    continue

                c_label = candidate.get("label", "")
                c_props = candidate.get("properties", {})

                contradiction = self._detect_contradiction(
                    new_label, new_props, new_node_id,
                    c_label, c_props, candidate_id,
                )
                if contradiction:
                    found.append(contradiction)
                    self._contradictions.append(contradiction)
                    self._mark_contested(new_node_id, candidate_id)
                    self._add_contradiction_edge(new_node_id, candidate_id, contradiction)
                    logger.warning(
                        "[KGContradiction] Detected: '%s' ↔ '%s' type=%s",
                        new_label[:60], c_label[:60], contradiction.contradiction_type,
                    )
                    self._emit_telemetry(contradiction)
            except Exception as e:
                logger.debug("[KGContradiction] candidate check error: %s", e)

        return found

    def supersede(
        self,
        old_node_id: str,
        new_node_id: str,
        reason: str = "newer_information",
    ) -> SupersessionRecord:
        """
        Mark old_node as superseded by new_node.

        Adds a SUPERSEDES edge and updates lifecycle_state on old_node.
        """
        try:
            old_node = self._get_node(old_node_id)
            new_node = self._get_node(new_node_id)
            old_label = old_node.get("label", old_node_id) if old_node else old_node_id
            new_label = new_node.get("label", new_node_id) if new_node else new_node_id

            # Mark old node as superseded
            self._set_lifecycle(old_node_id, KG_NODE_SUPERSEDED)

            # Add SUPERSEDES edge new → old
            self._add_supersedes_edge(new_node_id, old_node_id, reason)

            # Resolve any existing contradictions between these nodes
            for c in self._contradictions:
                if {c.node_a_id, c.node_b_id} == {old_node_id, new_node_id}:
                    c.resolved    = True
                    c.resolution  = "superseded_by_newer"

            record = SupersessionRecord(
                old_node_id=old_node_id,
                new_node_id=new_node_id,
                old_label=old_label,
                new_label=new_label,
                reason=reason,
            )
            logger.info(
                "[KGContradiction] Superseded: '%s' → '%s' reason=%s",
                old_label[:60], new_label[:60], reason,
            )
            return record

        except Exception as e:
            logger.warning("[KGContradiction] supersede error: %s", e)
            return SupersessionRecord(
                old_node_id=old_node_id, new_node_id=new_node_id,
                old_label="", new_label="", reason=f"error:{e}",
            )

    def get_unresolved_contradictions(self) -> List[ContradictionRecord]:
        """Return all contradiction records that have not been resolved."""
        return [c for c in self._contradictions if not c.resolved]

    def get_all_contradictions(self) -> List[Dict[str, Any]]:
        return [c.to_dict() for c in self._contradictions]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        try:
            g = self._kg.graph
            if node_id in g.nodes:
                return dict(g.nodes[node_id])
        except Exception:
            pass
        return None

    def _find_similar_nodes(
        self, node_id: str, label: str, node_type: str, limit: int = 10
    ) -> List[str]:
        """Find nodes of same type with overlapping label words."""
        words = set(label.lower().split())
        if len(words) < 2:
            return []
        try:
            g = self._kg.graph
            results = []
            for nid, data in g.nodes(data=True):
                if nid == node_id:
                    continue
                if data.get("type") != node_type:
                    continue
                other_words = set(data.get("label", "").lower().split())
                overlap = len(words & other_words) / max(len(words | other_words), 1)
                if overlap >= 0.4:   # 40% word overlap
                    results.append((overlap, nid))
            results.sort(reverse=True)
            return [nid for _, nid in results[:limit]]
        except Exception:
            return []

    def _detect_contradiction(
        self,
        label_a: str, props_a: dict, id_a: str,
        label_b: str, props_b: dict, id_b: str,
    ) -> Optional[ContradictionRecord]:
        """Return a ContradictionRecord if the two nodes likely contradict."""

        # Check value conflicts in shared properties
        shared_keys = set(props_a.keys()) & set(props_b.keys())
        for key in shared_keys:
            va = str(props_a[key]).lower()
            vb = str(props_b[key]).lower()
            if va != vb and va and vb:
                return ContradictionRecord(
                    node_a_id=id_a, node_b_id=id_b,
                    label_a=label_a, label_b=label_b,
                    contradiction_type="value_conflict",
                    confidence=0.75,
                )

        # Check negation pairs in labels
        words_a = set(label_a.lower().split())
        words_b = set(label_b.lower().split())
        for pos_set, neg_set in self.NEGATION_PAIRS:
            if (words_a & pos_set and words_b & neg_set) or \
               (words_b & pos_set and words_a & neg_set):
                return ContradictionRecord(
                    node_a_id=id_a, node_b_id=id_b,
                    label_a=label_a, label_b=label_b,
                    contradiction_type="direct_negation",
                    confidence=0.80,
                )

        return None

    def _mark_contested(self, id_a: str, id_b: str) -> None:
        try:
            g = self._kg.graph
            if id_a in g.nodes:
                g.nodes[id_a]["lifecycle_state"] = KG_NODE_CONTESTED
            if id_b in g.nodes:
                g.nodes[id_b]["lifecycle_state"] = KG_NODE_CONTESTED
        except Exception:
            pass

    def _set_lifecycle(self, node_id: str, state: str) -> None:
        try:
            g = self._kg.graph
            if node_id in g.nodes:
                g.nodes[node_id]["lifecycle_state"] = state
        except Exception:
            pass

    def _add_contradiction_edge(
        self, id_a: str, id_b: str, rec: ContradictionRecord
    ) -> None:
        try:
            g = self._kg.graph
            g.add_edge(id_a, id_b,
                       type=CONTRADICTS_EDGE,
                       contradiction_id=rec.contradiction_id,
                       confidence=rec.confidence,
                       detected_at=rec.detected_at)
        except Exception as e:
            logger.debug("[KGContradiction] edge add error: %s", e)

    def _add_supersedes_edge(self, new_id: str, old_id: str, reason: str) -> None:
        try:
            g = self._kg.graph
            g.add_edge(new_id, old_id,
                       type=SUPERSEDES_EDGE,
                       reason=reason,
                       superseded_at=time.time())
        except Exception as e:
            logger.debug("[KGContradiction] supersedes edge error: %s", e)

    def _emit_telemetry(self, rec: ContradictionRecord) -> None:
        try:
            from aura.reliability.telemetry import emit, TelemetryKind
            emit(
                TelemetryKind.CONTRADICTION,
                contradictions_detected=1,
                extra={
                    "contradiction_id": rec.contradiction_id,
                    "type": rec.contradiction_type,
                    "node_a": rec.label_a[:60],
                    "node_b": rec.label_b[:60],
                },
            )
        except Exception:
            pass


__all__ = [
    "KGContradictionDetector",
    "ContradictionRecord",
    "SupersessionRecord",
    "KG_NODE_ACTIVE",
    "KG_NODE_SUPERSEDED",
    "KG_NODE_CONTESTED",
    "CONTRADICTS_EDGE",
    "SUPERSEDES_EDGE",
]
