"""Temporal (valid-time) overlay for Aura's NetworkX Knowledge Graph.

This is the missing piece next to `kg_contradiction.py`: contradictions
tell you that two facts can't both be true, and supersession marks one
as replaced. Temporal edges tell you *when* each fact was true, so you
can answer "what did we believe about X on 2026-01-15?".

Design — overlay, not rewrite:
  - Stores `valid_from` and `valid_until` as regular node properties.
  - Uses the KG's existing `update_node()` so no schema changes.
  - `valid_from` defaults to node.created_at when a fact enters the graph.
  - `valid_until = None` means "still valid" (open interval).

Interaction with supersession:
  When `temporal.supersede(old, new)` is called, the old node's
  `valid_until` is closed at the supersession timestamp AND the graph
  gets the normal SUPERSEDES edge (via KGContradictionDetector if
  available). This keeps the two overlays composable.

Usage:
    from aura.memory.kg_temporal import KGTemporalOverlay
    temporal = KGTemporalOverlay(kg)

    # Stamp a fresh fact
    temporal.stamp(node_id)

    # Answer "what was true about 'laptop owner' on 2026-02-10?"
    hits = temporal.query_at(label_contains="laptop owner", at_time=iso_dt)

    # Get full history of a fact
    history = temporal.timeline(label_contains="laptop owner")

    # Close a fact's validity window now
    temporal.invalidate(node_id)

    # Supersede + invalidate in one step
    temporal.supersede(old_id=..., new_id=..., reason="user_correction")
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _utc_iso(ts: Optional[float | str | datetime] = None) -> str:
    """Return an ISO 8601 UTC timestamp. Accepts float epoch, ISO string, or datetime."""
    if ts is None:
        return datetime.now(timezone.utc).isoformat()
    if isinstance(ts, datetime):
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return ts.isoformat()
    if isinstance(ts, (int, float)):
        return datetime.fromtimestamp(float(ts), tz=timezone.utc).isoformat()
    if isinstance(ts, str):
        # Trust the caller — but sanity-parse to catch garbage
        try:
            datetime.fromisoformat(ts.replace("Z", "+00:00"))
            return ts
        except ValueError:
            return datetime.now(timezone.utc).isoformat()
    return datetime.now(timezone.utc).isoformat()


def _parse_iso(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None


@dataclass
class TemporalFact:
    """A KG node viewed through the temporal lens."""
    node_id: str
    label: str
    node_type: str
    valid_from: Optional[datetime]
    valid_until: Optional[datetime]
    confidence: float
    raw: Dict[str, Any]

    @property
    def is_open(self) -> bool:
        """True if the validity window is still open (no valid_until set)."""
        return self.valid_until is None

    def covers(self, at_time: datetime) -> bool:
        """True if this fact was valid at `at_time`."""
        if self.valid_from and at_time < self.valid_from:
            return False
        if self.valid_until and at_time > self.valid_until:
            return False
        return True


VALID_FROM_KEY = "valid_from"
VALID_UNTIL_KEY = "valid_until"
TEMPORAL_SOURCE_KEY = "temporal_source"


class KGTemporalOverlay:
    """Valid-time overlay for a NetworkX-based KnowledgeGraph.

    Does not rewrite the KG — stores valid_from / valid_until as node
    properties and reads them back on query.
    """

    def __init__(self, kg, contradiction_detector=None) -> None:
        """
        Args:
            kg: An aura.tools.knowledge_graph.KnowledgeGraph instance.
            contradiction_detector: Optional KGContradictionDetector. If
                provided, `supersede()` will also add a SUPERSEDES edge via
                the detector so both overlays agree on graph state.
        """
        self._kg = kg
        self._detector = contradiction_detector

    # ------------------------------------------------------------------
    # Write API
    # ------------------------------------------------------------------

    def stamp(
        self,
        node_id: str,
        valid_from: Optional[float | str | datetime] = None,
        valid_until: Optional[float | str | datetime] = None,
        source: str = "manual",
    ) -> bool:
        """Attach valid_from / valid_until to a node.

        Missing values default to 'now' for valid_from and None (open) for
        valid_until. Returns True on success.
        """
        node = self._get_node_dict(node_id)
        if node is None:
            logger.debug("[KGTemporal] stamp: node not found: %s", node_id)
            return False

        properties = {
            VALID_FROM_KEY: _utc_iso(valid_from) if valid_from is not None else _utc_iso(
                node.get("created_at") or node.get("properties", {}).get("created_at")
            ),
            TEMPORAL_SOURCE_KEY: source,
        }
        if valid_until is not None:
            properties[VALID_UNTIL_KEY] = _utc_iso(valid_until)

        return bool(self._kg.update_node(node_id, properties))

    def invalidate(
        self,
        node_id: str,
        at: Optional[float | str | datetime] = None,
        reason: str = "invalidated",
    ) -> bool:
        """Close a fact's validity window.

        Sets valid_until to `at` (or now). If the node has no valid_from
        yet, one is filled in from its created_at so the interval is
        well-formed.
        """
        node = self._get_node_dict(node_id)
        if node is None:
            return False

        properties: Dict[str, Any] = {VALID_UNTIL_KEY: _utc_iso(at)}
        existing_props = node.get("properties", {}) or {}
        if VALID_FROM_KEY not in existing_props:
            properties[VALID_FROM_KEY] = _utc_iso(
                node.get("created_at") or existing_props.get("created_at")
            )
        properties["invalidation_reason"] = reason
        return bool(self._kg.update_node(node_id, properties))

    def supersede(
        self,
        old_node_id: str,
        new_node_id: str,
        reason: str = "newer_information",
    ) -> bool:
        """Invalidate `old` aligned with `new`'s start, and (if a
        contradiction detector was supplied) add a SUPERSEDES edge.

        If the new node already has a `valid_from` set, the old node's
        `valid_until` is pinned to that same instant so the timeline has
        no gap or overlap. If the new node has no stamp yet, both sides
        use "now".
        """
        new_node = self._get_node_dict(new_node_id)
        existing_new_from: Optional[str] = None
        if new_node is not None:
            existing_new_from = (new_node.get("properties") or {}).get(VALID_FROM_KEY)

        boundary = existing_new_from or _utc_iso()

        ok_old = self.invalidate(old_node_id, at=boundary, reason=reason)
        # Only stamp the new node if it doesn't already carry a valid_from
        if existing_new_from is None:
            self.stamp(new_node_id, valid_from=boundary, source="supersession")

        if self._detector is not None:
            try:
                self._detector.supersede(old_node_id, new_node_id, reason=reason)
            except Exception as exc:  # noqa: BLE001
                logger.debug("[KGTemporal] detector.supersede failed: %s", exc)

        return ok_old

    # ------------------------------------------------------------------
    # Read API
    # ------------------------------------------------------------------

    def get_fact(self, node_id: str) -> Optional[TemporalFact]:
        node = self._get_node_dict(node_id)
        if node is None:
            return None
        return self._to_fact(node_id, node)

    def query_at(
        self,
        at_time: Optional[float | str | datetime] = None,
        label_contains: Optional[str] = None,
        node_type: Optional[str] = None,
    ) -> List[TemporalFact]:
        """Return all facts valid at a given point in time.

        Filters by label substring and/or node type if provided.
        """
        target = _parse_iso(_utc_iso(at_time))
        if target is None:
            return []

        needle = (label_contains or "").lower().strip()
        results: List[TemporalFact] = []

        for node_id, node in self._iter_nodes():
            label = (node.get("label") or "").lower()
            if needle and needle not in label:
                continue
            if node_type and node.get("type") != node_type:
                continue
            fact = self._to_fact(node_id, node)
            if fact.covers(target):
                results.append(fact)

        results.sort(key=lambda f: f.valid_from or datetime.min.replace(tzinfo=timezone.utc))
        return results

    def timeline(
        self,
        label_contains: str,
        node_type: Optional[str] = None,
    ) -> List[TemporalFact]:
        """Return the full history of a fact, sorted by valid_from."""
        needle = label_contains.lower().strip()
        history: List[TemporalFact] = []
        for node_id, node in self._iter_nodes():
            label = (node.get("label") or "").lower()
            if needle not in label:
                continue
            if node_type and node.get("type") != node_type:
                continue
            history.append(self._to_fact(node_id, node))
        history.sort(
            key=lambda f: f.valid_from or datetime.min.replace(tzinfo=timezone.utc)
        )
        return history

    def open_facts(self, node_type: Optional[str] = None) -> List[TemporalFact]:
        """All facts with an open validity window (valid_until is None)."""
        facts: List[TemporalFact] = []
        for node_id, node in self._iter_nodes():
            if node_type and node.get("type") != node_type:
                continue
            fact = self._to_fact(node_id, node)
            if fact.is_open:
                facts.append(fact)
        return facts

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _get_node_dict(self, node_id: str) -> Optional[Dict[str, Any]]:
        try:
            node_obj = self._kg.get_node(node_id)
        except Exception as exc:  # noqa: BLE001
            logger.debug("[KGTemporal] get_node failed: %s", exc)
            return None
        if node_obj is None:
            return None
        # KG's Node dataclass exposes to_dict()
        to_dict = getattr(node_obj, "to_dict", None)
        if callable(to_dict):
            return to_dict()
        if isinstance(node_obj, dict):
            return node_obj
        return None

    def _iter_nodes(self):
        """Yield (node_id, node_dict) for every node in the KG."""
        nodes_attr = getattr(self._kg, "_nodes", None)
        if not nodes_attr:
            return
        for node_id, node_obj in list(nodes_attr.items()):
            to_dict = getattr(node_obj, "to_dict", None)
            if callable(to_dict):
                yield node_id, to_dict()
            elif isinstance(node_obj, dict):
                yield node_id, node_obj

    @staticmethod
    def _to_fact(node_id: str, node: Dict[str, Any]) -> TemporalFact:
        props = node.get("properties", {}) or {}
        return TemporalFact(
            node_id=node_id,
            label=node.get("label", ""),
            node_type=node.get("type", ""),
            valid_from=_parse_iso(props.get(VALID_FROM_KEY)) or _parse_iso(node.get("created_at")),
            valid_until=_parse_iso(props.get(VALID_UNTIL_KEY)),
            confidence=float(node.get("confidence", 0.0) or 0.0),
            raw=node,
        )


__all__ = [
    "KGTemporalOverlay",
    "TemporalFact",
    "VALID_FROM_KEY",
    "VALID_UNTIL_KEY",
]
