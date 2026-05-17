"""
Curiosity as Information Gain — KG Gap Scanner (Roadmap 4.3).

Scans the Knowledge Graph for genuine knowledge gaps and generates
natural curiosity-driven questions. Replaces the generic curiosity
intensity score with specific, KG-grounded curiosity targets.

Gap types detected:
    1. Isolated nodes: entities with < 3 connections
    2. Context-less mentions: recently mentioned but no description
    3. Contradictions: same entity with conflicting relationships
    4. Stale topics: projects/concepts not mentioned in > 14 days

Each gap becomes a CuriosityTarget with a natural-language question
that AURA can ask proactively.

Integrates with:
    - AURAKnowledgeGraph (Kuzu): primary data source for gap detection
    - Runtime KG (NetworkX): fallback for gap detection
    - IntrinsicMotivation: feeds curiosity targets for drive actions
    - MotivationAccumulator: provides relevance scoring for curiosity messages
    - ProactiveMessages: templates for curiosity-driven questions
"""

import logging
import random
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ============================================================================
# Data Models
# ============================================================================

class GapType:
    """Types of knowledge gaps."""
    ISOLATED = "isolated_node"          # < 3 connections
    CONTEXTLESS = "missing_context"     # Mentioned but no description
    CONTRADICTORY = "contradiction"     # Conflicting relationships
    STALE = "stale_topic"              # Not mentioned in > 14 days
    SHALLOW = "shallow_knowledge"       # Important entity with low access count


@dataclass
class CuriosityTarget:
    """A specific knowledge gap that curiosity can drive toward filling."""
    entity_name: str
    entity_id: str
    entity_type: str
    gap_type: str                      # GapType constant
    urgency: float = 0.5              # 0-1
    question: str = ""                 # Natural-language question to ask
    context: str = ""                  # Why this gap matters
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_name": self.entity_name,
            "entity_id": self.entity_id,
            "entity_type": self.entity_type,
            "gap_type": self.gap_type,
            "urgency": round(self.urgency, 3),
            "question": self.question,
            "context": self.context,
        }


# ============================================================================
# Question Templates
# ============================================================================

# Templates for generating natural curiosity questions per gap type.
# {name} = entity name, {type} = entity type, {days} = days since last mention

_ISOLATED_QUESTIONS = [
    "I noticed '{name}' in my knowledge base but it's pretty disconnected from everything else. How does it relate to what you're working on?",
    "'{name}' keeps coming up but I can't figure out how it connects to your other work. Can you help me understand?",
    "I've got '{name}' stored as a {type} but it's floating around without much context. What's its role in the bigger picture?",
    "Quick question — '{name}' seems important but I don't have a clear picture of how it fits in. Mind filling me in?",
]

_CONTEXTLESS_QUESTIONS = [
    "You mentioned '{name}' recently but I don't have much background on it. What should I know?",
    "I keep seeing '{name}' pop up but I'm drawing a blank on the details. What is it exactly?",
    "'{name}' came up in conversation but I realized I don't have a solid understanding of it. Can you give me the quick version?",
    "I noticed I don't have a good description for '{name}'. Mind telling me more about it so I can be more helpful?",
]

_CONTRADICTORY_QUESTIONS = [
    "I found something weird — my notes about '{name}' seem to contradict themselves. Can we sort that out?",
    "I've got conflicting info about '{name}'. {context} — which one's right?",
    "My knowledge about '{name}' doesn't quite add up. I'm seeing {context}. Want to help me untangle it?",
]

_STALE_QUESTIONS = [
    "I haven't heard about '{name}' in a while — it's been {days} days. How's that going?",
    "'{name}' dropped off my radar about {days} days ago. Still in progress or did priorities shift?",
    "Just thinking about it — you haven't mentioned '{name}' recently. Everything on track there?",
    "It's been {days} days since '{name}' came up. I'm curious — any updates?",
    "'{name}' has been quiet for a while. Is it on pause or should I keep it on my radar?",
]

_SHALLOW_QUESTIONS = [
    "I know about '{name}' but I feel like I'm only scratching the surface. What am I missing?",
    "'{name}' seems important to you but I don't have deep context on it. What should I understand better?",
    "I've noted '{name}' as a {type} but I bet there's more to it. Can you tell me more?",
]


def _generate_question(target: CuriosityTarget) -> str:
    """Generate a natural-language question for a curiosity target."""
    templates = {
        GapType.ISOLATED: _ISOLATED_QUESTIONS,
        GapType.CONTEXTLESS: _CONTEXTLESS_QUESTIONS,
        GapType.CONTRADICTORY: _CONTRADICTORY_QUESTIONS,
        GapType.STALE: _STALE_QUESTIONS,
        GapType.SHALLOW: _SHALLOW_QUESTIONS,
    }
    pool = templates.get(target.gap_type, _ISOLATED_QUESTIONS)
    template = random.choice(pool)

    days = target.metadata.get("days_stale", "")
    return template.format(
        name=target.entity_name,
        type=target.entity_type,
        days=days,
        context=target.context,
    )


# ============================================================================
# Curiosity Scanner
# ============================================================================

class CuriosityScanner:
    """Scans the Knowledge Graph for genuine knowledge gaps.

    Runs two scan modes:
    - Quick scan: isolated nodes + stale topics (fast, after each conversation)
    - Full scan: all 5 gap types (slower, during idle time)

    Results are cached and refreshed based on scan_interval.
    """

    SCAN_INTERVAL = 300.0  # 5 minutes between full scans
    QUICK_SCAN_INTERVAL = 60.0  # 1 minute between quick scans
    MAX_TARGETS = 10
    STALE_DAYS = 14
    CONTEXTLESS_DAYS = 7
    MIN_CONNECTIONS = 3

    def __init__(self):
        self._lock = threading.RLock()
        self._targets: List[CuriosityTarget] = []
        self._last_full_scan: float = 0.0
        self._last_quick_scan: float = 0.0
        self._stats = {
            "full_scans": 0,
            "quick_scans": 0,
            "total_gaps_found": 0,
            "questions_generated": 0,
        }

    # ====================================================================
    # Main Scan Methods
    # ====================================================================

    def scan_full(self) -> List[CuriosityTarget]:
        """Run all gap detection checks. Rate-limited to SCAN_INTERVAL."""
        with self._lock:
            now = time.time()
            if now - self._last_full_scan < self.SCAN_INTERVAL:
                return self._targets
            targets = []
            targets.extend(self._scan_isolated_nodes())
            targets.extend(self._scan_contextless_mentions())
            targets.extend(self._scan_contradictions())
            targets.extend(self._scan_stale_topics())
            targets.extend(self._scan_shallow_knowledge())

            # Deduplicate by entity_id
            seen = set()
            unique = []
            for t in targets:
                if t.entity_id not in seen:
                    seen.add(t.entity_id)
                    unique.append(t)

            # Sort by urgency and cap
            unique.sort(key=lambda t: t.urgency, reverse=True)
            self._targets = unique[:self.MAX_TARGETS]

            # Generate questions for all targets
            for target in self._targets:
                if not target.question:
                    target.question = _generate_question(target)
                    self._stats["questions_generated"] += 1

            self._last_full_scan = now
            self._stats["full_scans"] += 1
            self._stats["total_gaps_found"] += len(self._targets)

            if self._targets:
                logger.info(
                    f"[CuriosityScanner] Full scan found {len(self._targets)} gaps: "
                    + ", ".join(f"{t.gap_type}:{t.entity_name}" for t in self._targets[:3])
                )

        return self._targets

    def scan_quick(self) -> List[CuriosityTarget]:
        """Run fast checks only (isolated + stale). Rate-limited."""
        with self._lock:
            now = time.time()
            if now - self._last_quick_scan < self.QUICK_SCAN_INTERVAL:
                return self._targets
            targets = []
            targets.extend(self._scan_isolated_nodes())
            targets.extend(self._scan_stale_topics())

            # Merge with existing (prefer new findings)
            existing_ids = {t.entity_id for t in targets}
            for old in self._targets:
                if old.entity_id not in existing_ids:
                    targets.append(old)

            seen = set()
            unique = []
            for t in targets:
                if t.entity_id not in seen:
                    seen.add(t.entity_id)
                    unique.append(t)

            unique.sort(key=lambda t: t.urgency, reverse=True)
            self._targets = unique[:self.MAX_TARGETS]

            for target in self._targets:
                if not target.question:
                    target.question = _generate_question(target)
                    self._stats["questions_generated"] += 1

            self._last_quick_scan = now
            self._stats["quick_scans"] += 1

        return self._targets

    # ====================================================================
    # Gap Detection — Kuzu KG
    # ====================================================================

    def _get_kuzu_kg(self):
        """Get the Kuzu-based persistent KG instance.

        Tries multiple paths:
        1. Agent's kg_brain instance (already initialized)
        2. Direct instantiation with default path
        """
        # Try to get the agent's existing KG instance
        try:
            from api.services.agent_service import agent_service
            if agent_service.agent and hasattr(agent_service.agent, 'kg_brain'):
                kg = agent_service.agent.kg_brain
                if kg is not None:
                    return kg
        except Exception:
            pass

        # Fallback: instantiate directly (will reuse existing DB files)
        try:
            from aura_knowledge_graph.graph_database import AURAKnowledgeGraph
            return AURAKnowledgeGraph()
        except Exception:
            return None

    def _scan_isolated_nodes(self) -> List[CuriosityTarget]:
        """Find entities with fewer than MIN_CONNECTIONS connections."""
        targets = []

        # Try Kuzu KG first
        kg = self._get_kuzu_kg()
        if kg:
            try:
                result = kg.execute_cypher(f"""
                    MATCH (e:Entity)
                    WHERE e.access_count > 0
                    OPTIONAL MATCH (e)-[r:RELATES_TO]-(other:Entity)
                    WHERE r.is_active = true
                    WITH e, COUNT(r) AS conn_count
                    WHERE conn_count < {self.MIN_CONNECTIONS}
                    RETURN e.id, e.name, e.entity_type, e.description,
                           e.importance, conn_count
                    ORDER BY e.importance DESC
                    LIMIT 10
                """)

                for row in result:
                    entity_id, name, etype, _desc, importance, conn_count = row
                    urgency = 0.4 + (self.MIN_CONNECTIONS - conn_count) * 0.15
                    urgency = min(0.8, urgency + (importance or 0) * 0.2)
                    targets.append(CuriosityTarget(
                        entity_name=name or entity_id,
                        entity_id=entity_id,
                        entity_type=etype or "unknown",
                        gap_type=GapType.ISOLATED,
                        urgency=urgency,
                        context=f"{conn_count} connections (need {self.MIN_CONNECTIONS}+)",
                        metadata={"connections": conn_count, "importance": importance},
                    ))
            except Exception as e:
                logger.debug(f"[CuriosityScanner] Kuzu isolated scan error: {e}")

        # Fallback: runtime NetworkX KG
        if not targets:
            targets = self._scan_isolated_nodes_nx()

        return targets

    def _scan_isolated_nodes_nx(self) -> List[CuriosityTarget]:
        """Fallback: scan NetworkX runtime KG for isolated nodes."""
        targets = []
        try:
            from aura.tools.knowledge_graph import get_knowledge_graph
            kg = get_knowledge_graph()
            all_nodes = []
            if hasattr(kg, '_lock') and hasattr(kg, '_nodes'):
                with kg._lock:
                    all_nodes = list(kg._nodes.values())

            for node in all_nodes:
                if getattr(node, 'access_count', 0) == 0:
                    continue
                edges = kg.get_edges(node.id) if hasattr(kg, 'get_edges') else []
                if len(edges) < self.MIN_CONNECTIONS:
                    targets.append(CuriosityTarget(
                        entity_name=getattr(node, 'label', str(node)),
                        entity_id=node.id,
                        entity_type=getattr(node, 'type', 'unknown'),
                        gap_type=GapType.ISOLATED,
                        urgency=0.5,
                        context=f"{len(edges)} connections",
                        metadata={"connections": len(edges)},
                    ))
        except Exception as e:
            logger.debug(f"[CuriosityScanner] NX isolated scan error: {e}")
        return targets[:5]

    def _scan_contextless_mentions(self) -> List[CuriosityTarget]:
        """Find entities mentioned in last CONTEXTLESS_DAYS days but with no description."""
        targets = []
        kg = self._get_kuzu_kg()
        if not kg:
            return targets

        try:
            cutoff = int(time.time()) - (self.CONTEXTLESS_DAYS * 86400)
            result = kg.execute_cypher(f"""
                MATCH (e:Entity)
                WHERE e.last_accessed > {cutoff}
                  AND (e.description IS NULL OR e.description = '')
                  AND e.access_count >= 1
                RETURN e.id, e.name, e.entity_type, e.access_count, e.last_accessed
                ORDER BY e.access_count DESC
                LIMIT 5
            """)

            for row in result:
                entity_id, name, etype, access_count, _last_accessed = row
                urgency = min(0.7, 0.3 + (access_count or 0) * 0.05)
                targets.append(CuriosityTarget(
                    entity_name=name or entity_id,
                    entity_id=entity_id,
                    entity_type=etype or "unknown",
                    gap_type=GapType.CONTEXTLESS,
                    urgency=urgency,
                    context=f"Mentioned {access_count} times, no description",
                    metadata={"access_count": access_count},
                ))
        except Exception as e:
            logger.debug(f"[CuriosityScanner] Contextless scan error: {e}")

        return targets

    def _scan_contradictions(self) -> List[CuriosityTarget]:
        """Find entities with conflicting active relationships.

        Detects: same source+target with different relationship types both active,
        or same entity with contradictory property relationships.
        """
        targets = []
        kg = self._get_kuzu_kg()
        if not kg:
            return targets

        try:
            # Find entities that have multiple active edges to the same target
            # with different relationship types — potential contradictions
            result = kg.execute_cypher("""
                MATCH (s:Entity)-[r1:RELATES_TO]->(t:Entity),
                      (s)-[r2:RELATES_TO]->(t)
                WHERE r1.is_active = true AND r2.is_active = true
                  AND r1.relationship_type <> r2.relationship_type
                RETURN DISTINCT s.id, s.name, s.entity_type,
                       r1.relationship_type, r2.relationship_type,
                       t.name
                LIMIT 5
            """)

            for row in result:
                src_id, src_name, src_type, rel1, rel2, target_name = row
                context_str = f"'{src_name}' has both '{rel1}' and '{rel2}' to '{target_name}'"
                targets.append(CuriosityTarget(
                    entity_name=src_name or src_id,
                    entity_id=src_id,
                    entity_type=src_type or "unknown",
                    gap_type=GapType.CONTRADICTORY,
                    urgency=0.7,
                    context=context_str,
                    metadata={"rel1": rel1, "rel2": rel2, "target": target_name},
                ))
        except Exception as e:
            logger.debug(f"[CuriosityScanner] Contradiction scan error: {e}")

        return targets

    def _scan_stale_topics(self) -> List[CuriosityTarget]:
        """Find projects/concepts not accessed in STALE_DAYS+ days."""
        targets = []
        kg = self._get_kuzu_kg()
        if not kg:
            return self._scan_stale_topics_nx()

        try:
            cutoff = int(time.time()) - (self.STALE_DAYS * 86400)
            result = kg.execute_cypher(f"""
                MATCH (e:Entity)
                WHERE e.last_accessed < {cutoff}
                  AND e.last_accessed > 0
                  AND e.access_count >= 2
                  AND e.entity_type IN ['Project', 'Concept', 'Technology', 'Task']
                RETURN e.id, e.name, e.entity_type, e.last_accessed,
                       e.importance, e.access_count
                ORDER BY e.importance DESC
                LIMIT 5
            """)

            now = int(time.time())
            for row in result:
                entity_id, name, etype, last_accessed, importance, access_count = row
                days_stale = (now - (last_accessed or 0)) // 86400
                urgency = min(0.9, 0.4 + days_stale / 60.0)
                urgency = min(0.9, urgency + (importance or 0) * 0.2)
                targets.append(CuriosityTarget(
                    entity_name=name or entity_id,
                    entity_id=entity_id,
                    entity_type=etype or "unknown",
                    gap_type=GapType.STALE,
                    urgency=urgency,
                    context=f"Not mentioned in {days_stale} days",
                    metadata={
                        "days_stale": days_stale,
                        "importance": importance,
                        "access_count": access_count,
                    },
                ))
        except Exception as e:
            logger.debug(f"[CuriosityScanner] Kuzu stale scan error: {e}")

        return targets

    def _scan_stale_topics_nx(self) -> List[CuriosityTarget]:
        """Fallback: scan NetworkX KG for stale topics."""
        targets = []
        try:
            from aura.tools.knowledge_graph import get_knowledge_graph
            kg = get_knowledge_graph()
            all_nodes = []
            if hasattr(kg, '_lock') and hasattr(kg, '_nodes'):
                with kg._lock:
                    all_nodes = list(kg._nodes.values())

            now = datetime.now()
            cutoff = now - timedelta(days=self.STALE_DAYS)

            for node in all_nodes:
                if getattr(node, 'access_count', 0) < 2:
                    continue
                if getattr(node, 'type', '') not in ("project", "concept", "technology", "task"):
                    continue
                la = getattr(node, 'last_accessed', '')
                if not la:
                    continue
                try:
                    node_dt = datetime.fromisoformat(str(la).replace('Z', '+00:00'))
                    if node_dt.tzinfo is not None:
                        node_dt = node_dt.replace(tzinfo=None)
                    if node_dt < cutoff:
                        days_stale = (now - node_dt).days
                        targets.append(CuriosityTarget(
                            entity_name=getattr(node, 'label', str(node)),
                            entity_id=node.id,
                            entity_type=getattr(node, 'type', 'unknown'),
                            gap_type=GapType.STALE,
                            urgency=min(0.9, 0.4 + days_stale / 60.0),
                            context=f"Not mentioned in {days_stale} days",
                            metadata={"days_stale": days_stale},
                        ))
                except (ValueError, TypeError, AttributeError):
                    pass
        except Exception as e:
            logger.debug(f"[CuriosityScanner] NX stale scan error: {e}")
        return targets[:5]

    def _scan_shallow_knowledge(self) -> List[CuriosityTarget]:
        """Find important entities with low access count (we should know more)."""
        targets = []
        kg = self._get_kuzu_kg()
        if not kg:
            return targets

        try:
            result = kg.execute_cypher("""
                MATCH (e:Entity)
                WHERE e.importance > 0.5
                  AND e.access_count <= 2
                  AND e.access_count > 0
                RETURN e.id, e.name, e.entity_type, e.importance, e.access_count
                ORDER BY e.importance DESC
                LIMIT 5
            """)

            for row in result:
                entity_id, name, etype, importance, access_count = row
                urgency = min(0.7, 0.3 + (importance or 0) * 0.4)
                targets.append(CuriosityTarget(
                    entity_name=name or entity_id,
                    entity_id=entity_id,
                    entity_type=etype or "unknown",
                    gap_type=GapType.SHALLOW,
                    urgency=urgency,
                    context=f"High importance ({importance:.2f}) but only {access_count} accesses",
                    metadata={"importance": importance, "access_count": access_count},
                ))
        except Exception as e:
            logger.debug(f"[CuriosityScanner] Shallow knowledge scan error: {e}")

        return targets

    # ====================================================================
    # Public API
    # ====================================================================

    def get_targets(self) -> List[CuriosityTarget]:
        """Get current curiosity targets (cached from last scan)."""
        return list(self._targets)

    def get_top_target(self) -> Optional[CuriosityTarget]:
        """Get the highest-urgency curiosity target."""
        if self._targets:
            return self._targets[0]
        return None

    def get_topics_for_message(self, max_topics: int = 3) -> List[str]:
        """Get entity names suitable for curiosity message templates."""
        return [t.entity_name for t in self._targets[:max_topics]]

    def get_question_for_top_target(self) -> Optional[str]:
        """Get a natural-language question for the top curiosity target."""
        top = self.get_top_target()
        if top:
            if not top.question:
                top.question = _generate_question(top)
            return top.question
        return None

    def mark_target_explored(self, entity_id: str) -> None:
        """Mark a curiosity target as explored (remove from active list)."""
        with self._lock:
            self._targets = [t for t in self._targets if t.entity_id != entity_id]

    def get_information_gain_score(self) -> float:
        """Compute overall information gain opportunity (0-1).

        Higher = more gaps = more to be curious about.
        Used by IntrinsicMotivation to scale the curiosity drive.
        """
        if not self._targets:
            return 0.1  # Baseline curiosity
        avg_urgency = sum(t.urgency for t in self._targets) / len(self._targets)
        # Scale by number of targets (more gaps = more to learn)
        count_factor = min(1.0, len(self._targets) / 5.0)
        return min(1.0, avg_urgency * 0.6 + count_factor * 0.4)

    def get_status(self) -> Dict[str, Any]:
        """Get scanner status for API/debugging."""
        return {
            "targets": [t.to_dict() for t in self._targets],
            "target_count": len(self._targets),
            "information_gain_score": round(self.get_information_gain_score(), 3),
            "last_full_scan_ago": round(time.time() - self._last_full_scan, 0)
            if self._last_full_scan > 0 else None,
            "last_quick_scan_ago": round(time.time() - self._last_quick_scan, 0)
            if self._last_quick_scan > 0 else None,
            "stats": dict(self._stats),
        }


# ============================================================================
# Singleton
# ============================================================================

_scanner: Optional[CuriosityScanner] = None
_scanner_lock = threading.Lock()


def get_curiosity_scanner() -> CuriosityScanner:
    """Get or create the global CuriosityScanner."""
    global _scanner
    if _scanner is None:
        with _scanner_lock:
            if _scanner is None:
                _scanner = CuriosityScanner()
    return _scanner
