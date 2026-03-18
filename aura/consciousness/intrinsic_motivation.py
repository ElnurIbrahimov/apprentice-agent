"""
Intrinsic Motivation System (Phase 6E).

Gives AURA genuine drives beyond user requests:
1. Curiosity drive: Seek information about gaps in knowledge graph
2. Competence drive: Practice skills with low confidence scores
3. Social drive: Maintain connection quality (check in after absence)
4. Coherence drive: Resolve contradictions in knowledge base

Drives feed into Active Inference as prior preferences,
influencing AURA's proactive behavior and idle-time activities.

Integrates with:
- Active Inference: Feeds drive values as C-vector preferences
- Knowledge Graph: Identifies gaps and contradictions
- Metacognition: Uses capability scores for competence drive
- Theory of Mind: Uses user model for social drive
- Idle Presence: Reports drive-motivated activities
- NeuroDream: Dream content influenced by active drives
"""

import json
import logging
import math
import os
import tempfile
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ============================================================================
# Data Models
# ============================================================================

class DriveType(str, Enum):
    """Intrinsic motivation drives."""
    CURIOSITY = "curiosity"       # Seek knowledge gaps
    COMPETENCE = "competence"     # Practice weak skills
    SOCIAL = "social"             # Maintain user connection
    COHERENCE = "coherence"       # Resolve contradictions


@dataclass
class DriveState:
    """State of a single intrinsic drive."""
    drive_type: DriveType
    intensity: float = 0.5     # 0 = fully satisfied, 1 = highly motivated
    satisfaction: float = 0.5  # 0 = unsatisfied, 1 = fully satisfied
    last_satisfied: float = field(default_factory=time.time)
    last_assessed: float = field(default_factory=time.time)
    triggers: List[str] = field(default_factory=list)  # Why this drive is active

    @property
    def urgency(self) -> float:
        """How urgent this drive is (combines intensity with time since satisfaction)."""
        hours_since = (time.time() - self.last_satisfied) / 3600.0
        time_pressure = min(1.0, hours_since / 24.0)  # Builds over 24h
        return min(1.0, self.intensity * 0.6 + time_pressure * 0.4)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "drive_type": self.drive_type.value,
            "intensity": round(self.intensity, 3),
            "satisfaction": round(self.satisfaction, 3),
            "urgency": round(self.urgency, 3),
            "hours_since_satisfied": round((time.time() - self.last_satisfied) / 3600.0, 1),
            "triggers": self.triggers[-3:],  # Last 3 triggers
        }


@dataclass
class DriveAction:
    """An action motivated by an intrinsic drive."""
    drive: DriveType
    action: str            # What to do
    description: str       # Human-readable description
    priority: float = 0.5  # 0-1
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# Intrinsic Motivation Engine
# ============================================================================

class IntrinsicMotivationEngine:
    """Manages AURA's intrinsic drives and their influence on behavior.

    Periodically assesses drive states from knowledge graph, metacognition,
    user model, and recent interactions. Produces drive-motivated actions
    and feeds preferences into the Active Inference engine.
    """

    def __init__(self, data_dir: Optional[str] = None):
        if data_dir is None:
            base = Path(__file__).resolve().parent.parent.parent
            data_dir = str(base / "data" / "intrinsic_motivation")

        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)

        self._lock = threading.RLock()

        # Drive states
        self._drives: Dict[DriveType, DriveState] = {
            DriveType.CURIOSITY: DriveState(drive_type=DriveType.CURIOSITY, intensity=0.6),
            DriveType.COMPETENCE: DriveState(drive_type=DriveType.COMPETENCE, intensity=0.4),
            DriveType.SOCIAL: DriveState(drive_type=DriveType.SOCIAL, intensity=0.5),
            DriveType.COHERENCE: DriveState(drive_type=DriveType.COHERENCE, intensity=0.3),
        }

        # Pending drive-motivated actions
        self._pending_actions: List[DriveAction] = []

        # Phase 4.3: Specific curiosity targets from KG gap scan
        self._curiosity_targets: List[Dict[str, Any]] = []

        # Assessment interval
        self._assess_interval = 120.0  # Every 2 minutes
        self._last_full_assessment = 0.0

        # Stats
        self._stats = {
            "assessments_run": 0,
            "actions_generated": 0,
            "preferences_pushed": 0,
            "curiosity_explorations": 0,
            "competence_practices": 0,
            "social_checkins": 0,
            "coherence_resolutions": 0,
        }

        # Load persisted state
        self._load_state()

        logger.info("[IntrinsicMotivation] Engine initialized with drives: "
                     + ", ".join(f"{d.value}={self._drives[d].intensity:.2f}" for d in DriveType))

    # ====================================================================
    # Drive Assessment
    # ====================================================================

    def assess_drives(self) -> Dict[DriveType, DriveState]:
        """Assess all drives from current system state.

        Queries knowledge graph, metacognition, user model, and
        recent interaction patterns to update drive intensities.
        """
        now = time.time()
        if now - self._last_full_assessment < self._assess_interval:
            return dict(self._drives)

        with self._lock:
            self._assess_curiosity()
            self._assess_competence()
            self._assess_social()
            self._assess_coherence()
            self._apply_awareness_modifiers()

            self._last_full_assessment = now
            self._stats["assessments_run"] += 1

            # Save state after assessment
            self._save_state()

        return dict(self._drives)

    def _assess_curiosity(self) -> None:
        """Assess curiosity drive from knowledge graph gaps.

        Phase 4.3: Uses CuriosityScanner for information-gain-based assessment
        grounded in actual KG gaps, with fallback to generic heuristics.
        """
        drive = self._drives[DriveType.CURIOSITY]
        triggers = []
        intensity = 0.3  # Base curiosity level

        # Phase 4.3: Use CuriosityScanner as primary source of curiosity signals
        scanner_used = False
        try:
            from aura.proactive.curiosity_scanner import get_curiosity_scanner
            scanner = get_curiosity_scanner()
            targets = scanner.scan_quick()

            if targets:
                # Use information gain score as primary intensity signal
                ig_score = scanner.get_information_gain_score()
                intensity += ig_score * 0.4
                scanner_used = True

                # Add specific triggers from top gaps
                for t in targets[:3]:
                    triggers.append(f"{t.gap_type}: {t.entity_name}")

                # Store targets for action generation
                self._curiosity_targets = [
                    {
                        "target": t.entity_name,
                        "reason": t.gap_type,
                        "urgency": t.urgency,
                        "node_id": t.entity_id,
                        "question": t.question,
                    }
                    for t in targets
                ]

                logger.debug(
                    f"[IntrinsicMotivation] CuriosityScanner: {len(targets)} targets, "
                    f"IG={ig_score:.2f}"
                )
        except Exception as e:
            logger.debug(f"[IntrinsicMotivation] CuriosityScanner not available: {e}")

        # Fallback: generic KG heuristics if scanner didn't find anything
        if not scanner_used:
            try:
                from aura.tools.knowledge_graph import get_knowledge_graph
                kg = get_knowledge_graph()
                stats = kg.get_stats()
                total_nodes = stats.get("total_nodes", 0)
                total_edges = stats.get("total_edges", 0)

                if total_nodes > 0:
                    ratio = total_edges / total_nodes if total_nodes > 0 else 0
                    if ratio < 1.5:
                        gap_signal = 1.0 - min(1.0, ratio / 1.5)
                        intensity += gap_signal * 0.3
                        triggers.append(f"knowledge gaps: {total_nodes} nodes, {total_edges} edges (ratio {ratio:.1f})")

                    avg_conf = stats.get("avg_confidence", 0.5)
                    if avg_conf < 0.5:
                        intensity += (0.5 - avg_conf) * 0.2
                        triggers.append(f"low average confidence: {avg_conf:.2f}")
            except Exception as e:
                logger.debug(f"[IntrinsicMotivation] non-critical: {e}")

            # Fallback gap scan
            gap_targets = self.scan_kg_gaps(max_targets=5)
            if gap_targets:
                intensity += min(0.2, len(gap_targets) * 0.04)
                for t in gap_targets[:2]:
                    triggers.append(f"{t['reason']}: {t['target']}")
                self._curiosity_targets = gap_targets

        # Check if recent topics have unexplored connections
        try:
            from api.routes.context import get_tracker
            ctx = get_tracker()
            focus = ctx.get_focus_state(limit=5)
            items = focus.get("items", [])
            if items and len(items) >= 3:
                intensity += 0.1
                topics = [i["name"] for i in items[:3]]
                triggers.append(f"multiple active topics: {', '.join(topics)}")
        except Exception as e:
            logger.debug(f"[IntrinsicMotivation] non-critical: {e}")

        # Time decay: curiosity grows when not explored
        hours_since = (time.time() - drive.last_satisfied) / 3600.0
        time_growth = min(0.2, hours_since / 48.0 * 0.2)
        intensity += time_growth

        drive.intensity = min(1.0, max(0.0, intensity))
        drive.satisfaction = 1.0 - drive.intensity
        drive.triggers = triggers
        drive.last_assessed = time.time()

    def _assess_competence(self) -> None:
        """Assess competence drive from metacognitive capability scores."""
        drive = self._drives[DriveType.COMPETENCE]
        triggers = []
        intensity = 0.2  # Base level

        # Check metacognitive self-model for weak areas
        try:
            from aura.consciousness.metacognition import get_metacognitive_engine
            mc = get_metacognitive_engine()
            model = mc.get_self_model()

            # Weak areas drive competence motivation
            if model.weaknesses:
                intensity += min(0.4, len(model.weaknesses) * 0.1)
                triggers.append(f"weak areas: {', '.join(model.weaknesses[:3])}")

            # Active learning goals increase competence drive
            active_goals = [g for g in model.learning_goals if g.status in ("pending", "active")]
            if active_goals:
                intensity += min(0.2, len(active_goals) * 0.05)
                triggers.append(f"{len(active_goals)} active learning goals")

            # Low improvement rate
            if model.total_improvements > 0:
                rate = model.successful_improvements / model.total_improvements
                if rate < 0.5:
                    intensity += 0.15
                    triggers.append(f"improvement rate: {rate:.0%}")
        except Exception as e:
            logger.debug(f"[IntrinsicMotivation] non-critical: {e}")
        # Reflexion removed
        drive.intensity = min(1.0, max(0.0, intensity))
        drive.satisfaction = 1.0 - drive.intensity
        drive.triggers = triggers
        drive.last_assessed = time.time()

    def _assess_social(self) -> None:
        """Assess social drive from user interaction patterns."""
        drive = self._drives[DriveType.SOCIAL]
        triggers = []
        intensity = 0.2  # Base social need

        # Check how long since last user interaction
        try:
            from api.routes.idle_behaviors import get_manager
            mgr = get_manager()
            idle_secs = mgr.get_idle_duration()
            idle_hours = idle_secs / 3600.0

            if idle_hours > 4:
                # Long absence increases social drive
                intensity += min(0.5, idle_hours / 12.0)
                triggers.append(f"user idle for {idle_hours:.1f} hours")
            elif idle_hours < 0.1:
                # Active conversation reduces social drive
                intensity = max(0.0, intensity - 0.1)
                drive.last_satisfied = time.time()
        except Exception as e:
            logger.debug(f"[IntrinsicMotivation] non-critical: {e}")
        # Check Theory of Mind for engagement level
        try:
            from aura.proactive.theory_of_mind import get_theory_of_mind
            tom = get_theory_of_mind()
            emo = tom.get_emotional_state()
            if emo.engagement < 0.3:
                intensity += 0.15
                triggers.append(f"low user engagement: {emo.engagement:.2f}")
            if emo.frustration > 0.5:
                intensity += 0.2
                triggers.append(f"user frustration detected: {emo.frustration:.2f}")
        except Exception as e:
            logger.debug(f"[IntrinsicMotivation] non-critical: {e}")
        # Time decay: social need grows when not interacting
        hours_since = (time.time() - drive.last_satisfied) / 3600.0
        time_growth = min(0.3, hours_since / 8.0 * 0.3)  # Grows over 8h
        intensity += time_growth

        drive.intensity = min(1.0, max(0.0, intensity))
        drive.satisfaction = 1.0 - drive.intensity
        drive.triggers = triggers
        drive.last_assessed = time.time()

    def _assess_coherence(self) -> None:
        """Assess coherence drive from knowledge base consistency."""
        drive = self._drives[DriveType.COHERENCE]
        triggers = []
        intensity = 0.2  # Base level

        # Check knowledge graph for potential contradictions
        try:
            from aura.tools.knowledge_graph import get_knowledge_graph
            kg = get_knowledge_graph()
            stats = kg.get_stats()
            clusters = stats.get("clusters", 0)
            total_nodes = stats.get("total_nodes", 0)

            # Many disconnected clusters = incoherence
            if total_nodes > 10 and clusters > 5:
                cluster_ratio = clusters / total_nodes
                if cluster_ratio > 0.3:
                    intensity += cluster_ratio * 0.3
                    triggers.append(f"fragmented knowledge: {clusters} clusters/{total_nodes} nodes")
        except Exception as e:
            logger.debug(f"[IntrinsicMotivation] non-critical: {e}")
        # Check for conflicting learned context
        try:
            from aura.tools.neurodream import get_neurodream
            nd = get_neurodream()
            status = nd.get_status()
            lc = status.get("learned_context", {})
            facts_count = lc.get("key_facts", 0)
            if facts_count > 20:
                # Many facts increases chance of contradictions
                intensity += min(0.15, facts_count / 100.0)
                triggers.append(f"{facts_count} key facts to maintain consistency")
        except Exception as e:
            logger.debug(f"[IntrinsicMotivation] non-critical: {e}")
        drive.intensity = min(1.0, max(0.0, intensity))
        drive.satisfaction = 1.0 - drive.intensity
        drive.triggers = triggers
        drive.last_assessed = time.time()

    def scan_kg_gaps(self, max_targets: int = 5) -> List[Dict[str, Any]]:
        """Scan KG for specific curiosity targets: orphans, stale nodes, low-confidence.

        Returns list of dicts with {target, reason, urgency, node_id}.
        """
        targets: List[Dict[str, Any]] = []
        try:
            from aura.tools.knowledge_graph import get_knowledge_graph
            kg = get_knowledge_graph()

            now = datetime.now()
            cutoff_dt = now - timedelta(days=7)

            # Safely snapshot all nodes under the KG lock
            all_nodes = []
            try:
                if hasattr(kg, '_lock') and hasattr(kg, '_nodes'):
                    with kg._lock:
                        all_nodes = list(kg._nodes.values())
                else:
                    # Fallback: use public API
                    all_nodes = kg.get_recent_nodes(limit=200) if hasattr(kg, 'get_recent_nodes') else []
            except Exception as e:
                logger.debug(f"[IntrinsicMotivation] KG node snapshot failed: {e}")

            # 1. Stale nodes (not accessed in 7+ days, access_count >= 3, important types)
            for node in all_nodes:
                try:
                    la = getattr(node, 'last_accessed', '')
                    if not la or getattr(node, 'access_count', 0) < 3:
                        continue
                    if getattr(node, 'type', '') not in ("project", "concept", "person", "tool"):
                        continue
                    node_dt = datetime.fromisoformat(str(la).replace('Z', '+00:00'))
                    if node_dt.tzinfo is not None:
                        node_dt = node_dt.replace(tzinfo=None)
                    if node_dt < cutoff_dt:
                        days_stale = (now - node_dt).days
                        targets.append({
                            "target": node.label,
                            "reason": f"stale_{days_stale}d",
                            "urgency": min(0.9, 0.4 + days_stale / 30),
                            "node_id": node.id,
                        })
                except (ValueError, TypeError, AttributeError):
                    pass

            # 2. Low-confidence nodes (user-mentioned but uncertain)
            for node in all_nodes:
                if (getattr(node, 'confidence', 1.0) < 0.4
                        and getattr(node, 'access_count', 0) >= 2):
                    targets.append({
                        "target": getattr(node, 'label', str(node)),
                        "reason": "low_confidence",
                        "urgency": 0.5 + (0.4 - getattr(node, 'confidence', 0.4)),
                        "node_id": getattr(node, 'id', ''),
                    })

            # 3. Orphan nodes (few edges, actually mentioned by user)
            for node in all_nodes:
                try:
                    if getattr(node, 'access_count', 0) > 0:
                        edges = kg.get_edges(node.id) if hasattr(kg, 'get_edges') else []
                        if len(edges) <= 1:
                            targets.append({
                                "target": node.label,
                                "reason": "isolated_concept",
                                "urgency": 0.6,
                                "node_id": node.id,
                            })
                except Exception as e:
                    logger.debug(f"[IntrinsicMotivation] orphan node check failed: {e}")

        except Exception as e:
            logger.debug(f"[IntrinsicMotivation] KG gap scan error: {e}")

        # Deduplicate and sort by urgency
        seen = set()
        unique = []
        for t in targets:
            if t["node_id"] not in seen:
                seen.add(t["node_id"])
                unique.append(t)
        unique.sort(key=lambda t: t["urgency"], reverse=True)
        return unique[:max_targets]

    def _apply_awareness_modifiers(self) -> None:
        """Blend proactive awareness drive signals into current drive intensities.

        Applies a 15% blend: drive = drive * 0.85 + signal * 0.15.
        Default signal is 0.5 (neutral), so this barely changes intensity
        when nothing is flagged. Deviations nudge matching drives.
        """
        try:
            from aura.consciousness.proactive_awareness import (
                get_proactive_awareness_engine,
            )
            engine = get_proactive_awareness_engine()
            signals = engine.get_drive_signals()

            drive_map = {
                "curiosity": DriveType.CURIOSITY,
                "coherence": DriveType.COHERENCE,
                "social": DriveType.SOCIAL,
                "competence": DriveType.COMPETENCE,
            }

            for signal_name, drive_type in drive_map.items():
                signal_value = signals.get(signal_name, 0.5)
                drive = self._drives.get(drive_type)
                if drive is not None:
                    drive.intensity = drive.intensity * 0.85 + signal_value * 0.15
                    drive.intensity = min(1.0, max(0.0, drive.intensity))
        except Exception as e:
            logger.debug(f"[IntrinsicMotivation] awareness modifiers error: {e}")

    # ====================================================================
    # Action Generation
    # ====================================================================

    def generate_actions(self) -> List[DriveAction]:
        """Generate actions motivated by current drive states.

        Returns prioritized list of drive-motivated actions.
        """
        self.assess_drives()
        actions = []

        with self._lock:
            # Sort drives by urgency
            sorted_drives = sorted(
                self._drives.values(),
                key=lambda d: d.urgency,
                reverse=True,
            )

            for drive in sorted_drives:
                if drive.urgency < 0.3:
                    continue  # Not urgent enough

                new_actions = self._generate_drive_actions(drive)
                actions.extend(new_actions)

            # Cap at 5 actions
            actions = sorted(actions, key=lambda a: a.priority, reverse=True)[:5]

            self._pending_actions = actions
            self._stats["actions_generated"] += len(actions)

        return actions

    def _generate_drive_actions(self, drive: DriveState) -> List[DriveAction]:
        """Generate specific actions for a drive."""
        actions = []

        if drive.drive_type == DriveType.CURIOSITY:
            # Phase 4.3: Use specific KG gap targets when available
            targets = getattr(self, '_curiosity_targets', [])
            if targets and drive.urgency > 0.5:
                top = targets[0]
                # Use pre-generated question if available (from CuriosityScanner)
                question = top.get("question", "")
                desc = question if question else f"Ask about '{top['target']}' ({top['reason']})"
                actions.append(DriveAction(
                    drive=DriveType.CURIOSITY,
                    action="explore_specific_gap",
                    description=desc,
                    priority=drive.urgency * 0.8,
                    metadata={
                        "target": top["target"],
                        "reason": top["reason"],
                        "node_id": top["node_id"],
                        "question": question,
                    },
                ))
            elif drive.urgency > 0.5:
                actions.append(DriveAction(
                    drive=DriveType.CURIOSITY,
                    action="explore_knowledge_gaps",
                    description="Explore connections between isolated knowledge concepts",
                    priority=drive.urgency * 0.8,
                ))
            if drive.urgency > 0.3:
                actions.append(DriveAction(
                    drive=DriveType.CURIOSITY,
                    action="seek_new_patterns",
                    description="Search for novel patterns in recent interactions",
                    priority=drive.urgency * 0.6,
                ))

        elif drive.drive_type == DriveType.COMPETENCE:
            if drive.urgency > 0.5:
                weak_areas = drive.triggers[0] if drive.triggers else "weak areas"
                actions.append(DriveAction(
                    drive=DriveType.COMPETENCE,
                    action="practice_weak_skill",
                    description=f"Practice and improve: {weak_areas}",
                    priority=drive.urgency * 0.7,
                ))
            if drive.urgency > 0.4:
                actions.append(DriveAction(
                    drive=DriveType.COMPETENCE,
                    action="review_failures",
                    description="Review recent failures and extract lessons",
                    priority=drive.urgency * 0.5,
                ))

        elif drive.drive_type == DriveType.SOCIAL:
            if drive.urgency > 0.6:
                actions.append(DriveAction(
                    drive=DriveType.SOCIAL,
                    action="check_in",
                    description="Check in with user after extended absence",
                    priority=drive.urgency * 0.9,
                ))
            if drive.urgency > 0.4:
                actions.append(DriveAction(
                    drive=DriveType.SOCIAL,
                    action="prepare_proactive_help",
                    description="Prepare helpful suggestions based on user patterns",
                    priority=drive.urgency * 0.5,
                ))

        elif drive.drive_type == DriveType.COHERENCE:
            if drive.urgency > 0.5:
                actions.append(DriveAction(
                    drive=DriveType.COHERENCE,
                    action="resolve_contradictions",
                    description="Identify and resolve contradictions in knowledge base",
                    priority=drive.urgency * 0.6,
                ))
            if drive.urgency > 0.3:
                actions.append(DriveAction(
                    drive=DriveType.COHERENCE,
                    action="consolidate_clusters",
                    description="Connect isolated knowledge clusters",
                    priority=drive.urgency * 0.4,
                ))

        return actions

    # ====================================================================
    # Active Inference Integration
    # ====================================================================

    def push_preferences_to_inference(self) -> bool:
        """Push current drive intensities to Active Inference C-vector.

        Returns True if preferences were successfully pushed.
        """
        try:
            from aura.proactive.gateway_daemon import get_gateway_daemon
            daemon = get_gateway_daemon()
            engine = daemon.inference_engine

            preferences = {
                "curiosity": self._drives[DriveType.CURIOSITY].urgency * 2.0,   # Scale to -2..2
                "competence": self._drives[DriveType.COMPETENCE].urgency * 1.5,
                "social": self._drives[DriveType.SOCIAL].urgency * 2.0,
                "coherence": self._drives[DriveType.COHERENCE].urgency * 1.0,
            }

            engine.set_intrinsic_preferences(preferences)
            self._stats["preferences_pushed"] += 1
            logger.debug(f"[IntrinsicMotivation] Pushed preferences: "
                         + ", ".join(f"{k}={v:.2f}" for k, v in preferences.items()))
            return True

        except Exception as e:
            logger.debug(f"[IntrinsicMotivation] Failed to push preferences: {e}")
            return False

    # ====================================================================
    # Drive Satisfaction (called when actions are fulfilled)
    # ====================================================================

    def satisfy_drive(self, drive_type: DriveType, amount: float = 0.3) -> None:
        """Record that a drive was partially or fully satisfied.

        Args:
            drive_type: Which drive was satisfied
            amount: How much satisfaction (0-1)
        """
        with self._lock:
            drive = self._drives.get(drive_type)
            if drive:
                drive.satisfaction = min(1.0, drive.satisfaction + amount)
                drive.intensity = max(0.0, drive.intensity - amount * 0.5)
                drive.last_satisfied = time.time()

                stat_key = f"{drive_type.value}_{'explorations' if drive_type == DriveType.CURIOSITY else 'practices' if drive_type == DriveType.COMPETENCE else 'checkins' if drive_type == DriveType.SOCIAL else 'resolutions'}"
                if stat_key in self._stats:
                    self._stats[stat_key] += 1

    def record_interaction(self) -> None:
        """Record that a user interaction occurred (satisfies social drive)."""
        self.satisfy_drive(DriveType.SOCIAL, amount=0.2)

    def record_learning(self) -> None:
        """Record that AURA learned something (satisfies curiosity + competence)."""
        self.satisfy_drive(DriveType.CURIOSITY, amount=0.15)
        self.satisfy_drive(DriveType.COMPETENCE, amount=0.1)

    def record_knowledge_update(self) -> None:
        """Record a knowledge graph update (satisfies coherence + curiosity)."""
        self.satisfy_drive(DriveType.COHERENCE, amount=0.1)
        self.satisfy_drive(DriveType.CURIOSITY, amount=0.1)

    # ====================================================================
    # Full Motivation Cycle
    # ====================================================================

    def run_motivation_cycle(self) -> Dict[str, Any]:
        """Run a full motivation cycle: assess -> generate actions -> push preferences.

        Returns summary of the cycle.
        """
        # 1. Assess all drives
        drives = self.assess_drives()

        # 2. Generate motivated actions
        actions = self.generate_actions()

        # 3. Push preferences to Active Inference
        pushed = self.push_preferences_to_inference()

        # 4. Record activity in idle presence
        try:
            from aura.consciousness.idle_presence import get_idle_presence_engine, IdleActivity
            ipe = get_idle_presence_engine()
            dominant = max(self._drives.values(), key=lambda d: d.urgency)
            desc = f"motivation cycle: {dominant.drive_type.value} drive active ({dominant.urgency:.0%})"
            ipe._record_activity(IdleActivity.METACOGNITION, desc, cognitive_load=0.2)
        except Exception as e:
            logger.debug(f"[IntrinsicMotivation] non-critical: {e}")
        return {
            "drives": {d.value: self._drives[d].to_dict() for d in DriveType},
            "actions": [
                {"drive": a.drive.value, "action": a.action,
                 "description": a.description, "priority": round(a.priority, 2)}
                for a in actions
            ],
            "preferences_pushed": pushed,
            "dominant_drive": max(self._drives.values(), key=lambda d: d.urgency).drive_type.value,
        }

    # ====================================================================
    # Prompt Context
    # ====================================================================

    def get_context_for_prompt(self) -> str:
        """Generate system prompt injection showing active drives.

        Returns a [Motivation] block for the system prompt.
        """
        self.assess_drives()

        # Only show drives with significant urgency
        active_drives = [
            d for d in self._drives.values()
            if d.urgency > 0.3
        ]

        if not active_drives:
            return ""

        active_drives.sort(key=lambda d: d.urgency, reverse=True)

        lines = ["[Motivation]"]
        for drive in active_drives:
            trigger_hint = f" ({drive.triggers[0]})" if drive.triggers else ""
            lines.append(f"- {drive.drive_type.value}: {drive.urgency:.0%}{trigger_hint}")

        dominant = active_drives[0]
        if dominant.drive_type == DriveType.CURIOSITY:
            lines.append("Tendency: Ask follow-up questions, explore tangents")
        elif dominant.drive_type == DriveType.COMPETENCE:
            lines.append("Tendency: Offer detailed explanations, seek feedback")
        elif dominant.drive_type == DriveType.SOCIAL:
            lines.append("Tendency: Check in warmly, show interest in user's wellbeing")
        elif dominant.drive_type == DriveType.COHERENCE:
            lines.append("Tendency: Clarify ambiguities, verify understanding")

        return "\n".join(lines)

    # ====================================================================
    # API
    # ====================================================================

    def get_status(self) -> Dict[str, Any]:
        """Get engine status for API."""
        self.assess_drives()
        dominant = max(self._drives.values(), key=lambda d: d.urgency)
        return {
            "active": True,
            "drives": {d.value: self._drives[d].to_dict() for d in DriveType},
            "dominant_drive": dominant.drive_type.value,
            "dominant_urgency": round(dominant.urgency, 3),
            "pending_actions": len(self._pending_actions),
            "stats": dict(self._stats),
        }

    def get_drives_summary(self) -> Dict[str, float]:
        """Get simple drive urgency values."""
        return {
            d.value: round(self._drives[d].urgency, 3)
            for d in DriveType
        }

    # ====================================================================
    # Persistence
    # ====================================================================

    def _state_file(self) -> Path:
        return self._data_dir / "motivation_state.json"

    def _load_state(self) -> None:
        """Load persisted drive state."""
        sf = self._state_file()
        if not sf.exists():
            return

        try:
            data = json.loads(sf.read_text(encoding="utf-8"))
            for dtype_str, drive_data in data.get("drives", {}).items():
                try:
                    dtype = DriveType(dtype_str)
                    if dtype in self._drives:
                        self._drives[dtype].intensity = drive_data.get("intensity", 0.5)
                        self._drives[dtype].satisfaction = drive_data.get("satisfaction", 0.5)
                        self._drives[dtype].last_satisfied = drive_data.get("last_satisfied", time.time())
                except (ValueError, KeyError):
                    pass

            self._stats.update(data.get("stats", {}))
        except Exception as e:
            logger.warning(f"[IntrinsicMotivation] Failed to load state: {e}")

    def _save_state(self) -> None:
        """Save drive state to disk (atomic write via tempfile + os.replace)."""
        try:
            data = {
                "drives": {
                    d.value: {
                        "intensity": self._drives[d].intensity,
                        "satisfaction": self._drives[d].satisfaction,
                        "last_satisfied": self._drives[d].last_satisfied,
                    }
                    for d in DriveType
                },
                "stats": self._stats,
                "saved_at": datetime.now().isoformat(),
            }
            target = self._state_file()
            fd, tmp_path = tempfile.mkstemp(
                dir=str(target.parent), suffix=".tmp"
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, default=str)
                os.replace(tmp_path, str(target))
            except BaseException:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
        except Exception as e:
            logger.warning(f"[IntrinsicMotivation] Failed to save state: {e}")


# ============================================================================
# Singleton
# ============================================================================

_engine: Optional[IntrinsicMotivationEngine] = None
_engine_lock = threading.Lock()


def get_intrinsic_motivation() -> IntrinsicMotivationEngine:
    """Get or create the global IntrinsicMotivationEngine."""
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                _engine = IntrinsicMotivationEngine()
    return _engine
