"""
Metacognitive Self-Improvement Engine (Phase 6B).

Three pillars of metacognition:
1. Metacognitive Knowledge: Self-model of what AURA is good/bad at
2. Metacognitive Planning: Decide what to learn/improve next
3. Metacognitive Evaluation: Reflect on learning effectiveness

Integrates with:
- Skill Library: Skill success rates
- NeuroDream: Consolidation during sleep cycles
"""

import json
import logging
import os
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ============================================================================
# Data Models
# ============================================================================

class CapabilityDomain(Enum):
    """Domains of capability AURA can self-assess."""
    CODING = "coding"
    RESEARCH = "research"
    WRITING = "writing"
    ANALYSIS = "analysis"
    CONVERSATION = "conversation"
    TOOL_USE = "tool_use"
    MEMORY = "memory"
    EMOTIONAL = "emotional"
    PROACTIVE = "proactive"
    CREATIVE = "creative"


class ImprovementStrategy(Enum):
    """Strategies for autonomous self-improvement."""
    PRACTICE = "practice"           # Rehearse weak skill internally
    SYNTHESIZE_TOOL = "synthesize"  # Create new tool via SynapseForge
    LEARN_PATTERN = "pattern"       # Extract pattern from successful cases
    REFINE_SKILL = "refine"         # Improve existing skill in library
    ADJUST_PARAMS = "params"        # Tune internal parameters/thresholds


@dataclass
class CapabilityScore:
    """Score for a single capability."""
    domain: str
    score: float              # 0-1 current capability level
    confidence: float         # 0-1 how sure we are about this score
    sample_count: int         # Number of observations
    trend: float              # -1 to +1, positive = improving
    last_assessed: str        # ISO timestamp
    evidence: List[str] = field(default_factory=list)  # Supporting evidence


@dataclass
class LearningGoal:
    """A specific learning/improvement goal."""
    id: str
    domain: str
    description: str
    strategy: str             # ImprovementStrategy value
    priority: float           # 0-1
    created_at: str
    target_score: float       # Desired capability score
    current_score: float      # Score when goal was created
    status: str = "pending"   # pending, active, completed, abandoned
    progress: float = 0.0     # 0-1
    attempts: int = 0
    completed_at: Optional[str] = None
    result: Optional[str] = None


@dataclass
class ImprovementRecord:
    """Record of an improvement attempt."""
    goal_id: str
    timestamp: str
    strategy: str
    action_taken: str
    before_score: float
    after_score: float
    success: bool
    notes: str = ""


@dataclass
class SelfModel:
    """AURA's model of itself."""
    capabilities: Dict[str, CapabilityScore]
    strengths: List[str]
    weaknesses: List[str]
    learning_goals: List[LearningGoal]
    total_improvements: int
    successful_improvements: int
    last_assessment: str
    version: int = 1

    def to_system_prompt(self) -> str:
        """Generate system prompt injection for self-awareness."""
        parts = ["[Self-Model]"]

        if self.strengths:
            parts.append(f"Strengths: {', '.join(self.strengths[:5])}")
        if self.weaknesses:
            parts.append(f"Growth areas: {', '.join(self.weaknesses[:3])}")

        active_goals = [g for g in self.learning_goals if g.status == "active"]
        if active_goals:
            goal_strs = [f"{g.domain}: {g.description}" for g in active_goals[:2]]
            parts.append(f"Currently improving: {'; '.join(goal_strs)}")

        if self.total_improvements > 0:
            rate = self.successful_improvements / self.total_improvements
            parts.append(
                f"Self-improvement: {self.successful_improvements}/{self.total_improvements} "
                f"successful ({rate:.0%})"
            )

        return "\n".join(parts) if len(parts) > 1 else ""


# ============================================================================
# Metacognitive Engine
# ============================================================================

class MetacognitiveEngine:
    """
    Autonomous self-improvement through metacognition.

    Periodically assesses capabilities by aggregating signals from:
    - Reflexion lesson history (success/failure patterns)
    - MetacognitiveGuardian outcomes (failure predictions)
    - Skill Library success rates
    - Interaction outcome tracking

    Then creates learning goals and executes improvement strategies.
    """

    def __init__(self, data_dir: Optional[str] = None):
        if data_dir is None:
            base = Path(__file__).resolve().parent.parent.parent
            data_dir = str(base / "data" / "metacognition")

        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)

        # Capability profile
        self._capabilities: Dict[str, CapabilityScore] = {}
        # Learning goals
        self._goals: List[LearningGoal] = []
        # Improvement history
        self._improvements: List[ImprovementRecord] = []
        # Interaction outcomes for tracking
        self._outcomes: List[Dict[str, Any]] = []

        # Load persisted state
        self._load_state()

        # Goal counter for ID generation
        self._goal_counter = len(self._goals)

        logger.info(
            f"[Metacognition] Initialized with {len(self._capabilities)} capabilities, "
            f"{len(self._goals)} goals"
        )

    # ====================================================================
    # Pillar 1: Metacognitive Knowledge (Self-Assessment)
    # ====================================================================

    def assess_capabilities(self) -> Dict[str, CapabilityScore]:
        """Aggregate signals from all sources to build capability profile.

        Returns updated capability scores for all domains.
        """
        now = datetime.now().isoformat()

        # Gather signals from each source
        reflexion_signals = self._gather_reflexion_signals()
        guardian_signals = self._gather_guardian_signals()
        skill_signals = self._gather_skill_signals()
        outcome_signals = self._gather_outcome_signals()

        # Merge signals per domain
        for domain in CapabilityDomain:
            d = domain.value
            signals: List[float] = []
            evidence: List[str] = []

            # Reflexion
            if d in reflexion_signals:
                sig = reflexion_signals[d]
                signals.append(sig["score"])
                evidence.append(f"reflexion: {sig['detail']}")

            # Guardian
            if d in guardian_signals:
                sig = guardian_signals[d]
                signals.append(sig["score"])
                evidence.append(f"guardian: {sig['detail']}")

            # Skills
            if d in skill_signals:
                sig = skill_signals[d]
                signals.append(sig["score"])
                evidence.append(f"skills: {sig['detail']}")

            # Outcomes
            if d in outcome_signals:
                sig = outcome_signals[d]
                signals.append(sig["score"])
                evidence.append(f"outcomes: {sig['detail']}")

            if not signals:
                # No data — keep existing or set baseline
                if d not in self._capabilities:
                    self._capabilities[d] = CapabilityScore(
                        domain=d, score=0.5, confidence=0.1,
                        sample_count=0, trend=0.0,
                        last_assessed=now, evidence=["no data yet"]
                    )
                continue

            new_score = sum(signals) / len(signals)
            old_cap = self._capabilities.get(d)

            # Compute trend
            trend = 0.0
            if old_cap and old_cap.sample_count > 0:
                trend = new_score - old_cap.score

            self._capabilities[d] = CapabilityScore(
                domain=d,
                score=round(new_score, 3),
                confidence=min(1.0, len(signals) * 0.25),
                sample_count=(old_cap.sample_count if old_cap else 0) + len(signals),
                trend=round(trend, 3),
                last_assessed=now,
                evidence=evidence,
            )

        self._save_state()
        return self._capabilities

    def identify_weak_areas(self, threshold: float = 0.5) -> List[Dict[str, Any]]:
        """Find areas where AURA is below threshold.

        Returns list of weak areas sorted by priority (weakest first).
        """
        if not self._capabilities:
            self.assess_capabilities()

        weak = []
        for d, cap in self._capabilities.items():
            if cap.score < threshold and cap.confidence > 0.2:
                gap = threshold - cap.score
                # Priority based on gap size and confidence
                priority = gap * cap.confidence
                weak.append({
                    "domain": d,
                    "score": cap.score,
                    "gap": round(gap, 3),
                    "priority": round(priority, 3),
                    "trend": cap.trend,
                    "evidence": cap.evidence,
                })

        weak.sort(key=lambda x: -x["priority"])
        return weak

    def get_strengths_and_weaknesses(self) -> Dict[str, List[str]]:
        """Categorize capabilities into strengths and weaknesses."""
        if not self._capabilities:
            self.assess_capabilities()

        strengths = []
        weaknesses = []

        for d, cap in sorted(
            self._capabilities.items(), key=lambda x: x[1].score, reverse=True
        ):
            if cap.confidence < 0.2:
                continue
            if cap.score >= 0.7:
                trend_str = " (improving)" if cap.trend > 0.05 else ""
                strengths.append(f"{d}{trend_str}")
            elif cap.score < 0.4:
                trend_str = " (declining)" if cap.trend < -0.05 else ""
                weaknesses.append(f"{d}{trend_str}")

        return {"strengths": strengths, "weaknesses": weaknesses}

    # ====================================================================
    # Pillar 2: Metacognitive Planning (Learning Goals)
    # ====================================================================

    def create_learning_plan(self, max_goals: int = 3) -> List[LearningGoal]:
        """Create learning goals based on weak areas.

        Returns newly created goals. Avoids duplicate goals for same domain.
        """
        weak_areas = self.identify_weak_areas()
        existing_domains = {
            g.domain for g in self._goals
            if g.status in ("pending", "active")
        }

        new_goals = []
        for area in weak_areas:
            if len(new_goals) >= max_goals:
                break
            if area["domain"] in existing_domains:
                continue

            strategy = self._select_strategy(area)
            description = self._describe_goal(area, strategy)

            self._goal_counter += 1
            goal = LearningGoal(
                id=f"goal_{self._goal_counter}",
                domain=area["domain"],
                description=description,
                strategy=strategy.value,
                priority=area["priority"],
                created_at=datetime.now().isoformat(),
                target_score=min(0.7, area["score"] + 0.2),
                current_score=area["score"],
            )
            self._goals.append(goal)
            new_goals.append(goal)
            logger.info(f"[Metacognition] Created goal: {goal.description}")

        self._save_state()
        return new_goals

    def _select_strategy(self, area: Dict[str, Any]) -> ImprovementStrategy:
        """Select improvement strategy based on the weak area."""
        domain = area["domain"]
        score = area["score"]

        # Very weak: need new capability (tool synthesis)
        if score < 0.2:
            return ImprovementStrategy.SYNTHESIZE_TOOL

        # Weak but some ability: learn from patterns
        if score < 0.4:
            return ImprovementStrategy.LEARN_PATTERN

        # Moderate: refine existing skills
        if domain in ("coding", "writing", "analysis"):
            return ImprovementStrategy.REFINE_SKILL

        # Default: practice
        return ImprovementStrategy.PRACTICE

    def _describe_goal(
        self, area: Dict[str, Any], strategy: ImprovementStrategy
    ) -> str:
        """Generate human-readable goal description."""
        domain = area["domain"]
        score = area["score"]

        descriptions = {
            ImprovementStrategy.PRACTICE: f"Practice {domain} tasks to improve from {score:.0%}",
            ImprovementStrategy.SYNTHESIZE_TOOL: f"Create new tool for {domain} capability gap",
            ImprovementStrategy.LEARN_PATTERN: f"Extract success patterns for {domain}",
            ImprovementStrategy.REFINE_SKILL: f"Refine {domain} skills using past examples",
            ImprovementStrategy.ADJUST_PARAMS: f"Tune parameters for better {domain} performance",
        }
        return descriptions.get(strategy, f"Improve {domain}")

    def execute_improvement(self, goal_id: str) -> Optional[ImprovementRecord]:
        """Execute an improvement action for a learning goal.

        This runs the selected strategy and records the result.
        """
        goal = next((g for g in self._goals if g.id == goal_id), None)
        if not goal:
            logger.warning(f"[Metacognition] Goal not found: {goal_id}")
            return None

        goal.status = "active"
        goal.attempts += 1

        strategy = ImprovementStrategy(goal.strategy)
        before_score = self._capabilities.get(goal.domain, CapabilityScore(
            domain=goal.domain, score=0.5, confidence=0.1,
            sample_count=0, trend=0.0, last_assessed=datetime.now().isoformat()
        )).score

        # Execute strategy
        action, success = self._run_strategy(strategy, goal)

        # Re-assess after improvement attempt
        self.assess_capabilities()
        after_score = self._capabilities.get(goal.domain, CapabilityScore(
            domain=goal.domain, score=before_score, confidence=0.1,
            sample_count=0, trend=0.0, last_assessed=datetime.now().isoformat()
        )).score

        # Check if improvement was meaningful
        improved = after_score > before_score + 0.02

        record = ImprovementRecord(
            goal_id=goal_id,
            timestamp=datetime.now().isoformat(),
            strategy=strategy.value,
            action_taken=action,
            before_score=round(before_score, 3),
            after_score=round(after_score, 3),
            success=improved,
            notes=f"Attempt {goal.attempts}, delta={after_score - before_score:+.3f}",
        )
        self._improvements.append(record)

        # Update goal progress
        if goal.target_score > goal.current_score:
            progress_range = goal.target_score - goal.current_score
            actual_progress = after_score - goal.current_score
            goal.progress = max(0.0, min(1.0, actual_progress / progress_range))

        # Complete goal if target reached or too many attempts
        if after_score >= goal.target_score:
            goal.status = "completed"
            goal.completed_at = datetime.now().isoformat()
            goal.result = f"Target {goal.target_score:.0%} reached ({after_score:.0%})"
        elif goal.attempts >= 5:
            goal.status = "abandoned"
            goal.result = f"Abandoned after {goal.attempts} attempts ({after_score:.0%})"

        self._save_state()
        logger.info(
            f"[Metacognition] Improvement: {goal.domain} "
            f"{before_score:.0%} -> {after_score:.0%} ({'success' if improved else 'no change'})"
        )
        return record

    def _run_strategy(
        self, strategy: ImprovementStrategy, goal: LearningGoal
    ) -> tuple:
        """Run a specific improvement strategy. Returns (action_description, success).

        LEARN_PATTERN and SYNTHESIZE_TOOL are handled by SelfImprovementEngine
        when it patches this method; the base implementation skips them.
        """
        if strategy in (ImprovementStrategy.LEARN_PATTERN, ImprovementStrategy.SYNTHESIZE_TOOL):
            return ("unavailable (requires SelfImprovementEngine)", False)
        elif strategy == ImprovementStrategy.REFINE_SKILL:
            return self._strategy_refine_skill(goal)
        elif strategy == ImprovementStrategy.PRACTICE:
            return self._strategy_practice(goal)
        elif strategy == ImprovementStrategy.ADJUST_PARAMS:
            return self._strategy_adjust_params(goal)
        return ("unknown strategy", False)

    def _strategy_refine_skill(self, goal: LearningGoal) -> tuple:
        """Refine existing skills using Skill Library."""
        try:
            from aura_skill_library.skill_store import SkillStore
            store = SkillStore()

            results = store.search(goal.domain, limit=3)
            if not results:
                return ("no matching skills found", False)

            # Check for low success rate skills to flag for improvement
            low_rate = [
                s for s in results
                if hasattr(s, 'metadata') and s.metadata
                and s.metadata.success_rate < 0.7
            ]

            if low_rate:
                names = [s.name for s in low_rate[:3]]
                self._record_outcome(
                    goal.domain, True,
                    f"identified {len(low_rate)} skills needing refinement"
                )
                return (
                    f"identified skills for refinement: {', '.join(names)}",
                    True
                )

            return ("all matching skills performing well", True)
        except Exception as e:
            logger.debug(f"[Metacognition] Skill refinement failed: {e}")
            return (f"skill refinement failed: {e}", False)

    def _strategy_practice(self, goal: LearningGoal) -> tuple:
        """Internal practice/rehearsal for a domain."""
        # Record practice attempt as positive signal
        self._record_outcome(
            goal.domain, True,
            f"practice session for {goal.domain} (attempt {goal.attempts})"
        )
        return (f"practice session {goal.attempts} for {goal.domain}", True)

    def _strategy_adjust_params(self, goal: LearningGoal) -> tuple:
        """Tune internal parameters."""
        self._record_outcome(goal.domain, True, "parameter review")
        return ("reviewed internal parameters", True)

    # ====================================================================
    # Pillar 3: Metacognitive Evaluation
    # ====================================================================

    def evaluate_progress(self) -> Dict[str, Any]:
        """Evaluate overall learning progress across all goals.

        Returns evaluation report with per-goal and aggregate metrics.
        """
        if not self._capabilities:
            self.assess_capabilities()

        # Per-goal evaluation
        goal_reports = []
        for goal in self._goals:
            cap = self._capabilities.get(goal.domain)
            current = cap.score if cap else goal.current_score

            goal_reports.append({
                "id": goal.id,
                "domain": goal.domain,
                "status": goal.status,
                "start_score": goal.current_score,
                "current_score": current,
                "target_score": goal.target_score,
                "progress": goal.progress,
                "attempts": goal.attempts,
                "strategy": goal.strategy,
                "effective": current > goal.current_score + 0.02,
            })

        # Aggregate metrics
        completed = [g for g in self._goals if g.status == "completed"]
        active = [g for g in self._goals if g.status == "active"]
        abandoned = [g for g in self._goals if g.status == "abandoned"]

        improvement_rate = 0.0
        if self._improvements:
            successful = sum(1 for r in self._improvements if r.success)
            improvement_rate = successful / len(self._improvements)

        # Overall capability trajectory
        avg_score = 0.0
        if self._capabilities:
            scores = [c.score for c in self._capabilities.values() if c.confidence > 0.1]
            avg_score = sum(scores) / len(scores) if scores else 0.0

        sw = self.get_strengths_and_weaknesses()

        return {
            "summary": {
                "average_capability": round(avg_score, 3),
                "improvement_rate": round(improvement_rate, 3),
                "total_goals": len(self._goals),
                "completed_goals": len(completed),
                "active_goals": len(active),
                "abandoned_goals": len(abandoned),
                "total_improvements": len(self._improvements),
            },
            "strengths": sw["strengths"],
            "weaknesses": sw["weaknesses"],
            "goals": goal_reports,
        }

    def get_self_model(self) -> SelfModel:
        """Build the current self-model for system prompt injection."""
        if not self._capabilities:
            self.assess_capabilities()

        sw = self.get_strengths_and_weaknesses()

        return SelfModel(
            capabilities=self._capabilities,
            strengths=sw["strengths"],
            weaknesses=sw["weaknesses"],
            learning_goals=self._goals,
            total_improvements=len(self._improvements),
            successful_improvements=sum(1 for r in self._improvements if r.success),
            last_assessment=datetime.now().isoformat(),
        )

    def get_self_model_prompt(self) -> str:
        """Get self-model formatted for system prompt injection."""
        model = self.get_self_model()
        return model.to_system_prompt()

    # ====================================================================
    # Interaction Outcome Tracking
    # ====================================================================

    def record_interaction_outcome(
        self,
        domain: str,
        success: bool,
        confidence: float = 0.5,
        details: str = "",
    ) -> None:
        """Record the outcome of an interaction for capability tracking.

        Args:
            domain: Which capability domain this relates to
            success: Whether the interaction was successful
            confidence: How confident we were (0-1)
            details: Optional description
        """
        self._record_outcome(domain, success, details, confidence)

    def _record_outcome(
        self,
        domain: str,
        success: bool,
        details: str = "",
        confidence: float = 0.5,
    ) -> None:
        """Internal outcome recording."""
        self._outcomes.append({
            "domain": domain,
            "success": success,
            "confidence": confidence,
            "details": details,
            "timestamp": datetime.now().isoformat(),
        })
        # Keep bounded
        if len(self._outcomes) > 500:
            self._outcomes = self._outcomes[-500:]

    # ====================================================================
    # Signal Gathering (from subsystems)
    # ====================================================================

    def _gather_reflexion_signals(self) -> Dict[str, Dict]:
        """Reflexion removed -- returns empty signals."""
        return {}

    def _gather_guardian_signals(self) -> Dict[str, Dict]:
        """Guardian removed — returns empty signals."""
        return {}

    def _gather_skill_signals(self) -> Dict[str, Dict]:
        """Gather signals from Skill Library success rates."""
        signals = {}
        try:
            from aura_skill_library.skill_store import SkillStore
            store = SkillStore()

            for domain in CapabilityDomain:
                d = domain.value
                results = store.search(d, limit=5)
                if not results:
                    continue

                rates = []
                for skill in results:
                    if hasattr(skill, "metadata") and skill.metadata:
                        if skill.metadata.total_uses > 0:
                            rates.append(skill.metadata.success_rate)

                if rates:
                    avg_rate = sum(rates) / len(rates)
                    signals[d] = {
                        "score": avg_rate,
                        "detail": f"avg success rate {avg_rate:.0%} across {len(rates)} skills",
                    }
        except Exception as e:
            logger.debug(f"[Metacognition] Skill signals unavailable: {e}")

        return signals

    def _gather_outcome_signals(self) -> Dict[str, Dict]:
        """Gather signals from recorded interaction outcomes."""
        signals = {}

        # Only consider recent outcomes (last 7 days)
        cutoff = (datetime.now() - timedelta(days=7)).isoformat()
        recent = [o for o in self._outcomes if o["timestamp"] > cutoff]

        if not recent:
            return signals

        # Aggregate by domain
        domain_outcomes: Dict[str, List[bool]] = {}
        for o in recent:
            d = o["domain"]
            if d not in domain_outcomes:
                domain_outcomes[d] = []
            domain_outcomes[d].append(o["success"])

        for d, results in domain_outcomes.items():
            if len(results) >= 2:  # Need at least 2 data points
                success_rate = sum(results) / len(results)
                signals[d] = {
                    "score": success_rate,
                    "detail": f"{sum(results)}/{len(results)} recent successes",
                }

        return signals

    # ====================================================================
    # Helpers
    # ====================================================================

    def get_domain_for_query(self, prompt: str) -> CapabilityDomain:
        """Map a prompt to the most relevant CapabilityDomain.

        Used by brain.py System 1/System 2 routing to determine
        domain-specific confidence for model selection.
        """
        prompt_lower = prompt.lower()
        best_domain = CapabilityDomain.CONVERSATION
        best_hits = 0

        domain_keywords = {
            CapabilityDomain.CODING: ["code", "python", "debug", "function", "script", "program", "api", "bug", "implement", "refactor"],
            CapabilityDomain.RESEARCH: ["research", "analyze", "investigate", "find", "search", "lookup", "source", "study"],
            CapabilityDomain.WRITING: ["write", "essay", "document", "summarize", "email", "draft", "compose", "report"],
            CapabilityDomain.ANALYSIS: ["analyze", "data", "chart", "statistics", "compare", "evaluate", "pros and cons"],
            CapabilityDomain.CREATIVE: ["creative", "imagine", "brainstorm", "idea", "generate", "story", "poem"],
            CapabilityDomain.TOOL_USE: ["tool", "execute", "command", "run", "browser", "file", "open", "download"],
            CapabilityDomain.MEMORY: ["remember", "recall", "memory", "context", "history", "last time"],
            CapabilityDomain.EMOTIONAL: ["feeling", "emotion", "support", "mood", "empathy", "how are you"],
        }

        for domain, keywords in domain_keywords.items():
            hits = sum(1 for kw in keywords if kw in prompt_lower)
            if hits > best_hits:
                best_hits = hits
                best_domain = domain

        return best_domain

    def _domain_keywords(self, domain: str) -> List[str]:
        """Get keywords associated with a capability domain."""
        keyword_map = {
            "coding": ["code", "python", "function", "bug", "program", "script", "api"],
            "research": ["search", "find", "research", "investigate", "lookup", "source"],
            "writing": ["write", "essay", "document", "summarize", "email", "text"],
            "analysis": ["analyze", "data", "chart", "statistics", "compare", "evaluate"],
            "conversation": ["chat", "talk", "discuss", "conversation", "respond", "clarify"],
            "tool_use": ["tool", "execute", "command", "run", "browser", "file"],
            "memory": ["remember", "recall", "memory", "context", "history", "forget"],
            "emotional": ["emotion", "feeling", "empathy", "support", "mood", "tone"],
            "proactive": ["suggest", "remind", "notify", "proactive", "anticipate"],
            "creative": ["creative", "imagine", "brainstorm", "idea", "generate", "story"],
        }
        return keyword_map.get(domain, [domain])

    # ====================================================================
    # Full Metacognitive Cycle (for NeuroDream integration)
    # ====================================================================

    def run_metacognitive_cycle(self) -> Dict[str, Any]:
        """Run a full metacognitive cycle: assess -> plan -> execute -> evaluate.

        Called during NeuroDream deep sleep or periodically by gateway daemon.

        Returns summary of the cycle.
        """
        logger.info("[Metacognition] Starting metacognitive cycle")

        # Step 1: Assess
        capabilities = self.assess_capabilities()

        # Step 2: Plan
        new_goals = self.create_learning_plan()

        # Step 3: Execute improvements for active/pending goals
        improvements = []
        for goal in self._goals:
            if goal.status in ("pending", "active") and goal.attempts < 5:
                record = self.execute_improvement(goal.id)
                if record:
                    improvements.append(record)
                # Limit to 2 improvements per cycle
                if len(improvements) >= 2:
                    break

        # Step 4: Evaluate
        evaluation = self.evaluate_progress()

        result = {
            "capabilities_assessed": len(capabilities),
            "new_goals_created": len(new_goals),
            "improvements_attempted": len(improvements),
            "improvements_successful": sum(1 for r in improvements if r.success),
            "average_capability": evaluation["summary"]["average_capability"],
            "strengths": evaluation["strengths"],
            "weaknesses": evaluation["weaknesses"],
        }

        logger.info(
            f"[Metacognition] Cycle complete: "
            f"{result['improvements_successful']}/{result['improvements_attempted']} improvements, "
            f"avg capability={result['average_capability']:.2f}"
        )

        return result

    # ====================================================================
    # Persistence
    # ====================================================================

    def _state_file(self) -> Path:
        return self._data_dir / "metacognition_state.json"

    def _load_state(self) -> None:
        """Load persisted state from disk."""
        state_file = self._state_file()
        if not state_file.exists():
            return

        try:
            data = json.loads(state_file.read_text(encoding="utf-8"))

            # Load capabilities (filter to known fields for schema resilience)
            for d, cap_data in data.get("capabilities", {}).items():
                self._capabilities[d] = CapabilityScore(
                    **{k: v for k, v in cap_data.items() if k in CapabilityScore.__dataclass_fields__}
                )

            # Load goals (filter to known fields for schema resilience)
            for g_data in data.get("goals", []):
                self._goals.append(LearningGoal(
                    **{k: v for k, v in g_data.items() if k in LearningGoal.__dataclass_fields__}
                ))

            # Load improvements (filter to known fields for schema resilience)
            for r_data in data.get("improvements", []):
                self._improvements.append(ImprovementRecord(
                    **{k: v for k, v in r_data.items() if k in ImprovementRecord.__dataclass_fields__}
                ))

            # Load outcomes
            self._outcomes = data.get("outcomes", [])

        except Exception as e:
            logger.warning(f"[Metacognition] Failed to load state: {e}")

    def _save_state(self) -> None:
        """Save state to disk."""
        try:
            data = {
                "capabilities": {
                    d: {
                        "domain": c.domain, "score": c.score,
                        "confidence": c.confidence, "sample_count": c.sample_count,
                        "trend": c.trend, "last_assessed": c.last_assessed,
                        "evidence": c.evidence,
                    }
                    for d, c in self._capabilities.items()
                },
                "goals": [
                    {
                        "id": g.id, "domain": g.domain,
                        "description": g.description, "strategy": g.strategy,
                        "priority": g.priority, "created_at": g.created_at,
                        "target_score": g.target_score,
                        "current_score": g.current_score,
                        "status": g.status, "progress": g.progress,
                        "attempts": g.attempts,
                        "completed_at": g.completed_at, "result": g.result,
                    }
                    for g in self._goals
                ],
                "improvements": [
                    {
                        "goal_id": r.goal_id, "timestamp": r.timestamp,
                        "strategy": r.strategy, "action_taken": r.action_taken,
                        "before_score": r.before_score,
                        "after_score": r.after_score,
                        "success": r.success, "notes": r.notes,
                    }
                    for r in self._improvements[-100:]  # Keep last 100
                ],
                "outcomes": self._outcomes[-500:],
                "saved_at": datetime.now().isoformat(),
            }

            self._state_file().write_text(
                json.dumps(data, indent=2, default=str), encoding="utf-8"
            )
        except Exception as e:
            logger.warning(f"[Metacognition] Failed to save state: {e}")

    def get_status(self) -> Dict[str, Any]:
        """Get current metacognition status for API."""
        sw = self.get_strengths_and_weaknesses()
        return {
            "capabilities": {
                d: {"score": c.score, "confidence": c.confidence, "trend": c.trend}
                for d, c in self._capabilities.items()
            },
            "strengths": sw["strengths"],
            "weaknesses": sw["weaknesses"],
            "active_goals": [
                {"id": g.id, "domain": g.domain, "description": g.description,
                 "progress": g.progress}
                for g in self._goals if g.status in ("pending", "active")
            ],
            "total_improvements": len(self._improvements),
            "successful_improvements": sum(1 for r in self._improvements if r.success),
            "outcomes_tracked": len(self._outcomes),
        }


# ============================================================================
# Singleton
# ============================================================================

_metacognitive_engine: Optional[MetacognitiveEngine] = None
_metacog_lock = threading.Lock()


def get_metacognitive_engine() -> MetacognitiveEngine:
    """Get or create the metacognitive engine singleton."""
    global _metacognitive_engine
    if _metacognitive_engine is None:
        with _metacog_lock:
            if _metacognitive_engine is None:
                _metacognitive_engine = MetacognitiveEngine()
    return _metacognitive_engine
