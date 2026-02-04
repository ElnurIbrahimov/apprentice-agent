"""
Active Inference Engine - Proactive decision making using Free Energy Principle.

Based on Karl Friston's Active Inference framework:
- Agents minimize "surprisal" (prediction error) through action and perception
- Balances exploitation (achieving goals) with exploration (reducing uncertainty)
- Naturally emergent proactive behavior from minimizing expected free energy

Uses pymdp when available, falls back to simplified implementation.
"""

import logging
import math
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum

logger = logging.getLogger(__name__)

# Try to import pymdp for full Active Inference
try:
    from pymdp.agent import Agent as PyMDPAgent
    from pymdp import utils as pymdp_utils
    PYMDP_AVAILABLE = True
    logger.info("[ActiveInference] pymdp available - using full implementation")
except ImportError:
    PYMDP_AVAILABLE = False
    logger.info("[ActiveInference] pymdp not available - using simplified implementation")


class ProactiveAction(Enum):
    """Available proactive actions."""
    WAIT = "wait"                      # Do nothing, continue observing
    NOTIFY = "notify"                  # Send notification to user
    SUGGEST = "suggest"                # Make a suggestion
    REMIND = "remind"                  # Send a reminder
    ASK = "ask"                        # Ask user a question
    PREPARE = "prepare"                # Prepare something in background
    INTERVENE = "intervene"            # Actively intervene/help


@dataclass
class BeliefState:
    """Current beliefs about the world state."""
    user_busy: float = 0.5           # Belief that user is busy (0-1)
    user_receptive: float = 0.5      # Belief that user wants interaction (0-1)
    task_urgent: float = 0.0         # Belief there's urgent task (0-1)
    context_stable: float = 0.5      # Belief context is stable (0-1)
    uncertainty: float = 0.5         # Overall uncertainty level (0-1)

    def to_array(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.array([
            self.user_busy,
            self.user_receptive,
            self.task_urgent,
            self.context_stable,
            self.uncertainty
        ])

    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'BeliefState':
        """Create from numpy array."""
        return cls(
            user_busy=float(arr[0]),
            user_receptive=float(arr[1]),
            task_urgent=float(arr[2]),
            context_stable=float(arr[3]),
            uncertainty=float(arr[4])
        )


@dataclass
class ProactiveDecision:
    """Result of active inference decision."""
    action: ProactiveAction
    confidence: float                 # Confidence in decision (0-1)
    expected_free_energy: float       # Expected free energy (lower = better)
    reasoning: str                    # Human-readable explanation
    metadata: Dict[str, Any] = field(default_factory=dict)


class SimplifiedActiveInference:
    """
    Simplified Active Inference implementation.

    Uses heuristic rules inspired by Free Energy Principle:
    - Minimize surprise by acting to confirm predictions
    - Balance pragmatic (goal) and epistemic (curiosity) value
    """

    def __init__(self):
        self.beliefs = BeliefState()
        self.action_history: List[Tuple[ProactiveAction, datetime]] = []
        self.last_action_time: Optional[datetime] = None

        # Preferences (what we "want" to observe)
        self.preferences = {
            "user_engaged": 0.7,      # Prefer user being engaged
            "low_uncertainty": 0.8,   # Prefer low uncertainty
            "tasks_handled": 0.9,     # Prefer tasks being handled
        }

        # Action cooldowns (seconds) to prevent spam
        self.cooldowns = {
            ProactiveAction.WAIT: 0,
            ProactiveAction.NOTIFY: 60,
            ProactiveAction.SUGGEST: 120,
            ProactiveAction.REMIND: 180,
            ProactiveAction.ASK: 300,
            ProactiveAction.PREPARE: 30,
            ProactiveAction.INTERVENE: 600,
        }

        logger.info("[ActiveInference] Simplified engine initialized")

    def update_beliefs(
        self,
        observations: Dict[str, float]
    ) -> BeliefState:
        """
        Update beliefs based on observations.

        Uses Bayesian-like blending of prior beliefs with new observations.

        Args:
            observations: Dict of observation_name -> value (0-1)
        """
        learning_rate = 0.3  # How much to weight new observations

        # Map observations to belief dimensions
        if "user_activity" in observations:
            activity = observations["user_activity"]
            self.beliefs.user_busy = self._blend(
                self.beliefs.user_busy, activity, learning_rate
            )

        if "interaction_recency" in observations:
            recency = observations["interaction_recency"]
            # Recent interaction = more receptive
            self.beliefs.user_receptive = self._blend(
                self.beliefs.user_receptive, recency, learning_rate
            )

        if "urgent_events" in observations:
            urgency = observations["urgent_events"]
            self.beliefs.task_urgent = self._blend(
                self.beliefs.task_urgent, urgency, learning_rate
            )

        if "context_changes" in observations:
            stability = 1.0 - observations["context_changes"]
            self.beliefs.context_stable = self._blend(
                self.beliefs.context_stable, stability, learning_rate
            )

        # Update uncertainty based on observation confidence
        if "observation_confidence" in observations:
            conf = observations["observation_confidence"]
            self.beliefs.uncertainty = self._blend(
                self.beliefs.uncertainty, 1.0 - conf, learning_rate
            )

        return self.beliefs

    def _blend(self, prior: float, observation: float, rate: float) -> float:
        """Blend prior belief with new observation."""
        return np.clip(prior * (1 - rate) + observation * rate, 0.0, 1.0)

    def compute_expected_free_energy(
        self,
        action: ProactiveAction
    ) -> Tuple[float, str]:
        """
        Compute expected free energy for an action.

        G = pragmatic_value + epistemic_value

        Pragmatic: How well does action achieve preferences?
        Epistemic: How much does action reduce uncertainty?

        Lower G = better action.

        Returns:
            (expected_free_energy, reasoning)
        """
        # Pragmatic component: deviation from preferences
        if action == ProactiveAction.WAIT:
            # Waiting is neutral - doesn't push toward preferences
            pragmatic = 0.5
            reasoning = "Waiting maintains status quo"

        elif action == ProactiveAction.NOTIFY:
            # Good if task urgent, bad if user busy
            if self.beliefs.task_urgent > 0.7:
                pragmatic = 0.2  # Low = good
                reasoning = "Urgent task needs attention"
            elif self.beliefs.user_busy > 0.7:
                pragmatic = 0.8  # High = bad
                reasoning = "User appears busy"
            else:
                pragmatic = 0.4
                reasoning = "Notification may be helpful"

        elif action == ProactiveAction.SUGGEST:
            # Good if user receptive and not too busy
            if self.beliefs.user_receptive > 0.6 and self.beliefs.user_busy < 0.5:
                pragmatic = 0.3
                reasoning = "User seems receptive to suggestions"
            else:
                pragmatic = 0.6
                reasoning = "Suggestion may not be welcome now"

        elif action == ProactiveAction.REMIND:
            # Good if task exists and context is stable
            if self.beliefs.task_urgent > 0.3 and self.beliefs.context_stable > 0.5:
                pragmatic = 0.35
                reasoning = "Reminder timing seems appropriate"
            else:
                pragmatic = 0.65
                reasoning = "Context may not be right for reminder"

        elif action == ProactiveAction.ASK:
            # Good when uncertainty is high
            if self.beliefs.uncertainty > 0.6:
                pragmatic = 0.25
                reasoning = "Asking would reduce uncertainty"
            else:
                pragmatic = 0.7
                reasoning = "Already have sufficient information"

        elif action == ProactiveAction.PREPARE:
            # Generally safe, good with anticipated needs
            pragmatic = 0.4
            reasoning = "Background preparation is low-risk"

        elif action == ProactiveAction.INTERVENE:
            # Only good in urgent situations
            if self.beliefs.task_urgent > 0.8:
                pragmatic = 0.2
                reasoning = "Urgent situation requires intervention"
            else:
                pragmatic = 0.9
                reasoning = "Intervention not warranted"
        else:
            pragmatic = 0.5
            reasoning = "Unknown action"

        # Epistemic component: information gain
        # Actions that interact with user reduce uncertainty
        epistemic_gain = {
            ProactiveAction.WAIT: 0.0,
            ProactiveAction.NOTIFY: 0.1,
            ProactiveAction.SUGGEST: 0.2,
            ProactiveAction.REMIND: 0.1,
            ProactiveAction.ASK: 0.4,      # Asking gains most info
            ProactiveAction.PREPARE: 0.05,
            ProactiveAction.INTERVENE: 0.3,
        }

        epistemic = -epistemic_gain.get(action, 0.0) * self.beliefs.uncertainty

        # Total expected free energy
        G = pragmatic + epistemic

        return G, reasoning

    def _can_take_action(self, action: ProactiveAction) -> bool:
        """Check if action is off cooldown."""
        if self.last_action_time is None:
            return True

        elapsed = (datetime.now() - self.last_action_time).total_seconds()
        cooldown = self.cooldowns.get(action, 0)
        return elapsed >= cooldown

    def select_action(self) -> ProactiveDecision:
        """
        Select best action using Active Inference.

        Computes expected free energy for each action and selects
        the one with lowest G (best expected outcome).

        Returns:
            ProactiveDecision with selected action and reasoning
        """
        # Compute G for each action
        action_values: List[Tuple[ProactiveAction, float, str]] = []

        for action in ProactiveAction:
            if not self._can_take_action(action):
                continue
            G, reasoning = self.compute_expected_free_energy(action)
            action_values.append((action, G, reasoning))

        if not action_values:
            # All actions on cooldown, default to wait
            return ProactiveDecision(
                action=ProactiveAction.WAIT,
                confidence=0.5,
                expected_free_energy=0.5,
                reasoning="All actions on cooldown"
            )

        # Select action with lowest G
        action_values.sort(key=lambda x: x[1])
        best_action, best_G, reasoning = action_values[0]

        # Compute confidence (inverse of G, normalized)
        confidence = 1.0 - best_G

        # Record action
        if best_action != ProactiveAction.WAIT:
            self.action_history.append((best_action, datetime.now()))
            self.last_action_time = datetime.now()
            # Trim history
            self.action_history = self.action_history[-100:]

        return ProactiveDecision(
            action=best_action,
            confidence=confidence,
            expected_free_energy=best_G,
            reasoning=reasoning,
            metadata={
                "beliefs": self.beliefs.__dict__,
                "alternatives": [(a.value, round(g, 3)) for a, g, _ in action_values[:3]]
            }
        )

    def should_act_proactively(self) -> Tuple[bool, str]:
        """
        Determine if proactive action is warranted.

        Returns:
            (should_act, reason)
        """
        decision = self.select_action()

        # Act if selected action is not WAIT and confidence is reasonable
        should_act = (
            decision.action != ProactiveAction.WAIT
            and decision.confidence > 0.4
        )

        return should_act, decision.reasoning


class ActiveInferenceEngine:
    """
    Main Active Inference engine for proactive behavior.

    Uses pymdp if available, otherwise falls back to simplified implementation.
    """

    def __init__(self, use_pymdp: bool = True):
        """
        Initialize the engine.

        Args:
            use_pymdp: Whether to use pymdp (if available)
        """
        self.use_pymdp = use_pymdp and PYMDP_AVAILABLE

        if self.use_pymdp:
            self._init_pymdp()
        else:
            self._simple_engine = SimplifiedActiveInference()

        logger.info(f"[ActiveInference] Engine initialized (pymdp={self.use_pymdp})")

    def _init_pymdp(self):
        """Initialize pymdp-based agent."""
        # State space: [user_state, task_state, context_state]
        # Each has 3 levels: low, medium, high
        num_states = [3, 3, 3]

        # Observation space: [user_signals, task_signals, context_signals]
        num_obs = [3, 3, 3]

        # Action space: wait, notify, suggest, remind, ask
        num_actions = 5

        # A matrix: P(observation | state)
        A = pymdp_utils.obj_array(len(num_obs))
        for i in range(len(num_obs)):
            # Noisy identity - state roughly corresponds to observation
            A[i] = np.eye(num_states[i]) * 0.8 + 0.2 / num_states[i]

        # B matrix: P(state' | state, action)
        B = pymdp_utils.obj_array(len(num_states))
        for i in range(len(num_states)):
            B[i] = np.zeros((num_states[i], num_states[i], num_actions))
            for a in range(num_actions):
                # Default: state tends to persist
                B[i][:, :, a] = np.eye(num_states[i]) * 0.7 + 0.3 / num_states[i]

        # C vector: Preferred observations
        C = pymdp_utils.obj_array(len(num_obs))
        C[0] = np.array([0.2, 0.5, 0.3])  # Prefer medium user engagement
        C[1] = np.array([0.6, 0.3, 0.1])  # Prefer low task urgency (handled)
        C[2] = np.array([0.2, 0.5, 0.3])  # Prefer medium context stability

        # D vector: Initial state beliefs
        D = pymdp_utils.obj_array(len(num_states))
        for i in range(len(num_states)):
            D[i] = np.ones(num_states[i]) / num_states[i]

        self._pymdp_agent = PyMDPAgent(A=A, B=B, C=C, D=D)

    def update_beliefs(self, observations: Dict[str, float]) -> BeliefState:
        """Update beliefs from observations."""
        if self.use_pymdp:
            # Convert observations to discrete format for pymdp
            # TODO: Implement full pymdp integration
            pass

        return self._simple_engine.update_beliefs(observations)

    def select_action(self) -> ProactiveDecision:
        """Select best proactive action."""
        if self.use_pymdp:
            # TODO: Implement full pymdp action selection
            pass

        return self._simple_engine.select_action()

    def should_act_proactively(self) -> Tuple[bool, str]:
        """Determine if proactive action is warranted."""
        return self._simple_engine.should_act_proactively()

    def get_beliefs(self) -> BeliefState:
        """Get current belief state."""
        return self._simple_engine.beliefs


if __name__ == "__main__":
    print("=" * 60)
    print("Active Inference Engine Test")
    print("=" * 60)

    engine = ActiveInferenceEngine(use_pymdp=False)

    # Simulate different scenarios
    scenarios = [
        {
            "name": "User busy, no urgent tasks",
            "observations": {
                "user_activity": 0.9,
                "interaction_recency": 0.2,
                "urgent_events": 0.1,
                "context_changes": 0.1,
            }
        },
        {
            "name": "User idle, urgent task pending",
            "observations": {
                "user_activity": 0.2,
                "interaction_recency": 0.3,
                "urgent_events": 0.9,
                "context_changes": 0.2,
            }
        },
        {
            "name": "High uncertainty",
            "observations": {
                "user_activity": 0.5,
                "interaction_recency": 0.5,
                "urgent_events": 0.5,
                "context_changes": 0.8,
                "observation_confidence": 0.2,
            }
        },
    ]

    for scenario in scenarios:
        print(f"\n--- {scenario['name']} ---")

        # Update beliefs
        beliefs = engine.update_beliefs(scenario["observations"])
        print(f"Beliefs: busy={beliefs.user_busy:.2f}, receptive={beliefs.user_receptive:.2f}, "
              f"urgent={beliefs.task_urgent:.2f}")

        # Select action
        decision = engine.select_action()
        print(f"Action: {decision.action.value}")
        print(f"Confidence: {decision.confidence:.2f}")
        print(f"Reasoning: {decision.reasoning}")

        # Should act?
        should_act, reason = engine.should_act_proactively()
        print(f"Should act: {should_act} - {reason}")

    print("\n" + "=" * 60)
    print("Test complete!")
