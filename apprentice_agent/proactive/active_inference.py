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

    def restore_action_history(
        self, history: List[Tuple[str, datetime]]
    ) -> None:
        """Restore action history from persistence.

        Args:
            history: List of (action_value_string, taken_at) tuples.
        """
        self.action_history = []
        for action_str, taken_at in history:
            try:
                action = ProactiveAction(action_str)
            except (ValueError, KeyError):
                continue
            self.action_history.append((action, taken_at))
        if self.action_history:
            self.last_action_time = self.action_history[-1][1]

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
    Main Active Inference engine for proactive behavior (Phase 6A).

    Uses pymdp if available for full free energy minimization,
    otherwise falls back to simplified implementation.

    Hidden states:
      - User state: idle, shallow, focused (3 levels)
      - Task state: none, pending, urgent (3 levels)
      - Emotional state: negative, neutral, positive (3 levels)
      - Context state: stable, shifting, chaotic (3 levels)

    Observations:
      - User signals: inactivity, browsing, typing_fast
      - Task signals: no_tasks, some_tasks, urgent_tasks
      - Emotional signals: distressed, calm, happy
      - Context signals: same_app, app_switch, many_switches

    Actions: wait, notify, suggest, remind, ask, prepare, intervene (7)
    """

    # Map ProactiveAction to action index for pymdp
    ACTION_MAP = {
        ProactiveAction.WAIT: 0,
        ProactiveAction.NOTIFY: 1,
        ProactiveAction.SUGGEST: 2,
        ProactiveAction.REMIND: 3,
        ProactiveAction.ASK: 4,
        ProactiveAction.PREPARE: 5,
        ProactiveAction.INTERVENE: 6,
    }
    ACTION_REVERSE = {v: k for k, v in ACTION_MAP.items()}

    NUM_STATES = [3, 3, 3, 3]   # user, task, emotional, context
    NUM_OBS = [3, 3, 3, 3]      # user_sig, task_sig, emo_sig, context_sig
    NUM_ACTIONS = 7

    def __init__(self, use_pymdp: bool = True):
        """
        Initialize the engine.

        Args:
            use_pymdp: Whether to use pymdp (if available)
        """
        # Always keep simplified engine for belief tracking
        self._simple_engine = SimplifiedActiveInference()
        self.use_pymdp = use_pymdp and PYMDP_AVAILABLE

        if self.use_pymdp:
            try:
                self._init_pymdp()
                logger.info("[ActiveInference] pymdp agent initialized")
            except Exception as e:
                logger.warning(f"[ActiveInference] pymdp init failed, falling back: {e}")
                self.use_pymdp = False

        logger.info(f"[ActiveInference] Engine initialized (pymdp={self.use_pymdp})")

    @staticmethod
    def _make_factor_A(likelihood_2d, target_factor, num_obs_m, num_states):
        """Build multi-factor A matrix where one modality depends on a single factor.

        Args:
            likelihood_2d: Shape (num_obs_m, num_states[target_factor])
            target_factor: Which hidden factor this modality depends on
            num_obs_m: Number of observation levels for this modality
            num_states: List of state dims per factor

        Returns:
            A_m with shape (num_obs_m, *num_states), normalized over obs axis.
        """
        full_shape = [num_obs_m] + list(num_states)
        A_m = np.zeros(full_shape)
        for obs_idx in range(num_obs_m):
            for state_idx in range(num_states[target_factor]):
                slices = [obs_idx] + [slice(None)] * len(num_states)
                slices[target_factor + 1] = state_idx  # +1 for obs dimension
                A_m[tuple(slices)] = likelihood_2d[obs_idx, state_idx]
        return A_m

    def _init_pymdp(self):
        """Initialize pymdp-based agent with AURA's generative model (Phase 6A).

        Architecture: 4 hidden state factors, 4 observation modalities, 7 actions.
        Only factor 0 (user state) is directly controllable by the agent's 7 actions.
        Factors 1-3 (task, emotional, context) have autonomous dynamics.

        Each observation modality depends on exactly one hidden factor (conditional
        independence), expressed via properly-shaped multi-factor A tensors.
        """
        n_states = self.NUM_STATES
        n_obs = self.NUM_OBS
        n_actions = self.NUM_ACTIONS
        n_factors = len(n_states)
        n_modalities = len(n_obs)

        # === A matrix: P(observation | state) ===
        # Each A[m] has shape (num_obs[m], *num_states) = (3, 3, 3, 3, 3)
        # Each modality depends on one factor; uniform over others.
        A = pymdp_utils.obj_array(n_modalities)

        # Per-factor likelihood matrices (2D: obs × states for that factor)
        likelihoods = [
            # User: idle->inactivity, shallow->browsing, focused->typing
            np.array([
                [0.8, 0.1, 0.1],
                [0.15, 0.7, 0.2],
                [0.05, 0.2, 0.7],
            ]),
            # Task: none->no_tasks, pending->some_tasks, urgent->urgent_tasks
            np.array([
                [0.8, 0.15, 0.05],
                [0.15, 0.7, 0.2],
                [0.05, 0.15, 0.75],
            ]),
            # Emotional: negative->distressed, neutral->calm, positive->happy
            np.array([
                [0.75, 0.15, 0.05],
                [0.2, 0.7, 0.2],
                [0.05, 0.15, 0.75],
            ]),
            # Context: stable->same_app, shifting->app_switch, chaotic->many_switches
            np.array([
                [0.8, 0.15, 0.05],
                [0.15, 0.7, 0.2],
                [0.05, 0.15, 0.75],
            ]),
        ]
        for m in range(n_modalities):
            A[m] = self._make_factor_A(likelihoods[m], m, n_obs[m], n_states)

        # === B matrix: P(state' | state, action) - Transition model ===
        # Factor 0 (user): controllable, 7 actions
        # Factors 1-3: uncontrollable, 1 action (natural dynamics)
        B = pymdp_utils.obj_array(n_factors)

        # Factor 0: User state transitions under 7 actions
        B[0] = np.zeros((n_states[0], n_states[0], n_actions))
        for a in range(n_actions):
            B[0][:, :, a] = np.eye(n_states[0]) * 0.7 + 0.1
        # SUGGEST (2) or ASK (4) can move user from idle to engaged
        for action_idx in [2, 4]:
            B[0][1, 0, action_idx] = 0.4  # idle -> shallow
            B[0][0, 0, action_idx] = 0.5  # idle stays idle
            B[0][2, 0, action_idx] = 0.1  # idle -> focused (unlikely)
        # NOTIFY (1) mildly engages idle user
        B[0][1, 0, 1] = 0.3
        B[0][0, 0, 1] = 0.6
        # INTERVENE (6) strongly engages
        B[0][2, 0, 6] = 0.3
        B[0][1, 0, 6] = 0.3
        B[0][0, 0, 6] = 0.3

        # Factors 1-3: Uncontrollable (1 "null" action - natural dynamics)
        for f in range(1, n_factors):
            B[f] = np.zeros((n_states[f], n_states[f], 1))
            B[f][:, :, 0] = np.eye(n_states[f]) * 0.7 + 0.1

        # Normalize B matrices
        for f in range(n_factors):
            n_ctrl = B[f].shape[2]
            for a in range(n_ctrl):
                col_sums = B[f][:, :, a].sum(axis=0, keepdims=True)
                col_sums[col_sums == 0] = 1
                B[f][:, :, a] /= col_sums

        # === C vector: Preferred observations (log preferences) ===
        C = pymdp_utils.obj_array(n_modalities)
        C[0] = np.array([-1.0, 0.5, 1.0])   # Prefer user engaged
        C[1] = np.array([1.0, 0.0, -2.0])    # Prefer tasks resolved
        C[2] = np.array([-1.0, 0.5, 1.5])    # Prefer positive emotion
        C[3] = np.array([1.0, 0.0, -0.5])    # Prefer stable context

        # === D vector: Prior beliefs about initial states ===
        D = pymdp_utils.obj_array(n_factors)
        D[0] = np.array([0.3, 0.5, 0.2])    # Probably shallow work
        D[1] = np.array([0.5, 0.35, 0.15])   # Probably no tasks
        D[2] = np.array([0.1, 0.7, 0.2])     # Probably neutral
        D[3] = np.array([0.5, 0.35, 0.15])   # Probably stable

        # Create pymdp agent - only factor 0 is controllable
        self._pymdp_agent = PyMDPAgent(
            A=A, B=B, C=C, D=D,
            control_fac_idx=[0],
            policy_len=1,
        )

        # Track last observations for pymdp
        self._last_obs = None

    def _discretize_observations(self, observations: Dict[str, float]) -> List[int]:
        """Convert continuous observations to discrete indices for pymdp.

        Maps float observations (0-1) to 3 discrete levels: 0=low, 1=medium, 2=high.
        """
        def to_level(val: float) -> int:
            if val < 0.33:
                return 0
            elif val < 0.67:
                return 1
            return 2

        # User signal: from activity level
        user_activity = observations.get("user_activity", 0.5)
        user_obs = to_level(user_activity)

        # Task signal: from urgency
        task_urgency = observations.get("urgent_events", 0.0)
        task_obs = to_level(task_urgency)

        # Emotional signal: from emotional state (if available)
        emotional = observations.get("emotional_valence", 0.5)
        emo_obs = to_level(emotional)

        # Context signal: from context changes
        context_change = observations.get("context_changes", 0.0)
        ctx_obs = to_level(context_change)

        return [user_obs, task_obs, emo_obs, ctx_obs]

    def update_beliefs(self, observations: Dict[str, float]) -> BeliefState:
        """Update beliefs from observations."""
        # Always update simplified engine (for backward compat)
        simple_beliefs = self._simple_engine.update_beliefs(observations)

        if self.use_pymdp:
            try:
                obs = self._discretize_observations(observations)
                self._last_obs = obs
                # pymdp infer_states
                self._pymdp_agent.infer_states(obs)
            except Exception as e:
                logger.debug(f"[ActiveInference] pymdp belief update error: {e}")

        return simple_beliefs

    def select_action(self) -> ProactiveDecision:
        """Select best proactive action."""
        if self.use_pymdp and self._last_obs is not None:
            try:
                # pymdp infer_policies and sample_action
                q_pi, efe = self._pymdp_agent.infer_policies()
                action_idx = self._pymdp_agent.sample_action()

                # Map to ProactiveAction
                # action_idx may be an array of per-factor actions; take first
                if hasattr(action_idx, '__len__'):
                    idx = int(action_idx[0])
                else:
                    idx = int(action_idx)

                idx = idx % self.NUM_ACTIONS  # Safety clamp
                action = self.ACTION_REVERSE.get(idx, ProactiveAction.WAIT)

                # Confidence from policy posterior
                confidence = float(q_pi.max()) if q_pi is not None else 0.5

                # EFE of selected policy
                best_efe = float(efe.min()) if efe is not None else 0.0

                # Also get simplified reasoning for the chosen action
                _, reasoning = self._simple_engine.compute_expected_free_energy(action)

                # Check cooldowns using simplified engine
                if not self._simple_engine._can_take_action(action):
                    action = ProactiveAction.WAIT
                    reasoning = "Selected action on cooldown"

                # Record in simplified engine for cooldown tracking
                if action != ProactiveAction.WAIT:
                    self._simple_engine.action_history.append((action, datetime.now()))
                    self._simple_engine.last_action_time = datetime.now()

                return ProactiveDecision(
                    action=action,
                    confidence=confidence,
                    expected_free_energy=best_efe,
                    reasoning=f"[pymdp] {reasoning}",
                    metadata={
                        "beliefs": self._simple_engine.beliefs.__dict__,
                        "pymdp_action_idx": idx,
                        "policy_posterior_max": float(q_pi.max()) if q_pi is not None else None,
                    }
                )
            except Exception as e:
                logger.debug(f"[ActiveInference] pymdp action selection error: {e}, falling back")

        return self._simple_engine.select_action()

    def should_act_proactively(self) -> Tuple[bool, str]:
        """Determine if proactive action is warranted."""
        return self._simple_engine.should_act_proactively()

    def get_beliefs(self) -> BeliefState:
        """Get current belief state."""
        return self._simple_engine.beliefs

    def restore_beliefs(self, beliefs: BeliefState) -> None:
        """Restore beliefs from persistence."""
        self._simple_engine.beliefs = beliefs

    def restore_action_history(
        self, history: List[Tuple[str, datetime]]
    ) -> None:
        """Restore action history from persistence.

        Args:
            history: List of (action_value_string, taken_at) tuples.
        """
        self._simple_engine.restore_action_history(history)

    def set_intrinsic_preferences(self, preferences: Dict[str, float]) -> None:
        """Update intrinsic motivation priors in the generative model (Phase 6E).

        Args:
            preferences: Dict mapping preference names to values (-2 to 2).
                E.g. {"curiosity": 1.5, "social": 0.8}
        """
        if not self.use_pymdp:
            # Update simplified engine preferences
            self._simple_engine.preferences.update(preferences)
            return

        try:
            # Modify C vector based on intrinsic drives
            if "curiosity" in preferences:
                # Curiosity: prefer context shifts (exploration)
                self._pymdp_agent.C[3][2] += preferences["curiosity"] * 0.3

            if "social" in preferences:
                # Social drive: prefer user engagement
                self._pymdp_agent.C[0][1] += preferences["social"] * 0.2
                self._pymdp_agent.C[0][2] += preferences["social"] * 0.3

            if "competence" in preferences:
                # Competence: prefer task resolution
                self._pymdp_agent.C[1][0] += preferences["competence"] * 0.3

            if "coherence" in preferences:
                # Coherence: prefer stable context
                self._pymdp_agent.C[3][0] += preferences["coherence"] * 0.2

        except Exception as e:
            logger.debug(f"[ActiveInference] Failed to set preferences: {e}")


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
