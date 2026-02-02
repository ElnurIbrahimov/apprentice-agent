"""
AURA Predictive Life Modeling - Causal Analysis

Uses DoWhy for causal inference on life decisions.
"""
import logging
from typing import Dict, List, Optional, Any, Tuple

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None

try:
    from dowhy import CausalModel
    DOWHY_AVAILABLE = True
except ImportError:
    DOWHY_AVAILABLE = False
    CausalModel = None

from .life_state import LifeState, LifeDomain
from .scenario import Scenario, DecisionType

logger = logging.getLogger(__name__)


class CausalLifeAnalyzer:
    """
    Analyzes causal relationships between life decisions and outcomes.
    Uses DoWhy for formal causal inference when available.
    """

    # Causal graph for life domains (simplified)
    LIFE_CAUSAL_GRAPH = """
    digraph {
        income -> savings;
        income -> lifestyle;
        savings -> financial_security;
        savings -> stress;
        job_satisfaction -> stress;
        job_satisfaction -> life_satisfaction;
        stress -> health;
        stress -> relationship_quality;
        health -> energy;
        health -> life_satisfaction;
        energy -> productivity;
        productivity -> income;
        relationship_quality -> life_satisfaction;
        financial_security -> stress;
        financial_security -> life_satisfaction;
        education -> income;
        education -> job_satisfaction;
        career_change -> income;
        career_change -> job_satisfaction;
        career_change -> stress;
        relocation -> cost_of_living;
        cost_of_living -> savings;
        relocation -> relationship_quality;
        children -> expenses;
        children -> life_satisfaction;
        children -> stress;
        expenses -> savings;
    }
    """

    def __init__(self, historical_data: Optional[Any] = None):
        """
        Initialize causal analyzer.

        Args:
            historical_data: DataFrame with columns matching life domains
                           for fitting causal models
        """
        self.historical_data = historical_data
        self.causal_model = None

        if not DOWHY_AVAILABLE:
            logger.info("DoWhy not available. Causal analysis will use heuristics.")

    def analyze_decision_effect(
        self,
        decision: str,
        outcome: str,
        current_state: LifeState,
        scenario: Optional[Scenario] = None
    ) -> Dict[str, Any]:
        """
        Analyze the causal effect of a decision on an outcome.

        Args:
            decision: The treatment variable (e.g., "career_change")
            outcome: The outcome variable (e.g., "life_satisfaction")
            current_state: Current life state for context
            scenario: Optional scenario for more specific analysis

        Returns:
            Dictionary with causal effect estimates
        """
        if not DOWHY_AVAILABLE or self.historical_data is None:
            return self._heuristic_analysis(decision, outcome, current_state, scenario)

        try:
            # Build causal model
            model = CausalModel(
                data=self.historical_data,
                treatment=decision,
                outcome=outcome,
                graph=self.LIFE_CAUSAL_GRAPH
            )

            # Identify causal effect
            identified_estimand = model.identify_effect()

            # Estimate effect
            estimate = model.estimate_effect(
                identified_estimand,
                method_name="backdoor.linear_regression"
            )

            # Refutation tests
            refutation = model.refute_estimate(
                identified_estimand,
                estimate,
                method_name="random_common_cause"
            )

            return {
                "decision": decision,
                "outcome": outcome,
                "causal_effect": estimate.value,
                "confidence_interval": estimate.get_confidence_intervals() if hasattr(estimate, 'get_confidence_intervals') else None,
                "refutation_result": refutation.refutation_result if refutation else None,
                "method": "dowhy_backdoor",
                "identified_estimand": str(identified_estimand)
            }

        except Exception as e:
            logger.warning(f"DoWhy analysis failed: {e}, falling back to heuristics")
            return self._heuristic_analysis(decision, outcome, current_state, scenario)

    def _heuristic_analysis(
        self,
        decision: str,
        outcome: str,
        current_state: LifeState,
        scenario: Optional[Scenario]
    ) -> Dict[str, Any]:
        """
        Heuristic causal analysis when DoWhy is not available.
        Based on common-sense causal relationships.
        """
        # Define expected effect directions and magnitudes
        effect_matrix = {
            ("career_change", "income"): (0.1, 0.3, "positive"),
            ("career_change", "stress"): (0.05, 0.2, "increase_then_decrease"),
            ("career_change", "life_satisfaction"): (0.05, 0.25, "variable"),
            ("quit_job", "income"): (-1.0, -0.5, "negative"),
            ("quit_job", "stress"): (-0.2, 0.3, "variable"),
            ("start_business", "income"): (-0.5, 2.0, "high_variance"),
            ("start_business", "stress"): (0.1, 0.4, "positive"),
            ("start_business", "life_satisfaction"): (-0.1, 0.4, "high_variance"),
            ("education", "income"): (0.1, 0.4, "delayed_positive"),
            ("education", "skills"): (0.2, 0.5, "positive"),
            ("relocation", "cost_of_living"): (-0.3, 0.3, "variable"),
            ("children", "expenses"): (0.3, 0.6, "positive"),
            ("children", "life_satisfaction"): (-0.1, 0.4, "variable"),
        }

        key = (decision, outcome)
        if key in effect_matrix:
            min_effect, max_effect, pattern = effect_matrix[key]
            mean_effect = (min_effect + max_effect) / 2
        else:
            # Unknown relationship
            min_effect, max_effect = -0.1, 0.1
            mean_effect = 0.0
            pattern = "unknown"

        return {
            "decision": decision,
            "outcome": outcome,
            "causal_effect": mean_effect,
            "effect_range": (min_effect, max_effect),
            "pattern": pattern,
            "method": "heuristic",
            "confidence": "low" if pattern == "unknown" else "medium",
            "note": "Based on common causal patterns. Collect data for more precise estimates."
        }

    def identify_confounders(
        self,
        decision: str,
        outcome: str
    ) -> List[str]:
        """Identify potential confounding variables."""
        # Based on causal graph structure
        confounders_map = {
            ("career_change", "income"): ["education", "experience", "industry", "location"],
            ("career_change", "stress"): ["health", "financial_security", "personality"],
            ("education", "income"): ["initial_ability", "field", "connections"],
            ("relocation", "life_satisfaction"): ["relationship_status", "career", "social_ties"],
        }

        return confounders_map.get((decision, outcome), ["unidentified"])

    def what_if_analysis(
        self,
        current_state: LifeState,
        intervention: Dict[str, Any],
        outcomes_of_interest: List[str]
    ) -> Dict[str, Any]:
        """
        Perform what-if analysis: "What if I change X to Y?"

        Args:
            current_state: Current life state
            intervention: Dict mapping variable names to new values
            outcomes_of_interest: List of outcome variables to estimate

        Returns:
            Predicted outcomes under intervention
        """
        results = {}

        for outcome in outcomes_of_interest:
            for var, new_value in intervention.items():
                effect = self._estimate_intervention_effect(
                    current_state,
                    var,
                    new_value,
                    outcome
                )

                if outcome not in results:
                    results[outcome] = {"baseline": self._get_current_value(current_state, outcome)}

                results[outcome][f"effect_of_{var}"] = effect

        # Compute combined effect (simplified additive)
        for outcome in results:
            total_effect = sum(
                v for k, v in results[outcome].items()
                if k.startswith("effect_of_")
            )
            baseline = results[outcome].get("baseline", 0) or 0
            results[outcome]["predicted_value"] = baseline + total_effect

        return results

    def _estimate_intervention_effect(
        self,
        state: LifeState,
        variable: str,
        new_value: Any,
        outcome: str
    ) -> float:
        """Estimate effect of setting a variable to a new value."""
        current_value = self._get_current_value(state, variable)

        if current_value is None or new_value is None:
            return 0.0

        # Get causal effect estimate
        analysis = self._heuristic_analysis(variable, outcome, state, None)
        effect_per_unit = analysis["causal_effect"]

        # Calculate change
        if isinstance(current_value, (int, float)) and isinstance(new_value, (int, float)):
            change = new_value - current_value
            return effect_per_unit * change

        return effect_per_unit  # Binary/categorical change

    def _get_current_value(self, state: LifeState, variable: str) -> Optional[float]:
        """Get current value of a variable from life state."""
        mapping = {
            "income": state.financial.monthly_income,
            "savings": state.financial.savings,
            "expenses": state.financial.monthly_expenses,
            "stress": state.health.stress_level,
            "health": state.health.physical_health,
            "life_satisfaction": state.personal.life_satisfaction,
            "job_satisfaction": state.career.satisfaction_score,
            "relationship_quality": state.relationships.social_satisfaction,
        }
        return mapping.get(variable)

    def get_causal_graph_summary(self) -> Dict[str, Any]:
        """Get a summary of the causal graph structure."""
        return {
            "domains": ["financial", "career", "health", "relationships", "personal"],
            "key_relationships": [
                {"from": "income", "to": "savings", "type": "direct"},
                {"from": "stress", "to": "health", "type": "negative"},
                {"from": "job_satisfaction", "to": "life_satisfaction", "type": "positive"},
                {"from": "financial_security", "to": "stress", "type": "negative"},
                {"from": "education", "to": "income", "type": "delayed_positive"},
            ],
            "intervention_points": [
                "career_change", "education", "relocation",
                "lifestyle_change", "financial_decision"
            ]
        }
