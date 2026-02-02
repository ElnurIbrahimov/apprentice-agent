"""
AURA Predictive Life Modeling - Report Generator

Creates structured decision analysis reports.
"""
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Any, Optional

from .life_state import LifeState
from .scenario import Scenario
from .simulation_engine import run_monte_carlo, SimulationConfig


@dataclass
class DecisionReport:
    """Structured decision analysis report."""
    scenario_name: str
    decision_type: str
    generated_at: datetime

    # Current state summary
    current_state_summary: Dict[str, Any]

    # Simulation results
    time_horizon_months: int
    num_simulations: int

    # Outcome distributions
    financial_outcomes: Dict[str, Any]
    wellbeing_outcomes: Dict[str, Any]

    # Risk analysis
    risk_metrics: Dict[str, float]

    # Key insights
    insights: List[str]

    # Recommendation
    recommendation: str
    confidence_level: str

    def to_markdown(self) -> str:
        """Generate markdown report."""
        md = f"""# Decision Analysis Report: {self.scenario_name}

**Generated:** {self.generated_at.strftime('%Y-%m-%d %H:%M')}
**Decision Type:** {self.decision_type}
**Time Horizon:** {self.time_horizon_months} months ({self.time_horizon_months // 12} years)
**Simulations Run:** {self.num_simulations}

---

## Current State Summary

| Domain | Status |
|--------|--------|
| Monthly Income | ${self.current_state_summary.get('income', 0):,.0f} |
| Monthly Expenses | ${self.current_state_summary.get('expenses', 0):,.0f} |
| Savings | ${self.current_state_summary.get('savings', 0):,.0f} |
| Net Worth | ${self.current_state_summary.get('net_worth', 0):,.0f} |
| Wellbeing Score | {self.current_state_summary.get('wellbeing', 0):.0%} |

---

## Projected Outcomes

### Financial Trajectory

| Metric | Pessimistic (10%) | Expected (50%) | Optimistic (90%) |
|--------|-------------------|----------------|------------------|
| Savings | ${self.financial_outcomes['savings']['p10']:,.0f} | ${self.financial_outcomes['savings']['p50']:,.0f} | ${self.financial_outcomes['savings']['p90']:,.0f} |
| Net Worth | ${self.financial_outcomes['net_worth']['p10']:,.0f} | ${self.financial_outcomes['net_worth']['p50']:,.0f} | ${self.financial_outcomes['net_worth']['p90']:,.0f} |

**Expected Change in Net Worth:** ${self.financial_outcomes['net_worth']['p50'] - self.current_state_summary.get('net_worth', 0):+,.0f}

### Wellbeing Trajectory

| Metric | Pessimistic (10%) | Expected (50%) | Optimistic (90%) |
|--------|-------------------|----------------|------------------|
| Wellbeing Score | {self.wellbeing_outcomes['p10']:.0%} | {self.wellbeing_outcomes['p50']:.0%} | {self.wellbeing_outcomes['p90']:.0%} |

---

## Risk Analysis

| Risk Metric | Value |
|-------------|-------|
| Probability of Negative Savings | {self.risk_metrics.get('probability_negative_savings', 0):.1%} |
| Probability of Wellbeing Decline | {self.risk_metrics.get('probability_wellbeing_decline', 0):.1%} |

---

## Key Insights

"""
        for i, insight in enumerate(self.insights, 1):
            md += f"{i}. {insight}\n"

        md += f"""
---

## Recommendation

**{self.recommendation}**

*Confidence Level: {self.confidence_level}*

---

*This analysis is based on Monte Carlo simulation with {self.num_simulations} runs.
Actual outcomes may vary based on factors not modeled. Use as one input among many
in your decision-making process.*
"""
        return md

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "scenario_name": self.scenario_name,
            "decision_type": self.decision_type,
            "generated_at": self.generated_at.isoformat(),
            "current_state_summary": self.current_state_summary,
            "time_horizon_months": self.time_horizon_months,
            "num_simulations": self.num_simulations,
            "financial_outcomes": self.financial_outcomes,
            "wellbeing_outcomes": self.wellbeing_outcomes,
            "risk_metrics": self.risk_metrics,
            "insights": self.insights,
            "recommendation": self.recommendation,
            "confidence_level": self.confidence_level
        }


class ReportGenerator:
    """Generates decision analysis reports."""

    def __init__(self, llm_client: Any = None):
        """
        Initialize report generator.

        Args:
            llm_client: Optional LLM client for insight generation
        """
        self.llm = llm_client

    def generate_report(
        self,
        current_state: LifeState,
        scenario: Scenario,
        simulation_results: Dict[str, Any],
        baseline_results: Optional[Dict[str, Any]] = None
    ) -> DecisionReport:
        """
        Generate a comprehensive decision report.

        Args:
            current_state: Current life state
            scenario: The scenario being analyzed
            simulation_results: Results from Monte Carlo simulation
            baseline_results: Optional results without the scenario (for comparison)
        """
        # Extract outcomes
        outcomes = simulation_results["outcomes"]

        # Generate insights
        insights = self._generate_insights(
            current_state,
            scenario,
            outcomes,
            simulation_results["risk_metrics"],
            baseline_results
        )

        # Generate recommendation
        recommendation, confidence = self._generate_recommendation(
            scenario,
            outcomes,
            simulation_results["risk_metrics"]
        )

        return DecisionReport(
            scenario_name=scenario.name,
            decision_type=scenario.decision_type.value,
            generated_at=datetime.now(),
            current_state_summary={
                "income": current_state.financial.monthly_income,
                "expenses": current_state.financial.monthly_expenses,
                "savings": current_state.financial.savings,
                "net_worth": current_state.financial.net_worth,
                "wellbeing": current_state.compute_wellbeing_score()
            },
            time_horizon_months=simulation_results["time_horizon_months"],
            num_simulations=simulation_results["num_runs"],
            financial_outcomes={
                "savings": outcomes["savings"],
                "net_worth": outcomes["net_worth"]
            },
            wellbeing_outcomes=outcomes["wellbeing"],
            risk_metrics=simulation_results["risk_metrics"],
            insights=insights,
            recommendation=recommendation,
            confidence_level=confidence
        )

    def _generate_insights(
        self,
        state: LifeState,
        scenario: Scenario,
        outcomes: Dict,
        risk_metrics: Dict,
        baseline: Optional[Dict]
    ) -> List[str]:
        """Generate key insights from simulation results."""
        insights = []

        # Financial runway insight
        current_runway = state.financial.runway_months
        projected_savings = outcomes["savings"]["p50"]
        projected_runway = projected_savings / max(1, state.financial.monthly_expenses)

        if projected_runway < 6:
            insights.append(
                f"Warning: Projected savings could leave you with only "
                f"{projected_runway:.1f} months of runway. Consider building "
                f"emergency fund before this decision."
            )
        elif projected_runway > current_runway:
            insights.append(
                f"This scenario is projected to increase your financial runway "
                f"from {current_runway:.1f} to {projected_runway:.1f} months."
            )

        # Risk insight
        neg_savings_prob = risk_metrics.get("probability_negative_savings", 0)
        if neg_savings_prob > 0.2:
            insights.append(
                f"There is a {neg_savings_prob:.0%} chance of depleting savings "
                f"within the time horizon. Consider risk mitigation strategies."
            )
        elif neg_savings_prob < 0.05:
            insights.append(
                f"Financial risk is low: only {neg_savings_prob:.0%} chance of "
                f"negative savings."
            )

        # Wellbeing insight
        wellbeing_change = outcomes["wellbeing"]["p50"] - state.compute_wellbeing_score()
        if wellbeing_change > 0.1:
            insights.append(
                f"Expected to improve overall wellbeing by {wellbeing_change:.0%}."
            )
        elif wellbeing_change < -0.1:
            insights.append(
                f"May decrease wellbeing by {abs(wellbeing_change):.0%}. "
                f"Consider non-financial factors carefully."
            )

        # Variance insight
        savings_spread = outcomes["savings"]["p90"] - outcomes["savings"]["p10"]
        if savings_spread > state.financial.savings * 2:
            insights.append(
                f"High outcome variance: best and worst cases differ by "
                f"${savings_spread:,.0f}. This is a high-uncertainty decision."
            )

        # Scenario-specific insights
        if scenario.decision_type.value == "start_business":
            if scenario.failure_probability > 0.5:
                insights.append(
                    f"Business ventures have {scenario.failure_probability:.0%} "
                    f"failure rate historically. Plan for contingencies."
                )

        return insights[:5]  # Max 5 insights

    def _generate_recommendation(
        self,
        scenario: Scenario,
        outcomes: Dict,
        risk_metrics: Dict
    ) -> tuple:
        """Generate recommendation based on analysis."""
        # Score the decision
        score = 0

        # Financial improvement
        if outcomes["net_worth"]["p50"] > 0:
            score += 1
        if outcomes["savings"]["p10"] > 0:
            score += 1

        # Wellbeing improvement
        if outcomes["wellbeing"]["p50"] > 0.6:
            score += 1
        if outcomes["wellbeing"]["p10"] > 0.5:
            score += 1

        # Risk tolerance
        if risk_metrics.get("probability_negative_savings", 1) < 0.1:
            score += 1
        if risk_metrics.get("probability_wellbeing_decline", 1) < 0.2:
            score += 1

        # Generate recommendation
        if score >= 5:
            recommendation = (
                f"This scenario shows strong positive indicators across financial "
                f"and wellbeing outcomes. Consider proceeding with appropriate preparation."
            )
            confidence = "High"
        elif score >= 3:
            recommendation = (
                f"This scenario shows mixed results. The expected outcome is positive, "
                f"but there are notable risks. Proceed with caution and contingency plans."
            )
            confidence = "Medium"
        else:
            recommendation = (
                f"This scenario carries significant risk with uncertain rewards. "
                f"Consider alternatives or additional preparation before proceeding."
            )
            confidence = "Low"

        return recommendation, confidence

    def compare_scenarios(
        self,
        current_state: LifeState,
        scenarios: List[Scenario],
        config: SimulationConfig
    ) -> Dict[str, Any]:
        """Compare multiple scenarios side by side."""
        results = {}
        name_counts = {}

        for scenario in scenarios:
            sim_results = run_monte_carlo(current_state, [scenario], config)
            report = self.generate_report(current_state, scenario, sim_results)

            # Handle duplicate names by appending a number
            base_name = scenario.name
            if base_name in name_counts:
                name_counts[base_name] += 1
                key = f"{base_name} ({name_counts[base_name]})"
            else:
                name_counts[base_name] = 1
                key = base_name

            results[key] = {
                "report": report.to_dict(),
                "summary": {
                    "expected_net_worth": sim_results["outcomes"]["net_worth"]["p50"],
                    "expected_wellbeing": sim_results["outcomes"]["wellbeing"]["p50"],
                    "risk_score": sim_results["risk_metrics"]["probability_negative_savings"]
                }
            }

        # Rank scenarios
        ranked = sorted(
            results.items(),
            key=lambda x: (
                x[1]["summary"]["expected_wellbeing"] * 0.5 +
                (1 - x[1]["summary"]["risk_score"]) * 0.3 +
                min(1, x[1]["summary"]["expected_net_worth"] / 100000) * 0.2
            ),
            reverse=True
        )

        return {
            "scenarios": results,
            "ranking": [name for name, _ in ranked],
            "best_choice": ranked[0][0] if ranked else None
        }

    def generate_summary(
        self,
        report: DecisionReport,
        max_length: int = 500
    ) -> str:
        """Generate a brief summary of the report."""
        summary = f"""
**{report.scenario_name}** ({report.decision_type})

Expected outcome over {report.time_horizon_months // 12} years:
- Net Worth: ${report.financial_outcomes['net_worth']['p50']:,.0f} (range: ${report.financial_outcomes['net_worth']['p10']:,.0f} to ${report.financial_outcomes['net_worth']['p90']:,.0f})
- Wellbeing: {report.wellbeing_outcomes['p50']:.0%}
- Risk of negative savings: {report.risk_metrics.get('probability_negative_savings', 0):.0%}

**Recommendation ({report.confidence_level} confidence):** {report.recommendation[:200]}...
"""
        return summary.strip()
