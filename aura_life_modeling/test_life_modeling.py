"""
Tests for AURA Predictive Life Modeling.
"""

import unittest
import sys
from pathlib import Path
from datetime import datetime

# Support both direct execution and package import
try:
    from .life_state import (
        LifeState, LifeDomain, FinancialState, CareerState,
        HealthState, RelationshipState, PersonalState, TimeState
    )
    from .scenario import (
        Scenario, ScenarioImpact, ImpactRange, DecisionType, ScenarioTemplates
    )
    from .simulation_engine import (
        LifeSimulationModel, SimulationConfig, run_monte_carlo
    )
    from .causal_analyzer import CausalLifeAnalyzer
    from .report_generator import ReportGenerator, DecisionReport
    from .mcp_tools import LifeModelingTools
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from aura_life_modeling.life_state import (
        LifeState, LifeDomain, FinancialState, CareerState,
        HealthState, RelationshipState, PersonalState, TimeState
    )
    from aura_life_modeling.scenario import (
        Scenario, ScenarioImpact, ImpactRange, DecisionType, ScenarioTemplates
    )
    from aura_life_modeling.simulation_engine import (
        LifeSimulationModel, SimulationConfig, run_monte_carlo
    )
    from aura_life_modeling.causal_analyzer import CausalLifeAnalyzer
    from aura_life_modeling.report_generator import ReportGenerator, DecisionReport
    from aura_life_modeling.mcp_tools import LifeModelingTools


class TestLifeState(unittest.TestCase):
    """Test LifeState model."""

    def test_financial_state_calculations(self):
        """Test financial calculations."""
        fin = FinancialState(
            monthly_income=5000,
            monthly_expenses=3000,
            savings=20000,
            investments=10000,
            debt=5000
        )

        self.assertEqual(fin.monthly_surplus, 2000)
        self.assertEqual(fin.net_worth, 25000)  # 20000 + 10000 - 5000
        self.assertAlmostEqual(fin.runway_months, 6.67, places=1)

    def test_life_state_creation(self):
        """Test LifeState creation."""
        state = LifeState(
            user_id="test_user",
            timestamp=datetime.now()
        )

        self.assertEqual(state.user_id, "test_user")
        self.assertIsInstance(state.financial, FinancialState)
        self.assertIsInstance(state.career, CareerState)

    def test_wellbeing_score(self):
        """Test wellbeing score calculation."""
        state = LifeState(
            user_id="test",
            timestamp=datetime.now()
        )

        # Set good conditions
        state.financial.monthly_income = 5000
        state.financial.monthly_expenses = 3000
        state.financial.savings = 36000  # 12 months runway
        state.career.satisfaction_score = 0.8
        state.health.stress_level = 0.2
        state.personal.life_satisfaction = 0.8

        score = state.compute_wellbeing_score()
        self.assertGreater(score, 0.5)
        self.assertLessEqual(score, 1.0)

    def test_serialization(self):
        """Test to_dict and from_dict."""
        state = LifeState(
            user_id="test",
            timestamp=datetime.now()
        )
        state.financial.savings = 10000
        state.career.current_role = "Developer"

        # Serialize
        data = state.to_dict()
        self.assertIn("financial", data)
        self.assertIn("career", data)

        # Deserialize
        restored = LifeState.from_dict(data)
        self.assertEqual(restored.user_id, "test")
        self.assertEqual(restored.financial.savings, 10000)
        self.assertEqual(restored.career.current_role, "Developer")


class TestScenario(unittest.TestCase):
    """Test Scenario models."""

    def test_impact_range_sampling(self):
        """Test ImpactRange sampling."""
        impact = ImpactRange(
            min_value=0.0,
            max_value=1.0,
            most_likely=0.5,
            distribution="triangular"
        )

        samples = [impact.sample() for _ in range(100)]
        self.assertTrue(all(0 <= s <= 1 for s in samples))

    def test_scenario_creation(self):
        """Test Scenario creation."""
        scenario = Scenario.create(
            name="Test Scenario",
            description="A test scenario",
            decision_type=DecisionType.CAREER_CHANGE,
            impacts=ScenarioImpact(
                income_change=ImpactRange(1.1, 1.3, 1.15)
            )
        )

        self.assertTrue(scenario.id.startswith("scenario_"))
        self.assertEqual(scenario.name, "Test Scenario")
        self.assertEqual(scenario.decision_type, DecisionType.CAREER_CHANGE)

    def test_scenario_templates(self):
        """Test pre-built scenario templates."""
        # Start business
        business = ScenarioTemplates.quit_job_start_business(
            startup_cost=10000,
            success_probability=0.3
        )
        self.assertEqual(business.decision_type, DecisionType.START_BUSINESS)
        self.assertEqual(business.failure_probability, 0.7)

        # Job change
        job = ScenarioTemplates.job_change(salary_change_pct=0.2)
        self.assertEqual(job.decision_type, DecisionType.CAREER_CHANGE)

        # Have child
        child = ScenarioTemplates.have_child()
        self.assertEqual(child.decision_type, DecisionType.HAVE_CHILD)


class TestSimulation(unittest.TestCase):
    """Test simulation engine."""

    def setUp(self):
        """Create test state."""
        self.state = LifeState(
            user_id="test",
            timestamp=datetime.now()
        )
        self.state.financial.monthly_income = 5000
        self.state.financial.monthly_expenses = 3000
        self.state.financial.savings = 20000

    def test_single_simulation(self):
        """Test single simulation run."""
        scenario = ScenarioTemplates.job_change(salary_change_pct=0.15)

        config = SimulationConfig(
            time_horizon_months=12,
            monte_carlo_runs=1,
            random_seed=42
        )

        model = LifeSimulationModel(self.state, [scenario], config)
        results = model.run()

        self.assertIn("final_state", results)
        self.assertIn("trajectory", results)
        self.assertIn("savings", results["final_state"])

    def test_monte_carlo(self):
        """Test Monte Carlo simulation."""
        scenario = ScenarioTemplates.job_change(salary_change_pct=0.15)

        config = SimulationConfig(
            time_horizon_months=12,
            monte_carlo_runs=20,
            random_seed=42
        )

        results = run_monte_carlo(self.state, [scenario], config)

        self.assertEqual(results["num_runs"], 20)
        self.assertIn("outcomes", results)
        self.assertIn("savings", results["outcomes"])
        self.assertIn("p50", results["outcomes"]["savings"])
        self.assertIn("risk_metrics", results)

    def test_start_business_simulation(self):
        """Test high-risk scenario simulation."""
        scenario = ScenarioTemplates.quit_job_start_business(
            startup_cost=10000,
            monthly_burn=2000,
            success_probability=0.3
        )

        config = SimulationConfig(
            time_horizon_months=24,
            monte_carlo_runs=50,
            random_seed=42
        )

        results = run_monte_carlo(self.state, [scenario], config)

        # Should have meaningful risk
        self.assertIn("probability_negative_savings", results["risk_metrics"])


class TestCausalAnalyzer(unittest.TestCase):
    """Test causal analysis."""

    def setUp(self):
        """Create test state and analyzer."""
        self.state = LifeState(
            user_id="test",
            timestamp=datetime.now()
        )
        self.state.financial.monthly_income = 5000
        self.state.personal.life_satisfaction = 0.6
        self.analyzer = CausalLifeAnalyzer()

    def test_heuristic_analysis(self):
        """Test heuristic causal analysis."""
        result = self.analyzer.analyze_decision_effect(
            decision="career_change",
            outcome="income",
            current_state=self.state
        )

        self.assertIn("causal_effect", result)
        self.assertIn("method", result)
        self.assertEqual(result["method"], "heuristic")

    def test_what_if_analysis(self):
        """Test what-if analysis."""
        results = self.analyzer.what_if_analysis(
            current_state=self.state,
            intervention={"income": 7000},
            outcomes_of_interest=["life_satisfaction", "savings"]
        )

        self.assertIn("life_satisfaction", results)
        self.assertIn("predicted_value", results["life_satisfaction"])

    def test_confounders(self):
        """Test confounder identification."""
        confounders = self.analyzer.identify_confounders(
            decision="career_change",
            outcome="income"
        )

        self.assertIsInstance(confounders, list)
        self.assertGreater(len(confounders), 0)


class TestReportGenerator(unittest.TestCase):
    """Test report generation."""

    def setUp(self):
        """Create test state and reporter."""
        self.state = LifeState(
            user_id="test",
            timestamp=datetime.now()
        )
        self.state.financial.monthly_income = 5000
        self.state.financial.monthly_expenses = 3000
        self.state.financial.savings = 20000
        self.reporter = ReportGenerator()

    def test_report_generation(self):
        """Test full report generation."""
        scenario = ScenarioTemplates.job_change(salary_change_pct=0.15)

        config = SimulationConfig(
            time_horizon_months=12,
            monte_carlo_runs=20,
            random_seed=42
        )

        sim_results = run_monte_carlo(self.state, [scenario], config)
        report = self.reporter.generate_report(self.state, scenario, sim_results)

        self.assertIsInstance(report, DecisionReport)
        self.assertEqual(report.scenario_name, "Change Jobs")
        self.assertGreater(len(report.insights), 0)

    def test_markdown_output(self):
        """Test markdown report output."""
        scenario = ScenarioTemplates.job_change()
        config = SimulationConfig(time_horizon_months=12, monte_carlo_runs=10)
        sim_results = run_monte_carlo(self.state, [scenario], config)
        report = self.reporter.generate_report(self.state, scenario, sim_results)

        md = report.to_markdown()
        self.assertIn("Decision Analysis Report", md)
        self.assertIn("Change Jobs", md)
        self.assertIn("Recommendation", md)

    def test_scenario_comparison(self):
        """Test comparing multiple scenarios."""
        scenarios = [
            ScenarioTemplates.job_change(salary_change_pct=0.15),
            ScenarioTemplates.job_change(salary_change_pct=0.25),
        ]

        config = SimulationConfig(time_horizon_months=12, monte_carlo_runs=10)
        comparison = self.reporter.compare_scenarios(self.state, scenarios, config)

        self.assertIn("ranking", comparison)
        self.assertIn("best_choice", comparison)
        self.assertEqual(len(comparison["ranking"]), 2)


class TestMCPTools(unittest.TestCase):
    """Test MCP tools interface."""

    def test_get_tools(self):
        """Test tool definitions."""
        tools = LifeModelingTools()
        tool_defs = tools.get_tools()

        self.assertEqual(len(tool_defs), 6)

        tool_names = [t["name"] for t in tool_defs]
        self.assertIn("life_state_update", tool_names)
        self.assertIn("simulate_decision", tool_names)
        self.assertIn("compare_decisions", tool_names)
        self.assertIn("what_if_analysis", tool_names)
        self.assertIn("get_life_state", tool_names)
        self.assertIn("generate_decision_report", tool_names)

    def test_life_state_update(self):
        """Test updating life state."""
        tools = LifeModelingTools()

        result = tools.handle_tool_call("life_state_update", {
            "financial": {
                "monthly_income": 6000,
                "monthly_expenses": 4000,
                "savings": 30000
            }
        }, user_id="test")

        self.assertTrue(result["success"])
        self.assertIn("wellbeing_score", result)

    def test_get_life_state(self):
        """Test getting life state."""
        tools = LifeModelingTools()

        # First update
        tools.handle_tool_call("life_state_update", {
            "financial": {"savings": 25000}
        }, user_id="test")

        # Then get
        result = tools.handle_tool_call("get_life_state", {}, user_id="test")

        self.assertIn("state", result)
        self.assertIn("wellbeing_score", result)

    def test_simulate_decision(self):
        """Test decision simulation via MCP."""
        tools = LifeModelingTools()

        # Setup state
        tools.handle_tool_call("life_state_update", {
            "financial": {
                "monthly_income": 5000,
                "monthly_expenses": 3000,
                "savings": 20000
            }
        }, user_id="test")

        # Simulate
        result = tools.handle_tool_call("simulate_decision", {
            "decision_type": "career_change",
            "parameters": {"salary_change_pct": 0.2},
            "time_horizon_years": 3,
            "num_simulations": 20
        }, user_id="test")

        self.assertIn("outcomes", result)
        self.assertIn("expected_savings", result["outcomes"])
        self.assertIn("risks", result)

    def test_generate_report(self):
        """Test report generation via MCP."""
        tools = LifeModelingTools()

        # Setup state
        tools.handle_tool_call("life_state_update", {
            "financial": {
                "monthly_income": 5000,
                "monthly_expenses": 3000,
                "savings": 20000
            }
        }, user_id="test")

        # Generate report
        result = tools.handle_tool_call("generate_decision_report", {
            "decision_type": "career_change",
            "format": "markdown"
        }, user_id="test")

        self.assertIn("report_markdown", result)
        self.assertIn("Decision Analysis Report", result["report_markdown"])


if __name__ == "__main__":
    unittest.main()
