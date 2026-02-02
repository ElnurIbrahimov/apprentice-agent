"""
AURA Predictive Life Modeling - Simulation Engine

Uses Mesa for agent-based modeling of life trajectories.
"""
import random
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import numpy as np

from .life_state import LifeState, LifeDomain
from .scenario import Scenario, ScenarioImpact, ImpactRange

# Mesa imports - graceful fallback if not available
try:
    from mesa import Agent, Model
    from mesa.time import RandomActivation
    from mesa.datacollection import DataCollector
    MESA_AVAILABLE = True
except ImportError:
    MESA_AVAILABLE = False
    Agent = object
    Model = object

logger = logging.getLogger(__name__)


@dataclass
class SimulationConfig:
    """Configuration for simulation runs."""
    time_horizon_months: int = 60        # 5 years default
    monte_carlo_runs: int = 100          # Number of simulations
    random_seed: Optional[int] = None
    include_market_volatility: bool = True
    include_life_events: bool = True
    inflation_rate: float = 0.03         # 3% annual


class LifeFactorAgent(Agent):
    """
    Agent representing a life domain factor.
    Each factor evolves over time based on rules and random events.
    """

    def __init__(self, unique_id: int, model: "LifeSimulationModel", domain: str, initial_value: float):
        if MESA_AVAILABLE:
            super().__init__(unique_id, model)
        self.unique_id = unique_id
        self.model = model
        self.domain = domain
        self.value = initial_value
        self.history: List[Tuple[int, float]] = [(0, initial_value)]
        self.trend = 0.0  # Current trend direction
        self.volatility = 0.1

    def step(self):
        """Advance one time step (1 month)."""
        # Apply trend with noise
        noise = random.gauss(0, self.volatility)
        change = self.trend + noise

        # Apply change with bounds
        self.value = max(0.0, min(1.0, self.value + change))
        self.history.append((self.model.current_step, self.value))

        # Mean reversion tendency
        self.trend *= 0.9

    def apply_shock(self, magnitude: float, duration: int = 1):
        """Apply an external shock to this factor."""
        self.value = max(0.0, min(1.0, self.value + magnitude))
        self.trend += magnitude * 0.1  # Lingering effect


class LifeSimulationModel(Model):
    """
    Mesa model simulating life trajectory over time.
    """

    def __init__(
        self,
        initial_state: LifeState,
        scenarios: List[Scenario],
        config: SimulationConfig
    ):
        if MESA_AVAILABLE:
            super().__init__()
        self.initial_state = initial_state
        self.scenarios = scenarios
        self.config = config
        self.current_step = 0

        # Set random seed
        if config.random_seed:
            random.seed(config.random_seed)
            np.random.seed(config.random_seed)

        # Initialize schedule
        if MESA_AVAILABLE:
            self.schedule = RandomActivation(self)
        else:
            self.schedule = SimpleSchedule()

        # Create agents for key life factors
        self._create_factor_agents()

        # Track financial trajectory separately (more complex)
        # Must be initialized BEFORE applying scenario impacts
        self.financial_history: List[Dict] = []
        self._init_financial_state()

        # Apply scenario impacts (after financial state is initialized)
        self._apply_scenario_impacts()

        # Data collector
        if MESA_AVAILABLE:
            self.datacollector = DataCollector(
                model_reporters={
                    "Savings": lambda m: m.savings,
                    "Monthly_Income": lambda m: m.monthly_income,
                    "Net_Worth": lambda m: m.net_worth,
                    "Wellbeing": lambda m: m.compute_wellbeing(),
                    "Stress": lambda m: m.stress_level
                }
            )
        else:
            self.datacollector = SimpleDataCollector()

    def _create_factor_agents(self):
        """Create agents for trackable life factors."""
        state = self.initial_state

        factors = [
            ("career_satisfaction", state.career.satisfaction_score),
            ("job_security", state.career.job_security),
            ("health_physical", state.health.physical_health),
            ("health_mental", state.health.mental_health),
            ("stress", state.health.stress_level),
            ("energy", state.health.energy_level),
            ("life_satisfaction", state.personal.life_satisfaction),
            ("relationship_quality", state.relationships.social_satisfaction)
        ]

        for i, (name, value) in enumerate(factors):
            agent = LifeFactorAgent(i, self, name, value)
            self.schedule.add(agent)

    def _init_financial_state(self):
        """Initialize financial tracking."""
        state = self.initial_state.financial
        self.savings = state.savings
        self.monthly_income = state.monthly_income
        self.monthly_expenses = state.monthly_expenses
        self.investments = state.investments
        self.debt = state.debt

        self.financial_history.append({
            "month": 0,
            "savings": self.savings,
            "income": self.monthly_income,
            "expenses": self.monthly_expenses,
            "net_worth": self.net_worth
        })

    @property
    def net_worth(self) -> float:
        """Calculate current net worth."""
        return self.savings + self.investments - self.debt

    @property
    def stress_level(self) -> float:
        """Get current stress level from agent."""
        for agent in self.schedule.agents:
            if agent.domain == "stress":
                return agent.value
        return 0.5

    def _apply_scenario_impacts(self):
        """Apply initial scenario impacts."""
        for scenario in self.scenarios:
            impacts = scenario.immediate_impacts

            # Financial impacts
            if impacts.income_change:
                multiplier = impacts.income_change.sample()
                self.monthly_income *= multiplier

            if impacts.one_time_cost:
                cost = impacts.one_time_cost.sample()
                self.savings -= cost

            if impacts.monthly_expense_change:
                delta = impacts.monthly_expense_change.sample()
                self.monthly_expenses += delta

            # Factor impacts
            for agent in self.schedule.agents:
                if agent.domain == "stress" and impacts.stress_change:
                    agent.apply_shock(impacts.stress_change.sample())
                elif agent.domain == "career_satisfaction" and impacts.job_satisfaction_change:
                    agent.apply_shock(impacts.job_satisfaction_change.sample())
                elif agent.domain == "life_satisfaction" and impacts.satisfaction_change:
                    agent.apply_shock(impacts.satisfaction_change.sample())
                elif agent.domain == "energy" and impacts.energy_change:
                    agent.apply_shock(impacts.energy_change.sample())

    def step(self):
        """Advance simulation by one month."""
        self.current_step += 1

        # Update factors
        self.schedule.step()

        # Update financials
        self._update_financials()

        # Random life events
        if self.config.include_life_events:
            self._check_random_events()

        # Collect data
        self.datacollector.collect(self)

    def _update_financials(self):
        """Update financial state for one month."""
        # Apply inflation to expenses (monthly)
        monthly_inflation = (1 + self.config.inflation_rate) ** (1/12) - 1
        self.monthly_expenses *= (1 + monthly_inflation)

        # Calculate surplus
        surplus = self.monthly_income - self.monthly_expenses

        # Update savings
        self.savings += surplus

        # Investment returns (if positive savings)
        if self.investments > 0 and self.config.include_market_volatility:
            # Annual return ~7% with volatility
            monthly_return = random.gauss(0.07/12, 0.15/np.sqrt(12))
            self.investments *= (1 + monthly_return)

        # Track history
        self.financial_history.append({
            "month": self.current_step,
            "savings": self.savings,
            "income": self.monthly_income,
            "expenses": self.monthly_expenses,
            "net_worth": self.net_worth,
            "surplus": surplus
        })

    def _check_random_events(self):
        """Check for and apply random life events."""
        # Small probability of various events each month
        events = [
            (0.005, "job_loss", {"income_mult": 0.0, "stress": 0.3}),
            (0.002, "health_issue", {"expense": 5000, "health": -0.2}),
            (0.003, "windfall", {"savings": 5000}),
            (0.01, "expense_shock", {"expense": 1000}),
        ]

        for prob, event_name, impacts in events:
            if random.random() < prob:
                self._apply_random_event(event_name, impacts)

    def _apply_random_event(self, event_name: str, impacts: Dict):
        """Apply a random event's impacts."""
        if "income_mult" in impacts:
            self.monthly_income *= impacts["income_mult"]
        if "expense" in impacts:
            self.savings -= impacts["expense"]
        if "savings" in impacts:
            self.savings += impacts["savings"]
        if "stress" in impacts:
            for agent in self.schedule.agents:
                if agent.domain == "stress":
                    agent.apply_shock(impacts["stress"])
        if "health" in impacts:
            for agent in self.schedule.agents:
                if agent.domain.startswith("health"):
                    agent.apply_shock(impacts["health"])

    def compute_wellbeing(self) -> float:
        """Compute current wellbeing score."""
        factor_values = {agent.domain: agent.value for agent in self.schedule.agents}

        # Financial component
        runway = self.savings / max(1, self.monthly_expenses)
        financial_security = min(1.0, runway / 12)

        # Weighted average
        weights = {
            "career_satisfaction": 0.12,
            "job_security": 0.08,
            "health_physical": 0.12,
            "health_mental": 0.12,
            "stress": -0.10,  # Negative weight
            "energy": 0.08,
            "life_satisfaction": 0.18,
            "relationship_quality": 0.10,
            "financial": 0.20
        }

        score = weights["financial"] * financial_security
        for domain, weight in weights.items():
            if domain in factor_values:
                if weight < 0:
                    score += abs(weight) * (1 - factor_values[domain])
                else:
                    score += weight * factor_values[domain]

        return max(0, min(1, score))

    def run(self) -> Dict[str, Any]:
        """Run full simulation."""
        for _ in range(self.config.time_horizon_months):
            self.step()

        return self.get_results()

    def get_results(self) -> Dict[str, Any]:
        """Get simulation results."""
        df = self.datacollector.get_model_vars_dataframe()

        return {
            "time_horizon_months": self.config.time_horizon_months,
            "final_state": {
                "savings": self.savings,
                "net_worth": self.net_worth,
                "monthly_income": self.monthly_income,
                "monthly_expenses": self.monthly_expenses,
                "wellbeing": self.compute_wellbeing()
            },
            "trajectory": {
                "savings": df["Savings"] if "Savings" in df else [],
                "net_worth": df["Net_Worth"] if "Net_Worth" in df else [],
                "wellbeing": df["Wellbeing"] if "Wellbeing" in df else [],
                "stress": df["Stress"] if "Stress" in df else []
            },
            "financial_history": self.financial_history,
            "factor_histories": {
                agent.domain: agent.history
                for agent in self.schedule.agents
            }
        }


class SimpleSchedule:
    """Simple schedule for when Mesa is not available."""

    def __init__(self):
        self.agents: List[LifeFactorAgent] = []

    def add(self, agent: LifeFactorAgent):
        self.agents.append(agent)

    def step(self):
        for agent in self.agents:
            agent.step()


class SimpleDataCollector:
    """Simple data collector for when Mesa is not available."""

    def __init__(self):
        self.data: Dict[str, List] = {
            "Savings": [],
            "Monthly_Income": [],
            "Net_Worth": [],
            "Wellbeing": [],
            "Stress": []
        }

    def collect(self, model: LifeSimulationModel):
        self.data["Savings"].append(model.savings)
        self.data["Monthly_Income"].append(model.monthly_income)
        self.data["Net_Worth"].append(model.net_worth)
        self.data["Wellbeing"].append(model.compute_wellbeing())
        self.data["Stress"].append(model.stress_level)

    def get_model_vars_dataframe(self) -> Dict[str, List]:
        return self.data


def run_monte_carlo(
    initial_state: LifeState,
    scenarios: List[Scenario],
    config: SimulationConfig
) -> Dict[str, Any]:
    """
    Run Monte Carlo simulation with multiple runs.
    Returns distribution of outcomes.
    """
    results = []

    for run in range(config.monte_carlo_runs):
        # Create new config with different seed
        run_config = SimulationConfig(
            time_horizon_months=config.time_horizon_months,
            monte_carlo_runs=1,
            random_seed=(config.random_seed or 0) + run,
            include_market_volatility=config.include_market_volatility,
            include_life_events=config.include_life_events,
            inflation_rate=config.inflation_rate
        )

        model = LifeSimulationModel(initial_state, scenarios, run_config)
        result = model.run()
        results.append(result)

    # Aggregate results
    final_savings = [r["final_state"]["savings"] for r in results]
    final_net_worth = [r["final_state"]["net_worth"] for r in results]
    final_wellbeing = [r["final_state"]["wellbeing"] for r in results]

    return {
        "num_runs": config.monte_carlo_runs,
        "time_horizon_months": config.time_horizon_months,
        "outcomes": {
            "savings": {
                "mean": float(np.mean(final_savings)),
                "std": float(np.std(final_savings)),
                "p10": float(np.percentile(final_savings, 10)),
                "p50": float(np.percentile(final_savings, 50)),
                "p90": float(np.percentile(final_savings, 90)),
                "min": float(min(final_savings)),
                "max": float(max(final_savings))
            },
            "net_worth": {
                "mean": float(np.mean(final_net_worth)),
                "std": float(np.std(final_net_worth)),
                "p10": float(np.percentile(final_net_worth, 10)),
                "p50": float(np.percentile(final_net_worth, 50)),
                "p90": float(np.percentile(final_net_worth, 90))
            },
            "wellbeing": {
                "mean": float(np.mean(final_wellbeing)),
                "std": float(np.std(final_wellbeing)),
                "p10": float(np.percentile(final_wellbeing, 10)),
                "p50": float(np.percentile(final_wellbeing, 50)),
                "p90": float(np.percentile(final_wellbeing, 90))
            }
        },
        "risk_metrics": {
            "probability_negative_savings": sum(1 for s in final_savings if s < 0) / len(final_savings),
            "probability_wellbeing_decline": sum(1 for w in final_wellbeing if w < 0.5) / len(final_wellbeing)
        },
        "all_runs": results
    }
