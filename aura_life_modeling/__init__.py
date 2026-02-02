"""
AURA Predictive Life Modeling
=============================

Simulates life decisions and models outcomes using agent-based modeling
and causal inference. Realizes AURA's World Simulator for personal
decision-making.

Features:
- Life state modeling across 6 domains (financial, career, health, relationships, personal, time)
- Mesa-based agent simulation for trajectory modeling
- Monte Carlo simulation for outcome distributions
- DoWhy integration for causal inference (optional)
- Pre-built scenario templates for common decisions
- Report generation with insights and recommendations
- MCP tools for AURA integration

Usage:
    from aura_life_modeling import LifeState, Scenario, SimulationConfig
    from aura_life_modeling import run_monte_carlo, LifeModelingTools
    from datetime import datetime

    # Create life state
    state = LifeState(user_id="user1", timestamp=datetime.now())
    state.financial.monthly_income = 5000
    state.financial.monthly_expenses = 3500
    state.financial.savings = 20000

    # Create scenario
    scenario = ScenarioTemplates.quit_job_start_business(
        startup_cost=15000,
        success_probability=0.4
    )

    # Run simulation
    results = run_monte_carlo(state, [scenario], SimulationConfig())

    # Generate report
    reporter = ReportGenerator()
    report = reporter.generate_report(state, scenario, results)
    print(report.to_markdown())
"""

__version__ = "1.0.0"

from .life_state import (
    LifeState,
    LifeDomain,
    FinancialState,
    CareerState,
    HealthState,
    RelationshipState,
    PersonalState,
    TimeState
)

from .scenario import (
    Scenario,
    ScenarioImpact,
    ImpactRange,
    DecisionType,
    ScenarioTemplates
)

from .simulation_engine import (
    LifeSimulationModel,
    SimulationConfig,
    run_monte_carlo,
    MESA_AVAILABLE
)

from .causal_analyzer import (
    CausalLifeAnalyzer,
    DOWHY_AVAILABLE
)

from .report_generator import (
    DecisionReport,
    ReportGenerator
)

from .mcp_tools import LifeModelingTools


__all__ = [
    # Life State
    "LifeState",
    "LifeDomain",
    "FinancialState",
    "CareerState",
    "HealthState",
    "RelationshipState",
    "PersonalState",
    "TimeState",

    # Scenarios
    "Scenario",
    "ScenarioImpact",
    "ImpactRange",
    "DecisionType",
    "ScenarioTemplates",

    # Simulation
    "LifeSimulationModel",
    "SimulationConfig",
    "run_monte_carlo",

    # Analysis
    "CausalLifeAnalyzer",
    "DecisionReport",
    "ReportGenerator",

    # Integration
    "LifeModelingTools",

    # Availability flags
    "MESA_AVAILABLE",
    "DOWHY_AVAILABLE",

    # Version
    "__version__"
]
