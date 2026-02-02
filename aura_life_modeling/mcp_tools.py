"""
AURA Predictive Life Modeling - MCP Tools

Exposes life modeling functionality to AURA's tool system.
"""
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from .life_state import LifeState, LifeDomain
from .scenario import Scenario, ScenarioTemplates, DecisionType, ScenarioImpact, ImpactRange
from .simulation_engine import run_monte_carlo, SimulationConfig
from .causal_analyzer import CausalLifeAnalyzer
from .report_generator import ReportGenerator

logger = logging.getLogger(__name__)


class LifeModelingTools:
    """MCP tools for life modeling operations."""

    def __init__(
        self,
        knowledge_graph: Any = None,
        episodic_memory: Any = None,
        llm_client: Any = None
    ):
        """
        Initialize life modeling tools.

        Args:
            knowledge_graph: Optional KG brain for state building
            episodic_memory: Optional episodic memory for history
            llm_client: Optional LLM client for insights
        """
        self.kg = knowledge_graph
        self.episodic = episodic_memory
        self.llm = llm_client
        self.analyzer = CausalLifeAnalyzer()
        self.reporter = ReportGenerator(llm_client)

        # Cache user's life state
        self._life_state_cache: Dict[str, LifeState] = {}

    def get_tools(self) -> List[Dict]:
        """Return MCP tool definitions."""
        return [
            {
                "name": "life_state_update",
                "description": "Update your life state model with current information. Call this before running simulations.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "financial": {
                            "type": "object",
                            "properties": {
                                "monthly_income": {"type": "number"},
                                "monthly_expenses": {"type": "number"},
                                "savings": {"type": "number"},
                                "investments": {"type": "number"},
                                "debt": {"type": "number"}
                            }
                        },
                        "career": {
                            "type": "object",
                            "properties": {
                                "current_role": {"type": "string"},
                                "satisfaction": {"type": "number", "minimum": 0, "maximum": 1},
                                "is_employed": {"type": "boolean"}
                            }
                        },
                        "health": {
                            "type": "object",
                            "properties": {
                                "stress_level": {"type": "number", "minimum": 0, "maximum": 1},
                                "age": {"type": "integer"}
                            }
                        },
                        "personal": {
                            "type": "object",
                            "properties": {
                                "life_satisfaction": {"type": "number", "minimum": 0, "maximum": 1},
                                "location": {"type": "string"}
                            }
                        }
                    }
                }
            },
            {
                "name": "simulate_decision",
                "description": "Simulate a life decision and see projected outcomes. Returns financial and wellbeing projections with confidence intervals.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "decision_type": {
                            "type": "string",
                            "enum": ["career_change", "quit_job", "start_business",
                                    "major_purchase", "relocation", "education",
                                    "have_child", "retirement", "lifestyle_change", "custom"],
                            "description": "Type of decision to simulate"
                        },
                        "parameters": {
                            "type": "object",
                            "description": "Decision-specific parameters (e.g., salary_change_pct, startup_cost)"
                        },
                        "time_horizon_years": {
                            "type": "integer",
                            "default": 5,
                            "description": "How many years to simulate"
                        },
                        "num_simulations": {
                            "type": "integer",
                            "default": 100,
                            "description": "Number of Monte Carlo runs"
                        }
                    },
                    "required": ["decision_type"]
                }
            },
            {
                "name": "compare_decisions",
                "description": "Compare multiple decision scenarios side by side to find the best option.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "decisions": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "decision_type": {"type": "string"},
                                    "parameters": {"type": "object"}
                                }
                            },
                            "description": "List of decisions to compare"
                        },
                        "time_horizon_years": {"type": "integer", "default": 5}
                    },
                    "required": ["decisions"]
                }
            },
            {
                "name": "what_if_analysis",
                "description": "Answer 'what if' questions about life changes using causal analysis.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "Natural language what-if question"
                        },
                        "variables": {
                            "type": "object",
                            "description": "Specific variable changes to analyze"
                        }
                    },
                    "required": ["question"]
                }
            },
            {
                "name": "get_life_state",
                "description": "Get current life state model and wellbeing score.",
                "parameters": {
                    "type": "object",
                    "properties": {}
                }
            },
            {
                "name": "generate_decision_report",
                "description": "Generate a detailed markdown report for a specific decision analysis.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "decision_type": {"type": "string"},
                        "parameters": {"type": "object"},
                        "format": {
                            "type": "string",
                            "enum": ["markdown", "json"],
                            "default": "markdown"
                        }
                    },
                    "required": ["decision_type"]
                }
            }
        ]

    def handle_tool_call(self, tool_name: str, arguments: Dict, user_id: str = "default") -> Dict[str, Any]:
        """Handle an MCP tool call."""
        handlers = {
            "life_state_update": lambda args: self._update_life_state(user_id, args),
            "get_life_state": lambda args: self._get_life_state(user_id),
            "simulate_decision": lambda args: self._simulate_decision(user_id, args),
            "compare_decisions": lambda args: self._compare_decisions(user_id, args),
            "what_if_analysis": lambda args: self._what_if_analysis(user_id, args),
            "generate_decision_report": lambda args: self._generate_report(user_id, args)
        }

        handler = handlers.get(tool_name)
        if not handler:
            return {"error": f"Unknown tool: {tool_name}"}

        try:
            return handler(arguments)
        except Exception as e:
            logger.error(f"Tool {tool_name} failed: {e}")
            return {"error": str(e)}

    def _get_or_create_state(self, user_id: str) -> LifeState:
        """Get existing state or create default."""
        if user_id not in self._life_state_cache:
            # Try to build from Knowledge Graph
            state = self._build_state_from_kg(user_id)
            if state is None:
                state = LifeState(
                    user_id=user_id,
                    timestamp=datetime.now()
                )
            self._life_state_cache[user_id] = state
        return self._life_state_cache[user_id]

    def _build_state_from_kg(self, user_id: str) -> Optional[LifeState]:
        """Try to build life state from Knowledge Graph."""
        if self.kg is None:
            return None

        try:
            # Query KG for user data if available
            if hasattr(self.kg, 'query'):
                # Implementation depends on KG structure
                pass
        except Exception as e:
            logger.debug(f"Could not build state from KG: {e}")

        return None

    def _update_life_state(self, user_id: str, params: Dict) -> Dict:
        """Update user's life state."""
        state = self._get_or_create_state(user_id)

        # Update financial
        if "financial" in params:
            fin = params["financial"]
            if "monthly_income" in fin:
                state.financial.monthly_income = fin["monthly_income"]
            if "monthly_expenses" in fin:
                state.financial.monthly_expenses = fin["monthly_expenses"]
            if "savings" in fin:
                state.financial.savings = fin["savings"]
            if "investments" in fin:
                state.financial.investments = fin["investments"]
            if "debt" in fin:
                state.financial.debt = fin["debt"]

        # Update career
        if "career" in params:
            car = params["career"]
            if "current_role" in car:
                state.career.current_role = car["current_role"]
            if "satisfaction" in car:
                state.career.satisfaction_score = car["satisfaction"]
            if "is_employed" in car:
                state.career.is_employed = car["is_employed"]

        # Update health
        if "health" in params:
            hea = params["health"]
            if "stress_level" in hea:
                state.health.stress_level = hea["stress_level"]
            if "age" in hea:
                state.health.age = hea["age"]

        # Update personal
        if "personal" in params:
            per = params["personal"]
            if "life_satisfaction" in per:
                state.personal.life_satisfaction = per["life_satisfaction"]
            if "location" in per:
                state.personal.location = per["location"]

        state.timestamp = datetime.now()
        self._life_state_cache[user_id] = state

        return {
            "success": True,
            "wellbeing_score": state.compute_wellbeing_score(),
            "financial_runway_months": state.financial.runway_months if state.financial.runway_months != float('inf') else "unlimited"
        }

    def _get_life_state(self, user_id: str) -> Dict:
        """Get current life state."""
        state = self._get_or_create_state(user_id)
        runway = state.financial.runway_months

        return {
            "state": state.to_dict(),
            "wellbeing_score": state.compute_wellbeing_score(),
            "financial_summary": {
                "net_worth": state.financial.net_worth,
                "monthly_surplus": state.financial.monthly_surplus,
                "runway_months": runway if runway != float('inf') else "unlimited"
            }
        }

    def _simulate_decision(self, user_id: str, params: Dict) -> Dict:
        """Run simulation for a decision."""
        state = self._get_or_create_state(user_id)

        # Build scenario from parameters
        decision_type = DecisionType(params["decision_type"])
        scenario_params = params.get("parameters", {})

        # Use templates or create custom
        scenario = self._create_scenario(decision_type, scenario_params)

        # Configure simulation
        config = SimulationConfig(
            time_horizon_months=params.get("time_horizon_years", 5) * 12,
            monte_carlo_runs=params.get("num_simulations", 100)
        )

        # Run simulation
        results = run_monte_carlo(state, [scenario], config)

        # Simplify output
        return {
            "scenario": scenario.name,
            "time_horizon_months": config.time_horizon_months,
            "simulations": config.monte_carlo_runs,
            "current_state": {
                "savings": state.financial.savings,
                "net_worth": state.financial.net_worth,
                "wellbeing": state.compute_wellbeing_score()
            },
            "outcomes": {
                "expected_savings": results["outcomes"]["savings"]["p50"],
                "savings_range": [
                    results["outcomes"]["savings"]["p10"],
                    results["outcomes"]["savings"]["p90"]
                ],
                "expected_net_worth": results["outcomes"]["net_worth"]["p50"],
                "net_worth_range": [
                    results["outcomes"]["net_worth"]["p10"],
                    results["outcomes"]["net_worth"]["p90"]
                ],
                "expected_wellbeing": results["outcomes"]["wellbeing"]["p50"],
                "wellbeing_range": [
                    results["outcomes"]["wellbeing"]["p10"],
                    results["outcomes"]["wellbeing"]["p90"]
                ]
            },
            "risks": results["risk_metrics"]
        }

    def _create_scenario(self, decision_type: DecisionType, params: Dict) -> Scenario:
        """Create scenario from type and parameters."""
        if decision_type == DecisionType.START_BUSINESS:
            return ScenarioTemplates.quit_job_start_business(
                startup_cost=params.get("startup_cost", 10000),
                monthly_burn=params.get("monthly_burn", 3000),
                success_probability=params.get("success_probability", 0.3)
            )
        elif decision_type == DecisionType.CAREER_CHANGE:
            return ScenarioTemplates.job_change(
                salary_change_pct=params.get("salary_change_pct", 0.15),
                satisfaction_change=params.get("satisfaction_change", 0.1)
            )
        elif decision_type == DecisionType.RELOCATION:
            return ScenarioTemplates.relocation(
                cost_of_living_change=params.get("cost_of_living_change", 0),
                salary_change=params.get("salary_change", 0),
                moving_cost=params.get("moving_cost", 5000)
            )
        elif decision_type == DecisionType.HAVE_CHILD:
            return ScenarioTemplates.have_child()
        elif decision_type == DecisionType.EDUCATION:
            return ScenarioTemplates.education(
                program_cost=params.get("program_cost", 20000),
                duration_months=params.get("duration_months", 24),
                salary_increase_after=params.get("salary_increase_after", 0.25)
            )
        elif decision_type == DecisionType.MAJOR_PURCHASE:
            return ScenarioTemplates.major_purchase(
                purchase_cost=params.get("purchase_cost", 30000),
                monthly_payment=params.get("monthly_payment", 500),
                satisfaction_boost=params.get("satisfaction_boost", 0.1)
            )
        elif decision_type == DecisionType.RETIREMENT:
            return ScenarioTemplates.retirement()
        elif decision_type == DecisionType.LIFESTYLE_CHANGE:
            return ScenarioTemplates.lifestyle_change(
                expense_change=params.get("expense_change", 0),
                stress_reduction=params.get("stress_reduction", 0.1),
                satisfaction_boost=params.get("satisfaction_boost", 0.1)
            )
        else:
            # Generic scenario
            return Scenario.create(
                name=params.get("name", "Custom Decision"),
                description=params.get("description", "User-defined scenario"),
                decision_type=decision_type,
                impacts=ScenarioImpact()
            )

    def _compare_decisions(self, user_id: str, params: Dict) -> Dict:
        """Compare multiple decisions."""
        state = self._get_or_create_state(user_id)

        scenarios = []
        for dec in params["decisions"]:
            decision_type = DecisionType(dec["decision_type"])
            scenario = self._create_scenario(decision_type, dec.get("parameters", {}))
            scenarios.append(scenario)

        config = SimulationConfig(
            time_horizon_months=params.get("time_horizon_years", 5) * 12,
            monte_carlo_runs=50  # Fewer for comparison
        )

        comparison = self.reporter.compare_scenarios(state, scenarios, config)

        return {
            "ranking": comparison["ranking"],
            "best_choice": comparison["best_choice"],
            "summaries": {
                name: data["summary"]
                for name, data in comparison["scenarios"].items()
            }
        }

    def _what_if_analysis(self, user_id: str, params: Dict) -> Dict:
        """Answer what-if questions."""
        state = self._get_or_create_state(user_id)

        # Parse question into interventions
        variables = params.get("variables", {})
        outcomes = ["life_satisfaction", "savings", "stress"]

        results = self.analyzer.what_if_analysis(state, variables, outcomes)

        return {
            "question": params["question"],
            "analysis": results,
            "interpretation": self._interpret_what_if(results)
        }

    def _interpret_what_if(self, results: Dict) -> str:
        """Generate natural language interpretation."""
        interpretations = []

        for outcome, data in results.items():
            baseline = data.get("baseline", 0) or 0
            predicted = data.get("predicted_value", baseline)
            change = predicted - baseline

            if abs(change) > 0.1:
                direction = "increase" if change > 0 else "decrease"
                interpretations.append(
                    f"{outcome.replace('_', ' ').title()} is expected to {direction} "
                    f"by {abs(change):.0%}"
                )

        return ". ".join(interpretations) if interpretations else "No significant changes expected."

    def _generate_report(self, user_id: str, params: Dict) -> Dict:
        """Generate detailed decision report."""
        state = self._get_or_create_state(user_id)

        decision_type = DecisionType(params["decision_type"])
        scenario = self._create_scenario(decision_type, params.get("parameters", {}))

        config = SimulationConfig(time_horizon_months=60, monte_carlo_runs=100)
        sim_results = run_monte_carlo(state, [scenario], config)

        report = self.reporter.generate_report(state, scenario, sim_results)

        if params.get("format") == "json":
            return report.to_dict()
        else:
            return {"report_markdown": report.to_markdown()}
