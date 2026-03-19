"""
AURA Predictive Life Modeling - Scenario Builder

Defines decisions and their expected impacts.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum
import uuid
import random


class DecisionType(Enum):
    """Types of life decisions."""
    CAREER_CHANGE = "career_change"
    QUIT_JOB = "quit_job"
    START_BUSINESS = "start_business"
    MAJOR_PURCHASE = "major_purchase"
    INVESTMENT = "investment"
    RELOCATION = "relocation"
    MARRIAGE = "marriage"
    HAVE_CHILD = "have_child"
    EDUCATION = "education"
    RETIREMENT = "retirement"
    LIFESTYLE_CHANGE = "lifestyle_change"
    CUSTOM = "custom"


@dataclass
class ImpactRange:
    """Range of possible impacts for a variable."""
    min_value: float
    max_value: float
    most_likely: float
    distribution: str = "triangular"  # triangular, normal, uniform

    def sample(self, rng=None) -> float:
        """Sample from the distribution."""
        if rng is None:
            rng = random

        if self.distribution == "triangular":
            return rng.triangular(self.min_value, self.max_value, self.most_likely)
        elif self.distribution == "normal":
            mean = self.most_likely
            std = (self.max_value - self.min_value) / 4
            return rng.gauss(mean, std)
        else:  # uniform
            return rng.uniform(self.min_value, self.max_value)


@dataclass
class ScenarioImpact:
    """Impact of a scenario on life domains."""
    # Financial impacts
    income_change: Optional[ImpactRange] = None           # Multiplier (1.0 = no change)
    one_time_cost: Optional[ImpactRange] = None           # One-time expense
    monthly_expense_change: Optional[ImpactRange] = None  # Delta
    savings_change: Optional[ImpactRange] = None          # Delta

    # Career impacts
    job_satisfaction_change: Optional[ImpactRange] = None
    skill_growth: Optional[ImpactRange] = None
    job_security_change: Optional[ImpactRange] = None

    # Health impacts
    stress_change: Optional[ImpactRange] = None
    energy_change: Optional[ImpactRange] = None

    # Time impacts
    work_hours_change: Optional[ImpactRange] = None
    commute_change: Optional[ImpactRange] = None
    leisure_change: Optional[ImpactRange] = None

    # Life satisfaction
    satisfaction_change: Optional[ImpactRange] = None

    # Delayed effects (months until impact)
    delay_months: int = 0

    # Duration (0 = permanent)
    duration_months: int = 0


@dataclass
class Scenario:
    """A life scenario to simulate."""
    id: str
    name: str
    description: str
    decision_type: DecisionType

    # Base impacts
    immediate_impacts: ScenarioImpact

    # Conditional impacts (if certain conditions are met)
    conditional_impacts: Dict[str, ScenarioImpact] = field(default_factory=dict)

    # Prerequisites (conditions that must be true)
    prerequisites: Dict[str, Any] = field(default_factory=dict)

    # Risk factors
    failure_probability: float = 0.0
    failure_impacts: Optional[ScenarioImpact] = None

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    tags: List[str] = field(default_factory=list)

    @classmethod
    def create(
        cls,
        name: str,
        description: str,
        decision_type: DecisionType,
        impacts: ScenarioImpact,
        **kwargs
    ) -> "Scenario":
        """Factory method to create a scenario."""
        return cls(
            id=f"scenario_{uuid.uuid4().hex[:12]}",
            name=name,
            description=description,
            decision_type=decision_type,
            immediate_impacts=impacts,
            **kwargs
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "decision_type": self.decision_type.value,
            "failure_probability": self.failure_probability,
            "tags": self.tags,
            "created_at": self.created_at.isoformat()
        }


class ScenarioTemplates:
    """Common life scenario templates."""

    @staticmethod
    def quit_job_start_business(
        startup_cost: float = 10000,
        monthly_burn: float = 3000,
        success_probability: float = 0.3,
        months_to_revenue: int = 12
    ) -> Scenario:
        """Template: Quit job to start a business."""
        return Scenario.create(
            name="Quit Job & Start Business",
            description="Leave employment to start your own business",
            decision_type=DecisionType.START_BUSINESS,
            impacts=ScenarioImpact(
                income_change=ImpactRange(0.0, 0.0, 0.0),  # No income initially
                one_time_cost=ImpactRange(startup_cost * 0.8, startup_cost * 1.5, startup_cost),
                monthly_expense_change=ImpactRange(monthly_burn * 0.8, monthly_burn * 1.3, monthly_burn),
                job_satisfaction_change=ImpactRange(0.1, 0.4, 0.25),
                stress_change=ImpactRange(0.1, 0.3, 0.2),
                work_hours_change=ImpactRange(10, 30, 20),
                satisfaction_change=ImpactRange(0.0, 0.3, 0.15),
                delay_months=0
            ),
            failure_probability=1 - success_probability,
            failure_impacts=ScenarioImpact(
                stress_change=ImpactRange(0.2, 0.4, 0.3),
                satisfaction_change=ImpactRange(-0.3, -0.1, -0.2),
                savings_change=ImpactRange(-startup_cost * 2, -startup_cost, -startup_cost * 1.5)
            ),
            tags=["career", "entrepreneurship", "high-risk"]
        )

    @staticmethod
    def job_change(
        salary_change_pct: float = 0.15,
        satisfaction_change: float = 0.1
    ) -> Scenario:
        """Template: Change jobs for better opportunity."""
        return Scenario.create(
            name="Change Jobs",
            description="Move to a new position at a different company",
            decision_type=DecisionType.CAREER_CHANGE,
            impacts=ScenarioImpact(
                income_change=ImpactRange(
                    1.0 + salary_change_pct * 0.5,
                    1.0 + salary_change_pct * 1.5,
                    1.0 + salary_change_pct
                ),
                job_satisfaction_change=ImpactRange(
                    satisfaction_change * 0.5,
                    satisfaction_change * 1.5,
                    satisfaction_change
                ),
                stress_change=ImpactRange(0.0, 0.15, 0.05),  # Initial adjustment stress
                skill_growth=ImpactRange(0.05, 0.2, 0.1),
                delay_months=0,
                duration_months=3  # Adjustment period
            ),
            failure_probability=0.1,  # Job doesn't work out
            tags=["career", "moderate-risk"]
        )

    @staticmethod
    def relocation(
        cost_of_living_change: float = 0.0,
        salary_change: float = 0.0,
        moving_cost: float = 5000
    ) -> Scenario:
        """Template: Relocate to a new city."""
        return Scenario.create(
            name="Relocate",
            description="Move to a new city/region",
            decision_type=DecisionType.RELOCATION,
            impacts=ScenarioImpact(
                income_change=ImpactRange(
                    1.0 + salary_change - 0.1,
                    1.0 + salary_change + 0.1,
                    1.0 + salary_change
                ),
                monthly_expense_change=ImpactRange(
                    cost_of_living_change * 0.8,
                    cost_of_living_change * 1.2,
                    cost_of_living_change
                ),
                one_time_cost=ImpactRange(moving_cost * 0.7, moving_cost * 1.5, moving_cost),
                stress_change=ImpactRange(0.05, 0.2, 0.1),
                satisfaction_change=ImpactRange(-0.1, 0.2, 0.05),
                delay_months=1
            ),
            tags=["lifestyle", "relocation"]
        )

    @staticmethod
    def have_child() -> Scenario:
        """Template: Having a child."""
        return Scenario.create(
            name="Have a Child",
            description="Start or expand family with a new child",
            decision_type=DecisionType.HAVE_CHILD,
            impacts=ScenarioImpact(
                monthly_expense_change=ImpactRange(800, 2000, 1200),
                one_time_cost=ImpactRange(5000, 15000, 8000),
                work_hours_change=ImpactRange(-10, 0, -5),  # May reduce work
                leisure_change=ImpactRange(-15, -5, -10),
                stress_change=ImpactRange(0.1, 0.3, 0.2),
                satisfaction_change=ImpactRange(0.0, 0.3, 0.15),
                energy_change=ImpactRange(-0.2, -0.05, -0.1)
            ),
            tags=["family", "major-life-event"]
        )

    @staticmethod
    def education(
        program_cost: float = 20000,
        duration_months: int = 24,
        salary_increase_after: float = 0.25
    ) -> Scenario:
        """Template: Pursue additional education."""
        return Scenario.create(
            name="Pursue Education",
            description="Go back to school or get certification",
            decision_type=DecisionType.EDUCATION,
            impacts=ScenarioImpact(
                one_time_cost=ImpactRange(
                    program_cost * 0.8,
                    program_cost * 1.2,
                    program_cost
                ),
                work_hours_change=ImpactRange(-10, 0, -5),
                leisure_change=ImpactRange(-10, -5, -8),
                stress_change=ImpactRange(0.05, 0.2, 0.1),
                skill_growth=ImpactRange(0.15, 0.35, 0.25),
                delay_months=duration_months
            ),
            conditional_impacts={
                "after_completion": ScenarioImpact(
                    income_change=ImpactRange(
                        1.0 + salary_increase_after * 0.5,
                        1.0 + salary_increase_after * 1.5,
                        1.0 + salary_increase_after
                    ),
                    job_satisfaction_change=ImpactRange(0.05, 0.2, 0.1)
                )
            },
            tags=["education", "career", "investment"]
        )

    @staticmethod
    def major_purchase(
        purchase_cost: float = 30000,
        monthly_payment: float = 500,
        satisfaction_boost: float = 0.1
    ) -> Scenario:
        """Template: Major purchase (car, etc.)."""
        return Scenario.create(
            name="Major Purchase",
            description="Make a significant purchase",
            decision_type=DecisionType.MAJOR_PURCHASE,
            impacts=ScenarioImpact(
                one_time_cost=ImpactRange(
                    purchase_cost * 0.9,
                    purchase_cost * 1.1,
                    purchase_cost
                ),
                monthly_expense_change=ImpactRange(
                    monthly_payment * 0.9,
                    monthly_payment * 1.1,
                    monthly_payment
                ),
                satisfaction_change=ImpactRange(
                    satisfaction_boost * 0.5,
                    satisfaction_boost * 1.2,
                    satisfaction_boost
                )
            ),
            tags=["financial", "lifestyle"]
        )

    @staticmethod
    def retirement(
        retirement_savings: float = 500000,
        monthly_retirement_income: float = 3000
    ) -> Scenario:
        """Template: Retirement."""
        return Scenario.create(
            name="Retire",
            description="Exit the workforce and begin retirement",
            decision_type=DecisionType.RETIREMENT,
            impacts=ScenarioImpact(
                income_change=ImpactRange(0.0, 0.0, 0.0),  # No employment income
                work_hours_change=ImpactRange(-40, -30, -40),
                leisure_change=ImpactRange(20, 40, 30),
                stress_change=ImpactRange(-0.3, -0.1, -0.2),
                satisfaction_change=ImpactRange(0.0, 0.3, 0.15),
                commute_change=ImpactRange(-5, 0, -5)
            ),
            tags=["retirement", "major-life-event"]
        )

    @staticmethod
    def lifestyle_change(
        expense_change: float = 0,
        stress_reduction: float = 0.1,
        satisfaction_boost: float = 0.1
    ) -> Scenario:
        """Template: General lifestyle change."""
        return Scenario.create(
            name="Lifestyle Change",
            description="Make changes to daily lifestyle habits",
            decision_type=DecisionType.LIFESTYLE_CHANGE,
            impacts=ScenarioImpact(
                monthly_expense_change=ImpactRange(
                    expense_change * 0.8,
                    expense_change * 1.2,
                    expense_change
                ),
                stress_change=ImpactRange(
                    -stress_reduction * 1.2,
                    -stress_reduction * 0.5,
                    -stress_reduction
                ),
                satisfaction_change=ImpactRange(
                    satisfaction_boost * 0.5,
                    satisfaction_boost * 1.5,
                    satisfaction_boost
                )
            ),
            tags=["lifestyle", "wellness"]
        )
