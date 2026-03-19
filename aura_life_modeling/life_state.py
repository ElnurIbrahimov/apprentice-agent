"""
AURA Predictive Life Modeling - Life State Model

Represents the user's current life state across all domains.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum


class LifeDomain(Enum):
    """Life domains for modeling."""
    FINANCIAL = "financial"
    CAREER = "career"
    HEALTH = "health"
    RELATIONSHIPS = "relationships"
    PERSONAL = "personal"
    TIME = "time"


@dataclass
class FinancialState:
    """Financial domain state."""
    monthly_income: float = 0.0
    monthly_expenses: float = 0.0
    savings: float = 0.0
    investments: float = 0.0
    debt: float = 0.0
    emergency_fund_months: float = 0.0
    income_sources: List[Dict[str, float]] = field(default_factory=list)
    expense_categories: Dict[str, float] = field(default_factory=dict)
    currency: str = "USD"

    @property
    def monthly_surplus(self) -> float:
        """Calculate monthly surplus."""
        return self.monthly_income - self.monthly_expenses

    @property
    def net_worth(self) -> float:
        """Calculate net worth."""
        return self.savings + self.investments - self.debt

    @property
    def runway_months(self) -> float:
        """Calculate financial runway in months.

        NOTE: Assumes savings and emergency_fund_months are separate pools.
        If savings already includes the emergency fund, this double-counts.
        """
        if self.monthly_expenses <= 0:
            return float('inf')
        return self.savings / self.monthly_expenses + self.emergency_fund_months


@dataclass
class CareerState:
    """Career domain state."""
    current_role: str = ""
    company: str = ""
    industry: str = ""
    years_experience: float = 0.0
    skills: List[str] = field(default_factory=list)
    satisfaction_score: float = 0.5  # 0-1
    growth_potential: float = 0.5    # 0-1
    job_security: float = 0.5        # 0-1
    work_life_balance: float = 0.5   # 0-1
    network_strength: float = 0.5    # 0-1
    is_employed: bool = True
    remote_work: bool = False


@dataclass
class HealthState:
    """Health domain state."""
    physical_health: float = 0.7     # 0-1
    mental_health: float = 0.7       # 0-1
    energy_level: float = 0.7        # 0-1
    stress_level: float = 0.3        # 0-1 (lower is better)
    sleep_quality: float = 0.7       # 0-1
    exercise_frequency: int = 3      # days per week
    chronic_conditions: List[str] = field(default_factory=list)
    age: int = 30


@dataclass
class RelationshipState:
    """Relationships domain state."""
    relationship_status: str = "single"  # single, dating, married, divorced
    has_children: bool = False
    num_children: int = 0
    family_support: float = 0.5      # 0-1
    friend_network: float = 0.5      # 0-1
    social_satisfaction: float = 0.5  # 0-1
    dependents: int = 0


@dataclass
class PersonalState:
    """Personal domain state."""
    life_satisfaction: float = 0.6   # 0-1
    purpose_clarity: float = 0.5     # 0-1
    values: List[str] = field(default_factory=list)
    goals: List[str] = field(default_factory=list)
    hobbies: List[str] = field(default_factory=list)
    location: str = ""
    cost_of_living_index: float = 1.0


@dataclass
class TimeState:
    """Time allocation state (hours per week)."""
    work_hours: float = 40.0
    commute_hours: float = 5.0
    sleep_hours: float = 49.0        # 7 hrs/day
    family_hours: float = 20.0
    leisure_hours: float = 20.0
    learning_hours: float = 5.0
    exercise_hours: float = 3.0

    @property
    def total_allocated(self) -> float:
        """Total allocated hours per week."""
        return (self.work_hours + self.commute_hours + self.sleep_hours +
                self.family_hours + self.leisure_hours + self.learning_hours +
                self.exercise_hours)

    @property
    def free_hours(self) -> float:
        """Free hours per week (168 hours total)."""
        return max(0, 168 - self.total_allocated)


@dataclass
class LifeState:
    """Complete life state model."""
    user_id: str
    timestamp: datetime
    financial: FinancialState = field(default_factory=FinancialState)
    career: CareerState = field(default_factory=CareerState)
    health: HealthState = field(default_factory=HealthState)
    relationships: RelationshipState = field(default_factory=RelationshipState)
    personal: PersonalState = field(default_factory=PersonalState)
    time: TimeState = field(default_factory=TimeState)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "user_id": self.user_id,
            "timestamp": self.timestamp.isoformat(),
            "financial": {
                "monthly_income": self.financial.monthly_income,
                "monthly_expenses": self.financial.monthly_expenses,
                "savings": self.financial.savings,
                "investments": self.financial.investments,
                "debt": self.financial.debt,
                "emergency_fund_months": self.financial.emergency_fund_months,
                "income_sources": self.financial.income_sources,
                "expense_categories": self.financial.expense_categories,
                "currency": self.financial.currency,
                "net_worth": self.financial.net_worth,
                "monthly_surplus": self.financial.monthly_surplus,
                "runway_months": self.financial.runway_months
            },
            "career": {
                "current_role": self.career.current_role,
                "company": self.career.company,
                "industry": self.career.industry,
                "years_experience": self.career.years_experience,
                "skills": self.career.skills,
                "satisfaction_score": self.career.satisfaction_score,
                "growth_potential": self.career.growth_potential,
                "job_security": self.career.job_security,
                "work_life_balance": self.career.work_life_balance,
                "network_strength": self.career.network_strength,
                "is_employed": self.career.is_employed,
                "remote_work": self.career.remote_work
            },
            "health": {
                "physical_health": self.health.physical_health,
                "mental_health": self.health.mental_health,
                "energy_level": self.health.energy_level,
                "stress_level": self.health.stress_level,
                "sleep_quality": self.health.sleep_quality,
                "exercise_frequency": self.health.exercise_frequency,
                "chronic_conditions": self.health.chronic_conditions,
                "age": self.health.age
            },
            "relationships": {
                "relationship_status": self.relationships.relationship_status,
                "has_children": self.relationships.has_children,
                "num_children": self.relationships.num_children,
                "family_support": self.relationships.family_support,
                "friend_network": self.relationships.friend_network,
                "social_satisfaction": self.relationships.social_satisfaction,
                "dependents": self.relationships.dependents
            },
            "personal": {
                "life_satisfaction": self.personal.life_satisfaction,
                "purpose_clarity": self.personal.purpose_clarity,
                "values": self.personal.values,
                "goals": self.personal.goals,
                "hobbies": self.personal.hobbies,
                "location": self.personal.location,
                "cost_of_living_index": self.personal.cost_of_living_index
            },
            "time": {
                "work_hours": self.time.work_hours,
                "commute_hours": self.time.commute_hours,
                "sleep_hours": self.time.sleep_hours,
                "family_hours": self.time.family_hours,
                "leisure_hours": self.time.leisure_hours,
                "learning_hours": self.time.learning_hours,
                "exercise_hours": self.time.exercise_hours,
                "total_allocated": self.time.total_allocated,
                "free_hours": self.time.free_hours
            }
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "LifeState":
        """Deserialize from dictionary."""
        financial = FinancialState(
            monthly_income=data.get("financial", {}).get("monthly_income", 0),
            monthly_expenses=data.get("financial", {}).get("monthly_expenses", 0),
            savings=data.get("financial", {}).get("savings", 0),
            investments=data.get("financial", {}).get("investments", 0),
            debt=data.get("financial", {}).get("debt", 0),
            emergency_fund_months=data.get("financial", {}).get("emergency_fund_months", 0),
            income_sources=data.get("financial", {}).get("income_sources", []),
            expense_categories=data.get("financial", {}).get("expense_categories", {}),
            currency=data.get("financial", {}).get("currency", "USD")
        )

        career = CareerState(
            current_role=data.get("career", {}).get("current_role", ""),
            company=data.get("career", {}).get("company", ""),
            industry=data.get("career", {}).get("industry", ""),
            years_experience=data.get("career", {}).get("years_experience", 0),
            skills=data.get("career", {}).get("skills", []),
            satisfaction_score=data.get("career", {}).get("satisfaction_score", 0.5),
            growth_potential=data.get("career", {}).get("growth_potential", 0.5),
            job_security=data.get("career", {}).get("job_security", 0.5),
            work_life_balance=data.get("career", {}).get("work_life_balance", 0.5),
            network_strength=data.get("career", {}).get("network_strength", 0.5),
            is_employed=data.get("career", {}).get("is_employed", True),
            remote_work=data.get("career", {}).get("remote_work", False)
        )

        health = HealthState(
            physical_health=data.get("health", {}).get("physical_health", 0.7),
            mental_health=data.get("health", {}).get("mental_health", 0.7),
            energy_level=data.get("health", {}).get("energy_level", 0.7),
            stress_level=data.get("health", {}).get("stress_level", 0.3),
            sleep_quality=data.get("health", {}).get("sleep_quality", 0.7),
            exercise_frequency=data.get("health", {}).get("exercise_frequency", 3),
            chronic_conditions=data.get("health", {}).get("chronic_conditions", []),
            age=data.get("health", {}).get("age", 30)
        )

        relationships = RelationshipState(
            relationship_status=data.get("relationships", {}).get("relationship_status", "single"),
            has_children=data.get("relationships", {}).get("has_children", False),
            num_children=data.get("relationships", {}).get("num_children", 0),
            family_support=data.get("relationships", {}).get("family_support", 0.5),
            friend_network=data.get("relationships", {}).get("friend_network", 0.5),
            social_satisfaction=data.get("relationships", {}).get("social_satisfaction", 0.5),
            dependents=data.get("relationships", {}).get("dependents", 0)
        )

        personal = PersonalState(
            life_satisfaction=data.get("personal", {}).get("life_satisfaction", 0.6),
            purpose_clarity=data.get("personal", {}).get("purpose_clarity", 0.5),
            values=data.get("personal", {}).get("values", []),
            goals=data.get("personal", {}).get("goals", []),
            hobbies=data.get("personal", {}).get("hobbies", []),
            location=data.get("personal", {}).get("location", ""),
            cost_of_living_index=data.get("personal", {}).get("cost_of_living_index", 1.0)
        )

        time = TimeState(
            work_hours=data.get("time", {}).get("work_hours", 40),
            commute_hours=data.get("time", {}).get("commute_hours", 5),
            sleep_hours=data.get("time", {}).get("sleep_hours", 49),
            family_hours=data.get("time", {}).get("family_hours", 20),
            leisure_hours=data.get("time", {}).get("leisure_hours", 20),
            learning_hours=data.get("time", {}).get("learning_hours", 5),
            exercise_hours=data.get("time", {}).get("exercise_hours", 3)
        )

        timestamp = data.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        elif timestamp is None:
            timestamp = datetime.now()

        return cls(
            user_id=data.get("user_id", "default"),
            timestamp=timestamp,
            financial=financial,
            career=career,
            health=health,
            relationships=relationships,
            personal=personal,
            time=time
        )

    def get_domain(self, domain: LifeDomain) -> Any:
        """Get state for a specific domain."""
        mapping = {
            LifeDomain.FINANCIAL: self.financial,
            LifeDomain.CAREER: self.career,
            LifeDomain.HEALTH: self.health,
            LifeDomain.RELATIONSHIPS: self.relationships,
            LifeDomain.PERSONAL: self.personal,
            LifeDomain.TIME: self.time
        }
        return mapping[domain]

    def compute_wellbeing_score(self) -> float:
        """Compute overall wellbeing score (0-1)."""
        weights = {
            "financial_security": 0.15,
            "career_satisfaction": 0.15,
            "health": 0.20,
            "relationships": 0.15,
            "life_satisfaction": 0.20,
            "time_balance": 0.15
        }

        # Financial security (runway + surplus)
        financial_security = min(1.0, self.financial.runway_months / 12) * 0.5
        financial_security += (1.0 if self.financial.monthly_surplus > 0 else 0.3) * 0.5

        # Career satisfaction
        career_satisfaction = (
            self.career.satisfaction_score * 0.4 +
            self.career.work_life_balance * 0.3 +
            self.career.growth_potential * 0.3
        )

        # Health composite
        health = (
            self.health.physical_health * 0.3 +
            self.health.mental_health * 0.3 +
            (1 - self.health.stress_level) * 0.2 +
            self.health.energy_level * 0.2
        )

        # Relationships
        relationships = (
            self.relationships.family_support * 0.4 +
            self.relationships.friend_network * 0.3 +
            self.relationships.social_satisfaction * 0.3
        )

        # Time balance (penalize overwork)
        time_balance = 1.0 - min(1.0, max(0, self.time.work_hours - 40) / 40)
        time_balance = time_balance * 0.5 + (self.time.leisure_hours / 30) * 0.5

        score = (
            weights["financial_security"] * financial_security +
            weights["career_satisfaction"] * career_satisfaction +
            weights["health"] * health +
            weights["relationships"] * relationships +
            weights["life_satisfaction"] * self.personal.life_satisfaction +
            weights["time_balance"] * min(1.0, time_balance)
        )

        return round(score, 3)
