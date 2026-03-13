"""
GEPA Evolution Types — Data structures for skill evolution.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from enum import Enum
import hashlib
import json


class CandidateSelectionStrategy(Enum):
    PARETO = "pareto"
    CURRENT_BEST = "current_best"
    EPSILON_GREEDY = "epsilon_greedy"


class ComponentSelectionMode(Enum):
    ROUND_ROBIN = "round_robin"
    ALL = "all"


@dataclass
class Trajectory:
    """Full execution trace of a skill run."""
    skill_id: str
    task_input: str
    task_output: str
    score: float
    success: bool
    error: Optional[str] = None
    reasoning_steps: List[str] = field(default_factory=list)
    tools_called: List[str] = field(default_factory=list)
    memory_retrieved: List[str] = field(default_factory=list)
    intermediate_outputs: List[str] = field(default_factory=list)
    execution_time_ms: float = 0.0
    evaluator_feedback: Optional[str] = None

    def summary(self) -> str:
        """Compact summary for LLM reflection."""
        parts = [f"Input: {self.task_input[:200]}"]
        parts.append(f"Output: {self.task_output[:300]}")
        parts.append(f"Score: {self.score:.2f} | Success: {self.success}")
        if self.error:
            parts.append(f"Error: {self.error}")
        if self.tools_called:
            parts.append(f"Tools: {', '.join(self.tools_called)}")
        if self.evaluator_feedback:
            parts.append(f"Feedback: {self.evaluator_feedback}")
        return "\n".join(parts)


@dataclass
class EvalExample:
    """A single evaluation task."""
    id: str
    task_input: str
    expected_behavior: str  # Rubric, not exact output
    source: str = "synthetic"  # synthetic, golden, session

    def cache_key(self) -> str:
        return hashlib.sha256(
            f"{self.id}:{self.task_input}".encode()
        ).hexdigest()[:16]


@dataclass
class Candidate:
    """A snapshot of skill texts being evolved."""
    id: int
    components: Dict[str, str]  # skill_id -> procedure text
    parent_id: int = -1  # -1 = root/seed
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Scores per eval example
    scores: Dict[str, float] = field(default_factory=dict)  # example_id -> score
    objective_scores: Dict[str, float] = field(default_factory=dict)

    @property
    def avg_score(self) -> float:
        if not self.scores:
            return 0.0
        return sum(self.scores.values()) / len(self.scores)

    def cache_key(self) -> str:
        content = json.dumps(self.components, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]


@dataclass
class ReflectionItem:
    """One failure case formatted for LLM reflection."""
    component_id: str
    current_text: str
    task_input: str
    task_output: str
    score: float
    error: Optional[str]
    feedback: Optional[str]
    trajectory_summary: str


@dataclass
class GEPAConfig:
    """Configuration for a GEPA optimization run."""
    # Budget
    max_iterations: int = 10
    max_metric_calls: int = 150
    timeout_seconds: int = 600  # 10 min default

    # Models (Ollama endpoints)
    reflection_model: str = "qwen3:8b"  # Strong model for mutation proposals
    eval_model: str = "qwen2.5-coder:7b"  # Fast model for evaluation

    # Reflection
    minibatch_size: int = 3
    candidate_strategy: CandidateSelectionStrategy = CandidateSelectionStrategy.PARETO
    component_mode: ComponentSelectionMode = ComponentSelectionMode.ROUND_ROBIN
    epsilon: float = 0.1  # For epsilon-greedy

    # Merge
    use_merge: bool = True
    max_merges: int = 5
    merge_interval: int = 3  # Every N iterations

    # Constraints
    max_skill_chars: int = 15000
    max_growth_ratio: float = 1.2  # 20% growth limit

    # Stopping
    no_improvement_patience: int = 4
    score_threshold: float = 0.95

    # Storage
    run_dir: str = "./aura_data/evolution_runs"

    # Ollama
    ollama_base_url: str = "http://localhost:11434"


@dataclass
class GEPAResult:
    """Result of a GEPA optimization run."""
    best_candidate: Candidate
    all_candidates: List[Candidate]
    iterations_run: int
    total_evals: int
    pareto_front: Dict[str, int]  # example_id -> best candidate_id
    improvement: float  # best_score - seed_score
    duration_seconds: float
    stop_reason: str
