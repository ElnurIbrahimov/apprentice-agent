# Meta-Cognitive Self-Improvement for AURA

## Concrete Architecture Designs

**Date**: February 2026
**Scope**: Strategy performance tracking, self-modifying system prompts, reasoning template libraries

---

## Executive Summary

AURA has multiple reasoning engines (CognitiveTheater, MCTS, chain-of-thought, debate, etc.) but no mechanism to learn which works best for which problem types. This research covers three interconnected capabilities:

1. **Strategy Bandit** - Adaptive strategy selection using Thompson Sampling
2. **Prompt Evolution Engine** - Self-modifying system prompts via DSPy + Constitutional AI
3. **Reasoning Pattern Store** - Template library inspired by Voyager/Mem^p

These form a unified meta-cognitive loop: the bandit selects strategies, templates guide reasoning, and prompts evolve based on accumulated performance data.

---

## 1. Strategy Performance Tracking

### The Core Problem

AURA has multiple reasoning engines but no mechanism to learn which works best for which problem types. The goal is a meta-reasoning layer that adaptively routes problems to the best-performing strategy.

### Key Research Foundations

- **SMART: Self-learning Meta-strategy Agent for Reasoning Tasks** (Oct 2024) - Models strategy selection as a Markov Decision Process. The agent internalizes outcomes of its own reasoning and adjusts strategy without external feedback. Open-source at github.com/kumar-shridhar/SMART. Achieved +15 points on GSM8K.
  - Paper: https://arxiv.org/abs/2410.16128

- **SYMBOLIC-MoE: Adaptive Skill-based Routing** (Mar 2025) - Symbolic routing mechanism that dynamically selects among specialized expert models based on skills required by each query.
  - Paper: https://arxiv.org/html/2503.05641v1

- **Group Thompson Sampling for LLM Reasoning** (2025) - Replaces single-point value estimates with value distributions (Uncertainty-Aware Value Models) and uses Thompson Sampling to select candidates.
  - Paper: https://arxiv.org/html/2502.11155

- **Multi-Armed Bandits Meet Large Language Models** (2025) - Demonstrates how LLMs can be meta-bandit agents, acquiring meta-policies capable of exploring novel environments.
  - Paper: https://arxiv.org/html/2505.13355v1

### Scoring "Success" Without Ground Truth

Four concrete proxy metrics to combine:

**A. Self-Consistency Score**
Run the same problem through the chosen strategy K times (K=3-5). Measure agreement across outputs. Grounded in SelfCheckGPT and Reasoning-Aware Self-Consistency (RASC).

**B. LLM-as-Judge with Rubric**
Use a separate Ollama model to evaluate output on binary criteria (Pass/Fail). Binary evaluations are significantly more reliable than numeric scales.

**C. Causal Stepwise Evaluation (CaSE)**
Score each reasoning step based only on its preceding context. Catches incoherent reasoning chains even when the final answer looks plausible.
- Paper: https://arxiv.org/html/2510.20603v1

**D. User/Environment Feedback**
Downstream signals: user acceptance, clarification requests, retries, thumbs-up/down, tool execution success.

### Data Schema

```json
{
  "request_id": "uuid",
  "timestamp": "iso8601",
  "problem_fingerprint": {
    "category": "math|code|analysis|creative|planning|debug",
    "complexity_estimate": 0.0-1.0,
    "keywords": ["recursion", "optimization"],
    "embedding": [float; 384]
  },
  "strategy_used": "chain_of_thought|tree_of_thought|mcts|debate|cognitive_theater",
  "strategy_params": {},
  "outcome": {
    "self_consistency_score": 0.0-1.0,
    "judge_score": 0|1,
    "stepwise_coherence": 0.0-1.0,
    "user_feedback": null|"accept"|"reject"|"retry",
    "tool_success": null|true|false,
    "latency_ms": 4500,
    "token_cost": 2340
  },
  "composite_reward": 0.0-1.0
}
```

### The Bandit Algorithm (Thompson Sampling)

```python
import random

# Per (problem_category, strategy) pair, maintain:
#   alpha: count of successes + 1 (prior)
#   beta:  count of failures + 1 (prior)

def select_strategy(problem_category, available_strategies, arm_stats):
    samples = {}
    for strategy in available_strategies:
        alpha = arm_stats[(problem_category, strategy)]["alpha"]
        beta  = arm_stats[(problem_category, strategy)]["beta"]
        samples[strategy] = random.betavariate(alpha, beta)

    # Epsilon-exploration: 10% chance of random strategy for cold-start
    if random.random() < 0.1:
        return random.choice(available_strategies)

    return max(samples, key=samples.get)

def update_arm(problem_category, strategy, reward, arm_stats):
    if reward > 0.5:
        arm_stats[(problem_category, strategy)]["alpha"] += 1
    else:
        arm_stats[(problem_category, strategy)]["beta"] += 1
```

### Feedback Loop Timing

- **Immediate (seconds):** Stepwise coherence, latency, token cost
- **Short-delay (minutes):** Self-consistency score (K parallel traces)
- **Delayed (hours/days):** User feedback, downstream task success
- **Periodic (weekly):** Decay old observations with half-life for adaptation

### Composite Reward Weighting

- Self-consistency: 0.3
- Judge score: 0.3
- Stepwise coherence: 0.2
- Latency penalty: 0.1 (normalized)
- User feedback: 0.1 (when available, upweight to 0.3 and renormalize)

---

## 2. Self-Modifying System Prompts

### Key Research Foundations

- **DSPy MIPROv2** - Jointly optimizes instructions and few-shot examples via Bayesian Optimization. Works natively with Ollama.
  - Docs: https://dspy.ai/api/optimizers/MIPROv2/

- **TextGrad** (Nature, 2024) - Backpropagates textual feedback to optimize prompts. +20% on LeetCode-Hard.
  - GitHub: https://github.com/zou-group/textgrad

- **Godel Agent** (Oct 2024) - Self-referential framework where agent modifies its own prompts. +11% on MGSM.
  - Paper: https://arxiv.org/abs/2410.04444

- **Constitutional AI Self-Improvement** - Critique-revise loop for prompt refinement.
  - Paper: https://arxiv.org/abs/2212.08073

### Prompt Performance Log

```json
{
  "prompt_version": "v23",
  "prompt_hash": "sha256:...",
  "prompt_text": "You are AURA, a reasoning engine that...",
  "module": "planner|reasoner|critic|tool_selector",
  "period_start": "2025-01-01",
  "period_end": "2025-01-07",
  "metrics": {
    "total_invocations": 342,
    "avg_composite_reward": 0.72,
    "avg_user_satisfaction": 0.81,
    "failure_categories": {
      "hallucination": 12,
      "incomplete_reasoning": 8,
      "wrong_tool_selection": 5,
      "verbose_unhelpful": 15
    },
    "example_failures": [
      {"input": "...", "output": "...", "failure_type": "hallucination"}
    ],
    "example_successes": [
      {"input": "...", "output": "...", "reward": 0.95}
    ]
  }
}
```

### Three-Stage Prompt Modification Pipeline

**Stage 1: Critique (Constitutional AI Loop)**
Weekly or every N invocations, run a critique pass analyzing failures and successes against principles.

**Stage 2: Revise (TextGrad-inspired)**
Generate 3 candidate revisions addressing the critique. Preserve instructions that correlate with successes.

**Stage 3: Evaluate (DSPy-style Bayesian Search)**
Test candidates against held-out problems. Accept only if mean improves by >1 std AND no failure category regresses by >20%.

### Safety Mechanisms

- Never deploy untested prompts (minimum 20 held-out examples)
- Keep last 5 versions for rollback
- Rate-limit: max 1 change per module per week
- Shadow mode burn-in before auto-deployment

### DSPy Integration for Ollama

```python
import dspy

lm = dspy.LM('ollama_chat/llama3.1', api_base='http://localhost:11434')
dspy.configure(lm=lm)

class AURAReasoner(dspy.Signature):
    """Given a problem, produce step-by-step reasoning and answer."""
    problem = dspy.InputField(desc="The user's problem or question")
    reasoning = dspy.OutputField(desc="Step-by-step reasoning chain")
    answer = dspy.OutputField(desc="Final answer")

from dspy.teleprompt import MIPROv2
optimizer = MIPROv2(metric=your_composite_metric, num_candidates=10)
optimized_reasoner = optimizer.compile(
    AURAReasoner(),
    trainset=your_training_examples
)
```

---

## 3. Reasoning Template Library

### Key Research Foundations

- **Voyager Skill Library** (NeurIPS 2023) - Stores executable code as skills, indexed by description. Skills are compositional and transfer to new environments.
  - Website: https://voyager.minedojo.org/

- **Mem^p: Exploring Agent Procedural Memory** (Aug 2025) - Distills past trajectories into fine-grained instructions AND higher-level abstractions. Migration from stronger to weaker models yields substantial gains.
  - Paper: https://arxiv.org/abs/2508.06433

- **MemSkill: Learning and Evolving Memory Skills** (Feb 2026) - Reframes memory operations as learnable skills with controller-executor-designer architecture.
  - Paper: https://arxiv.org/abs/2602.02474

- **Contextual Experience Replay (CER)** (ICLR/ACL 2025) - Each experience = dynamics + skills. 51% relative improvement over GPT-4o on WebArena.
  - Paper: https://openreview.net/forum?id=RXvFK5dnpz

### Data Schema

**Raw Trace (per successful reasoning chain):**

```json
{
  "trace_id": "uuid",
  "problem": "original user input",
  "problem_embedding": [384 floats],
  "problem_category": "optimization",
  "strategy_used": "mcts",
  "full_trace": [
    {"step": 1, "type": "decompose", "content": "Break into sub-problems: ..."},
    {"step": 2, "type": "analogize", "content": "This is similar to ..."},
    {"step": 3, "type": "solve_sub", "content": "For sub-problem A: ..."},
    {"step": 4, "type": "synthesize", "content": "Combining results: ..."},
    {"step": 5, "type": "verify", "content": "Checking: ..."}
  ],
  "composite_reward": 0.92,
  "user_feedback": "accept"
}
```

**Abstracted Template (generated from traces):**

```json
{
  "template_id": "uuid",
  "name": "decompose-analogize-solve-synthesize-verify",
  "description": "Break complex optimization problems into sub-problems...",
  "abstract_steps": [
    {"step": 1, "type": "decompose", "instruction": "Identify independent sub-problems"},
    {"step": 2, "type": "analogize", "instruction": "Find similar solved cases"},
    {"step": 3, "type": "solve_sub", "instruction": "Solve each using analogous approach"},
    {"step": 4, "type": "synthesize", "instruction": "Combine, check for conflicts"},
    {"step": 5, "type": "verify", "instruction": "Verify against original constraints"}
  ],
  "applicable_categories": ["optimization", "planning", "multi-step-math"],
  "source_traces": ["trace_id_1", "trace_id_2"],
  "embedding": [384 floats],
  "performance": {
    "times_used": 47,
    "avg_reward_when_used": 0.85,
    "avg_reward_baseline": 0.71
  }
}
```

### Template Lifecycle

1. **Collect** high-reward traces (composite_reward > 0.8)
2. **Abstract** patterns using LLM (replace specifics with general placeholders)
3. **Deduplicate** via embedding similarity (>0.85 = merge)
4. **Deprecate** underperformers (avg_reward drops below baseline over 20+ uses)

### Compositionality (Voyager-inspired)

Templates at multiple granularity levels:
- **Atomic steps:** Single reasoning moves (analogize, enumerate, contrapositive)
- **Patterns:** Sequences of 3-7 steps solving a class of problems
- **Meta-patterns:** High-level strategies composing patterns

### Inference-Time Template Retrieval

```python
def enhance_reasoning_with_templates(problem, problem_embedding, category):
    candidates = retrieve_templates(
        embedding=problem_embedding,
        category=category,
        top_k=3,
        min_reward=0.7
    )
    if not candidates:
        return None

    template_guidance = format_template_guidance(candidates[0])

    augmented_prompt = f"""
    For this type of problem, a successful reasoning pattern is:
    {template_guidance}

    Use this as guidance but adapt freely.

    Problem: {problem}
    """
    return augmented_prompt
```

---

## 4. Unified SQLite Schema (`aura_meta.db`)

```sql
-- Strategy Bandit arms
CREATE TABLE strategy_arms (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    problem_category TEXT NOT NULL,
    strategy_name TEXT NOT NULL,
    alpha REAL DEFAULT 1.0,
    beta REAL DEFAULT 1.0,
    total_uses INTEGER DEFAULT 0,
    avg_reward REAL DEFAULT 0.0,
    last_updated TEXT,
    UNIQUE(problem_category, strategy_name)
);

-- Per-invocation outcomes
CREATE TABLE strategy_outcomes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id TEXT UNIQUE NOT NULL,
    timestamp TEXT NOT NULL,
    problem_category TEXT NOT NULL,
    complexity_estimate REAL,
    strategy_used TEXT NOT NULL,
    self_consistency_score REAL,
    judge_score INTEGER,
    stepwise_coherence REAL,
    user_feedback TEXT,
    tool_success INTEGER,
    latency_ms INTEGER,
    token_cost INTEGER,
    composite_reward REAL NOT NULL,
    problem_embedding BLOB
);

-- Versioned system prompts
CREATE TABLE prompt_versions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    module TEXT NOT NULL,
    version INTEGER NOT NULL,
    prompt_text TEXT NOT NULL,
    prompt_hash TEXT NOT NULL,
    created_at TEXT NOT NULL,
    is_active INTEGER DEFAULT 0,
    total_invocations INTEGER DEFAULT 0,
    avg_composite_reward REAL,
    avg_user_satisfaction REAL,
    failure_counts TEXT,  -- JSON
    UNIQUE(module, version)
);

-- Prompt evolution log
CREATE TABLE prompt_evolution_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    module TEXT NOT NULL,
    old_version INTEGER,
    new_version INTEGER,
    change_type TEXT,  -- critique, revise, rollback
    change_reason TEXT,
    critique_text TEXT,
    timestamp TEXT NOT NULL
);

-- Raw successful reasoning traces
CREATE TABLE reasoning_traces (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trace_id TEXT UNIQUE NOT NULL,
    problem TEXT NOT NULL,
    problem_category TEXT,
    strategy_used TEXT,
    full_trace TEXT NOT NULL,  -- JSON array of steps
    composite_reward REAL NOT NULL,
    user_feedback TEXT,
    problem_embedding BLOB,
    created_at TEXT NOT NULL
);

-- Abstracted reasoning templates
CREATE TABLE reasoning_templates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    template_id TEXT UNIQUE NOT NULL,
    name TEXT NOT NULL,
    description TEXT NOT NULL,
    abstract_steps TEXT NOT NULL,  -- JSON array
    applicable_categories TEXT,    -- JSON array
    source_trace_ids TEXT,         -- JSON array
    embedding BLOB,
    times_used INTEGER DEFAULT 0,
    avg_reward_when_used REAL DEFAULT 0.0,
    avg_reward_baseline REAL DEFAULT 0.0,
    status TEXT DEFAULT 'active',  -- active, deprecated, archived
    created_at TEXT NOT NULL,
    last_used TEXT
);

-- Performance tracking
CREATE TABLE performance_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    metric_type TEXT NOT NULL,
    metric_value REAL NOT NULL,
    context TEXT  -- JSON with additional metadata
);
```

---

## 5. Python Class Skeleton: StrategyBandit

```python
class StrategyBandit:
    """Multi-armed bandit for adaptive reasoning strategy selection.

    Uses Thompson Sampling over Beta distributions per (category, strategy) pair.
    Integrates with brain.py for model selection and metacog_guardian for monitoring.
    """

    def __init__(self, db_path: str = "data/aura_meta.db"):
        self.db_path = db_path
        self.epsilon = 0.1  # exploration rate
        self._init_db()
        self._load_arms()

    def select_strategy(self, problem: str, category: str,
                        available_strategies: list) -> str:
        """Select best strategy via Thompson Sampling."""
        # 1. Classify problem if category not provided
        # 2. Sample from Beta distribution per strategy
        # 3. Epsilon-greedy exploration for cold-start
        # 4. Return selected strategy name
        pass

    def record_outcome(self, request_id: str, strategy: str,
                       category: str, outcome: dict):
        """Record strategy outcome and update arms."""
        # 1. Compute composite reward from outcome dict
        # 2. Update arm alpha/beta
        # 3. Log to strategy_outcomes table
        # 4. Check if trace qualifies for template extraction
        pass

    def compute_composite_reward(self, outcome: dict) -> float:
        """Weighted combination of proxy metrics."""
        weights = {
            'self_consistency': 0.3,
            'judge_score': 0.3,
            'stepwise_coherence': 0.2,
            'latency_penalty': 0.1,
            'user_feedback': 0.1,
        }
        # Compute weighted sum, handle missing values
        pass

    def decay_arms(self, half_life_days: int = 30):
        """Apply temporal decay to old observations."""
        pass

    def get_arm_stats(self, category: str = None) -> dict:
        """Get current arm statistics for display."""
        pass
```

### Integration Points with AURA

1. **brain.py `_select_model_for_complexity()`** - StrategyBandit feeds into model selection
2. **brain.py `think()`** - Strategy selection happens before reasoning
3. **metacog_guardian.py** - Monitors bandit performance, alerts on degradation
4. **inner_monologue.py** - Provides self-consistency evaluation data

---

## 6. Unified Architecture: How All Three Systems Interconnect

```
INCOMING PROBLEM
       |
       v
[Problem Classifier] -- embeds problem, categorizes it
       |
       v
[Template Retriever] -- finds relevant reasoning templates (System 3)
       |
       v
[Strategy Bandit] -- selects strategy via Thompson Sampling (System 1)
       |                uses template availability as a feature
       v
[Active System Prompt] -- loaded from prompt store (System 2)
       |                    augmented with template guidance
       v
[Reasoning Engine] -- CognitiveTheater / MCTS / CoT / Debate / etc.
       |
       v
[Output + Trace]
       |
       +---> [Reward Computation] -- self-consistency, judge, CaSE, latency
       |            |
       |            +---> Update Strategy Bandit arms (System 1)
       |            +---> Append to prompt performance log (System 2)
       |            +---> If high reward: extract template candidate (System 3)
       |
       v
[Response to User]
       |
       +---> [Delayed Feedback] -- user accept/reject/retry
                    |
                    +---> Update all three systems
```

### Timing of Learning Loops

- Strategy bandit: updates after every invocation (real-time)
- Template extraction: runs every 100 high-reward traces (batch)
- Prompt evolution: runs weekly or every 500 invocations (slow, deliberate)

### Cold Start

- Initialize bandit arms with uniform priors (alpha=1, beta=1)
- Start with no templates
- Use hand-crafted initial system prompts
- After ~50-100 interactions per category: meaningful bandit preferences
- After ~200-500 high-reward interactions: useful template library

---

## 7. Implementation Priority

| Priority | Component | Effort | Impact |
|----------|-----------|--------|--------|
| 1 | Strategy Bandit + SQLite schema | 2 weeks | High - immediate adaptive routing |
| 2 | Composite reward computation | 1 week | High - enables all learning |
| 3 | Template extraction pipeline | 2 weeks | Medium - grows with data |
| 4 | Prompt evolution engine | 3 weeks | Medium - requires most data |

**Recommended start:** Bandit + schema first. Templates once data accumulates. Prompts last.

---

## 8. References

- SMART: https://arxiv.org/abs/2410.16128
- SYMBOLIC-MoE: https://arxiv.org/html/2503.05641v1
- Group Thompson Sampling: https://arxiv.org/html/2502.11155
- Multi-Armed Bandits + LLMs: https://arxiv.org/html/2505.13355v1
- TextGrad (Nature): https://github.com/zou-group/textgrad
- DSPy MIPROv2: https://dspy.ai/api/optimizers/MIPROv2/
- DSPy BetterTogether: https://dspy.ai/api/optimizers/BetterTogether/
- Godel Agent: https://arxiv.org/abs/2410.04444
- Constitutional AI: https://arxiv.org/abs/2212.08073
- Voyager Skill Library: https://voyager.minedojo.org/
- Mem^p Procedural Memory: https://arxiv.org/abs/2508.06433
- MemSkill: https://arxiv.org/abs/2602.02474
- Contextual Experience Replay (CER): https://openreview.net/forum?id=RXvFK5dnpz
- Self-Improving Agents (Yohei Nakajima): https://yoheinakajima.com/better-ways-to-build-self-improving-ai-agents/
- SelfCheckGPT: https://arxiv.org/html/2502.06233v1
- LLM-as-Judge Best Practices: https://mer.vin/2025/11/llm-as-a-judge-best-practices-for-consistent-evaluation/
- CaSE Stepwise Evaluation: https://arxiv.org/html/2510.20603v1
- Episodic Memory for LLM Agents: https://arxiv.org/pdf/2502.06975
- SMART GitHub: https://github.com/kumar-shridhar/SMART/
