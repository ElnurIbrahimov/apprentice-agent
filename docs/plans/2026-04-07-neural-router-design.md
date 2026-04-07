# Neural Router — Intelligent Model Routing for Aura

**Date:** 2026-04-07
**Status:** Approved
**Author:** Claude + Elnur

## Problem

Aura has 14 cloud models + 5 local models but routing is a glorified lookup table:

1. `detect_action_mode()` burns an LLM call (5s!) just to classify intent
2. `ACTION_MODE_MODELS` maps classifications to hardcoded model names
3. `_select_model()` in brain uses word-count heuristics + keyword matching
4. `RoutingStats` (UCB bandit) exists but is dormant — agent_service sets `_model_override` which blocks it
5. No cost/speed awareness, no conversation memory, no learning from feedback
6. Binary System 1/System 2 with nothing in between for 14+ models

**Result:** Three disconnected layers, the fancy stuff never fires, effective routing is keyword → lookup table.

## Solution: Three-Layer Neural Router

### Design Principles

- **<5ms routing overhead** (down from 5,000ms)
- **Learns from every interaction** — regenerations, model switches, ratings, timing
- **Conversation-aware** — tracks complexity trends, code mode, topic drift, satisfaction
- **Three override levels** — Auto / Preference tier / Explicit model
- **Only routes Ollama models** — ChatGPT OAuth + direct API providers are manual picks
- **Benchmark-seeded** — good from day one, gets better over time
- **Human-editable** — all profiles, weights, and learning data in readable JSON files

---

## Layer 1: Instant Classifier (<5ms)

### Feature Extractor (pure Python, <0.1ms)

Extracts from raw prompt:
- `word_count` — len(words)
- `code_ratio` — % of tokens that are code-like (brackets, semicolons, camelCase, snake_case)
- `question_marks` — count
- `has_attachment` — image/file in payload
- `language_markers` — detected programming language keywords
- `imperative_score` — starts with action verb

### Task Dimension Scorer (<1ms)

Outputs a 6-dimension **task need vector**:

```python
{
    code: 0.0-1.0,      # how much coding ability is needed
    reason: 0.0-1.0,    # how much reasoning depth is needed
    speed: 0.0-1.0,     # how much the user needs a fast response
    context: 0.0-1.0,   # how much context window is needed
    quality: 0.0-1.0,   # how important output quality is
    vision: 0.0-1.0,    # whether multimodal is needed
}
```

Rules (deterministic, no ML):
- Short messages (<=10 words) → speed_need = 0.9
- Long messages (>50 words) → reason_need boosted, speed reduced
- Code ratio > 0.15 or language keywords → code_need = 0.9
- Complex patterns ("comprehensive analysis", "step by step") → reason_need = 0.8
- Attachment present → vision_need = 1.0
- Recent regeneration in conversation → quality_need boosted

### First-Message Exception

First message of a conversation also computes a nomic-embed-text embedding (~3ms) stored as the conversation's topic anchor for drift detection. Total first-message budget: ~5ms. Subsequent messages: <1ms from cache.

---

## Layer 2: Profile Matcher (<1ms)

### Model Capability Profiles

Every Ollama model gets a 6-dimension profile, seeded from benchmarks:

| Model | code | reason | speed | context | quality | vision |
|-------|------|--------|-------|---------|---------|--------|
| nemotron-3-super | 0.60 | 0.73 | 1.00 | 1.00 | 0.70 | 0 |
| kimi-k2.5 | 0.77 | 0.92 | 0.08 | 0.26 | 0.90 | 1 |
| glm-5 | 0.78 | 0.96 | 0.15 | 0.20 | 0.95 | 0 |
| glm-5.1 | 0.88 | 0.93 | 0.10 | 0.20 | 0.92 | 0 |
| qwen3.5:397b | 0.70 | 0.88 | 0.19 | 0.26 | 0.88 | 1 |
| qwen3.5 | 0.50 | 0.83 | 0.30 | 0.26 | 0.75 | 1 |
| deepseek-v3.2 | 0.68 | 0.85 | 0.11 | 0.13 | 0.82 | 0 |
| gemma4:31b | 0.72 | 0.85 | 0.23 | 0.26 | 0.83 | 1 |
| minimax-m2.7 | 0.78 | 0.80 | 0.22 | 0.21 | 0.86 | 0 |
| minimax-m2.5 | 0.80 | 0.74 | 0.20 | 0.20 | 0.82 | 0 |
| qwen3-coder:480b | 0.70 | 0.60 | 0.35 | 0.26 | 0.75 | 0 |
| qwen3-coder-next | 0.71 | 0.55 | 0.38 | 0.26 | 0.72 | 0 |
| gpt-oss:120b | 0.45 | 0.90 | 0.15 | 0.13 | 0.80 | 0 |
| gemma4:e4b | 0.35 | 0.69 | 0.50 | 0.13 | 0.60 | 1 |
| gemma4:e2b | 0.25 | 0.60 | 0.60 | 0.13 | 0.50 | 1 |

Profiles stored in `data/model_profiles.json`. Benchmark-seeded, learning-adjusted over time.

### Matching Algorithm

```python
def match(task_needs, user_preference, available_models):
    # 1. Apply preference tier weights
    weights = PREFERENCE_WEIGHTS[user_preference]  # fast/balanced/quality
    
    # 2. Weighted dot product per model
    scores = {}
    for model in available_models:
        profile = get_profile(model)
        score = sum(task_needs[d] * profile[d] * weights[d] for d in DIMENSIONS)
        score += routing_stats.get_bonus(model, task_category)  # UCB exploration
        scores[model] = score
    
    # 3. Hard filters
    if task_needs.vision > 0.5:
        scores = {m: s for m, s in scores.items() if profiles[m].vision > 0}
    if task_needs.context > 0.8:
        scores = {m: s for m, s in scores.items() if profiles[m].context > 0.5}
    
    return max(scores, key=scores.get)
```

### Override Hierarchy

| Level | User action | Router behavior |
|-------|------------|----------------|
| Auto | No model picked | Full 3-layer routing |
| Preference | "prefer fast" / "balanced" / "prefer quality" | Router runs with shifted weights |
| Explicit | Specific model picked | Router bypassed, 0ms overhead |

### Per-Feature Defaults

Each feature maps to a default task need vector (e.g. "code" panel → code_need=0.9). Layer 1 overrides these when the actual prompt says otherwise.

---

## Layer 3: Conversation Context Tracker

### Conversation Profile

```python
@dataclass
class ConversationProfile:
    topic_embedding: list[float]     # nomic-embed-text anchor
    topic_drift_score: float = 0.0
    complexity_history: list[float]  # last 10 messages
    complexity_trend: float = 0.0    # positive = escalating
    code_ratio_history: list[float]
    in_code_mode: bool = False
    models_used: list[str]
    last_model: str = None
    regen_count: int = 0
    model_switches: int = 0
    avg_response_gap_ms: float = 0
    thumbs: list[int] = []
    total_tokens: int = 0
    turn_count: int = 0
```

### Influence on Routing

| Signal | Adjustment |
|--------|-----------|
| `in_code_mode = True` | code_need += 0.3 |
| `complexity_trend > 0.3` | reason_need += 0.2, speed_need -= 0.2 |
| `complexity_trend < -0.3` | speed_need += 0.2 |
| `regen_count > 0` | quality_need += 0.3 |
| `model_switches > 0` | Learn from what user switched to |
| `total_tokens > 100K` | Hard boost context_need |
| `topic_drift_score > 0.6` | Reset code_mode, recompute topic |

### Model Stickiness

Prefer keeping the same model within a conversation. Switch only when:
- Context threshold crossed
- Complexity escalates significantly (3+ messages trending up)
- User regenerated
- Topic drifted hard (cosine similarity < 0.4 with anchor)
- Code mode entered/exited

When switching, only escalate within the same dimension.

### Background Updates

After each response: update profile (~1ms background task). Check topic drift every 3rd turn. Profile is ready before next message arrives.

---

## Learning Loop

### Six Feedback Signals

| Signal | Detection | Weight | Updates |
|--------|-----------|--------|---------|
| Regeneration | User clicks regen | -0.3 | Model profile down for task dimensions |
| Model switch | User picks different model | -0.2 old / +0.1 new | Preference shift |
| Response gap | Time to next user message | +/-0.05 | Subtle quality signal |
| Explicit rating | Thumbs up/down, star vote | +/-0.4 | Strongest signal |
| Conversation length | Turns without regen/switch | +0.02/turn | Slow positive signal |
| Abort/interrupt | New message while streaming | -0.15 | Too slow or off-track |

### Profile Evolution

```python
def update_profile(model, task_dimensions, feedback_score):
    profile = get_profile(model)
    learning_rate = 0.05
    for dim in DIMENSIONS:
        if task_dimensions[dim] > 0.3:
            profile[dim] += learning_rate * feedback_score * task_dimensions[dim]
            profile[dim] = clamp(0.0, 1.0)
    save_profile(model)
```

### Exploration

UCB bonus from RoutingStats (already exists). MIN_SAMPLES raised from 5 → 15 for 14 cloud models.

### Decay

14-day half-life. Last week = full weight. Last month = 25%. Allows quick learning about new models.

### Storage

- `data/model_profiles.json` — capability profiles
- `data/classifier_weights.json` — tunable thresholds
- `data/routing_learning.jsonl` — feedback history (with rotation)

---

## Integration

### WebSocket Payload (backward compatible)

```json
{
  "type": "chat",
  "message": "...",
  "routing": {
    "model": null,
    "preference": "balanced",
    "feature": "chat",
    "conversation_id": "abc-123"
  }
}
```

Old format (`model: "some-model"`) still works as explicit override.

### Extension UI

- Preference tier selector: Fast / Balanced / Quality (3 buttons at top of Models panel)
- ModelPill shows: `Auto → nemotron-3-super` with routing reason
- Per-feature dropdowns stay as explicit override

### CLI

- `--preference fast|balanced|quality` flag
- `/model` picker = explicit override
- `/routing` command = show conversation profile, last decision, learning stats
- Status line: `[nemotron-3-super · fast · 449t/s]`

### Response Metadata

```json
{
  "routing": {
    "model_used": "nemotron-3-super:cloud",
    "reason": "short_query + prefer_fast",
    "task_dimensions": {"code": 0.0, "reason": 0.2, "speed": 0.9},
    "alternatives": ["kimi-k2.5:cloud", "glm-5:cloud"],
    "conversation_turn": 3
  }
}
```

---

## What Gets Replaced

| Deleted | Replacement |
|---------|------------|
| `detect_action_mode()` + LLM classifier | Layer 1 instant classifier |
| `ACTION_MODE_MODELS` hardcoded dict | Model profiles + dot product |
| `_is_complex_query()` patterns | Layer 1 feature extractor |
| `_should_escalate_to_system2()` + neuromodulators | Continuous scoring across all dimensions |
| `_select_model()` in model_router_mixin | 3-layer router |
| `MODEL_FAST/REASON/CODE` as routing targets | Profiles (roles stay as human aliases) |

## What Stays

| Kept | Why |
|------|-----|
| `_get_client_for_model()` | Client resolution orthogonal to routing |
| `RoutingStats` + UCB | Core of learning loop, expanded |
| `set_model_override()` | Explicit override path |
| Config roles + chains | Human-readable config, fallback generation |
| Per-feature dropdowns | Explicit override UI |

## New Files

```
aura/routing/
    __init__.py
    classifier.py        # Layer 1: features + task scoring
    matcher.py           # Layer 2: profile matching
    conversation.py      # Layer 3: conversation tracker
    profiles.py          # Model profile management
    learning.py          # Feedback processing + updates
    router.py            # Entry point: route() → model

data/
    model_profiles.json
    classifier_weights.json
    routing_learning.jsonl
```

## Call Chain

```
Message arrives
  → router.route(prompt, routing_opts, conversation_id)
      → IF explicit model → return (0ms)
      → Layer 1: classifier.score(prompt) → task_needs (1ms)
      → Layer 3: conversation.adjust(task_needs) → adjusted (0ms cached)
      → Layer 2: matcher.match(adjusted, preference) → model (0.5ms)
      → return model
  → _get_client_for_model(model) → (client, model)
  → client.chat(model, messages)
  → Background: learning.record(model, task_needs, conv_id)
  → On feedback: learning.update(signal, model, task_needs)
```

**Total routing: <5ms. Down from 5,000ms.**
