# FluxMind
## Self-Improving Neural Architecture with Latent Program Induction

---

## What is FluxMind?

FluxMind is an AI that **writes its own internal programs**. Unlike traditional neural networks that learn fixed input-output mappings, FluxMind generates latent instructions during execution and runs them to produce outputs.

### The Core Idea

```
Traditional Neural Net:
    Input → [Fixed Weights] → Output

FluxMind:
    Input → [Generate Latent Program] → [Execute Program] → Output
                     ↑
           Model invents its own
           reasoning on the fly
```

---

## Original Vision

> **Ultra-Adaptive Learning**: It doesn't just learn from text or images—it evolves its own learning strategies based on hardware availability.

> **Low-Energy, High-IQ Mode**: Think of it as the Zen monk of AI. While GPTs are chugging electricity, FluxMind quietly sips power, then drops an insight.

> **Memory on the Fly**: It keeps just the "juice"—the nuggets of wisdom—and forgets the fluff.

> **Self-Improving Creativity**: It could invent new types of "reasoning circuits" on the go.

---

## What Has Been Achieved

### Phase 1: Adaptive Routing (v0.53-v0.67) ✅

| Version | Achievement |
|---------|-------------|
| v0.53 | Basic module routing works |
| v0.55 | Clean specialization emerges |
| v0.60 | Step conditioning (different modules per step) |
| v0.65 | Procedural vs episodic cognition |
| v0.66 | Budget-conditioned routing |
| v0.67 | Procedural generalization (98.7% at 3× length) |

**Key Finding**: Modules naturally specialize. Architecture alignment matters more than raw capacity.

### Phase 2: Efficient Cognition (v0.68-v0.69) ✅

| Version | Achievement |
|---------|-------------|
| v0.68 | Per-step compute scaling (avg 0.028 compute used) |
| v0.69 | Long sequence handling (91.6% at 128 steps = 8× training) |

**Key Finding**: Model chooses minimal compute and doesn't over-compress already-efficient state.

### Phase 3: Latent Program Induction (v0.70) ✅ BREAKTHROUGH

**The model writes its own internal programs.**

| Version | Result |
|---------|--------|
| v0.70 | Halting-based → Failed (collapsed to 1 step) |
| v0.70b | Pre-computed latents → Failed (random guessing) |
| v0.70c | Curriculum → Partial (errors compound) |
| v0.70d | Autoregressive → Working! |
| v0.70f | Simplified proof-of-concept → 96.7% |
| **v0.70g2** | **3 vars, 6 ops → 99.6% exact match** |

**Key Finding**: Latent instructions must be generated DURING execution (autoregressive), not pre-computed.

### Phase 4: Scaling (v0.74) ✅

| Version | Config | Accuracy | Params |
|---------|--------|----------|--------|
| v0.73 | 3×6 | 99.5% | 156K |
| **v0.74.2** | **4×8** | **99.7%** | **442K** |

**Key Finding**: Phased training required at scale (train each DSL separately, then joint fine-tune).

### Phase 5: Compositional OOD (v0.75) ✅

Tested out-of-distribution generalization on compositional tasks.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    FluxMind Architecture                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  For each execution step t:                                 │
│                                                             │
│    current_state ──┐                                        │
│                    ├──► LatentGenerator ──► latent_t        │
│    operation_t ────┘          │                             │
│                               ▼                             │
│                 LatentExecutor(state_t, latent_t)           │
│                               │                             │
│                               ▼                             │
│                          state_{t+1}                        │
│                               │                             │
│                               ▼                             │
│                     OutputPredictor ──► output_t            │
│                                                             │
└─────────────────────────────────────────────────────────────┘

KEY INSIGHT: Latent instructions are generated DURING execution,
allowing the model to adapt its "internal program" based on 
what's actually happening at runtime.
```

---

## Scaling Law (Empirical)

| Task Complexity | Parameters Needed | Result |
|-----------------|-------------------|--------|
| 2 vars, 4 ops | ~50K | 96.7% |
| 3 vars, 6 ops | ~190K | 99.6% |
| 4 vars, 8 ops | ~450K | 99.7% |

**Rule of thumb**: ~2.5-3× parameters per 50% task complexity increase.

---

## Current Status

### Completed ✅
- Adaptive routing that specializes automatically
- Efficient computation that scales with task difficulty  
- Memory handling for long sequences
- **Latent program induction** (model writes own programs)
- Scaling to 4×8 complexity

### Next Step 🔄
**v0.70j: Conditionals**
- IF_X_GT_Y_ADD_Z operations
- Runtime branching based on state values
- Code exists: `FluxMind_v070j_conditionals.py`

### Roadmap 📋
- v0.71: Confidence/metacognition (model knows when it's uncertain)
- v0.72: Transfer tests (do learned strategies generalize?)
- v0.73: Multi-task learning

---

## Key Files

### Documentation
| File | Description |
|------|-------------|
| `FluxMind_Vision_and_Roadmap.md` | **Master vision document** |
| `FluxMind_v070_Series_Documentation.md` | Latent program induction experiments |
| `FluxMind_v074_Series_Documentation.md` | Scaling experiments |
| `FluxMind_v075_Results.md` | Compositional OOD results |

### Code
| File | Description |
|------|-------------|
| `FluxMind_v070j_conditionals.py` | **← CONTINUE FROM HERE** |
| `FluxMind_v071_confidence.py` | Confidence/metacognition |
| `FluxMind_v0751_compositional_ood.py` | Compositional OOD |
| `FluxMind_v065_complete.py` | Procedural cognition |
| `FluxMind_v067d_spending.py` | Budget routing |

---

## Research Findings

### What Works
1. **Autoregressive latent generation** - Generate during execution, not before
2. **Phased training at scale** - Train DSLs separately, then combine
3. **Module specialization** - Emerges naturally without explicit supervision
4. **Budget-conditioned routing** - Model uses minimal necessary compute

### What Doesn't Work
1. **Pre-computed latent plans** - Can't handle conditionals
2. **Halting mechanisms with length penalty** - Collapses before learning task
3. **Op permutation rebinding at 4×8** - Latent space too entangled

### Documented Limitations
1. **SWAP operations** - Non-local permutations need attention (violates FluxMind philosophy)
2. **State-range OOD** - Model overconfident, needs calibration training
3. **Op rebinding at scale** - Works at 3×6, fails at 4×8

---

## What FluxMind is NOT

❌ Standard attention/transformer  
❌ Meta-learning from support sets  
❌ Hand-coded reasoning circuits  
❌ DSL inference from examples  

## What FluxMind IS

✅ Latent program induction  
✅ Model generates its own internal programs  
✅ Autoregressive execution  
✅ Self-improving reasoning  
✅ Emergent abstractions  

---

## Getting Started

1. Read `FluxMind_Vision_and_Roadmap.md`
2. Understand `FluxMind_v070_Series_Documentation.md`
3. Continue development from `FluxMind_v070j_conditionals.py`

---

*Real FluxMind progress through v0.75*
*Next milestone: v0.70j Conditionals*
