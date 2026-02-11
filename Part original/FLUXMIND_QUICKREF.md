# FluxMind Quick Reference Card

## WHAT IT IS

**FluxMind generates its own internal programs during execution.**

```
Input → [Model Generates Latent Program] → [Executes It] → Output
```

NOT meta-learning. NOT attention over examples. NOT transformers.

---

## CURRENT STATUS

| Milestone | Version | Accuracy | Status |
|-----------|---------|----------|--------|
| Adaptive Routing | v0.67 | 98.7% | ✅ Done |
| Efficient Compute | v0.68 | - | ✅ Done |
| Long Sequences | v0.69 | 91.6% @8× | ✅ Done |
| **Latent Programs** | **v0.70g2** | **99.6%** | **✅ Done** |
| Scaling 4×8 | v0.74 | 99.7% | ✅ Done |
| Compositional OOD | v0.75 | - | ✅ Done |
| **Conditionals** | **v0.70j** | - | **🔄 NEXT** |

---

## KEY INSIGHT

Latent instructions must be generated **DURING** execution (autoregressive), not pre-computed. This allows runtime adaptation.

---

## NEXT STEP

**File**: `FluxMind_v070j_conditionals.py`

**Goal**: IF_X_GT_Y_ADD_Z operations

---

## IMPORTANT FILES

| Purpose | File |
|---------|------|
| Vision | `FluxMind_Vision_and_Roadmap.md` |
| v0.70 Details | `FluxMind_v070_Series_Documentation.md` |
| Continue From | `FluxMind_v070j_conditionals.py` |

---

## WRONG DIRECTION (DELETE)

- Everything v0.79-v0.86
- Everything "MetaFluxMind"
- All `meta_fluxmind*.py`
- All `train_meta*.py`
- All `*hybrid*.py`

---

## PARAMETERS

| Config | Params | Accuracy |
|--------|--------|----------|
| 2×4 | ~50K | 96.7% |
| 3×6 | ~190K | 99.6% |
| 4×8 | ~450K | 99.7% |

**Scaling**: ~2.5-3× params per complexity tier

---

## THE VISION

> "Self-Improving Creativity: It invents new types of reasoning circuits on the go."

**v0.70g2 PROVED THIS** - Model writes its own internal programs, latent structure emerges naturally.
