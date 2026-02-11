# BroadMind: Research Document
## Latent Program Induction for Self-Improving Neural Computation

---

## 1. Concept Statement

### 1.1 Vision

BroadMind is a neural architecture that **generates and executes its own internal programs at runtime**. Unlike transformers, which apply fixed learned weight matrices to input sequences, BroadMind synthesizes latent instructions during forward execution and runs them to produce outputs. The model does not memorize input-output mappings; it constructs a procedural computation on the fly, conditioned on the current state and task context.

The original vision describes five core properties:

1. **Ultra-Adaptive Learning**: The system evolves its own learning and inference strategies based on available hardware and task complexity, rather than running a fixed computation graph.
2. **Low-Energy, High-IQ Mode**: Computation is proportional to problem difficulty. Easy problems consume minimal resources; hard problems receive deeper processing. No wasted compute.
3. **Memory on the Fly**: The system retains compressed essence ("wisdom") from experience and discards raw examples. It operates on distilled procedural knowledge, not memorized data.
4. **Physical Integration**: The architecture is small enough (~445K parameters, <2MB FP32) to run on microcontrollers, edge devices, and phones -- AI that does not require a data center.
5. **Self-Improving Creativity**: The system invents new reasoning circuits at runtime. Its internal programs occupy a continuous latent space, enabling smooth interpolation between known strategies and generation of novel ones.

### 1.2 Scientific Formulation

Formally, BroadMind implements a **latent program induction** framework. Given an initial state $s_0 \in \mathbb{R}^d$ and a program $P = (o_1, o_2, \ldots, o_T)$ consisting of a sequence of operation tokens, the model produces a final state $s_T$ through iterative application of a generate-execute loop:

For each step $t = 1, \ldots, T$:
1. **Generate**: $z_t = G_\theta(s_{t-1}, o_t, t, w)$ where $z_t \in \mathbb{R}^{d_z}$ is a latent instruction, $w \in \mathbb{R}^{d_w}$ is a task-specific wisdom code, and $G_\theta$ is the latent generator network.
2. **Execute**: $s_t = s_{t-1} + E_\phi(s_{t-1}, z_t, c(s_{t-1}))$ where $E_\phi$ is the latent executor and $c(s_{t-1})$ extracts comparison features from the current state.
3. **Halt**: $h_t = H_\psi(P, t)$ where $H_\psi$ predicts whether computation should terminate at step $t$.

The latent instruction $z_t$ is the critical differentiator: it is a real-valued vector that the model produces autoregressively during execution, not a pre-computed plan or a retrieved template. This enables the model to adapt its internal program based on the runtime state, supporting conditional logic, state-dependent branching, and compositional generalization.

### 1.3 What BroadMind Is Not

| BroadMind is NOT | BroadMind IS |
|---|---|
| A transformer (no self-attention) | A recurrent latent-program executor |
| Meta-learning from support sets | Wisdom distillation from compressed experience |
| Fixed computation per input | Adaptive compute (halter + mixture of recursions) |
| Pattern matching / memorization | Runtime program synthesis in continuous space |
| Retrieval-augmented generation | Essence-based reasoning from distilled knowledge |
| Symbolic program synthesis | Continuous latent program induction |

---

## 2. Architecture (Current State: v0.76)

### 2.1 System Overview

```
Input: initial_state (x, y, z) + program [op1, op2, ..., opN]
                |
    [WisdomMatcher] -- encode (state, first_op) -> query
                    -- dot-product similarity to WisdomBank -> softmax -> weighted wisdom
                |
                v
    wisdom_code (48-dim compressed task knowledge)
                |
    For each step t = 1..N:
        |
        [LatentGenerator]
            inputs: state_enc + op_emb + sinusoidal_step_enc + wisdom_enc + recursion_depth_enc
            output: latent instruction z_t (96-dim)
        |
        [RecursionRouter] -- (state_enc + op_enc) -> Gumbel-Softmax -> depth in {1,2,3,4}
        |
        For r = 1..depth:
            [LatentExecutor]
                inputs: state_enc + latent_enc + comparison_features
                output: delta (residual state update)
            state = state + delta
        |
        [Halter] -- (program_enc + sinusoidal_step_enc) -> sigmoid -> halt probability
        |
        if halt_prob > 0.5: STOP
                |
    Output: predicted final state via linear predictor
```

### 2.2 Component Inventory

| Component | Parameters | Function |
|---|---|---|
| **WisdomDistiller** | ~75K | Compress N (state, next_state, op) transitions into a 48-dim code |
| **WisdomBank** | 240 | Store 5 x 48-dim wisdom slots (one per task family + spare) |
| **WisdomMatcher** | ~40K | Route new problems to the correct wisdom via learned similarity |
| **Solver** (LatentGenerator + LatentExecutor) | ~280K | Generate and execute latent instructions per step |
| **RecursionRouter** | ~20K | Select inner recursion depth (1-4) per step |
| **Halter** | ~30K | Decide when to stop across program steps |
| **Total** | **~445K** | |

### 2.3 Key Design Decisions

**Autoregressive latent generation**: Latent instructions must be generated DURING execution, conditioned on the current state. Pre-computed latent plans fail because they cannot handle state-dependent conditionals (e.g., IF x > y THEN z += 1). This was validated empirically: v0.70b (pre-computed latents) achieved random-chance accuracy, while v0.70d (autoregressive) achieved 96.7%.

**Residual state updates**: The executor predicts a delta added to the current state ($s_t = s_{t-1} + \Delta$). This makes identity operations trivial (predict zero delta) and stabilizes training by constraining the output space.

**Explicit comparison features**: For conditional operations, the executor receives explicit boolean comparisons $(x > y, y > z, z > x)$ and normalized differences $((x-y)/30, (y-z)/30, (z-x)/30)$ alongside the latent instruction. This gives the network direct access to branching information without requiring it to rediscover comparison logic from raw state values.

**Sinusoidal step encodings**: Replacing learned positional embeddings with sinusoidal encodings (v0.75) enables generalization to program lengths never seen during training, as the encoding function is defined for any non-negative integer.

**Phased training**: Training all components simultaneously fails because objectives conflict (e.g., the solver and halter have competing loss landscapes). A strict 6-phase curriculum trains each component separately, then fine-tunes end-to-end.

### 2.4 Training Pipeline

| Phase | Duration | What Trains | What's Frozen | Key Detail |
|---|---|---|---|---|
| 1 | 2500 iter | Solver | All else | Zero-wisdom input; learn pure execution |
| 2 | 1500 iter | Distiller + Solver | Halter | Generate and store wisdom codes per family |
| 2b | 800 iter | Matcher + WisdomBank | Solver, Distiller, Halter | Learn to route problems to correct wisdom |
| 3 | 1500 iter | Halter | All else | Learn when to stop (BCE on halt signal) |
| 4 | 1500 iter | Everything | -- | End-to-end fine-tune; joint loss: task + 0.5*halt + 0.5*match + cost |
| 5 | 500 iter | Solver | All else | Gaussian noise (0.05->0.02) + consistency loss for length generalization |

### 2.5 Results Summary

| Metric | v0.74 | v0.75 | v0.76 |
|---|---|---|---|
| Parameters | 477K | ~421K | ~445K |
| In-distribution accuracy | 99.7% | 99%+ | 99%+ |
| Mixed-program accuracy | 99.7% | 99%+ | 99%+ |
| Wisdom matching | 100% | 100% | 100% |
| Adaptive step matching | Exact | Exact | Exact |
| Length 8 (OOD) | -- | 90%+ | 90%+ |
| Length 16 (OOD) | -- | 80%+ | 80%+ |
| Depth hierarchy | -- | -- | COMPARE > TRANSFER > ACC/DEC |

---

## 3. Related Work and Positioning

### 3.1 Latent Program Induction

**Latent Program Network (LPN)** (Macfarlane & Bonnet, NeurIPS 2025 Spotlight; 3rd place ARC Prize 2024) is the closest prior work. LPN learns a continuous latent space of implicit programs: an encoder maps input-output demonstration pairs into latent space, and a decoder executes the latent program on new inputs. At test time, gradient-based search refines the latent code to best explain observed I/O pairs. LPN doubles its out-of-distribution accuracy when test-time search is enabled.

**DreamCoder** (Ellis et al., PLDI 2021) uses wake-sleep Bayesian learning to build a library of symbolic program components. It operates in discrete symbolic space, not continuous latent space.

**Latent Execution for Neural Program Synthesis** (Meta AI, 2021) proposes executing programs in a latent (neural) space rather than symbolically, enabling gradient-based training of synthesis models.

**BroadMind's distinction**: BroadMind generates the latent program in a single forward pass without test-time search. LPN requires explicit gradient optimization at inference; DreamCoder requires symbolic search. BroadMind's latent programs are produced autoregressively as part of the forward computation, making inference cheap and deterministic. No existing work combines forward-pass latent program generation with adaptive computation depth and compressed task knowledge.

### 3.2 Adaptive Computation

**MIND** (ICLR 2025, Oral) uses an introspection network to decide whether to apply or skip fixed-point iteration layers. Achieves 96.62% on ImageNet with a 3-layer network surpassing ResNet-50.

**PonderNet** (DeepMind, 2021) reformulates adaptive halting as a probabilistic Markov Decision Process over recurrent steps. Fully differentiable and unbiased.

**LoopLM** (2025) scales looped transformers to 1.4B-2.6B parameters. A 1.4B LoopLM matches a 4B standard transformer (2-3x parameter efficiency) using entropy-regularized training with a uniform prior over exit steps.

**BroadMind's distinction**: In existing adaptive compute systems, the decision is "how much computation" but the computation itself is uniform (same layer applied repeatedly). BroadMind's computation is program-driven: each iteration can be qualitatively different because the latent program modulates what happens, not just how many times it happens. The Mixture of Recursions (v0.76) adds a second axis of adaptivity -- depth within each step -- orthogonal to the halter's across-step decision.

### 3.3 Knowledge Compression

**Function Vectors** (Todd et al., ICLR 2024) showed that in-context learning demonstrations compress into a compact "function vector" in transformer hidden states that can trigger task execution in zero-shot contexts.

**In-Context Vectors (ICV)** (Liu et al., 2024) extracts a single latent vector from demonstrations, then shifts LLM hidden states by this vector. Achieves 45% computation reduction.

**Dataset Distillation** (2025) achieves up to 170x compression of training data into synthetic latent codes.

**BroadMind's distinction**: Existing compressed representations are passive -- they steer activations but do not execute computation. BroadMind's wisdom codes are active: they parameterize the latent generator, directly controlling what program the solver produces. The compressed representation is an executable program specification, not a nudge vector. No existing work treats compressed knowledge as a generative program.

### 3.4 Length Generalization

**Looped Transformers for Length Generalization** (ICLR 2025) proves an L-layer transformer can be simulated by a looped model with L+O(1) layers. Adaptive loop steps improve generalization 3-5x beyond training lengths on arithmetic tasks.

**"Reasoning with Latent Thoughts"** (ICLR 2025) proves looped transformers implicitly generate "latent thoughts" and can simulate T steps of chain-of-thought with T loops. Important finding: looping helps reasoning but hurts memorization, suggesting beneficial regularization.

**BroadMind's distinction**: Existing looped transformers achieve length generalization through uniform weight sharing. BroadMind's latent program modulates what happens at each iteration, creating heterogeneous loop behavior. The program itself adapts to input length, providing a complementary mechanism beyond positional encoding and loop depth.

### 3.5 Mixture of Experts and Depths

**Mixture-of-Depths (MoD)** (Google DeepMind, 2024) enforces per-layer token budgets via top-k routing. Some tokens go through full processing; others skip.

**Mixture-of-Recursions (MoR)** (NeurIPS 2025) unifies parameter efficiency with adaptive compute. Routers assign different recursion depths to individual tokens.

**ReMoE** (ICLR 2025) replaces TopK+Softmax routing with continuous ReLU activation, making routing fully differentiable.

**BroadMind's distinction**: Every MoE/MoD/MoR variant routes tokens to pre-existing experts or layers. The expert bank is fixed at training time. BroadMind generates the "expert" (latent program) at runtime. There is no fixed expert bank. This could be framed as **"Mixture of Generated Experts"** -- the routing decision and the expert content are co-produced. This is a fundamentally different paradigm.

### 3.6 Non-Transformer Architectures

**Mamba/Mamba-2** (Gu and Dao, 2023/2024): Selective State Space Model with 4-5x inference throughput over transformers. Struggles with in-context learning and copying tasks.

**xLSTM** (NeurIPS 2024): Exponential gating with matrix memory. At 350M parameters, achieves lower perplexity than Llama and Mamba at comparable scale.

**Test-Time Training (TTT) Layers** (2024): The hidden state is itself a model, updated via self-supervised learning at test time. TTT-Linear matches Mamba speed; TTT-E2E (2025) scales to 3B parameters.

**RWKV v5/v6** (2024-2025): Linear attention RNN deployed to 1.5 billion Windows devices for Microsoft on-device Copilot.

**BroadMind's distinction**: BroadMind is architecture-agnostic -- the latent program induction mechanism could sit on top of any backbone. Mamba's documented weakness in in-context learning is precisely where latent program induction could provide an advantage, since the generated program explicitly encodes task structure. TTT learns at test time via gradient descent, while BroadMind generates programs in a single forward pass without gradient computation at inference.

---

## 4. Current Task Domain

### 4.1 Operation Vocabulary

BroadMind currently operates on a domain-specific language (DSL) with 3 integer variables $(x, y, z)$ and 13 operations across 4 families:

| Family | Operations | Semantics |
|---|---|---|
| ACCUMULATE | ACC_X, ACC_Y, ACC_Z | Increment: $x \leftarrow x + 1$ |
| TRANSFER | TRANSFER_XY, TRANSFER_YZ, TRANSFER_ZX | Conservation: $x \leftarrow x - 1, y \leftarrow y + 1$ |
| COMPARE | IF_X_GT_Y_INC_Z, IF_Y_GT_Z_INC_X, IF_Z_GT_X_INC_Y | Conditional: if $x > y$ then $z \leftarrow z + 1$ |
| DECREMENT | DEC_X, DEC_Y, DEC_Z | Decrement: $x \leftarrow x - 1$ |

Programs are sequences of 1-4 operations (training), tested up to 16 (OOD). Initial state values are integers in $[0, 10)$.

### 4.2 What This Domain Tests

Despite its simplicity, this DSL exercises several fundamental computational capabilities:

- **Sequential state tracking**: The model must maintain and update a multi-variable state across steps.
- **Conditional branching**: COMPARE operations require the model to evaluate boolean conditions on the current state and execute conditionally.
- **Conservation laws**: TRANSFER operations preserve the sum of variables, testing whether the model learns invariant structure.
- **Compositional execution**: Mixed-family programs require the model to compose operations from different families in arbitrary order.
- **Length generalization**: Testing at 4x the training length probes whether the learned execution strategy extrapolates.

### 4.3 Limitations of the Current Domain

- Fixed 3-variable state (no scaling to larger state dimensions)
- Integer arithmetic only (no floating-point, string, or structured data)
- No loops or recursion in the program language itself
- No memory beyond the current state (no stack, heap, or external memory)
- No I/O or environment interaction
- Deterministic execution (no stochastic programs)

---

## 5. Scaling Roadmap: From Toy to Real

### 5.1 The Scaling Challenge

The literature identifies a consistent pattern: accuracy degrades sharply as program complexity increases. On CRUXEval (3-13 line Python functions), GPT-4 achieves only 63-81% on output prediction. On ARC-AGI-2, top solutions reach 24-54%. The gap between BroadMind's 99%+ on 13 arithmetic ops and real-world program execution is substantial.

Five failure modes emerge when scaling from toy to real programs:

1. **Exponential search space**: The number of possible programs grows exponentially with language complexity.
2. **State tracking degradation**: Accuracy degrades as execution traces lengthen, particularly for recurrent architectures.
3. **Representation gap**: Continuous neural representations struggle to faithfully represent discrete symbolic programs.
4. **Compositional failure**: Models learn individual operations but fail on unseen compositions.
5. **Verification difficulty**: For real programs, checking correctness is itself computationally hard (unlike arithmetic where ground truth is trivially computable).

### 5.2 Proposed Scaling Phases

Based on the literature, we propose a five-phase scaling strategy:

#### Phase A: Compositional Depth (Current Priority)

**Goal**: Fix at 3 variables, scale from 2-operation compositions to 5, 10, 20 operations.

**Approach**:
- ExeDec-style execution decomposition: predict intermediate execution states as auxiliary training targets, not just final output.
- Inductive scratchpads: learn the single-step transition function rather than the full execution trace. This is what BroadMind already does, but making it explicit as a training objective could improve robustness.
- Combined curriculum: always mix shorter programs into longer-program training batches to prevent catastrophic forgetting.

**Success criteria**: 95%+ accuracy at 20-step programs, trained on 1-4.

#### Phase B: State Width

**Goal**: Scale variables from 3 to 5, 8, 16.

**Approach**:
- Structured positional encodings for variable slots (analogous to Abacus embeddings for digit positions).
- Modular state representation: encode each variable independently, then combine through interaction layers.
- The Globality Barrier (NeurIPS 2024) suggests that inductive state representations (learning single-step transitions) are critical for scaling, as holistic state representations hit an efficiency wall.

**Success criteria**: 95%+ accuracy with 8 variables, 13 operations, programs up to 8 steps.

#### Phase C: Operation Vocabulary Expansion

**Goal**: Expand from 13 arithmetic operations to include boolean, comparison, assignment, and basic data structure operations.

**Approach**:
- ExeDec's compositional generalization findings are critical: composing new operations with previously learned ones requires explicit decomposition.
- Phased introduction: add one operation family at a time, using the phased training pipeline that already works.
- Wisdom distillation naturally accommodates new families: add new wisdom slots as new operation families are introduced.

**Success criteria**: 90%+ accuracy with 30+ operations across 6+ families.

#### Phase D: Control Flow

**Goal**: Add conditionals with else-branches, bounded loops, and function calls.

**Approach**:
- This is the hardest scaling step. The program itself becomes variable-length and state-dependent.
- Looped transformer / recurrent-depth approaches may be necessary: the model needs to iterate its own computation adaptively based on loop bounds that depend on runtime state.
- The Mixture of Recursions (v0.76) is a proto-version of this: it already selects iteration depth based on operation complexity. Extending this to program-level control flow is the natural next step.

**Success criteria**: 85%+ accuracy on programs with bounded loops (up to 10 iterations) and conditional branches.

#### Phase E: Integration and Real-World Benchmarks

**Goal**: Evaluate on established benchmarks (Karel, simplified ARC tasks, CLRS algorithmic reasoning).

**Approach**:
- Use the learned execution engine as a reasoning module within a larger system.
- Hybrid neuro-symbolic approaches: use BroadMind for execution and a separate system for program search/synthesis.
- Test on ARC-like tasks where the "program" must be inferred from input-output examples, not given explicitly.

**Success criteria**: Competitive performance on at least one established benchmark.

### 5.3 Key Research Questions

1. **Does latent program induction scale?** The core bet is that generating programs in continuous latent space is more flexible than symbolic synthesis and more structured than pure neural computation. Does this advantage hold at scale?

2. **Can wisdom distillation handle hundreds of operation families?** The current 4-family, 5-slot wisdom bank works perfectly. Does it degrade gracefully or catastrophically as the number of families grows?

3. **Is autoregressive latent generation necessary at scale?** At 3 variables and 4 steps, autoregressive generation is clearly needed (pre-computed plans fail). At 16 variables and 100 steps, the cost of per-step generation may become prohibitive. Are there hybrid approaches (generate a partial plan, execute autoregressively)?

4. **How does MoR interact with program complexity?** The v0.76 depth hierarchy (COMPARE > TRANSFER > ACC) emerged naturally. Will this self-organization persist with more operation types, or will the router need explicit guidance?

---

## 6. Edge Deployment

### 6.1 Model Size

| Format | Model Size | RAM at Inference | Total Footprint |
|---|---|---|---|
| FP32 (current) | ~1.78 MB | ~500 KB | ~2.3 MB |
| INT8 (QAT) | ~445 KB | ~200 KB | ~645 KB |
| INT4 (mixed precision) | ~222 KB | ~150 KB | ~372 KB |

### 6.2 Target Platforms

| Platform | Processor | Memory | Expected Latency (INT8) | Framework |
|---|---|---|---|---|
| STM32H7 | Cortex-M7 @ 480MHz | 2MB Flash, 1MB SRAM | 2-5 ms | STM32Cube.AI or TFLite Micro |
| ESP32-S3 | Xtensa LX7 @ 240MHz | 16MB Flash, 512KB SRAM | 10-50 ms | ESP-DL |
| Raspberry Pi 5 | Cortex-A76 @ 2.4GHz | 4-16GB RAM | <0.5 ms | ExecuTorch or ONNX Runtime |
| iPhone / Android | NPU (up to 73 TOPS) | 6-16GB RAM | <0.2 ms | ExecuTorch / ONNX Runtime Mobile |

### 6.3 Deployment Strategy

```
PyTorch (FP32, ~1.78 MB)
    |
    v
Quantization-Aware Training (INT8, ~445 KB)
    |
    v
Export (ONNX or TorchScript)
    |
    +---> STM32Cube.AI or TFLite Micro  (MCU targets)
    +---> ESP-DL                         (ESP32 targets)
    +---> ExecuTorch                     (RPi, phones)
    +---> ONNX Runtime Mobile            (cross-platform)
```

### 6.4 Hardware-Adaptive Inference

For a single model that adapts to available hardware:

**MatFormer-style nested networks**: Structure the solver's FFN layers so that the first N neurons form a coherent sub-network. A single training run produces models at 25%, 50%, 75%, and 100% width. This directly maps to deployment: full model on phone, 50% on RPi, 25% on MCU.

**Early-exit with the existing Halter**: The halter already decides when to stop across steps. Extending this to allow early exit within the solver's layers (not just across program steps) would enable depth-adaptive inference.

---

## 7. Theoretical Positioning

### 7.1 BroadMind as a Computational Model

BroadMind can be understood as a differentiable implementation of a **register machine** operating in continuous space:

- **Registers**: The state vector $(x, y, z)$ serves as a fixed set of registers.
- **Instruction set**: The latent generator produces an instruction (96-dim vector) at each step.
- **Instruction decode/execute**: The latent executor interprets the instruction and applies a state update.
- **Program counter**: The step index $t$ and halter together implement a program counter with conditional halting.
- **Microcode**: The MoR inner recursion acts as microcode -- within each instruction, multiple sub-operations can execute with shared weights.

This analogy suggests a principled path to scaling: extend the register file (more variables), enrich the instruction set (more latent dimensions or structured latent codes), and add addressing modes (attention over variables).

### 7.2 Relationship to Universal Computation

**Looped transformers** have been proven capable of simulating arbitrary computations given sufficient depth/loops (ICML 2023). BroadMind's recurrent generate-execute loop is structurally similar, but with the critical addition that the "program" being executed is itself generated by the model. This creates a two-level computational hierarchy:

- **Level 1 (meta-computation)**: The latent generator decides WHAT to compute.
- **Level 2 (object-computation)**: The latent executor carries out the computation.

This separation of concerns is analogous to the fetch-decode-execute cycle in physical processors, and may explain why the architecture generalizes well: the executor learns a general-purpose state update mechanism, while the generator specializes in mapping task context to appropriate instructions.

### 7.3 Capacity-Generalization Tradeoff

ICLR 2025 work on provable generalization demonstrates that capacity-constrained models have an advantage for length and compositional generalization. BroadMind's ~445K parameter count is not a limitation but a feature: the model cannot memorize the exponential space of possible programs and is forced to learn general execution strategies. This aligns with the finding that looping helps reasoning but hurts memorization (Saunshi et al., ICLR 2025).

---

## 8. How to Proceed: Concrete Next Steps

### 8.1 Immediate (v0.77): Hardware-Adaptive Compute

**Goal**: One model, multiple deployment profiles. The model detects available resources and adjusts its own depth and width at inference time.

**Implementation plan**:
1. Restructure the solver's FFN layers using MatFormer-style nesting (25/50/75/100% width).
2. Add switchable LayerNorm per width configuration.
3. Train with stochastic depth (randomly drop layers during training).
4. Self-distillation: distill between full and minimal model during training.
5. Device profiling: 100ms micro-benchmark at load time to select configuration.

**Success criteria**: Full model matches v0.76 accuracy (>99%); 50%-width model at >95%; depth-2 model at >90%.

### 8.2 Near-term (v0.78): Edge Deployment

**Goal**: Run BroadMind on at least 3 hardware targets with measured benchmarks.

**Implementation plan**:
1. Quantization-Aware Training (INT8) with PyTorch's built-in QAT.
2. Export to ONNX format.
3. Deploy to Raspberry Pi 5 via ONNX Runtime (easiest target).
4. Deploy to STM32H7 via STM32Cube.AI (hardest target).
5. Deploy to ESP32-S3 via ESP-DL.
6. Measure and publish latency, power consumption, and accuracy numbers.

**Success criteria**: INT8 accuracy within 1% of FP32; <1ms on RPi5; <5ms on STM32H7.

### 8.3 Medium-term: Scaling to Real Complexity

Follow the scaling phases outlined in Section 5.2:

1. **Phase A**: 20-step programs (compositional depth)
2. **Phase B**: 8+ variables (state width)
3. **Phase C**: 30+ operations (vocabulary expansion)
4. **Phase D**: Bounded loops and branches (control flow)
5. **Phase E**: Benchmark evaluation (Karel, ARC, CLRS)

### 8.4 Long-term: "Evolves Its Own Learning Strategies"

This is the final unrealized element of the original vision. The model should adapt not just its inference strategy but its learning strategy. Potential approaches:

- **Learned optimizers**: The latent generator could produce not just execution instructions but training update rules.
- **Meta-learning the curriculum**: Instead of a fixed 6-phase training pipeline, learn which phase transitions and loss weightings work best.
- **Self-play for program generation**: Use the model to generate its own training programs, creating a self-improving data flywheel.

---

## 9. Key References

### Latent Program Induction
- Macfarlane & Bonnet. "Latent Program Network." NeurIPS 2025 Spotlight.
- Ellis et al. "DreamCoder: Bootstrapping Inductive Program Synthesis." PLDI 2021.
- "Latent Execution for Neural Program Synthesis." Meta AI, 2021.
- "Searching Latent Program Spaces." NeurIPS 2024.

### Adaptive Computation
- "MIND: Adaptive Thinking with Dynamic Computation." ICLR 2025, Oral.
- Banino et al. "PonderNet: Learning to Ponder." DeepMind, 2021.
- "LoopLM: Scaling Looped Transformers." 2025.
- Damani, Peng et al. "Learning How Hard to Think." ICLR 2025.
- Dehghani et al. "Universal Transformers." ICLR 2019.

### Knowledge Compression
- Todd et al. "Function Vectors in Large Language Models." ICLR 2024.
- Liu et al. "In-Context Vectors." 2024.
- "Implicit In-Context Learning." ICLR 2025.
- Ilharco et al. "Task Arithmetic." ICLR 2023.

### Length Generalization
- Fan, Du, Ramchandran, Lee. "Looped Transformers for Length Generalization." ICLR 2025.
- Saunshi et al. "Reasoning with Latent Thoughts." ICLR 2025.
- Huang et al. "A Formal Framework for Understanding Length Generalization." ICLR 2025.
- "On Provable Length and Compositional Generalization." ICLR 2025.

### Mixture of Experts / Depths / Recursions
- Raposo et al. "Mixture-of-Depths." Google DeepMind, 2024.
- "Mixture-of-Recursions." NeurIPS 2025.
- "ReMoE: Fully Differentiable Mixture-of-Experts." ICLR 2025.
- "Mixture of Nested Experts (MoNE)." NeurIPS 2024.

### Non-Transformer Architectures
- Gu and Dao. "Mamba / Mamba-2: Selective State Space Models." 2023/2024.
- Beck et al. "xLSTM: Extended Long Short-Term Memory." NeurIPS 2024.
- Sun et al. "Test-Time Training Layers." 2024.
- "RWKV v5/v6." 2024-2025.

### Program Execution and Compositional Generalization
- Zaremba and Sutskever. "Learning to Execute." 2014.
- Nye et al. "Show Your Work: Scratchpads for Intermediate Computation." 2021.
- "ExeDec: Execution Decomposition." ICLR 2024, Oral.
- "The Globality Barrier of Transformers." NeurIPS 2024.
- "Abacus Embeddings." NeurIPS 2024.
- Geiping et al. "Latent Reasoning Without Token Generation." NeurIPS 2025.

### Edge Deployment
- "ExecuTorch 1.0." PyTorch, 2025.
- "MCUNet: Tiny Deep Learning on IoT Devices." NeurIPS 2020.
- "SpinQuant: LLM Quantization with Learned Rotations." Meta, 2024.
- "MLPerf Tiny Benchmarks." 2024-2025.

---

## 10. Summary

BroadMind occupies a unique position in the research landscape: no existing system generates an executable latent program in a single forward pass that simultaneously controls **what computation to perform**, **how much computation to perform**, and **what compressed knowledge to apply**. Existing work addresses each dimension in isolation -- LPN does latent programs but needs test-time search; MIND and LoopLM do adaptive compute but with fixed layer content; function vectors compress knowledge but passively steer rather than execute; MoE/MoD/MoR route to pre-existing experts.

BroadMind's contribution is unifying all three dimensions -- **program generation**, **adaptive computation**, and **knowledge compression** -- into a single mechanism. The current implementation demonstrates this on a toy domain with 99%+ accuracy at 445K parameters. The path to real-world impact runs through compositional scaling, edge deployment, and eventually, self-improving learning strategies.

The architecture is small enough to run on a microcontroller, general enough to accommodate new operations without structural changes, and principled enough to have a clear scaling trajectory supported by recent theoretical results. The question is not whether the approach works -- it does -- but whether it scales.

---

*Research document prepared February 2026*
*BroadMind v0.76 -- Mixture of Recursions*
