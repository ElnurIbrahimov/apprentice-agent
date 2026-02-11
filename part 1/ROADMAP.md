# BroadMind Roadmap

## Where We Are (v0.76)

BroadMind v0.76 is a neural program executor that learns latent reasoning circuits at runtime. It is not a transformer -- it generates and executes internal programs on the fly.

**Current capabilities:**
- 4 task families, 13 operations (ACCUMULATE, TRANSFER, COMPARE, DECREMENT)
- Mixed-program execution across all families
- Wisdom distillation (compresses experience into 48-float codes)
- Adaptive compute (uses exactly as many steps as needed)
- Length generalization via sinusoidal encodings + noise injection (v0.75)
- Mixture of Recursions: adaptive inner depth per solver step (v0.76)
- ~445K parameters, runs on CPU or CUDA

**Current results:**
- 99%+ single-family and mixed-program accuracy
- 100% wisdom matching
- Perfect adaptive step matching
- Length generalization: 90%+ at 8 steps, 80%+ at 16 steps (trained on 1-4)
- MoR depth hierarchy: COMPARE > TRANSFER > ACC/DEC

---

## Where We're Going

Three development tracks, prioritized by research impact and feasibility.

### Track 1: Task Scaling (v0.75) -- DONE

> *"Invents reasoning circuits on the go"* -- scaled up.

**Goal:** Train on 4-step programs, generalize to 16-32 steps with zero retraining.

**Why this is first:** ICLR 2025 proved that small capacity-constrained models like BroadMind have a *provable advantage* for length and compositional generalization. Our 477K parameter count is a feature, not a limitation.

#### Planned techniques

| Technique | Based On | What It Does |
|---|---|---|
| Looped execution | Looped Transformers (ICML 2023) | Same weights process each step; add more loops at inference for longer programs |
| Scratchpad states | "Show Your Work" (Nye et al.) | Emit intermediate execution states as auxiliary training targets; +21.5% accuracy on program execution tasks |
| Gaussian noise on initial states | Length generalization in RNNs (2025) | Expose the model to unseen internal states during training; enables generalization from 2K to 128K tokens with only 500 fine-tune steps |
| Execution decomposition | ExeDec (ICLR 2024 Oral, DeepMind) | Break long programs into verified substeps; +44% compositional generalization |
| Diverse training distribution | Provable generalization (ICLR 2025) | Diversity of operation combinations matters more than data volume |

#### Success criteria
- Train on programs of length 1-4
- Generalize to length 8 at >90% accuracy
- Generalize to length 16 at >80% accuracy
- No increase in parameter count

#### Key references
- [On Provable Length and Compositional Generalization (ICLR 2025)](https://arxiv.org/abs/2402.04875)
- [ExeDec: Execution Decomposition (ICLR 2024)](https://arxiv.org/abs/2307.13883)
- [Looped Transformers as Programmable Computers (ICML 2023)](https://arxiv.org/abs/2301.13196)
- [Understanding and Improving Length Generalization in Recurrent Models (2025)](https://arxiv.org/pdf/2507.02782)
- [Show Your Work: Scratchpads for Intermediate Computation](https://arxiv.org/abs/2112.00114)
- [Searching Latent Program Spaces (NeurIPS 2024)](https://arxiv.org/html/2411.08706v1)

---

### Track 2: Mixture of Recursions (v0.76) -- DONE

> *"Adaptive depth per operation"* -- the model decides how deeply to process each step.

**What was built:** An adaptive inner recursion system where shared solver weights iterate 1-4 times per step. A learned router (Gumbel-Softmax) selects depth based on operation complexity. Simple ops exit early; complex ops get iterative refinement.

**Results achieved:**
- Depth hierarchy emerges: COMPARE > TRANSFER > ACC/DEC
- Efficient routing: overall mean depth < 3.0
- Compute cost regularization prevents always-max-depth
- ~445K params (+24K over v0.75)

---

### Track 3: Hardware-Adaptive Compute (v0.77) -- DONE

> *"Evolves its own learning strategies based on hardware availability."*

**Goal:** One model, many deployment profiles. The model detects available resources and adjusts its own depth and width at inference time.

**Results achieved:**
- Full model (w=1.0, d=4): 100.0% accuracy (target >99%)
- 50%-width model (w=0.5, d=4): 98.5% accuracy (target >95%)
- Half-depth model (w=1.0, d=2): 100.0% accuracy (target >90%)
- 75%-width model (w=0.75, d=4): 100.0% accuracy
- Device profiler auto-selects config within latency target
- ~447K params (+2.3K over v0.76 from SwitchableLayerNorm)
- Length generalization preserved: L4 100%, L8 98.7%, L12 94.3%, L16 89.0%

#### Planned techniques

| Technique | Based On | What It Does |
|---|---|---|
| Stochastic depth training | TU Dresden (2025) | Drop layers randomly during training; model learns to function at any depth. Zero additional parameters |
| Nested/Matryoshka FFN | MatFormer (NeurIPS 2024, Google/Harvard) | Structure FFN layers so the first N neurons form a coherent sub-network. Extract 25/50/75/100% width models from one training run |
| Self-distillation | Adaptive Depth Networks (ICLR 2025) | Distill between full model and minimal model during training. Only two endpoints, not every configuration |
| MIND introspection | MIND (ICLR 2025 Oral) | Tiny introspection network (~1-5K params) reads hidden states and decides: iterate again or stop. The model restructures its own compute graph |
| Dynamic width gating | DS-Net (CVPR 2021, extended 2025) | Per-layer gating MLP (~500 params each) outputs a slimming ratio based on input. Model selects its own width per input |
| Device profiling | CLONE (USENIX ATC 2025) | 100ms micro-benchmark at load time; map result to pre-computed configuration |

#### Architecture target

```
Input -> Embedding (shared, ~50K params)
       -> [Core Block x N, each ~80K params, iterable]
           Each block:
             - Nested FFN (supports 25/50/75/100% width)
             - Switchable LayerNorm (one per width config)
             - Halting predictor (~1K params)
             - Optional skip (second sub-path)
       -> Output head (~20K params)

Full config:    ~470K effective params
Minimal config: ~120K effective params
Adaptive range: 120K - 470K per input
```

#### Success criteria
- Full model matches v0.74 accuracy (>99%)
- 50%-width model at >95% accuracy
- Depth-2 model (half depth) at >90% accuracy
- Automatic configuration selection based on device profile

#### Key references
- [MatFormer: Nested Transformer for Elastic Inference (NeurIPS 2024)](https://arxiv.org/abs/2310.07707)
- [MIND: Adaptive Thinking with Dynamic Computation (ICLR 2025)](https://proceedings.iclr.cc/paper_files/paper/2025/file/955499a8e2860ed746717c1374224c43-Paper-Conference.pdf)
- [Adaptive Depth Networks with Skippable Sub-Paths (ICLR 2025)](https://arxiv.org/abs/2312.16392)
- [MoSE: Mixture of Slimmable Experts (Feb 2026)](https://arxiv.org/abs/2602.06154)
- [Dynamic Slimmable Network (CVPR 2021)](https://arxiv.org/abs/2103.13258)
- [Elastoformer (SEC 2025)](https://dl.acm.org/doi/pdf/10.1145/3769102.3770612)
- [CLONE: Customizing LLMs for Edge (USENIX ATC 2025)](https://arxiv.org/abs/2506.02847)
- [Stochastic Depth for Adaptive Inference (TU Dresden 2025)](https://cfaed.tu-dresden.de/files/Images/people/chair-cc/publications/2505_Korol_SDArXiv.pdf)
- [SMGrNN: Self-Motivated Growing Neural Network (Dec 2025)](https://arxiv.org/abs/2512.12713)
- [Discovering Cognitive Strategies with Tiny RNNs (Nature 2025)](https://www.nature.com/articles/s41586-025-09142-4)

---

### Track 4: Edge Deployment (v0.78)

> *"AI that doesn't need a data center to be brilliant."*

**Goal:** Run BroadMind on a Raspberry Pi, a phone, and a microcontroller.

#### Model size projections

| Format | Model Size | RAM at Inference | Total Footprint |
|---|---|---|---|
| FP32 (current) | 1.9 MB | 500 KB | 2.4 MB |
| INT8 | 477 KB | 200 KB | 677 KB |
| INT4 | 239 KB | 150 KB | 389 KB |
| Binary (1-bit) | 60 KB | 100 KB | 160 KB |

#### Expected latency (INT8)

| Device | Expected Latency |
|---|---|
| iPhone / Android flagship | <0.2 ms |
| Raspberry Pi 5 | <0.5 ms |
| Jetson Orin Nano | <0.2 ms |
| STM32H7 (Cortex-M7 @ 480MHz) | 1-5 ms |
| ESP32-S3 | 5-15 ms |

#### Deployment pipeline

```
PyTorch (FP32)
    |
    v
Quantization-Aware Training (INT8)
    |
    v
ONNX Export (torch.onnx.export with dynamo=True)
    |
    +---> ONNX Runtime Mobile (Pi, Android, iOS)
    |
    +---> ExecuTorch (microcontrollers, 50KB runtime)
    |
    +---> LiteRT / TFLite Micro (broadest MCU support)
```

#### Target platforms

| Platform | Framework | Notes |
|---|---|---|
| Raspberry Pi 5 | ONNX Runtime or PyTorch | Easiest first target |
| Android / iOS | ONNX Runtime Mobile | Sub-millisecond inference |
| STM32H7 | ExecuTorch or STM32Cube.AI | 512KB SRAM, 2MB flash -- BroadMind INT8 fits |
| ESP32-S3 | ESP-DL | WiFi/BLE integrated, good for IoT demos |
| Google Coral | TFLite + Edge TPU | Hardware-accelerated inference |

#### Success criteria
- INT8 model accuracy within 1% of FP32
- Runs on Raspberry Pi 5 at <1ms latency
- Runs on at least one microcontroller (STM32 or ESP32)
- Published benchmark numbers

#### Key references
- [Quantized Neural Networks for Microcontrollers (2025)](https://arxiv.org/abs/2508.15008)
- [ExecuTorch - PyTorch Edge Deployment](https://executorch.ai/)
- [ONNX Runtime Performance](https://onnxruntime.ai/docs/performance/)
- [Real Time Inference on Raspberry Pi](https://docs.pytorch.org/tutorials/intermediate/realtime_rpi.html)
- [Deep Learning on Microcontrollers: State of Embedded ML in 2025](https://shawnhymel.com/2994/deep-learning-on-microcontrollers-the-state-of-embedded-ml-in-2025/)
- [Binary Normalized Neural Networks (2025)](https://arxiv.org/pdf/2509.07025)
- [From Tiny Machine Learning to Tiny Deep Learning Survey (2025)](https://arxiv.org/html/2506.18927v1)

---

## Vision Completion Status

| Vision Element | Status | Version |
|---|---|---|
| "Invents reasoning circuits on the go" | Done | v0.70j2 |
| "Keeps the juice, forgets the fluff" | Done | v0.72b |
| "Sips power, drops insight" | Done | v0.73d |
| Mixed-program execution | Done | v0.74 |
| Length generalization (long programs) | Done | v0.75 |
| Mixture of Recursions (adaptive inner depth) | Done | v0.76 |
| Hardware-adaptive compute (elastic inference) | Done | v0.77 |
| Physical integration (edge devices) | Planned | v0.78 |
| "Evolves learning strategies" | Future | TBD |

---

## Version History

| Version | What | Parameters | Accuracy |
|---|---|---|---|
| v0.70j2 | Latent program induction + conditionals | 357K | 96.5% |
| v0.72b | Wisdom distillation | 422K | 99.6% |
| v0.73d | Adaptive compute | ~450K | 100% |
| v0.74 | Complete model (all above + DECREMENT + mixed programs) | 477K | 99.7% |
| v0.75 | Task scaling (length generalization) | ~421K | 99%+ ID, 80%+ at 16 steps |
| v0.76 | Mixture of Recursions (adaptive inner depth) | ~445K | 99%+ with depth hierarchy |
| v0.77 | Elastic inference (width/depth auto-config) | ~447K | 100% full, 98.5% at 50% width |
| v0.78 | Edge deployment | ~447K (INT8: ~447KB) | TBD |
