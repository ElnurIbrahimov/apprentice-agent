"""
BroadMind v0.77: Hardware-Adaptive Compute (Elastic Inference)
==============================================================

Builds on v0.76 (Mixture of Recursions) with:
- Elastic width: model runs at 25/50/75/100% width from a single training run
- Stochastic depth: 1-4 recursion depths, randomly masked during elastic training
- Self-distillation: full-width teacher guides reduced-width students
- Device profiler: auto-selects largest config within latency target

One training run → deploy from microcontrollers (~120K eff params) to full CPU/GPU (~447K params).
~447K params (+2.3K over v0.76's ~445K from SwitchableLayerNorm).
"""

import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
import random

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    n_variables = 3
    n_task_families = 4
    max_program_length = 4        # training max (unchanged)
    max_eval_length = 16          # max inference length for OOD eval
    value_range = 10
    comparison_scale = 30.0       # replaces hardcoded /10 in comparisons

    # Architecture
    d_model = 192
    d_latent = 96
    d_wisdom = 48
    n_wisdom_slots = 5

    # Halting
    halt_threshold = 0.5

    # Distillation
    distill_batch_size = 32

    # Training
    batch_size = 256

    # Phased training (critical for success)
    n_iterations_phase1 = 2500   # Solver only
    n_iterations_phase2 = 1500   # Wisdom distillation
    n_iterations_phase2b = 800   # Wisdom matching
    n_iterations_phase3 = 1500   # Halter only
    n_iterations_phase4 = 1500   # End-to-end
    n_iterations_phase5 = 500    # Length generalization fine-tuning

    # Noise for Phase 5
    noise_std = 0.05

    lr = 1e-3
    lr_fine = 1e-4
    grad_clip = 1.0
    dropout = 0.1
    weight_decay = 1e-4

    # MoR (Mixture of Recursions) — v0.76
    max_recursion_depth = 4       # inner recursion passes (1-4)
    recursion_enc_dim = 24        # d_model // 8, sinusoidal encoding for recursion depth
    compute_cost_weight = 0.005   # penalty to discourage always-max-depth routing

    # v0.77: Elastic inference
    width_multipliers = [0.25, 0.50, 0.75, 1.0]
    self_distill_weight = 2.0           # stronger teacher signal
    n_iterations_phase6 = 2500   # elastic (combined width + interleaved depth)
    n_iterations_phase7 = 1000   # cascaded self-distillation
    n_iterations_phase8 = 300    # recovery
    lr_elastic = 2e-5            # gentle lr to preserve length gen
    latency_target_ms = 10.0

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config = Config()

# ============================================================================
# TASK FAMILIES
# ============================================================================

TASK_FAMILIES = {
    0: {'name': 'ACCUMULATE', 'ops': ['ACC_X', 'ACC_Y', 'ACC_Z']},
    1: {'name': 'TRANSFER', 'ops': ['TRANSFER_XY', 'TRANSFER_YZ', 'TRANSFER_ZX']},
    2: {'name': 'COMPARE', 'ops': ['IF_X_GT_Y_INC_Z', 'IF_Y_GT_Z_INC_X', 'IF_Z_GT_X_INC_Y']},
    3: {'name': 'DECREMENT', 'ops': ['DEC_X', 'DEC_Y', 'DEC_Z']},
}

ALL_OPS = []
for fam in TASK_FAMILIES.values():
    ALL_OPS.extend(fam['ops'])
ALL_OPS.append('PAD')
PAD_IDX = len(ALL_OPS) - 1

N_OPS = len(ALL_OPS)
OP_TO_IDX = {op: i for i, op in enumerate(ALL_OPS)}

def get_family_id(op_idx):
    """Get family ID from operation index."""
    if op_idx >= PAD_IDX:
        return -1
    for fam_id, fam in TASK_FAMILIES.items():
        if ALL_OPS[op_idx] in fam['ops']:
            return fam_id
    return -1

# ============================================================================
# OPERATIONS
# ============================================================================

def execute_op(state, op_name):
    x, y, z = state
    if op_name == 'ACC_X': return (x + 1, y, z)
    if op_name == 'ACC_Y': return (x, y + 1, z)
    if op_name == 'ACC_Z': return (x, y, z + 1)
    if op_name == 'TRANSFER_XY': return (x - 1, y + 1, z)
    if op_name == 'TRANSFER_YZ': return (x, y - 1, z + 1)
    if op_name == 'TRANSFER_ZX': return (x + 1, y, z - 1)
    if op_name == 'IF_X_GT_Y_INC_Z': return (x, y, z + 1) if x > y else (x, y, z)
    if op_name == 'IF_Y_GT_Z_INC_X': return (x + 1, y, z) if y > z else (x, y, z)
    if op_name == 'IF_Z_GT_X_INC_Y': return (x, y + 1, z) if z > x else (x, y, z)
    if op_name == 'DEC_X': return (x - 1, y, z)
    if op_name == 'DEC_Y': return (x, y - 1, z)
    if op_name == 'DEC_Z': return (x, y, z - 1)
    if op_name == 'PAD': return (x, y, z)
    raise ValueError(f"Unknown op: {op_name}")

def execute_program(initial_state, program_ops):
    states = [initial_state]
    state = initial_state
    for op_name in program_ops:
        state = execute_op(state, op_name)
        states.append(state)
    return states

# ============================================================================
# SINUSOIDAL STEP ENCODING (replaces nn.Embedding for step position)
# ============================================================================

def sinusoidal_step_encoding(step_num, d, batch_size, device):
    """Sinusoidal positional encoding for a single step.

    Works for any step_num (no upper bound), zero learnable parameters.
    Solver uses d=48 (d_model//4), Halter uses d=96 (d_model//2).
    """
    position = torch.tensor([step_num], dtype=torch.float, device=device)
    div_term = torch.exp(
        torch.arange(0, d, 2, device=device).float() * -(math.log(10000.0) / d)
    )
    pe = torch.zeros(1, d, device=device)
    pe[0, 0::2] = torch.sin(position * div_term)
    pe[0, 1::2] = torch.cos(position * div_term)
    return pe.expand(batch_size, -1)

# ============================================================================
# DATA GENERATION
# ============================================================================

def generate_batch(batch_size, min_len=1, max_len=4, mixed_prob=0.0):
    program_indices = []
    initial_states = []
    all_intermediate = []
    lengths = []
    family_ids = []

    # Collect all ops across families for mixed programs
    all_family_ops = []
    for fam in TASK_FAMILIES.values():
        all_family_ops.extend(fam['ops'])

    for _ in range(batch_size):
        prog_len = random.randint(min_len, max_len)

        if random.random() < mixed_prob:
            # Mixed program: draw ops from ALL families
            prog_ops = [random.choice(all_family_ops) for _ in range(prog_len)]
            # family_id = family of first op (for wisdom matching)
            family_id = get_family_id(OP_TO_IDX[prog_ops[0]])
        else:
            # Single-family program
            family_id = random.randint(0, config.n_task_families - 1)
            ops = TASK_FAMILIES[family_id]['ops']
            prog_ops = [random.choice(ops) for _ in range(prog_len)]

        init = tuple(np.random.randint(0, config.value_range) for _ in range(3))
        prog_indices = [OP_TO_IDX[op] for op in prog_ops]

        states = execute_program(init, prog_ops)

        # Pad to max_len
        while len(prog_indices) < max_len:
            prog_indices.append(PAD_IDX)

        intermediate = list(states[1:])
        while len(intermediate) < max_len:
            intermediate.append(intermediate[-1])

        program_indices.append(prog_indices)
        initial_states.append(init)
        all_intermediate.append(intermediate)
        lengths.append(prog_len)
        family_ids.append(family_id)

    return {
        'program_indices': torch.tensor(program_indices, dtype=torch.long, device=config.device),
        'initial_states': torch.tensor(initial_states, dtype=torch.float, device=config.device),
        'intermediate': torch.tensor(all_intermediate, dtype=torch.float, device=config.device),
        'lengths': torch.tensor(lengths, dtype=torch.long, device=config.device),
        'family_ids': torch.tensor(family_ids, dtype=torch.long, device=config.device),
    }


def generate_family_batch(family_id, batch_size, prog_len=4):
    """Generate batch from single family (for wisdom distillation)."""
    ops = TASK_FAMILIES[family_id]['ops']

    program_indices = []
    initial_states = []
    all_intermediate = []

    for _ in range(batch_size):
        init = tuple(np.random.randint(0, config.value_range) for _ in range(3))
        prog_ops = [random.choice(ops) for _ in range(prog_len)]
        prog_indices = [OP_TO_IDX[op] for op in prog_ops]

        states = execute_program(init, prog_ops)

        program_indices.append(prog_indices)
        initial_states.append(init)
        all_intermediate.append(states[1:])

    return {
        'program_indices': torch.tensor(program_indices, dtype=torch.long, device=config.device),
        'initial_states': torch.tensor(initial_states, dtype=torch.float, device=config.device),
        'intermediate': torch.tensor(all_intermediate, dtype=torch.float, device=config.device),
    }

# ============================================================================
# WISDOM COMPONENTS (from v0.72b)
# ============================================================================

class WisdomDistiller(nn.Module):
    """Compress N experiences into ONE wisdom code."""

    def __init__(self, config):
        super().__init__()

        self.experience_encoder = nn.Sequential(
            nn.Linear(config.n_variables * 2 + N_OPS, config.d_model),
            nn.LayerNorm(config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
        )

        self.aggregator = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_wisdom),
        )

        self.compressor = nn.Sequential(
            nn.LayerNorm(config.d_wisdom),
            nn.Linear(config.d_wisdom, config.d_wisdom),
            nn.Tanh(),
        )

    def forward(self, states, next_states, op_indices):
        N = states.shape[0]
        op_onehot = F.one_hot(op_indices, N_OPS).float()
        experience = torch.cat([states, next_states, op_onehot], dim=-1)
        encoded = self.experience_encoder(experience)
        aggregated = self.aggregator(encoded)
        pooled = aggregated.mean(dim=0)
        wisdom = self.compressor(pooled)
        return wisdom


class WisdomBank(nn.Module):
    """Store wisdom nuggets."""

    def __init__(self, config):
        super().__init__()
        self.wisdom_codes = nn.Parameter(
            torch.randn(config.n_wisdom_slots, config.d_wisdom) * 0.1
        )

    def write_wisdom(self, slot_idx, wisdom_code):
        with torch.no_grad():
            self.wisdom_codes.data[slot_idx] = wisdom_code

    def read_all(self):
        return self.wisdom_codes


class WisdomMatcher(nn.Module):
    """Match problems to wisdom."""

    def __init__(self, config):
        super().__init__()

        self.problem_encoder = nn.Sequential(
            nn.Linear(config.n_variables + N_OPS, config.d_model),
            nn.LayerNorm(config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, config.d_wisdom),
        )

        self.temperature = nn.Parameter(torch.tensor(1.0))

    def forward(self, initial_state, first_op_idx, wisdom_bank):
        batch_size = initial_state.shape[0]

        op_onehot = F.one_hot(first_op_idx, N_OPS).float()
        problem = torch.cat([initial_state, op_onehot], dim=-1)
        problem_enc = self.problem_encoder(problem)

        all_wisdom = wisdom_bank.read_all()
        scores = torch.matmul(problem_enc, all_wisdom.T)
        scores = scores / (self.temperature.abs() + 0.1)

        attention = F.softmax(scores, dim=-1)
        wisdom = torch.matmul(attention, all_wisdom)

        return wisdom, attention

# ============================================================================
# SWITCHABLE LAYER NORM (new in v0.77)
# ============================================================================

class SwitchableLayerNorm(nn.Module):
    """One LayerNorm per width config. Normalization stats depend on feature count,
    so sub-width norms must be separate from full-width norm."""

    def __init__(self, full_dim, width_multipliers):
        super().__init__()
        self.full_dim = full_dim
        self.norms = nn.ModuleDict()
        for w in width_multipliers:
            key = f"w{int(w * 100)}"
            dim = int(full_dim * w)
            self.norms[key] = nn.LayerNorm(dim)

    def forward(self, x, width_mult=1.0):
        key = f"w{int(width_mult * 100)}"
        return self.norms[key](x)

# ============================================================================
# ELASTIC SOLVER (replaces v0.76 Solver — width-sliceable, depth-maskable)
# ============================================================================

class ElasticSolver(nn.Module):
    """Wisdom-guided program executor with Mixture of Recursions (MoR)
    and elastic width/depth for hardware-adaptive compute.

    All nn.Sequential blocks decomposed into individual named layers so each
    can be width-sliced independently. Junction layers (latent_gen, executor,
    router) use precomputed column indices for correct multi-source slicing.

    At width_mult=1.0 with no depth_mask, behavior is identical to v0.76 Solver.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        d = config.d_model  # 192

        # --- State encoder (decomposed from nn.Sequential) ---
        self.state_enc_linear = nn.Linear(config.n_variables, d)
        self.state_enc_norm = SwitchableLayerNorm(d, config.width_multipliers)

        # --- Op embedding ---
        self.op_embedding = nn.Embedding(N_OPS, d)

        # --- Wisdom encoder (decomposed) ---
        self.wisdom_enc_linear = nn.Linear(config.d_wisdom, d // 2)

        # --- Recursion router (decomposed) ---
        # Input: state_enc (d) + op_enc (d) = 2d [junction]
        self.router_linear1 = nn.Linear(d * 2, d // 4)
        self.router_linear2 = nn.Linear(d // 4, config.max_recursion_depth)
        # Bias init: favor depth 1 at start for stable early training
        with torch.no_grad():
            self.router_linear2.bias.data = torch.tensor([2.0, 0.0, -1.0, -2.0])

        # --- Latent generator (decomposed) ---
        # Input: state_enc(d) + op_enc(d) + step_enc(d//4) + wisdom_enc(d//2) + rec_enc(rec_dim)
        # = 2d + d//4 + d//2 + rec_dim = 552 at full width [junction]
        latent_gen_in = d * 2 + d // 4 + d // 2 + config.recursion_enc_dim
        self.latent_gen_linear1 = nn.Linear(latent_gen_in, d)
        self.latent_gen_norm = SwitchableLayerNorm(d, config.width_multipliers)
        self.latent_gen_linear2 = nn.Linear(d, config.d_latent)

        # --- Comparison encoder (decomposed) ---
        self.comp_enc_linear = nn.Linear(6, d // 4)

        # --- Latent encoder (decomposed) ---
        self.latent_enc_linear = nn.Linear(config.d_latent, d)
        self.latent_enc_norm = SwitchableLayerNorm(d, config.width_multipliers)

        # --- Executor (decomposed) ---
        # Input: state_enc(d) + latent_enc(d) + comp_enc(d//4) = 2d + d//4 [junction]
        executor_in = d * 2 + d // 4
        self.exec_linear1 = nn.Linear(executor_in, d)
        self.exec_norm = SwitchableLayerNorm(d, config.width_multipliers)
        self.exec_linear2 = nn.Linear(d, d)
        self.exec_linear3 = nn.Linear(d, config.n_variables)

        # --- Predictor ---
        self.predictor = nn.Linear(config.n_variables, config.n_variables)

        # --- Precompute junction column indices per width ---
        self._precompute_junction_indices(config)

    def _precompute_junction_indices(self, config):
        """Precompute column indices for junction layers at each width config.

        Junction layers receive concatenated inputs from multiple sources.
        At reduced width, each source is sliced to its elastic dim, but the
        weight matrix columns must reference the correct positions in the
        full-width layout.
        """
        d = config.d_model          # 192
        rec_dim = config.recursion_enc_dim  # 24

        # Full-width input layouts:
        # router:     [state_enc(0:d) | op_enc(d:2d)]                         = 2d = 384
        # latent_gen: [state(0:d) | op(d:2d) | step(2d:2d+d//4)
        #              | wisdom(2d+d//4:2d+d//4+d//2) | rec(2d+3d//4:2d+3d//4+rec)]  = 552
        # executor:   [state(0:d) | latent_enc(d:2d) | comp(2d:2d+d//4)]     = 2d+d//4 = 432

        step_start = 2 * d                    # 384
        wisdom_start = step_start + d // 4    # 432
        rec_start = wisdom_start + d // 2     # 528

        for w in config.width_multipliers:
            d_eff = int(d * w)
            key = f"w{int(w * 100)}"

            # Router: state_enc[:d_eff] + op_enc[:d_eff]
            router_cols = list(range(0, d_eff)) + list(range(d, d + d_eff))
            self.register_buffer(f'router_cols_{key}',
                                 torch.tensor(router_cols, dtype=torch.long))

            # Latent gen: state[:d_eff] + op[:d_eff] + step[:d_eff//4]
            #           + wisdom[:d_eff//2] + rec[:rec_dim] (always full)
            latent_cols = (
                list(range(0, d_eff)) +
                list(range(d, d + d_eff)) +
                list(range(step_start, step_start + d_eff // 4)) +
                list(range(wisdom_start, wisdom_start + d_eff // 2)) +
                list(range(rec_start, rec_start + rec_dim))
            )
            self.register_buffer(f'latent_cols_{key}',
                                 torch.tensor(latent_cols, dtype=torch.long))

            # Executor: state[:d_eff] + latent_enc[:d_eff] + comp[:d_eff//4]
            exec_cols = (
                list(range(0, d_eff)) +
                list(range(d, d + d_eff)) +
                list(range(2 * d, 2 * d + d_eff // 4))
            )
            self.register_buffer(f'exec_cols_{key}',
                                 torch.tensor(exec_cols, dtype=torch.long))

    def _encode_state(self, state, width_mult):
        """Encode state at elastic width: Linear -> SwitchableLayerNorm -> GELU -> Dropout."""
        d_eff = int(self.config.d_model * width_mult)
        enc = F.linear(state,
                       self.state_enc_linear.weight[:d_eff],
                       self.state_enc_linear.bias[:d_eff])
        enc = self.state_enc_norm(enc, width_mult)
        enc = F.gelu(enc)
        enc = F.dropout(enc, p=self.config.dropout, training=self.training)
        return enc

    def _comparison_features(self, state):
        """Extract comparison features from state (always 6 features)."""
        x, y, z = state[:, 0], state[:, 1], state[:, 2]
        return torch.stack([
            (x > y).float(), (y > z).float(), (z > x).float(),
            (x - y) / self.config.comparison_scale,
            (y - z) / self.config.comparison_scale,
            (z - x) / self.config.comparison_scale,
        ], dim=-1)

    def get_router_logits(self, state, op_idx):
        """Get router logits at full width (for analysis only)."""
        state_enc = self._encode_state(state, 1.0)
        op_enc = self.op_embedding(op_idx)
        r_input = torch.cat([state_enc, op_enc], dim=-1)
        r1 = F.gelu(self.router_linear1(r_input))
        return self.router_linear2(r1)

    def step(self, state, op_idx, step_num, wisdom, training_noise_std=0.0,
             width_mult=1.0, depth_mask=None):
        """Execute one solver step at elastic width/depth.

        Args:
            state: (batch, n_variables) current state
            op_idx: (batch,) operation indices
            step_num: int, current program step
            wisdom: (batch, d_wisdom) wisdom vector
            training_noise_std: float, Gaussian noise for length gen
            width_mult: float in {0.25, 0.50, 0.75, 1.0}
            depth_mask: BoolTensor(max_recursion_depth) or None; True=keep, depth 0 always kept
        """
        batch_size = state.shape[0]
        d = self.config.d_model
        d_eff = int(d * width_mult)
        d_latent_eff = int(self.config.d_latent * width_mult)
        key = f"w{int(width_mult * 100)}"

        # === Fixed encodings (computed ONCE per step) ===
        op_enc = self.op_embedding(op_idx)[:, :d_eff]               # (batch, d_eff)
        step_enc = sinusoidal_step_encoding(
            step_num, d_eff // 4, batch_size, state.device
        )                                                             # (batch, d_eff//4)
        wisdom_enc = F.linear(
            wisdom,
            self.wisdom_enc_linear.weight[:d_eff // 2],
            self.wisdom_enc_linear.bias[:d_eff // 2]
        )                                                             # (batch, d_eff//2)
        wisdom_enc = F.gelu(wisdom_enc)

        # === Router decision (based on initial state + operation) ===
        state_enc = self._encode_state(state, width_mult)            # (batch, d_eff)
        router_input = torch.cat([state_enc, op_enc], dim=-1)        # (batch, 2*d_eff)
        router_cols = getattr(self, f'router_cols_{key}')
        r1_dim = d_eff // 4
        r1 = F.linear(router_input,
                       self.router_linear1.weight[:r1_dim, router_cols],
                       self.router_linear1.bias[:r1_dim])
        r1 = F.gelu(r1)
        router_logits = F.linear(r1,
                                  self.router_linear2.weight[:, :r1_dim],
                                  self.router_linear2.bias)          # (batch, 4)

        # Apply depth mask (masked depths get -inf logits)
        if depth_mask is not None:
            router_logits = router_logits.masked_fill(
                ~depth_mask.unsqueeze(0), float('-inf')
            )

        if self.training:
            router_weights = F.gumbel_softmax(router_logits, tau=1.0, hard=True)
        else:
            router_weights = F.one_hot(
                router_logits.argmax(dim=-1), self.config.max_recursion_depth
            ).float()

        # === Inner recursion (shared weights, variable depth) ===
        recursive_state = state
        outputs = []
        latent_cols = getattr(self, f'latent_cols_{key}')
        exec_cols = getattr(self, f'exec_cols_{key}')

        for r in range(self.config.max_recursion_depth):
            # Skip masked depths (optimization — their weight is 0)
            if depth_mask is not None and not depth_mask[r]:
                outputs.append(recursive_state)
                continue

            # Re-encode state at current recursion
            state_enc_r = self._encode_state(recursive_state, width_mult)

            # Recursion depth encoding (fixed dim, not width-scaled)
            rec_enc = sinusoidal_step_encoding(
                r, self.config.recursion_enc_dim, batch_size, state.device
            )

            # Latent generation (junction layer)
            latent_input = torch.cat(
                [state_enc_r, op_enc, step_enc, wisdom_enc, rec_enc], dim=-1
            )
            lg1 = F.linear(latent_input,
                           self.latent_gen_linear1.weight[:d_eff, latent_cols],
                           self.latent_gen_linear1.bias[:d_eff])
            lg1 = self.latent_gen_norm(lg1, width_mult)
            lg1 = F.gelu(lg1)
            lg1 = F.dropout(lg1, p=self.config.dropout, training=self.training)
            latent = F.linear(lg1,
                              self.latent_gen_linear2.weight[:d_latent_eff, :d_eff],
                              self.latent_gen_linear2.bias[:d_latent_eff])

            # Comparison encoding (input always 6, output d_eff//4)
            comp_features = self._comparison_features(recursive_state)
            comp_enc = F.linear(comp_features,
                                self.comp_enc_linear.weight[:d_eff // 4],
                                self.comp_enc_linear.bias[:d_eff // 4])
            comp_enc = F.gelu(comp_enc)

            # Latent encoding
            latent_enc = F.linear(latent,
                                   self.latent_enc_linear.weight[:d_eff, :d_latent_eff],
                                   self.latent_enc_linear.bias[:d_eff])
            latent_enc = self.latent_enc_norm(latent_enc, width_mult)
            latent_enc = F.gelu(latent_enc)

            # Executor (junction layer)
            exec_input = torch.cat([state_enc_r, latent_enc, comp_enc], dim=-1)
            e1 = F.linear(exec_input,
                          self.exec_linear1.weight[:d_eff, exec_cols],
                          self.exec_linear1.bias[:d_eff])
            e1 = self.exec_norm(e1, width_mult)
            e1 = F.gelu(e1)
            e1 = F.dropout(e1, p=self.config.dropout, training=self.training)
            e2 = F.linear(e1,
                          self.exec_linear2.weight[:d_eff, :d_eff],
                          self.exec_linear2.bias[:d_eff])
            e2 = F.gelu(e2)
            # Final output: always n_variables (3)
            delta = F.linear(e2,
                             self.exec_linear3.weight[:, :d_eff],
                             self.exec_linear3.bias)

            recursive_state = recursive_state + delta
            outputs.append(recursive_state)

        # === Weighted combination ===
        stacked = torch.stack(outputs, dim=1)                        # (batch, 4, 3)
        new_state = (stacked * router_weights.unsqueeze(-1)).sum(dim=1)

        # Compute cost (for regularization)
        depths = torch.tensor([1.0, 2.0, 3.0, 4.0], device=state.device)
        compute_cost = (router_weights * depths).sum(-1).mean()

        # Noise injection for length generalization
        if training_noise_std > 0.0 and self.training:
            new_state = new_state + torch.randn_like(new_state) * training_noise_std

        return new_state, compute_cost

    def forward(self, programs, initial_states, wisdom, training_noise_std=0.0,
                width_mult=1.0, depth_mask=None):
        n_steps = programs.shape[1]

        state = initial_states
        all_preds = []
        all_states = []
        total_compute_cost = 0.0

        for t in range(n_steps):
            op_idx = programs[:, t]
            state, compute_cost = self.step(
                state, op_idx, t, wisdom,
                training_noise_std=training_noise_std,
                width_mult=width_mult,
                depth_mask=depth_mask
            )
            pred = self.predictor(state)
            all_preds.append(pred)
            all_states.append(state)
            total_compute_cost = total_compute_cost + compute_cost

        return torch.stack(all_preds, dim=1), torch.stack(all_states, dim=1), total_compute_cost

# ============================================================================
# HALTER (length-agnostic, mean-pooled program encoding — unchanged from v0.76)
# ============================================================================

class Halter(nn.Module):
    """Length-agnostic halter with mean-pooled program encoding."""

    def __init__(self, config):
        super().__init__()

        self.op_embedding = nn.Embedding(N_OPS, config.d_model // 2)  # 96 dims

        # Mean-pool op embeddings (mask PAD) + concat op count -> Linear(97, 192)
        self.program_encoder = nn.Sequential(
            nn.Linear(config.d_model // 2 + 1, config.d_model),
            nn.LayerNorm(config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )

        # Classifier: d_model (192 from program) + d_model//2 (96 from step) = 288
        self.classifier = nn.Sequential(
            nn.Linear(config.d_model + config.d_model // 2, config.d_model // 2),
            nn.LayerNorm(config.d_model // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model // 2, 1),
        )

    def forward(self, programs, step):
        batch_size = programs.shape[0]
        seq_len = programs.shape[1]

        # Embed all ops
        op_emb = self.op_embedding(programs)  # (batch, seq_len, 96)

        # Mask out PAD tokens
        pad_mask = (programs != PAD_IDX).float().unsqueeze(-1)  # (batch, seq_len, 1)
        op_count = pad_mask.sum(dim=1)  # (batch, 1) — number of real ops

        # Mean pool over non-PAD ops
        masked_emb = op_emb * pad_mask
        pooled = masked_emb.sum(dim=1) / op_count.clamp(min=1)  # (batch, 96)

        # Normalize op count to [0,1] range
        op_count_norm = op_count / config.max_eval_length  # (batch, 1)

        # Concat pooled embedding + op count -> program encoding
        program_input = torch.cat([pooled, op_count_norm], dim=-1)  # (batch, 97)
        program_enc = self.program_encoder(program_input)  # (batch, 192)

        # Sinusoidal step encoding (d_model//2 = 96)
        step_enc = sinusoidal_step_encoding(
            step, config.d_model // 2, batch_size, programs.device
        )

        x = torch.cat([program_enc, step_enc], dim=-1)  # (batch, 288)
        logit = self.classifier(x)

        return logit

# ============================================================================
# COMPLETE MODEL
# ============================================================================

class BroadMindV077(nn.Module):
    """
    BroadMind v0.77: Hardware-Adaptive Compute (Elastic Inference)

    Builds on v0.76 with:
    - Elastic width (25/50/75/100%) via Matryoshka-style weight slicing
    - Stochastic depth (1-4 recursion depths) via depth masking
    - Self-distillation from full-width teacher to reduced-width students
    - Automatic device profiling and config selection
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Wisdom components (always full-width)
        self.distiller = WisdomDistiller(config)
        self.wisdom_bank = WisdomBank(config)
        self.matcher = WisdomMatcher(config)

        # Solver (elastic width/depth)
        self.solver = ElasticSolver(config)

        # Halter (adaptive compute, always full-width)
        self.halter = Halter(config)

    def get_wisdom(self, programs, initial_states):
        """Get wisdom for batch (via matching)."""
        first_ops = programs[:, 0]
        wisdom, attention = self.matcher(initial_states, first_ops, self.wisdom_bank)
        return wisdom, attention

    def forward_all_steps(self, programs, initial_states, wisdom=None,
                          training_noise_std=0.0, width_mult=1.0, depth_mask=None):
        """Run all steps, return predictions, halt logits, and compute cost."""
        if wisdom is None:
            wisdom, _ = self.get_wisdom(programs, initial_states)

        preds, states, compute_cost = self.solver(
            programs, initial_states, wisdom,
            training_noise_std=training_noise_std,
            width_mult=width_mult,
            depth_mask=depth_mask
        )

        halt_logits = []
        for t in range(programs.shape[1]):
            logit = self.halter(programs, t)
            halt_logits.append(logit)

        return preds, torch.stack(halt_logits, dim=1), wisdom, compute_cost

    def forward_adaptive(self, programs, initial_states, training_noise_std=0.0,
                         width_mult=1.0, depth_mask=None):
        """Forward with adaptive halting. Returns (pred, steps_used, compute_cost)."""
        batch_size = programs.shape[0]
        max_steps = programs.shape[1]

        # Get wisdom (always full-width)
        wisdom, _ = self.get_wisdom(programs, initial_states)

        state = initial_states
        halted = torch.zeros(batch_size, dtype=torch.bool, device=state.device)
        final_states = state.clone()
        steps_used = torch.zeros(batch_size, device=state.device)
        total_compute_cost = 0.0

        for t in range(max_steps):
            if halted.all():
                break

            op_idx = programs[:, t]
            new_state, compute_cost = self.solver.step(
                state, op_idx, t, wisdom,
                training_noise_std=training_noise_std,
                width_mult=width_mult,
                depth_mask=depth_mask
            )
            total_compute_cost = total_compute_cost + compute_cost

            state = torch.where(halted.unsqueeze(1), state, new_state)

            halt_logit = self.halter(programs, t)
            halt_prob = torch.sigmoid(halt_logit).squeeze(-1)

            should_halt = (halt_prob > self.config.halt_threshold) & ~halted

            final_states = torch.where(should_halt.unsqueeze(1), state, final_states)
            steps_used = steps_used + (~halted).float()
            halted = halted | should_halt

        final_states = torch.where(~halted.unsqueeze(1), state, final_states)

        pred = self.solver.predictor(final_states)
        return pred, steps_used, total_compute_cost

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

# ============================================================================
# TRAINING PHASES 1-5 (adapted for ElasticSolver API — default width_mult=1.0)
# ============================================================================

def train_solver_only(model, optimizer, batch):
    """Phase 1: Train solver with zero wisdom. Includes compute cost regularization."""
    model.train()
    optimizer.zero_grad()

    programs = batch['program_indices']
    initial_states = batch['initial_states']
    intermediate = batch['intermediate']
    lengths = batch['lengths']

    batch_size = programs.shape[0]
    zero_wisdom = torch.zeros(batch_size, config.d_wisdom, device=config.device)

    preds, _, compute_cost = model.solver(programs, initial_states, zero_wisdom)

    batch_size, max_steps, n_vars = preds.shape
    mask = torch.arange(max_steps, device=preds.device).unsqueeze(0) < lengths.unsqueeze(1)
    mask = mask.unsqueeze(-1).float()

    task_loss = ((preds - intermediate) ** 2 * mask).sum() / mask.sum()
    cost_loss = config.compute_cost_weight * compute_cost
    loss = task_loss + cost_loss

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
    optimizer.step()

    return task_loss.item(), cost_loss.item()


def train_wisdom(model, optimizer):
    """Phase 2: Train wisdom distillation. Includes compute cost regularization."""
    model.train()
    optimizer.zero_grad()

    total_loss = 0.0
    total_compute_cost = 0.0

    for family_id in range(config.n_task_families):
        # Generate distillation batch
        distill_batch = generate_family_batch(family_id, config.distill_batch_size, prog_len=4)

        # Collect transitions
        states_list = []
        next_states_list = []
        ops_list = []

        for b in range(config.distill_batch_size):
            for t in range(4):
                if t == 0:
                    state = distill_batch['initial_states'][b]
                else:
                    state = distill_batch['intermediate'][b, t-1]
                next_state = distill_batch['intermediate'][b, t]
                op_idx = distill_batch['program_indices'][b, t]

                states_list.append(state)
                next_states_list.append(next_state)
                ops_list.append(op_idx)

        states = torch.stack(states_list)
        next_states = torch.stack(next_states_list)
        ops = torch.stack(ops_list)

        # Distill
        wisdom = model.distiller(states, next_states, ops)
        model.wisdom_bank.write_wisdom(family_id, wisdom.detach())

        # Test wisdom
        test_batch = generate_family_batch(family_id, config.batch_size // 4, prog_len=4)
        batch_size = test_batch['program_indices'].shape[0]
        wisdom_expanded = wisdom.unsqueeze(0).expand(batch_size, -1)

        preds, _, compute_cost = model.solver(
            test_batch['program_indices'], test_batch['initial_states'], wisdom_expanded
        )

        loss = F.mse_loss(preds, test_batch['intermediate'])
        total_loss = total_loss + loss
        total_compute_cost = total_compute_cost + config.compute_cost_weight * compute_cost

    combined_loss = total_loss + total_compute_cost
    combined_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
    optimizer.step()

    return total_loss.item() / config.n_task_families


def train_matcher(model, optimizer, batch):
    """Phase 2b: Train matcher to route problems to correct wisdom."""
    model.matcher.train()
    model.wisdom_bank.train()
    model.distiller.eval()
    model.solver.eval()
    model.halter.eval()

    optimizer.zero_grad()

    programs = batch['program_indices']
    initial_states = batch['initial_states']
    family_ids = batch['family_ids']

    first_ops = programs[:, 0]

    # Compute matching scores (logits)
    op_onehot = F.one_hot(first_ops, N_OPS).float()
    problem = torch.cat([initial_states, op_onehot], dim=-1)
    problem_enc = model.matcher.problem_encoder(problem)

    all_wisdom = model.wisdom_bank.read_all()
    scores = torch.matmul(problem_enc, all_wisdom.T)
    scores = scores / (model.matcher.temperature.abs() + 0.1)

    # Classification loss over task families
    loss = F.cross_entropy(scores[:, :config.n_task_families], family_ids)

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
    optimizer.step()

    return loss.item()


def train_halter_only(model, optimizer, batch):
    """Phase 3: Train halter (solver frozen). No compute cost — solver is frozen."""
    model.halter.train()
    model.solver.eval()
    model.distiller.eval()
    model.matcher.eval()

    optimizer.zero_grad()

    programs = batch['program_indices']
    lengths = batch['lengths']
    max_steps = programs.shape[1]

    total_loss = 0.0

    for t in range(max_steps):
        halt_logit = model.halter(programs, t)
        should_halt = (t >= lengths - 1).float().unsqueeze(-1)
        loss = F.binary_cross_entropy_with_logits(halt_logit, should_halt)
        total_loss = total_loss + loss

    total_loss = total_loss / max_steps

    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.halter.parameters(), config.grad_clip)
    optimizer.step()

    return total_loss.item()


def train_end_to_end(model, optimizer, batch):
    """Phase 4: Fine-tune everything. Includes compute cost regularization."""
    model.train()
    optimizer.zero_grad()

    programs = batch['program_indices']
    initial_states = batch['initial_states']
    intermediate = batch['intermediate']
    lengths = batch['lengths']

    preds, halt_logits, wisdom, compute_cost = model.forward_all_steps(programs, initial_states)

    batch_size, max_steps, n_vars = preds.shape

    # Task loss
    mask = torch.arange(max_steps, device=preds.device).unsqueeze(0) < lengths.unsqueeze(1)
    mask = mask.unsqueeze(-1).float()
    task_loss = ((preds - intermediate) ** 2 * mask).sum() / mask.sum()

    # Halt loss
    halt_loss = 0.0
    for t in range(max_steps):
        should_halt = (t >= lengths - 1).float().unsqueeze(-1)
        loss = F.binary_cross_entropy_with_logits(halt_logits[:, t], should_halt)
        halt_loss = halt_loss + loss
    halt_loss = halt_loss / max_steps

    # Matching loss (preserve wisdom routing during fine-tuning)
    family_ids = batch['family_ids']
    first_ops = programs[:, 0]
    op_onehot = F.one_hot(first_ops, N_OPS).float()
    problem = torch.cat([initial_states, op_onehot], dim=-1)
    problem_enc = model.matcher.problem_encoder(problem)
    all_wisdom = model.wisdom_bank.read_all()
    scores = torch.matmul(problem_enc, all_wisdom.T)
    scores = scores / (model.matcher.temperature.abs() + 0.1)
    match_loss = F.cross_entropy(scores[:, :config.n_task_families], family_ids)

    loss = task_loss + 0.5 * halt_loss + 0.5 * match_loss + config.compute_cost_weight * compute_cost

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
    optimizer.step()

    return task_loss.item(), halt_loss.item()


def train_phase5(model, optimizer, iteration, total_iterations):
    """Phase 5: Length generalization fine-tuning with Gaussian noise."""
    model.solver.train()
    model.halter.eval()
    model.distiller.eval()
    model.matcher.eval()

    optimizer.zero_grad()

    # Linearly anneal noise: 0.05 -> 0.02
    progress = iteration / max(total_iterations - 1, 1)
    noise_std = config.noise_std * (1.0 - progress) + 0.02 * progress

    batch = generate_batch(config.batch_size, min_len=1, max_len=4, mixed_prob=0.5)

    programs = batch['program_indices']
    initial_states = batch['initial_states']
    intermediate = batch['intermediate']
    lengths = batch['lengths']

    # Get wisdom (detached — we're not training matcher/bank)
    with torch.no_grad():
        wisdom, _ = model.get_wisdom(programs, initial_states)

    # Run WITH noise
    preds_noisy, _, compute_cost = model.solver(
        programs, initial_states, wisdom,
        training_noise_std=noise_std
    )

    # Task loss (masked)
    batch_size, max_steps, n_vars = preds_noisy.shape
    mask = torch.arange(max_steps, device=preds_noisy.device).unsqueeze(0) < lengths.unsqueeze(1)
    mask = mask.unsqueeze(-1).float()
    task_loss = ((preds_noisy - intermediate) ** 2 * mask).sum() / mask.sum()

    # Consistency loss: run WITHOUT noise, penalize divergence
    with torch.no_grad():
        preds_clean, _, _ = model.solver(
            programs, initial_states, wisdom,
            training_noise_std=0.0
        )

    consistency_loss = ((preds_noisy - preds_clean.detach()) ** 2 * mask).sum() / mask.sum()

    cost_loss = config.compute_cost_weight * compute_cost
    loss = task_loss + 0.1 * consistency_loss + cost_loss

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.solver.parameters(), config.grad_clip)
    optimizer.step()

    return task_loss.item(), consistency_loss.item(), noise_std, cost_loss.item()

# ============================================================================
# ELASTIC TRAINING UTILITIES (new in v0.77)
# ============================================================================

def generate_structured_depth_mask(max_depth, device):
    """Generate structured depth mask for elastic training.
    78% no mask (full depth), 12% half-depth, 6% single-depth, 4% three-depth.
    Returns None for no mask (full depth), or BoolTensor for restricted depth.
    22% depth restriction balances depth training with width quality preservation."""
    r = random.random()
    if r < 0.78:
        return None  # full depth — majority preserves natural routing
    elif r < 0.90:
        # Half-depth: depths 0,1 only — key eval config
        mask = torch.zeros(max_depth, dtype=torch.bool, device=device)
        mask[:2] = True
        return mask
    elif r < 0.96:
        # Single-depth: depth 0 only — hardest constraint
        mask = torch.zeros(max_depth, dtype=torch.bool, device=device)
        mask[0] = True
        return mask
    else:
        # Three-depth: depths 0,1,2
        mask = torch.zeros(max_depth, dtype=torch.bool, device=device)
        mask[:3] = True
        return mask


def init_subnorm_from_full(solver):
    """Initialize sub-width LayerNorms from full-width norm (copy first N elements).
    Must be called before Phase 6. Random init would cause instability."""
    for norm_name in ['state_enc_norm', 'latent_gen_norm', 'latent_enc_norm', 'exec_norm']:
        switchable_norm = getattr(solver, norm_name)
        full_norm = switchable_norm.norms['w100']
        for key, sub_norm in switchable_norm.norms.items():
            if key == 'w100':
                continue
            dim = sub_norm.normalized_shape[0]
            with torch.no_grad():
                sub_norm.weight.data.copy_(full_norm.weight.data[:dim])
                sub_norm.bias.data.copy_(full_norm.bias.data[:dim])


def train_phase6_elastic(model, optimizer):
    """Phase 6: Combined width + depth elastic training.
    78% full depth with biased width sampling — trains width adaptation.
    22% restricted depth with biased width — trains depth adaptation.
    Both width and depth vary each iteration, interleaved to prevent
    catastrophic forgetting of either. Only solver trainable."""
    model.solver.train()
    model.halter.eval()
    model.distiller.eval()
    model.matcher.eval()

    optimizer.zero_grad()

    # Biased width sampling: narrow widths get more exposure
    r = random.random()
    if r < 0.30:
        width_mult = 0.25
    elif r < 0.60:
        width_mult = 0.50
    elif r < 0.80:
        width_mult = 0.75
    else:
        width_mult = 1.0

    # Depth: 78% full, 12% half, 6% single, 4% three (22% restricted)
    depth_mask = generate_structured_depth_mask(
        config.max_recursion_depth, config.device
    )

    batch = generate_batch(config.batch_size, min_len=1, max_len=4, mixed_prob=0.5)

    programs = batch['program_indices']
    initial_states = batch['initial_states']
    intermediate = batch['intermediate']
    lengths = batch['lengths']

    with torch.no_grad():
        wisdom, _ = model.get_wisdom(programs, initial_states)

    preds, _, compute_cost = model.solver(
        programs, initial_states, wisdom,
        width_mult=width_mult,
        depth_mask=depth_mask
    )

    batch_size, max_steps, n_vars = preds.shape
    mask = torch.arange(max_steps, device=preds.device).unsqueeze(0) < lengths.unsqueeze(1)
    mask = mask.unsqueeze(-1).float()

    task_loss = ((preds - intermediate) ** 2 * mask).sum() / mask.sum()
    cost_loss = config.compute_cost_weight * compute_cost
    loss = task_loss + cost_loss

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.solver.parameters(), config.grad_clip)
    optimizer.step()

    depth_str = "full" if depth_mask is None else "restricted"
    return task_loss.item(), cost_loss.item(), width_mult, depth_str


def train_phase7_distill(model, optimizer):
    """Phase 7: Cascaded self-distillation with full-width interleaving + depth distillation.
    Each width learns from the NEXT-WIDER width (not always from full):
      w=0.75 learns from w=1.0, w=0.50 learns from w=0.75, w=0.25 learns from w=0.50
    Plus 15% depth distillation: half-depth student <- full-depth teacher (both full width).
    20% full-width maintenance to prevent degradation.
    Only solver trainable, lr=lr_elastic."""
    model.solver.train()
    model.halter.eval()
    model.distiller.eval()
    model.matcher.eval()

    optimizer.zero_grad()

    batch = generate_batch(config.batch_size, min_len=1, max_len=4, mixed_prob=0.5)

    programs = batch['program_indices']
    initial_states = batch['initial_states']
    intermediate = batch['intermediate']
    lengths = batch['lengths']

    with torch.no_grad():
        wisdom, _ = model.get_wisdom(programs, initial_states)

    batch_size, max_steps = programs.shape[0], programs.shape[1]
    mask = torch.arange(max_steps, device=programs.device).unsqueeze(0) < lengths.unsqueeze(1)
    mask = mask.unsqueeze(-1).float()

    # 20% full-width maintenance, 15% depth distillation, 65% cascaded width distillation
    r = random.random()
    if r < 0.20:
        # Full-width maintenance: normal task loss
        width_mult = 1.0
        preds, _, compute_cost = model.solver(
            programs, initial_states, wisdom, width_mult=1.0
        )
        task_loss = ((preds - intermediate) ** 2 * mask).sum() / mask.sum()
        loss = task_loss + config.compute_cost_weight * compute_cost
        distill_loss_val = 0.0
    elif r < 0.35:
        # Depth distillation: full-width half-depth student <- full-width full-depth teacher
        width_mult = 1.0
        half_depth_mask = torch.zeros(config.max_recursion_depth, dtype=torch.bool, device=config.device)
        half_depth_mask[:2] = True  # depths 0,1 only

        with torch.no_grad():
            teacher_preds, _, _ = model.solver(
                programs, initial_states, wisdom, width_mult=1.0
            )

        student_preds, _, compute_cost = model.solver(
            programs, initial_states, wisdom, width_mult=1.0,
            depth_mask=half_depth_mask
        )

        task_loss = ((student_preds - intermediate) ** 2 * mask).sum() / mask.sum()
        distill_loss = ((student_preds - teacher_preds.detach()) ** 2 * mask).sum() / mask.sum()
        loss = task_loss + config.self_distill_weight * distill_loss
        distill_loss_val = distill_loss.item()
    else:
        # Cascaded width distillation: student learns from next-wider teacher
        # Distribution: w=0.25:25%, w=0.50:22%, w=0.75:18%
        if r < 0.60:
            width_mult = 0.25
            teacher_width = 0.50
        elif r < 0.82:
            width_mult = 0.50
            teacher_width = 0.75
        else:
            width_mult = 0.75
            teacher_width = 1.0

        with torch.no_grad():
            teacher_preds, _, _ = model.solver(
                programs, initial_states, wisdom, width_mult=teacher_width
            )

        student_preds, _, compute_cost = model.solver(
            programs, initial_states, wisdom, width_mult=width_mult
        )

        task_loss = ((student_preds - intermediate) ** 2 * mask).sum() / mask.sum()
        distill_loss = ((student_preds - teacher_preds.detach()) ** 2 * mask).sum() / mask.sum()
        loss = task_loss + config.self_distill_weight * distill_loss
        distill_loss_val = distill_loss.item()

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.solver.parameters(), config.grad_clip)
    optimizer.step()

    return task_loss.item(), distill_loss_val, width_mult


def train_phase8_recovery(model, optimizer):
    """Phase 8: Recovery with all-width emphasis. Maintains elastic quality
    while restoring full-width accuracy. Biased sampling:
    w=1.0:40%, w=0.25:25%, w=0.50:20%, w=0.75:15%.
    Only solver trainable."""
    model.solver.train()
    model.halter.eval()
    model.distiller.eval()
    model.matcher.eval()

    optimizer.zero_grad()

    # 40% full-width, 60% biased toward narrow widths
    r = random.random()
    if r < 0.40:
        width_mult = 1.0
    elif r < 0.65:
        width_mult = 0.25
    elif r < 0.85:
        width_mult = 0.50
    else:
        width_mult = 0.75

    batch = generate_batch(config.batch_size, min_len=1, max_len=4, mixed_prob=0.5)

    programs = batch['program_indices']
    initial_states = batch['initial_states']
    intermediate = batch['intermediate']
    lengths = batch['lengths']

    with torch.no_grad():
        wisdom, _ = model.get_wisdom(programs, initial_states)

    preds, _, compute_cost = model.solver(
        programs, initial_states, wisdom,
        width_mult=width_mult
    )

    batch_size, max_steps, n_vars = preds.shape
    mask = torch.arange(max_steps, device=preds.device).unsqueeze(0) < lengths.unsqueeze(1)
    mask = mask.unsqueeze(-1).float()

    task_loss = ((preds - intermediate) ** 2 * mask).sum() / mask.sum()
    cost_loss = config.compute_cost_weight * compute_cost
    loss = task_loss + cost_loss

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.solver.parameters(), config.grad_clip)
    optimizer.step()

    return task_loss.item(), cost_loss.item(), width_mult

# ============================================================================
# DEVICE PROFILER (new in v0.77)
# ============================================================================

class DeviceProfiler:
    """Benchmarks model at 4 preset width/depth configs, selects largest
    config that fits within latency target. ~100ms runtime, zero trainable params."""

    CONFIGS = {
        'minimal': {'width_mult': 0.25, 'max_depth': 1},
        'small':   {'width_mult': 0.50, 'max_depth': 2},
        'medium':  {'width_mult': 0.75, 'max_depth': 3},
        'full':    {'width_mult': 1.00, 'max_depth': 4},
    }

    def __init__(self, model, latency_target_ms=10.0):
        self.model = model
        self.latency_target_ms = latency_target_ms

    def benchmark(self, width_mult, max_depth, n_runs=10):
        """Benchmark a single config, return mean latency in ms."""
        self.model.eval()
        device = config.device

        depth_mask = torch.zeros(config.max_recursion_depth, dtype=torch.bool, device=device)
        depth_mask[:max_depth] = True

        # Warmup
        with torch.no_grad():
            batch = generate_batch(1, min_len=1, max_len=4)
            wisdom, _ = self.model.get_wisdom(
                batch['program_indices'], batch['initial_states']
            )
            self.model.solver.step(
                batch['initial_states'], batch['program_indices'][:, 0], 0, wisdom,
                width_mult=width_mult, depth_mask=depth_mask
            )

        # Timed runs
        latencies = []
        with torch.no_grad():
            for _ in range(n_runs):
                batch = generate_batch(1, min_len=1, max_len=4)
                wisdom, _ = self.model.get_wisdom(
                    batch['program_indices'], batch['initial_states']
                )
                start = time.perf_counter()
                self.model.solver.step(
                    batch['initial_states'], batch['program_indices'][:, 0], 0, wisdom,
                    width_mult=width_mult, depth_mask=depth_mask
                )
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                end = time.perf_counter()
                latencies.append((end - start) * 1000)

        return np.mean(latencies)

    def select_config(self):
        """Benchmark all configs and select largest within latency target."""
        results = {}
        selected = 'minimal'  # fallback

        for name, cfg in self.CONFIGS.items():
            latency = self.benchmark(cfg['width_mult'], cfg['max_depth'])
            results[name] = latency
            if latency <= self.latency_target_ms:
                selected = name  # keep updating — later (larger) configs override

        return selected, results

# ============================================================================
# EVALUATION
# ============================================================================

def evaluate(model, n_batches=10, mixed_prob=0.0, width_mult=1.0, depth_mask=None):
    model.eval()

    results = {
        'exact': 0, 'total': 0,
        'steps_used': [], 'steps_needed': [],
        'compute_costs': [],
        'by_length': defaultdict(lambda: {'correct': 0, 'total': 0, 'steps': []}),
        'by_family': defaultdict(lambda: {'correct': 0, 'total': 0}),
    }

    with torch.no_grad():
        for _ in range(n_batches):
            batch = generate_batch(config.batch_size, min_len=1, max_len=4, mixed_prob=mixed_prob)

            final_targets = torch.stack([
                batch['intermediate'][b, batch['lengths'][b] - 1]
                for b in range(config.batch_size)
            ])

            predictions, steps_used, compute_cost = model.forward_adaptive(
                batch['program_indices'],
                batch['initial_states'],
                width_mult=width_mult,
                depth_mask=depth_mask
            )

            pred_rounded = predictions.round()
            exact = (pred_rounded == final_targets).all(dim=-1)

            results['exact'] += exact.sum().item()
            results['total'] += config.batch_size
            results['steps_used'].extend(steps_used.cpu().tolist())
            results['steps_needed'].extend(batch['lengths'].cpu().tolist())
            results['compute_costs'].append(compute_cost.item() if torch.is_tensor(compute_cost) else compute_cost)

            for b in range(config.batch_size):
                length = batch['lengths'][b].item()
                family_id = batch['family_ids'][b].item()
                family_name = TASK_FAMILIES[family_id]['name']

                results['by_length'][length]['total'] += 1
                results['by_length'][length]['steps'].append(steps_used[b].item())
                results['by_family'][family_name]['total'] += 1

                if exact[b]:
                    results['by_length'][length]['correct'] += 1
                    results['by_family'][family_name]['correct'] += 1

    results['accuracy'] = results['exact'] / results['total'] * 100
    results['avg_steps_used'] = np.mean(results['steps_used'])
    results['avg_steps_needed'] = np.mean(results['steps_needed'])
    results['avg_compute_cost'] = np.mean(results['compute_costs'])

    for length in results['by_length']:
        r = results['by_length'][length]
        r['accuracy'] = r['correct'] / r['total'] * 100 if r['total'] > 0 else 0
        r['avg_steps'] = np.mean(r['steps']) if r['steps'] else 0

    for family in results['by_family']:
        r = results['by_family'][family]
        r['accuracy'] = r['correct'] / r['total'] * 100 if r['total'] > 0 else 0

    return results


def evaluate_solver_only(model, n_batches=10):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for _ in range(n_batches):
            batch = generate_batch(config.batch_size, min_len=1, max_len=4)
            batch_size = batch['program_indices'].shape[0]

            zero_wisdom = torch.zeros(batch_size, config.d_wisdom, device=config.device)
            preds, _, _ = model.solver(batch['program_indices'], batch['initial_states'], zero_wisdom)

            for b in range(batch_size):
                length = batch['lengths'][b].item()
                pred = preds[b, length - 1].round()
                target = batch['intermediate'][b, length - 1]
                if (pred == target).all():
                    correct += 1
                total += 1

    return correct / total * 100


def evaluate_wisdom_matching(model, n_examples=100):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for family_id in range(config.n_task_families):
            batch = generate_family_batch(family_id, n_examples, prog_len=4)

            first_ops = batch['program_indices'][:, 0]
            _, attention = model.matcher(batch['initial_states'], first_ops, model.wisdom_bank)

            predicted = attention[:, :config.n_task_families].argmax(dim=-1)
            correct += (predicted == family_id).sum().item()
            total += n_examples

    return correct / total * 100


def generate_scaling_batch(batch_size, prog_len, mixed_prob=0.5):
    """Generate a batch with a fixed program length for scaling evaluation."""
    program_indices = []
    initial_states = []
    all_intermediate = []
    lengths = []
    family_ids = []

    all_family_ops = []
    for fam in TASK_FAMILIES.values():
        all_family_ops.extend(fam['ops'])

    for _ in range(batch_size):
        if random.random() < mixed_prob:
            prog_ops = [random.choice(all_family_ops) for _ in range(prog_len)]
            family_id = get_family_id(OP_TO_IDX[prog_ops[0]])
        else:
            family_id = random.randint(0, config.n_task_families - 1)
            ops = TASK_FAMILIES[family_id]['ops']
            prog_ops = [random.choice(ops) for _ in range(prog_len)]

        init = tuple(np.random.randint(0, config.value_range) for _ in range(3))
        prog_indices = [OP_TO_IDX[op] for op in prog_ops]

        states = execute_program(init, prog_ops)

        # No padding — exact length
        program_indices.append(prog_indices)
        initial_states.append(init)
        all_intermediate.append(states[1:])
        lengths.append(prog_len)
        family_ids.append(family_id)

    return {
        'program_indices': torch.tensor(program_indices, dtype=torch.long, device=config.device),
        'initial_states': torch.tensor(initial_states, dtype=torch.float, device=config.device),
        'intermediate': torch.tensor(all_intermediate, dtype=torch.float, device=config.device),
        'lengths': torch.tensor(lengths, dtype=torch.long, device=config.device),
        'family_ids': torch.tensor(family_ids, dtype=torch.long, device=config.device),
    }


def evaluate_scaling(model, n_batches=20):
    """Evaluate at lengths [4, 8, 12, 16] for length generalization."""
    model.eval()

    test_lengths = [4, 8, 12, 16]
    scaling_results = {}

    with torch.no_grad():
        for prog_len in test_lengths:
            results = {
                'exact': 0, 'total': 0,
                'steps_used': [], 'steps_needed': [],
                'compute_costs': [],
                'by_family': defaultdict(lambda: {'correct': 0, 'total': 0}),
            }

            for _ in range(n_batches):
                batch = generate_scaling_batch(config.batch_size, prog_len, mixed_prob=0.5)

                final_targets = batch['intermediate'][:, prog_len - 1]

                predictions, steps_used, compute_cost = model.forward_adaptive(
                    batch['program_indices'],
                    batch['initial_states']
                )

                pred_rounded = predictions.round()
                exact = (pred_rounded == final_targets).all(dim=-1)

                results['exact'] += exact.sum().item()
                results['total'] += config.batch_size
                results['steps_used'].extend(steps_used.cpu().tolist())
                results['steps_needed'].extend([prog_len] * config.batch_size)
                results['compute_costs'].append(compute_cost.item() if torch.is_tensor(compute_cost) else compute_cost)

                for b in range(config.batch_size):
                    family_id = batch['family_ids'][b].item()
                    family_name = TASK_FAMILIES[family_id]['name']
                    results['by_family'][family_name]['total'] += 1
                    if exact[b]:
                        results['by_family'][family_name]['correct'] += 1

            results['accuracy'] = results['exact'] / results['total'] * 100
            results['avg_steps_used'] = np.mean(results['steps_used'])
            results['avg_steps_needed'] = np.mean(results['steps_needed'])
            results['avg_compute_cost'] = np.mean(results['compute_costs'])
            results['avg_depth'] = results['avg_compute_cost'] / max(np.mean(results['steps_used']), 1.0)

            for family in results['by_family']:
                r = results['by_family'][family]
                r['accuracy'] = r['correct'] / r['total'] * 100 if r['total'] > 0 else 0

            scaling_results[prog_len] = results

    return scaling_results


def analyze_recursion_depths(model, n_batches=20):
    """Analyze recursion router depth selections per operation."""
    model.eval()

    op_depths = defaultdict(list)
    family_depths = defaultdict(list)
    length_depths = defaultdict(list)

    with torch.no_grad():
        for _ in range(n_batches):
            batch = generate_batch(config.batch_size, min_len=1, max_len=4, mixed_prob=0.5)
            programs = batch['program_indices']
            initial_states = batch['initial_states']
            lengths = batch['lengths']

            wisdom, _ = model.get_wisdom(programs, initial_states)

            state = initial_states
            for t in range(programs.shape[1]):
                op_idx = programs[:, t]

                # Get router decision (full width for analysis)
                router_logits = model.solver.get_router_logits(state, op_idx)
                selected_depth = router_logits.argmax(dim=-1) + 1  # 1-indexed

                # Step forward to get updated state
                state, _ = model.solver.step(state, op_idx, t, wisdom)

                # Record depths per sample
                for b in range(config.batch_size):
                    prog_len = lengths[b].item()
                    if t >= prog_len:
                        continue  # skip PAD ops

                    op_name = ALL_OPS[op_idx[b].item()]
                    fam_id = get_family_id(op_idx[b].item())
                    if fam_id >= 0:
                        fam_name = TASK_FAMILIES[fam_id]['name']
                    else:
                        fam_name = 'UNKNOWN'

                    depth_val = selected_depth[b].item()
                    op_depths[op_name].append(depth_val)
                    family_depths[fam_name].append(depth_val)
                    length_depths[prog_len].append(depth_val)

    # Compute statistics
    results = {
        'by_op': {},
        'by_family': {},
        'by_length': {},
        'overall_mean_depth': 0.0,
    }

    all_depths = []
    for op_name in sorted(op_depths.keys()):
        depths = op_depths[op_name]
        all_depths.extend(depths)
        results['by_op'][op_name] = {
            'mean_depth': np.mean(depths),
            'depth_distribution': {
                d: depths.count(d) / len(depths) * 100
                for d in range(1, config.max_recursion_depth + 1)
            },
            'count': len(depths),
        }

    for fam_name in sorted(family_depths.keys()):
        depths = family_depths[fam_name]
        results['by_family'][fam_name] = {
            'mean_depth': np.mean(depths),
            'count': len(depths),
        }

    for prog_len in sorted(length_depths.keys()):
        depths = length_depths[prog_len]
        results['by_length'][prog_len] = {
            'mean_depth': np.mean(depths),
            'count': len(depths),
        }

    if all_depths:
        results['overall_mean_depth'] = np.mean(all_depths)

    return results


def evaluate_elastic(model, n_batches=10):
    """Evaluate at grid of width x depth configs for elastic inference quality."""
    model.eval()

    configs = {
        'Full (w=1.0, d=4)':       {'width_mult': 1.0,  'max_depth': 4},
        '75% width (w=0.75, d=4)': {'width_mult': 0.75, 'max_depth': 4},
        '50% width (w=0.5, d=4)':  {'width_mult': 0.5,  'max_depth': 4},
        'Half depth (w=1.0, d=2)': {'width_mult': 1.0,  'max_depth': 2},
        '25% width (w=0.25, d=4)': {'width_mult': 0.25, 'max_depth': 4},
        'Minimal (w=0.25, d=1)':   {'width_mult': 0.25, 'max_depth': 1},
    }

    results = {}

    with torch.no_grad():
        for name, cfg in configs.items():
            depth_mask = torch.zeros(
                config.max_recursion_depth, dtype=torch.bool, device=config.device
            )
            depth_mask[:cfg['max_depth']] = True

            correct = 0
            total = 0

            for _ in range(n_batches):
                batch = generate_batch(
                    config.batch_size, min_len=1, max_len=4, mixed_prob=0.5
                )

                final_targets = torch.stack([
                    batch['intermediate'][b, batch['lengths'][b] - 1]
                    for b in range(config.batch_size)
                ])

                predictions, steps_used, compute_cost = model.forward_adaptive(
                    batch['program_indices'],
                    batch['initial_states'],
                    width_mult=cfg['width_mult'],
                    depth_mask=depth_mask
                )

                pred_rounded = predictions.round()
                exact = (pred_rounded == final_targets).all(dim=-1)
                correct += exact.sum().item()
                total += config.batch_size

            accuracy = correct / total * 100

            results[name] = {
                'accuracy': accuracy,
                'width_mult': cfg['width_mult'],
                'max_depth': cfg['max_depth'],
            }

    return results

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("BroadMind v0.77: Hardware-Adaptive Compute (Elastic Inference)")
    print("Wisdom + Adaptive Compute + Length Gen + MoR + Elastic Width/Depth")
    print("=" * 70)

    print(f"\nDevice: {config.device}")

    model = BroadMindV077(config).to(config.device)
    print(f"Parameters: {model.count_parameters():,}")

    # ========================================================================
    # PHASE 1: Train SOLVER only
    # ========================================================================
    print("\n" + "=" * 70)
    print("PHASE 1: Train SOLVER only (no wisdom)")
    print("=" * 70)

    solver_optimizer = torch.optim.AdamW(
        model.solver.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay
    )

    for i in range(config.n_iterations_phase1):
        batch = generate_batch(config.batch_size)
        task_loss, cost_loss = train_solver_only(model, solver_optimizer, batch)

        if (i + 1) % 500 == 0:
            acc = evaluate_solver_only(model, n_batches=5)
            print(f"[{i+1}/{config.n_iterations_phase1}] Loss: {task_loss:.4f} | "
                  f"Cost: {cost_loss:.4f} | Solver Acc: {acc:.1f}%")

    # ========================================================================
    # PHASE 2: Train WISDOM distillation
    # ========================================================================
    print("\n" + "=" * 70)
    print("PHASE 2: Train WISDOM distillation")
    print("=" * 70)

    wisdom_optimizer = torch.optim.AdamW(
        list(model.distiller.parameters()) + list(model.matcher.parameters()) + list(model.solver.parameters()),
        lr=config.lr,
        weight_decay=config.weight_decay
    )

    for i in range(config.n_iterations_phase2):
        loss = train_wisdom(model, wisdom_optimizer)

        if (i + 1) % 500 == 0:
            wisdom_acc = evaluate_wisdom_matching(model)
            results = evaluate(model, n_batches=5)
            print(f"[{i+1}/{config.n_iterations_phase2}] Wisdom Loss: {loss:.4f} | "
                  f"Wisdom Match: {wisdom_acc:.1f}% | Task Acc: {results['accuracy']:.1f}%")

    # ========================================================================
    # PHASE 2b: Train MATCHER (wisdom routing)
    # ========================================================================
    print("\n" + "=" * 70)
    print("PHASE 2b: Train MATCHER (wisdom routing)")
    print("=" * 70)

    for param in model.solver.parameters():
        param.requires_grad = False
    for param in model.distiller.parameters():
        param.requires_grad = False
    for param in model.halter.parameters():
        param.requires_grad = False
    for param in model.matcher.parameters():
        param.requires_grad = True
    model.wisdom_bank.wisdom_codes.requires_grad = True

    matcher_optimizer = torch.optim.AdamW(
        list(model.matcher.parameters()) + [model.wisdom_bank.wisdom_codes],
        lr=config.lr,
        weight_decay=config.weight_decay
    )

    for i in range(config.n_iterations_phase2b):
        batch = generate_batch(config.batch_size)
        loss = train_matcher(model, matcher_optimizer, batch)

        if (i + 1) % 200 == 0:
            wisdom_acc = evaluate_wisdom_matching(model)
            print(f"[{i+1}/{config.n_iterations_phase2b}] Match Loss: {loss:.4f} | Wisdom Match: {wisdom_acc:.1f}%")

    # ========================================================================
    # PHASE 3: Train HALTER only
    # ========================================================================
    print("\n" + "=" * 70)
    print("PHASE 3: Train HALTER only (others frozen)")
    print("=" * 70)

    for param in model.parameters():
        param.requires_grad = False
    for param in model.halter.parameters():
        param.requires_grad = True

    halter_optimizer = torch.optim.AdamW(
        model.halter.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay
    )

    for i in range(config.n_iterations_phase3):
        batch = generate_batch(config.batch_size)
        loss = train_halter_only(model, halter_optimizer, batch)

        if (i + 1) % 500 == 0:
            results = evaluate(model, n_batches=5)
            print(f"[{i+1}/{config.n_iterations_phase3}] Halt Loss: {loss:.4f} | "
                  f"Acc: {results['accuracy']:.1f}% | Steps: {results['avg_steps_used']:.1f}/{results['avg_steps_needed']:.1f}")

    # ========================================================================
    # PHASE 4: Fine-tune EVERYTHING
    # ========================================================================
    print("\n" + "=" * 70)
    print("PHASE 4: Fine-tune everything")
    print("=" * 70)

    for param in model.parameters():
        param.requires_grad = True

    full_optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr_fine,
        weight_decay=config.weight_decay
    )

    for i in range(config.n_iterations_phase4):
        batch = generate_batch(config.batch_size, mixed_prob=0.3)
        task_loss, halt_loss = train_end_to_end(model, full_optimizer, batch)

        if (i + 1) % 500 == 0:
            results = evaluate(model, n_batches=5)
            mixed_results = evaluate(model, n_batches=5, mixed_prob=0.5)
            print(f"[{i+1}/{config.n_iterations_phase4}] Task: {task_loss:.4f} | Halt: {halt_loss:.4f} | "
                  f"Acc: {results['accuracy']:.1f}% | Mixed: {mixed_results['accuracy']:.1f}% | "
                  f"Steps: {results['avg_steps_used']:.1f}/{results['avg_steps_needed']:.1f}")

    # ========================================================================
    # PHASE 5: Length generalization fine-tuning
    # ========================================================================
    print("\n" + "=" * 70)
    print("PHASE 5: Length generalization (noise injection)")
    print("=" * 70)

    # Only solver trainable
    for param in model.parameters():
        param.requires_grad = False
    for param in model.solver.parameters():
        param.requires_grad = True

    phase5_optimizer = torch.optim.AdamW(
        model.solver.parameters(),
        lr=5e-5,
        weight_decay=config.weight_decay
    )

    for i in range(config.n_iterations_phase5):
        task_loss, cons_loss, noise, cost_loss = train_phase5(
            model, phase5_optimizer, i, config.n_iterations_phase5
        )

        if (i + 1) % 100 == 0:
            results = evaluate(model, n_batches=5)
            print(f"[{i+1}/{config.n_iterations_phase5}] Task: {task_loss:.4f} | "
                  f"Cons: {cons_loss:.4f} | Noise: {noise:.3f} | "
                  f"Cost: {cost_loss:.4f} | Acc: {results['accuracy']:.1f}%")

    # ========================================================================
    # PHASE 6: Combined width + depth elastic training (new in v0.77)
    # ========================================================================
    print("\n" + "=" * 70)
    print("PHASE 6: Combined width + depth elastic training")
    print("Width: w=0.25:30%, w=0.50:30%, w=0.75:20%, w=1.0:20%")
    print("Depth: 78% full, 12% half, 6% single, 4% three (22% restricted)")
    print("=" * 70)

    # Initialize sub-width LayerNorms from full-width norm
    init_subnorm_from_full(model.solver)
    print("Initialized sub-width LayerNorms from full-width norms")

    # Only solver trainable
    for param in model.parameters():
        param.requires_grad = False
    for param in model.solver.parameters():
        param.requires_grad = True

    elastic_optimizer = torch.optim.AdamW(
        model.solver.parameters(),
        lr=config.lr_elastic,
        weight_decay=config.weight_decay
    )

    for i in range(config.n_iterations_phase6):
        task_loss, cost_loss, w, d = train_phase6_elastic(model, elastic_optimizer)

        if (i + 1) % 500 == 0:
            results_full = evaluate(model, n_batches=5)
            print(f"[{i+1}/{config.n_iterations_phase6}] Loss: {task_loss:.4f} | "
                  f"Cost: {cost_loss:.4f} | w={w:.2f} d={d} | "
                  f"Full Acc: {results_full['accuracy']:.1f}%")

    # Quick elastic check after Phase 6
    print("\n  Post-Phase 6 elastic snapshot:")
    p6_elastic = evaluate_elastic(model, n_batches=5)
    for name, er in p6_elastic.items():
        print(f"    {name}: {er['accuracy']:.1f}%")

    # ========================================================================
    # PHASE 7: Cascaded self-distillation (new in v0.77)
    # ========================================================================
    print("\n" + "=" * 70)
    print("PHASE 7: Cascaded distillation + depth distillation")
    print("65% width cascade (w=0.25<-0.50, w=0.50<-0.75, w=0.75<-1.0)")
    print("15% depth distill (half-depth <- full-depth), 20% full maintenance")
    print("=" * 70)

    distill_optimizer = torch.optim.AdamW(
        model.solver.parameters(),
        lr=config.lr_elastic,
        weight_decay=config.weight_decay
    )

    for i in range(config.n_iterations_phase7):
        task_loss, distill_loss, w = train_phase7_distill(model, distill_optimizer)

        if (i + 1) % 300 == 0:
            results_full = evaluate(model, n_batches=5)
            print(f"[{i+1}/{config.n_iterations_phase7}] Task: {task_loss:.4f} | "
                  f"Distill: {distill_loss:.4f} | w={w:.2f} | "
                  f"Full Acc: {results_full['accuracy']:.1f}%")

    # Quick elastic check after Phase 7
    print("\n  Post-Phase 7 elastic snapshot:")
    p7_elastic = evaluate_elastic(model, n_batches=5)
    for name, er in p7_elastic.items():
        print(f"    {name}: {er['accuracy']:.1f}%")

    # ========================================================================
    # PHASE 8: Recovery with width emphasis (new in v0.77)
    # ========================================================================
    print("\n" + "=" * 70)
    print("PHASE 8: Recovery (40% full, 60% narrow-biased)")
    print("=" * 70)

    recovery_optimizer = torch.optim.AdamW(
        model.solver.parameters(),
        lr=config.lr_elastic,
        weight_decay=config.weight_decay
    )

    for i in range(config.n_iterations_phase8):
        task_loss, cost_loss, w = train_phase8_recovery(model, recovery_optimizer)

        if (i + 1) % 100 == 0:
            results_full = evaluate(model, n_batches=5)
            print(f"[{i+1}/{config.n_iterations_phase8}] Loss: {task_loss:.4f} | "
                  f"Cost: {cost_loss:.4f} | w={w:.2f} | "
                  f"Full Acc: {results_full['accuracy']:.1f}%")

    # Re-enable all params for eval
    for param in model.parameters():
        param.requires_grad = True

    # ========================================================================
    # FINAL EVALUATION (in-distribution, lengths 1-4)
    # ========================================================================
    print("\n" + "=" * 70)
    print("FINAL EVALUATION (in-distribution, lengths 1-4)")
    print("=" * 70)

    results = evaluate(model, n_batches=50)
    mixed_results = evaluate(model, n_batches=50, mixed_prob=0.5)
    wisdom_acc = evaluate_wisdom_matching(model, n_examples=200)

    print(f"\nOverall Accuracy: {results['accuracy']:.1f}%")
    print(f"Mixed-Program Accuracy: {mixed_results['accuracy']:.1f}%")
    print(f"Wisdom Matching: {wisdom_acc:.1f}%")
    print(f"Steps Used: {results['avg_steps_used']:.2f}")
    print(f"Steps Needed: {results['avg_steps_needed']:.2f}")
    print(f"Avg Compute Cost: {results['avg_compute_cost']:.3f}")

    print("\nBy Task Family:")
    for family, r in results['by_family'].items():
        print(f"  {family}: {r['accuracy']:.1f}%")

    print("\nBy Program Length:")
    for length in sorted(results['by_length'].keys()):
        r = results['by_length'][length]
        match = "[Y]" if abs(r['avg_steps'] - length) < 0.5 else "[N]"
        print(f"  Length {length}: {r['accuracy']:.1f}% acc, {r['avg_steps']:.1f} steps {match}")

    # ========================================================================
    # RECURSION DEPTH ANALYSIS (MoR-specific)
    # ========================================================================
    print("\n" + "=" * 70)
    print("RECURSION DEPTH ANALYSIS (MoR)")
    print("=" * 70)

    depth_results = analyze_recursion_depths(model, n_batches=20)

    print("\n  Operation          | Avg Depth")
    print("  -------------------+----------")
    for op_name, stats in sorted(depth_results['by_op'].items(), key=lambda x: -x[1]['mean_depth']):
        print(f"  {op_name:<19s} | {stats['mean_depth']:>6.2f}")

    print("\n  Family             | Avg Depth")
    print("  -------------------+----------")
    for fam_name, stats in sorted(depth_results['by_family'].items(), key=lambda x: -x[1]['mean_depth']):
        print(f"  {fam_name:<19s} | {stats['mean_depth']:>6.2f}")

    print(f"\n  Overall Mean Depth: {depth_results['overall_mean_depth']:.2f}")

    # ========================================================================
    # SCALING EVALUATION (lengths 4, 8, 12, 16)
    # ========================================================================
    print("\n" + "=" * 70)
    print("SCALING EVALUATION (length generalization)")
    print("=" * 70)

    scaling_results = evaluate_scaling(model, n_batches=20)

    print("\n  Length | Accuracy | Steps Used | Steps Needed | Avg Depth")
    print("  -------+----------+------------+--------------+----------")
    for prog_len in [4, 8, 12, 16]:
        sr = scaling_results[prog_len]
        print(f"  {prog_len:>5d}  | {sr['accuracy']:>6.1f}%  | "
              f"{sr['avg_steps_used']:>10.1f} | {sr['avg_steps_needed']:>10.1f}   | "
              f"{sr['avg_depth']:>6.2f}")

    print("\n  Per-family breakdown at each length:")
    for prog_len in [4, 8, 12, 16]:
        sr = scaling_results[prog_len]
        fam_strs = []
        for fam_name in ['ACCUMULATE', 'TRANSFER', 'COMPARE', 'DECREMENT']:
            if fam_name in sr['by_family']:
                fam_strs.append(f"{fam_name}={sr['by_family'][fam_name]['accuracy']:.0f}%")
        print(f"    Length {prog_len:>2d}: {', '.join(fam_strs)}")

    # ========================================================================
    # ELASTIC EVALUATION (new in v0.77)
    # ========================================================================
    print("\n" + "=" * 70)
    print("ELASTIC EVALUATION (width x depth grid)")
    print("=" * 70)

    elastic_results = evaluate_elastic(model, n_batches=20)

    print("\n  Config                    | Accuracy | Target")
    print("  --------------------------+----------+-------")
    targets = {
        'Full (w=1.0, d=4)':       '>99%',
        '75% width (w=0.75, d=4)': '>97%',
        '50% width (w=0.5, d=4)':  '>95%',
        'Half depth (w=1.0, d=2)': '>90%',
        '25% width (w=0.25, d=4)': '>85%',
        'Minimal (w=0.25, d=1)':   '>80%',
    }
    for name, er in elastic_results.items():
        print(f"  {name:<26s} | {er['accuracy']:>6.1f}%  | {targets[name]}")

    # ========================================================================
    # DEVICE PROFILER (new in v0.77)
    # ========================================================================
    print("\n" + "=" * 70)
    print("DEVICE PROFILER")
    print("=" * 70)

    profiler = DeviceProfiler(model, latency_target_ms=config.latency_target_ms)
    selected_config, latency_results = profiler.select_config()

    print(f"\n  Latency target: {config.latency_target_ms:.1f} ms")
    print(f"\n  Config   | Latency (ms)")
    print(f"  ---------+-------------")
    for name, latency in latency_results.items():
        marker = " <-- selected" if name == selected_config else ""
        print(f"  {name:<8s} | {latency:>8.2f} ms{marker}")
    print(f"\n  Selected config: {selected_config}")
    sel_cfg = DeviceProfiler.CONFIGS[selected_config]
    print(f"  Width: {sel_cfg['width_mult']:.0%}, Max depth: {sel_cfg['max_depth']}")

    # ========================================================================
    # THE COMPLETE BROADMIND v0.77 TEST
    # ========================================================================
    print("\n" + "=" * 70)
    print("THE COMPLETE BROADMIND v0.77 TEST")
    print("=" * 70)

    l1 = results['by_length'][1]
    l4 = results['by_length'][4]

    accurate = results['accuracy'] > 95
    mixed_accurate = mixed_results['accuracy'] > 90
    wise = wisdom_acc > 90
    efficient_easy = l1['avg_steps'] < 1.5
    efficient_hard = l4['avg_steps'] > 3.5
    steps_match = abs(results['avg_steps_used'] - results['avg_steps_needed']) < 0.3

    # Scaling criteria
    scale_4 = scaling_results[4]['accuracy'] > 99
    scale_8 = scaling_results[8]['accuracy'] > 90
    scale_12 = scaling_results[12]['accuracy'] > 85
    scale_16 = scaling_results[16]['accuracy'] > 80

    # MoR criteria
    compare_depth = depth_results['by_family'].get('COMPARE', {}).get('mean_depth', 0)
    transfer_depth = depth_results['by_family'].get('TRANSFER', {}).get('mean_depth', 0)
    acc_depth = depth_results['by_family'].get('ACCUMULATE', {}).get('mean_depth', 0)
    dec_depth = depth_results['by_family'].get('DECREMENT', {}).get('mean_depth', 0)
    simple_depth = (acc_depth + dec_depth) / 2.0 if (acc_depth + dec_depth) > 0 else 0

    depth_hierarchy = compare_depth > transfer_depth > simple_depth
    efficient_routing = depth_results['overall_mean_depth'] < 3.0

    # Elastic criteria (v0.77)
    elastic_full = elastic_results['Full (w=1.0, d=4)']['accuracy'] > 99
    elastic_75w = elastic_results['75% width (w=0.75, d=4)']['accuracy'] > 97
    elastic_half_w = elastic_results['50% width (w=0.5, d=4)']['accuracy'] > 95
    elastic_half_d = elastic_results['Half depth (w=1.0, d=2)']['accuracy'] > 90
    elastic_25w = elastic_results['25% width (w=0.25, d=4)']['accuracy'] > 85
    elastic_min = elastic_results['Minimal (w=0.25, d=1)']['accuracy'] > 80

    all_pass = (accurate and mixed_accurate and wise and
                efficient_easy and efficient_hard and steps_match)
    all_scale = scale_4 and scale_8 and scale_12 and scale_16
    all_mor = depth_hierarchy and efficient_routing
    all_elastic = elastic_full and elastic_75w and elastic_half_w and elastic_half_d and elastic_25w and elastic_min

    print(f"""
[+] Wisdom Distillation ("keeps the juice"):
    Wisdom Matching: {wisdom_acc:.1f}% {'[PASS]' if wise else '[FAIL]'}

[+] Adaptive Compute ("sips power"):
    Easy (len=1): {l1['avg_steps']:.1f} steps {'[PASS]' if efficient_easy else '[FAIL]'}
    Hard (len=4): {l4['avg_steps']:.1f} steps {'[PASS]' if efficient_hard else '[FAIL]'}
    Steps Match: {results['avg_steps_used']:.2f} ~ {results['avg_steps_needed']:.2f} {'[PASS]' if steps_match else '[FAIL]'}

[+] Overall Performance:
    Accuracy: {results['accuracy']:.1f}% {'[PASS]' if accurate else '[FAIL]'}
    Mixed-Program: {mixed_results['accuracy']:.1f}% {'[PASS]' if mixed_accurate else '[FAIL]'}

[+] Length Generalization:
    Length  4: {scaling_results[4]['accuracy']:.1f}% {'[PASS]' if scale_4 else '[FAIL]'} (target >99%)
    Length  8: {scaling_results[8]['accuracy']:.1f}% {'[PASS]' if scale_8 else '[FAIL]'} (target >90%)
    Length 12: {scaling_results[12]['accuracy']:.1f}% {'[PASS]' if scale_12 else '[FAIL]'} (target >85%)
    Length 16: {scaling_results[16]['accuracy']:.1f}% {'[PASS]' if scale_16 else '[FAIL]'} (target >80%)

[+] Mixture of Recursions (MoR):
    Depth Hierarchy (COMPARE > TRANSFER > SIMPLE):
      COMPARE:  {compare_depth:.2f}
      TRANSFER: {transfer_depth:.2f}
      SIMPLE:   {simple_depth:.2f}
      {'[PASS]' if depth_hierarchy else '[FAIL]'}
    Efficient Routing (mean < 3.0): {depth_results['overall_mean_depth']:.2f} {'[PASS]' if efficient_routing else '[FAIL]'}

[+] Elastic Inference (v0.77):
    Full (w=1.0, d=4):       {elastic_results['Full (w=1.0, d=4)']['accuracy']:.1f}% {'[PASS]' if elastic_full else '[FAIL]'} (target >99%)
    75% width (w=0.75, d=4): {elastic_results['75% width (w=0.75, d=4)']['accuracy']:.1f}% {'[PASS]' if elastic_75w else '[FAIL]'} (target >97%)
    50% width (w=0.5, d=4):  {elastic_results['50% width (w=0.5, d=4)']['accuracy']:.1f}% {'[PASS]' if elastic_half_w else '[FAIL]'} (target >95%)
    Half depth (w=1.0, d=2): {elastic_results['Half depth (w=1.0, d=2)']['accuracy']:.1f}% {'[PASS]' if elastic_half_d else '[FAIL]'} (target >90%)
    25% width (w=0.25, d=4): {elastic_results['25% width (w=0.25, d=4)']['accuracy']:.1f}% {'[PASS]' if elastic_25w else '[FAIL]'} (target >85%)
    Minimal (w=0.25, d=1):   {elastic_results['Minimal (w=0.25, d=1)']['accuracy']:.1f}% {'[PASS]' if elastic_min else '[FAIL]'} (target >80%)
    Device Profile: {selected_config} (w={sel_cfg['width_mult']:.0%}, d={sel_cfg['max_depth']})

BROADMIND v0.74 COMPAT: {'PASS' if all_pass else 'PARTIAL'}
SCALING STATUS:          {'PASS' if all_scale else 'PARTIAL'}
MoR STATUS:              {'PASS' if all_mor else 'PARTIAL'}
ELASTIC STATUS:          {'PASS' if all_elastic else 'PARTIAL'}
BROADMIND v0.77 STATUS:  {'COMPLETE' if all_pass and all_scale and all_mor and all_elastic else 'PARTIAL'}
""")

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"""
BroadMind v0.77: Hardware-Adaptive Compute (Elastic Inference)
--------------------------------------------------------------
Parameters: {model.count_parameters():,}

Capabilities:
  [+] Latent program induction (invents reasoning circuits)
  [+] Wisdom distillation (keeps the juice, forgets the fluff)
  [+] Adaptive compute (sips power, drops insight)
  [+] Mixed-program execution (ops from any family)
  [+] Length generalization (sinusoidal encoding + noise injection)
  [+] Mixture of Recursions (adaptive inner depth per operation)
  [+] Elastic inference (25/50/75/100% width, 1-4 depth, auto-config)

Results (in-distribution, lengths 1-4):
  Accuracy:        {results['accuracy']:.1f}%
  Mixed-Program:   {mixed_results['accuracy']:.1f}%
  Wisdom Match:    {wisdom_acc:.1f}%
  Steps Used:      {results['avg_steps_used']:.2f}
  Steps Needed:    {results['avg_steps_needed']:.2f}
  Avg Compute Cost: {results['avg_compute_cost']:.3f}

Per-Family:
  ACCUMULATE: {results['by_family']['ACCUMULATE']['accuracy']:.0f}%
  TRANSFER:   {results['by_family']['TRANSFER']['accuracy']:.0f}%
  COMPARE:    {results['by_family']['COMPARE']['accuracy']:.0f}%
  DECREMENT:  {results['by_family']['DECREMENT']['accuracy']:.0f}%

Per-Length (in-distribution):
  Len 1: {results['by_length'][1]['accuracy']:.0f}% @ {results['by_length'][1]['avg_steps']:.1f} steps
  Len 2: {results['by_length'][2]['accuracy']:.0f}% @ {results['by_length'][2]['avg_steps']:.1f} steps
  Len 3: {results['by_length'][3]['accuracy']:.0f}% @ {results['by_length'][3]['avg_steps']:.1f} steps
  Len 4: {results['by_length'][4]['accuracy']:.0f}% @ {results['by_length'][4]['avg_steps']:.1f} steps

Recursion Depth (MoR):
  Overall Mean Depth: {depth_results['overall_mean_depth']:.2f}
  ACCUMULATE: {depth_results['by_family'].get('ACCUMULATE', {}).get('mean_depth', 0):.2f}
  TRANSFER:   {depth_results['by_family'].get('TRANSFER', {}).get('mean_depth', 0):.2f}
  COMPARE:    {depth_results['by_family'].get('COMPARE', {}).get('mean_depth', 0):.2f}
  DECREMENT:  {depth_results['by_family'].get('DECREMENT', {}).get('mean_depth', 0):.2f}

Elastic Inference (v0.77):
  Full (w=1.0, d=4):       {elastic_results['Full (w=1.0, d=4)']['accuracy']:.1f}%
  75% width (w=0.75, d=4): {elastic_results['75% width (w=0.75, d=4)']['accuracy']:.1f}%
  50% width (w=0.5, d=4):  {elastic_results['50% width (w=0.5, d=4)']['accuracy']:.1f}%
  Half depth (w=1.0, d=2): {elastic_results['Half depth (w=1.0, d=2)']['accuracy']:.1f}%
  25% width (w=0.25, d=4): {elastic_results['25% width (w=0.25, d=4)']['accuracy']:.1f}%
  Minimal (w=0.25, d=1):   {elastic_results['Minimal (w=0.25, d=1)']['accuracy']:.1f}%
  Device Profile: {selected_config}

Scaling (out-of-distribution):
  Length  4: {scaling_results[4]['accuracy']:.1f}% acc, {scaling_results[4]['avg_steps_used']:.1f} steps, {scaling_results[4].get('avg_depth', 0):.2f} avg depth
  Length  8: {scaling_results[8]['accuracy']:.1f}% acc, {scaling_results[8]['avg_steps_used']:.1f} steps, {scaling_results[8].get('avg_depth', 0):.2f} avg depth
  Length 12: {scaling_results[12]['accuracy']:.1f}% acc, {scaling_results[12]['avg_steps_used']:.1f} steps, {scaling_results[12].get('avg_depth', 0):.2f} avg depth
  Length 16: {scaling_results[16]['accuracy']:.1f}% acc, {scaling_results[16]['avg_steps_used']:.1f} steps, {scaling_results[16].get('avg_depth', 0):.2f} avg depth
""")

    torch.save({
        'model_state_dict': model.state_dict(),
        'accuracy': results['accuracy'],
        'mixed_accuracy': mixed_results['accuracy'],
        'wisdom_accuracy': wisdom_acc,
        'avg_steps_used': results['avg_steps_used'],
        'scaling_results': {k: v['accuracy'] for k, v in scaling_results.items()},
        'depth_results': {
            'overall_mean_depth': depth_results['overall_mean_depth'],
            'by_family': {k: v['mean_depth'] for k, v in depth_results['by_family'].items()},
            'by_op': {k: v['mean_depth'] for k, v in depth_results['by_op'].items()},
        },
        'elastic_results': {k: v['accuracy'] for k, v in elastic_results.items()},
        'device_profile': {
            'selected': selected_config,
            'latencies': latency_results,
        },
    }, 'broadmind_v077_elastic.pt')
    print("Saved to broadmind_v077_elastic.pt")

    return model, results, scaling_results, depth_results, elastic_results


if __name__ == "__main__":
    model, results, scaling_results, depth_results, elastic_results = main()
