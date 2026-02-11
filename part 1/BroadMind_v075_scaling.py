"""
BroadMind v0.75: Task Scaling / Length Generalization
=====================================================

Builds on v0.74 (Wisdom Distillation + Adaptive Compute) with:
- Sinusoidal step encodings (replaces learned embeddings)
- Length-agnostic Halter (mean-pooled program encoding)
- Gaussian noise injection for length generalization (Phase 5)
- Evaluation at lengths 4, 8, 12, 16

Key insight (ICLR 2025): smaller capacity-constrained models have a
provable advantage for length generalization. v0.75 is ~421K params
(down from v0.74's ~477K).
"""

import math
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
# SOLVER (wisdom-guided, sinusoidal step encoding, noise injection)
# ============================================================================

class Solver(nn.Module):
    """Wisdom-guided program executor with sinusoidal step encoding."""

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.state_encoder = nn.Sequential(
            nn.Linear(config.n_variables, config.d_model),
            nn.LayerNorm(config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )

        self.op_embedding = nn.Embedding(N_OPS, config.d_model)
        # step_embedding REMOVED — replaced by sinusoidal_step_encoding

        # Wisdom integration
        self.wisdom_encoder = nn.Sequential(
            nn.Linear(config.d_wisdom, config.d_model // 2),
            nn.GELU(),
        )

        self.latent_gen = nn.Sequential(
            nn.Linear(config.d_model * 2 + config.d_model // 4 + config.d_model // 2, config.d_model),
            nn.LayerNorm(config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, config.d_latent),
        )

        self.comparison_enc = nn.Sequential(
            nn.Linear(6, config.d_model // 4),
            nn.GELU(),
        )

        self.latent_enc = nn.Sequential(
            nn.Linear(config.d_latent, config.d_model),
            nn.LayerNorm(config.d_model),
            nn.GELU(),
        )

        self.executor = nn.Sequential(
            nn.Linear(config.d_model * 2 + config.d_model // 4, config.d_model),
            nn.LayerNorm(config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.n_variables),
        )

        self.predictor = nn.Linear(config.n_variables, config.n_variables)

    def step(self, state, op_idx, step_num, wisdom, training_noise_std=0.0):
        batch_size = state.shape[0]

        state_enc = self.state_encoder(state)
        op_enc = self.op_embedding(op_idx)

        # Sinusoidal step encoding (d_model // 4 = 48)
        step_enc = sinusoidal_step_encoding(
            step_num, self.config.d_model // 4, batch_size, state.device
        )

        wisdom_enc = self.wisdom_encoder(wisdom)

        # Generate latent (wisdom-guided!)
        latent = self.latent_gen(torch.cat([state_enc, op_enc, step_enc, wisdom_enc], dim=-1))

        x, y, z = state[:, 0], state[:, 1], state[:, 2]
        comp = torch.stack([
            (x > y).float(), (y > z).float(), (z > x).float(),
            (x - y) / self.config.comparison_scale,
            (y - z) / self.config.comparison_scale,
            (z - x) / self.config.comparison_scale,
        ], dim=-1)
        comp_enc = self.comparison_enc(comp)

        latent_enc = self.latent_enc(latent)

        delta = self.executor(torch.cat([state_enc, latent_enc, comp_enc], dim=-1))
        new_state = state + delta

        # Noise injection for length generalization
        if training_noise_std > 0.0 and self.training:
            new_state = new_state + torch.randn_like(new_state) * training_noise_std

        return new_state

    def forward(self, programs, initial_states, wisdom, training_noise_std=0.0):
        n_steps = programs.shape[1]

        state = initial_states
        all_preds = []
        all_states = []

        for t in range(n_steps):
            op_idx = programs[:, t]
            state = self.step(state, op_idx, t, wisdom, training_noise_std=training_noise_std)
            pred = self.predictor(state)
            all_preds.append(pred)
            all_states.append(state)

        return torch.stack(all_preds, dim=1), torch.stack(all_states, dim=1)

# ============================================================================
# HALTER (length-agnostic, mean-pooled program encoding)
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

        # step_embedding REMOVED — replaced by sinusoidal_step_encoding (d_model//2 = 96)

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

class BroadMindV075(nn.Module):
    """
    BroadMind v0.75: Task Scaling / Length Generalization

    Builds on v0.74 with:
    - Sinusoidal step encodings (generalizes to any length)
    - Length-agnostic Halter (mean-pooled)
    - Gaussian noise injection (Phase 5)
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Wisdom components
        self.distiller = WisdomDistiller(config)
        self.wisdom_bank = WisdomBank(config)
        self.matcher = WisdomMatcher(config)

        # Solver (wisdom-guided)
        self.solver = Solver(config)

        # Halter (adaptive compute)
        self.halter = Halter(config)

    def get_wisdom(self, programs, initial_states):
        """Get wisdom for batch (via matching)."""
        first_ops = programs[:, 0]
        wisdom, attention = self.matcher(initial_states, first_ops, self.wisdom_bank)
        return wisdom, attention

    def forward_all_steps(self, programs, initial_states, wisdom=None, training_noise_std=0.0):
        """Run all steps, return predictions and halt logits."""
        if wisdom is None:
            wisdom, _ = self.get_wisdom(programs, initial_states)

        preds, states = self.solver(programs, initial_states, wisdom,
                                    training_noise_std=training_noise_std)

        halt_logits = []
        for t in range(programs.shape[1]):
            logit = self.halter(programs, t)
            halt_logits.append(logit)

        return preds, torch.stack(halt_logits, dim=1), wisdom

    def forward_adaptive(self, programs, initial_states, training_noise_std=0.0):
        """Forward with adaptive halting."""
        batch_size = programs.shape[0]
        max_steps = programs.shape[1]

        # Get wisdom
        wisdom, _ = self.get_wisdom(programs, initial_states)

        state = initial_states
        halted = torch.zeros(batch_size, dtype=torch.bool, device=state.device)
        final_states = state.clone()
        steps_used = torch.zeros(batch_size, device=state.device)

        for t in range(max_steps):
            if halted.all():
                break

            op_idx = programs[:, t]
            new_state = self.solver.step(state, op_idx, t, wisdom,
                                         training_noise_std=training_noise_std)

            state = torch.where(halted.unsqueeze(1), state, new_state)

            halt_logit = self.halter(programs, t)
            halt_prob = torch.sigmoid(halt_logit).squeeze(-1)

            should_halt = (halt_prob > self.config.halt_threshold) & ~halted

            final_states = torch.where(should_halt.unsqueeze(1), state, final_states)
            steps_used = steps_used + (~halted).float()
            halted = halted | should_halt

        final_states = torch.where(~halted.unsqueeze(1), state, final_states)

        pred = self.solver.predictor(final_states)
        return pred, steps_used

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

# ============================================================================
# TRAINING
# ============================================================================

def train_solver_only(model, optimizer, batch):
    """Phase 1: Train solver with zero wisdom."""
    model.train()
    optimizer.zero_grad()

    programs = batch['program_indices']
    initial_states = batch['initial_states']
    intermediate = batch['intermediate']
    lengths = batch['lengths']

    batch_size = programs.shape[0]
    zero_wisdom = torch.zeros(batch_size, config.d_wisdom, device=config.device)

    preds, _ = model.solver(programs, initial_states, zero_wisdom)

    batch_size, max_steps, n_vars = preds.shape
    mask = torch.arange(max_steps, device=preds.device).unsqueeze(0) < lengths.unsqueeze(1)
    mask = mask.unsqueeze(-1).float()

    loss = ((preds - intermediate) ** 2 * mask).sum() / mask.sum()

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
    optimizer.step()

    return loss.item()


def train_wisdom(model, optimizer):
    """Phase 2: Train wisdom distillation."""
    model.train()
    optimizer.zero_grad()

    total_loss = 0.0

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

        preds, _ = model.solver(test_batch['program_indices'], test_batch['initial_states'], wisdom_expanded)

        loss = F.mse_loss(preds, test_batch['intermediate'])
        total_loss = total_loss + loss

    total_loss.backward()
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
    """Phase 3: Train halter (solver frozen)."""
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
    """Phase 4: Fine-tune everything."""
    model.train()
    optimizer.zero_grad()

    programs = batch['program_indices']
    initial_states = batch['initial_states']
    intermediate = batch['intermediate']
    lengths = batch['lengths']

    preds, halt_logits, wisdom = model.forward_all_steps(programs, initial_states)

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

    loss = task_loss + 0.5 * halt_loss + 0.5 * match_loss

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
    optimizer.step()

    return task_loss.item(), halt_loss.item()


def train_phase5(model, optimizer, iteration, total_iterations):
    """Phase 5: Length generalization fine-tuning with Gaussian noise.

    - Solver only trainable, lr=5e-5
    - Training data: still length 1-4 (in-distribution)
    - Gaussian noise on recurrent state, linearly annealed 0.05 -> 0.02
    - Mixed programs at 50%
    - Consistency loss: run batch twice (with/without noise), penalize divergence
    """
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
    preds_noisy, _ = model.solver(programs, initial_states, wisdom,
                                   training_noise_std=noise_std)

    # Task loss (masked)
    batch_size, max_steps, n_vars = preds_noisy.shape
    mask = torch.arange(max_steps, device=preds_noisy.device).unsqueeze(0) < lengths.unsqueeze(1)
    mask = mask.unsqueeze(-1).float()
    task_loss = ((preds_noisy - intermediate) ** 2 * mask).sum() / mask.sum()

    # Consistency loss: run WITHOUT noise, penalize divergence
    with torch.no_grad():
        preds_clean, _ = model.solver(programs, initial_states, wisdom,
                                       training_noise_std=0.0)

    consistency_loss = ((preds_noisy - preds_clean.detach()) ** 2 * mask).sum() / mask.sum()

    loss = task_loss + 0.1 * consistency_loss

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.solver.parameters(), config.grad_clip)
    optimizer.step()

    return task_loss.item(), consistency_loss.item(), noise_std

# ============================================================================
# EVALUATION
# ============================================================================

def evaluate(model, n_batches=10, mixed_prob=0.0):
    model.eval()

    results = {
        'exact': 0, 'total': 0,
        'steps_used': [], 'steps_needed': [],
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

            predictions, steps_used = model.forward_adaptive(
                batch['program_indices'],
                batch['initial_states']
            )

            pred_rounded = predictions.round()
            exact = (pred_rounded == final_targets).all(dim=-1)

            results['exact'] += exact.sum().item()
            results['total'] += config.batch_size
            results['steps_used'].extend(steps_used.cpu().tolist())
            results['steps_needed'].extend(batch['lengths'].cpu().tolist())

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
            preds, _ = model.solver(batch['program_indices'], batch['initial_states'], zero_wisdom)

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
                'by_family': defaultdict(lambda: {'correct': 0, 'total': 0}),
            }

            for _ in range(n_batches):
                batch = generate_scaling_batch(config.batch_size, prog_len, mixed_prob=0.5)

                final_targets = batch['intermediate'][:, prog_len - 1]

                predictions, steps_used = model.forward_adaptive(
                    batch['program_indices'],
                    batch['initial_states']
                )

                pred_rounded = predictions.round()
                exact = (pred_rounded == final_targets).all(dim=-1)

                results['exact'] += exact.sum().item()
                results['total'] += config.batch_size
                results['steps_used'].extend(steps_used.cpu().tolist())
                results['steps_needed'].extend([prog_len] * config.batch_size)

                for b in range(config.batch_size):
                    family_id = batch['family_ids'][b].item()
                    family_name = TASK_FAMILIES[family_id]['name']
                    results['by_family'][family_name]['total'] += 1
                    if exact[b]:
                        results['by_family'][family_name]['correct'] += 1

            results['accuracy'] = results['exact'] / results['total'] * 100
            results['avg_steps_used'] = np.mean(results['steps_used'])
            results['avg_steps_needed'] = np.mean(results['steps_needed'])

            for family in results['by_family']:
                r = results['by_family'][family]
                r['accuracy'] = r['correct'] / r['total'] * 100 if r['total'] > 0 else 0

            scaling_results[prog_len] = results

    return scaling_results

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("BroadMind v0.75: Task Scaling / Length Generalization")
    print("Wisdom Distillation + Adaptive Compute + Length Generalization")
    print("=" * 70)

    print(f"\nDevice: {config.device}")

    model = BroadMindV075(config).to(config.device)
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
        loss = train_solver_only(model, solver_optimizer, batch)

        if (i + 1) % 500 == 0:
            acc = evaluate_solver_only(model, n_batches=5)
            print(f"[{i+1}/{config.n_iterations_phase1}] Loss: {loss:.4f} | Solver Acc: {acc:.1f}%")

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
        task_loss, cons_loss, noise = train_phase5(
            model, phase5_optimizer, i, config.n_iterations_phase5
        )

        if (i + 1) % 100 == 0:
            results = evaluate(model, n_batches=5)
            print(f"[{i+1}/{config.n_iterations_phase5}] Task: {task_loss:.4f} | "
                  f"Cons: {cons_loss:.4f} | Noise: {noise:.3f} | "
                  f"Acc: {results['accuracy']:.1f}%")

    # Re-enable all params for eval
    for param in model.parameters():
        param.requires_grad = True

    # ========================================================================
    # FINAL EVALUATION (v0.74-compatible, lengths 1-4)
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

    print("\nBy Task Family:")
    for family, r in results['by_family'].items():
        print(f"  {family}: {r['accuracy']:.1f}%")

    print("\nBy Program Length:")
    for length in sorted(results['by_length'].keys()):
        r = results['by_length'][length]
        match = "[Y]" if abs(r['avg_steps'] - length) < 0.5 else "[N]"
        print(f"  Length {length}: {r['accuracy']:.1f}% acc, {r['avg_steps']:.1f} steps {match}")

    # ========================================================================
    # SCALING EVALUATION (lengths 4, 8, 12, 16)
    # ========================================================================
    print("\n" + "=" * 70)
    print("SCALING EVALUATION (length generalization)")
    print("=" * 70)

    scaling_results = evaluate_scaling(model, n_batches=20)

    print("\n  Length | Accuracy | Steps Used | Steps Needed")
    print("  -------+----------+------------+-------------")
    for prog_len in [4, 8, 12, 16]:
        sr = scaling_results[prog_len]
        print(f"  {prog_len:>5d}  | {sr['accuracy']:>6.1f}%  | "
              f"{sr['avg_steps_used']:>10.1f} | {sr['avg_steps_needed']:>10.1f}")

    print("\n  Per-family breakdown at each length:")
    for prog_len in [4, 8, 12, 16]:
        sr = scaling_results[prog_len]
        fam_strs = []
        for fam_name in ['ACCUMULATE', 'TRANSFER', 'COMPARE', 'DECREMENT']:
            if fam_name in sr['by_family']:
                fam_strs.append(f"{fam_name}={sr['by_family'][fam_name]['accuracy']:.0f}%")
        print(f"    Length {prog_len:>2d}: {', '.join(fam_strs)}")

    # ========================================================================
    # THE COMPLETE BROADMIND v0.75 TEST
    # ========================================================================
    print("\n" + "=" * 70)
    print("THE COMPLETE BROADMIND v0.75 TEST")
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

    all_pass = (accurate and mixed_accurate and wise and
                efficient_easy and efficient_hard and steps_match)
    all_scale = scale_4 and scale_8 and scale_12 and scale_16

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

BROADMIND v0.74 COMPAT: {'PASS' if all_pass else 'PARTIAL'}
SCALING STATUS:          {'PASS' if all_scale else 'PARTIAL'}
BROADMIND v0.75 STATUS:  {'COMPLETE' if all_pass and all_scale else 'PARTIAL'}
""")

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"""
BroadMind v0.75: Task Scaling / Length Generalization
-----------------------------------------------------
Parameters: {model.count_parameters():,}

Capabilities:
  [+] Latent program induction (invents reasoning circuits)
  [+] Wisdom distillation (keeps the juice, forgets the fluff)
  [+] Adaptive compute (sips power, drops insight)
  [+] Mixed-program execution (ops from any family)
  [+] Length generalization (sinusoidal encoding + noise injection)

Results (in-distribution, lengths 1-4):
  Accuracy:        {results['accuracy']:.1f}%
  Mixed-Program:   {mixed_results['accuracy']:.1f}%
  Wisdom Match:    {wisdom_acc:.1f}%
  Steps Used:      {results['avg_steps_used']:.2f}
  Steps Needed:    {results['avg_steps_needed']:.2f}

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

Scaling (out-of-distribution):
  Length  4: {scaling_results[4]['accuracy']:.1f}% acc, {scaling_results[4]['avg_steps_used']:.1f} steps
  Length  8: {scaling_results[8]['accuracy']:.1f}% acc, {scaling_results[8]['avg_steps_used']:.1f} steps
  Length 12: {scaling_results[12]['accuracy']:.1f}% acc, {scaling_results[12]['avg_steps_used']:.1f} steps
  Length 16: {scaling_results[16]['accuracy']:.1f}% acc, {scaling_results[16]['avg_steps_used']:.1f} steps
""")

    torch.save({
        'model_state_dict': model.state_dict(),
        'accuracy': results['accuracy'],
        'mixed_accuracy': mixed_results['accuracy'],
        'wisdom_accuracy': wisdom_acc,
        'avg_steps_used': results['avg_steps_used'],
        'scaling_results': {k: v['accuracy'] for k, v in scaling_results.items()},
    }, 'broadmind_v075_scaling.pt')
    print("Saved to broadmind_v075_scaling.pt")

    return model, results, scaling_results


if __name__ == "__main__":
    model, results, scaling_results = main()
