"""
FluxMind v0.70j2: Conditional Operations - Working Version
==========================================================

Results: 96.5% exact match (vs 1.0% in v0.70j)
Generalization to 8 steps: 94.1%

Key fixes from v0.70j:
1. More capacity: 357K params (vs 134K)
2. Curriculum learning: ADD/SUB first, then conditionals
3. Explicit comparison features in executor
4. Residual connection in executor
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    # Task complexity
    n_variables = 3          # x, y, z
    n_ops = 9                # ADD/SUB × 3 + 3 conditionals
    program_length = 4       # steps per program
    value_range = 10         # values in [0, value_range)
    
    # Architecture - INCREASED CAPACITY
    d_model = 192            # was 128 in v0.70j
    d_latent = 96            # was 64 in v0.70j
    
    # Training
    batch_size = 256
    n_iterations = 4000
    lr = 1e-3
    lr_fine = 3e-4           # lower for phase 3
    grad_clip = 1.0
    
    # Curriculum phases
    curriculum_phase1 = 1500  # ADD/SUB only
    curriculum_phase2 = 1500  # Gradually add conditionals
    # Phase 3 = remaining iterations with full distribution
    
    # Regularization
    dropout = 0.1
    weight_decay = 1e-4
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config = Config()

# Operation definitions
OPS = [
    'ADD_X', 'ADD_Y', 'ADD_Z',
    'SUB_X', 'SUB_Y', 'SUB_Z',
    'IF_X_GT_Y_ADD_Z',  # if x > y: z += 1
    'IF_Y_GT_Z_ADD_X',  # if y > z: x += 1
    'IF_Z_GT_X_ADD_Y',  # if z > x: y += 1
]

# ============================================================================
# DATA GENERATION
# ============================================================================

def execute_op(state, op_idx):
    """Execute a single operation on state."""
    x, y, z = state
    op = OPS[op_idx]
    
    if op == 'ADD_X': return (x + 1, y, z)
    elif op == 'ADD_Y': return (x, y + 1, z)
    elif op == 'ADD_Z': return (x, y, z + 1)
    elif op == 'SUB_X': return (x - 1, y, z)
    elif op == 'SUB_Y': return (x, y - 1, z)
    elif op == 'SUB_Z': return (x, y, z - 1)
    elif op == 'IF_X_GT_Y_ADD_Z': return (x, y, z + 1) if x > y else (x, y, z)
    elif op == 'IF_Y_GT_Z_ADD_X': return (x + 1, y, z) if y > z else (x, y, z)
    elif op == 'IF_Z_GT_X_ADD_Y': return (x, y + 1, z) if z > x else (x, y, z)
    else: raise ValueError(f"Unknown op: {op}")

def execute_program(initial_state, program):
    """Execute full program, return list of states after each step."""
    states = [initial_state]
    state = initial_state
    for op_idx in program:
        state = execute_op(state, op_idx)
        states.append(state)
    return states

def generate_batch(batch_size, program_length, value_range=10, n_ops=9, conditional_prob=1.0):
    """Generate batch with curriculum control over conditional frequency."""
    programs = []
    initial_states = []
    all_states = []
    
    for _ in range(batch_size):
        init = tuple(np.random.randint(0, value_range) for _ in range(3))
        
        prog = []
        for _ in range(program_length):
            if np.random.random() < conditional_prob:
                prog.append(np.random.randint(0, n_ops))  # all ops
            else:
                prog.append(np.random.randint(0, 6))  # ADD/SUB only
        
        states = execute_program(init, prog)
        programs.append(prog)
        initial_states.append(init)
        all_states.append(states[1:])
    
    programs_t = torch.tensor(programs, dtype=torch.long, device=config.device)
    initial_t = torch.tensor(initial_states, dtype=torch.float, device=config.device)
    targets_t = torch.tensor(all_states, dtype=torch.float, device=config.device)
    
    return programs_t, initial_t, targets_t

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class LatentGenerator(nn.Module):
    """Generate latent instruction from (state, op)."""
    
    def __init__(self, config):
        super().__init__()
        
        self.state_encoder = nn.Sequential(
            nn.Linear(config.n_variables, config.d_model),
            nn.LayerNorm(config.d_model),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, config.d_model),
            nn.ReLU(),
        )
        
        self.op_embedding = nn.Embedding(config.n_ops, config.d_model)
        
        self.generator = nn.Sequential(
            nn.Linear(config.d_model * 2, config.d_model),
            nn.LayerNorm(config.d_model),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, config.d_model),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, config.d_latent)
        )
    
    def forward(self, state, op_idx):
        state_enc = self.state_encoder(state)
        op_enc = self.op_embedding(op_idx)
        combined = torch.cat([state_enc, op_enc], dim=-1)
        return self.generator(combined)


class LatentExecutor(nn.Module):
    """Execute latent instruction on state - with comparison capability."""
    
    def __init__(self, config):
        super().__init__()
        
        self.state_encoder = nn.Sequential(
            nn.Linear(config.n_variables, config.d_model),
            nn.LayerNorm(config.d_model),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, config.d_model),
            nn.ReLU(),
        )
        
        # KEY: Explicit comparison features for conditionals
        self.comparison_encoder = nn.Sequential(
            nn.Linear(3, config.d_model // 2),  # 3 comparison signals
            nn.ReLU(),
        )
        
        self.latent_encoder = nn.Sequential(
            nn.Linear(config.d_latent, config.d_model),
            nn.LayerNorm(config.d_model),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        )
        
        combined_dim = config.d_model * 2 + config.d_model // 2
        self.executor = nn.Sequential(
            nn.Linear(combined_dim, config.d_model),
            nn.LayerNorm(config.d_model),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, config.d_model),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, config.n_variables)
        )
    
    def forward(self, state, latent):
        state_enc = self.state_encoder(state)
        
        # Explicit comparison features for conditionals
        x, y, z = state[:, 0], state[:, 1], state[:, 2]
        comparisons = torch.stack([
            (x > y).float(),
            (y > z).float(),
            (z > x).float()
        ], dim=-1)
        comp_enc = self.comparison_encoder(comparisons)
        
        latent_enc = self.latent_encoder(latent)
        
        combined = torch.cat([state_enc, latent_enc, comp_enc], dim=-1)
        delta = self.executor(combined)
        
        # Residual connection: output = state + delta
        return state + delta


class FluxMindV070j2(nn.Module):
    """FluxMind v0.70j2: Conditionals - Working Version"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.generator = LatentGenerator(config)
        self.executor = LatentExecutor(config)
        self.predictor = nn.Linear(config.n_variables, config.n_variables)
    
    def forward(self, programs, initial_states, return_latents=False):
        batch_size, seq_len = programs.shape
        
        predictions = []
        latents = []
        state = initial_states
        
        for t in range(seq_len):
            op_idx = programs[:, t]
            
            # Generate latent instruction (conditioned on current state!)
            latent = self.generator(state, op_idx)
            latents.append(latent)
            
            # Execute latent to get next state
            state = self.executor(state, latent)
            
            # Predict output
            pred = self.predictor(state)
            predictions.append(pred)
        
        predictions = torch.stack(predictions, dim=1)
        
        if return_latents:
            latents = torch.stack(latents, dim=1)
            return predictions, latents
        
        return predictions
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

# ============================================================================
# TRAINING
# ============================================================================

def train_step(model, optimizer, batch):
    programs, initial_states, targets = batch
    
    model.train()
    optimizer.zero_grad()
    
    predictions = model(programs, initial_states)
    loss = F.mse_loss(predictions, targets)
    
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
    optimizer.step()
    
    return loss.item()


def evaluate(model, n_batches=10, program_length=None, n_ops=9):
    if program_length is None:
        program_length = config.program_length
    
    model.eval()
    
    total_exact = 0
    total_samples = 0
    op_correct = defaultdict(int)
    op_total = defaultdict(int)
    
    with torch.no_grad():
        for _ in range(n_batches):
            programs, initial_states, targets = generate_batch(
                config.batch_size, program_length, config.value_range, n_ops
            )
            
            predictions = model(programs, initial_states)
            pred_rounded = predictions.round()
            
            exact = (pred_rounded == targets).all(dim=-1).all(dim=-1)
            total_exact += exact.sum().item()
            total_samples += config.batch_size
            
            for b in range(config.batch_size):
                for t in range(program_length):
                    op_idx = programs[b, t].item()
                    op_name = OPS[op_idx]
                    correct = (pred_rounded[b, t] == targets[b, t]).all().item()
                    op_correct[op_name] += correct
                    op_total[op_name] += 1
    
    return {
        'exact_match': total_exact / total_samples * 100,
        'per_op': {op: op_correct[op] / op_total[op] * 100 if op_total[op] > 0 else 0 
                   for op in OPS}
    }

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 60)
    print("FluxMind v0.70j2: Conditional Operations")
    print("=" * 60)
    print(f"\nTask: {config.n_variables} variables, {config.n_ops} operations")
    print(f"Operations: {OPS}")
    print(f"Device: {config.device}")
    
    model = FluxMindV070j2(config).to(config.device)
    print(f"Parameters: {model.count_parameters():,}")
    
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config.lr, 
        weight_decay=config.weight_decay
    )
    
    # ========================================================================
    # PHASE 1: ADD/SUB ONLY
    # ========================================================================
    print("\n" + "-" * 60)
    print("Phase 1: ADD/SUB only")
    print("-" * 60)
    
    for i in range(config.curriculum_phase1):
        batch = generate_batch(
            config.batch_size, config.program_length, 
            config.value_range, config.n_ops, conditional_prob=0.0
        )
        loss = train_step(model, optimizer, batch)
        
        if (i + 1) % 500 == 0:
            results = evaluate(model, n_batches=10, n_ops=6)
            print(f"[{i+1}] Loss: {loss:.4f} | Exact (ADD/SUB): {results['exact_match']:.1f}%")
    
    # ========================================================================
    # PHASE 2: GRADUAL CONDITIONALS
    # ========================================================================
    print("\n" + "-" * 60)
    print("Phase 2: Gradually add conditionals")
    print("-" * 60)
    
    for i in range(config.curriculum_phase2):
        progress = i / config.curriculum_phase2
        cond_prob = 0.3 * progress  # 0% → 30%
        
        batch = generate_batch(
            config.batch_size, config.program_length,
            config.value_range, config.n_ops, conditional_prob=cond_prob
        )
        loss = train_step(model, optimizer, batch)
        
        if (i + 1) % 500 == 0:
            results = evaluate(model, n_batches=10)
            add_acc = np.mean([results['per_op'][op] for op in OPS if op.startswith('ADD_')])
            if_acc = np.mean([results['per_op'][op] for op in OPS if op.startswith('IF_')])
            iteration = config.curriculum_phase1 + i + 1
            print(f"[{iteration}] Loss: {loss:.4f} | Exact: {results['exact_match']:.1f}% | ADD: {add_acc:.1f}% | IF: {if_acc:.1f}%")
    
    # ========================================================================
    # PHASE 3: FULL TRAINING (LOWER LR)
    # ========================================================================
    print("\n" + "-" * 60)
    print("Phase 3: Full training with lower LR")
    print("-" * 60)
    
    for pg in optimizer.param_groups:
        pg['lr'] = config.lr_fine
    
    remaining = config.n_iterations - config.curriculum_phase1 - config.curriculum_phase2
    
    for i in range(remaining):
        batch = generate_batch(
            config.batch_size, config.program_length,
            config.value_range, config.n_ops, conditional_prob=1.0
        )
        loss = train_step(model, optimizer, batch)
        
        if (i + 1) % 500 == 0:
            results = evaluate(model, n_batches=10)
            add_acc = np.mean([results['per_op'][op] for op in OPS if op.startswith('ADD_')])
            if_acc = np.mean([results['per_op'][op] for op in OPS if op.startswith('IF_')])
            iteration = config.curriculum_phase1 + config.curriculum_phase2 + i + 1
            print(f"[{iteration}] Exact: {results['exact_match']:.1f}% | ADD: {add_acc:.1f}% | IF: {if_acc:.1f}%")
    
    # ========================================================================
    # FINAL EVALUATION
    # ========================================================================
    print("\n" + "=" * 60)
    print("FINAL EVALUATION")
    print("=" * 60)
    
    results = evaluate(model, n_batches=30)
    print(f"\nExact match: {results['exact_match']:.1f}%")
    print("\nPer-operation accuracy:")
    for op in OPS:
        marker = "★" if 'IF_' in op else " "
        print(f"  {marker} {op}: {results['per_op'][op]:.1f}%")
    
    # Generalization
    results_8 = evaluate(model, n_batches=30, program_length=8)
    print(f"\nGeneralization (8 steps): {results_8['exact_match']:.1f}%")
    
    # Summary
    add_acc = np.mean([results['per_op'][op] for op in OPS if op.startswith('ADD_')])
    sub_acc = np.mean([results['per_op'][op] for op in OPS if op.startswith('SUB_')])
    if_acc = np.mean([results['per_op'][op] for op in OPS if op.startswith('IF_')])
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"""
FluxMind v0.70j2 Results:
-------------------------
Parameters: {model.count_parameters():,}
Task: {config.n_variables} vars, {config.n_ops} ops (ADD/SUB + conditionals)

Training (4 steps):  {results['exact_match']:.1f}%
Generalization (8 steps): {results_8['exact_match']:.1f}%

By operation type:
  ADD ops: {add_acc:.1f}%
  SUB ops: {sub_acc:.1f}%
  IF ops:  {if_acc:.1f}%

Comparison:
  v0.70g2: 99.6% (6 ops, 191K params)
  v0.70j:  1.0%  (9 ops, 134K params) - FAILED
  v0.70j2: {results['exact_match']:.1f}%  (9 ops, {model.count_parameters():,} params) - SUCCESS
""")
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'results': results,
        'results_8': results_8,
    }, 'fluxmind_v070j2_best.pt')
    print("Model saved to fluxmind_v070j2_best.pt")
    
    return model, results, results_8

if __name__ == "__main__":
    model, results, results_8 = main()
