"""
FluxMind v0.75.1: Compositional Programs + OOD Calibration
===========================================================
Production-ready cognitive engine with calibrated uncertainty.

Author: Elnur Ibrahimov
Version: 0.75.1
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import random
import numpy as np
import json
from pathlib import Path


@dataclass
class FluxMindConfig:
    """Configuration for FluxMind model."""
    # Model architecture
    d_model: int = 192
    d_latent: int = 96
    n_vars: int = 4
    n_ops_per_dsl: int = 8
    n_dsls: int = 2

    # Training hyperparameters
    batch_size: int = 256
    lr: float = 1e-3
    lr_finetune: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0

    # Phased training iterations
    phase1_iterations: int = 2000  # DSL A only
    phase2_iterations: int = 3000  # DSL B only
    phase3_iterations: int = 2500  # Compositional + OOD

    # Program settings
    n_steps: int = 4

    # Value ranges
    min_val: int = 1
    max_val: int = 15
    ood_min_val: int = 16
    ood_max_val: int = 30
    ood_prob: float = 0.15

    # Mixing strategies for compositional training
    mix_strategies: List[str] = field(default_factory=lambda: [
        "pure_A", "pure_B", "alternating",
        "random", "A_then_B", "B_then_A"
    ])

    # Runtime
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, d: dict) -> 'FluxMindConfig':
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ============================================================================
# DSL DEFINITIONS
# ============================================================================

DSL_NAMES = ['A', 'B']

def apply_op_A(x: int, y: int, z: int, w: int, op_idx: int,
               min_val: int = 1, max_val: int = 15) -> Tuple[int, int, int, int]:
    """DSL A: Additive operations (+1, -1 on each variable)"""
    if op_idx == 0: x = min(x + 1, max_val)
    elif op_idx == 1: x = max(x - 1, min_val)
    elif op_idx == 2: y = min(y + 1, max_val)
    elif op_idx == 3: y = max(y - 1, min_val)
    elif op_idx == 4: z = min(z + 1, max_val)
    elif op_idx == 5: z = max(z - 1, min_val)
    elif op_idx == 6: w = min(w + 1, max_val)
    elif op_idx == 7: w = max(w - 1, min_val)
    return x, y, z, w


def apply_op_B(x: int, y: int, z: int, w: int, op_idx: int,
               min_val: int = 1, max_val: int = 15) -> Tuple[int, int, int, int]:
    """DSL B: Multiplicative operations (*2, //2 on each variable)"""
    if op_idx == 0: x = min(x * 2, max_val)
    elif op_idx == 1: x = max(x // 2, min_val)
    elif op_idx == 2: y = min(y * 2, max_val)
    elif op_idx == 3: y = max(y // 2, min_val)
    elif op_idx == 4: z = min(z * 2, max_val)
    elif op_idx == 5: z = max(z // 2, min_val)
    elif op_idx == 6: w = min(w * 2, max_val)
    elif op_idx == 7: w = max(w // 2, min_val)
    return x, y, z, w


def apply_op(state: List[int], op_idx: int, dsl: int,
             min_val: int = 1, max_val: int = 15) -> Tuple[int, int, int, int]:
    """Apply operation from specified DSL."""
    x, y, z, w = [int(v) for v in state]
    if dsl == 0:
        return apply_op_A(x, y, z, w, op_idx, min_val, max_val)
    else:
        return apply_op_B(x, y, z, w, op_idx, min_val, max_val)


# ============================================================================
# DATA GENERATION
# ============================================================================

def generate_dsl_sequence(n_steps: int, strategy: str) -> List[int]:
    """Generate DSL assignments per step based on strategy."""
    if strategy == "pure_A":
        return [0] * n_steps
    elif strategy == "pure_B":
        return [1] * n_steps
    elif strategy == "alternating":
        return [i % 2 for i in range(n_steps)]
    elif strategy == "random":
        return [random.randint(0, 1) for _ in range(n_steps)]
    elif strategy == "A_then_B":
        mid = n_steps // 2
        return [0] * mid + [1] * (n_steps - mid)
    elif strategy == "B_then_A":
        mid = n_steps // 2
        return [1] * mid + [0] * (n_steps - mid)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def generate_batch(
    config: FluxMindConfig,
    batch_size: int,
    n_steps: int,
    strategy: str = "random",
    is_ood: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate a batch of compositional programs."""
    min_val = config.ood_min_val if is_ood else config.min_val
    max_val = config.ood_max_val if is_ood else config.max_val
    exec_max = config.ood_max_val if is_ood else config.max_val

    all_states = []
    all_ops = []
    all_dsls = []

    for _ in range(batch_size):
        dsl_seq = generate_dsl_sequence(n_steps, strategy)
        state = [random.randint(min_val, max_val) for _ in range(4)]
        states = [state.copy()]
        ops = []

        for t in range(n_steps):
            op = random.randint(0, config.n_ops_per_dsl - 1)
            dsl = dsl_seq[t]
            x, y, z, w = state
            if dsl == 0:
                x, y, z, w = apply_op_A(x, y, z, w, op, config.min_val, exec_max)
            else:
                x, y, z, w = apply_op_B(x, y, z, w, op, config.min_val, exec_max)
            state = [x, y, z, w]
            states.append(state.copy())
            ops.append(op)

        all_states.append(states)
        all_ops.append(ops)
        all_dsls.append(dsl_seq)

    return (
        torch.tensor(all_states, dtype=torch.float32, device=config.device),
        torch.tensor(all_ops, dtype=torch.long, device=config.device),
        torch.tensor(all_dsls, dtype=torch.long, device=config.device)
    )


# ============================================================================
# MODEL COMPONENTS
# ============================================================================

class LatentGenerator(nn.Module):
    """Generates latent programs from state, operation, and DSL context."""

    def __init__(self, cfg: FluxMindConfig):
        super().__init__()
        self.cfg = cfg

        self.op_embed = nn.Embedding(cfg.n_ops_per_dsl, cfg.d_model)
        self.dsl_embed = nn.Embedding(cfg.n_dsls, cfg.d_model)
        self.state_encoder = nn.Linear(cfg.n_vars, cfg.d_model)

        # With DSL context
        self.generator = nn.Sequential(
            nn.Linear(cfg.d_model * 3, cfg.d_model),
            nn.ReLU(),
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.ReLU(),
            nn.Linear(cfg.d_model, cfg.d_latent)
        )

        # Without DSL context (for collision detection)
        self.generator_no_dsl = nn.Sequential(
            nn.Linear(cfg.d_model * 2, cfg.d_model),
            nn.ReLU(),
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.ReLU(),
            nn.Linear(cfg.d_model, cfg.d_latent)
        )

        # Calibrated confidence head
        self.confidence_head = nn.Sequential(
            nn.Linear(cfg.d_latent + cfg.n_vars, cfg.d_model // 2),
            nn.ReLU(),
            nn.Linear(cfg.d_model // 2, 1),
            nn.Sigmoid()
        )

        # Context presence detector
        self.context_present_head = nn.Sequential(
            nn.Linear(cfg.d_latent, cfg.d_model // 2),
            nn.ReLU(),
            nn.Linear(cfg.d_model // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, state: torch.Tensor, op: torch.Tensor,
                dsl_id: Optional[torch.Tensor] = None):
        state_enc = self.state_encoder(state)
        op_enc = self.op_embed(op)

        if dsl_id is not None:
            dsl_enc = self.dsl_embed(dsl_id)
            combined = torch.cat([state_enc, op_enc, dsl_enc], dim=-1)
            latent = self.generator(combined)
        else:
            combined = torch.cat([state_enc, op_enc], dim=-1)
            latent = self.generator_no_dsl(combined)

        conf_input = torch.cat([latent, state], dim=-1)
        confidence = self.confidence_head(conf_input)
        context_present = self.context_present_head(latent)

        return latent, confidence, context_present


class StatePredictor(nn.Module):
    """Predicts next state from current state and latent program."""

    def __init__(self, cfg: FluxMindConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.n_vars + cfg.d_latent, cfg.d_model),
            nn.ReLU(),
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.ReLU(),
            nn.Linear(cfg.d_model, cfg.d_model // 2),
            nn.ReLU(),
            nn.Linear(cfg.d_model // 2, cfg.n_vars)
        )

    def forward(self, state: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([state, latent], dim=-1))


# ============================================================================
# MAIN MODEL
# ============================================================================

class FluxMind(nn.Module):
    """
    FluxMind v0.75.1: Compositional Programs + OOD Calibration

    Key capabilities:
    - Multi-DSL learning (additive + multiplicative)
    - Compositional programs (mix DSLs mid-sequence)
    - Calibrated uncertainty (knows when it doesn't know)
    - OOD detection (confidence drops on unfamiliar inputs)
    """

    def __init__(self, cfg: FluxMindConfig):
        super().__init__()
        self.cfg = cfg
        self.generator = LatentGenerator(cfg)
        self.predictor = StatePredictor(cfg)

    def forward_step(self, state: torch.Tensor, op: torch.Tensor,
                     dsl_id: Optional[torch.Tensor] = None):
        """Execute single reasoning step."""
        latent, confidence, context_present = self.generator(state, op, dsl_id)
        next_state = self.predictor(state, latent)
        return next_state, confidence, context_present, latent

    def forward_program(self, init_state: torch.Tensor, ops: torch.Tensor,
                        dsl_ids: Optional[torch.Tensor] = None):
        """Execute full program."""
        n_steps = ops.shape[1]

        trajectory = [init_state]
        confidences = []
        context_presents = []
        latents = []

        state = init_state
        for t in range(n_steps):
            op = ops[:, t]
            dsl_id = dsl_ids[:, t] if dsl_ids is not None else None

            next_state, conf, ctx, latent = self.forward_step(state, op, dsl_id)

            trajectory.append(next_state)
            confidences.append(conf)
            context_presents.append(ctx)
            latents.append(latent)

            state = next_state

        return (
            torch.stack(trajectory, dim=1),
            torch.stack(confidences, dim=1),
            torch.stack(context_presents, dim=1),
            torch.stack(latents, dim=1)
        )

    def step(self, state: List[int], op: int, dsl: int) -> Dict:
        """
        Single step inference for production use.

        Args:
            state: Current state [x, y, z, w]
            op: Operation index (0-7)
            dsl: DSL index (0=additive, 1=multiplicative)

        Returns:
            dict with next_state, confidence, context_present
        """
        self.eval()
        with torch.no_grad():
            state_t = torch.tensor([state], dtype=torch.float32, device=self.cfg.device)
            op_t = torch.tensor([op], dtype=torch.long, device=self.cfg.device)
            dsl_t = torch.tensor([dsl], dtype=torch.long, device=self.cfg.device)

            next_state, confidence, context_present, _ = self.forward_step(state_t, op_t, dsl_t)

            next_state = next_state.round().clamp(
                self.cfg.min_val, self.cfg.max_val
            ).squeeze().tolist()

            return {
                "next_state": [int(v) for v in next_state],
                "confidence": float(confidence.item()),
                "context_present": float(context_present.item())
            }

    def execute(self, initial_state: List[int], ops: List[int],
                dsls: List[int]) -> Dict:
        """
        Execute full program for production use.

        Args:
            initial_state: Starting state [x, y, z, w]
            ops: List of operation indices
            dsls: List of DSL indices per step

        Returns:
            dict with trajectory, confidences, mean_confidence
        """
        self.eval()
        with torch.no_grad():
            state_t = torch.tensor([initial_state], dtype=torch.float32, device=self.cfg.device)
            ops_t = torch.tensor([ops], dtype=torch.long, device=self.cfg.device)
            dsls_t = torch.tensor([dsls], dtype=torch.long, device=self.cfg.device)

            traj, confs, _, _ = self.forward_program(state_t, ops_t, dsls_t)

            traj = traj.round().clamp(self.cfg.min_val, self.cfg.max_val)
            trajectory = [[int(v) for v in s] for s in traj.squeeze().tolist()]
            confidences = [float(c) for c in confs.squeeze().tolist()]

            return {
                "trajectory": trajectory,
                "confidences": confidences,
                "mean_confidence": sum(confidences) / len(confidences),
                "should_abstain": any(c < 0.5 for c in confidences)
            }

    def save(self, path: str):
        """Save model and config."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.cfg.to_dict()
        }, path)

    @classmethod
    def load(cls, path: str, device: str = None) -> 'FluxMind':
        """Load model from file."""
        checkpoint = torch.load(path, map_location='cpu')
        cfg = FluxMindConfig.from_dict(checkpoint['config'])
        if device:
            cfg.device = device
        model = cls(cfg).to(cfg.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model


# ============================================================================
# TRAINING
# ============================================================================

class FluxMindTrainer:
    """Trainer for FluxMind with phased training and OOD calibration."""

    def __init__(self, config: FluxMindConfig = None):
        self.config = config or FluxMindConfig()
        self.model = FluxMind(self.config).to(self.config.device)
        self.history = []

    def train_epoch(self, optimizer, strategy: str, use_ood: bool = False):
        """Single training epoch."""
        self.model.train()
        cfg = self.config

        is_ood = use_ood and random.random() < cfg.ood_prob
        states, ops, dsl_ids = generate_batch(cfg, cfg.batch_size, cfg.n_steps, strategy, is_ood)

        total_loss = 0
        all_conf, all_err = [], []

        state = states[:, 0]
        for t in range(cfg.n_steps):
            latent, confidence, _ = self.model.generator(state, ops[:, t], dsl_ids[:, t])
            next_pred = self.model.predictor(state, latent)

            loss = F.huber_loss(next_pred, states[:, t + 1])
            total_loss += loss

            with torch.no_grad():
                max_v = cfg.ood_max_val if is_ood else cfg.max_val
                discrete = next_pred.round().clamp(cfg.min_val, max_v)
                err = (discrete - states[:, t + 1]).abs().max(dim=-1)[0]
                all_err.append(err)
                all_conf.append(confidence)

            state = next_pred.round().clamp(cfg.min_val, max_v).detach()

        all_err = torch.stack(all_err, dim=1)
        all_conf = torch.stack(all_conf, dim=1)

        # OOD calibration: confidence target = 0 for OOD batches
        if is_ood:
            conf_target = torch.zeros_like(all_conf)
        else:
            conf_target = (all_err < 0.5).float().unsqueeze(-1)

        conf_loss = F.binary_cross_entropy(all_conf, conf_target)
        total_loss = total_loss + 0.2 * conf_loss

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
        optimizer.step()

        return total_loss.item() / cfg.n_steps, conf_loss.item(), is_ood

    def evaluate(self, strategy: str, n_batches: int = 10, is_ood: bool = False):
        """Evaluate model."""
        self.model.eval()
        cfg = self.config
        accs, confs = [], []

        with torch.no_grad():
            for _ in range(n_batches):
                states, ops, dsl_ids = generate_batch(cfg, cfg.batch_size, cfg.n_steps, strategy, is_ood)
                state = states[:, 0]
                correct = torch.ones(cfg.batch_size, dtype=torch.bool, device=cfg.device)
                conf_sum = 0

                for t in range(cfg.n_steps):
                    latent, confidence, _ = self.model.generator(state, ops[:, t], dsl_ids[:, t])
                    next_pred = self.model.predictor(state, latent)
                    max_v = cfg.ood_max_val if is_ood else cfg.max_val
                    next_pred = next_pred.round().clamp(cfg.min_val, max_v)
                    correct &= (next_pred - states[:, t + 1]).abs().max(dim=-1)[0] < 0.5
                    conf_sum += confidence.mean().item()
                    state = next_pred

                accs.append(correct.float().mean().item() * 100)
                confs.append(conf_sum / cfg.n_steps)

        return {"accuracy": np.mean(accs), "confidence": np.mean(confs)}

    def train(self, verbose: bool = True):
        """Full phased training."""
        cfg = self.config

        if verbose:
            print("=" * 60)
            print("FluxMind v0.75.1 Training")
            print("=" * 60)
            print(f"Device: {cfg.device}")
            print(f"Parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        # Phase 1: DSL A
        if verbose:
            print(f"\n=== Phase 1: DSL A ({cfg.phase1_iterations} iterations) ===")
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        for i in range(cfg.phase1_iterations):
            self.train_epoch(optimizer, "pure_A")
            if verbose and (i + 1) % 500 == 0:
                m = self.evaluate("pure_A", 5)
                print(f"[{i+1}] A: {m['accuracy']:.1f}%")

        # Phase 2: DSL B
        if verbose:
            print(f"\n=== Phase 2: DSL B ({cfg.phase2_iterations} iterations) ===")
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        for i in range(cfg.phase2_iterations):
            self.train_epoch(optimizer, "pure_B")
            if verbose and (i + 1) % 500 == 0:
                m_a = self.evaluate("pure_A", 5)
                m_b = self.evaluate("pure_B", 5)
                print(f"[{i+1}] A: {m_a['accuracy']:.1f}%  B: {m_b['accuracy']:.1f}%")

        # Phase 3: Compositional + OOD
        if verbose:
            print(f"\n=== Phase 3: Compositional + OOD ({cfg.phase3_iterations} iterations) ===")
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr_finetune, weight_decay=cfg.weight_decay)
        strategies = cfg.mix_strategies

        for i in range(cfg.phase3_iterations):
            strategy = strategies[i % len(strategies)]
            self.train_epoch(optimizer, strategy, use_ood=True)

            if verbose and (i + 1) % 500 == 0:
                m_a = self.evaluate("pure_A", 5)
                m_b = self.evaluate("pure_B", 5)
                m_mix = self.evaluate("random", 5)
                m_ood = self.evaluate("pure_A", 5, is_ood=True)
                print(f"[{i+1}] A:{m_a['accuracy']:.1f}%/{m_a['confidence']:.2f} "
                      f"B:{m_b['accuracy']:.1f}%/{m_b['confidence']:.2f} "
                      f"Mix:{m_mix['accuracy']:.1f}%/{m_mix['confidence']:.2f} | "
                      f"OOD:{m_ood['accuracy']:.1f}%/{m_ood['confidence']:.2f}")

        if verbose:
            print("\nTraining complete!")

        return self.model


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def train_fluxmind(save_path: str = "fluxmind_v0751.pt", verbose: bool = True) -> FluxMind:
    """Train and save a FluxMind model."""
    config = FluxMindConfig()
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed)

    trainer = FluxMindTrainer(config)
    model = trainer.train(verbose=verbose)
    model.save(save_path)

    if verbose:
        print(f"\nModel saved to {save_path}")

    return model


def load_fluxmind(path: str = "fluxmind_v0751.pt") -> FluxMind:
    """Load a trained FluxMind model."""
    return FluxMind.load(path)


if __name__ == "__main__":
    # Train and test
    model = train_fluxmind("fluxmind_v0751.pt")

    # Test inference
    result = model.step([5, 3, 7, 2], op=0, dsl=0)  # ADD_X in DSL A
    print(f"\nTest step: {result}")

    result = model.execute(
        initial_state=[5, 3, 7, 2],
        ops=[0, 2, 4, 6],  # ADD_X, ADD_Y, ADD_Z, ADD_W
        dsls=[0, 0, 1, 1]  # A, A, B, B (mixed!)
    )
    print(f"Test execute: {result}")
