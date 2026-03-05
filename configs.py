"""Centralized configuration for Soft-JEPA-Flow training."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field


@dataclass
class Config:
    # --- Mode ---
    mode: str = "baseline"  # "baseline" | "jepa"

    # --- Data (Kaggle default) ---
    data_dir: str = "/kaggle/input/miniimagenet256-latents-arrayrecord-sdvae"
    num_classes: int = 100

    # --- Model ---
    patch_size: int = 2
    hidden_size: int = 768
    depth: int = 12
    num_heads: int = 12
    mlp_ratio: float = 4.0

    # --- Optimizer (kvfrans-compatible) ---
    opt: str = "adam"  # "adam" | "adamw"
    lr: float = 1e-4
    beta1: float = 0.9
    beta2: float = 0.99
    weight_decay: float = 0.01

    # --- Training ---
    global_batch: int = 128
    steps: int = 200_000
    seed: int = 42
    class_dropout_prob: float = 0.15
    aug_flip_p: float = 0.3
    aug_jitter_eps: float = 0.01
    t_schedule: str = "lognormal"  # "lognormal" (default) | "uniform"
    t_lognorm_mean: float = -0.4
    t_lognorm_std: float = 1.0

    # --- JEPA-specific ---
    lambda_jepa: float = 0.1
    ema_decay: float = 0.999
    mask_ratio: float = 0.25
    student_layer: int = 4
    teacher_layer: int = 8

    # --- Eval / Sampling ---
    cfg_scale: float = 2.0
    sample_steps: int = 128
    num_sample_images: int = 16
    fid_n: int = 4096
    fid_cache_path: str = "checkpoints/fid_real_stats_4096.npz"
    fid_decode_batch: int = 32
    fid_inception_batch: int = 64

    # --- Logging / Checkpoint ---
    log_every: int = 1000
    eval_every: int = 5_000
    sample_every: int = 40_000
    fid_every: int = 50_000
    ckpt_every: int = 50_000
    run_name: str = "run"
    ckpt_dir: str = "checkpoints"
    best_metric: str = "quick_fid_4096"  # or "val_loss"
    hf_repo_id: str = ""  # full repo id, e.g. "Bangchis/soft-jepa-flow"
    hf_username: str = "Bangchis"  # used when hf_repo_id is empty
    hf_repo_name: str = "soft-jepa-flow"  # used with hf_username
    hf_private: bool = False

    # --- Derived (computed, not CLI) ---
    latent_size: int = 32
    latent_channels: int = 4
    vae_scaling_factor: float = 0.18215

    @property
    def num_tokens(self) -> int:
        grid = self.latent_size // self.patch_size
        return grid * grid

    @property
    def grid_size(self) -> int:
        return self.latent_size // self.patch_size

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_args(cls) -> Config:
        parser = argparse.ArgumentParser(description="Soft-JEPA-Flow training")

        # Mode
        parser.add_argument("--mode", type=str, default="baseline",
                            choices=["baseline", "jepa"])

        # Data
        parser.add_argument("--data_dir", type=str,
                            default="/kaggle/input/datasets/bangchi/miniimagenet256-latents-arrayrecord-sdvae")
        parser.add_argument("--num_classes", type=int, default=100)

        # Model
        parser.add_argument("--patch_size", type=int, default=2)
        parser.add_argument("--hidden_size", type=int, default=768)
        parser.add_argument("--depth", type=int, default=12)
        parser.add_argument("--num_heads", type=int, default=12)
        parser.add_argument("--mlp_ratio", type=float, default=4.0)

        # Optimizer
        parser.add_argument("--opt", type=str, default="adam",
                            choices=["adam", "adamw"])
        parser.add_argument("--lr", type=float, default=1e-4)
        parser.add_argument("--beta1", type=float, default=0.9)
        parser.add_argument("--beta2", type=float, default=0.99)
        parser.add_argument("--weight_decay", type=float, default=0.01)

        # Training
        parser.add_argument("--global_batch", type=int, default=128)
        parser.add_argument("--steps", type=int, default=200_000)
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--class_dropout_prob", type=float, default=0.1)
        parser.add_argument("--aug_flip_p", type=float, default=0.5)
        parser.add_argument("--aug_jitter_eps", type=float, default=0.01)
        parser.add_argument("--t_schedule", type=str, default="lognormal",
                            choices=["lognormal", "uniform"])
        parser.add_argument("--t_lognorm_mean", type=float, default=-0.4)
        parser.add_argument("--t_lognorm_std", type=float, default=1.0)

        # JEPA-specific
        parser.add_argument("--lambda_jepa", type=float, default=0.1)
        parser.add_argument("--ema_decay", type=float, default=0.999)
        parser.add_argument("--mask_ratio", type=float, default=0.25)
        parser.add_argument("--student_layer", type=int, default=3)
        parser.add_argument("--teacher_layer", type=int, default=7)

        # Eval / Sampling
        parser.add_argument("--cfg_scale", type=float, default=1.0)
        parser.add_argument("--sample_steps", type=int, default=128)
        parser.add_argument("--num_sample_images", type=int, default=16)
        parser.add_argument("--fid_n", type=int, default=4096)
        parser.add_argument("--fid_cache_path", type=str,
                            default="checkpoints/fid_real_stats_4096.npz")
        parser.add_argument("--fid_decode_batch", type=int, default=32)
        parser.add_argument("--fid_inception_batch", type=int, default=64)

        # Logging / Checkpoint
        parser.add_argument("--log_every", type=int, default=500)
        parser.add_argument("--eval_every", type=int, default=5_000)
        parser.add_argument("--sample_every", type=int, default=10_000)
        parser.add_argument("--fid_every", type=int, default=50_000)
        parser.add_argument("--ckpt_every", type=int, default=50_000)
        parser.add_argument("--run_name", type=str, default="run")
        parser.add_argument("--ckpt_dir", type=str, default="checkpoints")
        parser.add_argument("--best_metric", type=str, default="quick_fid_4096",
                            choices=["quick_fid_4096", "val_loss"])
        parser.add_argument("--hf_repo_id", type=str, default="")
        parser.add_argument("--hf_username", type=str, default="Bangchis")
        parser.add_argument("--hf_repo_name", type=str,
                            default="soft-jepa-flow")
        parser.add_argument("--hf_private", action="store_true")

        args = parser.parse_args()

        # Build Config from parsed args (only set CLI-exposed fields)
        cli_fields = {k: v for k, v in vars(args).items()}
        return cls(**cli_fields)
