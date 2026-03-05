"""W&B logging, activation debug, and sample grid utilities."""

from __future__ import annotations

from dataclasses import asdict
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

if TYPE_CHECKING:
    from configs import Config


def init_wandb(config: Config):
    """Initialize W&B run with full config dump."""
    import wandb

    wandb.init(
        project="soft-jepa-flow",
        name=config.run_name,
        config=asdict(config),
    )


def log_metrics(metrics: dict, step: int, prefix: str = "train"):
    """Log scalar metrics to W&B."""
    import wandb

    log_dict = {f"{prefix}/{k}": float(v) for k, v in metrics.items()}
    log_dict["step"] = step
    wandb.log(log_dict, step=step)


def log_grad_stats(grads, params, step: int):
    """Log gradient and parameter norms."""
    import wandb

    grad_norm = _tree_norm(grads)
    param_norm = _tree_norm(params)

    wandb.log(
        {
            "train/grad_norm": float(grad_norm),
            "train/param_norm": float(param_norm),
        },
        step=step,
    )


def log_activation_debug(params, step: int, block_indices: list[int] | None = None):
    """Log lightweight activation/gate statistics (scalar only, no histograms).

    Checks adaLN gate stats to detect 'gates stuck at 0' issues.
    """
    import wandb

    if block_indices is None:
        block_indices = [0, 4, 8, 11]

    log_dict = {}
    for idx in block_indices:
        block_key = f"block_{idx}"
        if block_key not in params:
            continue

        block_params = params[block_key]

        # Check adaLN modulation Dense kernel (the zero-init one)
        for key in block_params:
            if "Dense_0" in key:  # adaLN projection
                kernel = block_params[key].get("kernel", None)
                if kernel is not None:
                    k_np = np.array(kernel)
                    log_dict[f"debug/block{idx}_adaln_mean"] = float(np.mean(np.abs(k_np)))
                    log_dict[f"debug/block{idx}_adaln_std"] = float(np.std(k_np))

    if log_dict:
        wandb.log(log_dict, step=step)


def log_nan_inf_counts(metrics: dict, step: int):
    """Log NaN/Inf counts for divergence detection."""
    import wandb

    nan_count = sum(1 for v in metrics.values() if np.isnan(float(v)))
    inf_count = sum(1 for v in metrics.values() if np.isinf(float(v)))

    wandb.log(
        {
            "train/nan_count": nan_count,
            "train/inf_count": inf_count,
        },
        step=step,
    )


def log_sample_grid(
    images: np.ndarray,
    step: int,
    key: str = "samples",
    scaling_factor: float = 0.18215,
):
    """Log a grid of latent samples as a W&B image.

    Args:
        images: (N, 32, 32, 4) latent samples
        step:   current training step
        key:    W&B log key
        scaling_factor: VAE scaling factor for denormalization
    """
    import wandb
    from PIL import Image

    n = images.shape[0]
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))

    # Convert latents to pseudo-RGB for visualization
    vis_list = []
    for i in range(n):
        z = images[i] / scaling_factor
        z = (z + 1.0) / 2.0
        z = np.clip(z, 0, 1)
        rgb = z[:, :, :3]  # first 3 channels
        vis_list.append((rgb * 255).astype(np.uint8))

    # Build grid
    h, w = vis_list[0].shape[:2]
    grid = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
    for i, img in enumerate(vis_list):
        r, c = divmod(i, cols)
        grid[r * h : (r + 1) * h, c * w : (c + 1) * w] = img

    wandb.log({key: wandb.Image(grid)}, step=step)


def _tree_norm(tree) -> float:
    """Compute global L2 norm of a pytree."""
    leaves = jax.tree.leaves(tree)
    return float(jnp.sqrt(sum(jnp.sum(x ** 2) for x in leaves)))
