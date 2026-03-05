"""Main entry point for Soft-JEPA-Flow training on Kaggle TPU.

Usage:
    git clone <repo> && cd soft_jepa_flow
    pip install -r requirements.txt
    python train.py --mode baseline --data_dir /kaggle/input/miniimagenet256-latents-arrayrecord-sdvae
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import jax_utils
from flax.training import common_utils

from configs import Config
from data import create_loader
from dit import JepaDiT
from train_state import TrainState
from train_step import train_step_baseline, train_step_jepa
from sample import sample_images
from checkpoint import save_checkpoint, maybe_restore, BestMetricTracker
from logging_utils import (
    init_wandb,
    log_metrics,
    log_nan_inf_counts,
    log_sample_grid,
)


def make_optimizer(config: Config):
    """Create optax optimizer from config."""
    if config.opt == "adamw" or config.weight_decay > 0:
        return optax.adamw(
            learning_rate=config.lr,
            b1=config.beta1,
            b2=config.beta2,
            weight_decay=config.weight_decay,
        )
    return optax.adam(
        learning_rate=config.lr,
        b1=config.beta1,
        b2=config.beta2,
    )


def make_config_static(config: Config) -> dict:
    """Extract static hyperparams as a plain dict for pmap'd steps."""
    return {
        "aug_flip_p": config.aug_flip_p,
        "aug_jitter_eps": config.aug_jitter_eps,
        "mask_ratio": config.mask_ratio,
        "lambda_jepa": config.lambda_jepa,
        "ema_decay": config.ema_decay,
        "patch_size": config.patch_size,
        "latent_size": config.latent_size,
        # Timestep schedule: 0=lognormal (default), 1=uniform
        "t_schedule": 0 if config.t_schedule == "lognormal" else 1,
        "t_lognorm_mean": config.t_lognorm_mean,
        "t_lognorm_std": config.t_lognorm_std,
    }


def shard_batch(batch, num_devices):
    """Reshape batch for pmap: (local_batch, ...) → (num_devices, per_device, ...)."""
    return jax.tree.map(
        lambda x: x.reshape(num_devices, -1, *x.shape[1:]),
        batch,
    )


def main():
    config = Config.from_args()

    # --- Device setup ---
    num_devices = jax.local_device_count()
    print(f"[init] {num_devices} devices, global_batch={config.global_batch}")

    per_device_batch = config.global_batch // jax.device_count()
    assert config.global_batch % jax.device_count() == 0

    # --- Model ---
    model_def = JepaDiT(
        patch_size=config.patch_size,
        hidden_size=config.hidden_size,
        depth=config.depth,
        num_heads=config.num_heads,
        mlp_ratio=config.mlp_ratio,
        num_classes=config.num_classes,
        class_dropout_prob=config.class_dropout_prob,
        student_layer=config.student_layer,
        teacher_layer=config.teacher_layer,
        latent_size=config.latent_size,
        latent_channels=config.latent_channels,
    )

    # --- Init params ---
    rng = jax.random.PRNGKey(config.seed)
    rng, param_rng, dropout_rng, state_rng = jax.random.split(rng, 4)

    dummy_z = jnp.zeros((per_device_batch, config.latent_size, config.latent_size,
                          config.latent_channels))
    dummy_t = jnp.zeros((per_device_batch,))
    dummy_y = jnp.zeros((per_device_batch, config.num_classes))

    params = model_def.init(
        {"params": param_rng, "label_dropout": dropout_rng},
        dummy_z, dummy_t, dummy_y,
        train=False, mode="baseline",
    )["params"]

    param_count = sum(x.size for x in jax.tree.leaves(params))
    print(f"[init] Model params: {param_count:,}")

    # --- Optimizer + TrainState ---
    tx = make_optimizer(config)
    state = TrainState.create(model_def=model_def, params=params, tx=tx, rng=state_rng)

    # --- Replicate across devices ---
    state = jax_utils.replicate(state)

    # --- Try resume from checkpoint ---
    state, start_step = maybe_restore(state, config)

    # --- W&B init ---
    init_wandb(config)

    # --- Data loaders ---
    train_loader = create_loader(
        config.data_dir, "train", config.global_batch,
        config.num_classes, config.seed, shuffle=True,
    )
    val_loader = create_loader(
        config.data_dir, "val", config.global_batch,
        config.num_classes, config.seed, shuffle=False,
    )

    # --- Prefetch to device ---
    def sharded_iter(loader):
        for batch in loader:
            # Convert grain output (numpy dict) to JAX arrays, sharded
            batch_jax = jax.tree.map(lambda x: jnp.array(x), batch)
            yield shard_batch(batch_jax, num_devices)

    prefetcher = jax_utils.prefetch_to_device(sharded_iter(train_loader), size=2)

    # --- Config static (replicated dict for pmap) ---
    config_static = make_config_static(config)
    config_static_rep = jax_utils.replicate(config_static)

    # --- Best metric tracker ---
    best_tracker = BestMetricTracker(config)

    # --- Fixed noise for sample grids ---
    rng, sample_rng = jax.random.split(rng)
    fixed_sample_rng = sample_rng

    # --- Training loop ---
    train_step_fn = train_step_jepa if config.mode == "jepa" else train_step_baseline

    print(f"[train] Starting {config.mode} training for {config.steps} steps...")

    for step, sharded_batch in enumerate(prefetcher, start=start_step + 1):
        if step > config.steps:
            break

        state, metrics = train_step_fn(state, sharded_batch, config_static_rep)

        # --- Periodic logging ---
        if step % config.log_every == 0:
            metrics_cpu = jax.tree.map(lambda x: float(x[0]), metrics)
            log_metrics(metrics_cpu, step, prefix="train")
            log_nan_inf_counts(metrics_cpu, step)
            print(f"  step {step}: {metrics_cpu}")

        # --- Sample grid ---
        if step % config.sample_every == 0:
            state_single = jax_utils.unreplicate(state)
            samples = sample_images(
                state_single.apply_fn,
                state_single.ema_params,
                fixed_sample_rng,
                num_images=config.num_sample_images,
                num_classes=config.num_classes,
                num_steps=config.sample_steps,
                cfg_scale=config.cfg_scale,
            )
            log_sample_grid(np.array(samples), step)

        # --- Validation ---
        if step % config.eval_every == 0:
            val_metrics = run_validation(state, val_loader, config, num_devices)
            log_metrics(val_metrics, step, prefix="val")

            # Update best tracker with val loss as fallback
            if config.best_metric == "val_loss":
                best_tracker.update(state, val_metrics.get("l_total", float("inf")), step)

        # --- FID ---
        if step % config.fid_every == 0 and step > 0:
            from fid import compute_fid as compute_fid_fn

            state_single = jax_utils.unreplicate(state)
            rng, fid_rng = jax.random.split(rng)

            fid_val_loader = create_loader(
                config.data_dir, "val", config.global_batch,
                config.num_classes, config.seed, shuffle=False,
            )

            fid_score = compute_fid_fn(
                state_single.apply_fn,
                state_single.ema_params,
                fid_rng,
                fid_val_loader,
                fid_n=config.fid_n,
                num_classes=config.num_classes,
                sample_steps=config.sample_steps,
                cfg_scale=config.cfg_scale,
                seed=config.seed,
            )

            import wandb
            wandb.log({"eval/quick_fid_4096": fid_score}, step=step)
            print(f"  [FID] step {step}: {fid_score:.2f}")

            if config.best_metric == "quick_fid_4096":
                best_tracker.update(state, fid_score, step)

        # --- Checkpoint ---
        if step % config.ckpt_every == 0:
            save_checkpoint(state, config, step)

    print("[train] Training complete.")


def run_validation(state, val_loader, config: Config, num_devices: int) -> dict:
    """Run one pass over val set and compute average metrics."""
    from train_step import train_step_baseline, train_step_jepa

    total_metrics = {}
    count = 0

    for batch in val_loader:
        batch_jax = jax.tree.map(lambda x: jnp.array(x), batch)
        sharded = shard_batch(batch_jax, num_devices)

        # Use model in eval mode (no augment, no dropout)
        # For validation, compute loss only (no gradient)
        state_single = jax_utils.unreplicate(state)

        if config.mode == "baseline":
            # Simple forward pass for val loss
            t = jax.random.uniform(jax.random.PRNGKey(count), (sharded["latent"].shape[1],))
            z0 = sharded["latent"][0]
            y = sharded["label"][0]
            z1 = jax.random.normal(jax.random.PRNGKey(count + 1), z0.shape)
            t_exp = t[:, None, None, None]
            z_t = (1.0 - t_exp) * z0 + t_exp * z1
            v_target = z1 - z0

            out = state_single.apply_fn(
                {"params": state_single.params}, z_t, t, y,
                train=False, mode="baseline",
            )
            l_gen = float(jnp.mean((out["v_pred"] - v_target) ** 2))
            batch_metrics = {"l_gen": l_gen, "l_total": l_gen}
        else:
            # Simplified JEPA val: just L_gen (no masking in val)
            t = jax.random.uniform(jax.random.PRNGKey(count), (sharded["latent"].shape[1],))
            z0 = sharded["latent"][0]
            y = sharded["label"][0]
            z1 = jax.random.normal(jax.random.PRNGKey(count + 1), z0.shape)
            t_exp = t[:, None, None, None]
            z_t = (1.0 - t_exp) * z0 + t_exp * z1
            v_target = z1 - z0

            out = state_single.apply_fn(
                {"params": state_single.params}, z_t, t, y,
                train=False, mode="baseline",
            )
            l_gen = float(jnp.mean((out["v_pred"] - v_target) ** 2))
            batch_metrics = {"l_gen": l_gen, "l_total": l_gen}

        for k, v in batch_metrics.items():
            total_metrics[k] = total_metrics.get(k, 0.0) + v
        count += 1

        if count >= 50:  # Cap val batches for speed
            break

    if count > 0:
        total_metrics = {k: v / count for k, v in total_metrics.items()}

    return total_metrics


if __name__ == "__main__":
    main()
