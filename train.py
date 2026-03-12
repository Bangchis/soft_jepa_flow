"""Main entry point for Soft-JEPA-Flow training on Kaggle TPU.

Usage:
    git clone <repo> && cd soft_jepa_flow
    pip install -r requirements.txt
    python train.py --mode baseline --data_dir /kaggle/input/miniimagenet256-latents-arrayrecord-sdvae
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import jax_utils
from flax.traverse_util import flatten_dict, unflatten_dict
from flax.training import common_utils
from tqdm.auto import tqdm

from configs import Config
from data import create_loader
from dit import JepaDiT
from train_state import TrainState
from train_step import (
    StaticConfig, train_step_baseline, train_step_jepa, train_step_jepa2,
    _sample_t_logit_normal_shifted, _build_variable_k_mask,
    _sigreg_loss, _cls_sigreg_loss, _token_sigreg_loss,
)
from sample import sample_images
from checkpoint import save_checkpoint, maybe_restore, BestMetricTracker
from logging_utils import (
    init_wandb,
    log_metrics,
    log_nan_inf_counts,
    log_activation_debug,
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


def make_config_static(config: Config) -> StaticConfig:
    """Extract static hyperparams for pmap'd steps."""
    return StaticConfig(
        aug_flip_p=config.aug_flip_p,
        aug_jitter_eps=config.aug_jitter_eps,
        mask_ratio=config.mask_ratio,
        lambda_jepa=config.lambda_jepa,
        ema_decay=config.ema_decay,
        patch_size=config.patch_size,
        latent_size=config.latent_size,
        # Timestep schedule: 0=lognormal (default), 1=uniform
        t_schedule=0 if config.t_schedule == "lognormal" else 1,
        t_lognorm_mean=config.t_lognorm_mean,
        t_lognorm_std=config.t_lognorm_std,
        # JEPA2
        lambda_jepa2=config.lambda_jepa2,
        jepa2_fm_warmup_steps=config.jepa2_fm_warmup_steps,
        jepa2_split_layer=config.jepa2_split_layer,
        jepa2_split_layer_global=config.jepa2_split_layer_global,
        jepa2_mask_lo=config.jepa2_mask_lo,
        jepa2_mask_hi=config.jepa2_mask_hi,
        jepa2_t_shift=config.jepa2_t_shift,
        jepa2_alpha_lo=config.jepa2_alpha_lo,
        jepa2_alpha_hi=config.jepa2_alpha_hi,
        jepa2_sigreg_slices=config.jepa2_sigreg_slices,
        jepa2_sigreg_sigma=config.jepa2_sigreg_sigma,
        jepa2_sigreg_num_points=config.jepa2_sigreg_num_points,
        jepa2_sigreg_domain_lo=config.jepa2_sigreg_domain_lo,
        jepa2_sigreg_domain_hi=config.jepa2_sigreg_domain_hi,
        hidden_size=config.hidden_size,
        lambda_cf=config.lambda_cf,
        lambda_cf_ortho=config.lambda_cf_ortho,
        cf_shallow_layer=config.cf_shallow_layer,
        cf_deep_layer=config.cf_deep_layer,
    )


def shard_batch(batch, num_devices):
    """Reshape batch for pmap: (local_batch, ...) → (num_devices, per_device, ...)."""
    return jax.tree.map(
        lambda x: x.reshape(num_devices, -1, *x.shape[1:]),
        batch,
    )


def merge_param_trees(tree_a, tree_b, *, tree_a_name: str, tree_b_name: str):
    """Merge two param trees; shared leaves must match shape/dtype."""
    flat_a = flatten_dict(tree_a)
    flat_b = flatten_dict(tree_b)

    merged = dict(flat_a)
    for key, value_b in flat_b.items():
        if key in merged:
            value_a = merged[key]
            shape_a = getattr(value_a, "shape", None)
            shape_b = getattr(value_b, "shape", None)
            dtype_a = getattr(value_a, "dtype", None)
            dtype_b = getattr(value_b, "dtype", None)
            if shape_a != shape_b or dtype_a != dtype_b:
                key_str = "/".join(key)
                raise ValueError(
                    f"[init] Param mismatch at '{key_str}' while merging "
                    f"{tree_a_name}+{tree_b_name}: "
                    f"{shape_a}/{dtype_a} vs {shape_b}/{dtype_b}"
                )
            continue
        merged[key] = value_b

    return unflatten_dict(merged)


def init_full_params(model_def, config: Config, *, param_rng, dummy_z, dummy_t, dummy_y):
    """Initialize full param tree so JEPA and teacher branches are present."""
    num_tokens = (config.latent_size // config.patch_size) ** 2
    dummy_mask = jnp.ones((dummy_z.shape[0], num_tokens), dtype=jnp.float32)
    rng_jepa, rng_teacher = jax.random.split(param_rng)

    params_jepa = model_def.init(
        {"params": rng_jepa},
        dummy_z, dummy_t, dummy_y,
        train=False, mode="jepa", mask=dummy_mask,
    )["params"]
    params_teacher = model_def.init(
        {"params": rng_teacher},
        dummy_z, dummy_t, dummy_y,
        train=False, mode="teacher",
    )["params"]

    return merge_param_trees(
        params_jepa,
        params_teacher,
        tree_a_name="jepa",
        tree_b_name="teacher",
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
        jepa2_split_layer=config.jepa2_split_layer,
        jepa2_split_layer_global=config.jepa2_split_layer_global,
        cf_shallow_layer=config.cf_shallow_layer,
        cf_deep_layer=config.cf_deep_layer,
        latent_size=config.latent_size,
        latent_channels=config.latent_channels,
    )

    # --- Init params ---
    rng = jax.random.PRNGKey(config.seed)
    rng, param_rng, state_rng = jax.random.split(rng, 3)

    dummy_z = jnp.zeros((per_device_batch, config.latent_size, config.latent_size,
                          config.latent_channels))
    dummy_t = jnp.zeros((per_device_batch,))
    dummy_y = jnp.zeros((per_device_batch, config.num_classes))

    if config.mode == "jepa2":
        # JEPA2: init with mode="jepa2" to get readout params,
        # then merge with baseline init for sampling compatibility
        rng_j2, rng_bl = jax.random.split(param_rng)

        params_jepa2 = model_def.init(
            {"params": rng_j2},
            dummy_z, dummy_t, dummy_y,
            train=False, mode="jepa2",
            z_s=dummy_z, t_s=dummy_t,
        )["params"]

        params_baseline = model_def.init(
            {"params": rng_bl},
            dummy_z, dummy_t, dummy_y,
            train=False, mode="baseline",
        )["params"]

        params = merge_param_trees(
            params_jepa2, params_baseline,
            tree_a_name="jepa2", tree_b_name="baseline",
        )
        print("[init] Initialized full param tree for jepa2+baseline.")
    else:
        params = init_full_params(
            model_def,
            config,
            param_rng=param_rng,
            dummy_z=dummy_z,
            dummy_t=dummy_t,
            dummy_y=dummy_y,
        )
        print("[init] Initialized full param tree for jepa+teacher.")

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

    # --- Config static (broadcasted as pmap static arg) ---
    config_static = make_config_static(config)

    # --- Best metric tracker ---
    best_tracker = BestMetricTracker(config)

    # --- Fixed noise for sample grids ---
    rng, sample_rng = jax.random.split(rng)
    fixed_sample_rng = sample_rng

    # --- Training loop ---
    if config.mode == "jepa2":
        train_step_fn = train_step_jepa2
    elif config.mode == "jepa":
        train_step_fn = train_step_jepa
    else:
        train_step_fn = train_step_baseline

    print(f"[train] Starting {config.mode} training for {config.steps} steps...")

    remaining_steps = max(0, config.steps - start_step)
    pbar = tqdm(
        total=remaining_steps,
        desc=f"train[{config.mode}]",
        dynamic_ncols=True,
        unit="step",
    )

    for step, sharded_batch in enumerate(prefetcher, start=start_step + 1):
        if step > config.steps:
            break

        step_start = time.perf_counter()
        state, metrics = train_step_fn(state, sharded_batch, config_static)
        step_time = time.perf_counter() - step_start
        pbar.update(1)

        # --- Periodic logging ---
        if step % config.log_every == 0:
            metrics_cpu = jax.tree.map(lambda x: float(x[0]), metrics)
            log_metrics(metrics_cpu, step, prefix="train")
            log_nan_inf_counts(metrics_cpu, step)
            # Default-on model debug: lightweight activation stats at log cadence.
            state_single = jax_utils.unreplicate(state)
            log_activation_debug(state_single.params, step)
            print(f"  step {step}: {metrics_cpu}")
            pbar.set_postfix(
                l_total=f"{metrics_cpu.get('l_total', float('nan')):.4f}",
                step_s=f"{step_time:.3f}",
            )

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
                config,
            )

            log_metrics({"quick_fid_4096": fid_score}, step, prefix="eval")
            print(f"  [FID] step {step}: {fid_score:.2f}")

            if config.best_metric == "quick_fid_4096":
                best_tracker.update(state, fid_score, step)

        # --- Checkpoint ---
        if step % config.ckpt_every == 0:
            save_checkpoint(state, config, step)

    pbar.close()
    print("[train] Training complete.")


def run_validation(state, val_loader, config: Config, num_devices: int) -> dict:
    """Run one pass over val set and compute average metrics."""
    total_metrics = {}
    count = 0
    state_single = jax_utils.unreplicate(state)
    val_pbar = tqdm(
        total=50,
        desc=f"val[{config.mode}]",
        dynamic_ncols=True,
        leave=False,
        unit="batch",
    )

    for batch in val_loader:
        batch_jax = jax.tree.map(lambda x: jnp.array(x), batch)
        sharded = shard_batch(batch_jax, num_devices)

        if config.mode == "jepa2":
            # JEPA2 val: L_gen (both views) + L_pred + L_sig + L_total
            z0 = sharded["latent"][0]
            y = sharded["label"][0]
            B = z0.shape[0]
            p = config.patch_size
            H = W = config.latent_size
            gh, gw = H // p, W // p
            N = gh * gw
            D = config.hidden_size

            rng_base = jax.random.PRNGKey(count)
            rng_t, rng_alpha, rng_noise, rng_mask_r, rng_mask_s, rng_sig = (
                jax.random.split(rng_base, 6)
            )

            # Timestep sampling
            t_val = _sample_t_logit_normal_shifted(rng_t, (B,), shift=config.jepa2_t_shift)
            alpha = jax.random.uniform(rng_alpha, (B,),
                                       minval=config.jepa2_alpha_lo,
                                       maxval=config.jepa2_alpha_hi)
            s_val = jnp.clip(t_val / alpha, 1e-5, 1.0 - 1e-5)

            z1 = jax.random.normal(rng_noise, z0.shape)
            v_target = z1 - z0

            # Variable mask
            M_tok, _, _ = _build_variable_k_mask(
                rng_mask_r, rng_mask_s, B, N,
                lo=config.jepa2_mask_lo, hi=config.jepa2_mask_hi,
            )
            M_2d = M_tok.reshape(B, gh, gw)
            M_lat = jnp.repeat(jnp.repeat(M_2d, p, axis=1), p, axis=2)[..., None]

            t_4d = t_val[:, None, None, None]
            s_4d = s_val[:, None, None, None]
            z_t = (1.0 - t_4d) * z0 + t_4d * z1
            z_s = (1.0 - s_4d) * z0 + s_4d * z1
            z_mix = M_lat * z_t + (1.0 - M_lat) * z_s

            out = state_single.apply_fn(
                {"params": state_single.params}, z_mix, t_val, y,
                train=False, mode="jepa2",
                z_s=z_s, t_s=s_val,
            )
            v_pred_g = out["v_pred_g"]
            v_pred_l = out["v_pred_l"]
            h_g = out["h_g"]             # (B, N, D)
            h_l = out["h_l"]             # (B, N, D)

            l_gen_g = float(jnp.mean((v_pred_g - v_target) ** 2))
            l_gen_l = float(jnp.mean((v_pred_l - v_target) ** 2))
            l_gen = 0.5 * (l_gen_g + l_gen_l)

            # L_pred: token-wise MSE
            l_pred = float(jnp.mean((h_l - h_g) ** 2))

            # SIGReg per token position on global hidden
            l_sig = float(_token_sigreg_loss(
                h_g,         # (B, N, D) — no pmap so already full batch
                rng_sig,
                num_slices=config.jepa2_sigreg_slices,
                sigma=config.jepa2_sigreg_sigma,
                num_points=config.jepa2_sigreg_num_points,
                domain=(config.jepa2_sigreg_domain_lo, config.jepa2_sigreg_domain_hi),
            ))

            l_total = l_gen + config.lambda_jepa2 * (0.95 * l_pred + 0.05 * l_sig)
            batch_metrics = {"l_gen": l_gen, "l_pred": l_pred, "l_sig": l_sig, "l_total": l_total}

        elif config.mode == "baseline":
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
            # JEPA val: L_gen + L_repa + L_total
            z0 = sharded["latent"][0]
            y = sharded["label"][0]
            B = z0.shape[0]
            p = config.patch_size
            H = W = config.latent_size
            gh, gw = H // p, W // p
            N = gh * gw

            rng_base = jax.random.PRNGKey(count)
            rng_t, rng_s, rng_noise, rng_mask = jax.random.split(rng_base, 4)

            t = jax.random.uniform(rng_t, (B,))
            s = jax.random.uniform(rng_s, (B,))
            tau_min = jnp.minimum(t, s)
            tau_max = jnp.maximum(t, s)

            M_tok = (jax.random.uniform(rng_mask, (B, N)) < config.mask_ratio).astype(jnp.float32)
            M_2d = M_tok.reshape(B, gh, gw)
            M_lat = jnp.repeat(jnp.repeat(M_2d, p, axis=1), p, axis=2)[..., None]

            z1 = jax.random.normal(rng_noise, z0.shape)
            v_target = z1 - z0

            tau_min_4d = tau_min[:, None, None, None]
            tau_max_4d = tau_max[:, None, None, None]
            tau_mixed_map = M_lat * tau_max_4d + (1.0 - M_lat) * tau_min_4d

            z_clean = (1.0 - tau_min_4d) * z0 + tau_min_4d * z1
            z_mixed = (1.0 - tau_mixed_map) * z0 + tau_mixed_map * z1

            out_stu = state_single.apply_fn(
                {"params": state_single.params}, z_mixed, tau_max, y,
                train=False, mode="jepa", mask=M_tok,
            )
            v_pred = out_stu["v_pred"]
            h_pred = out_stu["h_pred"]

            out_tea = state_single.apply_fn(
                {"params": state_single.ema_params}, z_clean, tau_min, y,
                train=False, mode="teacher",
            )
            h_target = out_tea["h_target"]

            l_gen = float(jnp.mean((v_pred - v_target) ** 2))
            h2 = jnp.sum(h_pred * h_pred, axis=-1)
            t2 = jnp.sum(h_target * h_target, axis=-1)
            denom = jnp.sqrt(jnp.maximum(h2 * t2, 1e-6))
            cos_sim = jnp.clip(jnp.sum(h_pred * h_target, axis=-1) / denom, -1.0, 1.0)
            l_repa = float(1.0 - jnp.sum(M_tok * cos_sim) / (jnp.sum(M_tok) + 1e-8))
            l_total = l_gen + config.lambda_jepa * l_repa
            batch_metrics = {"l_gen": l_gen, "l_repa": l_repa, "l_total": l_total}

        for k, v in batch_metrics.items():
            total_metrics[k] = total_metrics.get(k, 0.0) + v
        count += 1
        val_pbar.update(1)
        val_pbar.set_postfix(l_total=f"{batch_metrics.get('l_total', float('nan')):.4f}")

        if count >= 50:  # Cap val batches for speed
            break
    val_pbar.close()

    if count > 0:
        total_metrics = {k: v / count for k, v in total_metrics.items()}

    return total_metrics


if __name__ == "__main__":
    main()
