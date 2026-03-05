"""pmap'd training steps for baseline (rectified flow) and JEPA modes.

Both functions are decorated with jax.pmap(axis_name='batch').
RNG is per-device via fold_in(axis_index).
"""

from __future__ import annotations

import functools
from typing import NamedTuple

import jax
import jax.numpy as jnp

from augment import augment_latents


# -------------------------------------------------------------------------
# Timestep sampling: lognormal (default) or uniform
# -------------------------------------------------------------------------

class StaticConfig(NamedTuple):
    """Static hyperparameters broadcasted into pmap."""
    aug_flip_p: float
    aug_jitter_eps: float
    mask_ratio: float
    lambda_jepa: float
    ema_decay: float
    patch_size: int
    latent_size: int
    t_schedule: int
    t_lognorm_mean: float
    t_lognorm_std: float


def sample_timesteps(rng, shape, config_static):
    """Sample timesteps in [0, 1].

    config_static.t_schedule: int (0 = lognormal, 1 = uniform)
    Lognormal: sigmoid(N(mean, std)) — biases toward mid-schedule.
    """
    def _lognormal(rng):
        u = jax.random.normal(rng, shape)
        u = u * config_static.t_lognorm_std + config_static.t_lognorm_mean
        return jax.nn.sigmoid(u)

    def _uniform(rng):
        return jax.random.uniform(rng, shape)

    return jax.lax.cond(config_static.t_schedule == 0, _lognormal, _uniform, rng)


# -------------------------------------------------------------------------
# Baseline training step (rectified flow)
# -------------------------------------------------------------------------

@functools.partial(
    jax.pmap,
    axis_name="batch",
    donate_argnums=(0,),
    static_broadcasted_argnums=(2,),
)
def train_step_baseline(state, batch, config_static):
    """One training step for baseline mode (rectified flow MSE).

    Args:
        state:         TrainState (replicated)
        batch:         {"latent": (B, 32, 32, 4), "label": (B, 100)}
        config_static: StaticConfig with static hyperparams

    Returns:
        (state, metrics)
    """
    # Per-device RNG
    step_rng = jax.random.fold_in(state.rng, jax.lax.axis_index("batch"))
    rng_aug, rng_t, rng_noise, rng_label, rng_next = jax.random.split(step_rng, 5)

    z0 = batch["latent"]
    y = batch["label"]
    B = z0.shape[0]

    # Augment
    z0 = augment_latents(
        rng_aug,
        z0,
        flip_p=config_static.aug_flip_p,
        jitter_eps=config_static.aug_jitter_eps,
    )

    # Sample timestep t ∈ [0,1] (lognormal or uniform) and noise z1 ~ N(0,I)
    t = sample_timesteps(rng_t, (B,), config_static)
    z1 = jax.random.normal(rng_noise, z0.shape)

    # Interpolate: z_t = (1-t)*z0 + t*z1
    t_expand = t[:, None, None, None]
    z_t = (1.0 - t_expand) * z0 + t_expand * z1

    # Velocity target
    v_target = z1 - z0

    def loss_fn(params):
        out = state.apply_fn(
            {"params": params}, z_t, t, y,
            train=True, mode="baseline",
            rngs={"label_dropout": rng_label},
        )
        v_pred = out["v_pred"]
        l_gen = jnp.mean((v_pred - v_target) ** 2)
        return l_gen, {"l_gen": l_gen, "l_total": l_gen}

    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)

    # Sync across devices
    grads = jax.lax.pmean(grads, axis_name="batch")
    metrics = jax.lax.pmean(metrics, axis_name="batch")

    state = state.apply_gradients(grads)

    # Advance RNG
    new_rng = jax.random.split(state.rng)[0]
    state = state.replace(rng=new_rng)

    return state, metrics


# -------------------------------------------------------------------------
# JEPA training step (dual-timestep + cross-attention predictor)
# -------------------------------------------------------------------------

@functools.partial(
    jax.pmap,
    axis_name="batch",
    donate_argnums=(0,),
    static_broadcasted_argnums=(2,),
)
def train_step_jepa(state, batch, config_static):
    """One training step for JEPA mode.

    Args:
        state:         TrainState (replicated)
        batch:         {"latent": (B, 32, 32, 4), "label": (B, 100)}
        config_static: StaticConfig with static hyperparams:
            aug_flip_p, aug_jitter_eps, mask_ratio, lambda_jepa, ema_decay,
            patch_size, latent_size

    Returns:
        (state, metrics)
    """
    # Per-device RNG
    step_rng = jax.random.fold_in(state.rng, jax.lax.axis_index("batch"))
    rng_aug, rng_t, rng_s, rng_noise, rng_mask, rng_label, rng_next = (
        jax.random.split(step_rng, 7)
    )

    z0 = batch["latent"]
    y = batch["label"]
    B = z0.shape[0]
    p = config_static.patch_size
    H = W = config_static.latent_size
    gh = H // p
    gw = W // p
    N = gh * gw

    # Augment
    z0 = augment_latents(
        rng_aug,
        z0,
        flip_p=config_static.aug_flip_p,
        jitter_eps=config_static.aug_jitter_eps,
    )

    # --- Dual timestep (lognormal or uniform) ---
    t = sample_timesteps(rng_t, (B,), config_static)
    s = sample_timesteps(rng_s, (B,), config_static)
    tau_min = jnp.minimum(t, s)
    tau_max = jnp.maximum(t, s)

    # --- Token mask M_tok: (B, N) ---
    mask_ratio = config_static.mask_ratio
    M_tok = (jax.random.uniform(rng_mask, (B, N)) < mask_ratio).astype(jnp.float32)

    # --- Build M_lat from M_tok (patch-aligned upsample) ---
    M_2d = M_tok.reshape(B, gh, gw)
    M_up = jnp.repeat(M_2d, p, axis=1)   # (B, H, gw)
    M_up = jnp.repeat(M_up, p, axis=2)   # (B, H, W)
    M_lat = M_up[..., None]               # (B, H, W, 1)

    # --- Noise ---
    z1 = jax.random.normal(rng_noise, z0.shape)
    v_target = z1 - z0

    # --- Build inputs in latent 2D (before patchify) ---
    tau_min_4d = tau_min[:, None, None, None]
    tau_max_4d = tau_max[:, None, None, None]
    tau_mixed_map = M_lat * tau_max_4d + (1.0 - M_lat) * tau_min_4d  # (B,H,W,1)

    z_clean = (1.0 - tau_min_4d) * z0 + tau_min_4d * z1        # teacher input
    z_mixed = (1.0 - tau_mixed_map) * z0 + tau_mixed_map * z1   # student input

    def loss_fn(params):
        # Student forward (jepa mode)
        out_stu = state.apply_fn(
            {"params": params}, z_mixed, tau_max, y,
            train=True, mode="jepa", mask=M_tok,
            rngs={"label_dropout": rng_label},
        )
        v_pred = out_stu["v_pred"]     # (B, H, W, C)
        h_pred = out_stu["h_pred"]     # (B, N, D) — already gated by M_tok

        # Teacher forward (stop-grad, EMA params)
        out_tea = state.apply_fn(
            {"params": jax.lax.stop_gradient(state.ema_params)},
            z_clean, tau_min, y,
            train=False, mode="teacher",
        )
        h_target = jax.lax.stop_gradient(out_tea["h_target"])  # (B, N, D)

        # L_gen: MSE on velocity
        l_gen = jnp.mean((v_pred - v_target) ** 2)

        # L_JEPA: 1 - cosine similarity, weighted by M_tok (target tokens)
        h_pred_norm = h_pred / (jnp.linalg.norm(h_pred, axis=-1, keepdims=True) + 1e-8)
        h_tgt_norm = h_target / (jnp.linalg.norm(h_target, axis=-1, keepdims=True) + 1e-8)
        cos_sim = jnp.sum(h_pred_norm * h_tgt_norm, axis=-1)  # (B, N)

        l_jepa = 1.0 - jnp.sum(M_tok * cos_sim) / (jnp.sum(M_tok) + 1e-8)

        lambda_j = config_static.lambda_jepa
        l_total = l_gen + lambda_j * l_jepa

        return l_total, {"l_gen": l_gen, "l_jepa": l_jepa, "l_total": l_total}

    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)

    # Sync across devices
    grads = jax.lax.pmean(grads, axis_name="batch")
    metrics = jax.lax.pmean(metrics, axis_name="batch")

    state = state.apply_gradients(grads)

    # EMA update
    ema_decay = config_static.ema_decay
    state = state.update_ema(ema_decay)

    # Advance RNG
    new_rng = jax.random.split(state.rng)[0]
    state = state.replace(rng=new_rng)

    return state, metrics
