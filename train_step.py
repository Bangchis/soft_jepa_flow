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

_COS_EPS = 1e-3
_GRAD_CLIP_NORM = 1.0


def _global_norm(tree):
    leaves = jax.tree.leaves(tree)
    return jnp.sqrt(sum(jnp.sum(jnp.square(x)) for x in leaves) + 1e-12)


def _safe_cosine(a: jax.Array, b: jax.Array) -> jax.Array:
    """Numerically stable cosine similarity on last dim."""
    a2 = jnp.sum(a * a, axis=-1)
    b2 = jnp.sum(b * b, axis=-1)
    denom = jnp.sqrt(jnp.maximum(a2 * b2, _COS_EPS * _COS_EPS))
    cos = jnp.sum(a * b, axis=-1) / denom
    return jnp.clip(cos, -1.0, 1.0)


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

    z0 = jnp.nan_to_num(batch["latent"], nan=0.0, posinf=1e4, neginf=-1e4)
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

    z0 = jnp.nan_to_num(batch["latent"], nan=0.0, posinf=1e4, neginf=-1e4)
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
    mask_ratio = jnp.clip(config_static.mask_ratio, 0.0, 1.0)
    M_tok = (jax.random.uniform(rng_mask, (B, N)) < mask_ratio).astype(jnp.float32)
    if N == 1:
        # With one token, there is no separate context set; disable target mask for stability.
        M_tok = jnp.zeros_like(M_tok)
    else:
        target_count = jnp.sum(M_tok, axis=1, keepdims=True)
        no_target = target_count == 0
        all_target = target_count == N
        first_col = M_tok[:, :1]
        first_col = jnp.where(no_target, jnp.ones_like(first_col), first_col)
        first_col = jnp.where(all_target, jnp.zeros_like(first_col), first_col)
        M_tok = M_tok.at[:, :1].set(first_col)

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
        cos_sim = _safe_cosine(h_pred, h_target)  # (B, N)

        l_jepa = 1.0 - jnp.sum(M_tok * cos_sim) / (jnp.sum(M_tok) + 1e-8)

        lambda_j = config_static.lambda_jepa
        l_total = l_gen + lambda_j * l_jepa

        # Debug metrics to diagnose NaN/Inf quickly.
        mask_count = jnp.sum(M_tok, axis=1)
        v_pred_finite = jnp.mean(jnp.isfinite(v_pred).astype(jnp.float32))
        h_pred_finite = jnp.mean(jnp.isfinite(h_pred).astype(jnp.float32))
        h_target_finite = jnp.mean(jnp.isfinite(h_target).astype(jnp.float32))
        cos_finite = jnp.mean(jnp.isfinite(cos_sim).astype(jnp.float32))

        debug_metrics = {
            "dbg_mask_mean": jnp.mean(M_tok),
            "dbg_mask_min_count": jnp.min(mask_count),
            "dbg_mask_max_count": jnp.max(mask_count),
            "dbg_no_target_rate": jnp.mean((mask_count == 0).astype(jnp.float32)),
            "dbg_all_target_rate": jnp.mean((mask_count == N).astype(jnp.float32)),
            "dbg_tau_min": jnp.min(tau_min),
            "dbg_tau_max": jnp.max(tau_max),
            "dbg_z0_absmax": jnp.max(jnp.abs(z0)),
            "dbg_v_pred_absmax": jnp.max(jnp.abs(jnp.nan_to_num(v_pred))),
            "dbg_h_pred_absmax": jnp.max(jnp.abs(jnp.nan_to_num(h_pred))),
            "dbg_h_target_absmax": jnp.max(jnp.abs(jnp.nan_to_num(h_target))),
            "dbg_v_pred_finite": v_pred_finite,
            "dbg_h_pred_finite": h_pred_finite,
            "dbg_h_target_finite": h_target_finite,
            "dbg_cos_finite": cos_finite,
            "dbg_l_gen_finite": jnp.isfinite(l_gen).astype(jnp.float32),
            "dbg_l_jepa_finite": jnp.isfinite(l_jepa).astype(jnp.float32),
            "dbg_l_total_finite": jnp.isfinite(l_total).astype(jnp.float32),
        }

        return l_total, {"l_gen": l_gen, "l_jepa": l_jepa, "l_total": l_total, **debug_metrics}

    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)

    # Sync across devices
    grads = jax.lax.pmean(grads, axis_name="batch")
    grad_norm = _global_norm(grads)
    grad_scale = jnp.minimum(1.0, _GRAD_CLIP_NORM / (grad_norm + 1e-6))
    grads = jax.tree.map(lambda g: g * grad_scale, grads)
    metrics = jax.lax.pmean(metrics, axis_name="batch")
    metrics["dbg_grad_norm"] = grad_norm
    metrics["dbg_grad_scale"] = grad_scale

    state = state.apply_gradients(grads)

    # EMA update
    ema_decay = config_static.ema_decay
    state = state.update_ema(ema_decay)

    # Advance RNG
    new_rng = jax.random.split(state.rng)[0]
    state = state.replace(rng=new_rng)

    return state, metrics
