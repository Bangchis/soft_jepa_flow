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
    # JEPA2
    lambda_jepa2: float = 0.05
    jepa2_split_layer: int = 4
    jepa2_mask_lo: float = 0.2
    jepa2_mask_hi: float = 0.4
    jepa2_t_shift: float = 1.78
    jepa2_alpha_lo: float = 1.4
    jepa2_alpha_hi: float = 2.0
    jepa2_sigreg_slices: int = 512
    jepa2_sigreg_sigma: float = 1.0
    jepa2_sigreg_num_points: int = 17
    jepa2_sigreg_domain_lo: float = -5.0
    jepa2_sigreg_domain_hi: float = 5.0
    hidden_size: int = 768


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
    grad_norm = jax.lax.pmean(_global_norm(grads), axis_name="batch")
    param_norm = jax.lax.pmean(_global_norm(state.params), axis_name="batch")
    metrics = {
        **metrics,
        "grad_norm": grad_norm,
        "param_norm": param_norm,
    }

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

        return l_total, {"l_gen": l_gen, "l_jepa": l_jepa, "l_total": l_total}

    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)

    # Sync across devices
    grads = jax.lax.pmean(grads, axis_name="batch")
    grad_norm = _global_norm(grads)
    grad_scale = jnp.minimum(1.0, _GRAD_CLIP_NORM / (grad_norm + 1e-6))
    grads = jax.tree.map(lambda g: g * grad_scale, grads)
    metrics = jax.lax.pmean(metrics, axis_name="batch")
    param_norm = jax.lax.pmean(_global_norm(state.params), axis_name="batch")
    metrics = {
        **metrics,
        "grad_norm": jax.lax.pmean(grad_norm, axis_name="batch"),
        "grad_scale": jax.lax.pmean(grad_scale, axis_name="batch"),
        "param_norm": param_norm,
    }

    state = state.apply_gradients(grads)

    # EMA update
    ema_decay = config_static.ema_decay
    state = state.update_ema(ema_decay)

    # Advance RNG
    new_rng = jax.random.split(state.rng)[0]
    state = state.replace(rng=new_rng)

    return state, metrics


# -------------------------------------------------------------------------
# JEPA2 helpers
# -------------------------------------------------------------------------

def _sample_t_logit_normal_shifted(rng, shape, shift=1.78, eps=1e-5):
    """Logit-normal shifted timestep: t = sigmoid(N(0,1) + shift)."""
    u = jax.random.normal(rng, shape)
    t = jax.nn.sigmoid(u + shift)
    return jnp.clip(t, eps, 1.0 - eps)


def _build_variable_k_mask(rng_ratio, rng_scores, B, N, lo=0.2, hi=0.4):
    """Variable exact-k mask: random ratio per sample, JAX-friendly.

    Returns:
        mask: (B, N) float32 binary mask (1 = masked/target)
        ratio: (B,) actual ratio per sample
        k: (B,) number of masked tokens per sample
    """
    ratio = jax.random.uniform(rng_ratio, (B,), minval=lo, maxval=hi)
    k = jnp.rint(ratio * N).astype(jnp.int32)
    k = jnp.clip(k, 1, N - 1)

    scores = jax.random.uniform(rng_scores, (B, N))
    # argsort twice gives rank
    idx = jnp.argsort(-scores, axis=-1)          # (B, N)
    batch_idx = jnp.arange(B)[:, None]
    rank = jnp.zeros_like(idx)
    rank = rank.at[batch_idx, idx].set(jnp.arange(N)[None, :])

    mask = (rank < k[:, None]).astype(jnp.float32)
    return mask, ratio, k


def _epps_pulley_1d(y, sigma=1.0):
    """Epps-Pulley 1D normality statistic.

    y: (B,) float32  —  1D projections of batch embeddings.
    Returns scalar.
    """
    y = y.astype(jnp.float32)
    B = y.shape[0]

    diff = y[:, None] - y[None, :]                              # (B, B)
    term1 = jnp.mean(jnp.exp(-(diff ** 2) / (2.0 * sigma ** 2)))

    coeff2 = jnp.sqrt(sigma ** 2 / (sigma ** 2 + 1.0))
    term2 = (2.0 / B) * coeff2 * jnp.sum(
        jnp.exp(-(y ** 2) / (2.0 * (sigma ** 2 + 1.0)))
    )

    term3 = jnp.sqrt(sigma ** 2 / (sigma ** 2 + 2.0))

    return term1 - term2 + term3


def _sigreg_loss(
    R,
    rng,
    num_slices=512,
    sigma=1.0,
    num_points=17,
    domain=(-5.0, 5.0),
):
    """SIGReg loss via sliced Epps-Pulley.

    R: (B_global, D) batch CLS embeddings.
    num_points/domain are kept as explicit configs for paper-compatible
    SIGReg settings, even though the closed-form EP statistic itself
    does not require quadrature points.
    Returns scalar.
    """
    R = R.astype(jnp.float32)
    B, D = R.shape
    domain_lo, domain_hi = domain
    if domain_hi <= domain_lo:
        raise ValueError(f"Invalid SIGReg domain: ({domain_lo}, {domain_hi})")
    if num_points < 2:
        raise ValueError(f"SIGReg num_points must be >= 2, got {num_points}")

    # Center and standardize per dimension
    mean = jnp.mean(R, axis=0, keepdims=True)
    std = jnp.std(R, axis=0, keepdims=True) + 1e-6
    Rn = (R - mean) / std

    # Random unit directions
    A = jax.random.normal(rng, (num_slices, D), dtype=jnp.float32)
    A = A / (jnp.linalg.norm(A, axis=-1, keepdims=True) + 1e-8)

    # Project: (B, Q)
    Y = Rn @ A.T

    # Closed-form EP on slices. Keep a quadrature grid for config parity
    # with paper settings (num_points/domain) and future numerical variants.
    _ = jnp.linspace(domain_lo, domain_hi, num_points, dtype=jnp.float32)
    ep_vals = jax.vmap(lambda col: _epps_pulley_1d(col, sigma=sigma), in_axes=1)(Y)
    return jnp.mean(ep_vals)


def _cls_sigreg_loss(
    r_s,
    r_t,
    rng,
    num_slices=512,
    sigma=1.0,
    num_points=17,
    domain=(-5.0, 5.0),
):
    """SIGReg on both CLS views: 0.5 * SIGReg(r_s) + 0.5 * SIGReg(r_t)."""
    rng1, rng2 = jax.random.split(rng)
    loss_s = _sigreg_loss(
        r_s,
        rng1,
        num_slices=num_slices,
        sigma=sigma,
        num_points=num_points,
        domain=domain,
    )
    loss_t = _sigreg_loss(
        r_t,
        rng2,
        num_slices=num_slices,
        sigma=sigma,
        num_points=num_points,
        domain=domain,
    )
    return 0.5 * (loss_s + loss_t)


# -------------------------------------------------------------------------
# JEPA2 training step (dual-timestep + CLS JEPA + SIGReg)
# -------------------------------------------------------------------------

@functools.partial(
    jax.pmap,
    axis_name="batch",
    donate_argnums=(0,),
    static_broadcasted_argnums=(2,),
)
def train_step_jepa2(state, batch, config_static):
    """One training step for JEPA2 mode.

    Dual-timestep (logit-normal shifted), variable exact-k masking,
    cosine JEPA loss on CLS vectors, SIGReg on global-batch CLS.

    Args:
        state:         TrainState (replicated)
        batch:         {"latent": (B, 32, 32, 4), "label": (B, 100)}
        config_static: StaticConfig with static hyperparams

    Returns:
        (state, metrics)
    """
    # Per-device RNG
    step_rng = jax.random.fold_in(state.rng, jax.lax.axis_index("batch"))
    (rng_aug, rng_t, rng_alpha, rng_noise_img, rng_noise_cls,
     rng_mask_ratio, rng_mask_scores, rng_label, rng_sig, rng_next) = (
        jax.random.split(step_rng, 10)
    )

    z0 = jnp.nan_to_num(batch["latent"], nan=0.0, posinf=1e4, neginf=-1e4)
    y = batch["label"]
    B = z0.shape[0]
    p = config_static.patch_size
    H = W = config_static.latent_size
    gh = H // p
    gw = W // p
    N = gh * gw
    D = config_static.hidden_size

    # Augment
    z0 = augment_latents(
        rng_aug,
        z0,
        flip_p=config_static.aug_flip_p,
        jitter_eps=config_static.aug_jitter_eps,
    )

    # --- Timestep sampling ---
    t = _sample_t_logit_normal_shifted(rng_t, (B,), shift=config_static.jepa2_t_shift)
    alpha = jax.random.uniform(
        rng_alpha, (B,),
        minval=config_static.jepa2_alpha_lo,
        maxval=config_static.jepa2_alpha_hi,
    )
    s = jnp.clip(t / alpha, 1e-5, 1.0 - 1e-5)

    # --- Noise ---
    eps_img = jax.random.normal(rng_noise_img, z0.shape)       # (B, H, W, C)
    eps_reg = jax.random.normal(rng_noise_cls, (B, 1, D))      # (B, 1, D)

    # --- Variable exact-k mask ---
    M_tok, mask_ratio, k_mask = _build_variable_k_mask(
        rng_mask_ratio, rng_mask_scores, B, N,
        lo=config_static.jepa2_mask_lo,
        hi=config_static.jepa2_mask_hi,
    )   # M_tok: (B, N)

    # Upsample token mask to spatial mask
    M_2d = M_tok.reshape(B, gh, gw)
    M_up = jnp.repeat(M_2d, p, axis=1)   # (B, H, gw)
    M_up = jnp.repeat(M_up, p, axis=2)   # (B, H, W)
    M_lat = M_up[..., None]               # (B, H, W, 1)

    # --- Noised latents in spatial domain ---
    t_4d = t[:, None, None, None]
    s_4d = s[:, None, None, None]
    z_t = (1.0 - t_4d) * z0 + t_4d * eps_img
    z_s = (1.0 - s_4d) * z0 + s_4d * eps_img

    # Mixed view: masked tokens get z_t, unmasked get z_s
    z_mix = M_lat * z_t + (1.0 - M_lat) * z_s

    # Velocity target
    v_target = eps_img - z0

    def loss_fn(params):
        # Learnable CLS token from params
        c_reg = jnp.broadcast_to(params['cls_token'], (B, 1, D))  # (B, 1, D)

        # Noised CLS tokens
        t_3d = t[:, None, None]
        s_3d = s[:, None, None]
        cls_t = (1.0 - t_3d) * c_reg + t_3d * eps_reg
        cls_s = (1.0 - s_3d) * c_reg + s_3d * eps_reg

        # Forward
        out = state.apply_fn(
            {"params": params}, z_mix, t, y,
            train=True, mode="jepa2",
            z_s=z_s, t_s=s,
            cls_s_noised=cls_s, cls_t_noised=cls_t,
            rngs={"label_dropout": rng_label},
        )
        v_pred = out["v_pred"]     # (B, H, W, C)
        r_s = out["r_s"]           # (B, D)
        r_t = out["r_t"]           # (B, D)

        # L_gen: MSE on velocity
        l_gen = jnp.mean((v_pred - v_target) ** 2)

        # L_jepa: cosine loss on CLS vectors
        cos_cls = _safe_cosine(r_t, r_s)   # (B,)
        l_jepa = jnp.mean(1.0 - cos_cls)

        # L_sig: SIGReg on globally-gathered CLS embeddings
        r_s_global = jax.lax.all_gather(r_s, axis_name="batch")
        r_s_global = r_s_global.reshape(-1, D)   # (B_global, D)
        r_t_global = jax.lax.all_gather(r_t, axis_name="batch")
        r_t_global = r_t_global.reshape(-1, D)   # (B_global, D)

        l_sig = _cls_sigreg_loss(
            r_s_global, r_t_global, rng_sig,
            num_slices=config_static.jepa2_sigreg_slices,
            sigma=config_static.jepa2_sigreg_sigma,
            num_points=config_static.jepa2_sigreg_num_points,
            domain=(
                config_static.jepa2_sigreg_domain_lo,
                config_static.jepa2_sigreg_domain_hi,
            ),
        )

        lam = config_static.lambda_jepa2
        l_total = l_gen + lam * (l_jepa + l_sig)

        return l_total, {
            "l_gen": l_gen,
            "l_jepa": l_jepa,
            "l_sig": l_sig,
            "l_total": l_total,
            "t_mean": jnp.mean(t),
            "s_mean": jnp.mean(s),
            "alpha_mean": jnp.mean(alpha),
            "mask_ratio_mean": jnp.mean(mask_ratio),
            "k_mask_mean": jnp.mean(k_mask.astype(jnp.float32)),
        }

    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)

    # Sync across devices
    grads = jax.lax.pmean(grads, axis_name="batch")
    grad_norm = _global_norm(grads)
    grad_scale = jnp.minimum(1.0, _GRAD_CLIP_NORM / (grad_norm + 1e-6))
    grads = jax.tree.map(lambda g: g * grad_scale, grads)
    metrics = jax.lax.pmean(metrics, axis_name="batch")
    param_norm = jax.lax.pmean(_global_norm(state.params), axis_name="batch")
    metrics = {
        **metrics,
        "grad_norm": jax.lax.pmean(grad_norm, axis_name="batch"),
        "grad_scale": jax.lax.pmean(grad_scale, axis_name="batch"),
        "param_norm": param_norm,
    }

    state = state.apply_gradients(grads)

    # Sync params → ema_params (no EMA decay in jepa2, but sampling uses ema_params)
    state = state.replace(ema_params=state.params)

    # Advance RNG
    new_rng = jax.random.split(state.rng)[0]
    state = state.replace(rng=new_rng)

    return state, metrics
