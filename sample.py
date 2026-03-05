"""Euler ODE sampler for rectified flow with Classifier-Free Guidance."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def euler_sample(
    apply_fn,
    params,
    y_cond: jax.Array,
    rng: jax.Array,
    num_steps: int = 50,
    cfg_scale: float = 1.0,
    latent_shape: tuple = (32, 32, 4),
) -> jax.Array:
    """Generate latent samples via Euler ODE integration (t: 1 → 0).

    Args:
        apply_fn:     model.apply function
        params:       model parameters (typically EMA params)
        y_cond:       (B, num_classes) one-hot conditional labels
        rng:          PRNGKey for initial noise
        num_steps:    Number of Euler steps K
        cfg_scale:    CFG guidance scale (1.0 = no guidance)
        latent_shape: (H, W, C) spatial shape of latents

    Returns:
        z_0: (B, H, W, C) denoised latent approximation
    """
    B = y_cond.shape[0]
    z = jax.random.normal(rng, (B, *latent_shape))
    y_uncond = jnp.zeros_like(y_cond)
    dt = -1.0 / num_steps

    def step_fn(carry, k):
        z_t, t_val = carry

        t = jnp.full((B,), t_val)

        # Conditional prediction
        out_cond = apply_fn(
            {"params": params}, z_t, t, y_cond,
            train=False, mode="baseline",
        )
        v_cond = out_cond["v_pred"]

        if cfg_scale != 1.0:
            # Unconditional prediction
            out_uncond = apply_fn(
                {"params": params}, z_t, t, y_uncond,
                train=False, mode="baseline",
            )
            v_uncond = out_uncond["v_pred"]
            # CFG mixing
            v_pred = v_uncond + cfg_scale * (v_cond - v_uncond)
        else:
            v_pred = v_cond

        z_next = z_t + v_pred * dt
        t_next = t_val + dt
        return (z_next, t_next), None

    # Use lax.scan for efficient compilation
    t_init = 1.0
    (z_0, _), _ = jax.lax.scan(step_fn, (z, t_init), jnp.arange(num_steps))

    return z_0


def sample_images(
    apply_fn,
    params,
    rng: jax.Array,
    num_images: int = 16,
    num_classes: int = 100,
    num_steps: int = 50,
    cfg_scale: float = 1.0,
    latent_shape: tuple = (32, 32, 4),
) -> jax.Array:
    """Generate sample latents with random class labels.

    Returns:
        z_0: (num_images, H, W, C) denoised latents
    """
    rng_label, rng_sample = jax.random.split(rng)

    # Random class labels → one-hot
    class_ids = jax.random.randint(rng_label, (num_images,), 0, num_classes)
    y_cond = jax.nn.one_hot(class_ids, num_classes)

    return euler_sample(
        apply_fn, params, y_cond, rng_sample,
        num_steps=num_steps, cfg_scale=cfg_scale,
        latent_shape=latent_shape,
    )
