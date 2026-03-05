"""JAX-native latent augmentation (jit-safe, static shapes)."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def augment_latents(
    rng: jax.Array,
    z: jax.Array,
    flip_p: float = 0.5,
    jitter_eps: float = 0.01,
) -> jax.Array:
    """Apply training augmentations to a batch of latents.

    Args:
        rng:        PRNGKey
        z:          (B, H, W, C) latent batch
        flip_p:     Probability of horizontal flip per sample
        jitter_eps: Standard deviation of additive Gaussian noise

    Returns:
        Augmented z with same shape.
    """
    rng_flip, rng_jitter = jax.random.split(rng)

    # Random horizontal flip on width axis (axis=2 in NHWC)
    flip_mask = jax.random.bernoulli(rng_flip, flip_p, (z.shape[0], 1, 1, 1))
    z_flipped = jnp.flip(z, axis=2)
    z = jnp.where(flip_mask, z_flipped, z)

    # Additive jitter: z += eps * N(0, I)
    noise = jax.random.normal(rng_jitter, z.shape)
    z = z + jitter_eps * noise

    return z
