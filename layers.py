"""Reusable building blocks for DiT-JEPA, following kvfrans conventions.

Initialization conventions (from kvfrans/jax-diffusion-transformer):
- TimestepEmbedder, LabelEmbedder Dense: normal(0.02)
- MlpBlock, PatchEmbed, Attention:       xavier_uniform()
- DiTBlock adaLN projection:             constant(0)   ← adaLN-Zero
- FinalLayer cond + output projection:   constant(0)   ← zero-init
"""

from __future__ import annotations

import math

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def modulate(x, shift, scale):
    """adaLN modulation: x * (1 + scale) + shift.

    x:     (B, N, D)
    shift: (B, D)
    scale: (B, D)
    """
    return x * (1.0 + scale[:, None]) + shift[:, None]


# ---------------------------------------------------------------------------
# 2-D sinusoidal positional embedding (fixed, not learned)
# ---------------------------------------------------------------------------

def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray):
    """embed_dim//2 frequencies for each position in *pos*."""
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega = 1.0 / (10000.0 ** (omega / (embed_dim // 2)))
    out = np.outer(pos, omega)  # (M, D/2)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)  # (M, D)


def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int) -> np.ndarray:
    """Generate 2-D sin-cos positional embedding.

    Returns: (1, grid_size*grid_size, embed_dim)  float32
    """
    grid_h = np.arange(grid_size, dtype=np.float64)
    grid_w = np.arange(grid_size, dtype=np.float64)
    grid = np.meshgrid(grid_w, grid_h)  # (w, h) ordering, row-major
    grid = np.stack(grid, axis=0).reshape(2, -1)  # (2, N)

    half = embed_dim // 2
    emb_h = get_1d_sincos_pos_embed_from_grid(half, grid[1])  # (N, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(half, grid[0])  # (N, D/2)
    emb = np.concatenate([emb_h, emb_w], axis=1)  # (N, D)
    return emb[None].astype(np.float32)  # (1, N, D)


# ---------------------------------------------------------------------------
# TimestepEmbedder — sinusoidal → MLP
# ---------------------------------------------------------------------------

class TimestepEmbedder(nn.Module):
    """Embed scalar timestep t ∈ [0,1] into a D-dim vector."""
    hidden_size: int
    frequency_embedding_size: int = 256

    @nn.compact
    def __call__(self, t):
        # Sinusoidal embedding (like kvfrans timestep_embedding)
        half = self.frequency_embedding_size // 2
        freqs = jnp.exp(-math.log(10000.0) * jnp.arange(half) / half)
        args = t[:, None] * freqs[None]  # (B, half)
        embed = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)  # (B, freq_dim)

        # MLP: Dense → SiLU → Dense
        x = nn.Dense(self.hidden_size,
                     kernel_init=nn.initializers.normal(0.02))(embed)
        x = nn.silu(x)
        x = nn.Dense(self.hidden_size,
                     kernel_init=nn.initializers.normal(0.02))(x)
        return x  # (B, D)


# ---------------------------------------------------------------------------
# LabelEmbedder — one-hot → Dense, with CFG dropout
# ---------------------------------------------------------------------------

class LabelEmbedder(nn.Module):
    """Embed one-hot class label with optional CFG dropout (y → zeros)."""
    hidden_size: int
    dropout_prob: float = 0.1

    @nn.compact
    def __call__(self, y, train: bool = False):
        """y: (B, num_classes) one-hot float32."""
        if train and self.dropout_prob > 0:
            drop_rng = self.make_rng("label_dropout")
            drop_mask = jax.random.bernoulli(
                drop_rng, self.dropout_prob, shape=(y.shape[0], 1)
            )
            y = jnp.where(drop_mask, jnp.zeros_like(y), y)

        return nn.Dense(self.hidden_size,
                        kernel_init=nn.initializers.normal(0.02))(y)  # (B, D)


# ---------------------------------------------------------------------------
# MlpBlock — Dense → GELU → Dense  (dropout OFF, kvfrans convention)
# ---------------------------------------------------------------------------

class MlpBlock(nn.Module):
    mlp_dim: int
    out_dim: int | None = None

    @nn.compact
    def __call__(self, x):
        d = self.out_dim or x.shape[-1]
        x = nn.Dense(self.mlp_dim,
                     kernel_init=nn.initializers.xavier_uniform())(x)
        x = nn.gelu(x)
        x = nn.Dense(d, kernel_init=nn.initializers.xavier_uniform())(x)
        return x


# ---------------------------------------------------------------------------
# PatchEmbed — Conv2D patchifier
# ---------------------------------------------------------------------------

class PatchEmbed(nn.Module):
    """Patchify latent (B, H, W, C) → (B, N, D) via strided conv."""
    patch_size: int
    embed_dim: int

    @nn.compact
    def __call__(self, x):
        p = self.patch_size
        x = nn.Conv(
            features=self.embed_dim,
            kernel_size=(p, p),
            strides=(p, p),
            padding="VALID",
            kernel_init=nn.initializers.xavier_uniform(),
        )(x)  # (B, H/p, W/p, D)
        B = x.shape[0]
        return x.reshape(B, -1, self.embed_dim)  # (B, N, D)


# ---------------------------------------------------------------------------
# DiTBlock — adaLN-Zero transformer block
# ---------------------------------------------------------------------------

class DiTBlock(nn.Module):
    """Transformer block with adaptive layer-norm zero-init (adaLN-Zero)."""
    hidden_size: int
    num_heads: int
    mlp_ratio: float = 4.0

    @nn.compact
    def __call__(self, x, c):
        """
        x: (B, N, D) token features
        c: (B, D)    conditioning vector  (t_emb + y_emb)
        """
        D = self.hidden_size

        # adaLN modulation — 6 params, ZERO-INIT projection
        mod = nn.silu(c)
        mod = nn.Dense(6 * D, kernel_init=nn.initializers.constant(0))(mod)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            jnp.split(mod, 6, axis=-1)
        )

        # --- Self-Attention residual ---
        h = nn.LayerNorm(use_bias=False, use_scale=False)(x)
        h = modulate(h, shift_msa, scale_msa)
        h = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=D,
            kernel_init=nn.initializers.xavier_uniform(),
        )(h, h)
        x = x + gate_msa[:, None] * h

        # --- MLP residual ---
        h = nn.LayerNorm(use_bias=False, use_scale=False)(x)
        h = modulate(h, shift_mlp, scale_mlp)
        h = MlpBlock(mlp_dim=int(D * self.mlp_ratio))(h)
        x = x + gate_mlp[:, None] * h

        return x


# ---------------------------------------------------------------------------
# FinalLayer — modulated LN → Dense output  (both zero-init)
# ---------------------------------------------------------------------------

class FinalLayer(nn.Module):
    """Final projection: unpatchify tokens to (p*p*C) output per token."""
    patch_size: int
    out_channels: int
    hidden_size: int

    @nn.compact
    def __call__(self, x, c):
        """
        x: (B, N, D)
        c: (B, D)
        Returns: (B, N, p*p*C)
        """
        D = self.hidden_size
        p = self.patch_size

        # Conditioning → shift/scale (ZERO-INIT)
        mod = nn.silu(c)
        mod = nn.Dense(2 * D, kernel_init=nn.initializers.constant(0))(mod)
        shift, scale = jnp.split(mod, 2, axis=-1)

        # Modulated LayerNorm
        x = nn.LayerNorm(use_bias=False, use_scale=False)(x)
        x = modulate(x, shift, scale)

        # Output projection (ZERO-INIT)
        x = nn.Dense(p * p * self.out_channels,
                     kernel_init=nn.initializers.constant(0))(x)
        return x


# ---------------------------------------------------------------------------
# TeacherHead — MLP projection for JEPA teacher targets
# ---------------------------------------------------------------------------

class TeacherHead(nn.Module):
    """Project teacher hidden states to JEPA target space."""
    hidden_size: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_size,
                     kernel_init=nn.initializers.xavier_uniform())(x)
        x = nn.gelu(x)
        x = nn.Dense(self.hidden_size,
                     kernel_init=nn.initializers.xavier_uniform())(x)
        return x


# ---------------------------------------------------------------------------
# CrossAttentionPredictor — 1-layer MHA predictor for JEPA
#   Reuses backbone's fixed sin-cos pos_embed (NO learned P)
# ---------------------------------------------------------------------------

class CrossAttentionPredictor(nn.Module):
    """Static-shape cross-attention predictor for JEPA hidden-state prediction.

    Receives pos_embed from backbone — does NOT create its own.
    Q/K get positional info; V does not (semantic only).
    """
    hidden_size: int
    num_heads: int

    @nn.compact
    def __call__(self, h_stu, mask, pos_embed):
        """
        h_stu:     (B, N, D) student hidden states
        mask:      (B, N)    float32 {0,1}  — 1 = target token
        pos_embed: (1, N, D) fixed sin-cos from backbone (stop_gradient)

        Returns: h_pred (B, N, D) — gated to target tokens only
        """
        # Q: only target tokens contribute (gated by mask)
        Q = (h_stu + pos_embed) * mask[..., None]  # (B, N, D)

        # K: all tokens get positional info
        K = h_stu + pos_embed  # (B, N, D)

        # V: semantic content only (no positional info)
        V = h_stu  # (B, N, D)

        # Attention mask: allow only context keys (1-M)
        # context_mask True = attend, False = masked
        context_mask = (1.0 - mask).astype(jnp.bool_)  # (B, N)
        attn_mask = context_mask[:, None, None, :]  # (B, 1, 1, N)

        h_pred = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.hidden_size,
            kernel_init=nn.initializers.xavier_uniform(),
        )(Q, K, V, mask=attn_mask)

        # Gate output to target tokens only
        h_pred = h_pred * mask[..., None]  # (B, N, D)

        return h_pred
