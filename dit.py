"""JepaDiT — Diffusion Transformer with 3 forward modes.

Modes (Python string, static at JIT trace time):
  "baseline" — full forward → velocity prediction
  "jepa"     — full forward + student tap + cross-attention predictor
  "teacher"  — early exit at teacher_layer → TeacherHead projection
"""

from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp

from layers import (
    CrossAttentionPredictor,
    DiTBlock,
    FinalLayer,
    MlpBlock,
    PatchEmbed,
    TeacherHead,
    TimestepEmbedder,
    LabelEmbedder,
    get_2d_sincos_pos_embed,
)


class JepaDiT(nn.Module):
    """Dual-mode Diffusion Transformer (baseline + JEPA).

    Uses setup() with explicit block list (not nn.compact) to support
    selective execution per mode.
    """
    patch_size: int = 2
    hidden_size: int = 768
    depth: int = 12
    num_heads: int = 12
    mlp_ratio: float = 4.0
    num_classes: int = 100
    class_dropout_prob: float = 0.1
    student_layer: int = 4
    teacher_layer: int = 8
    latent_size: int = 32
    latent_channels: int = 4

    def setup(self):
        D = self.hidden_size
        p = self.patch_size
        C = self.latent_channels

        # Patchifier
        self.patch_embed = PatchEmbed(patch_size=p, embed_dim=D)

        # Embedders
        self.t_embed = TimestepEmbedder(hidden_size=D)
        self.y_embed = LabelEmbedder(hidden_size=D, dropout_prob=self.class_dropout_prob)

        # Transformer blocks
        self.blocks = [
            DiTBlock(hidden_size=D, num_heads=self.num_heads, mlp_ratio=self.mlp_ratio,
                     name=f"block_{i}")
            for i in range(self.depth)
        ]

        # Final output layer
        self.final_layer = FinalLayer(patch_size=p, out_channels=C, hidden_size=D)

        # JEPA-specific
        self.teacher_head = TeacherHead(hidden_size=D)
        self.cross_attn_pred = CrossAttentionPredictor(hidden_size=D, num_heads=self.num_heads)

    def __call__(self, x, t, y, *, train: bool = False, mode: str = "baseline",
                 mask=None):
        """
        Args:
            x:    (B, 32, 32, 4)  latent input (possibly noised)
            t:    (B,)            timestep in [0, 1]
            y:    (B, num_classes) one-hot class label
            train: whether training (enables label dropout)
            mode:  "baseline" | "jepa" | "teacher"
            mask:  (B, N) float32 {0,1} token mask — only used in "jepa" mode

        Returns dict with keys depending on mode:
            "baseline" → {"v_pred": (B, 32, 32, 4)}
            "jepa"     → {"v_pred": (B, 32, 32, 4), "h_pred": (B, N, D)}
            "teacher"  → {"h_target": (B, N, D)}
        """
        B = x.shape[0]
        H = W = self.latent_size
        C = self.latent_channels
        p = self.patch_size
        grid = H // p

        # --- Fixed positional embedding (stop_gradient) ---
        pos_embed = get_2d_sincos_pos_embed(self.hidden_size, grid)  # (1, N, D) np
        pos_embed = jax.lax.stop_gradient(jnp.array(pos_embed))

        # --- Patchify + pos embed ---
        x = self.patch_embed(x)      # (B, N, D)
        x = x + pos_embed            # (B, N, D)

        # --- Conditioning ---
        c = self.t_embed(t) + self.y_embed(y, train=train)  # (B, D)

        # --- Transformer blocks ---
        h_stu = None

        if mode == "teacher":
            # Early exit: only run first teacher_layer blocks
            for i in range(self.teacher_layer):
                x = self.blocks[i](x, c)
            h_target = self.teacher_head(x)
            return {"h_target": h_target}

        # baseline or jepa: run all blocks
        for i in range(self.depth):
            x = self.blocks[i](x, c)

            # Tap student hidden state at student_layer (jepa mode)
            if mode == "jepa" and i == self.student_layer - 1:
                h_stu = x

        # --- Unpatchify ---
        x = self.final_layer(x, c)  # (B, N, p*p*C)
        x = x.reshape(B, grid, grid, p, p, C)
        x = jnp.einsum("bhwpqc->bhpwqc", x)
        v_pred = x.reshape(B, H, W, C)

        if mode == "baseline":
            return {"v_pred": v_pred}

        # mode == "jepa"
        h_pred = self.cross_attn_pred(h_stu, mask, pos_embed)
        return {"v_pred": v_pred, "h_pred": h_pred}
