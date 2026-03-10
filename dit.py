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
    jepa2_split_layer: int = 4
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

        # JEPA2: no extra learnable readout — token-level hidden maps used directly

    def __call__(self, x, t, y, *, train: bool = False, mode: str = "baseline",
                 mask=None, z_s=None, t_s=None,
                 debug_collect_act_rms: bool = False):
        """
        Args:
            x:    (B, 32, 32, 4)  latent input (possibly noised)
            t:    (B,)            timestep in [0, 1]
            y:    (B, num_classes) one-hot class label
            train: whether training (enables label dropout)
            mode:  "baseline" | "jepa" | "teacher" | "jepa2"
            mask:  (B, N) float32 {0,1} token mask — only used in "jepa" mode
            z_s:   (B, 32, 32, 4) global view latent — only used in "jepa2" mode
            t_s:   (B,)           global view timestep — only used in "jepa2" mode

        Returns dict with keys depending on mode:
            "baseline" → {"v_pred": (B, 32, 32, 4)}
            "jepa"     → {"v_pred": (B, 32, 32, 4), "h_pred": (B, N, D)}
            "teacher"  → {"h_target": (B, N, D)}
            "jepa2"    → {"v_pred_g": ..., "v_pred_l": ..., "h_g": (B,N,D), "h_l": (B,N,D)}
        """
        B = x.shape[0]
        H = W = self.latent_size
        C = self.latent_channels
        p = self.patch_size
        grid = H // p

        def _rms(v):
            v = v.astype(jnp.float32)
            return jnp.sqrt(jnp.mean(jnp.square(v)) + 1e-12)

        # --- Fixed positional embedding (stop_gradient) ---
        pos_embed = get_2d_sincos_pos_embed(self.hidden_size, grid)  # (1, N, D) np
        pos_embed = jax.lax.stop_gradient(jnp.array(pos_embed))

        # --- JEPA2: dual-view full-depth forward with readout ---
        if mode == "jepa2":
            D = self.hidden_size

            # Patchify both views
            tokens_l = self.patch_embed(x) + pos_embed       # local (B, N, D)
            tokens_g = self.patch_embed(z_s) + pos_embed     # global (B, N, D)

            # Conditioning: shared y_embed, different t_embed
            y_emb = self.y_embed(y, train=train)
            c_l = self.t_embed(t) + y_emb        # local conditioned on heavier t
            c_g = self.t_embed(t_s) + y_emb      # global conditioned on lighter s

            act_rms = [] if debug_collect_act_rms else None
            h_g = h_l = None
            split = self.jepa2_split_layer

            # Run ALL blocks on BOTH views, tap hidden at split_layer
            for i in range(self.depth):
                tokens_g = self.blocks[i](tokens_g, c_g)
                tokens_l = self.blocks[i](tokens_l, c_l)
                if i == split - 1:
                    h_g = tokens_g   # (B, N, D)
                    h_l = tokens_l   # (B, N, D)
                if debug_collect_act_rms:
                    act_rms.append(_rms(tokens_l))

            # FinalLayer on both views → velocity predictions
            def _unpatchify(tokens, c):
                x_out = self.final_layer(tokens, c)
                x_out = x_out.reshape(B, grid, grid, p, p, C)
                x_out = jnp.einsum("bhwpqc->bhpwqc", x_out)
                return x_out.reshape(B, H, W, C)

            v_pred_g = _unpatchify(tokens_g, c_g)
            v_pred_l = _unpatchify(tokens_l, c_l)

            # Return token-level hidden maps directly
            out = {"v_pred_g": v_pred_g, "v_pred_l": v_pred_l,
                   "h_g": h_g, "h_l": h_l}
            if debug_collect_act_rms:
                out["act_rms"] = jnp.stack(act_rms, axis=0)
            return out

        # --- Patchify + pos embed (baseline / jepa / teacher) ---
        x = self.patch_embed(x)      # (B, N, D)
        x = x + pos_embed            # (B, N, D)

        # --- Conditioning ---
        c = self.t_embed(t) + self.y_embed(y, train=train)  # (B, D)

        # --- Transformer blocks ---
        h_stu = None
        act_rms = [] if debug_collect_act_rms else None

        if mode == "teacher":
            # Early exit: only run first teacher_layer blocks
            for i in range(self.teacher_layer):
                x = self.blocks[i](x, c)
            h_target = self.teacher_head(x)
            return {"h_target": h_target}

        # baseline or jepa: run all blocks
        for i in range(self.depth):
            x = self.blocks[i](x, c)
            if debug_collect_act_rms:
                act_rms.append(_rms(x))

            # Tap student hidden state at student_layer (jepa mode)
            if mode == "jepa" and i == self.student_layer - 1:
                h_stu = x

        # --- Unpatchify ---
        x = self.final_layer(x, c)  # (B, N, p*p*C)
        x = x.reshape(B, grid, grid, p, p, C)
        x = jnp.einsum("bhwpqc->bhpwqc", x)
        v_pred = x.reshape(B, H, W, C)

        if mode == "baseline":
            out = {"v_pred": v_pred}
            if debug_collect_act_rms:
                out["act_rms"] = jnp.stack(act_rms, axis=0)
            return out

        # mode == "jepa"
        h_pred = self.cross_attn_pred(h_stu, mask, pos_embed)
        out = {"v_pred": v_pred, "h_pred": h_pred}
        if debug_collect_act_rms:
            out["act_rms"] = jnp.stack(act_rms, axis=0)
        return out
