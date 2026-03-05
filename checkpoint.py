"""Checkpoint management: Orbax save/load, best-metric tracking, HF upload."""

from __future__ import annotations

import io
import json
import os
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import jax
import numpy as np

if TYPE_CHECKING:
    from configs import Config
    from train_state import TrainState


def save_checkpoint(state, config: Config, step: int, ckpt_dir: str | None = None):
    """Save checkpoint using orbax."""
    import orbax.checkpoint as ocp

    if ckpt_dir is None:
        ckpt_dir = config.ckpt_dir

    os.makedirs(ckpt_dir, exist_ok=True)
    save_dir = os.path.join(ckpt_dir, f"step_{step}")

    # Unreplicate state for saving (take device 0)
    from flax import jax_utils
    state_single = jax_utils.unreplicate(state)

    ckpt_data = {
        "params": state_single.params,
        "ema_params": state_single.ema_params,
        "opt_state": state_single.opt_state,
        "step": state_single.step,
        "rng": state_single.rng,
    }

    checkpointer = ocp.PyTreeCheckpointer()
    # Orbax requires destination not to exist. Do not pre-create save_dir.
    # Retry with unique suffix if the target already exists (e.g. rerun/race).
    save_attempt_dir = save_dir
    for attempt in range(5):
        try:
            checkpointer.save(save_attempt_dir, ckpt_data)
            save_dir = save_attempt_dir
            break
        except ValueError as e:
            if "already exists" not in str(e):
                raise
            ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-UTC")
            save_attempt_dir = os.path.join(
                ckpt_dir, f"step_{step}_{ts}_p{os.getpid()}_a{attempt+1}"
            )
            print(
                f"[ckpt] Destination exists, retrying with: {save_attempt_dir}"
            )
    else:
        raise RuntimeError(
            f"Failed to save checkpoint for step {step}: destination keeps colliding."
        )

    # Save config alongside
    config_path = os.path.join(save_dir, "config.json")
    with open(config_path, "w") as f:
        f.write(config.to_json())

    # Update "latest" symlink
    latest_path = os.path.join(ckpt_dir, "latest")
    if os.path.lexists(latest_path):
        os.unlink(latest_path)
    os.symlink(os.path.abspath(save_dir), latest_path)

    print(f"[ckpt] Saved checkpoint at step {step} → {save_dir}")
    return save_dir


def maybe_restore(state, config: Config):
    """Try to restore from latest checkpoint. Returns (state, start_step).

    If no checkpoint found, returns (state, 0).
    """
    import orbax.checkpoint as ocp

    latest_path = os.path.join(config.ckpt_dir, "latest")
    if not os.path.exists(latest_path):
        print("[ckpt] No checkpoint found, starting from scratch.")
        return state, 0

    restore_dir = os.path.realpath(latest_path)
    print(f"[ckpt] Restoring from {restore_dir}")

    from flax import jax_utils
    state_single = jax_utils.unreplicate(state)

    target = {
        "params": state_single.params,
        "ema_params": state_single.ema_params,
        "opt_state": state_single.opt_state,
        "step": state_single.step,
        "rng": state_single.rng,
    }

    checkpointer = ocp.PyTreeCheckpointer()
    try:
        restored = checkpointer.restore(restore_dir, item=target)
    except Exception as e:
        raise RuntimeError(
            "[ckpt] Incompatible checkpoint with current code (strict mode). "
            "This build requires full JEPA+teacher param tree. "
            "Please start from scratch or use a checkpoint created by this version. "
            f"restore_dir={restore_dir}. Original error: {e}"
        ) from e

    state_single = state_single.replace(
        params=restored["params"],
        ema_params=restored["ema_params"],
        opt_state=restored["opt_state"],
        step=restored["step"],
        rng=restored["rng"],
    )

    start_step = int(restored["step"])
    state = jax_utils.replicate(state_single)

    print(f"[ckpt] Restored successfully. Resuming from step {start_step}.")
    return state, start_step


class BestMetricTracker:
    """Track best metric and save/upload when improved."""

    def __init__(self, config: Config, metric_name: str | None = None):
        self.config = config
        self.metric_name = metric_name or config.best_metric
        self.best_value = float("inf")  # lower is better for both FID and val_loss
        self.best_dir = os.path.join(config.ckpt_dir, "best")

    def update(self, state, metric_value: float, step: int) -> bool:
        """Check if metric improved; if so, save best checkpoint.

        Returns True if this was a new best.
        """
        if metric_value >= self.best_value:
            return False

        self.best_value = metric_value
        print(f"[best] New best {self.metric_name} = {metric_value:.4f} at step {step}")

        # Save best checkpoint
        best_save_dir = save_checkpoint(state, self.config, step, ckpt_dir=self.best_dir)

        # Optional HF upload (auto-create repo supported)
        if self.config.hf_repo_id or self.config.hf_username:
            self._upload_to_hf(step=step, metric_value=metric_value, folder_path=best_save_dir)

        return True

    def _resolve_repo_id(self) -> str:
        """Resolve HF repo id from explicit repo_id or username/repo_name."""
        if self.config.hf_repo_id:
            return self.config.hf_repo_id
        if self.config.hf_username and self.config.hf_repo_name:
            return f"{self.config.hf_username}/{self.config.hf_repo_name}"
        return ""

    def _build_model_card(
        self,
        *,
        repo_id: str,
        step: int,
        metric_value: float,
        uploaded_at_utc: str,
    ) -> str:
        """Build an English HF model card with architecture + metrics + upload timing."""
        c = self.config
        objective_text = (
            "L_total = L_gen + lambda_jepa * L_jepa (teacher-student JEPA objective with EMA teacher)."
            if c.mode == "jepa"
            else "L_total = L_gen (rectified-flow velocity prediction objective)."
        )
        jepa_details = (
            f"- Student layer: `{c.student_layer}`\n"
            f"- Teacher layer: `{c.teacher_layer}`\n"
            f"- Mask ratio: `{c.mask_ratio}`\n"
            f"- Lambda JEPA: `{c.lambda_jepa}`\n"
            if c.mode == "jepa"
            else "- JEPA branch: disabled in baseline mode.\n"
        )

        return f"""---
license: apache-2.0
library_name: flax
tags:
- jax
- flax
- diffusion-transformer
- rectified-flow
- jepa
---

# Soft-JEPA-Flow ({c.run_name})

This repository stores checkpoints automatically uploaded from training.

## Model Architecture
- Backbone: DiT-style Transformer in JAX/Flax.
- Patch size: `{c.patch_size}`
- Hidden size: `{c.hidden_size}`
- Depth: `{c.depth}`
- Attention heads: `{c.num_heads}`
- MLP ratio: `{c.mlp_ratio}`
- Latent shape: `{c.latent_size}x{c.latent_size}x{c.latent_channels}`
- Number of classes: `{c.num_classes}`
{jepa_details}
## Training Objective
- Mode: `{c.mode}`
- Objective: {objective_text}
- Optimizer: `{c.opt}` (lr=`{c.lr}`, beta1=`{c.beta1}`, beta2=`{c.beta2}`, weight_decay=`{c.weight_decay}`)
- Timestep schedule: `{c.t_schedule}` (mean=`{c.t_lognorm_mean}`, std=`{c.t_lognorm_std}`)

## Metrics
- Best metric key: `{self.metric_name}`
- Best metric value: `{metric_value:.6f}`
- Best checkpoint step: `{step}`

## Upload Timing (Automatic)
- Upload policy: upload whenever a **new best metric** is observed.
- Upload timestamp (UTC): `{uploaded_at_utc}`
- Run name: `{c.run_name}`
- Path for this artifact: `{c.run_name}/best/step_{step}_{uploaded_at_utc}`

## Evaluation Notes
- FID pipeline uses latent decode with `stabilityai/sd-vae-ft-mse`, then InceptionV3 features.
- Inception input range is `[-1, 1]` after resize to `299x299`.

## Minimal Sampling/Inference Note
- The sampler uses the velocity head in `mode="baseline"` (including checkpoints trained with JEPA).

## Source
- HF repo id: `{repo_id}`
"""

    def _upload_model_card(
        self,
        *,
        api,
        repo_id: str,
        readme_text: str,
        step: int,
        metric_value: float,
    ):
        """Upload/overwrite README.md model card at repo root."""
        api.upload_file(
            path_or_fileobj=io.BytesIO(readme_text.encode("utf-8")),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model",
            commit_message=(
                f"update model card: best {self.metric_name}={metric_value:.6f} step={step}"
            ),
        )

    def _upload_to_hf(self, *, step: int, metric_value: float, folder_path: str):
        """Upload best checkpoint to HuggingFace Hub (create repo on first upload)."""
        token = os.environ.get("HF_TOKEN", "")
        if not token:
            print("[best] HF_TOKEN not set, skipping upload.")
            return

        repo_id = self._resolve_repo_id()
        if not repo_id:
            print("[best] Missing HF repo target. Set --hf_repo_id or (--hf_username + --hf_repo_name).")
            return

        try:
            from huggingface_hub import HfApi

            api = HfApi(token=token)
            api.create_repo(
                repo_id=repo_id,
                repo_type="model",
                private=self.config.hf_private,
                exist_ok=True,
            )

            ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-UTC")
            model_card = self._build_model_card(
                repo_id=repo_id,
                step=step,
                metric_value=metric_value,
                uploaded_at_utc=ts,
            )
            self._upload_model_card(
                api=api,
                repo_id=repo_id,
                readme_text=model_card,
                step=step,
                metric_value=metric_value,
            )

            path_in_repo = f"{self.config.run_name}/best/step_{step}_{ts}"
            commit_message = (
                f"best {self.metric_name}={metric_value:.6f} "
                f"step={step} at {ts}"
            )

            api.upload_folder(
                folder_path=folder_path,
                repo_id=repo_id,
                repo_type="model",
                path_in_repo=path_in_repo,
                commit_message=commit_message,
            )
            print(f"[best] Uploaded to HF: {repo_id}/{path_in_repo}")
        except Exception as e:
            print(f"[best] HF upload failed: {e}")
