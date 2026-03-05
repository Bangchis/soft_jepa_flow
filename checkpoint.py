"""Checkpoint management: Orbax save/load, best-metric tracking, HF upload."""

from __future__ import annotations

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
    restored = checkpointer.restore(restore_dir, item=target)

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
