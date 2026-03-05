"""SD-VAE CPU decode: single source of truth for latent → RGB conversion.

Uses stabilityai/sd-vae-ft-mse via diffusers on CPU (torch).
Lazy-loads model on first call; subsequent calls reuse the cached model.
"""

from __future__ import annotations

import numpy as np

_vae = None
_torch = None


def _get_vae():
    """Lazy-load SD-VAE on CPU."""
    global _vae, _torch
    if _vae is not None:
        return _vae

    import torch
    from diffusers import AutoencoderKL

    _torch = torch
    _vae = AutoencoderKL.from_pretrained(
        "stabilityai/sd-vae-ft-mse",
        torch_dtype=torch.float32,
    ).to("cpu").eval()

    return _vae


def decode_latents_nhwc(
    latents_nhwc: np.ndarray,
    scaling_factor: float = 0.18215,
    batch_size: int = 32,
) -> np.ndarray:
    """Decode latents to RGB images via SD-VAE on CPU.

    Args:
        latents_nhwc: (N, 32, 32, 4) float32 scaled latents (as stored in training)
        scaling_factor: VAE scaling factor (0.18215 for sd-vae-ft-mse)
        batch_size: sub-batch size to limit CPU memory

    Returns:
        (N, 256, 256, 3) float32 images in [0, 1]
    """
    import torch

    vae = _get_vae()
    N = latents_nhwc.shape[0]
    all_images = []

    for i in range(0, N, batch_size):
        batch_np = latents_nhwc[i : i + batch_size]

        # NHWC → NCHW, unscale
        z = np.transpose(batch_np, (0, 3, 1, 2))  # (B, 4, 32, 32)
        z = z / scaling_factor

        with torch.no_grad():
            z_torch = torch.from_numpy(z).float()
            decoded = vae.decode(z_torch).sample  # (B, 3, 256, 256) in [-1, 1]

        # [-1, 1] → [0, 1], NCHW → NHWC
        images = decoded.clamp(-1, 1).cpu().numpy()
        images = (images + 1.0) / 2.0
        images = np.transpose(images, (0, 2, 3, 1))  # (B, 256, 256, 3)
        all_images.append(images)

    return np.concatenate(all_images, axis=0)
