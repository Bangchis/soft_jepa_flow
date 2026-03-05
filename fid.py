"""True FID@4096 using SD-VAE decode + vendored kvfrans InceptionV3.

Pipeline:
1. Decode latents → 256×256 RGB via SD-VAE (CPU, torch)
2. Resize to 299×299, map [0,1] → [-1,1]
3. Extract InceptionV3 pool_3 activations (2048-dim) via JAX pmap
4. Compute Frechet distance vs cached real statistics
"""

from __future__ import annotations

import os

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image

from sample import euler_sample
from vae_decode import decode_latents_nhwc
from inception_fid import get_fid_network, fid_from_stats


# -------------------------------------------------------------------------
# Image preprocessing for Inception
# -------------------------------------------------------------------------

def _prepare_for_inception(images_01: np.ndarray) -> np.ndarray:
    """Resize and rescale decoded images for InceptionV3.

    Args:
        images_01: (N, 256, 256, 3) float32 in [0, 1]

    Returns:
        (N, 299, 299, 3) float32 in [-1, 1]
    """
    out = []
    for img in images_01:
        pil_img = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
        pil_img = pil_img.resize((299, 299), Image.BICUBIC)
        arr = np.array(pil_img).astype(np.float32) / 255.0
        out.append(arr)
    images_299 = np.stack(out)  # (N, 299, 299, 3) in [0, 1]
    return images_299 * 2.0 - 1.0  # → [-1, 1]


# -------------------------------------------------------------------------
# Inception activation extraction (batched, pmap'd)
# -------------------------------------------------------------------------

_inception_fn = None


def _get_inception_fn():
    """Lazy-init the pmap'd InceptionV3 network."""
    global _inception_fn
    if _inception_fn is None:
        _inception_fn = get_fid_network()
    return _inception_fn


def inception_activations(images_m1p1: np.ndarray, batch_size: int = 64) -> np.ndarray:
    """Extract 2048-dim InceptionV3 activations.

    Args:
        images_m1p1: (N, 299, 299, 3) float32 in [-1, 1]
        batch_size:  per-call batch size (will be padded to num_devices)

    Returns:
        (N, 2048) float64 activations
    """
    apply_fn = _get_inception_fn()
    num_devices = jax.local_device_count()
    N = len(images_m1p1)
    all_acts = []

    for i in range(0, N, batch_size):
        batch = images_m1p1[i : i + batch_size]
        cur_bs = len(batch)

        # Pad to multiple of num_devices
        pad_size = (num_devices - cur_bs % num_devices) % num_devices
        if pad_size > 0:
            batch = np.concatenate([batch, np.zeros((pad_size, 299, 299, 3), dtype=np.float32)])

        # Reshape for pmap: (num_devices, per_device, 299, 299, 3)
        batch = batch.reshape(num_devices, -1, 299, 299, 3)
        batch_jax = jnp.array(batch)

        acts = apply_fn(batch_jax)  # (num_devices, per_device, 1, 1, 2048)
        acts = np.array(acts).reshape(-1, 2048)[:cur_bs]
        all_acts.append(acts)

    return np.concatenate(all_acts, axis=0)


def compute_stats(activations: np.ndarray):
    """Compute mean and covariance of activations."""
    mu = np.mean(activations, axis=0)
    sigma = np.cov(activations, rowvar=False)
    return mu, sigma


# -------------------------------------------------------------------------
# Real stats: load from cache or compute + save
# -------------------------------------------------------------------------

def load_or_compute_real_stats(val_loader, config) -> tuple[np.ndarray, np.ndarray]:
    """Get real image stats for FID, with disk caching.

    On first call: decode val latents → SD-VAE → Inception → save .npz
    On subsequent calls: load from fid_cache_path.
    """
    cache_path = config.fid_cache_path

    if os.path.exists(cache_path):
        print(f"[FID] Loading cached real stats from {cache_path}")
        data = np.load(cache_path)
        return data["mu"], data["sigma"]

    print(f"[FID] Computing real stats from val data ({config.fid_n} samples)...")

    # Collect val latents (deterministic order, no shuffle)
    latents = []
    count = 0
    for batch in val_loader:
        batch_latents = np.array(batch["latent"])  # (B, 32, 32, 4)
        for lat in batch_latents:
            if count >= config.fid_n:
                break
            latents.append(lat)
            count += 1
        if count >= config.fid_n:
            break

    latents = np.stack(latents[:config.fid_n])  # (fid_n, 32, 32, 4)

    # Decode via SD-VAE (CPU)
    images_01 = decode_latents_nhwc(latents, batch_size=config.fid_decode_batch)

    # Prepare for Inception
    images_m1p1 = _prepare_for_inception(images_01)

    # Inception activations
    acts = inception_activations(images_m1p1, batch_size=config.fid_inception_batch)
    mu, sigma = compute_stats(acts)

    # Save cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez(
        cache_path,
        mu=mu,
        sigma=sigma,
        fid_n=config.fid_n,
        seed=config.seed,
        scaling_factor=config.vae_scaling_factor,
    )
    print(f"[FID] Saved real stats to {cache_path}")

    return mu, sigma


# -------------------------------------------------------------------------
# Main FID@N function
# -------------------------------------------------------------------------

def compute_fid(
    apply_fn,
    params,
    rng: jax.Array,
    val_loader,
    config,
) -> float:
    """Compute FID@fid_n using true SD-VAE decode + vendored InceptionV3.

    1. Get/cache real stats from val data
    2. Generate fid_n fake samples via Euler sampler
    3. Decode via SD-VAE, extract Inception features
    4. Compute Frechet distance

    Returns:
        FID score (float)
    """
    # Real stats (cached after first call)
    mu_real, sigma_real = load_or_compute_real_stats(val_loader, config)

    # Generate fake samples in batches
    gen_batch_size = min(64, config.fid_n)
    all_latents = []
    count = 0

    while count < config.fid_n:
        rng, sample_rng, label_rng = jax.random.split(rng, 3)
        n = min(gen_batch_size, config.fid_n - count)

        class_ids = jax.random.randint(label_rng, (n,), 0, config.num_classes)
        y_cond = jax.nn.one_hot(class_ids, config.num_classes)

        z_0 = euler_sample(
            apply_fn, params, y_cond, sample_rng,
            num_steps=config.sample_steps, cfg_scale=config.cfg_scale,
        )
        all_latents.append(np.array(z_0))
        count += n

    fake_latents = np.concatenate(all_latents, axis=0)[:config.fid_n]

    # Decode via SD-VAE (CPU)
    fake_images_01 = decode_latents_nhwc(fake_latents, batch_size=config.fid_decode_batch)

    # Prepare for Inception
    fake_images_m1p1 = _prepare_for_inception(fake_images_01)

    # Inception activations
    acts_fake = inception_activations(fake_images_m1p1, batch_size=config.fid_inception_batch)
    mu_fake, sigma_fake = compute_stats(acts_fake)

    fid = float(fid_from_stats(mu_fake, sigma_fake, mu_real, sigma_real))
    return fid
