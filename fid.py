"""FID@4096 computation using TF Hub InceptionV3.

Pipeline:
1. Generate latent samples via Euler sampler
2. Decode latents to RGB (VAE scaling factor)
3. Resize to 299×299, extract InceptionV3 pool_3 features (2048-dim)
4. Compute Frechet distance vs cached real statistics
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
from scipy import linalg

from sample import euler_sample


# -------------------------------------------------------------------------
# Inception feature extraction (lazy-loaded TF Hub)
# -------------------------------------------------------------------------

_inception_model = None


def _get_inception_model():
    """Lazy-load TF Hub InceptionV3 for FID features."""
    global _inception_model
    if _inception_model is not None:
        return _inception_model

    import tensorflow as tf
    import tensorflow_hub as hub

    _inception_model = hub.load(
        "https://tfhub.dev/tensorflow/tfgan/eval/inception/1"
    )
    return _inception_model


def inception_activations(images_uint8: np.ndarray, batch_size: int = 64) -> np.ndarray:
    """Extract 2048-dim InceptionV3 pool_3 activations.

    Args:
        images_uint8: (N, 299, 299, 3) uint8 RGB images
        batch_size:   TF inference batch size

    Returns:
        (N, 2048) float64 activations
    """
    import tensorflow as tf

    model = _get_inception_model()
    all_acts = []

    for i in range(0, len(images_uint8), batch_size):
        batch = images_uint8[i : i + batch_size]
        batch_tf = tf.constant(batch, dtype=tf.uint8)
        acts = model(batch_tf)["pool_3"]
        all_acts.append(acts.numpy().squeeze(axis=(1, 2)))

    return np.concatenate(all_acts, axis=0)


# -------------------------------------------------------------------------
# FID calculation
# -------------------------------------------------------------------------

def fid_from_stats(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Compute Frechet distance between two multivariate Gaussians."""
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)

    # Numerical stability
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff @ diff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return float(fid)


def compute_stats(activations: np.ndarray):
    """Compute mean and covariance of activations."""
    mu = np.mean(activations, axis=0)
    sigma = np.cov(activations, rowvar=False)
    return mu, sigma


# -------------------------------------------------------------------------
# Real stats caching
# -------------------------------------------------------------------------

_cached_real_stats = None


def get_or_compute_real_stats(
    data_loader,
    fid_n: int = 4096,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Get cached real stats or compute from val data.

    Selects a fixed set of fid_n val samples, decodes to RGB,
    extracts Inception activations, computes μ and Σ.
    """
    global _cached_real_stats
    if _cached_real_stats is not None:
        return _cached_real_stats

    from PIL import Image

    images_299 = []
    count = 0

    for batch in data_loader:
        latents = np.array(batch["latent"])  # (B, 32, 32, 4)
        for latent in latents:
            if count >= fid_n:
                break
            # Decode latent → pseudo-RGB (simplified: normalize to [0,255])
            # In production, use actual VAE decoder
            img = _latent_to_rgb_299(latent)
            images_299.append(img)
            count += 1
        if count >= fid_n:
            break

    images_299 = np.stack(images_299[:fid_n])
    acts = inception_activations(images_299)
    mu, sigma = compute_stats(acts)

    _cached_real_stats = (mu, sigma)
    return mu, sigma


def _latent_to_rgb_299(latent: np.ndarray, scaling_factor: float = 0.18215) -> np.ndarray:
    """Convert a single latent to 299×299 uint8 RGB.

    NOTE: This is a placeholder. For accurate FID, use the actual
    VAE decoder (stabilityai/sd-vae-ft-mse). Without it, we take
    the first 3 channels, rescale, and resize as an approximation.
    """
    from PIL import Image

    # Take first 3 channels as pseudo-RGB proxy
    z = latent / scaling_factor
    z = (z + 1.0) / 2.0  # [-1,1] → [0,1]
    z = np.clip(z, 0, 1)
    rgb = z[:, :, :3]  # (32, 32, 3)
    rgb_uint8 = (rgb * 255).astype(np.uint8)

    img = Image.fromarray(rgb_uint8)
    img = img.resize((299, 299), Image.BICUBIC)
    return np.array(img)


# -------------------------------------------------------------------------
# Main FID@N function
# -------------------------------------------------------------------------

def compute_fid(
    apply_fn,
    params,
    rng: jax.Array,
    val_loader,
    fid_n: int = 4096,
    num_classes: int = 100,
    sample_steps: int = 50,
    cfg_scale: float = 1.0,
    seed: int = 42,
) -> float:
    """Compute FID@fid_n.

    1. Get/cache real stats from val data
    2. Generate fid_n fake samples
    3. Compute FID

    Returns:
        FID score (float)
    """
    # Real stats (cached after first call)
    mu_real, sigma_real = get_or_compute_real_stats(val_loader, fid_n, seed)

    # Generate fake samples in batches
    gen_batch_size = min(64, fid_n)
    all_images = []
    count = 0

    while count < fid_n:
        rng, sample_rng, label_rng = jax.random.split(rng, 3)
        n = min(gen_batch_size, fid_n - count)

        class_ids = jax.random.randint(label_rng, (n,), 0, num_classes)
        y_cond = jax.nn.one_hot(class_ids, num_classes)

        z_0 = euler_sample(
            apply_fn, params, y_cond, sample_rng,
            num_steps=sample_steps, cfg_scale=cfg_scale,
        )
        z_0_np = np.array(z_0)

        for latent in z_0_np:
            img = _latent_to_rgb_299(latent)
            all_images.append(img)

        count += n

    fake_images = np.stack(all_images[:fid_n])
    acts_fake = inception_activations(fake_images)
    mu_fake, sigma_fake = compute_stats(acts_fake)

    fid = fid_from_stats(mu_fake, sigma_fake, mu_real, sigma_real)
    return fid
