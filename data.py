"""Grain-based ArrayRecord data loader for precomputed SD-VAE latents.

Each record: 2 bytes uint16 label + float16 latent (32, 32, 4) NHWC.
"""

from __future__ import annotations

import glob
import os

import grain.python as grain
import jax
import numpy as np


class ParseLatentRecord(grain.MapTransform):
    """Parse a single ArrayRecord blob into label (one-hot) and latent."""

    def __init__(self, num_classes: int = 100):
        self.num_classes = num_classes

    def map(self, raw_bytes: bytes) -> dict:
        # Label: first 2 bytes, uint16 little-endian
        label_int = int.from_bytes(raw_bytes[:2], byteorder="little")
        label_onehot = np.zeros(self.num_classes, dtype=np.float32)
        label_onehot[label_int] = 1.0

        # Latent: remaining bytes, float16 → float32
        latent = np.frombuffer(raw_bytes[2:], dtype=np.float16).copy()
        latent = latent.reshape(32, 32, 4).astype(np.float32)

        return {"label": label_onehot, "latent": latent}


def create_loader(
    data_dir: str,
    split: str,
    global_batch: int,
    num_classes: int = 100,
    seed: int = 42,
    shuffle: bool = True,
) -> grain.DataLoader:
    """Create a Grain DataLoader for ArrayRecord latent shards.

    Args:
        data_dir:     Root dir containing train/ and val/ subdirs.
        split:        "train" or "val".
        global_batch: Total batch size across all devices.
        num_classes:  Number of classes for one-hot.
        seed:         Random seed for shuffling.
        shuffle:      True for train, False for val.

    Returns:
        grain.DataLoader yielding batches of {"label": ..., "latent": ...}.
    """
    shard_dir = os.path.join(data_dir, split)
    shard_paths = sorted(glob.glob(os.path.join(shard_dir, "*.array_record")))
    assert len(shard_paths) > 0, f"No .array_record files in {shard_dir}"

    num_devices = jax.device_count()
    assert global_batch % num_devices == 0, (
        f"global_batch ({global_batch}) must be divisible by device_count ({num_devices})"
    )
    local_batch = global_batch // jax.process_count()

    source = grain.ArrayRecordDataSource(shard_paths)

    sampler = grain.IndexSampler(
        num_records=len(source),
        num_epochs=None if shuffle else 1,
        shard_options=grain.ShardByJaxProcess(),
        shuffle=shuffle,
        seed=seed,
    )

    loader = grain.DataLoader(
        data_source=source,
        sampler=sampler,
        operations=[
            ParseLatentRecord(num_classes),
            grain.Batch(batch_size=local_batch, drop_remainder=True),
        ],
        worker_count=4,
    )
    return loader
