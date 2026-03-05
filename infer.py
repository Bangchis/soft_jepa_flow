"""Standalone inference entrypoint for Soft-JEPA-Flow checkpoints.

Loads a checkpoint from ckpt_dir/ckpt_path, samples latent images via Euler ODE,
and optionally decodes with SD-VAE to PNG outputs.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from dit import JepaDiT
from sample import euler_sample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Soft-JEPA-Flow inference")
    parser.add_argument("--ckpt_dir", type=str, default="", help="Checkpoint root directory")
    parser.add_argument("--ckpt_path", type=str, default="", help="Explicit step_* checkpoint directory")
    parser.add_argument("--num_images", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sample_steps",
        type=int,
        default=None,
        help="Override sampling steps (default from checkpoint config or 128)",
    )
    parser.add_argument(
        "--cfg_scale",
        type=float,
        default=None,
        help="Override CFG scale (default from checkpoint config or 1.0)",
    )
    parser.add_argument(
        "--class_ids",
        type=str,
        default="",
        help="Optional comma-separated class ids, e.g. '1,7,9'",
    )
    parser.add_argument(
        "--use_raw_params",
        action="store_true",
        help="Use raw params instead of EMA params",
    )
    parser.add_argument("--decode", action="store_true", help="Decode latents to RGB PNG files")
    parser.add_argument("--decode_batch", type=int, default=32, help="VAE decode batch size")
    parser.add_argument("--out_dir", type=str, default="inference_outputs")
    return parser.parse_args()


def _extract_step_value(path_name: str) -> int:
    match = re.match(r"^step_(\d+)(?:$|_)", path_name)
    if not match:
        return -1
    return int(match.group(1))


def resolve_ckpt_step_dir(ckpt_dir: str, ckpt_path: str) -> Path:
    """Resolve checkpoint step directory in priority order.

    Priority:
    1) explicit --ckpt_path
    2) --ckpt_dir/latest symlink
    3) largest step_* directory under --ckpt_dir
    """
    if ckpt_path:
        p = Path(ckpt_path).expanduser().resolve()
        if not p.is_dir():
            raise FileNotFoundError(f"--ckpt_path is not a directory: {p}")
        if _extract_step_value(p.name) < 0:
            raise ValueError(f"--ckpt_path must point to a step_* directory: {p}")
        return p

    if not ckpt_dir:
        raise ValueError("Provide either --ckpt_path or --ckpt_dir")

    root = Path(ckpt_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"--ckpt_dir is not a directory: {root}")

    latest = root / "latest"
    if latest.exists():
        resolved = latest.resolve()
        if resolved.is_dir() and _extract_step_value(resolved.name) >= 0:
            return resolved
        raise ValueError(f"Invalid latest link target (expect step_* dir): {resolved}")

    candidates: list[Path] = []
    for child in root.iterdir():
        if child.is_dir() and _extract_step_value(child.name) >= 0:
            candidates.append(child)
    if not candidates:
        raise FileNotFoundError(f"No step_* checkpoint found in {root}")

    candidates.sort(
        key=lambda p: (_extract_step_value(p.name), p.stat().st_mtime),
        reverse=True,
    )
    return candidates[0]


def load_ckpt_config(step_dir: Path) -> dict:
    config_path = step_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing config.json in checkpoint dir: {step_dir}")
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def merge_param_trees(tree_a, tree_b, *, tree_a_name: str, tree_b_name: str):
    """Merge two param trees; shared leaves must have the same shape/dtype."""
    from flax.traverse_util import flatten_dict, unflatten_dict

    flat_a = flatten_dict(tree_a)
    flat_b = flatten_dict(tree_b)
    merged = dict(flat_a)

    for key, value_b in flat_b.items():
        if key in merged:
            value_a = merged[key]
            shape_a = getattr(value_a, "shape", None)
            shape_b = getattr(value_b, "shape", None)
            dtype_a = getattr(value_a, "dtype", None)
            dtype_b = getattr(value_b, "dtype", None)
            if shape_a != shape_b or dtype_a != dtype_b:
                key_str = "/".join(key)
                raise ValueError(
                    f"Param mismatch at '{key_str}' while merging "
                    f"{tree_a_name}+{tree_b_name}: "
                    f"{shape_a}/{dtype_a} vs {shape_b}/{dtype_b}"
                )
            continue
        merged[key] = value_b
    return unflatten_dict(merged)


def init_full_params(model_def: JepaDiT, cfg: dict, *, rng: jax.Array):
    """Initialize full parameter tree (JEPA + teacher branches)."""
    latent_size = int(cfg.get("latent_size", 32))
    latent_channels = int(cfg.get("latent_channels", 4))
    patch_size = int(cfg.get("patch_size", 2))
    num_classes = int(cfg.get("num_classes", 100))

    B = 1
    dummy_z = jnp.zeros((B, latent_size, latent_size, latent_channels), dtype=jnp.float32)
    dummy_t = jnp.zeros((B,), dtype=jnp.float32)
    dummy_y = jnp.zeros((B, num_classes), dtype=jnp.float32)

    num_tokens = (latent_size // patch_size) ** 2
    dummy_mask = jnp.ones((B, num_tokens), dtype=jnp.float32)
    rng_jepa, rng_teacher = jax.random.split(rng)

    params_jepa = model_def.init(
        {"params": rng_jepa},
        dummy_z,
        dummy_t,
        dummy_y,
        train=False,
        mode="jepa",
        mask=dummy_mask,
    )["params"]
    params_teacher = model_def.init(
        {"params": rng_teacher},
        dummy_z,
        dummy_t,
        dummy_y,
        train=False,
        mode="teacher",
    )["params"]
    return merge_param_trees(
        params_jepa,
        params_teacher,
        tree_a_name="jepa",
        tree_b_name="teacher",
    )


def restore_required_tensors(step_dir: Path, target_params):
    """Restore strict checkpoint tensors required for inference."""
    import orbax.checkpoint as ocp

    target = {
        "params": target_params,
        "ema_params": target_params,
        "step": np.int32(0),
        "rng": jax.random.PRNGKey(0),
    }

    checkpointer = ocp.PyTreeCheckpointer()
    try:
        restored = checkpointer.restore(str(step_dir), item=target)
    except Exception as e:
        raise RuntimeError(
            "Failed to restore checkpoint (strict mode). "
            "Expected keys: params, ema_params, step, rng. "
            f"checkpoint={step_dir}. Original error: {e}"
        ) from e

    for key in ("params", "ema_params", "step", "rng"):
        if key not in restored:
            raise RuntimeError(f"Checkpoint missing required key '{key}' at {step_dir}")
    return restored


def parse_class_ids(class_ids_str: str, *, num_classes: int, num_images: int) -> np.ndarray:
    """Parse and normalize class ids to exactly num_images entries."""
    if not class_ids_str.strip():
        return np.array([], dtype=np.int32)

    values: list[int] = []
    for tok in class_ids_str.split(","):
        tok = tok.strip()
        if not tok:
            continue
        values.append(int(tok))

    if not values:
        raise ValueError("--class_ids was provided but no valid ids were parsed")

    arr = np.asarray(values, dtype=np.int32)
    if np.any(arr < 0) or np.any(arr >= num_classes):
        raise ValueError(
            f"--class_ids must be in [0, {num_classes - 1}], got: {values}"
        )

    if arr.size < num_images:
        reps = int(math.ceil(num_images / arr.size))
        arr = np.tile(arr, reps)
    return arr[:num_images]


def build_grid(images: np.ndarray) -> np.ndarray:
    """Build a square-ish uint8 image grid from NHWC float images [0,1]."""
    n = images.shape[0]
    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))
    h, w = images.shape[1], images.shape[2]

    grid = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
    for i in range(n):
        r, c = divmod(i, cols)
        img = (np.clip(images[i], 0.0, 1.0) * 255.0).astype(np.uint8)
        grid[r * h : (r + 1) * h, c * w : (c + 1) * w] = img
    return grid


def save_decoded_images(images: np.ndarray, out_dir: Path):
    """Save per-image PNGs and a grid PNG."""
    from PIL import Image

    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    grid = build_grid(images)
    Image.fromarray(grid).save(out_dir / "grid.png")

    for i in range(images.shape[0]):
        img = (np.clip(images[i], 0.0, 1.0) * 255.0).astype(np.uint8)
        Image.fromarray(img).save(images_dir / f"img_{i:04d}.png")


def main():
    args = parse_args()
    step_dir = resolve_ckpt_step_dir(args.ckpt_dir, args.ckpt_path)
    cfg = load_ckpt_config(step_dir)

    sample_steps = int(args.sample_steps if args.sample_steps is not None else cfg.get("sample_steps", 128))
    cfg_scale = float(args.cfg_scale if args.cfg_scale is not None else cfg.get("cfg_scale", 1.0))

    model_def = JepaDiT(
        patch_size=int(cfg.get("patch_size", 2)),
        hidden_size=int(cfg.get("hidden_size", 768)),
        depth=int(cfg.get("depth", 12)),
        num_heads=int(cfg.get("num_heads", 12)),
        mlp_ratio=float(cfg.get("mlp_ratio", 4.0)),
        num_classes=int(cfg.get("num_classes", 100)),
        class_dropout_prob=float(cfg.get("class_dropout_prob", 0.1)),
        student_layer=int(cfg.get("student_layer", 4)),
        teacher_layer=int(cfg.get("teacher_layer", 8)),
        latent_size=int(cfg.get("latent_size", 32)),
        latent_channels=int(cfg.get("latent_channels", 4)),
    )

    rng = jax.random.PRNGKey(args.seed)
    rng, init_rng, sample_rng = jax.random.split(rng, 3)

    target_params = init_full_params(model_def, cfg, rng=init_rng)
    restored = restore_required_tensors(step_dir, target_params)

    params_source = "params" if args.use_raw_params else "ema_params"
    sample_params = restored["params"] if args.use_raw_params else restored["ema_params"]

    num_classes = int(cfg.get("num_classes", 100))
    class_ids_np = parse_class_ids(
        args.class_ids,
        num_classes=num_classes,
        num_images=args.num_images,
    )
    if class_ids_np.size == 0:
        class_ids = jax.random.randint(sample_rng, (args.num_images,), 0, num_classes)
        sample_rng = jax.random.split(sample_rng)[0]
    else:
        class_ids = jnp.array(class_ids_np, dtype=jnp.int32)

    y_cond = jax.nn.one_hot(class_ids, num_classes, dtype=jnp.float32)

    latents = euler_sample(
        model_def.apply,
        sample_params,
        y_cond,
        sample_rng,
        num_steps=sample_steps,
        cfg_scale=cfg_scale,
        latent_shape=(
            int(cfg.get("latent_size", 32)),
            int(cfg.get("latent_size", 32)),
            int(cfg.get("latent_channels", 4)),
        ),
    )
    latents_np = np.asarray(latents, dtype=np.float32)
    class_ids_np = np.asarray(class_ids, dtype=np.int32)

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "latents.npy", latents_np)
    np.save(out_dir / "class_ids.npy", class_ids_np)

    restored_step = int(np.asarray(restored["step"]).item())
    meta = {
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "checkpoint_dir": str(step_dir),
        "checkpoint_step": restored_step,
        "seed": int(args.seed),
        "num_images": int(args.num_images),
        "sample_steps": int(sample_steps),
        "cfg_scale": float(cfg_scale),
        "params_source": params_source,
        "decoded": bool(args.decode),
        "decode_batch": int(args.decode_batch),
        "class_ids": class_ids_np.tolist(),
    }
    with (out_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    if args.decode:
        from vae_decode import decode_latents_nhwc

        images = decode_latents_nhwc(
            latents_np,
            scaling_factor=float(cfg.get("vae_scaling_factor", 0.18215)),
            batch_size=int(args.decode_batch),
            show_progress=True,
            progress_desc="vae-decode[infer]",
        )
        save_decoded_images(images, out_dir)

    print("[infer] Done.")
    print(f"[infer] checkpoint: {step_dir}")
    print(f"[infer] step: {restored_step}")
    print(f"[infer] params source: {params_source}")
    print(f"[infer] num_images: {args.num_images}")
    print(f"[infer] sample_steps: {sample_steps}, cfg_scale: {cfg_scale}")
    print(f"[infer] out_dir: {out_dir}")


if __name__ == "__main__":
    main()
