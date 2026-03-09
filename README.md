# Soft-JEPA-Flow (JAX/Flax, TPU-friendly)

This repository trains a latent generative model in two modes:

1. `baseline`: DiT + rectified flow / flow matching.
2. `jepa`: baseline objective + JEPA representation loss (student/EMA-teacher + cross-attention predictor).
3. `jepa2`: baseline objective + dual-timestep CLS JEPA loss + SIGReg regularization (Self-Flow + LeJEPA inspired, no teacher/EMA).

This README reflects the current codebase status, especially:
- `train.py`
- `train_step.py`
- `sample.py`
- `infer.py`
- `fid.py`
- `vae_decode.py`
- `inception_fid.py`
- `configs.py`

## 1) What Has Been Updated vs Older Versions

- Pseudo-RGB FID path has been removed.
  - FID now uses true SD-VAE decode in `vae_decode.py`.
  - Inception is vendored (`inception_fid.py`) and used in the FID pipeline.
- Pseudo-RGB sample logging has been removed.
  - `logging_utils.log_sample_grid` now decodes real RGB via SD-VAE.
- JEPA validation path is integrated in `run_validation`.
  - In JEPA mode, validation logs `L_gen`, `L_repa`, and `L_total`.
- Standalone inference script is now available.
  - `infer.py` supports loading checkpoints and running sampling/optional decode outside `train.py`.

## 2) Input Dataset Format

Training data is precomputed SD-VAE latent data (`stabilityai/sd-vae-ft-mse`):
- latent shape: `(32, 32, 4)` (NHWC)
- each ArrayRecord example:
  - `label`: `uint16` (2 bytes, little-endian)
  - `latent`: remaining bytes, `float16`, reshaped to `(32, 32, 4)`

Expected directory structure:

```text
<DATA_DIR>/
  train/*.array_record
  val/*.array_record
  meta_train.json
  meta_val.json
```

`data.py` parses records and converts labels to one-hot vectors.

## 3) Training Flow

### 3.1 Baseline (rectified flow)

Per batch:
1. Apply latent augmentations (`flip`, `jitter`).
2. Sample timestep `t` (lognormal or uniform schedule).
3. Sample Gaussian noise `z1 ~ N(0, I)`.
4. Build noisy latent `z_t = (1 - t) * z0 + t * z1`.
5. Target velocity is `v_target = z1 - z0`.
6. Model predicts `v_pred` in `mode="baseline"`.
7. Loss is `L_gen = MSE(v_pred, v_target)`.

### 3.2 JEPA

Per batch:
1. Apply the same latent augmentations.
2. Sample `t, s` from the timestep schedule.
3. Compute `tau_min = min(t, s)`, `tau_max = max(t, s)`.
4. Sample token mask `M_tok` with `mask_ratio`.
5. Build:
   - `z_clean` for teacher path (`tau_min` noise level)
   - `z_mixed` for student path (token-wise `tau_min/tau_max` mixing)
6. Student forward in `mode="jepa"` returns:
   - `v_pred`
   - `h_pred`
7. Teacher forward with `ema_params` in `mode="teacher"` returns `h_target`.
8. Loss terms:
   - `L_gen = MSE(v_pred, v_target)`
   - `L_JEPA = 1 - cosine(h_pred, h_target)` on target tokens (`M_tok`)
   - `L_total = L_gen + lambda_jepa * L_JEPA`
9. Optimizer update on student params, then EMA update.

### 3.3 JEPA2 (Self-Flow + LeJEPA)

Per batch:
1. Apply the same latent augmentations.
2. Sample `t` via logit-normal shifted: `t = sigmoid(N(0,1) + 1.78)`.
3. Sample `alpha ~ U(1.4, 2.0)`, compute `s = t / alpha` (ensures `s < t`).
4. Generate shared noise `eps_img ~ N(0, I)` for image tokens and `eps_reg ~ N(0, I)` for CLS token.
5. Build two noise levels: `z_s = (1-s)*z0 + s*eps_img`, `z_t = (1-t)*z0 + t*eps_img`.
6. Noise learnable CLS token: `cls_s = (1-s)*c_reg + s*eps_reg`, `cls_t = (1-t)*c_reg + t*eps_reg`.
7. Sample random mask ratio `ρ_i ~ U(0.2, 0.4)` per sample, exact-k integer masking via argsort+rank.
8. Build mixed view: `z_mix = M * z_t + (1-M) * z_s` (masked tokens get noisier `z_t`).
9. Prepend CLS tokens to form sequences: `Seq_s = [cls_s | z_s]`, `Seq_t = [cls_t | z_mix]` → 257 tokens each.
10. Run first 4 DiT blocks on **both** views (shared weights, different timestep conditioning).
11. Extract CLS vectors: `r_s = H_s[:,0,:]`, `r_t = H_t[:,0,:]`.
12. Continue only t-view through blocks 5→12.
13. FinalLayer on image tokens only (skip CLS) → `v_pred`.
14. Loss terms:
    - `L_gen = MSE(v_pred, eps_img - z0)` (flow matching velocity)
    - `L_JEPA = mean(1 - cosine(r_t, r_s))` (cosine on CLS vectors)
    - `L_SIG = 0.5 * SIGReg(r_s) + 0.5 * SIGReg(r_t)` (Epps-Pulley, 512 slices, global batch via `all_gather`)
    - `L_total = L_gen + 0.05 * (L_JEPA + L_SIG)`
15. No EMA teacher, no stop-grad, no cross-attention predictor.

Key differences from `jepa` mode:
- No teacher network or EMA — both views share the same backbone pass.
- JEPA loss is on CLS vectors only, not per-token.
- SIGReg regularization prevents CLS collapse without stop-grad.
- Timestep scheduling uses logit-normal shifted + alpha-based dual-timestep.
- Variable mask ratio (20-40%) with exact integer token masking.

## 4) Sampling and Inference

### 4.1 Sampling core (`sample.py`)

Sampling uses Euler ODE integration in latent space:
- initialize `z ~ N(0, I)` at `t=1`
- integrate to `t=0` for `sample_steps`
- CFG uses conditional and unconditional passes with `cfg_scale`

Important: sampling always uses `mode="baseline"` velocity head, including checkpoints trained in JEPA mode.

### 4.2 Standalone inference (`infer.py`)

You can run inference directly from checkpoint folders:

```bash
python infer.py --ckpt_dir <CKPT_ROOT> --num_images 16 --decode
```

Checkpoint resolution order:
1. `--ckpt_path` (explicit `step_*` directory)
2. `--ckpt_dir/latest`
3. newest `step_*` inside `--ckpt_dir`

`infer.py` does the following:
1. Reads `config.json` from checkpoint step directory.
2. Builds model with matching architecture and performs strict Orbax restore (`params`, `ema_params`, `step`, `rng`).
3. Runs latent sampling with `euler_sample(...)`.
4. Saves outputs:
   - `latents.npy`
   - `class_ids.npy`
   - `meta.json`
   - when `--decode` is enabled: `grid.png` and `images/img_XXXX.png`

Default inference uses `ema_params`. Use `--use_raw_params` to sample with raw `params`.

## 5) Evaluation: FID and Visualization

### 5.1 FID (`fid.py`)

Current FID pipeline:
1. Load or compute real stats (`mu`, `sigma`) from validation data.
2. Decode latents with SD-VAE (`vae_decode.decode_latents_nhwc`).
3. Resize to `299x299` and map from `[0,1]` to `[-1,1]` for Inception input.
4. Extract Inception features via vendored Inception (`inception_fid.py`) with JAX `pmap`.
5. Compute Frechet distance.

Real stats cache:
- file: `--fid_cache_path` (default shown below)
- decode batch: `--fid_decode_batch`
- inception batch: `--fid_inception_batch`

### 5.2 Sample grid logging (`logging_utils.py`)

`log_sample_grid` decodes true RGB images via SD-VAE before logging to W&B.

## 6) Validation Behavior

`run_validation` is the active validation path:
- `baseline`: logs `L_gen`, `L_total`
- `jepa`: logs `L_gen`, `L_repa`, `L_total`

`L_repa` is the JEPA-like representation term computed from cosine similarity.

## 7) W&B Logging and Model Debug

At `log_every`, training logs regular metrics plus debug metrics:
- gradient and parameter norms:
  - `train/grad_norm`
  - `train/param_norm`
  - `train/grad_scale` (JEPA step)
- divergence counters:
  - `train/nan_count`
  - `train/inf_count`
- activation/adaLN debug (default-on, lightweight):
  - `debug/block{idx}_adaln_mean`
  - `debug/block{idx}_adaln_std`

## 8) Checkpointing and Hugging Face Upload

Checkpoint content includes:
- `params`, `ema_params`, `opt_state`, `step`, `rng`

Behavior:
- periodic save is controlled by `--ckpt_every`
- Orbax destination collision is handled by retrying with timestamp suffix
- `latest` symlink is updated after each save
- strict restore is enforced for current full JEPA+teacher param tree

Best checkpoint tracking:
- metric: `quick_fid_4096` or `val_loss`
- best checkpoints are stored under `ckpt_dir/best`

Hugging Face upload:
- triggered only on new best metric
- auto-creates repo (`create_repo(..., exist_ok=True)`)
- uploads checkpoint folder and updates an English model card
- requires `HF_TOKEN` with write permission

## 9) Installation

```bash
pip install -r requirements.txt
```

Current `requirements.txt` includes:
- JAX/Flax/Optax/Orbax/Grain
- W&B + Hugging Face Hub
- `torch` (CPU index), `diffusers`, `transformers`, `safetensors`
- `requests`, `tqdm`, `Pillow`, `scipy`

## 10) Run Commands

### 10.1 Baseline

```bash
python train.py \
  --mode baseline \
  --data_dir /kaggle/input/datasets/bangchi/miniimagenet256-latents-arrayrecord-sdvae \
  --run_name baseline_run \
  --global_batch 256 \
  --steps 200000 \
  --opt adam --lr 1e-4 --beta1 0.9 --beta2 0.99 --weight_decay 0.0 \
  --t_schedule lognormal --t_lognorm_mean -0.4 --t_lognorm_std 1.0 \
  --cfg_scale 1.0 --sample_steps 50 \
  --fid_n 4096 --fid_cache_path checkpoints/fid_real_stats_4096.npz \
  --fid_decode_batch 32 --fid_inception_batch 64
```

### 10.2 JEPA

```bash
python train.py \
  --mode jepa \
  --data_dir /kaggle/input/datasets/bangchi/miniimagenet256-latents-arrayrecord-sdvae \
  --run_name jepa_run \
  --global_batch 256 \
  --steps 200000 \
  --opt adam --lr 1e-4 --beta1 0.9 --beta2 0.99 --weight_decay 0.0 \
  --t_schedule lognormal --t_lognorm_mean -0.4 --t_lognorm_std 1.0 \
  --mask_ratio 0.25 --ema_decay 0.999 --lambda_jepa 0.1 \
  --cfg_scale 1.0 --sample_steps 50 \
  --fid_n 4096 --fid_cache_path checkpoints/fid_real_stats_4096.npz \
  --fid_decode_batch 32 --fid_inception_batch 64
```

### 10.3 Inference from checkpoint

Use `latest` (or fallback to newest `step_*` if `latest` is missing):

```bash
python infer.py \
  --ckpt_dir /kaggle/working/checkpoints_smoke_jepa_1772736269 \
  --num_images 16 \
  --sample_steps 128 \
  --cfg_scale 1.0 \
  --decode \
  --decode_batch 32 \
  --out_dir /kaggle/working/infer_out
```

Use an explicit step directory:

```bash
python infer.py \
  --ckpt_path /kaggle/working/checkpoints_smoke_jepa_1772736269/step_200 \
  --num_images 6 \
  --class_ids 1,2,3 \
  --decode
```

## 11) Full CLI Flags and Effective Defaults (argparse)

`train.py` uses `Config.from_args()`, so effective runtime defaults come from argparse in `configs.py`.

### Mode/Data
- `--mode` = `baseline` (`baseline|jepa`)
- `--data_dir` = `/kaggle/input/miniimagenet256-latents-arrayrecord-sdvae`
- `--num_classes` = `100`

### Model
- `--patch_size` = `2`
- `--hidden_size` = `768`
- `--depth` = `12`
- `--num_heads` = `12`
- `--mlp_ratio` = `4.0`

### Optimizer
- `--opt` = `adam` (`adam|adamw`)
- `--lr` = `1e-4`
- `--beta1` = `0.9`
- `--beta2` = `0.99`
- `--weight_decay` = `0.0`

### Training/Augment/Timestep
- `--global_batch` = `256`
- `--steps` = `200000`
- `--seed` = `42`
- `--class_dropout_prob` = `0.1`
- `--aug_flip_p` = `0.5`
- `--aug_jitter_eps` = `0.01`
- `--t_schedule` = `lognormal` (`lognormal|uniform`)
- `--t_lognorm_mean` = `-0.4`
- `--t_lognorm_std` = `1.0`

### JEPA
- `--lambda_jepa` = `0.1`
- `--ema_decay` = `0.999`
- `--mask_ratio` = `0.25`
- `--student_layer` = `4`
- `--teacher_layer` = `8`

### Eval/Sampling/FID
- `--cfg_scale` = `1.0`
- `--sample_steps` = `50`
- `--num_sample_images` = `16`
- `--fid_n` = `4096`
- `--fid_cache_path` = `checkpoints/fid_real_stats_4096.npz`
- `--fid_decode_batch` = `32`
- `--fid_inception_batch` = `64`

### Logging/Checkpoint
- `--log_every` = `100`
- `--eval_every` = `5000`
- `--sample_every` = `10000`
- `--fid_every` = `50000`
- `--ckpt_every` = `50000`
- `--run_name` = `run`
- `--ckpt_dir` = `checkpoints`
- `--best_metric` = `quick_fid_4096` (`quick_fid_4096|val_loss`)
- `--hf_repo_id` = `""` (if empty, it falls back to `hf_username/hf_repo_name`)
- `--hf_username` = `"Bangchis"`
- `--hf_repo_name` = `"soft-jepa-flow"`
- `--hf_private` = `False` (enable by passing `--hf_private`)

## 12) Technical Note: Dataclass Defaults vs CLI Defaults

`Config` dataclass includes some values different from argparse defaults.

When you run `python train.py ...`, the actual defaults come from `Config.from_args()` (argparse).

## 13) Known Limitations

1. SD-VAE decode on CPU can be a bottleneck for large `fid_n` or large decode batch.
2. Inception weights in `inception_fid.py` are downloaded on first use.
3. `infer.py` strict restore expects checkpoints compatible with the current full JEPA+teacher parameter tree.

