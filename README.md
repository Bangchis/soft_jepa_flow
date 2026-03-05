# Standard DiT vs Soft-Masked Flow-JEPA (Cross-Attention) — JAX/Flax (TPU-ready)

This repo implements two training modes on **precomputed SD-VAE latents** (MiniImageNet, 100 classes):

1) **Baseline:** Standard DiT trained with **rectified flow / flow matching** objective.
2) **Proposed:** **Soft-Masked Flow-JEPA** with an asymmetric **Student–Teacher (EMA)** setup and a **1-layer Cross-Attention predictor** trained with a JEPA representation loss on masked target tokens.

The implementation is designed to be **XLA/TPU friendly**:
- **No dynamic shapes** (no boolean indexing like `H[M==1]`).
- Masking is done via **attention masks** and `where`/gating while keeping **static tensor shapes**.

---

## 0. Dataset (Precomputed Latents)

We train on latents encoded offline using:

- **VAE:** `stabilityai/sd-vae-ft-mse` (Diffusers / PyTorch)
- **Image size:** 256×256 (bicubic resize)
- **Normalization:** `x = (x/255 - 0.5)/0.5` ∈ [-1, 1]
- **Latent size:** 32×32×4 (downsample factor 8)
- **Scaling:** `z = posterior.sample() * scaling_factor`, with `scaling_factor = 0.18215`

### Storage format: ArrayRecord
Each record is a binary blob:
- `label`: uint16 little-endian (2 bytes)
- `latent`: raw bytes in **NHWC** layout (32,32,4), typically float16 on disk

Layout (Kaggle dataset):

<latents_dataset>/
meta_train.json
meta_val.json
train/.array_record
val/.array_record


> Training does NOT require the original image dataset nor the split dataset.  
> Only the latents dataset is needed.

---

## 1. Notation & Shapes

- Batch: **B**
- Latents: **Z0 ∈ R^{B×H×W×C}**, with **H=W=32**, **C=4**
- Patch size: **p**
  - tokens: **N = (H/p)·(W/p)**
- Hidden dim: **D**
- Tokens: **X ∈ R^{B×N×D}**

Conditioning:
- 100 classes
- One-hot **y ∈ {0,1}^{100}**
- Class embedding **e_y = MLP(y) ∈ R^{D}**

Timestep embedding:
- **t ∈ [0,1]**
- **e_t = MLP(t) ∈ R^{D}**

---

## 2. Train-time Latent Augmentation (enabled in both modes)

Applied only during training (NOT validation):

1) Random horizontal flip with probability `p=0.5` on the latent grid (width axis).
2) Latent jitter:
\[
Z_0 \leftarrow Z_0 + \varepsilon \cdot \mathcal{N}(0, I), \quad \varepsilon = 0.01
\]

These operations keep static shapes and are cheap.

---

## 3. Mode A — Baseline: Standard DiT + Rectified Flow

### 3.1 Rectified flow path
Sample:
- `t ~ Uniform(0,1)`
- `Z1 ~ N(0, I)`

Interpolate:
\[
Z_t = (1-t)\,Z_0 + t\,Z_1
\]

Velocity target:
\[
v = Z_1 - Z_0
\]

Prediction:
\[
v_{\text{pred}} = f_\theta(Z_t, t, y)
\]

### 3.2 Loss
\[
\mathcal{L}_{gen} = \mathrm{MSE}(v_{\text{pred}}, v)
\]

---

## 4. Mode B — Soft-Masked Flow-JEPA (Cross-Attention Predictor)

### 4.1 Teacher EMA
Student params: θ  
Teacher params: ϕ updated by EMA:
\[
\phi \leftarrow \gamma\phi + (1-\gamma)\theta,\quad \gamma=0.999
\]
Teacher forward is stop-grad.

### 4.2 Dual-timestep soft masking (token-wise)
Sample:
- `t, s ~ Uniform(0,1)`
- `τ_min = min(t,s)`, `τ_max = max(t,s)`

Token mask **M ∈ {0,1}^{B×N}** with mask ratio `r` (default 0.25):
- `M=1`: target tokens (heavier noise)
- `1-M`: context tokens (cleaner)

Token-wise timestep:
\[
\tau_{\text{mixed}} = \tau_{max}\cdot M + \tau_{min}\cdot (1-M)
\]

Noises:
- `Z1 ~ N(0,I)`

Teacher input:
\[
Z_{\text{clean}} = (1-\tau_{min})Z_0 + \tau_{min}Z_1
\]

Student input:
\[
Z_{\text{mixed}} = (1-\tau_{\text{mixed}})Z_0 + \tau_{\text{mixed}}Z_1
\]

### 4.3 Generation loss (on Z_mixed)
\[
v = Z_1 - Z_0,\quad v_{\text{pred}} = f_\theta(Z_{\text{mixed}}, \tau, y)
\]
\[
\mathcal{L}_{gen} = \mathrm{MSE}(v_{\text{pred}}, v)
\]

### 4.4 JEPA loss via 1-layer Cross-Attention (static shape)
Tap hidden states:
- Student at layer `l=4`: \(h_{stu} \in \mathbb{R}^{B\times N\times D}\)
- Teacher at layer `k=8`: \(h_{tea} \in \mathbb{R}^{B\times N\times D}\)

Teacher head:
\[
h_{\text{target}} = \mathrm{MLP}_{tea}(h_{tea})
\]
(stop-grad)

Positional embedding \(P \in \mathbb{R}^{1\times N\times D}\).

Static-shape cross-attn construction:
\[
Q = (h_{stu}+P)\odot M
\]
\[
K = h_{stu}+P
\]
\[
V = h_{stu}
\]

Key mask (allow only context keys):
\[
\text{key\_mask} = (1-M)\in \{0,1\}^{B\times N}
\]
Broadcast to attention mask \(\text{attn\_mask}\in\{0,1\}^{B\times1\times1\times N}\).

Cross-attn:
\[
h_{\text{pred}} = \mathrm{MHA}(Q,K,V;\text{attn\_mask})
\]
Gate to target tokens:
\[
h_{\text{pred}} \leftarrow h_{\text{pred}}\odot M
\]

Cosine JEPA loss (weighted mean over target tokens):
\[
\mathcal{L}_{JEPA}
= 1 - \frac{\sum_i M_i\,\cos(h_{\text{pred},i}, \mathrm{stopgrad}(h_{\text{target},i}))}{\sum_i M_i + \epsilon}
\]

### 4.5 Total loss
\[
\mathcal{L}_{total} = \mathcal{L}_{gen} + \lambda \mathcal{L}_{JEPA},\quad \lambda = 0.1
\]

---

## 5. Inference (Sampling)

Euler sampler in latent space (rectified flow):
- Start: \(Z_1\sim \mathcal{N}(0,I)\)
- Integrate `t: 1 → 0` with K steps:
\[
v_{\text{pred}} = f_\theta(Z_t,t,y)
\]
\[
Z_{t+\Delta t} = Z_t + v_{\text{pred}}\cdot \Delta t,\quad \Delta t=-1/K
\]

Decode:
\[
x = \mathrm{VAE.decode}(Z_0 / \text{scaling\_factor})
\]
Map [-1,1] → [0,1].

---

## 6. Evaluation & Logging

- Track train/val losses: `L_gen`, `L_JEPA` (JEPA mode), `L_total`
- Log sample grids periodically (fixed noise)
- Compute and log **FID@4096** (`quick_fid_4096`) periodically

---

## 7. CLI Flags / Hyperparameters

All key hyperparameters are configurable via flags.

### 7.1 Common flags (baseline + JEPA)
- `--mode`: `baseline` or `jepa`
- `--data_dir`: path to Kaggle latents dataset root (contains `train/`, `val/`)
- `--run_name`: W&B run name
- `--seed`: RNG seed
- `--num_classes`: default 100
- `--global_batch`: total batch size across devices
- `--steps`: total training steps
- `--lr`: learning rate
- `--weight_decay`: weight decay

**Model**
- `--model_size`: e.g., `DiT-B`
- `--patch_size`: latent patch size `p`
- `--dim`: hidden width D
- `--depth`: number of layers
- `--heads`: number of attention heads
- `--dropout`: dropout rate

**Augment**
- `--aug_flip_p`: default `0.5`
- `--aug_jitter_eps`: default `0.01`

**Logging/Eval**
- `--log_every`
- `--eval_every`
- `--sample_every`
- `--fid_every`
- `--fid_n`: default `4096`
- `--wandb_project`, `--wandb_entity`

**Checkpoint**
- `--ckpt_every`
- `--ckpt_dir`
- `--best_metric`: `quick_fid_4096` or `val_loss`
- `--hf_repo_id` (optional): HuggingFace repo to upload best checkpoint
- `--hf_token` (optional): token via env/secrets recommended

### 7.2 JEPA-specific flags
- `--lambda_jepa`: default `0.1`
- `--ema_decay`: default `0.999`
- `--mask_ratio`: default `0.25`
- `--student_layer`: default `4`
- `--teacher_layer`: default `8`

### 7.3 Sampling flags
- `--sample_steps`: Euler steps K (e.g., 32/64/128)
- `--cfg_scale` (optional if you implement CFG)
- `--num_sample_images`: default 16

---

## 8. Example commands

### Baseline
```bash
python train.py \
  --mode baseline \
  --data_dir /kaggle/input/miniimagenet256-latents-arrayrecord-sdvae \
  --run_name baseline_ditb_p2 \
  --seed 42 \
  --global_batch 256 \
  --steps 200000 \
  --lr 1e-4 \
  --patch_size 2 \
  --aug_flip_p 0.5 \
  --aug_jitter_eps 0.01 \
  --fid_n 4096 \
  --lambda_jepa 0.1
JEPA
python train.py \
  --mode jepa \
  --data_dir /kaggle/input/miniimagenet256-latents-arrayrecord-sdvae \
  --run_name jepa_ditb_p2_lam01 \
  --seed 42 \
  --global_batch 256 \
  --steps 200000 \
  --lr 1e-4 \
  --patch_size 2 \
  --mask_ratio 0.25 \
  --ema_decay 0.999 \
  --lambda_jepa 0.1 \
  --aug_flip_p 0.5 \
  --aug_jitter_eps 0.01 \
  --fid_n 4096
9. Static-shape guarantee (XLA/TPU)

Never slice by mask (H[M==1] is forbidden).

Keep tensors full length N and use:

attention key masks (context-only keys)

where/gating by M

weighted reductions sum(M * ...) / sum(M)

This avoids dynamic shapes and recompilation issues.