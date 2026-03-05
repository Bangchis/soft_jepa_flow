# Standard DiT vs Soft-Masked Flow-JEPA (Cross-Attention) — JAX/Flax (TPU-ready)

This repo compares two models trained on **precomputed SD-VAE latents** (MiniImageNet, 100 classes):

1) **Baseline:** Standard DiT with a **rectified-flow / flow-matching** objective.
2) **Proposed:** **Soft-Masked Flow-JEPA** with Student–Teacher (EMA) and a **1-layer Cross-Attention predictor** + JEPA representation loss.

The implementation is **XLA/TPU-friendly**:
- No dynamic shapes from boolean indexing (no `H[M==1]`).
- Masking is done via **attention masks** and `where`/gating with **static shapes**.

This README also matches key conventions from `kvfrans/jax-diffusion-transformer`:
- Adam optimizer setup :contentReference[oaicite:2]{index=2}
- label dropout for CFG (separate RNG stream) :contentReference[oaicite:3]{index=3}
- CFG mixing formula in inference/eval :contentReference[oaicite:4]{index=4}
- adaLN-Zero / FinalLayer zero-init stability trick :contentReference[oaicite:5]{index=5}
- MLP dropout effectively OFF in the reference code :contentReference[oaicite:6]{index=6}

---

## 0) Dataset (Precomputed Latents)

Latents are encoded offline using:
- **VAE:** `stabilityai/sd-vae-ft-mse`
- **Image resize:** 256×256 (bicubic)
- **Normalize:** `x = (x/255 - 0.5)/0.5` in [-1, 1]
- **Latent size:** 32×32×4 (downsample factor 8; same convention discussed in kvfrans README) :contentReference[oaicite:7]{index=7}
- **Scaling:** `z = posterior.sample() * scaling_factor`, with scaling_factor ≈ 0.18215

### ArrayRecord storage format
Each record is a binary blob:
- `label`: uint16 little-endian (2 bytes)
- `latent`: raw bytes, NHWC layout (32,32,4), usually float16 on disk

Dataset layout:

<latents_dataset>/
meta_train.json
meta_val.json
train/.array_record
val/.array_record


> Training only needs the latents dataset (no need to add the original images or the split dataset).

---

## 1) Notation & Shapes

- Batch: **B**
- Latents: **Z0 ∈ R^{B×H×W×C}**, H=W=32, C=4 (NHWC)
- Patch size: **p**
  - tokens: **N = (H/p)·(W/p)**
- Hidden dim: **D**
- Tokens: **X ∈ R^{B×N×D}**

Conditioning:
- One-hot class: **y ∈ {0,1}^{100}**
- Class embed: **e_y = MLP(y) ∈ R^{D}**
- Timestep embed: **e_t = MLP(t) ∈ R^{D}**, t∈[0,1]

---

## 2) Train-time Latent Augmentation

Applied only in training (not validation):
1) Random horizontal flip (p=0.5) on latent grid (width axis)
2) Latent jitter:
\[
Z_0 \leftarrow Z_0 + \varepsilon \cdot \mathcal{N}(0,I),\quad \varepsilon = 0.01
\]

---

## 3) Mode A — Baseline (Rectified Flow)

Sample:
- \( t \sim U(0,1) \)
- \( Z_1 \sim \mathcal{N}(0,I) \)

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

Loss:
\[
\mathcal{L}_{gen} = \mathrm{MSE}(v_{\text{pred}}, v)
\]

---

## 4) Mode B — Soft-Masked Flow-JEPA (Cross-Attention Predictor)

### 4.1 Teacher EMA
Student params: θ  
Teacher params: ϕ updated by EMA:
\[
\phi \leftarrow \gamma\phi + (1-\gamma)\theta,\quad \gamma=0.999
\]

Teacher forward is stop-grad.

### 4.2 Dual-timestep soft masking (token-wise)
Sample:
- \( t,s \sim U(0,1) \)
- \( \tau_{min}=\min(t,s) \), \( \tau_{max}=\max(t,s) \)

Token mask \( M \in \{0,1\}^{B\times N} \) with mask ratio r (default 0.25):
- M=1 target tokens (heavier noise)
- 1−M context tokens (cleaner)

Token-wise timestep:
\[
\tau_{\text{mixed}} = \tau_{max}\,M + \tau_{min}\,(1-M)
\]

Noise:
- \( Z_1 \sim \mathcal{N}(0,I) \)

Teacher input:
\[
Z_{\text{clean}} = (1-\tau_{min})Z_0 + \tau_{min}Z_1
\]

Student input:
\[
Z_{\text{mixed}} = (1-\tau_{\text{mixed}})Z_0 + \tau_{\text{mixed}}Z_1
\]

### 4.3 Generation loss (same target v)
\[
v = Z_1 - Z_0,\quad v_{\text{pred}} = f_\theta(Z_{\text{mixed}}, \tau, y)
\]
\[
\mathcal{L}_{gen} = \mathrm{MSE}(v_{\text{pred}}, v)
\]

### 4.4 JEPA loss via 1-layer Cross-Attention (static shape)
Tap hidden states:
- Student at layer l=4: \( h_{stu}\in \mathbb{R}^{B\times N\times D} \)
- Teacher at layer k=8: \( h_{tea}\in \mathbb{R}^{B\times N\times D} \)

Teacher head:
\[
h_{\text{target}} = \mathrm{MLP}_{tea}(h_{tea})
\]
(stop-grad)

Positional embedding \( P\in\mathbb{R}^{1\times N\times D} \).

Static-shape cross-attn:
\[
Q = (h_{stu}+P)\odot M,\quad K = h_{stu}+P,\quad V = h_{stu}
\]
Key mask allows only context keys:
\[
\text{key\_mask} = (1-M)\in\{0,1\}^{B\times N}
\]
Broadcast to \( \text{attn\_mask}\in\{0,1\}^{B\times 1\times 1\times N} \).

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
\mathcal{L}_{JEPA} =
1-\frac{\sum_i M_i\,\cos(h_{\text{pred},i},\mathrm{stopgrad}(h_{\text{target},i}))}{\sum_i M_i+\epsilon}
\]

### 4.5 Total loss
\[
\mathcal{L}_{total}=\mathcal{L}_{gen}+\lambda\,\mathcal{L}_{JEPA},\quad \lambda=0.1
\]

---

## 5) DiT-kvfrans Compatibility (Optimizer, Dropout, Init)

### 5.1 Optimizer (matches kvfrans)
Reference initializes Adam as:
`optax.adam(learning_rate=lr, b1=beta1, b2=beta2)` :contentReference[oaicite:8]{index=8}

Defaults used here (and recommended):
- `--opt adam`
- `--lr 1e-4`
- `--beta1 0.9`
- `--beta2 0.99`
- `--weight_decay 0.01`

### 5.2 Dropout conventions
- Transformer dropout is effectively OFF by default in the kvfrans MLP block (dropout lines are commented). :contentReference[oaicite:9]{index=9}
- The key stochastic mechanism is **label dropout** for CFG (separate RNG stream `label_dropout`). :contentReference[oaicite:10]{index=10}

For one-hot conditioning, we emulate kvfrans label dropout:
- With probability `class_dropout_prob`, set one-hot `y = 0` vector (unconditional).

### 5.3 Zero-init stability (adaLN-Zero + FinalLayer)
Reference uses zero-init for:
- adaLN modulation projection: `Dense(6*hidden, kernel_init=constant(0))`
- final conditioning projection: `Dense(2*hidden, kernel_init=constant(0))`
- final output projection: `Dense(p^2*out_channels, kernel_init=constant(0))` :contentReference[oaicite:11]{index=11}

These are preserved to keep early training close to identity / stable.

---

## 6) Inference / Sampling (Euler) + CFG Scale

We sample in latent space.

### 6.1 Euler sampler (rectified flow)
Start:
\[
Z_1 \sim \mathcal{N}(0,I)
\]
Integrate t: 1 → 0 with K steps, Δt = −1/K:
\[
v_{\text{pred}} = f_\theta(Z_t,t,y)
\]
\[
Z_{t+\Delta t} = Z_t + v_{\text{pred}}\,\Delta t
\]

Decode:
\[
x = \mathrm{VAE.decode}(Z_0/\text{scaling\_factor})
\]
and map [-1,1] → [0,1].

### 6.2 Classifier-Free Guidance (CFG)
kvfrans performs CFG by evaluating the model twice (cond + uncond) and mixing: :contentReference[oaicite:12]{index=12}
\[
\text{pred} = \text{pred}_{uncond} + s\cdot(\text{pred}_{cond}-\text{pred}_{uncond})
\]
where \(s=\text{cfg\_scale}\).

In this repo:
- `pred` is the velocity \(v_{\text{pred}}\) (baseline) or the same velocity head (JEPA mode).
- `uncond` is produced by setting one-hot `y=0` vector.
- `cfg_scale=1.0` means no guidance (just conditional).

Recommended flag:
- `--cfg_scale <float>` (typical tuning range is task-dependent; start from 1.0 and increase if you want stronger class adherence.)

---

## 7) Evaluation & Logging
- Track train/val: `L_gen`, `L_JEPA` (JEPA), `L_total`
- Log sample grids periodically (fixed noise)
- Compute **FID@4096** (`quick_fid_4096`) periodically

---

## 8) CLI Flags / Hyperparameters

### 8.1 Common flags
**Data & run**
- `--mode`: `baseline` or `jepa`
- `--data_dir`: latents dataset root (has `train/`, `val/`)
- `--run_name`, `--seed`
- `--num_classes` (default 100)

**Model**
- `--model_size` (e.g., `DiT-B`)
- `--patch_size` (latent patch p)
- `--hidden_size`, `--depth`, `--num_heads`, `--mlp_ratio`

**Optimizer (kvfrans-compatible defaults)**
- `--opt` (default `adam`)
- `--lr` (default `1e-4`)
- `--beta1` (default `0.9`)
- `--beta2` (default `0.99`)
- `--weight_decay` (default `0.0`)

**CFG / conditioning**
- `--class_dropout_prob` (default `0.1`)  (train-time uncond dropout; kvfrans-style) :contentReference[oaicite:13]{index=13}
- `--cfg_scale` (default `1.0`)            (sampling-time guidance; kvfrans-style) :contentReference[oaicite:14]{index=14}

**Augment**
- `--aug_flip_p` (default `0.5`)
- `--aug_jitter_eps` (default `0.01`)

**Batching / steps**
- `--global_batch`
- `--steps`

**Eval/logging**
- `--log_every`, `--eval_every`, `--sample_every`
- `--fid_every`, `--fid_n` (default `4096`)
- `--sample_steps` (Euler steps K), `--num_sample_images` (default 16)

**Checkpoint**
- `--ckpt_every`, `--ckpt_dir`
- `--best_metric` (`quick_fid_4096` or `val_loss`)
- `--hf_repo_id` (optional)

### 8.2 JEPA-specific flags
- `--lambda_jepa` (default `0.1`)
- `--ema_decay` (default `0.999`)
- `--mask_ratio` (default `0.25`)
- `--student_layer` (default `4`)
- `--teacher_layer` (default `8`)

---

## 9) Example Commands

### Baseline
```bash
python train.py \
  --mode baseline \
  --data_dir /kaggle/input/miniimagenet256-latents-arrayrecord-sdvae \
  --run_name baseline_ditb_p2 \
  --seed 42 \
  --global_batch 256 \
  --steps 200000 \
  --opt adam --lr 1e-4 --beta1 0.9 --beta2 0.99 --weight_decay 0.0 \
  --class_dropout_prob 0.1 \
  --cfg_scale 2.0 \
  --aug_flip_p 0.5 --aug_jitter_eps 0.01 \
  --fid_n 4096
JEPA
python train.py \
  --mode jepa \
  --data_dir /kaggle/input/miniimagenet256-latents-arrayrecord-sdvae \
  --run_name jepa_ditb_p2_lam01 \
  --seed 42 \
  --global_batch 256 \
  --steps 200000 \
  --opt adam --lr 1e-4 --beta1 0.9 --beta2 0.99 --weight_decay 0.0 \
  --class_dropout_prob 0.1 \
  --cfg_scale 2.0 \
  --mask_ratio 0.25 --ema_decay 0.999 --lambda_jepa 0.1 \
  --aug_flip_p 0.5 --aug_jitter_eps 0.01 \
  --fid_n 4096
10) Static-shape guarantee (XLA/TPU)

Never slice tokens by mask (no H[M==1]).

Keep tensors length N and use:

key masks in attention (context-only keys)

gating by M with where / multiplication

weighted reductions sum(M * ...) / sum(M)


## Kaggle usage (where the dataset is and how it is used)

### 1) Add the latents dataset to your notebook
In Kaggle Notebook:
- Click **Add Data**
- Search and add: **bangchi/miniimagenet256-latents-arrayrecord-sdvae**

Kaggle will mount it under:

/kaggle/input/miniimagenet256-latents-arrayrecord-sdvae/


Expected contents:

/kaggle/input/miniimagenet256-latents-arrayrecord-sdvae/
meta_train.json
meta_val.json
train/
train-00000-of-xxxxx.array_record
...
val/
val-00000-of-xxxxx.array_record
...


### 2) Training reads ONLY latents (no images needed)
The training code reads ArrayRecord shards from:
- `train/` for training steps
- `val/` for validation steps

You do **not** need to add the original MiniImageNet images or the splits dataset for training.

### 3) Pass the dataset root via `--data_dir`
Example:
```bash
python train.py \
  --mode baseline \
  --data_dir /kaggle/input/miniimagenet256-latents-arrayrecord-sdvae \
  ...
4) What one record contains

Each ArrayRecord record corresponds to one image:

label (uint16)

latent (32×32×4, NHWC, fp16 on disk)

The loader forms batches by reading multiple records and stacking them.