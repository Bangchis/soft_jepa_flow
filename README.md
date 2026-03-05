# Standard DiT vs Soft-Masked Flow-JEPA (Cross-Attention) — JAX/Flax (TPU-ready)

This repo implements two training modes on **precomputed SD-VAE latents** (MiniImageNet, 100 classes):

1) **Baseline:** Standard DiT trained with **rectified flow / flow matching** objective.
2) **Proposed:** **Soft-Masked Flow-JEPA** with an **asymmetric Student–Teacher** setup and a **1-layer Cross-Attention predictor** trained with a representation (JEPA) loss on masked target tokens.

The project is designed to be **XLA/TPU friendly**:
- **No dynamic shapes** (no `H[M==1]` style boolean indexing).
- All masking is done via **attention masks** and `where`/gating with **static tensor shapes**.

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
- `latent`: raw bytes of latent in **NHWC** layout (32,32,4), typically float16 on disk

Directory layout (Kaggle dataset):

<latents_dataset>/
meta_train.json
meta_val.json
train/.array_record
val/.array_record


> Training does NOT require the original image dataset nor the split dataset.  
> Only the latents dataset is needed.

---

## 1. Notation & Shapes

Let:
- Batch size: **B**
- Latent tensor: **Z0 ∈ R^{B×H×W×C}**, with **H=W=32**, **C=4**
- Patch size (latent-patch): **p**
  - Patchify maps latent grid to tokens:
    - token grid size: (H/p)×(W/p)
    - number of tokens: **N = (H/p)·(W/p)**
- Token dimension: **D** (DiT hidden width)
- Hidden tokens: **X ∈ R^{B×N×D}**

For **DiT-Base Patch-2** on 32×32 latents:
- p=2 → token grid 16×16 → **N=256**
- Attention cost scales ~O(N²), still feasible.

Conditioning:
- Classes: 100
- One-hot: **y ∈ {0,1}^{100}**
- Class embedding: **e_y = MLP(y) ∈ R^{D}**

Timestep embedding:
- **t ∈ [0,1]**
- **e_t = MLP(t) ∈ R^{D}**

---

## 2. Train-time Latent Augmentation (enabled in both modes)

Applied only during training (NOT validation):

1) Random horizontal flip with p=0.5 on latent grid:
- `Z0 ← flip(Z0)` along width axis

2) Latent jitter noise:
- `Z0 ← Z0 + ε·N(0,I)` with **ε = 0.02**

These operations preserve static shapes and are cheap.

---

## 3. Mode A — Baseline: Standard DiT + Rectified Flow

### 3.1 Forward diffusion (rectified flow path)
Sample:
- `t ~ Uniform(0,1)`
- `Z1 ~ N(0, I)` (same shape as Z0)

Interpolate:
\[
Z_t = (1-t)\,Z_0 + t\,Z_1
\]

Flow target (velocity):
\[
v = Z_1 - Z_0
\]

Model prediction:
\[
v_{\text{pred}} = f_\theta(Z_t, t, y)
\]

### 3.2 Loss
\[
\mathcal{L}_{gen} = \mathrm{MSE}(v_{\text{pred}}, v)
\]

### 3.3 Training loop (Baseline)
Per step:
1) Load batch latents Z0 + labels y
2) Apply latent augment (flip + jitter)
3) Sample t, Z1; build Zt
4) Forward DiT → v_pred
5) Compute L_gen
6) Update θ with optimizer
7) Periodic: val loss, sampling, FID@4096, checkpoint, W&B logs

---

## 4. Mode B — Soft-Masked Flow-JEPA (Cross-Attention Predictor)

This mode extends baseline with:
- Student network (trainable)
- Teacher network (EMA, stop-grad)
- Dual-timestep soft masking (token-wise noise levels)
- A 1-layer cross-attention predictor trained to match teacher representations on target tokens

### 4.1 Student–Teacher setup
- Student: parameters θ
- Teacher: parameters ϕ updated by EMA:
\[
\phi \leftarrow \gamma\phi + (1-\gamma)\theta,\quad \gamma=0.999
\]
Teacher is used with `stop_gradient` (no backprop through teacher).

### 4.2 Dual-timestep soft masking (token-wise mixing)
Sample two timesteps:
- `t, s ~ Uniform(0,1)`
- `τ_min = min(t,s)`, `τ_max = max(t,s)`

Sample a binary token mask **M ∈ {0,1}^{B×N}** with mask ratio 0.25:
- M=1 indicates **Target tokens** (heavier noise)
- (1-M)=1 indicates **Context tokens** (cleaner)

Define token-wise timestep (soft mask):
\[
\tau_{\text{mixed}} = \tau_{max}\cdot M + \tau_{min}\cdot (1-M)
\]
Broadcast to match token/latent shape.

Generate noises:
- `Z1 ~ N(0,I)` (same shape as Z0)

Teacher input (all tokens at τ_min):
\[
Z_{\text{clean}} = (1-\tau_{min})Z_0 + \tau_{min}Z_1
\]

Student input (token-wise τ_mixed):
\[
Z_{\text{mixed}} = (1-\tau_{\text{mixed}})Z_0 + \tau_{\text{mixed}}Z_1
\]

> Implementation note: the model sees latents, then patchifies into tokens; mask M is defined on tokens and must align with patch grid.

### 4.3 Generation loss (same as baseline, but on Z_mixed)
We still train a flow field:
- Use a scalar timestep for conditioning (e.g., τ_min or t) consistently across runs.
- Target velocity remains:
\[
v = Z_1 - Z_0
\]
Prediction:
\[
v_{\text{pred}} = f_\theta(Z_{\text{mixed}}, \tau, y)
\]
Loss:
\[
\mathcal{L}_{gen} = \mathrm{MSE}(v_{\text{pred}}, v)
\]

### 4.4 JEPA representation loss via 1-layer Cross-Attention

We tap intermediate hidden states:
- Student hidden at layer **l = 4**: \(h_{stu} \in \mathbb{R}^{B\times N\times D}\)
- Teacher hidden at layer **k = 8**: \(h_{tea} \in \mathbb{R}^{B\times N\times D}\)

Teacher head (MLP):
\[
h_{\text{target}} = \mathrm{MLP}_{tea}(h_{tea})
\]
with stop-grad.

#### Cross-Attention predictor (static shape, no indexing)
We keep full length N (static), but restrict attention via masks.

Let positional embedding be \(P \in \mathbb{R}^{1\times N\times D}\).

Construct:
- Query uses **Target tokens** and **adds position**:
\[
Q = (h_{stu} + P)\odot M
\]
- Key uses **Context tokens**, **adds position**, but still stored full-N:
\[
K = h_{stu} + P
\]
- Value uses **Context tokens**, **no position**:
\[
V = h_{stu}
\]

Key mask (allow only context keys):
\[
\text{key\_mask} = (1 - M) \in \{0,1\}^{B\times N}
\]
Broadcast to attention mask:
\[
\text{attn\_mask} \in \{0,1\}^{B\times 1\times 1\times N}
\]
so every query position can only attend to **context** keys.

1-layer cross-attention:
\[
h_{\text{pred}} = \mathrm{MHA}(Q, K, V; \text{attn\_mask})
\]
Finally gate output to target tokens only:
\[
h_{\text{pred}} \leftarrow h_{\text{pred}}\odot M
\]

#### JEPA cosine loss (target tokens only, weighted mean)
Cosine similarity per token:
\[
c_i = \cos(h_{\text{pred},i}, \mathrm{stopgrad}(h_{\text{target},i}))
\]
Aggregate over target tokens:
\[
\mathcal{L}_{JEPA}
= 1 - \frac{\sum_i M_i\,c_i}{\sum_i M_i + \epsilon}
\]

### 4.5 Total loss
\[
\mathcal{L}_{total} = \mathcal{L}_{gen} + \lambda \mathcal{L}_{JEPA}
\]
with **λ = 0.1**.

### 4.6 Training loop (JEPA)
Per step:
1) Load Z0 + label → one-hot y
2) Apply latent augment (flip + jitter)
3) Sample t,s and mask M; build Z_clean (teacher) and Z_mixed (student)
4) Student forward on Z_mixed → v_pred + hidden h_stu@l=4
5) Teacher forward on Z_clean (stop-grad) → hidden h_tea@k=8 → h_target via TeacherHead
6) Cross-attention predictor → h_pred
7) Compute L_gen and L_JEPA, then L_total
8) Update student params θ
9) EMA update teacher params ϕ
10) Periodic eval/logging/checkpoints

---

## 5. Inference (Sampling)

We generate in latent space, then decode with the same VAE.

### 5.1 Euler ODE sampler (rectified flow)
We start from standard Gaussian:
\[
Z_1 \sim \mathcal{N}(0,I)
\]
Integrate from t=1 → 0 with K steps:
- Step size: \(\Delta t = -1/K\)
- At each step:
\[
v_{\text{pred}} = f_\theta(Z_t, t, y)
\]
\[
Z_{t+\Delta t} = Z_t + v_{\text{pred}} \cdot \Delta t
\]

At the end:
- Obtain \(Z_0\) (approx)
- Decode image:
\[
x = \mathrm{VAE.decode}(Z_0 / \text{scaling\_factor})
\]
and map from [-1,1] → [0,1].

> For fair comparison, both modes use the same sampler and step count.

---

## 6. Evaluation & Logging

### 6.1 Loss curves
Logged to W&B:
- train/val: `L_gen`, `L_JEPA` (JEPA mode), `L_total`
- optimizer stats: lr, grad_norm

### 6.2 Samples
Every `sample_every` steps:
- generate a fixed grid of 16 images (fixed noise seed)
- log to W&B as an image grid

### 6.3 FID@4096
Every `fid_every` steps:
- generate **4096** samples from the model
- compute FID against **4096** real images from val subset (fixed indices)
- log `quick_fid_4096`

---

## 7. Checkpointing (Best + Last)
We store:
- `last` checkpoint periodically
- `best` checkpoint when validation metric improves (prefer `quick_fid_4096`, fallback to `val/L_total`)

Teacher EMA params are saved for JEPA mode.

---

## 8. Reproducibility Notes
- Latents are generated via posterior **sampling** (not mean), and stored in fp16.
- Exact values may differ slightly across runs/hardware due to sampling and fp16.
- Static-shape masking is enforced: no dynamic indexing by masks.

---

## 9. Quick start (high-level)
1) Add the Kaggle latents dataset as input.
2) Run training in one mode:
   - Baseline: `train_baseline.py`
   - JEPA: `train_jepa.py`
3) Compare:
   - `quick_fid_4096`, sample grids, and loss curves.