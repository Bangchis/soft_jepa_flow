# Soft-JEPA-Flow (JAX/Flax, TPU-friendly)

Repo này huấn luyện mô hình sinh latent theo 2 chế độ:

1. `baseline`: DiT + rectified-flow/flow-matching.
2. `jepa`: baseline + JEPA loss (student/teacher EMA + cross-attention predictor).

Tài liệu này phản ánh **trạng thái code hiện tại** trong repo, đặc biệt các file:
- `train.py`
- `train_step.py`
- `sample.py`
- `infer.py`
- `fid.py`
- `vae_decode.py`
- `inception_fid.py`
- `configs.py`

## 1) Những gì đã được bù đắp so với bản cũ

- Đã bỏ pipeline pseudo-RGB cho FID:
  - FID hiện dùng decode thật bằng SD-VAE trong `vae_decode.py`.
  - FID dùng InceptionV3 vendored (`inception_fid.py`), không dùng TFHub như bản cũ.
- Đã bỏ pseudo-RGB khi log ảnh mẫu:
  - `logging_utils.log_sample_grid` decode latent thật bằng SD-VAE.
- Validation JEPA:
  - `run_validation` đã tính trực tiếp `L_gen`, `L_repa`, `L_total`.

## 2) Dữ liệu đầu vào

Dữ liệu train là latent đã mã hóa sẵn từ SD-VAE (`stabilityai/sd-vae-ft-mse`):
- shape latent: `(32, 32, 4)` (NHWC)
- mỗi record ArrayRecord:
  - `label`: `uint16` (2 bytes, little-endian)
  - `latent`: phần còn lại, `float16`, reshape thành `(32,32,4)`

Cấu trúc thư mục mong đợi:

```text
<DATA_DIR>/
  train/*.array_record
  val/*.array_record
  meta_train.json
  meta_val.json
```

`data.py` sẽ parse record và tạo one-hot label.

## 3) Cơ chế huấn luyện

### 3.1 Baseline (rectified flow)

Mỗi batch:
1. Augment latent (`flip`, `jitter`).
2. Sample `t` theo lịch (`lognormal` hoặc `uniform`).
3. Sample noise `z1 ~ N(0, I)`.
4. Tạo `z_t = (1 - t) * z0 + t * z1`.
5. Target velocity: `v_target = z1 - z0`.
6. Dự đoán `v_pred` bằng model ở `mode="baseline"`.
7. Loss: `L_gen = MSE(v_pred, v_target)`.

### 3.2 JEPA

Mỗi batch:
1. Augment latent như baseline.
2. Sample `t, s` theo lịch timestep.
3. `tau_min = min(t,s)`, `tau_max = max(t,s)`.
4. Sample token mask `M_tok` với `mask_ratio`.
5. Tạo:
   - `z_clean` cho teacher (noise theo `tau_min`).
   - `z_mixed` cho student (token-wise trộn `tau_min/tau_max`).
6. Student forward ở `mode="jepa"` trả về:
   - `v_pred`
   - `h_pred`
7. Teacher forward bằng `ema_params` ở `mode="teacher"` trả về `h_target`.
8. Loss:
   - `L_gen = MSE(v_pred, v_target)`
   - `L_JEPA = 1 - cosine(h_pred, h_target)` (chỉ tính trên token target theo `M_tok`)
   - `L_total = L_gen + lambda_jepa * L_JEPA`
9. Cập nhật optimizer cho student params, sau đó EMA update teacher.

## 4) Sampling / Infer

### 4.1 Sampling core (`sample.py`)

Sampling dùng Euler ODE trong latent space:
- khởi tạo `z ~ N(0, I)` tại `t=1`
- tích phân về `t=0` trong `sample_steps`
- CFG: chạy cond/uncond rồi trộn theo `cfg_scale`

Lưu ý quan trọng: trong sampling, model luôn chạy `mode="baseline"` để lấy head velocity (kể cả checkpoint train bằng JEPA).

### 4.2 Inference CLI độc lập (`infer.py`)

Repo đã có entrypoint inference riêng để chạy trực tiếp từ checkpoint:

```bash
python infer.py --ckpt_dir <CKPT_ROOT> --num_images 16 --decode
```

Checkpoint resolve theo thứ tự:
1. `--ckpt_path` (nếu truyền trực tiếp `step_*`)
2. `--ckpt_dir/latest`
3. `step_*` mới nhất trong `--ckpt_dir`

`infer.py` sẽ:
1. Đọc `config.json` từ checkpoint step dir.
2. Khởi tạo model đúng kiến trúc và restore strict Orbax (`params`, `ema_params`, `step`, `rng`).
3. Sample latent bằng `euler_sample(...)`.
4. Lưu output:
   - `latents.npy`
   - `class_ids.npy`
   - `meta.json`
   - nếu bật `--decode`: `grid.png` và `images/img_XXXX.png`

Mặc định inference dùng `ema_params`. Có thể dùng raw weights bằng `--use_raw_params`.

## 5) FID và Visualization (trạng thái hiện tại)

### 5.1 FID (`fid.py`)

Pipeline FID thật:
1. Lấy hoặc tính real stats (`mu`, `sigma`) từ val set.
2. Decode latent thật bằng SD-VAE (`vae_decode.decode_latents_nhwc`).
3. Resize về `299x299`, scale về `[-1,1]` cho Inception.
4. Trích xuất feature bằng InceptionV3 vendored (`inception_fid.py`) qua JAX `pmap`.
5. Tính Frechet distance.

Cache real stats:
- file: `--fid_cache_path` (mặc định `checkpoints/fid_real_stats_4096.npz`)
- batch decode: `--fid_decode_batch`
- batch inception: `--fid_inception_batch`

### 5.2 Log sample grid (`logging_utils.py`)

`log_sample_grid` decode latent thật bằng SD-VAE rồi mới log ảnh RGB lên W&B.

## 6) Validation behavior

`run_validation` là đường validation duy nhất:
- `baseline`: log `L_gen`, `L_total`
- `jepa`: log `L_gen`, `L_repa`, `L_total`

Trong đó `L_repa` là representation loss theo công thức cosine (tương đương thành phần JEPA trong validation).

## 7) Cài đặt

```bash
pip install -r requirements.txt
```

`requirements.txt` hiện đã gồm:
- JAX/Flax/Optax/Orbax/Grain
- W&B + HF Hub
- `torch` (CPU index), `diffusers`, `transformers`, `safetensors`
- `requests`, `tqdm`, `Pillow`, `scipy`

## 8) Cách chạy

### 8.1 Baseline

```bash
python train.py \
  --mode baseline \
  --data_dir /kaggle/input/miniimagenet256-latents-arrayrecord-sdvae \
  --run_name baseline_run \
  --global_batch 256 \
  --steps 200000 \
  --opt adam --lr 1e-4 --beta1 0.9 --beta2 0.99 --weight_decay 0.0 \
  --t_schedule lognormal --t_lognorm_mean -0.4 --t_lognorm_std 1.0 \
  --cfg_scale 1.0 --sample_steps 50 \
  --fid_n 4096 --fid_cache_path checkpoints/fid_real_stats_4096.npz \
  --fid_decode_batch 32 --fid_inception_batch 64
```

### 8.2 JEPA

```bash
python train.py \
  --mode jepa \
  --data_dir /kaggle/input/miniimagenet256-latents-arrayrecord-sdvae \
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

### 8.3 Inference từ checkpoint

Dùng `latest` (hoặc tự fallback step mới nhất nếu thiếu `latest`):

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

Trỏ thẳng vào một step cụ thể:

```bash
python infer.py \
  --ckpt_path /kaggle/working/checkpoints_smoke_jepa_1772736269/step_200 \
  --num_images 6 \
  --class_ids 1,2,3 \
  --decode
```

## 9) Toàn bộ CLI flags và effective defaults (argparse)

`train.py` dùng `Config.from_args()`, nên mặc định hiệu lực là từ `argparse` trong `configs.py`.

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
- `--hf_repo_id` = `""` (nếu để rỗng sẽ dùng `hf_username/hf_repo_name`)
- `--hf_username` = `"Bangchis"`
- `--hf_repo_name` = `"soft-jepa-flow"`
- `--hf_private` = `False` (bật flag `--hf_private` để tạo repo private)

## 10) Ghi chú kỹ thuật: Dataclass defaults vs CLI defaults

`Config` trong dataclass có vài giá trị khác parser defaults (ví dụ `global_batch`, `cfg_scale`, `sample_steps`, `log_every`, `weight_decay`).

Khi chạy `python train.py ...`, chương trình dùng `Config.from_args()` nên **parser defaults mới là mặc định thực tế**.

## 11) Known limitations

1. Decode SD-VAE trên CPU có thể là bottleneck khi `fid_n` lớn hoặc decode batch lớn.
2. Inception weights trong `inception_fid.py` được tải về lúc cần (lần chạy đầu cần mạng).

## 12) Tương thích và checkpoint

- Model checkpoint lưu cả `params`, `ema_params`, `opt_state`, `step`, `rng`.
- Sampling/FID trong training loop dùng `ema_params`.
- Có hỗ trợ chọn metric tốt nhất (`quick_fid_4096` hoặc `val_loss`) và upload HF tùy chọn.
- Upload HF:
  - Lần upload đầu sẽ tự tạo repo model nếu chưa tồn tại.
  - Nếu `--hf_repo_id` rỗng, repo đích sẽ là `{hf_username}/{hf_repo_name}`.
  - Mỗi lần có best mới sẽ upload theo path có timestamp UTC, ví dụ:
    `run_name/best/step_<step>_YYYYMMDD-HHMMSS-UTC`.
  - Cần `HF_TOKEN` có quyền write.
