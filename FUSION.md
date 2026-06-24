# Fusion photo-z — architecture & data pipeline

Predict galaxy redshift `z` by **fusing** the tabular (photometry + morphology) and image (CNN)
branches. Each branch alone tops out around σ_MAD ≈ 0.0127 (tabular) / 0.0122 (CNN); they fail on
*partly different* galaxies, so a small head over both — plus an explicit **missing-data signal** —
is the lever. This doc covers the input design, the model, and exactly how the training data is built.

---

## 1. Why fusion

- **Tabular ceiling** (catalog_v4, per-base-selected stack): σ_MAD **0.0127** on the 45k val. See
  `~/Downloads/outlier_analysis_v4/report.md`.
- **CNN ceiling** (MDN + TTA + p99, default arch): val50k σ_MAD **0.0122** (run `41f6573a`, experiment 16).
- The residual **~0.9 % "hard" core** is faint, low-S/N, photometric-noise-limited — *data-limited, not
  model-limited* for tabular. Morphology / neighbours from the image are the only remaining lever, and
  even then only for the brighter ambiguous tail.
- **Missing tabular data**: at inference some bands / morphology may be absent. The fusion head is told
  *which* features are missing (a mask) so it can lean on the image exactly when the tabular preds are
  unreliable. This is trained in via missing-feature augmentation (§4.3).

---

## 2. Fusion input — the 83-d vector

Per `(objid, missing-pattern)`: `[ 3 base preds | 16 absence mask | 64 CNN embedding ]`.

| block | dim | source |
|---|---|---|
| **base preds** | 3 | RF / HGB / MLP frozen v4 base models, run on the 16-feature vector (absent → 0) |
| **absence mask** | 16 | `1` if that tabular feature is **absent** (missing or dropped), else `0` |
| **CNN embedding** | 64 | the `embedding` (default arch) / `fuse64` (side-e2) activation of the photo-z CNN |

The 16 tabular features (fixed order):

```
dered_u dered_g dered_r dered_i dered_z          # 5 extinction-corrected mags
g-r u-g r-i i-z                                  # 4 colours (clip(-1,4))
log_expRad_r log_deVRad_r log_petroRad_r         # 3 log1p sizes
log_petroR50_r log_petroR90_r                    # 2 log1p Petrosian radii
fracDeV_r conc_r                                 # de Vaucouleurs fraction, conc_r = petroR90/petroR50
```

---

## 3. Model — `build_fusion` (in `fusion.py`)

A 3-hidden-layer MLP with an MDN head, ~38 k params:

```
Input(83)
  → BatchNorm                         # base preds (~z), 0/1 mask and CNN acts live on different scales
  → Dense(128, relu) → BN → Dropout(0.3)
  → Dense(128, relu) → BN → Dropout(0.3)
  → Dense(64,  relu) → BN → Dropout(0.3)
  → MDN head:  pi=Dense(K, softmax) · mu=Dense(K) · sig=Dense(K, exp)  → Concatenate
  → output (N, 3K)          # K=5 mixture, so (N, 15)
```

- **Target** `log1p(z)`. **Loss** `mdn_nll(K)` (negative log-likelihood of the K-Gaussian mixture,
  log-sum-exp for stability) — reused from `photoz_cnn.py`.
- **Point estimate** `mdn_point(raw)` = μ of the highest-π component, then `expm1` → z (in `eval.py`).
- `mdn=0` falls back to a plain `Dense(1)` regression head (Huber). All metrics (σ_MAD, outlier rate)
  are computed from `z_pred` vs `z_true` regardless of head, so they stay comparable to both baselines.

---

## 4. Data pipeline — building `fusion_input_v4.parquet`

`catalog_v4` row order == the `v45_data` (24 px) image-shard order, so everything aligns by row.
Because the keras env's numpy is too old to unpickle the sklearn models (and the sklearn env has no TF),
the build is **two stages**:

```
embed_v4.py  (TF / keras env)   →  emb_v4.npy        # 600k × 64 CNN embeddings, catalog order
prepare_fusion_data.py (sklearn env) + emb_v4.npy →  fusion_input_v4.parquet
```

### 4.1 Tabular branch — 3 base preds + 16 mask

- 16 features built from `catalog_v4` (SDSS `-9999` → NaN; colours = dered diffs clip(-1,4); log1p sizes;
  `conc_r = petroR90/petroR50`).
- **3 frozen base models** (`base_RF_v4.pkl`, `base_HGB_v4.pkl`, `base_MLP_v4.pkl`) trained by
  `~/Downloads/outlier_analysis_v4/train_baseline_v4.py` with **per-base hard-removal** (RF & HGB
  trained with the hard set removed, MLP on the full set — chosen by each model's all-σ_MAD).
  A `meta_lr_v4.pkl` (LinearRegression, weights RF 0.22 / HGB 0.21 / MLP 0.57) combines them for the
  standalone tabular prediction; **the fusion uses the 3 base preds directly**, not the meta.
- Each base pred = `base.predict(X16)` where **absent features are set to 0** (not median-imputed — the
  mask carries the missingness, the head learns to discount an out-of-range base pred when mask=1).

### 4.2 Image branch — 64-d CNN embedding

- Model: run `41f6573a` ("v4.5-mdn-tta-p99"), **default arch**, crop 24, **preproc `p99`** (5 channels),
  MDN K=5. The 64-d **`embedding`** Dense layer is the fusion tap. (For a side-e2 CNN the tap is `fuse64`,
  9-channel `color-feat+p99` input — `cnn_embedder` auto-detects.)
- The checkpoint shipped by MLflow is `checkpoints/latest_checkpoint.h5` (epoch 40, val σ_MAD 0.0126 —
  slightly off the run's best 0.0122). It was saved by a newer keras; load it after stripping the
  `quantization_config` keys (see `embed_v4.py` notes) → `model_clean.h5`.
- The embedding is computed **once per objid** (the image doesn't change with the missing pattern) and
  **shared** across that objid's input combinations.

### 4.3 Missing-feature augmentation

Trains the head to handle absent tabular inputs. Drops happen at the level of **11 observation units**
(the 5 bands `u g r i z`, plus `expRad deVRad petroRad petroR50 petroR90 fracDeV`). A derived feature is
**absent iff any unit it needs was dropped** — which makes composites atomically all-or-nothing
(`u-g` absent ⟺ `u` or `g` dropped; `conc_r` absent ⟺ `petroR50` or `petroR90` dropped).

Per objid (train **and** val), ~3 deduped input combinations:
1. coin flip (p=0.5) whether to keep the **full** version (no drops, mask all 0);
2. generate lossy versions to reach 3 total (full-kept → 2 lossy, else 3 lossy);
3. each lossy independently drops each unit with p=0.2 (≥1 enforced);
4. dedup identical drop-sets within the objid.

For each combination: `absent = (unit dropped) OR (genuinely NaN)` → feature value fed to the bases is
`0`, the mask bit is `1`. The CNN embedding is the objid's shared 64-vector.

### 4.4 Output — `fusion_input_v4.parquet`

`1,786,455 × 87` (≈3 combos / objid):

| columns | meaning |
|---|---|
| `objid`, `redshift` | identity + target (raw z; train on `log1p`) |
| `split` | `train` (1.65 M) / `val` (135 k) / `excluded` (539 = the 182 RA~0 image-broken × ~3) |
| `n_absent` | number of absent features (0 → full; ~17 % of rows are full) |
| `base_RF/HGB/MLP` | 3 base preds |
| `mask_<feat>` × 16 | absence mask (1 = absent) |
| `emb_0..63` | CNN embedding |

Uploaded to `gs://macrocosm-lewagon/data/fusion_input_v4.parquet`.

---

## 5. Training

`train_fusion_parquet(...)` in `fusion.py`, or the Colab notebook `train_fusion_v4.ipynb`
(self-contained, logs to MLflow experiment `fusion`):

- `X = [3 base | 16 mask | 64 emb]` (83-d), `y = log1p(redshift)`; rows by `split`.
- MDN NLL, Adam 3e-4, batch 1024, BN + dropout 0.3.
- **Early-stop on the full-feature val subset** (`n_absent == 0`) so the headline σ_MAD is directly
  comparable to the tabular (0.0127) and CNN (0.0122) baselines.
- Report val σ_MAD **by missing bucket** (`all / 0 / 1-2 / 3-5 / 6+`) to see how gracefully the model
  degrades as tabular features disappear (and how much the image carries it).

---

## 6. Files

**Repo (`CNN-Model`)**
- `fusion.py` — `build_fusion`, `train_fusion_parquet`, `tabular_features`, `tabular_base_preds`
  (v4 FrozenStack or v1 StackingRegressor), `cnn_embedder`.
- `photoz_cnn.py` — `mdn_nll`, `compile_model`, `make_callbacks` (reused).
- `eval.py` — `mdn_point`, `sigma_mad`, `outlier_rate`, `make_np_preprocess` (p99 etc.).
- `train_fusion_v4.ipynb` — Colab training notebook (MLflow).

**Local working files (`~`)**
- `embed_v4.py` — stage A (CNN embeddings → `fusion_cnn/emb_v4.npy`).
- `prepare_fusion_data.py` — stage B (features + augmentation + base preds → parquet).
- `make_fusion_notebook.py` — regenerates the notebook.
- `fusion_cnn/model_clean.h5` — the loadable CNN checkpoint (quantization_config stripped).

**GCS (`gs://macrocosm-lewagon/`)**
- `data/fusion_input_v4.parquet` — the fusion inputs.
- `models/base_{RF,HGB,MLP}_v4.pkl`, `models/meta_lr_v4.pkl`, `models/baseline_stack_v4.pkl` — tabular.

---

## 7. Reproduce

```bash
# stage A — embeddings (TF/keras env)
CUTOUT_SIZE=24 python embed_v4.py                 # -> fusion_cnn/emb_v4.npy
# stage B — features + augmentation + base preds (sklearn env)
python prepare_fusion_data.py                     # -> fusion_data/fusion_input_v4.parquet
gsutil cp fusion_data/fusion_input_v4.parquet gs://macrocosm-lewagon/data/
# train — Colab: open train_fusion_v4.ipynb, paste MLflow token, Run all
```

To rebuild with a better CNN, re-export the run's best `.keras`, regenerate `emb_v4.npy`, rerun stage B.
```
