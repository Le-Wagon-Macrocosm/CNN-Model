"""Evaluate a photo-z model on the FIXED 50k validation set (splits/val_objids.csv).

Same metric everywhere — Delta_z = (z_pred - z_true) / (1 + z_true):
  sigma_MAD = 1.4826 * median(|Delta_z - median(Delta_z)|)   (pooled over all 50k)
  outlier   = mean(|Delta_z| > 0.05)

    from eval import evaluate, val_predictions
    print(evaluate(model, data_dir="/content/data"))        # -> {'n':..., 'sigma_MAD':..., ...}
    df = val_predictions(model, data_dir="/content/data")   # per-object: objid,z_true,z_pred,dz

`model` predicts log1p(z) by default (target="log1p"); pass target="z" for a direct-z head.
Reads val images straight from the memory-mapped shards (only a batch in RAM). Val objids whose
image shard isn't downloaded are skipped, with a warning, so partial data still gives a number.
"""
import os
import glob
import re

import numpy as np
import pandas as pd

SHARD = 6000
# on-disk cutout edge length: sample_v1 = 64, registered+cropped sample_v3 = 24.
# Set env CUTOUT_SIZE=24 when training on v3.
SRC_SIZE = int(os.environ.get("CUTOUT_SIZE", 64))
OUTLIER_THR = 0.05
# u,g,r,i,z 99th-pct over the v4 train set, per crop size (smaller crop = more central = higher p99).
# Selected by CUTOUT_SIZE, so it MUST match the training crop for preproc='p99' to normalize correctly.
_BAND_P99_BY_SIZE = {
    16: [0.314, 1.297, 2.980, 4.523, 6.156],   # v4.7
    24: [0.235, 0.906, 2.059, 3.113, 4.242],   # v4.5
    32: [0.192, 0.687, 1.535, 2.305, 3.141],   # v4.6
}
BAND_P99 = _BAND_P99_BY_SIZE.get(SRC_SIZE, _BAND_P99_BY_SIZE[24])
BAND_SKY_SIGMA = [0.051, 0.042, 0.085, 0.123, 0.218] # u,g,r,i,z sky noise (sigma-clip) over 550k train (v3 — recompute for v4 if using color-feat+p99)
BAND_COLOR_SCALE = [1.400, 0.748, 0.929, 2.817]      # |p99| of asinh colours z-i,i-r,r-g,g-u (550k train v3 — recompute for v4 if using color-feat+p99)
PREPROC_CHANNELS = {"color-feat+p99": 9}             # modes that change the channel count (else 5)


def preproc_channels(mode):
    """Number of input channels the network sees for a given preprocessing mode."""
    return PREPROC_CHANNELS.get(mode, 5)
_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_VAL_CSV = os.path.join(_HERE, "splits", "val_objids.csv")
DEFAULT_TRAIN_CSV = os.path.join(_HERE, "splits", "train_objids.csv")


def is_train_subset(train_csv, base_csv=DEFAULT_TRAIN_CSV):
    """Check that every objid in `train_csv` is inside the canonical train split
    (splits/train_objids.csv) — i.e. NONE leak into the held-out 50k val. Returns
    {ok, n, n_outside_train, sample_outside}. Always call this before training on a custom subset."""
    given = set(int(x) for x in pd.read_csv(train_csv)["objid"].values)
    base = set(int(x) for x in pd.read_csv(base_csv)["objid"].values)
    outside = given - base
    return {"ok": len(outside) == 0, "n": len(given),
            "n_outside_train": len(outside), "sample_outside": list(outside)[:5]}


def make_np_preprocess(mode="zscore", scale=1000.0):
    """numpy preprocessing for eval — must mirror photoz_cnn.make_preprocess. Modes:
    'zscore' arcsinh + per-image per-channel std | 'div' x/scale | 'sqrt' sign*sqrt(|x|/scale)
    | 'p99' x / per-band p99."""
    p99 = np.asarray(BAND_P99, "float32")
    sig = np.asarray(BAND_SKY_SIGMA, "float32")
    cscale = np.asarray(BAND_COLOR_SCALE, "float32")

    def fn(arr):
        a = np.asarray(arr, dtype="float32")
        if mode == "div":
            return a / scale
        if mode == "sqrt":
            a = a / scale
            return np.sign(a) * np.sqrt(np.abs(a))
        if mode == "p99":
            return a / p99
        if mode == "color-feat+p99":           # 5 p99 bands + 4 asinh colours (z-i,i-r,r-g,g-u) -> 9 ch
            am = np.arcsinh(a / sig)           # asinh-mag per band (handles negatives)
            colors = np.stack([am[..., 4] - am[..., 3], am[..., 3] - am[..., 2],
                               am[..., 2] - am[..., 1], am[..., 1] - am[..., 0]], axis=-1)
            return np.concatenate([a / p99, colors / cscale], axis=-1)
        a = np.arcsinh(a)                      # 'zscore' (original)
        m = a.mean(axis=(1, 2), keepdims=True)
        s = a.std(axis=(1, 2), keepdims=True) + 1e-6
        return (a - m) / s
    return fn


default_preprocess = make_np_preprocess()      # 'zscore' — matches the original training pipeline


# ---- shared metric helpers (Delta_z based) ----
def delta_z(z_true, z_pred):
    z_true, z_pred = np.asarray(z_true, "float64"), np.asarray(z_pred, "float64")
    return (z_pred - z_true) / (1 + z_true)


def sigma_mad(z_true, z_pred):
    d = delta_z(z_true, z_pred)
    return float(1.4826 * np.median(np.abs(d - np.median(d))))


def outlier_rate(z_true, z_pred, thr=OUTLIER_THR):
    return float(np.mean(np.abs(delta_z(z_true, z_pred)) > thr))


def metrics_from_df(df, thr=OUTLIER_THR):
    """Summary metrics from a predictions DataFrame with columns z_true, z_pred (+ dz)."""
    dz = df["dz"].values if "dz" in df else delta_z(df["z_true"], df["z_pred"])
    return {
        "n": int(len(df)),
        "sigma_MAD": round(float(1.4826 * np.median(np.abs(dz - np.median(dz)))), 5),
        "outlier": round(float(np.mean(np.abs(dz) > thr)), 4),
        "bias": round(float(np.median(dz)), 5),
        "MAE": round(float(np.mean(np.abs(df["z_pred"].values - df["z_true"].values))), 5),
    }


def _shard_mm(data_dir):
    paths = sorted(glob.glob(f"{data_dir}/images_*.npy"),
                   key=lambda p: int(re.findall(r"images_(\d+)_", p)[0]))
    return {int(re.findall(r"images_(\d+)_", p)[0]) // SHARD: np.load(p, mmap_mode="r") for p in paths}


def mdn_point(raw):
    """log1p(z)-space point estimate from a model's raw output, auto-detecting the head:
    regression (N,1) -> the raveled output; MDN (N, 3*K) -> mean of the highest-weight Gaussian.
    Callers apply expm1 themselves (so TTA can average in log1p space)."""
    raw = np.asarray(raw)
    if raw.ndim == 2 and raw.shape[1] > 1 and raw.shape[1] % 3 == 0:   # MDN: [pi(K), mu(K), sigma(K)]
        K = raw.shape[1] // 3
        pi, mu = raw[:, :K], raw[:, K:2 * K]
        return mu[np.arange(len(mu)), pi.argmax(1)]
    return raw.ravel()


def tta_predict(model, imgs, preprocess, target="log1p"):
    """Test-time augmentation: average the model output over the 8 dihedral (D4) views
    (rot90 x4, each with/without horizontal flip) in the model's output space, then map to z.
    `imgs` are RAW cutouts (n,H,W,5); the model was trained invariant to these transforms."""
    acc = np.zeros(len(imgs), dtype="float64")
    for k in range(4):
        r = np.rot90(imgs, k, axes=(1, 2))
        for v in (r, np.flip(r, axis=2)):
            acc += mdn_point(model.predict(preprocess(np.ascontiguousarray(v)), verbose=0))
    pred = acc / 8.0
    return np.expm1(pred) if target == "log1p" else pred


def val_predictions(model, data_dir, val_csv=DEFAULT_VAL_CSV, catalog_path=None,
                    crop=64, batch=512, preprocess=default_preprocess, target="log1p", tta=False):
    """Per-object predictions on the val set.
    -> DataFrame[objid, z_true, z_pred, dz]  (rows whose image shard is missing are skipped).
    tta=True averages over the 8 D4 views (8x slower, usually a small sigma_MAD gain)."""
    catalog_path = catalog_path or os.path.join(data_dir, "catalog_v1.parquet")
    cat = pd.read_parquet(catalog_path, columns=["objid", "redshift"])
    objid_all = cat["objid"].values
    z = cat["redshift"].values
    o2i = {int(o): i for i, o in enumerate(objid_all)}

    val_obj = pd.read_csv(val_csv)["objid"].values
    val_idx = np.array([o2i[int(o)] for o in val_obj], dtype=np.int64)

    mm = _shard_mm(data_dir)
    have = np.array([(int(i) // SHARD) in mm for i in val_idx])
    if not have.all():
        print(f"WARNING: {(~have).sum()}/{len(val_idx)} val images missing "
              f"(shards not downloaded) -> evaluating on {int(have.sum())}.")
    val_obj, val_idx = val_obj[have], val_idx[have]
    if len(val_idx) == 0:
        raise RuntimeError("no val images available in data_dir — download the shards first")

    zt = z[val_idx].astype("float64")
    off = (SRC_SIZE - crop) // 2
    zp = np.empty(len(val_idx), dtype="float64")
    for k in range(0, len(val_idx), batch):
        bi = val_idx[k:k + batch]
        imgs = np.stack([np.asarray(mm[int(i) // SHARD][int(i) % SHARD][off:off + crop, off:off + crop, :])
                         for i in bi])
        if tta:
            zp[k:k + batch] = tta_predict(model, imgs, preprocess, target=target)
        else:
            p = mdn_point(model.predict(preprocess(imgs), verbose=0))
            zp[k:k + batch] = np.expm1(p) if target == "log1p" else p

    df = pd.DataFrame({"objid": val_obj.astype("int64"), "z_true": zt, "z_pred": zp})
    df["dz"] = delta_z(df["z_true"], df["z_pred"])
    return df


def evaluate(model, data_dir, val_csv=DEFAULT_VAL_CSV, catalog_path=None,
             crop=64, batch=512, preprocess=default_preprocess, target="log1p", tta=False):
    """Return dict(n, sigma_MAD, outlier, bias, MAE) for `model` on the val set."""
    df = val_predictions(model, data_dir, val_csv, catalog_path, crop, batch, preprocess, target, tta)
    return metrics_from_df(df)


def outliers_from_df(df, thr=OUTLIER_THR):
    """Subset of a predictions DataFrame whose |dz| > thr (the outlier objects)."""
    return df[np.abs(df["dz"].values) > thr].copy()
