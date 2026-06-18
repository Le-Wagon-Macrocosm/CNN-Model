"""Evaluate a photo-z model on the FIXED 50k validation set (splits/val_objids.csv).

Same metric everywhere — Delta_z = (z_pred - z_true) / (1 + z_true):
  sigma_MAD = 1.4826 * median(|Delta_z - median(Delta_z)|)   (pooled over all 50k)
  outlier   = mean(|Delta_z| > 0.05)

    from eval import evaluate
    print(evaluate(model, data_dir="/content/data"))     # -> {'n':..., 'sigma_MAD':..., ...}

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


def default_preprocess(arr):
    """arcsinh stretch + per-image per-channel normalize — matches the training pipeline."""
    a = np.asarray(arr, dtype="float32")
    a = np.arcsinh(a)
    m = a.mean(axis=(1, 2), keepdims=True)
    s = a.std(axis=(1, 2), keepdims=True) + 1e-6
    return (a - m) / s


def evaluate(model, data_dir, val_csv=DEFAULT_VAL_CSV, catalog_path=None,
             crop=64, batch=512, preprocess=default_preprocess, target="log1p"):
    """Return dict(n, sigma_MAD, outlier, bias, MAE) for `model` on the 50k val set.

    data_dir must hold catalog_v1.parquet + the images_*.npy shards covering the val rows.
    If `crop` < 64 the cutouts are center-cropped to match the model's input."""
    catalog_path = catalog_path or os.path.join(data_dir, "catalog_v1.parquet")
    objid = pd.read_parquet(catalog_path, columns=["objid"])["objid"].values
    z = pd.read_parquet(catalog_path, columns=["redshift"])["redshift"].values
    o2i = {int(o): i for i, o in enumerate(objid)}

    val_obj = pd.read_csv(val_csv)["objid"].values
    val_idx = np.array([o2i[int(o)] for o in val_obj], dtype=np.int64)

    paths = sorted(glob.glob(f"{data_dir}/images_*.npy"),
                   key=lambda p: int(re.findall(r"images_(\d+)_", p)[0]))
    mm = {int(re.findall(r"images_(\d+)_", p)[0]) // SHARD: np.load(p, mmap_mode="r") for p in paths}

    have = np.array([(int(i) // SHARD) in mm for i in val_idx])
    if not have.all():
        print(f"WARNING: {(~have).sum()}/{len(val_idx)} val images missing "
              f"(shards not downloaded) -> evaluating on {int(have.sum())}.")
    val_idx = val_idx[have]
    if len(val_idx) == 0:
        raise RuntimeError("no val images available in data_dir — download the shards first")

    zt = z[val_idx].astype("float32")
    off = (64 - crop) // 2
    preds = np.empty(len(val_idx), dtype="float32")
    for k in range(0, len(val_idx), batch):
        bi = val_idx[k:k + batch]
        imgs = np.stack([np.asarray(mm[int(i) // SHARD][int(i) % SHARD][off:off + crop, off:off + crop, :])
                         for i in bi])
        preds[k:k + batch] = model.predict(preprocess(imgs), verbose=0).ravel()

    zp = np.expm1(preds) if target == "log1p" else preds
    dz = (zp - zt) / (1 + zt)
    smad = 1.4826 * np.median(np.abs(dz - np.median(dz)))
    return {
        "n": int(len(val_idx)),
        "sigma_MAD": round(float(smad), 5),
        "outlier": round(float(np.mean(np.abs(dz) > 0.05)), 4),
        "bias": round(float(np.median(dz)), 5),
        "MAE": round(float(np.mean(np.abs(zp - zt))), 5),
    }
