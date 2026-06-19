"""3-fold out-of-fold (OOF) outlier finder for the training set.

Splits the training objects into 3 folds (the split is decided by `seed`); for each fold it
trains a fresh CNN on the OTHER two folds and predicts the held-out fold. Every training object
is therefore predicted exactly once by a model that never saw it. An object is an OUTLIER when
|Delta_z| > 0.05 in its OOF prediction; the union across the 3 folds is the full outlier set.
Results (all OOF predictions + the outlier subset + a summary) are saved to GCS.

    python cv_outliers.py --seed 0 --data-dir /content/data \
        --out gs://macrocosm-lewagon/results/cv_outliers
    # or:  from cv_outliers import run; run(seed=0, data_dir="/content/data")

Loads ALL training cutouts into RAM once (550k @ 64px ~= 22 GB -> needs a High-RAM runtime) and
gathers each fold by index (no copies). `seed` ONLY controls the fold partition.
"""
import argparse
import json
import subprocess
import tempfile

import numpy as np
import pandas as pd
import tensorflow as tf

import eval as ev
from eval import SHARD, DEFAULT_TRAIN_CSV, OUTLIER_THR
from photoz_cnn import (load_catalog, resolve_train_index, load_into_ram, build_cnn,
                        compile_model, preprocess, augment, make_callbacks)

N_FOLDS = 3


def _subset_ds(X, y, rows, training=False, batch=256, shuffle_buf=50000):
    """tf.data over a SUBSET of an in-RAM array, addressed by `rows` (positions into X) — no copy."""
    rows = np.asarray(rows, np.int64); n = len(rows); H, W = X.shape[1], X.shape[2]
    ds = tf.data.Dataset.range(n)
    if training:
        ds = ds.shuffle(min(n, shuffle_buf), reshuffle_each_iteration=True)
    ds = ds.batch(batch)

    def gather(i):
        xb = tf.numpy_function(lambda ii: X[rows[ii]].astype('float16'), [i], tf.float16)
        yb = tf.numpy_function(lambda ii: y[rows[ii]].astype('float32'), [i], tf.float32)
        xb.set_shape([None, H, W, 5]); yb.set_shape([None])
        return xb, yb

    ds = ds.map(gather, num_parallel_calls=tf.data.AUTOTUNE).map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)


def _predict_subset(model, X, rows, batch=512):
    ds = _subset_ds(X, np.zeros(len(X), np.float32), rows, training=False, batch=batch).map(lambda x, y: x)
    return np.expm1(model.predict(ds, verbose=0).ravel()).astype('float64')


def _save_to_gcs(df, out_gcs, seed):
    """Write OOF preds + outlier subset + summary under {out_gcs}/seed_{seed}/."""
    base = f"{out_gcs.rstrip('/')}/seed_{seed}"
    outliers = df[df['is_outlier']].copy()
    summary = {
        "seed": int(seed), "n_train": int(len(df)), "n_folds": N_FOLDS,
        "n_outliers": int(outliers.shape[0]),
        "outlier_rate": round(float(outliers.shape[0] / len(df)), 4),
        "oof_sigma_MAD": round(ev.sigma_mad(df['z_true'], df['z_pred']), 5),
        "per_fold_outliers": df.groupby('fold')['is_outlier'].sum().astype(int).to_dict(),
    }
    with tempfile.TemporaryDirectory() as tmp:
        df.to_parquet(f"{tmp}/oof_predictions.parquet", index=False)
        outliers.to_csv(f"{tmp}/outliers.csv", index=False)
        json.dump(summary, open(f"{tmp}/summary.json", "w"), indent=2)
        for fn in ("oof_predictions.parquet", "outliers.csv", "summary.json"):
            subprocess.run(["gsutil", "-q", "cp", f"{tmp}/{fn}", f"{base}/{fn}"], check=True)
    print(f"saved -> {base}/  ({summary['n_outliers']:,} outliers / {summary['n_train']:,})")
    return summary, base


def run(seed, data_dir, crop=64, N=None, batch=256, lr=3e-4, epochs=50, es_size=5000,
        patience=8, train_csv=DEFAULT_TRAIN_CSV,
        out_gcs="gs://macrocosm-lewagon/results/cv_outliers"):
    cat, z_all, o2i = load_catalog(data_dir)
    objid_all = cat['objid'].values
    rows = resolve_train_index(train_csv, data_dir, o2i, N=N, seed=0)   # the train objects (fixed)
    zrow = z_all[rows].astype('float64')
    oid = objid_all[rows].astype('int64')
    print(f"loading {len(rows):,} train cutouts into RAM (crop={crop})...")
    Xall, yall = load_into_ram(rows, crop, data_dir, z_all)            # yall = log1p(z), aligned with Xall
    print(f"  {Xall.shape}  ({Xall.nbytes / 1e9:.1f} GB float16)")

    # 3-fold partition decided by `seed` (positions into Xall)
    order = np.arange(len(rows)); np.random.RandomState(seed).shuffle(order)
    folds = np.array_split(order, N_FOLDS)

    zpred = np.full(len(rows), np.nan)
    foldid = np.full(len(rows), -1, int)
    for k in range(N_FOLDS):
        test_pos = folds[k]
        train_pos = np.concatenate([folds[j] for j in range(N_FOLDS) if j != k])
        es_pos, fit_pos = train_pos[:es_size], train_pos[es_size:]
        print(f"\n=== fold {k + 1}/{N_FOLDS}: train {len(fit_pos):,} | held-out {len(test_pos):,} ===")
        model = compile_model(build_cnn((crop, crop, 5)), lr=lr)
        model.fit(_subset_ds(Xall, yall, fit_pos, training=True, batch=batch),
                  validation_data=_subset_ds(Xall, yall, es_pos, training=False, batch=512),
                  epochs=epochs,
                  callbacks=make_callbacks(_subset_ds(Xall, yall, es_pos, training=False, batch=512),
                                           zrow[es_pos], patience))
        zpred[test_pos] = _predict_subset(model, Xall, test_pos)
        foldid[test_pos] = k
        del model; tf.keras.backend.clear_session()

    df = pd.DataFrame({"objid": oid, "z_true": zrow, "z_pred": zpred, "fold": foldid})
    df["dz"] = ev.delta_z(df["z_true"], df["z_pred"])
    df["is_outlier"] = np.abs(df["dz"].values) > OUTLIER_THR
    summary, base = _save_to_gcs(df, out_gcs, seed)
    print("summary:", summary)
    return df


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="3-fold OOF outlier finder -> GCS")
    p.add_argument("--seed", type=int, required=True, help="controls the 3-fold partition")
    p.add_argument("--data-dir", default="/content/data")
    p.add_argument("--crop", type=int, default=64)
    p.add_argument("--N", type=int, default=None, help="cap #train objects (debug); default = all")
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--out", default="gs://macrocosm-lewagon/results/cv_outliers")
    a = p.parse_args()
    run(seed=a.seed, data_dir=a.data_dir, crop=a.crop, N=a.N, batch=a.batch,
        lr=a.lr, epochs=a.epochs, out_gcs=a.out)
