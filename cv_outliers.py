"""3-fold out-of-fold (OOF) outlier finder for the training set.

Splits the training objects into 3 folds (the split is decided by `seed`); for each fold it
trains a fresh CNN on the OTHER two folds and predicts the held-out fold. Every training object
is therefore predicted exactly once by a model that never saw it. An object is an OUTLIER when
|Delta_z| > 0.05 in its OOF prediction; the union across the 3 folds is the full outlier set.

MLflow: experiment "oa"; one run per fold named "<seed>-<fold>" (e.g. "0-0", "0-1", "0-2").
GCS:    the outlier objids are written to {out}/outlier-<seed>.csv.

    python cv_outliers.py --seed 0 --data-dir /content/data --mlflow-token <token> \
        --out gs://macrocosm-lewagon/results/cv_outliers
    # or:  from cv_outliers import run; run(seed=0, data_dir="/content/data", mlflow_token="<token>")

Loads ALL training cutouts into RAM once (550k @ 64px ~= 22 GB -> needs a High-RAM runtime) and
gathers each fold by index (no copies). `seed` ONLY controls the fold partition.
"""
import argparse
import subprocess
import tempfile
from contextlib import nullcontext

import numpy as np
import pandas as pd
import tensorflow as tf

import eval as ev
from eval import DEFAULT_TRAIN_CSV, OUTLIER_THR
from photoz_cnn import (load_catalog, resolve_train_index, load_into_ram, build_cnn,
                        compile_model, preprocess, make_preprocess, augment, make_callbacks, setup_mlflow)

N_FOLDS = 3
EXPERIMENT = "oa"


def _subset_ds(X, y, rows, training=False, batch=256, shuffle_buf=50000, preprocess=preprocess):
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


def _predict_subset(model, X, rows, batch=512, preprocess=preprocess):
    ds = _subset_ds(X, np.zeros(len(X), np.float32), rows, training=False, batch=batch,
                    preprocess=preprocess).map(lambda x, y: x)
    return np.expm1(model.predict(ds, verbose=0).ravel()).astype('float64')


def _save_outliers_to_gcs(df, out_gcs, seed):
    """Write the outlier objids to {out_gcs}/outlier-<seed>.csv."""
    outliers = df[df['is_outlier']].copy()
    dst = f"{out_gcs.rstrip('/')}/outlier-{seed}.csv"
    with tempfile.TemporaryDirectory() as tmp:
        local = f"{tmp}/outlier-{seed}.csv"
        outliers.to_csv(local, index=False)
        subprocess.run(["gsutil", "-q", "cp", local, dst], check=True)
    print(f"saved {len(outliers):,} outliers -> {dst}")
    return dst


def run(seed, data_dir, crop=64, N=None, batch=256, lr=3e-4, epochs=50, es_size=5000,
        patience=8, train_csv=DEFAULT_TRAIN_CSV, mlflow_token=None, experiment=EXPERIMENT,
        preproc='zscore', preproc_scale=1000.0,
        out_gcs="gs://macrocosm-lewagon/results/cv_outliers"):
    pp = make_preprocess(preproc, preproc_scale)
    cat, z_all, o2i = load_catalog(data_dir)
    objid_all = cat['objid'].values
    rows = resolve_train_index(train_csv, data_dir, o2i, N=N, seed=0)   # the train objects (fixed)
    zrow = z_all[rows].astype('float64')
    oid = objid_all[rows].astype('int64')
    print(f"loading {len(rows):,} train cutouts into RAM (crop={crop})...")
    Xall, yall = load_into_ram(rows, crop, data_dir, z_all)            # yall = log1p(z), aligned with Xall
    print(f"  {Xall.shape}  ({Xall.nbytes / 1e9:.1f} GB float16)")

    use_mlflow = setup_mlflow(mlflow_token, experiment=experiment)     # autolog + experiment (default "oa")
    if use_mlflow:
        import mlflow

    # 3-fold partition decided by `seed` (positions into Xall)
    order = np.arange(len(rows)); np.random.RandomState(seed).shuffle(order)
    folds = np.array_split(order, N_FOLDS)

    zpred = np.full(len(rows), np.nan)
    foldid = np.full(len(rows), -1, int)
    for k in range(N_FOLDS):
        test_pos = folds[k]
        train_pos = np.concatenate([folds[j] for j in range(N_FOLDS) if j != k])
        es_pos, fit_pos = train_pos[:es_size], train_pos[es_size:]
        print(f"\n=== fold {k + 1}/{N_FOLDS} (run '{seed}-{k}'): train {len(fit_pos):,} | held-out {len(test_pos):,} ===")
        model = compile_model(build_cnn((crop, crop, 5)), lr=lr)
        es_ds = _subset_ds(Xall, yall, es_pos, training=False, batch=512, preprocess=pp)

        ctx = mlflow.start_run(run_name=f"{seed}-{k}") if use_mlflow else nullcontext()
        with ctx:
            if use_mlflow:
                mlflow.log_params(dict(seed=seed, fold=k, n_folds=N_FOLDS, crop=crop, batch=batch,
                                       lr=lr, epochs=epochs, preproc=preproc, preproc_scale=preproc_scale,
                                       n_train=len(fit_pos), n_test=len(test_pos)))
            model.fit(_subset_ds(Xall, yall, fit_pos, training=True, batch=batch, preprocess=pp),
                      validation_data=es_ds, epochs=epochs,
                      callbacks=make_callbacks(es_ds, zrow[es_pos], patience))
            zpred[test_pos] = _predict_subset(model, Xall, test_pos, preprocess=pp)
            foldid[test_pos] = k
            if use_mlflow:
                dzf = ev.delta_z(zrow[test_pos], zpred[test_pos])
                mlflow.log_metrics({"oof_sigma_MAD": ev.sigma_mad(zrow[test_pos], zpred[test_pos]),
                                    "oof_outlier": float(np.mean(np.abs(dzf) > OUTLIER_THR)),
                                    "oof_n_outliers": int(np.sum(np.abs(dzf) > OUTLIER_THR))})
        del model; tf.keras.backend.clear_session()

    df = pd.DataFrame({"objid": oid, "z_true": zrow, "z_pred": zpred, "fold": foldid})
    df["dz"] = ev.delta_z(df["z_true"], df["z_pred"])
    df["is_outlier"] = np.abs(df["dz"].values) > OUTLIER_THR
    n_out = int(df["is_outlier"].sum())
    print(f"\nOOF done: {n_out:,}/{len(df):,} outliers ({n_out / len(df) * 100:.1f}%), "
          f"oof sigma_MAD={ev.sigma_mad(df.z_true, df.z_pred):.4f}")
    _save_outliers_to_gcs(df, out_gcs, seed)
    return df


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="3-fold OOF outlier finder -> GCS (MLflow experiment 'oa')")
    p.add_argument("--seed", type=int, required=True, help="controls the 3-fold partition")
    p.add_argument("--data-dir", default="/content/data")
    p.add_argument("--crop", type=int, default=64)
    p.add_argument("--N", type=int, default=None, help="cap #train objects (debug); default = all")
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--mlflow-token", default=None, help="MLflow API token")
    p.add_argument("--experiment", default=EXPERIMENT, help=f"MLflow experiment name (default '{EXPERIMENT}')")
    p.add_argument("--preproc", default="zscore", choices=["zscore", "div", "sqrt", "p99"],
                   help="input preprocessing (default zscore)")
    p.add_argument("--preproc-scale", type=float, default=1000.0)
    p.add_argument("--out", default="gs://macrocosm-lewagon/results/cv_outliers")
    a = p.parse_args()
    run(seed=a.seed, data_dir=a.data_dir, crop=a.crop, N=a.N, batch=a.batch,
        lr=a.lr, epochs=a.epochs, mlflow_token=a.mlflow_token, experiment=a.experiment,
        preproc=a.preproc, preproc_scale=a.preproc_scale, out_gcs=a.out)
