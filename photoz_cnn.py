"""Photo-z CNN: model + in-RAM data pipeline + training entry point.

The notebook only downloads data and calls `train(...)` from here; everything else lives in
this module so the same code is reused by `cv_outliers.py`.

    from photoz_cnn import train
    train(data_dir="/content/data", crop=64, run_name="my-run", mlflow_token="<token>")

Targets log1p(z); Huber loss; arcsinh + per-image norm; rot90/flip augmentation. Trains in RAM
via an index-based tf.data pipeline (one copy of the array — no from_tensor_slices doubling).
"""
import os
import glob
import inspect

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers as L, Model, Input
from tensorflow.keras import regularizers

import eval as ev
from eval import (SHARD, DEFAULT_TRAIN_CSV, is_train_subset, sigma_mad, outlier_rate,
                  evaluate, val_predictions, outliers_from_df)

MLFLOW_URI = "https://146-148-10-86.sslip.io"

# input preprocessing mode (set env PREPROC before importing). All read pixels in nanomaggies.
#   'zscore' (default, original): arcsinh + per-image per-channel standardization
#            — optimization-friendly but DESTROYS absolute flux & cross-band color.
#   'div'  : x / SCALE                       — linear unit rescale; keeps flux & color intact.
#   'sqrt' : sign(x) * sqrt(|x| / SCALE)     — signed sqrt; compresses range, keeps sign & color.
#   'p99'  : x / per-band p99                 — fixed per-band rescale so each band's p99 ~ 1
#            (dataset-wide constants, not per-image; balances band scales).
# 'div'/'sqrt'/'p99' do NOT subtract a per-image mean, so the flux zero-point survives.
PREPROC = os.environ.get("PREPROC", "zscore")
PREPROC_SCALE = float(os.environ.get("PREPROC_SCALE", 1000.0))
BAND_P99 = [0.231, 0.865, 1.955, 2.949, 4.012]   # u,g,r,i,z 99th-pct over the 600k v3 sample


# ============================== model ==============================
def inception(x, f1, f3r, f3, f5r, f5, fp, name, reg=None):
    b1 = L.Conv2D(f1, 1, padding='same', activation='relu', kernel_regularizer=reg, name=f'{name}_1x1')(x)
    b3 = L.Conv2D(f3r, 1, padding='same', activation='relu', kernel_regularizer=reg, name=f'{name}_3x3_reduce')(x)
    b3 = L.Conv2D(f3, 3, padding='same', activation='relu', kernel_regularizer=reg, name=f'{name}_3x3')(b3)
    b5 = L.Conv2D(f5r, 1, padding='same', activation='relu', kernel_regularizer=reg, name=f'{name}_5x5_reduce')(x)
    b5 = L.Conv2D(f5, 5, padding='same', activation='relu', kernel_regularizer=reg, name=f'{name}_5x5')(b5)
    bp = L.MaxPool2D(3, strides=1, padding='same', name=f'{name}_pool')(x)
    bp = L.Conv2D(fp, 1, padding='same', activation='relu', kernel_regularizer=reg, name=f'{name}_pool_proj')(bp)
    return L.Concatenate(axis=-1, name=f'{name}_concat')([b1, b3, b5, bp])


def build_cnn(input_shape, embed_dim=64, l2=1e-4, drop=0.4, spatial_drop=0.1):
    reg = regularizers.l2(l2) if l2 else None
    inp = Input(shape=input_shape, name='cutout')
    x = L.Conv2D(32, 3, padding='same', activation='relu', kernel_regularizer=reg, name='stem1a')(inp)
    x = L.Conv2D(32, 3, padding='same', activation='relu', kernel_regularizer=reg, name='stem1b')(x)
    x = L.BatchNormalization(name='stem1_bn')(x); x = L.MaxPool2D(name='stem1_pool')(x)
    x = L.Conv2D(64, 3, padding='same', activation='relu', kernel_regularizer=reg, name='stem2')(x)
    x = L.BatchNormalization(name='stem2_bn')(x); x = L.MaxPool2D(name='stem2_pool')(x)
    x = inception(x, 32, 32, 48, 8, 24, 24, name='inc1', reg=reg); x = L.BatchNormalization(name='inc1_bn')(x)
    x = L.SpatialDropout2D(spatial_drop, name='inc1_sdrop')(x); x = L.MaxPool2D(name='inc1_down')(x)
    x = inception(x, 64, 48, 96, 16, 48, 48, name='inc2', reg=reg); x = L.BatchNormalization(name='inc2_bn')(x)
    x = L.SpatialDropout2D(spatial_drop, name='inc2_sdrop')(x)
    x = inception(x, 64, 48, 96, 16, 48, 48, name='inc3', reg=reg); x = L.BatchNormalization(name='inc3_bn')(x)
    x = L.GlobalAveragePooling2D(name='gap')(x)
    x = L.Dense(128, activation='relu', kernel_regularizer=reg, name='dense')(x)
    x = L.Dropout(drop, name='dropout')(x)
    emb = L.Dense(embed_dim, activation='relu', kernel_regularizer=reg, name='embedding')(x)
    x = L.Dropout(drop, name='embedding_drop')(emb)
    zout = L.Dense(1, name='z')(x)
    return Model(inp, zout, name='photoz_cnn')


def build_embedder(cnn):
    return Model(cnn.input, cnn.get_layer('embedding').output, name='photoz_embedder')


# ============================== pipeline ==============================
def preprocess(x, y):                       # (B,H,W,5) nanomaggies -> network input (see PREPROC)
    x = tf.cast(x, tf.float32)
    if PREPROC == "div":                    # 1. linear unit rescale (keeps flux & color)
        return x / PREPROC_SCALE, y
    if PREPROC == "sqrt":                   # 2. /SCALE then signed sqrt (keeps sign & color)
        x = x / PREPROC_SCALE
        return tf.sign(x) * tf.sqrt(tf.abs(x)), y
    if PREPROC == "p99":                    # 3. fixed per-band /p99 (u,g,r,i,z), each band p99 ~ 1
        return x / tf.constant(BAND_P99, tf.float32), y
    # 'zscore' (default, original): arcsinh + per-image per-channel standardization
    x = tf.math.asinh(x)
    m = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
    s = tf.math.reduce_std(x, axis=[1, 2], keepdims=True) + 1e-6
    return (x - m) / s, y


def augment(x, y):
    x = tf.image.rot90(x, tf.random.uniform([], 0, 4, dtype=tf.int32))
    x = tf.image.random_flip_left_right(x); x = tf.image.random_flip_up_down(x)
    return x, y


def ram_dataset(X, y, training=False, batch=256, shuffle_buf=50000):
    """Index-based in-RAM tf.data (one copy of X). y is a 1-D float array (log1p z)."""
    n = len(X); H, W = X.shape[1], X.shape[2]
    ds = tf.data.Dataset.range(n)
    if training:
        ds = ds.shuffle(min(n, shuffle_buf), reshuffle_each_iteration=True)
    ds = ds.batch(batch)

    def gather(i):
        xb = tf.numpy_function(lambda ii: X[ii].astype('float16'), [i], tf.float16)
        yb = tf.numpy_function(lambda ii: y[ii].astype('float32'), [i], tf.float32)
        xb.set_shape([None, H, W, 5]); yb.set_shape([None])
        return xb, yb

    ds = ds.map(gather, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)


def predict_z(model, X, batch=512):
    """Predict z for an in-RAM (n,S,S,5) array (preprocess, no augment) -> z (np.float64)."""
    dummy = np.zeros(len(X), np.float32)
    ds = ram_dataset(X, dummy, training=False, batch=batch).map(lambda x, y: x)
    return np.expm1(model.predict(ds, verbose=0).ravel()).astype("float64")


# ============================== data loading ==============================
def load_catalog(data_dir):
    cat = pd.read_parquet(f'{data_dir}/catalog_v1.parquet', columns=['objid', 'redshift'])
    z_all = cat['redshift'].values
    o2i = {int(o): i for i, o in enumerate(cat['objid'].values)}
    return cat, z_all, o2i


def present_shards(data_dir):
    import re
    return set(int(re.findall(r'images_(\d+)_', p)[0]) // SHARD
              for p in glob.glob(f'{data_dir}/images_*.npy'))


def resolve_train_index(train_csv, data_dir, o2i, N=None, seed=0):
    """objids from `train_csv` -> catalog row indices, kept to downloaded shards, shuffled (seed),
    capped to N. Asserts the csv is a subset of the canonical train split (no val leakage)."""
    chk = is_train_subset(train_csv)
    assert chk['ok'], f"LEAK: {chk['n_outside_train']} objids in {train_csv} not in train split"
    idx = np.array([o2i[int(o)] for o in pd.read_csv(train_csv)['objid'].values], dtype=np.int64)
    present = present_shards(data_dir)
    idx = idx[[(i // SHARD) in present for i in idx]]
    rng = np.random.RandomState(seed); rng.shuffle(idx)
    return idx[:N] if N else idx


def load_into_ram(rows, crop, data_dir, z_all):
    """Center-crop + load these catalog rows into a float16 array in RAM (sequential per shard).
    -> X (n,crop,crop,5) float16, y = log1p(z) float32."""
    mm = ev._shard_mm(data_dir)
    o = (ev.SRC_SIZE - crop) // 2; rows = np.asarray(rows, np.int64)
    X = np.empty((len(rows), crop, crop, 5), np.float16)
    for s in np.unique(rows // SHARD):
        sel = np.where(rows // SHARD == s)[0]; rr = rows[sel] % SHARD; srt = np.argsort(rr)
        X[sel[srt]] = mm[int(s)][rr[srt]][:, o:o + crop, o:o + crop, :]
    return X, np.log1p(z_all[rows]).astype('float32')


# ============================== training plumbing ==============================
class SigmaMadCallback(tf.keras.callbacks.Callback):
    """Per-epoch sigma_MAD / outlier on a held-out set (correct global median, not batch-avg)."""
    def __init__(self, val_ds, z_true):
        super().__init__(); self.val_ds = val_ds; self.z_true = np.asarray(z_true)

    def on_epoch_end(self, epoch, logs=None):
        zp = np.expm1(self.model.predict(self.val_ds, verbose=0).ravel())
        sm, out = sigma_mad(self.z_true, zp), outlier_rate(self.z_true, zp)
        logs = logs if logs is not None else {}
        logs['val_sigma_MAD'] = sm; logs['val_outlier'] = out
        try:
            import mlflow
            if mlflow.active_run():
                mlflow.log_metrics({k: float(v) for k, v in logs.items()}, step=epoch)
        except Exception as e:
            print('  (mlflow metric log skipped:', e, ')')
        print(f'  -> val sigma_MAD={sm:.4f}  outlier={out * 100:.1f}%')


def setup_mlflow(token=None, uri=None, experiment='photoz-cnn'):
    # fall back to env vars, so setting $MLFLOW_TRACKING_TOKEN (or a Colab secret) is enough
    token = token or os.environ.get('MLFLOW_TRACKING_TOKEN')
    uri = uri or os.environ.get('MLFLOW_TRACKING_URI') or MLFLOW_URI
    if not token or 'PASTE' in token:
        print('MLflow token not set (pass mlflow_token=... or set $MLFLOW_TRACKING_TOKEN) '
              '-> training without logging')
        return False
    import mlflow, mlflow.tensorflow
    os.environ['MLFLOW_TRACKING_URI'] = uri
    os.environ['MLFLOW_TRACKING_TOKEN'] = token
    mlflow.set_experiment(experiment)
    mlflow.tensorflow.autolog(log_models=True, log_datasets=False)
    print('MLflow: logging to', uri, '| experiment:', experiment)
    return True


def make_callbacks(es_ds, zes, patience=8):
    return [
        SigmaMadCallback(es_ds, zes),
        tf.keras.callbacks.EarlyStopping('val_sigma_MAD', mode='min', patience=patience, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau('val_sigma_MAD', mode='min', factor=0.5, patience=3, min_lr=1e-5),
    ]


def compile_model(model, lr=3e-4):
    model.compile(optimizer=tf.keras.optimizers.Adam(lr),
                  loss=tf.keras.losses.Huber(delta=0.02), metrics=['mae'])
    return model


# ============================== entry point ==============================
def train(data_dir, crop=64, train_csv=DEFAULT_TRAIN_CSV, N=None, seed=0, es_size=5000,
          batch=256, lr=3e-4, epochs=50, l2=1e-4, drop=0.4, patience=8,
          run_name='cnn', mlflow_token=None, experiment='photoz-cnn', mlflow_uri=MLFLOW_URI):
    """Load data into RAM, train, evaluate on the fixed 50k val, and (if a token is given) log
    everything to MLflow — INCLUDING the outlier objids on the 50k as artifacts. Returns (metrics, model)."""
    _, z_all, o2i = load_catalog(data_dir)
    idx = resolve_train_index(train_csv, data_dir, o2i, N, seed)
    es_idx, train_idx = idx[:es_size], idx[es_size:]
    print(f'loading {len(train_idx):,} train + {len(es_idx):,} early-stop into RAM (crop={crop})...')
    Xtr, ytr = load_into_ram(train_idx, crop, data_dir, z_all)
    Xes, _ = load_into_ram(es_idx, crop, data_dir, z_all); zes = z_all[es_idx]
    print(f'train {Xtr.shape} ({Xtr.nbytes / 1e9:.1f} GB float16)')

    model = compile_model(build_cnn((crop, crop, 5), l2=l2, drop=drop), lr=lr)
    train_ds = ram_dataset(Xtr, ytr, training=True, batch=batch)
    es_ds = ram_dataset(Xes, np.log1p(zes), training=False, batch=512)
    config = dict(crop=crop, batch=batch, lr=lr, epochs=epochs, l2=l2, drop=drop, seed=seed,
                  optimizer='adam', loss='huber(0.02)', target='log1p(z)', stretch='arcsinh',
                  norm='per-image per-channel', augment='rot90+flip', arch='vgg-stem+3inception',
                  train_csv=str(train_csv), n_train=len(train_idx), params=int(model.count_params()))

    def _fit_eval():
        model.fit(train_ds, validation_data=es_ds, epochs=epochs, callbacks=make_callbacks(es_ds, zes, patience))
        valdf = val_predictions(model, data_dir=data_dir, crop=crop)   # per-object on fixed 50k
        return ev.metrics_from_df(valdf), valdf

    if setup_mlflow(mlflow_token, mlflow_uri, experiment):
        import mlflow
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params(config)
            mlflow.log_text('\n\n'.join(inspect.getsource(f) for f in
                            (preprocess, augment, ram_dataset, build_cnn)), 'recipe.py')
            metrics, valdf = _fit_eval()
            outliers = outliers_from_df(valdf)
            mlflow.log_metrics({'val50k_sigma_MAD': metrics['sigma_MAD'],
                                'val50k_outlier': metrics['outlier'], 'val50k_n': metrics['n'],
                                'val50k_n_outliers': int(len(outliers))})
            # requirement 1: persist WHICH objids are outliers (+ all per-object preds)
            mlflow.log_text(outliers.to_csv(index=False), 'outliers_val50k.csv')
            mlflow.log_text(valdf.to_csv(index=False), 'val50k_predictions.csv')
    else:
        metrics, valdf = _fit_eval()

    print('50k val:', metrics)
    return metrics, model
