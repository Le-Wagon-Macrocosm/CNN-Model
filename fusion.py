"""Fusion photo-z model: a small MLP + MDN head over [tabular base preds | presence mask | CNN fuse64].

Input vector (83-d):
  - 3  tabular base predictions  : RF / HGB / MLP outputs of the StackingRegressor (baseline_stack.pkl)
  - 16 presence mask (0/1)        : is each of the 16 tabular features present (not NaN) for this object
  - 64 CNN fuse64                 : the side-e2 CNN's `fuse64` activation (falls back to `embedding`)

Target log1p(z); MDN (K-Gaussian mixture) output, same NLL loss / point estimate as the CNN
(reused from photoz_cnn / eval). The presence mask lets the head down-weight the tabular preds when
photometry/morphology is missing and lean on the image — which is exactly where the tabular baseline
fails (the ~900 "hard" colour-degenerate objects).

    from fusion import train_fusion
    train_fusion(data_dir="/content/data", cnn_path="cnn.keras",
                 stack_path="baseline_stack.pkl", mlflow_token="<token>")
"""
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers as L, Model, Input
from tensorflow.keras import regularizers

import eval as ev
from photoz_cnn import mdn_nll, compile_model, load_into_ram, make_callbacks, setup_mlflow

# the 16 tabular features, in the exact order baseline_stack.pkl expects
TAB_FEATURES = ["dered_u", "dered_g", "dered_r", "dered_i", "dered_z",
                "g-r", "u-g", "r-i", "i-z",
                "log_expRad_r", "log_deVRad_r", "log_petroRad_r", "log_petroR50_r", "log_petroR90_r",
                "fracDeV_r", "conc_r"]
N_BASE, N_MASK, N_EMB = 3, 16, 64
SENTINEL = -100.0          # SDSS -9999 unmeasured -> values <= -100 are NaN
# catalog_v4 columns needed to build the tabular features + redshift (and align to the image shards)
CAT_COLS = ["objid", "redshift", "dered_u", "dered_g", "dered_r", "dered_i", "dered_z",
            "expRad_r", "deVRad_r", "petroRad_r", "petroR50_r", "petroR90_r", "fracDeV_r"]


def load_catalog_v4(data_dir, catalog="catalog_v4.parquet"):
    """Single alignment source for fusion: catalog_v4 carries redshift, the 16-feature source columns,
    AND its row order matches the v4.x image shards. Returns (cat, z_all, o2i={objid: row})."""
    cat = pd.read_parquet(f"{data_dir}/{catalog}", columns=CAT_COLS)
    z_all = cat["redshift"].to_numpy()
    o2i = {int(o): i for i, o in enumerate(cat["objid"].to_numpy())}
    return cat, z_all, o2i


# ============================== model ==============================
def build_fusion(n_in=N_BASE + N_MASK + N_EMB, hidden=(128, 128, 64), mdn=5, l2=1e-4, drop=0.3):
    """3-hidden-layer MLP over the 83-d fusion vector, MDN head (or Dense-1 if mdn=0)."""
    reg = regularizers.l2(l2) if l2 else None
    inp = Input((n_in,), name='fusion_in')                  # [3 base | 16 mask | 64 fuse64]
    x = L.BatchNormalization(name='in_bn')(inp)             # base preds (~z), fuse64 acts & 0/1 mask differ in scale
    for i, h in enumerate(hidden):
        x = L.Dense(h, activation='relu', kernel_regularizer=reg, name=f'fc{i + 1}')(x)
        x = L.BatchNormalization(name=f'fc{i + 1}_bn')(x)
        x = L.Dropout(drop, name=f'fc{i + 1}_drop')(x)
    if mdn:
        pi = L.Dense(mdn, activation='softmax', name='mdn_pi')(x)
        mu = L.Dense(mdn, name='mdn_mu')(x)
        sig = L.Dense(mdn, activation='exponential', name='mdn_sigma')(x)
        out = L.Concatenate(name='z')([pi, mu, sig])        # (3*mdn,)
    else:
        out = L.Dense(1, name='z')(x)
    return Model(inp, out, name=f'fusion-mdn{mdn}' if mdn else 'fusion')


# ============================== tabular side ==============================
def tabular_features(cat):
    """Build the 16 features + presence mask from catalog_v4 columns, mirroring the baseline recipe:
      -9999 sentinel -> NaN; colours = dered diffs clip(-1,4); log1p on radii (clip>=0);
      conc_r = petroR90_r/petroR50_r.
    Returns (X16 float32 with NaN where unmeasured, mask uint8 [N,16])."""
    def col(name):
        v = cat[name].to_numpy("float32").copy()
        v[v <= SENTINEL] = np.nan
        return v
    du, dg, dr, di, dz = (col(f"dered_{b}") for b in "ugriz")
    def logr(name):
        v = col(name); v[v < 0] = np.nan          # negative radius is unmeasured
        return np.log1p(v)
    R50, R90 = col("petroR50_r"), col("petroR90_r")
    feats = {
        "dered_u": du, "dered_g": dg, "dered_r": dr, "dered_i": di, "dered_z": dz,
        "g-r": np.clip(dg - dr, -1, 4), "u-g": np.clip(du - dg, -1, 4),
        "r-i": np.clip(dr - di, -1, 4), "i-z": np.clip(di - dz, -1, 4),
        "log_expRad_r": logr("expRad_r"), "log_deVRad_r": logr("deVRad_r"),
        "log_petroRad_r": logr("petroRad_r"), "log_petroR50_r": np.log1p(np.where(R50 < 0, np.nan, R50)),
        "log_petroR90_r": np.log1p(np.where(R90 < 0, np.nan, R90)),
        "fracDeV_r": col("fracDeV_r"), "conc_r": R90 / R50,
    }
    X = np.stack([feats[f] for f in TAB_FEATURES], axis=1).astype("float32")
    mask = (~np.isnan(X)).astype("uint8")
    return X, mask


def tabular_base_preds(stack, X16, medians=None):
    """3 base predictions (RF/HGB/MLP) from the StackingRegressor. NaNs are median-imputed first
    (the base learners were fit on dropna'd data); the presence mask carries the missingness signal.
    Pass `medians` (len-16) to reuse train medians on val; else computed from X16."""
    if medians is None:
        medians = np.nanmedian(X16, axis=0)
    Xi = np.where(np.isnan(X16), medians, X16).astype("float32")
    base = stack.transform(Xi)                     # (N,3): one column per base estimator
    return base.astype("float32"), medians


# ============================== CNN side ==============================
def cnn_embedder(cnn, layer=None):
    """Model that maps a CNN input cutout -> its fusion embedding (side-e2 'fuse64', else 'embedding')."""
    if layer is None:
        layer = 'fuse64' if any(l.name == 'fuse64' for l in cnn.layers) else 'embedding'
    return Model(cnn.input, cnn.get_layer(layer).output, name='cnn_embedder')


def cnn_embeddings(cnn, rows, crop, data_dir, z_all, preproc, batch=512, layer=None):
    """fuse64 vectors for `rows` (positions into the shards). Loads cutouts into RAM then predicts."""
    emb = cnn_embedder(cnn, layer)
    X, _ = load_into_ram(rows, crop, data_dir, z_all)        # (N,crop,crop,5) float16
    pp = ev.make_np_preprocess(preproc)
    out = np.empty((len(rows), emb.output_shape[-1]), "float32")
    for s in range(0, len(X), batch):
        out[s:s + batch] = emb.predict(pp(np.asarray(X[s:s + batch], "float32")), verbose=0)
    return out


# ============================== assemble + train ==============================
def assemble(base, mask, emb):
    """[3 base | 16 mask | 64 emb] -> (N,83) float32."""
    return np.concatenate([base, mask.astype("float32"), emb], axis=1)


def train_fusion(data_dir, cnn_path, stack_path, crop=24, preproc='color-feat+p99',
                 train_csv=None, val_csv=None, mdn=5, hidden=(128, 128, 64), l2=1e-4, drop=0.3,
                 lr=3e-4, batch=512, epochs=80, patience=10, min_lr=1e-5,
                 run_name='fusion', mlflow_token=None, experiment='fusion'):
    import joblib
    train_csv = train_csv or ev.DEFAULT_TRAIN_CSV
    val_csv = val_csv or ev.DEFAULT_VAL_CSV
    cat, z_all, o2i = load_catalog_v4(data_dir)
    stack = joblib.load(stack_path)["model"]
    cnn = tf.keras.models.load_model(cnn_path, compile=False)   # MDN custom loss can't deserialize

    def split_xy(csv, medians=None):
        obj = pd.read_csv(csv)["objid"].astype("int64").values
        rows = np.array([o2i[int(o)] for o in obj if int(o) in o2i], np.int64)
        sub = cat.iloc[rows]
        X16, mask = tabular_features(sub)
        base, medians = tabular_base_preds(stack, X16, medians)
        emb = cnn_embeddings(cnn, rows, crop, data_dir, z_all, preproc)
        X = assemble(base, mask, emb)
        y = np.log1p(z_all[rows]).astype("float32")
        return X, y, z_all[rows].astype("float64"), medians

    print("assembling train ..."); Xtr, ytr, _, med = split_xy(train_csv)
    print("assembling val   ..."); Xva, yva, zva, _ = split_xy(val_csv, medians=med)
    print(f"train {Xtr.shape}  val {Xva.shape}")

    model = compile_model(build_fusion(Xtr.shape[1], hidden, mdn, l2, drop), lr=lr, mdn=mdn)
    use_mlflow = setup_mlflow(mlflow_token, experiment=experiment)
    cbs = make_callbacks(Xva, zva, patience, min_lr)             # SigmaMadCallback predicts on Xva, scores vs zva
    ctx = __import__('mlflow').start_run(run_name=run_name) if use_mlflow else __import__('contextlib').nullcontext()
    with ctx:
        if use_mlflow:
            import mlflow
            mlflow.log_params(dict(mdn=mdn, hidden=str(hidden), l2=l2, drop=drop, lr=lr, batch=batch,
                                   preproc=preproc, n_train=len(Xtr), n_val=len(Xva), inputs='3base+16mask+64fuse64'))
        model.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=epochs, batch_size=batch, callbacks=cbs)
        zp = np.expm1(ev.mdn_point(model.predict(Xva, verbose=0)))
        m = ev.metrics_from_df(pd.DataFrame({"z_true": zva, "z_pred": zp,
                                             "dz": ev.delta_z(zva, zp)}))
        print("val:", m)
        if use_mlflow:
            mlflow.log_metrics({f"val_{k}": v for k, v in m.items()})
    return model, m
