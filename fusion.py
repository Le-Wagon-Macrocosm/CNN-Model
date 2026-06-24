"""Fusion photo-z model: a small MLP + MDN head over [tabular base preds | presence mask | CNN fuse64].

Input vector (83-d):
  - 3  tabular base predictions  : RF / HGB / MLP outputs of the v4 FrozenStack (baseline_stack_v4.pkl)
  - 16 absence mask (0/1)         : 1 if that tabular feature is ABSENT (missing/dropped), else 0
  - 64 CNN embedding              : the CNN's `fuse64` (side-e2) or `embedding` (default) activation

For v4 the inputs are precomputed per (objid, missing-pattern) into a parquet by prepare_fusion_data.py;
train with `train_fusion_parquet(...)`. The older `train_fusion(...)` assembles them on the fly instead.

Target log1p(z); MDN (K-Gaussian mixture) output, same NLL loss / point estimate as the CNN
(reused from photoz_cnn / eval). The presence mask lets the head down-weight the tabular preds when
photometry/morphology is missing and lean on the image — which is exactly where the tabular baseline
fails (the ~900 "hard" colour-degenerate objects).

    from fusion import train_fusion
    train_fusion(data_dir="/content/data", cnn_path="cnn.keras",
                 stack_path="baseline_stack_v4.pkl", mlflow_token="<token>")
"""
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers as L, Model, Input
from tensorflow.keras import regularizers

import eval as ev
from photoz_cnn import mdn_nll, compile_model, load_into_ram, make_callbacks, setup_mlflow

# the 16 tabular features, in the exact order the baseline bases expect
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
        "fracDeV_r": col("fracDeV_r"), "conc_r": R90 / np.where(R50 == 0, np.nan, R50),
    }
    X = np.stack([feats[f] for f in TAB_FEATURES], axis=1).astype("float32")
    X[~np.isfinite(X)] = np.nan          # match train_baseline: replace([inf,-inf], nan) before dropna
    mask = (~np.isnan(X)).astype("uint8")
    return X, mask


def tabular_base_preds(stack, X16, medians=None):
    """3 base predictions (RF/HGB/MLP) from the tabular baseline. Accepts either:
      - v4 FrozenStack dict {"bases": {name: est}, "base_order": [...]}  -> column_stack of base.predict
      - v1 sklearn StackingRegressor                                    -> stack.transform
    NaNs are median-imputed first (the bases were fit on dropna'd data); the presence mask carries the
    missingness signal. Pass `medians` (len-16) to reuse train medians on val; else computed from X16."""
    if medians is None:
        medians = np.nanmedian(X16, axis=0)
    Xi = np.where(np.isnan(X16), medians, X16).astype("float32")
    if isinstance(stack, dict) and "bases" in stack:                       # v4 per-base FrozenStack
        base = np.column_stack([stack["bases"][m].predict(Xi) for m in stack["base_order"]])
    else:                                                                  # v1 StackingRegressor
        base = stack.transform(Xi)                 # (N,3): one column per base estimator
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
    art = joblib.load(stack_path)
    # v4 FrozenStack pkl is a dict with "bases"; the v1 bundle keeps its StackingRegressor under ["model"]
    stack = art if (isinstance(art, dict) and "bases" in art) else art["model"]
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


# ============================== parquet training (precomputed inputs) ==============================
def _bucket_smad(z_true, z_pred, n_absent):
    """sigma_MAD overall + by missing-feature bucket (0 / 1-2 / 3-5 / 6+)."""
    rows = [("all", np.ones(len(z_true), bool))]
    for lab, lo, hi in [("absent=0", 0, 0), ("absent 1-2", 1, 2), ("absent 3-5", 3, 5), ("absent 6+", 6, 99)]:
        rows.append((lab, (n_absent >= lo) & (n_absent <= hi)))
    out = {}
    for lab, mk in rows:
        if mk.sum():
            out[lab] = {"n": int(mk.sum()), "sigma_MAD": round(ev.sigma_mad(z_true[mk], z_pred[mk]), 5),
                        "outlier_%": round(ev.outlier_rate(z_true[mk], z_pred[mk]) * 100, 2)}
    return out


def train_fusion_parquet(parquet, mdn=5, hidden=(128, 128, 64), l2=1e-4, drop=0.3, lr=3e-4,
                         batch=1024, epochs=80, patience=10, min_lr=1e-5, val_full_only_es=True,
                         run_name="fusion-v4", mlflow_token=None, experiment="fusion", save_path=None):
    """Train the fusion MLP+MDN on the precomputed parquet (cols: split, n_absent, redshift,
    base_*, mask_*, emb_*). Early-stops on val sigma_MAD; reports val sigma_MAD by missing bucket."""
    df = pd.read_parquet(parquet)
    base_c = ["base_RF", "base_HGB", "base_MLP"]
    mask_c = [c for c in df.columns if c.startswith("mask_")]
    emb_c = sorted([c for c in df.columns if c.startswith("emb_")], key=lambda c: int(c[4:]))
    feat = base_c + mask_c + emb_c
    assert len(feat) == N_BASE + N_MASK + N_EMB, len(feat)

    def xy(name):
        d = df[df.split == name]
        return (d[feat].to_numpy("float32"), np.log1p(d["redshift"].to_numpy("float32")),
                d["redshift"].to_numpy("float64"), d["n_absent"].to_numpy("int16"))
    Xtr, ytr, _, _ = xy("train")
    Xva, yva, zva, nva = xy("val")
    print(f"train {Xtr.shape}  val {Xva.shape}  (in-dim {len(feat)})")

    model = compile_model(build_fusion(len(feat), hidden, mdn, l2, drop), lr=lr, mdn=mdn)
    # early-stop on the FULL-feature val subset (comparable to the tabular/CNN baselines), if asked
    es_mask = (nva == 0) if val_full_only_es else np.ones(len(nva), bool)
    cbs = make_callbacks(Xva[es_mask], zva[es_mask], patience, min_lr)

    use_mlflow = setup_mlflow(mlflow_token, experiment=experiment)
    ctx = __import__("mlflow").start_run(run_name=run_name) if use_mlflow else __import__("contextlib").nullcontext()
    with ctx:
        if use_mlflow:
            import mlflow
            mlflow.log_params(dict(mdn=mdn, hidden=str(hidden), l2=l2, drop=drop, lr=lr, batch=batch,
                                   n_train=len(Xtr), n_val=len(Xva), inputs="3base+16mask+64emb"))
        model.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=epochs, batch_size=batch, callbacks=cbs)
        zp = np.expm1(ev.mdn_point(model.predict(Xva, verbose=0)))
        buckets = _bucket_smad(zva, zp, nva)
        print("\n=== val sigma_MAD by missing bucket ===")
        for k, v in buckets.items():
            print(f"  {k:12s} {v}")
        if use_mlflow:
            mlflow.log_metrics({f"val_{k.replace(' ', '_')}_sMAD": v["sigma_MAD"] for k, v in buckets.items()})
    if save_path:
        model.save(save_path); print("saved", save_path)
    return model, buckets
