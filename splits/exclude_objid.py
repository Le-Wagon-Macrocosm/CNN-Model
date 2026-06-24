#!/usr/bin/env python3
"""Add an objid to the v4-5 exclusion set and rebuild csv / md / gallery.

Usage:
    python exclude_objid.py <objid> "<reason>" [category]   # add one and rebuild
    python exclude_objid.py                                 # rebuild only (no add)
    python exclude_objid.py --frame <run/camcol/field> "<reason>" [category]

Category defaults to "frame contamination"; other common ones:
"satellite trail", "EDGE truncation". The frame is looked up automatically from
catalog_v4, so the reason text does not need to include it.

Data sources:
  - 514 RA~0 miscuts: /home/hzhang/v4_near_ra0.csv (fixed, not in this table)
  - manual per-object exclusions: manual_exclusions.csv (maintained by this script;
    columns objid,category,frame,reason)
Outputs:
  - v4-5-train.csv / v4-5-validate.csv  (= original train/val minus all exclusions)
  - v4-5-excluded.csv                   (per object: objid,split,category,reason)
  - v4-5-excluded.md                    (documentation, auto-generated)
  - reruns /home/hzhang/v4_baseline_gallery.py (gallery reads v4-5-excluded.csv)
"""
import sys, os, subprocess
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CAT_V4 = "/home/hzhang/catalog_v4.parquet"
RA_CSV = "/home/hzhang/v4_near_ra0.csv"
MANUAL = os.path.join(HERE, "manual_exclusions.csv")
GALLERY = "/home/hzhang/v4_baseline_gallery.py"
DEFAULT_CAT = "frame contamination"
RA_CAT = "RA~0 miscut"
RA_REASON = "spans an RA 0/360 frame; HDU0 TAN-WCS off by ~95-176px over the whole frame -> cutout miscut"

# Samples ACTUALLY removed from the v4-5 training/validation split: RA~0 mis-cuts CONFIRMED FROM THE
# IMAGE (the cutout is pure sky with no central source -> the galaxy fell outside the frame after the
# ~95-176px WCS mispositioning). Detected by check_ra0_real.py: of the 514 RA~0 suspects, 182 have
# central g+r+i SNR < 12 (vs ~57 median for normal galaxies); the other ~330 show a clear centred
# galaxy and are RETAINED (their frame does not actually straddle 0/360). The rest of the suspect
# catalog below (full RA~0 list + manual_exclusions.csv) stays FLAGGED for QC / out of the gallery
# but RETAINED in the split — precautionary, not image-confirmed defective.
IMG_VICTIMS = os.path.join(HERE, "ra0_image_victims.csv")


def load_confirmed():
    """objid -> (category, reason) for the image-verified RA~0 victims removed from the split."""
    if not os.path.exists(IMG_VICTIMS):
        return {}
    v = pd.read_csv(IMG_VICTIMS, dtype={"objid": "int64"})
    return {int(o): ("RA~0 miscut (image-verified)", str(rs))
            for o, rs in zip(v.objid.tolist(), v.reason.tolist())}


CONFIRMED = load_confirmed()


def load_manual():
    if os.path.exists(MANUAL):
        return pd.read_csv(MANUAL, dtype={"objid": "int64", "frame": "string"})
    return pd.DataFrame(columns=["objid", "category", "frame", "reason"])


def add_one(oid, reason, category):
    v = pd.read_parquet(CAT_V4, columns=["objid", "run", "camcol", "field"])
    r = v[v.objid == oid]
    frame = ""
    if len(r):
        rr = r.iloc[0]; frame = f"{int(rr.run)}/{int(rr.camcol)}/{int(rr.field)}"
    else:
        print(f"! {oid} not in catalog_v4 (frame left blank)")
    m = load_manual()
    m = m[m.objid != oid]                       # dedup (re-add = update)
    m = pd.concat([m, pd.DataFrame([{"objid": oid, "category": category,
                                     "frame": frame, "reason": reason}])], ignore_index=True)
    m.to_csv(MANUAL, index=False)
    print(f"added: {oid}  [{category}]  frame {frame or '?'}  - {reason}")
    return m


def add_frame(spec, reason, category):
    run, cc, fld = (int(x) for x in spec.split("/"))
    v = pd.read_parquet(CAT_V4, columns=["objid", "run", "camcol", "field"])
    s = v[(v.run == run) & (v.camcol == cc) & (v.field == fld)]
    frame = f"{run}/{cc}/{fld}"
    if not len(s):
        print(f"! frame {frame} has no objid in catalog_v4"); return
    m = load_manual()
    for oid in s.objid.astype("int64").tolist():
        m = m[m.objid != int(oid)]
        m = pd.concat([m, pd.DataFrame([{"objid": int(oid), "category": category,
                                         "frame": frame, "reason": reason}])], ignore_index=True)
    m.to_csv(MANUAL, index=False)
    print(f"frame {frame}: added all {len(s)} objids  [{category}]  - {reason}")


def build_exc(man):
    ra = pd.read_csv(RA_CSV); ra["objid"] = ra["objid"].astype("int64")
    exc = {}
    for o, rav in zip(ra.objid.tolist(), ra.ra.tolist()):
        exc[int(o)] = (RA_CAT, f"{RA_REASON} (ra={rav:.4f})")
    for o, c, fr, rs in zip(man.objid.tolist(), man.category.tolist(),
                            man.frame.tolist(), man.reason.tolist()):
        frametxt = f" (frame {fr})" if isinstance(fr, str) and fr else ""
        exc[int(o)] = (str(c), f"{rs}{frametxt}")
    return exc


def rebuild():
    man = load_manual()
    exc = build_exc(man)                        # full suspect catalog (RA~0 list + manual) -> docs/gallery
    rows = []                                   # all suspects present in a split (v4-5-excluded.csv)
    counts = {}
    for split, src, dst in [("train", "train_objids.csv", "v4-5-train.csv"),
                            ("validate", "val_objids.csv", "v4-5-validate.csv")]:
        d = pd.read_csv(os.path.join(HERE, src)); d["objid"] = d["objid"].astype("int64")
        rm = d.objid.isin(CONFIRMED)            # the split removes ONLY the confirmed-broken samples
        d[~rm].to_csv(os.path.join(HERE, dst), index=False)
        counts[split] = (len(d), int((~rm).sum()), int(rm.sum()))
        for o in d[d.objid.isin(exc)].objid.tolist():   # full suspect catalog for this split
            c, rs = exc[int(o)]; rows.append((int(o), split, c, rs, int(o) in CONFIRMED))
        print(f"{dst}: {(~rm).sum():,} (removed {int(rm.sum())} confirmed; "
              f"{int(d.objid.isin(exc).sum())} suspects flagged but kept)")
    rem = pd.DataFrame(rows, columns=["objid", "split", "category", "reason", "removed_from_split"]) \
            .sort_values(["category", "objid"])
    rem.to_csv(os.path.join(HERE, "v4-5-excluded.csv"), index=False)
    write_md(man, rem, counts)
    print(f"\nv4-5-excluded.csv: {len(rem)} rows")
    print(rem.category.value_counts().to_string())
    if os.environ.get("SKIP_GALLERY") == "1":
        print("(skipping gallery rerun, SKIP_GALLERY=1)")
    else:
        print("\nrerunning gallery ...")
        subprocess.run(["python3", GALLERY], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("gallery updated (/home/hzhang/v4_baseline_gallery.html)")


def write_md(man, rem, counts):
    n_ra = int((rem.category == RA_CAT).sum())
    tr0, trk, trd = counts["train"]; va0, vak, vad = counts["validate"]
    L = []
    n_susp = len(rem); n_conf = int(rem.removed_from_split.sum())
    L.append("# v4-5 splits - confirmed removals & suspect catalog\n")
    L.append("> Auto-generated by `exclude_objid.py`; do not edit by hand.\n")
    L.append(f"`v4-5-train.csv` / `v4-5-validate.csv` are v4's `train_objids.csv` / `val_objids.csv` "
             f"with **only the {n_conf} confirmed-broken cutouts removed** (pure-noise thumbnails from "
             f"the RA 0/360 WCS bug). Everything else below is a **suspect catalog**: frames flagged "
             f"during image QC and kept out of the baseline gallery, but **RETAINED in the split** "
             f"(precautionary, not confirmed defective — dropping them would discard good data).\n")
    L.append("| original | count | v4-5 | count | removed |")
    L.append("|---|---|---|---|---|")
    L.append(f"| train_objids.csv | {tr0:,} | **v4-5-train.csv** | {trk:,} | {trd} |")
    L.append(f"| val_objids.csv | {va0:,} | **v4-5-validate.csv** | {vak:,} | {vad} |")
    L.append(f"\n**Removed from split: {n_conf}** (confirmed). **Flagged as suspect (kept): {n_susp - n_conf}**. "
             f"Per-object rows in **`v4-5-excluded.csv`** (column `removed_from_split` marks the {n_conf} "
             f"actually dropped).\n")
    L.append("---\n")
    L.append(f"## Confirmed removals ({n_conf})\n")
    L.append("**Image-verified RA~0 mis-cuts**: of the 514 RA~0 suspects, these have a pure-noise cutout "
             "(central g+r+i SNR < 12, vs ~57 median for normal galaxies) — the galaxy fell outside the "
             "frame after the ~95-176px WCS mispositioning, so the image is unusable. The other ~330 "
             "suspects show a clear centred galaxy and are RETAINED (their frame does not straddle 0/360). "
             "Detected by `check_ra0_real.py`; full per-object list (objid, snr, centroid) in "
             "**`ra0_image_victims.csv`**.\n")
    L.append("---\n")
    L.append(f"## Suspect catalog (flagged, retained)\n")
    L.append(f"### RA~0 miscut ({n_ra})\n")
    L.append("Objects in **frames that span the 0/360 meridian**. On those frames HDU0's TAN-WCS "
             "breaks down near RA=0, so `prepare_data`'s `world_to_pixel` mispositions the whole frame "
             "by **~95-176px**, miscutting the cutout.")
    L.append("- Criterion: `ra < 0.15` or `ra > 359.85` (source `/home/hzhang/v4_near_ra0.csv`).")
    L.append("- Fix: use the frame's own HDU3 full drift-scan astrometric solution for positioning.")
    L.append(f"- Full list: rows with `category={RA_CAT}` in `v4-5-excluded.csv` (each with its ra).\n")
    # other categories (from the manual table, grouped by category)
    for cat in list(man.category.unique()):
        sub = man[man.category == cat]
        L.append(f"### {cat} ({len(sub)})\n")
        if cat == "EDGE truncation":
            L.append("Has the SDSS `EDGE` flag - the object is near the frame edge, so the cutout may be "
                     "truncated off-frame. `clean=1` does not exclude EDGE, and the object-level quality "
                     "cuts have no EDGE check, so these slip through (catalog_v4 has 870 EDGE objects; "
                     "only the ones manually inspected are flagged here).\n")
        else:
            L.append("Frame pixels have large-area scattered light / diffuse glow / ghosts / bright-star "
                     "bleed / trails, yet SDSS still labels them `quality=3 GOOD`, `clean=1` - flags miss it.\n")
        L.append("| objid | frame | symptom |")
        L.append("|---|---|---|")
        for o, fr, rs in zip(sub.objid, sub.frame, sub.reason):
            L.append(f"| {o} | {fr} | {rs} |")
        L.append("")
    L.append("---\n")
    L.append("## Notes\n")
    L.append("- These are all **image-level** defects; tabular (modelMag/colors) and field flags "
             "(quality/calibStatus) cannot detect them.")
    L.append("- Add one: `python exclude_objid.py <objid> \"<reason>\" [category]`")
    L.append("- Exclude a whole frame: `python exclude_objid.py --frame <run/camcol/field> \"<reason>\" [category]`")
    open(os.path.join(HERE, "v4-5-excluded.md"), "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "--frame":
        # whole-frame mode: python exclude_objid.py --frame <run/camcol/field> "<reason>" [category]
        spec = sys.argv[2]; reason = sys.argv[3]
        category = sys.argv[4] if len(sys.argv) > 4 else DEFAULT_CAT
        add_frame(spec, reason, category)
    elif len(sys.argv) >= 3:
        oid = int(sys.argv[1]); reason = sys.argv[2]
        category = sys.argv[3] if len(sys.argv) > 3 else DEFAULT_CAT
        add_one(oid, reason, category)
    elif len(sys.argv) == 1:
        print("(rebuild only, no add)")
    else:
        print('usage: python exclude_objid.py <objid> "<reason>" [category]\n'
              '       python exclude_objid.py --frame <run/camcol/field> "<reason>" [category]')
        sys.exit(1)
    rebuild()
