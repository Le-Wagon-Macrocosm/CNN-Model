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
    exc = build_exc(man)
    rows = []
    counts = {}
    for split, src, dst in [("train", "train_objids.csv", "v4-5-train.csv"),
                            ("validate", "val_objids.csv", "v4-5-validate.csv")]:
        d = pd.read_csv(os.path.join(HERE, src)); d["objid"] = d["objid"].astype("int64")
        mask = d.objid.isin(exc)
        d[~mask].to_csv(os.path.join(HERE, dst), index=False)
        counts[split] = (len(d), int((~mask).sum()), int(mask.sum()))
        for o in d[mask].objid.tolist():
            c, rs = exc[int(o)]; rows.append((int(o), split, c, rs))
        print(f"{dst}: {(~mask).sum():,} (removed {int(mask.sum())})")
    rem = pd.DataFrame(rows, columns=["objid", "split", "category", "reason"]).sort_values(["category", "objid"])
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
    L.append("# v4-5 splits - excluded samples\n")
    L.append("> Auto-generated by `exclude_objid.py`; do not edit by hand.\n")
    L.append("`v4-5-train.csv` / `v4-5-validate.csv` are v4's `train_objids.csv` / `val_objids.csv` "
             "with image-defective samples removed (**pixel-level** problems that catalog/flag-level "
             "quality cuts cannot see).\n")
    L.append("| original | count | v4-5 | count | removed |")
    L.append("|---|---|---|---|---|")
    L.append(f"| train_objids.csv | {tr0:,} | **v4-5-train.csv** | {trk:,} | {trd} |")
    L.append(f"| val_objids.csv | {va0:,} | **v4-5-validate.csv** | {vak:,} | {vad} |")
    L.append(f"\nTotal removed: **{len(rem)}** objids. Per-object reasons in **`v4-5-excluded.csv`**.\n")
    L.append("---\n")
    L.append(f"## Category 1: RA~0 miscut ({n_ra})\n")
    L.append("Objects in **frames that span the 0/360 meridian**. On those frames HDU0's TAN-WCS "
             "breaks down near RA=0, so `prepare_data`'s `world_to_pixel` mispositions the whole frame "
             "by **~95-176px**, miscutting the cutout.")
    L.append("- Criterion: `ra < 0.15` or `ra > 359.85` (source `/home/hzhang/v4_near_ra0.csv`).")
    L.append("- Fix: use the frame's own HDU3 full drift-scan astrometric solution for positioning.")
    L.append(f"- Full list: rows with `category={RA_CAT}` in `v4-5-excluded.csv` (each with its ra).\n")
    # other categories (from the manual table, grouped by category)
    for cat in list(man.category.unique()):
        sub = man[man.category == cat]
        L.append(f"## Category: {cat} ({len(sub)})\n")
        if cat == "EDGE truncation":
            L.append("Has the SDSS `EDGE` flag - the object is near the frame edge, so the cutout may be "
                     "truncated off-frame. `clean=1` does not exclude EDGE, and the object-level quality "
                     "cuts have no EDGE check, so these slip through (catalog_v4 has 870 EDGE objects; "
                     "only confirmed cases are removed here).\n")
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
