"""signs filter — raw detections → accepted / dropped, with an isolation flag.

Ported from 09_bikesign_detect §3–4. Acceptance threshold per class comes
from config (signs.conf_keep). `--sweep` prints, for each threshold, the
share of detections with no other detection within neighbor_r_m: bike
symbols come in runs, so the isolated share is a proxy for false positives.
Pick the threshold where it bottoms out; past that, real symbols are being
cut too (Lexington: minimum near 0.55, chosen 0.60).

Isolated accepted signs are FLAGGED, not removed — removing later is easy,
resurrecting is not.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
from scipy.spatial import cKDTree

from ..config import Config
from ..io import read_signs_csv, write_points, write_signs_csv


def isolation(rows, r_m):
    if not rows:
        return
    xy = np.array([[r["utm_x"], r["utm_y"]] for r in rows])
    tree = cKDTree(xy)
    for r, q in zip(rows, xy):
        r["n_nb"] = len(tree.query_ball_point(q, r_m)) - 1
        r["isolated"] = int(r["n_nb"] == 0)


def sweep_table(rows, r_m, ths=(0.25, 0.35, 0.45, 0.55, 0.60, 0.65, 0.75, 0.85)):
    xy = np.array([[r["utm_x"], r["utm_y"]] for r in rows])
    conf = np.array([r["conf"] for r in rows])
    classes = sorted({r["cls_name"] for r in rows})
    print(f'{"conf≥":>6s} {"n":>6s} {"isolated":>9s} {"med nb":>7s}   per class')
    for th in ths:
        m = conf >= th
        if m.sum() < 5:
            continue
        tree = cKDTree(xy[m])
        nb = np.array([len(tree.query_ball_point(q, r_m)) - 1 for q in xy[m]])
        cnt = Counter(r["cls_name"] for r, k in zip(rows, m) if k)
        print(f"{th:6.2f} {m.sum():6d} {100*(nb==0).mean():8.1f}% {np.median(nb):7.0f}   "
              + "  ".join(f"{c}:{cnt[c]}" for c in classes))
    print("  → choose the threshold where the isolated share is lowest; when it rises "
          "again you are cutting real symbols")


def run(cfg: Config, check=False, sweep=False):
    p = cfg.get("signs")
    raw = cfg.path("signs_raw_csv")
    print(f"signs filter  {cfg.region}\n  raw: {raw}")
    if check:
        print(f"  {'ok ' if raw.exists() else 'MISSING'} {raw}")
        return
    rows = read_signs_csv(raw)
    print(f"  {len(rows)} raw detections  {dict(Counter(r['cls_name'] for r in rows))}")
    if sweep:
        sweep_table(rows, p["neighbor_r_m"])
        return
    keep_th = p["conf_keep"]
    kept = [r for r in rows if r["conf"] >= keep_th.get(r["cls_name"], 1.01)]
    dropped = [r for r in rows if r["conf"] < keep_th.get(r["cls_name"], 1.01)]
    isolation(kept, p["neighbor_r_m"])
    print(f"  accepted {len(kept)} / dropped {len(dropped)}  "
          f"{dict(Counter(r['cls_name'] for r in kept))}")
    if kept:
        print(f"  isolated among accepted: {sum(r['isolated'] for r in kept)} "
              f"({100*np.mean([r['isolated'] for r in kept]):.1f}%) — flagged only")
    f_csv = cfg.path("signs_filtered").with_suffix(".csv")
    write_signs_csv(f_csv, kept, extra=("n_nb", "isolated"))
    write_points(cfg.path("signs_filtered"), [(r["utm_x"], r["utm_y"]) for r in kept],
                 [{k: r[k] for k in ("cls_name", "conf", "tile", "n_nb", "isolated")} for r in kept],
                 cfg.crs)
    write_points(cfg.path("signs_dropped"), [(r["utm_x"], r["utm_y"]) for r in dropped],
                 [{k: r[k] for k in ("cls_name", "conf", "tile")} for r in dropped], cfg.crs)
    print("→ next: bikelane join match")
