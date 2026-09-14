"""join match — bike signs × centerlines → <region>_bikelanes.geojson

Ported from 10_bikelane_join (v2). Each sign is matched to EVERY centerline
within radius_m of its centre (default 1.2 m; lane half-width is 1.75 m, so
anything inside is the same lane). A line's type is the majority of its
signs' classes; ties break BikeOnly > Sharrow > OnlyBikeBus.

Why a radius and not the YOLO box: the box is axis-aligned, so on a diagonal
road it is loose along the road and tight across it — the criterion would
depend on heading. Why all lines and not the best one: when a box straddles
two lanes, "longest intersection" picks by where the symbol happened to sit,
which is not evidence. Two lines inside R is information; keep both.

Unmatched signs are saved with the distance to the nearest centerline:
  ~R..3 m   R slightly too small
  5..20 m   a centerline exists but in the wrong lane (centerline accuracy)
  > 20 m    no centerline there (marking not detected, or sign false positive)

`--sweep` prints match rate, lines-per-sign and total km for several radii;
lines-per-sign jumping (Lexington: 1.10 → 1.34 between R=2 and R=3) means
neighbouring lanes are being pulled in.
"""
from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np
from shapely.geometry import LineString, Point
from shapely.strtree import STRtree

from ..config import Config
from ..facility import majority_type
from ..io import read_lines, read_signs_csv, write_lines, write_points


def match(centerlines, signs, radius, miss_search_m=60.0):
    """Returns hit=[(sign_idx, [line_idx...])], miss=[(sign_idx, nearest_m)]."""
    geoms = [LineString(c) for c in centerlines]
    tree = STRtree(geoms)
    hit, miss = [], []
    for si, s in enumerate(signs):
        pt = Point(s["utm_x"], s["utm_y"])
        idxs = [int(ci) for ci in tree.query(pt.buffer(radius))
                if geoms[int(ci)].distance(pt) <= radius]
        if idxs:
            hit.append((si, idxs))
        else:
            near = tree.query(pt.buffer(miss_search_m))
            d = min((geoms[int(ci)].distance(pt) for ci in near), default=np.inf)
            miss.append((si, float(d)))
    return hit, miss


def sweep_table(centerlines, signs, radii=(0.6, 0.9, 1.2, 1.5, 2.0, 3.0)):
    L = np.array([LineString(c).length for c in centerlines])
    print(f'{"R(m)":>5s} {"matched":>13s} {"lines/sign":>10s} {"lines":>6s} {"km":>7s}')
    for r in radii:
        h, _ = match(centerlines, signs, r)
        npl = np.mean([len(v) for _, v in h]) if h else 0
        uniq = {li for _, v in h for li in v}
        print(f"{r:5.1f} {len(h):6d} ({100*len(h)/max(len(signs),1):3.0f}%) {npl:10.2f} "
              f"{len(uniq):6d} {L[list(uniq)].sum()/1000 if uniq else 0:7.1f}")
    print("  → largest R with lines/sign still ~1.0–1.1; a jump means adjacent lanes")


def build_bikelanes(centerlines, signs, hit):
    line_signs = defaultdict(list)
    for si, idxs in hit:
        for li in idxs:
            line_signs[li].append(signs[si])
    lines, props = [], []
    for li, ss in sorted(line_signs.items()):
        cnt = Counter(s["cls_name"] for s in ss)
        lines.append(centerlines[li])
        props.append(dict(type=majority_type(cnt), n_signs=len(ss), classes=dict(cnt),
                          conf_max=round(max(s["conf"] for s in ss), 3), centerline_id=li))
    return lines, props


def run(cfg: Config, check=False, sweep=False, radius=None):
    R = float(radius or cfg.get("join.radius_m"))
    center = cfg.path("centerlines_final")
    sig = cfg.path("signs_filtered").with_suffix(".csv")
    print(f"join match  {cfg.region}\n  centerlines: {center}\n  signs: {sig}\n  R = {R} m")
    if check:
        for f in (center, sig):
            print(f"  {'ok ' if f.exists() else 'MISSING'} {f}")
        return
    centerlines, _ = read_lines(center)
    signs = read_signs_csv(sig)
    print(f"  {len(centerlines)} centerlines, {len(signs)} signs "
          f"{dict(Counter(s['cls_name'] for s in signs))}")
    if sweep:
        sweep_table(centerlines, signs)
        return
    hit, miss = match(centerlines, signs, R)
    lines, props = build_bikelanes(centerlines, signs, hit)
    tot = sum(LineString(l).length for l in lines)
    ns = np.array([p["n_signs"] for p in props])
    print(f"  matched {len(hit)} signs / unmatched {len(miss)}")
    print(f"  {len(lines)} bike lanes, {tot/1000:.2f} km, "
          f"{dict(Counter(p['type'] for p in props))}")
    if len(ns):
        print(f"  signs per line: median {np.median(ns):.0f}; backed by one sign only: "
              f"{int((ns==1).sum())} ({100*(ns==1).mean():.0f}%)")
    if miss:
        d = np.array([x for _, x in miss])
        d = d[np.isfinite(d)]
        print("  unmatched, distance to nearest centerline:")
        for lo, hi, why in ((R, 3, "R slightly small"), (3, 5, "R slightly small"),
                            (5, 20, "wrong lane — centerline accuracy"),
                            (20, 1e9, "no centerline — marking missed or sign FP")):
            n = int(((d >= lo) & (d < hi)).sum())
            if n:
                print(f"    {lo:4.1f}–{min(hi,999):4.0f} m  {n:5d}  {why}")
    write_lines(cfg.path("bikelanes"), lines, props, cfg.crs)
    write_points(cfg.path("bikelanes_unmatched"),
                 [(signs[si]["utm_x"], signs[si]["utm_y"]) for si, _ in miss],
                 [dict(cls=signs[si]["cls_name"], conf=signs[si]["conf"], tile=signs[si]["tile"],
                       isolated=signs[si].get("isolated", 0),
                       nearest_m=round(d, 1) if np.isfinite(d) else -1) for si, d in miss],
                 cfg.crs)
    print("→ next: bikelane join clean")
