"""join clean — tidy the bike-lane lines, merging their evidence.

Ported from 10b_clean_bikelanes. Same operations as `centerlines clean`
(hook tails, overlapping parallels) but the survivor inherits the dropped
line's signs: n_signs adds, classes merge, type is re-decided.

One difference from the centerline stage: two parallel lines running in
OPPOSITE directions may be a two-way facility with symbols on both sides.
They are kept unless join.clean.merge_opposite is true.

Short lines are reported, and only dropped with drop_short: true.
On Lexington this stage removed nothing — the centerline clean had already
handled it — but it is the place to look when the join output is messy.
"""
from __future__ import annotations

from collections import Counter

import numpy as np

from ..config import Config
from ..facility import majority_type
from ..geom import cut_hooks, find_overlaps, plen, total_km
from ..io import read_lines, write_lines


def merge_props(keep: dict, drop: dict) -> dict:
    kc = Counter(keep.get("classes") or {})
    kc.update(Counter(drop.get("classes") or {}))
    keep["classes"] = dict(kc)
    keep["n_signs"] = int(keep.get("n_signs", 0)) + int(drop.get("n_signs", 0))
    keep["conf_max"] = max(float(keep.get("conf_max", 0)), float(drop.get("conf_max", 0)))
    if kc:
        keep["type"] = majority_type(kc)
    keep["merged"] = int(keep.get("merged", 0)) + 1
    return keep


def clean(lines, props, p: dict):
    lines, nc, cl = cut_hooks(lines, p["hook_ang"], p["hook_max_m"])
    print(f"  A: {nc} hook tails cut ({cl:.0f} m)")
    dead, n_opp = find_overlaps(lines, lat_m=p["dup_lat_m"], frac=p["dup_frac"],
                                ang_tol=p["dup_ang"], merge_opposite=p.get("merge_opposite", False))
    for j, i in dead.items():
        props[i] = merge_props(props[i], props[j])
    lines = [l for k, l in enumerate(lines) if k not in dead]
    props = [q for k, q in enumerate(props) if k not in dead]
    print(f"  B: {len(dead)} overlapping lines merged into their neighbour; "
          f"{n_opp} opposite-direction pairs kept as two-way")
    short = [k for k, l in enumerate(lines) if plen(l) < p["short_m"]]
    if p.get("drop_short"):
        lines = [l for k, l in enumerate(lines) if k not in short]
        props = [q for k, q in enumerate(props) if k not in short]
        print(f"  C: {len(short)} lines shorter than {p['short_m']} m dropped")
    else:
        print(f"  C: {len(short)} lines shorter than {p['short_m']} m (kept; drop_short: true to remove)")
    return lines, props


def run(cfg: Config, check=False):
    p = cfg.get("join.clean")
    src, out = cfg.path("bikelanes"), cfg.path("bikelanes_clean")
    print(f"join clean  {cfg.region}\n  input: {src}")
    if check:
        print(f"  {'ok ' if src.exists() else 'MISSING'} {src}")
        return
    lines, props = read_lines(src)
    n0, L0 = len(lines), total_km(lines)
    ns = np.array([int(q.get("n_signs", 1)) for q in props])
    print(f"  {n0} lines {L0:.2f} km  {dict(Counter(q.get('type') for q in props))}; "
          f"one-sign lines {int((ns==1).sum())} ({100*(ns==1).mean():.0f}%)")
    lines, props = clean(lines, props, p)
    print(f"  → {len(lines)} lines {total_km(lines):.2f} km ({100*(total_km(lines)/L0-1):+.1f}%)")
    write_lines(out, lines, props, cfg.crs)
    print("→ next: bikelane gaps scan")
