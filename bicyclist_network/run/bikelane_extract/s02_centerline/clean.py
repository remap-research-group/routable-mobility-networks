"""centerlines clean — hooks, overlapping parallels → centerlines_<region>_final.geojson

Ported from 08d_clean_stitch.py.

  A1  hook tails      cut a short tail after a direction reversal
  A2  overlaps        of two lines running on top of each other, drop the shorter
  B   stitching       OFF by default (config centerlines.clean.stitch)
  A2' overlaps again  only if stitching ran; longer lines make the test sharper

Stitching is off because it joins on the endpoint *tangent*, which on a
curve points off the road: the filler becomes a chord across the bend and
can link two separate parallel lines. Reducing the gap 30 m → 12 m did not
fix it. Gaps are closed later, on the bike lanes only, in `gaps` (a few
hundred lines, direction measured over 8 m, every candidate inspected).

Intersection crossings are never joined here: which leg continues where is a
topological judgement that local evidence cannot settle.

Reading the numbers: total length dropping a lot → cleaning too aggressive
(tighten dup_lat / dup_frac). Total length rising above the input → lines
were invented (only possible with stitching).
"""
from __future__ import annotations

import time

import numpy as np

from ..config import Config
from ..geom import (Grid, angdiff_deg, cut_hooks, dedup_pts, drop_overlaps,
                    n_components, plen, tangent_dir, total_km)
from ..io import read_lines, write_lines


def stitch(lines, gap_m=30.0, ang_tol=15.0, lat_tol=1.5, rounds=3):
    """Conservative endpoint stitching (kept for reference; off by default).
    Joins only when the two ends face each other, the connector follows the
    heading, and the far end sits within lat_tol of this line's extension."""
    cur = [dedup_pts(l) for l in lines]
    joined_total = 0
    for _ in range(rounds):
        ends = []
        for i, ln in enumerate(cur):
            ends.append((i, 0, ln[0], tangent_dir(ln, 0)))
            ends.append((i, 1, ln[-1], tangent_dir(ln, 1)))
        cell = max(gap_m, 1.0)
        g = {}
        for k, (_, _, p, _) in enumerate(ends):
            g.setdefault((int(p[0] // cell), int(p[1] // cell)), []).append(k)
        pairs = []
        for k, (i, w, p, u) in enumerate(ends):
            best = None
            cx, cy = int(p[0] // cell), int(p[1] // cell)
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for k2 in g.get((cx + dx, cy + dy), []):
                        if k2 == k:
                            continue
                        j, w2, q, v = ends[k2]
                        if j == i:
                            continue
                        d = float(np.linalg.norm(q - p))
                        if d > gap_m or d == 0:
                            continue
                        if angdiff_deg(u, -v) > ang_tol:
                            continue
                        conn = (q - p) / d
                        if angdiff_deg(u, conn) > ang_tol:
                            continue
                        w_ = q - p
                        lat = abs(float(u[0] * w_[1] - u[1] * w_[0]))
                        if lat > lat_tol:
                            continue
                        sc = d + lat * 10
                        if best is None or sc < best[0]:
                            best = (sc, k, k2)
            if best:
                pairs.append(best)
        pairs.sort()
        parent = list(range(len(cur)))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        merged = {i: cur[i] for i in range(len(cur))}
        endfree, joined = {}, 0
        for _, k, k2 in pairs:
            i, w, _, _ = ends[k]
            j, w2, _, _ = ends[k2]
            ri, rj = find(i), find(j)
            if ri == rj or (ri, w) in endfree or (rj, w2) in endfree:
                continue
            A, B = merged[ri], merged[rj]
            if w == 0:
                A = A[::-1]
            if w2 == 1:
                B = B[::-1]
            merged[ri] = np.vstack([A, B])
            del merged[rj]
            parent[rj] = ri
            endfree[(ri, w)] = endfree[(rj, w2)] = True
            joined += 1
        cur = list(merged.values())
        joined_total += joined
        if joined == 0:
            break
    return cur, joined_total


def _report(tag, lines, t0):
    print(f"  {tag:22s} {len(lines):6d} lines  {total_km(lines):7.1f} km  "
          f"{n_components(lines):6d} components  {time.time()-t0:4.0f}s")


def clean(lines, p: dict):
    t0 = time.time()
    _report("input", lines, t0)
    lines = [l for l in lines if plen(l) >= p["min_len_m"]]
    lines, nc, cl = cut_hooks(lines, p["hook_ang"], p["hook_max_m"])
    print(f"  A1: {nc} hook tails cut ({cl/1000:.2f} km)")
    _report("after hooks", lines, t0)
    lines, nd = drop_overlaps(lines, lat_m=p["dup_lat_m"], frac=p["dup_frac"],
                              ang_tol=p["dup_ang"], merge_opposite=True)
    print(f"  A2: {nd} overlapping lines dropped")
    _report("after overlaps", lines, t0)
    if p.get("stitch"):
        lines, nj = stitch(lines, p.get("stitch_gap_m", 30.0), p.get("stitch_ang", 15.0),
                           p.get("stitch_lat_m", 1.5))
        print(f"  B: {nj} joins (straight fillers, so total length rises a little)")
        _report("after stitching", lines, t0)
        lines, nc2, _ = cut_hooks(lines, p["hook_ang"], p["hook_max_m"])
        lines, nd2 = drop_overlaps(lines, lat_m=p["dup_lat_m"], frac=p["dup_frac"],
                                   ang_tol=p["dup_ang"], merge_opposite=True)
        print(f"  A2': {nc2} hooks, {nd2} overlaps removed after stitching")
        _report("after 2nd overlaps", lines, t0)
    return lines


def run(cfg: Config, check=False):
    p = cfg.get("centerlines.clean")
    src, out = cfg.path("centerlines_voronoi"), cfg.path("centerlines_final")
    print(f"centerlines clean  {cfg.region}\n  input: {src}")
    if check:
        print(f"  {'ok ' if src.exists() else 'MISSING'} {src}")
        return
    lines, _ = read_lines(src)
    L0 = total_km(lines)
    lines = clean(lines, p)
    print(f"  total length vs input: {100*(total_km(lines)/L0-1):+.1f}%")
    print("  remaining breaks are mostly intersection crossings — handled in `gaps` "
          "after the bike-sign join")
    write_lines(out, lines, None, cfg.crs)
    print("→ next: bikelane signs detect")
