"""gaps scan — find endpoint pairs that could be joined. Joins nothing.

Ported from 11_gap_scan. Writes two candidate files:
  <region>_gap_join.geojson      straight joins, to be applied by `gaps join`
  <region>_gap_crossing.geojson  gaps through an intersection, for `gaps intersections`

Why this is separate from joining: the earlier stitching (08d) joined 2,749
pairs across all car-lane centerlines and could not be checked. Here the
input is only the bike lanes (a few hundred lines), the gap is short, the
direction is a trend over the last 8 m rather than a tangent, and the
candidate count is small enough to look at every one.
"""
from __future__ import annotations

import json
from collections import Counter

import numpy as np

from ..config import Config
from ..geom import gap_candidates, is_crossing, pick_unique_ends, plen
from ..io import read_lines, backup_if_exists, crs_block


def sweep_table(lines, props, gaps=(10, 15, 20, 25, 30, 40), ang=12.0, lat_max_m=1.0,
                tail_m=8.0):
    """How each condition filters as the gap ceiling grows. Print-only aid for
    choosing gap_max_m: the true pairs run out where the count stops growing."""
    cands = gap_candidates(lines, max(gaps), ang, lat_max_m, tail_m, prefilter=False)
    for c in cands:
        c["same_type"] = props[c["i"]].get("type") == props[c["j"]].get("type")
    print(f'{"gap":>5s} {"pairs":>6s} {"facing":>7s} {"+along":>7s} {"+lat":>6s} {"+type":>6s}')
    for g in gaps:
        s0 = [c for c in cands if c["gap"] <= g]
        s1 = [c for c in s0 if c["facing"] <= ang]
        s2 = [c for c in s1 if c["along"] <= ang]
        s3 = [c for c in s2 if c["lat"] <= lat_max_m]
        s4 = [c for c in s3 if c["same_type"]]
        print(f"{g:5d} {len(s0):6d} {len(s1):7d} {len(s2):7d} {len(s3):6d} {len(s4):6d}")
    return cands


def scan(lines, props, background, p: dict):
    """Return (joinable, crossing) candidate lists."""
    cands = gap_candidates(lines, p["gap_max_m"], p["ang"], p["lat_max_m"], p["tail_m"])
    keep = pick_unique_ends(cands)
    xr = p.get("crossing", {})
    for c in keep:
        c["ti"], c["tj"] = props[c["i"]].get("type"), props[c["j"]].get("type")
        c["xr"] = bool(background) and is_crossing(
            c["p"], c["q"], background, xr.get("radius_m", 30.0),
            xr.get("ang", 30.0), xr.get("min_len_m", 3.0))
    return [c for c in keep if not c["xr"]], [c for c in keep if c["xr"]]


def _dump(path, recs, crs):
    backup_if_exists(path)
    feats = [{"type": "Feature",
              "properties": {"id": n, "gap_m": round(c["gap"], 1), "facing": round(c["facing"], 1),
                             "along": round(c["along"], 1), "lat_m": round(c["lat"], 2),
                             "line_a": int(c["i"]), "line_b": int(c["j"]),
                             "type_a": c["ti"], "type_b": c["tj"], "crossing": int(c["xr"])},
              "geometry": {"type": "LineString",
                           "coordinates": [[float(c["p"][0]), float(c["p"][1])],
                                           [float(c["q"][0]), float(c["q"][1])]]}}
             for n, c in enumerate(recs)]
    json.dump({"type": "FeatureCollection", "crs": crs_block(crs), "features": feats},
              open(path, "w"))
    print(f"  saved {path.name}  ({len(feats)} candidates)")


def run(cfg: Config, argv=None, check: bool = False, sweep: bool = False):
    p = cfg.get("gaps.scan")
    src = cfg.path("bikelanes_clean")
    if not src.exists():
        src = cfg.path("bikelanes")
    center = cfg.path("centerlines_final")
    print(f"gaps scan  {cfg.region}\n  input: {src}\n  background: {center}")
    if check:
        for f in (src, center):
            print(f"  {'ok ' if f.exists() else 'MISSING'} {f}")
        return

    lines, props = read_lines(src)
    print(f"  {len(lines)} bike lanes, {sum(map(plen, lines))/1000:.2f} km, "
          f"{dict(Counter(q.get('type') for q in props))}")
    background = read_lines(center)[0] if center.exists() else []
    if not background:
        print("  WARNING: no centerlines file — every candidate will be treated as a straight join")

    if sweep:
        sweep_table(lines, props, ang=p["ang"], lat_max_m=p["lat_max_m"], tail_m=p["tail_m"])
        return

    joinable, crossing = scan(lines, props, background, p)
    gj = np.array([c["gap"] for c in joinable]) if joinable else np.zeros(0)
    gx = np.array([c["gap"] for c in crossing]) if crossing else np.zeros(0)
    print(f"  candidates: {len(joinable)} straight ({gj.sum():.0f} m), "
          f"{len(crossing)} through an intersection ({gx.sum():.0f} m)")
    _dump(cfg.path("gap_join"), joinable, cfg.crs)
    _dump(cfg.path("gap_crossing"), crossing, cfg.crs)
    print("  → look at every candidate (QGIS over the orthophoto) before `bikelane gaps join`")
    return joinable, crossing
