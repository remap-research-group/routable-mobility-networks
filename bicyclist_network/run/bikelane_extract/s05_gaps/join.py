"""gaps join — apply the straight-join candidates confirmed after `gaps scan`.

Ported from 12_join_gaps. Reads <region>_gap_join.geojson, joins each pair
end-to-end with a straight segment, and writes <region>_bikelanes_joined.geojson
with n_joins / gap_total_m / obs_len_m / obs_ratio recorded per line.

Deliberately single-pass: after a join, the direction estimate at the new
line's ends is biased by the straight filler, so a second pass compounds
error for a few hundred metres of gain.
"""
from __future__ import annotations

import json
from collections import Counter

import numpy as np

from ..config import Config
from ..geom import build_chains, plen
from ..io import read_lines, write_lines
from .common import assemble, which_end


def load_pairs(lines, cand_path, eps: float = 0.5):
    """Restore (i, j, wi, wj, gap) pairs from a candidate file by matching the
    stored endpoint coordinates against the current bike-lane geometry."""
    pairs, bad = [], 0
    for c in json.load(open(cand_path))["features"]:
        pr = c["properties"]
        co = np.asarray(c["geometry"]["coordinates"], float)
        i, j = pr.get("line_a"), pr.get("line_b")
        if i is None or j is None or i >= len(lines) or j >= len(lines):
            bad += 1
            continue
        wi, wj = which_end(lines[i], co[0], eps), which_end(lines[j], co[1], eps)
        if wi is None or wj is None:
            wi, wj = which_end(lines[i], co[1], eps), which_end(lines[j], co[0], eps)
        if wi is None or wj is None:
            bad += 1
            continue
        pairs.append(dict(i=i, j=j, wi=wi, wj=wj, gap=float(pr["gap_m"])))
    if bad:
        print(f"  WARNING: {bad} candidates did not match — bike-lane file changed since "
              f"`gaps scan`; re-run scan")
    return pairs


def join(lines, props, pairs):
    nbr, applied = build_chains(len(lines), pairs)
    out_lines, out_props = assemble(lines, props, nbr)
    return out_lines, out_props, applied


def run(cfg: Config, argv=None, check: bool = False):
    src = cfg.path("bikelanes_clean")
    if not src.exists():
        src = cfg.path("bikelanes")
    cand = cfg.path("gap_join")
    out = cfg.path("bikelanes_joined")
    print(f"gaps join  {cfg.region}\n  input: {src}\n  candidates: {cand}")
    if check:
        for f in (src, cand):
            print(f"  {'ok ' if f.exists() else 'MISSING'} {f}")
        return

    lines, props = read_lines(src)
    pairs = load_pairs(lines, cand)
    out_lines, out_props, applied = join(lines, props, pairs)

    L0, L1 = sum(map(plen, lines)), sum(map(plen, out_lines))
    filled = sum(p["gap_total_m"] for p in out_props)
    nj = np.array([p["n_joins"] for p in out_props])
    obs = np.array([p["obs_ratio"] for p in out_props])
    print(f"  applied {len(applied)} / {len(pairs)} joins")
    print(f"  {len(lines)} lines {L0/1000:.2f} km → {len(out_lines)} lines {L1/1000:.2f} km "
          f"(filled {filled:.0f} m, {100*filled/L1:.1f}%)")
    print(f"  types: {dict(Counter(p['type'] for p in out_props))}")
    if (nj > 0).any():
        print(f"  obs_ratio on joined lines: median {100*np.median(obs[nj>0]):.0f}%, "
              f"min {100*obs[nj>0].min():.0f}%")
    write_lines(out, out_lines, out_props, cfg.crs)
    print ("next: bikelane gaps intersections")
    return out_lines, out_props
