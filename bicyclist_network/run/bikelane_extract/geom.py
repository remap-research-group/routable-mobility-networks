"""Polyline geometry shared by the centerline, join and gap stages.

All functions take polylines as (N, 2) float arrays in a projected CRS (metres).
Nothing here reads or writes files.

Consolidated from 08d_clean_stitch.py, 10b_clean_bikelanes, 11_gap_scan,
12_join_gaps and 16_connect_intersections, which each carried a copy.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

Line = np.ndarray  # (N, 2)


# ───────────────────────── basics ─────────────────────────

def dedup_pts(ln: Sequence, eps: float = 1e-6) -> Line:
    """Drop consecutive duplicate vertices. A zero-length segment breaks every
    angle computation downstream, so this is applied on load."""
    ln = np.asarray(ln, float)
    if len(ln) < 2:
        return ln
    keep = np.concatenate([[True], np.linalg.norm(np.diff(ln, axis=0), axis=1) > eps])
    out = ln[keep]
    return out if len(out) >= 2 else ln[:2]


def plen(ln: Line) -> float:
    ln = np.asarray(ln, float)
    return float(np.sum(np.linalg.norm(np.diff(ln, axis=0), axis=1)))


def total_km(lines: Iterable[Line]) -> float:
    return sum(plen(l) for l in lines) / 1000.0


def unit(v: np.ndarray) -> np.ndarray:
    n = np.hypot(*v)
    return v / n if n > 0 else np.zeros(2)


def angdiff_deg(u: np.ndarray, v: np.ndarray) -> float:
    """Angle between two unit vectors, 0..180."""
    return float(np.degrees(np.arccos(float(np.clip(u @ v, -1, 1)))))


def axial_diff_deg(u: np.ndarray, v: np.ndarray) -> float:
    """Angle between two directions ignoring sense, 0..90."""
    d = angdiff_deg(u, v)
    return min(d, 180.0 - d)


def chord_dir(ln: Line) -> np.ndarray:
    """Overall direction: last vertex minus first."""
    return unit(ln[-1] - ln[0])


def tangent_dir(ln: Line, which: int, k: int = 3) -> np.ndarray:
    """Outward direction at an endpoint from the last `k` vertices (the tangent).

    Only kept for centerline overlap merging. On curves this points off the
    road; for joining, use `tail_dir` instead.
    """
    if which == 0:
        return unit(ln[0] - ln[min(k, len(ln) - 1)])
    return unit(ln[-1] - ln[max(-k - 1, -len(ln))])


def tail_dir(ln: Line, which: int, tail_m: float = 8.0) -> np.ndarray:
    """Outward direction at an endpoint, measured over the last `tail_m` metres
    (a trend, not a tangent). This is what keeps joins following a curve
    instead of drawing chords."""
    p = ln if which == 1 else ln[::-1]
    end = p[-1]
    d = np.linalg.norm(np.diff(p, axis=0), axis=1)[::-1]
    cum = np.cumsum(d)
    k = min(max(int(np.searchsorted(cum, tail_m)) + 1, 1), len(p) - 1)
    return unit(end - p[-1 - k])


def endpoint(ln: Line, which: int) -> np.ndarray:
    return ln[0] if which == 0 else ln[-1]


def lateral_offset(u: np.ndarray, p: np.ndarray, q: np.ndarray) -> float:
    """Perpendicular distance of q from the ray (p, u)."""
    w = q - p
    return abs(float(u[0] * w[1] - u[1] * w[0]))


def seg_dists(pts: np.ndarray, ln: Line) -> np.ndarray:
    """Distance from each point to the nearest segment of `ln` (vectorised)."""
    A = ln[:-1]
    B = ln[1:]
    AB = B - A
    L2 = (AB ** 2).sum(1)
    L2[L2 == 0] = 1e-12
    P = pts[:, None, :] - A[None, :, :]
    t = np.clip((P * AB[None]).sum(2) / L2[None], 0, 1)
    proj = A[None] + t[..., None] * AB[None]
    return np.linalg.norm(pts[:, None, :] - proj, axis=2).min(1)


def sample_along(ln: Line, k: int | None = None) -> np.ndarray:
    """k points evenly spaced by arc length (default 4..20 depending on length)."""
    L = plen(ln)
    k = k or max(4, min(20, int(L / 2)))
    cum = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(ln, axis=0), axis=1))])
    if cum[-1] == 0:
        return ln[:1]
    t = np.linspace(0, 1, k) * cum[-1]
    return np.stack([np.interp(t, cum, ln[:, 0]), np.interp(t, cum, ln[:, 1])], 1)


class Grid:
    """Cell index over polylines for neighbour lookup."""

    def __init__(self, lines: Sequence[Line], cell: float):
        self.cell = cell
        self.g: dict[tuple[int, int], set[int]] = {}
        for i, ln in enumerate(lines):
            for c in self._cells(ln):
                self.g.setdefault(c, set()).add(i)

    def _cells(self, ln: Line):
        return {(int(x // self.cell), int(y // self.cell)) for x, y in ln}

    def near(self, ln: Line, pad: int = 1) -> set[int]:
        out: set[int] = set()
        for cx, cy in self._cells(ln):
            for dx in range(-pad, pad + 1):
                for dy in range(-pad, pad + 1):
                    out |= self.g.get((cx + dx, cy + dy), set())
        return out


# ───────────────────────── A: hook tails ─────────────────────────

def cut_hooks(lines: Sequence[Line], ang_th: float = 110.0, max_tail_m: float = 15.0):
    """Cut a short tail after a direction reversal inside a polyline.

    Returns (lines, n_cut, cut_len_m). Only tails shorter than `max_tail_m`
    beyond a bend of >= `ang_th` degrees are removed; the polyline is never
    dropped entirely.
    """
    out, n_cut, cut_len = [], 0, 0.0
    for ln in lines:
        ln = dedup_pts(ln)
        if len(ln) < 3:
            out.append(ln)
            continue
        d = np.diff(ln, axis=0)
        nrm = np.linalg.norm(d, axis=1)
        nrm[nrm == 0] = 1e-12
        u = d / nrm[:, None]
        ang = np.degrees(np.arccos(np.clip((u[:-1] * u[1:]).sum(1), -1, 1)))
        rev = np.where(ang >= ang_th)[0]  # bend i is at vertex i+1
        if len(rev) == 0:
            out.append(ln)
            continue
        cum = np.concatenate([[0], np.cumsum(nrm)])
        L = cum[-1]
        lo, hi = 0, len(ln) - 1
        k = rev[-1] + 1
        if 0 < L - cum[k] <= max_tail_m:
            hi = k
            n_cut += 1
            cut_len += L - cum[k]
        k0 = rev[0] + 1
        if 0 < cum[k0] <= max_tail_m and k0 < hi:
            lo = k0
            n_cut += 1
            cut_len += cum[k0]
        keep = ln[lo:hi + 1]
        out.append(keep if len(keep) >= 2 else ln)
    return out, n_cut, cut_len


# ───────────────────────── B: overlapping parallels ─────────────────────────

def find_overlaps(lines: Sequence[Line], lat_m: float = 1.0, frac: float = 0.9,
                  ang_tol: float = 20.0, merge_opposite: bool = True):
    """Find shorter polylines that run on top of a longer one.

    A line j is "covered" by a longer line i when at least `frac` of sample
    points along j lie within `lat_m` of i and the two are aligned within
    `ang_tol` degrees. With `merge_opposite=False`, aligned-but-opposite pairs
    are reported but not marked dead — on bike lanes those are two-way
    facilities with symbols in both directions.

    Returns (dead, n_opposite_kept) where dead maps j -> i (the survivor).
    """
    order = sorted(range(len(lines)), key=lambda i: -plen(lines[i]))
    grid = Grid(lines, cell=50.0)
    dead: dict[int, int] = {}
    n_opp = 0
    for i in order:
        if i in dead:
            continue
        gi = lines[i]
        Li = plen(gi)
        ui = chord_dir(gi)
        for j in grid.near(gi):
            if j == i or j in dead:
                continue
            gj = lines[j]
            if plen(gj) > Li:
                continue
            da = angdiff_deg(ui, chord_dir(gj))
            if min(da, 180 - da) > ang_tol:
                continue
            if (seg_dists(sample_along(gj), gi) <= lat_m).mean() < frac:
                continue
            if da > 90 and not merge_opposite:
                n_opp += 1
                continue
            dead[j] = i
    return dead, n_opp


def drop_overlaps(lines: Sequence[Line], **kw):
    """`find_overlaps` then drop the dead lines. Returns (lines, n_dropped)."""
    dead, _ = find_overlaps(lines, **kw)
    return [ln for k, ln in enumerate(lines) if k not in dead], len(dead)


# ───────────────────────── C: endpoint gap candidates ─────────────────────────

def gap_candidates(lines: Sequence[Line], gap_max_m: float, ang: float, lat_max_m: float,
                   tail_m: float = 8.0, prefilter: bool = True) -> list[dict]:
    """Pairs of endpoints that could be joined by a straight segment.

    Three conditions, each measured with `tail_dir`:
      facing  — the two outward directions oppose each other (one ends, other begins)
      along   — the connector runs the way the line was already heading
      lat     — the other endpoint sits close to this line's extension
                (this is what stops two parallel lines being joined)

    With `prefilter=False` every pair within `gap_max_m` is returned with its
    measurements, for sweeping thresholds in a notebook. Each candidate has
    keys k, k2 (endpoint ids), i, j, wi, wj, gap, facing, along, lat, p, q.
    """
    ends = []
    for i, ln in enumerate(lines):
        ends.append((i, 0, ln[0], tail_dir(ln, 0, tail_m)))
        ends.append((i, 1, ln[-1], tail_dir(ln, 1, tail_m)))
    P = np.array([e[2] for e in ends])
    cands = []
    for k, (i, w, p, u) in enumerate(ends):
        d = np.linalg.norm(P - p, axis=1)
        for k2 in np.where((d > 0) & (d <= gap_max_m))[0]:
            k2 = int(k2)
            if k2 <= k:
                continue
            j, w2, q, v = ends[k2]
            if j == i:
                continue
            gap = float(d[k2])
            conn = (q - p) / gap
            facing, along = angdiff_deg(u, -v), angdiff_deg(u, conn)
            lat = lateral_offset(u, p, q)
            if prefilter and (facing > ang or along > ang or lat > lat_max_m):
                continue
            cands.append(dict(k=k, k2=k2, i=i, j=j, wi=w, wj=w2, gap=gap,
                              facing=facing, along=along, lat=lat, p=p, q=q))
    return cands


def pick_unique_ends(cands: list[dict]) -> list[dict]:
    """Each endpoint is used at most once; smallest gap then smallest lateral first."""
    cands = sorted(cands, key=lambda c: (c["gap"], c["lat"]))
    used: set[int] = set()
    keep = []
    for c in cands:
        if c["k"] in used or c["k2"] in used:
            continue
        used.add(c["k"])
        used.add(c["k2"])
        keep.append(c)
    return keep


# ───────────────────────── intersection test ─────────────────────────

def _clip_to_disc(ln: Line, cx: float, cy: float, R: float):
    """Portions of each segment inside the disc, as (t0, t1) in [0,1] per segment.

    Exact segment/circle clipping, so it works on sparse polylines whose
    vertices all lie outside the disc (the vertex-only test misses those).
    """
    A = ln[:-1] - (cx, cy)
    D = np.diff(ln, axis=0)
    a = (D ** 2).sum(1)
    b = 2 * (A * D).sum(1)
    c = (A ** 2).sum(1) - R * R
    disc = b * b - 4 * a * c
    ok = (disc >= 0) & (a > 0)
    t0 = np.full(len(a), np.nan)
    t1 = np.full(len(a), np.nan)
    s = np.sqrt(np.where(ok, disc, 0))
    t0[ok] = np.clip((-b[ok] - s[ok]) / (2 * a[ok]), 0, 1)
    t1[ok] = np.clip((-b[ok] + s[ok]) / (2 * a[ok]), 0, 1)
    good = ok & (t1 > t0)
    return good, t0, t1, D


def seg_len_in_disc(ln: Line, cx: float, cy: float, R: float) -> float:
    """Length of the polyline that lies inside the disc."""
    good, t0, t1, D = _clip_to_disc(ln, cx, cy, R)
    if not good.any():
        return 0.0
    seg = np.linalg.norm(D[good], axis=1)
    return float((seg * (t1[good] - t0[good])).sum())


def line_dir_near(ln: Line, cx: float, cy: float, R: float):
    """Direction of the polyline across the disc (entry point → exit point)."""
    good, t0, t1, D = _clip_to_disc(ln, cx, cy, R)
    idx = np.where(good)[0]
    if len(idx) == 0:
        return None
    p_in = ln[idx[0]] + t0[idx[0]] * D[idx[0]]
    p_out = ln[idx[-1]] + t1[idx[-1]] * D[idx[-1]]
    v = p_out - p_in
    n = np.hypot(*v)
    return v / n if n > 0 else None


def is_crossing(p: np.ndarray, q: np.ndarray, background: Sequence[Line],
                radius_m: float = 30.0, ang: float = 30.0, min_len_m: float = 3.0) -> bool:
    """True if the gap p→q passes through an intersection.

    Test: within `radius_m` of the gap midpoint, some background line (the full
    car-lane centerline set) runs at >= `ang` degrees to the gap direction for
    at least `min_len_m`. If so, a straight join would create no node there
    and the crossing must be handled by `gaps.intersections`.
    """
    cx, cy = (p[0] + q[0]) / 2, (p[1] + q[1]) / 2
    u = np.asarray(q, float) - np.asarray(p, float)
    n = np.hypot(*u)
    if n == 0:
        return False
    u = u / n
    for ln in background:
        if (ln[:, 0].max() < cx - radius_m or ln[:, 0].min() > cx + radius_m or
                ln[:, 1].max() < cy - radius_m or ln[:, 1].min() > cy + radius_m):
            continue
        if seg_len_in_disc(ln, cx, cy, radius_m) < min_len_m:
            continue
        v = line_dir_near(ln, cx, cy, radius_m)
        if v is None:
            continue
        if axial_diff_deg(u, v) >= ang:
            return True
    return False


# ───────────────────────── chaining ─────────────────────────

def build_chains(n_lines: int, pairs: list[dict]):
    """Endpoint pairs → chain adjacency. Each endpoint used once, no loops.

    `pairs` items need i, j, wi, wj, gap. Returns (nbr, applied) where
    nbr[(i, w)] = ((j, wj), gap).
    """
    parent = list(range(n_lines))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    nbr, endused, applied = {}, set(), []
    for pr in sorted(pairs, key=lambda p: p["gap"]):
        ki, kj = (pr["i"], pr["wi"]), (pr["j"], pr["wj"])
        if ki in endused or kj in endused:
            continue
        if find(pr["i"]) == find(pr["j"]):
            continue
        endused.add(ki)
        endused.add(kj)
        parent[find(pr["j"])] = find(pr["i"])
        nbr[ki] = (kj, pr["gap"])
        nbr[kj] = (ki, pr["gap"])
        applied.append(pr)
    return nbr, applied


def walk_chains(lines: Sequence[Line], nbr: dict):
    """Follow `nbr` from every free end. Returns [(seq, gaps)] where seq is a
    list of (line_index, oriented_geometry) and gaps the joined distances."""
    visited: set[int] = set()
    out = []
    starts = [(i, w) for i in range(len(lines)) for w in (0, 1) if (i, w) not in nbr]
    for si, sw in starts:
        if si in visited:
            continue
        seq, gaps = [], []
        i, w = si, sw
        while True:
            visited.add(i)
            g = lines[i] if w == 0 else lines[i][::-1]
            seq.append((i, g))
            nx = nbr.get((i, 1 - w))
            if nx is None:
                break
            (j, wj), gap = nx
            if j in visited:
                break
            gaps.append(gap)
            i, w = j, wj
        out.append((seq, gaps))
    for i in range(len(lines)):
        if i not in visited:
            out.append(([(i, lines[i])], []))
            visited.add(i)
    return out


# ───────────────────────── connectivity ─────────────────────────

def n_components(lines: Sequence[Line], snap_m: float = 1.5) -> int:
    import networkx as nx
    G = nx.MultiGraph()
    for ln in lines:
        G.add_edge((round(ln[0][0] / snap_m) * snap_m, round(ln[0][1] / snap_m) * snap_m),
                   (round(ln[-1][0] / snap_m) * snap_m, round(ln[-1][1] / snap_m) * snap_m))
    return nx.number_connected_components(G)
