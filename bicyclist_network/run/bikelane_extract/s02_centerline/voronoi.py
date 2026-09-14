"""Class mask → lane centerline (06 v8, Voronoi-boundary definition).

A centerline is the set of pixels equidistant from two *different*
reconstructed markings, between DMIN and DMAX pixels from both. Defining it
this way (instead of "a fixed distance from one marking") means no ghost
line can appear outside the pavement: there is nothing on the other side to
be equidistant from.

Steps in `mask_to_centerline`:
  1. preclose        small morphological close per class
  2. build_lines     skeletonize → drop junction pixels (stroke labelling, so a
                     crosswalk touching a lane line does not swallow it) →
                     PCA per stroke → merge collinear strokes → binned-median
                     polyline fit → rasterise each marking with its own id
  3. owner map       distance transform with indices; each free pixel knows
                     its nearest marking
  4. boundary        pixels whose 4-neighbour has a different owner, within
                     [DMIN, DMAX], both owners parallel (ANG_TOL), not
                     curb–curb (that is the road middle, not a lane), and not
                     near a marking's tip (cap filter: kills closed hooks)
  5. skeletonize + drop fragments shorter than MIN_BRANCH

Curb is used as an *owner only*: it rescues roads with a single lane line but
never becomes a centerline by itself.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import skeletonize

S8 = np.ones((3, 3), np.uint8)


@dataclass
class VoronoiParams:
    close_k: int = 5            # preclose kernel (0 = off)
    minpix: int = 8             # min stroke length, lane lines (px)
    curb_minpix: int = 25       # min stroke length, curb (px)
    ang_merge: float = 15.0     # collinear merge: max heading difference (deg)
    lat_tol: float = 7.0        # collinear merge: max lateral offset (px)
    gap_max: float = 130.0      # collinear merge: max end-to-end gap (px)
    gaps: tuple = (130.0, 200.0, 280.0)   # multipass gaps (only if multipass)
    multipass: bool = False
    bin_w: float = 20.0         # polyline fit: bin width along the axis (px)
    thick: int = 1              # rasterised marking thickness
    dmin: int = 3               # boundary distance band (px); 3 px = 0.9 m half-width keeps bike lanes
    dmax: int = 20
    ang_tol: float = 30.0       # owners must be parallel within this (deg)
    min_branch: int = 30        # drop centerline fragments shorter than this (px)
    tip: int = 5                # cap filter radius around marking tips (px)
    border_exempt: bool = True  # tips touching the canvas border are not caps
    cap_filter: bool = True
    use_curb: bool = True

    @classmethod
    def from_dict(cls, d: dict | None) -> "VoronoiParams":
        d = d or {}
        names = {f.name for f in fields(cls)}
        kw = {k: (tuple(v) if k == "gaps" else v) for k, v in d.items() if k in names}
        return cls(**kw)


def angdiff(a, b):
    d = np.abs(a - b) % 180
    return np.minimum(d, 180 - d)


# ── 2. markings ──

def stroke_label(mask, min_len=8):
    """Skeletonize → remove junction pixels → label. Returns (lab, n)."""
    sk = skeletonize(mask > 0)
    deg = cv2.filter2D(sk.astype(np.uint8), -1, S8) - sk.astype(np.uint8)
    lab, n = ndi.label(sk & (deg <= 2), structure=S8)
    if n:
        sz = ndi.sum(np.ones_like(lab), lab, range(1, n + 1))
        keep = np.isin(lab, [i + 1 for i, s in enumerate(sz) if s >= min_len])
        lab, n = ndi.label(keep, structure=S8)
    return lab, n


def _pca(xs, ys):
    c = np.array([xs.mean(), ys.mean()])
    X = np.stack([xs - c[0], ys - c[1]])
    _, v = np.linalg.eigh(X @ X.T / len(xs))
    u = v[:, -1]
    s = X.T @ u
    return c, u, s


def comp_props(lab, minpix):
    props = {}
    for i, sl in enumerate(ndi.find_objects(lab), start=1):
        if sl is None:
            continue
        ys, xs = np.where(lab[sl] == i)
        if len(xs) < minpix:
            continue
        ys = ys + sl[0].start
        xs = xs + sl[1].start
        c, u, s = _pca(xs.astype(float), ys.astype(float))
        props[i] = dict(c=c, u=u, smin=s.min(), smax=s.max(), xs=xs, ys=ys,
                        th=np.degrees(np.arctan2(u[1], u[0])) % 180)
    return props


class _UF:
    def __init__(self, ids):
        self.p = {i: i for i in ids}

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def merge_collinear(props, gap_max, ang_merge, lat_tol):
    ids = list(props)
    if not ids:
        return []
    uf = _UF(ids)
    C = np.array([props[i]["c"] for i in ids])
    for ii in range(len(ids)):
        near = np.where(np.linalg.norm(C - C[ii], axis=1) <= gap_max + 200)[0]
        for jj in near:
            if jj <= ii:
                continue
            a, b = props[ids[ii]], props[ids[jj]]
            if angdiff(a["th"], b["th"]) > ang_merge:
                continue
            ua = a["u"]
            va = np.array([-ua[1], ua[0]])
            d = b["c"] - a["c"]
            if abs(d @ va) > lat_tol:
                continue
            ub = b["u"]
            vb = np.array([-ub[1], ub[0]])
            if abs((a["c"] - b["c"]) @ vb) > lat_tol:
                continue
            sb = d @ ua
            ia = (a["smin"], a["smax"])
            ib = (sb + b["smin"], sb + b["smax"])
            if max(ib[0] - ia[1], ia[0] - ib[1], 0.0) > gap_max:
                continue
            uf.union(ids[ii], ids[jj])
    g = {}
    for i in ids:
        g.setdefault(uf.find(i), []).append(i)
    return list(g.values())


def fit_polyline(props, g, binw):
    """Binned-median fit along the group's principal axis, interpolated at 1 px.
    (Replaced deg-2 polyfit, which was 8.7 m off on S-curves.)"""
    xs = np.concatenate([props[i]["xs"] for i in g]).astype(float)
    ys = np.concatenate([props[i]["ys"] for i in g]).astype(float)
    c, u, s = _pca(xs, ys)
    vv = np.array([-u[1], u[0]])
    t = np.stack([xs - c[0], ys - c[1]]).T @ vv
    edges = np.arange(s.min(), s.max() + binw, binw)
    if len(edges) < 2:
        return None, None
    idx = np.clip(np.digitize(s, edges) - 1, 0, len(edges) - 2)
    sm, tm = [], []
    for b in range(len(edges) - 1):
        m = idx == b
        if m.sum() < 2:
            continue
        sm.append(np.median(s[m]))
        tm.append(np.median(t[m]))
    if len(sm) < 2:
        return None, None
    sm, tm = np.array(sm), np.array(tm)
    ss = np.arange(sm.min(), sm.max() + 1.0, 1.0)
    tt = np.interp(ss, sm, tm)
    pts = c[None, :] + ss[:, None] * u[None, :] + tt[:, None] * vv[None, :]
    return pts, np.degrees(np.arctan2(u[1], u[0])) % 180


def preclose(pred, k):
    if not k:
        return pred
    out = pred.copy()
    el = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    for cls in (1, 2):
        m = cv2.morphologyEx((pred == cls).astype(np.uint8), cv2.MORPH_CLOSE, el)
        out[(m > 0) & (out == 0)] = cls
    return out


def group_props(props, groups):
    out = {}
    for gi, g in enumerate(groups, 1):
        xs = np.concatenate([props[i]["xs"] for i in g]).astype(float)
        ys = np.concatenate([props[i]["ys"] for i in g]).astype(float)
        c, u, s = _pca(xs, ys)
        out[gi] = dict(c=c, u=u, smin=s.min(), smax=s.max(),
                       xs=xs.astype(int), ys=ys.astype(int),
                       th=np.degrees(np.arctan2(u[1], u[0])) % 180)
    return out


def build_lines(pred, P: VoronoiParams):
    """Rasterise reconstructed markings. Returns (lines id-map, theta, kind, tips)."""
    lines = np.zeros(pred.shape, np.int32)
    theta, kind, tips = [np.nan], [0], [None]
    gid = 0
    specs = [(1, P.minpix)] + ([(2, P.curb_minpix)] if P.use_curb else [])
    for cls, minpix in specs:
        lab, _ = stroke_label(pred == cls, min_len=minpix)
        props = comp_props(lab, minpix)
        if not props:
            continue
        if P.multipass:
            cur = props
            for gmax in P.gaps:
                cur = group_props(cur, merge_collinear(cur, gmax, P.ang_merge, P.lat_tol))
            src, groups = cur, [[i] for i in cur]
        else:
            src, groups = props, merge_collinear(props, P.gap_max, P.ang_merge, P.lat_tol)
        for g in groups:
            pts, th = fit_polyline(src, g, P.bin_w)
            if pts is None:
                continue
            gid += 1
            cv2.polylines(lines, [pts.round().astype(np.int32).reshape(-1, 1, 2)],
                          False, int(gid), P.thick)
            theta.append(th)
            kind.append(cls)
            tips.append((pts[0], pts[-1]))
    return lines, np.array(theta), np.array(kind), tips


def drop_short(sk, min_branch):
    l, n = ndi.label(sk, structure=S8)
    if n == 0:
        return sk
    sz = ndi.sum(np.ones_like(l), l, range(1, n + 1))
    return np.isin(l, [i + 1 for i, s in enumerate(sz) if s >= min_branch])


# ── 3–5. centerline ──

def mask_to_centerline(pred, P: VoronoiParams | None = None, debug=False):
    """Class mask (0 bg / 1 lane / 2 curb) → bool centerline mask.

    With debug=True also returns (lines, theta, kind) for inspection.
    """
    P = P or VoronoiParams()
    empty = np.zeros(pred.shape, bool)
    H, W = pred.shape
    lines, theta, kind, tips = build_lines(preclose(pred, P.close_k), P)
    ng = int(lines.max())
    if ng < 2:
        return (empty, (lines, theta, kind)) if debug else empty

    free = lines == 0
    D, (iy, ix) = ndi.distance_transform_edt(free, return_indices=True)
    owner = lines[iy, ix]

    A = np.zeros(owner.shape, np.int32)
    B = np.zeros(owner.shape, np.int32)
    NA = np.zeros(owner.shape + (2,), np.int32)
    NB = np.zeros(owner.shape + (2,), np.int32)
    NN = np.stack([iy, ix], -1)

    def put(mask, sa, sb):
        A[sa][mask] = owner[sa][mask]
        B[sa][mask] = owner[sb][mask]
        NA[sa][mask] = NN[sa][mask]
        NB[sa][mask] = NN[sb][mask]

    sh, sh2 = (slice(None), slice(None, -1)), (slice(None), slice(1, None))
    put(owner[sh] != owner[sh2], sh, sh2)
    sv, sv2 = (slice(None, -1), slice(None)), (slice(1, None), slice(None))
    put((owner[sv] != owner[sv2]) & (A[sv] == 0), sv, sv2)

    bnd = (A > 0) & (B > 0) & free & (D >= P.dmin) & (D <= P.dmax)

    ka = kind[np.clip(A, 0, ng)]
    kb = kind[np.clip(B, 0, ng)]
    bnd &= ~((ka == 2) & (kb == 2))                     # curb–curb = road middle

    ta = theta[np.clip(A, 0, ng)]
    tb = theta[np.clip(B, 0, ng)]
    bnd &= np.isfinite(ta) & np.isfinite(tb) & (angdiff(ta, tb) <= P.ang_tol)

    if P.cap_filter:
        tm = np.zeros(pred.shape, np.uint8)
        M = P.tip + 2
        for gi in range(1, ng + 1):
            if tips[gi] is None:
                continue
            for q in tips[gi]:
                x, y = int(round(q[0])), int(round(q[1]))
                if P.border_exempt and (x < M or y < M or x >= W - M or y >= H - M):
                    continue
                cv2.circle(tm, (x, y), P.tip, 1, -1)
        tm = tm.astype(bool)
        bnd &= ~(tm[NA[..., 0], NA[..., 1]] | tm[NB[..., 0], NB[..., 1]])

    sk = drop_short(skeletonize(bnd), P.min_branch)
    return (sk, (lines, theta, kind)) if debug else sk


def count_frag(sk):
    return ndi.label(sk, structure=S8)[1]
