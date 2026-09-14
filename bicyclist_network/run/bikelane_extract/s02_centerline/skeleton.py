"""Skeleton (bool mask) → list of (N, 2) pixel polylines, x/y order.

Direction-first tracing with a U-turn stop: at each pixel the next pixel is
the one that deviates least from the current heading; if the best option
turns more than `u_turn_ang`, the trace ends. Starting from every node
(endpoint or junction pixel) and then sweeping unvisited pixels catches
pure loops.

Known limitation (deferred): at a junction the tracer picks the straightest
continuation, so an overpass crossing can be split arbitrarily.
"""
from __future__ import annotations

import cv2
import numpy as np


def _dir(a, b):
    v = (b[0] - a[0], b[1] - a[1])
    n = np.hypot(*v)
    return (v[0] / n, v[1] / n) if n > 0 else (0.0, 0.0)


def _ang(d1, d2):
    c = max(-1.0, min(1.0, d1[0] * d2[0] + d1[1] * d2[1]))
    return np.degrees(np.arccos(c))


def skeleton_to_polylines(skel, simplify_eps: float = 2.0, u_turn_ang: float = 100.0):
    sk = skel.astype(np.uint8)
    H, W = sk.shape
    kern = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], np.uint8)
    nb = cv2.filter2D(sk, -1, kern) * sk
    nodes = sk & ((nb == 1) | (nb >= 3))
    node_pts = set(map(tuple, np.argwhere(nodes)))
    visited = np.zeros_like(sk, bool)
    polylines = []

    def neigh(y, x):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                ny, nx_ = y + dy, x + dx
                if 0 <= ny < H and 0 <= nx_ < W and sk[ny, nx_]:
                    yield (ny, nx_)

    def trace(sy, sx, ny, nx_):
        path = [(sy, sx), (ny, nx_)]
        visited[ny, nx_] = True
        cy, cx = ny, nx_
        prev = (sy, sx)
        while (cy, cx) not in node_pts:
            cand = [p for p in neigh(cy, cx) if p != prev and not visited[p]]
            if not cand:
                break
            cur = _dir(prev, (cy, cx))
            best = min(cand, key=lambda p: _ang(cur, _dir((cy, cx), p)))
            if _ang(cur, _dir((cy, cx), best)) > u_turn_ang:
                break
            prev = (cy, cx)
            cy, cx = best
            visited[cy, cx] = True
            path.append((cy, cx))
        return path

    def finalize(path):
        if len(path) >= 2:
            pts = np.array([[x, y] for y, x in path], np.float32)
            if simplify_eps > 0 and len(pts) > 2:
                pts = cv2.approxPolyDP(pts.reshape(-1, 1, 2), simplify_eps, False).reshape(-1, 2)
            polylines.append(pts)

    for sy, sx in node_pts:
        for ny, nx_ in neigh(sy, sx):
            if not visited[ny, nx_]:
                finalize(trace(sy, sx, ny, nx_))
    rem = set(map(tuple, np.argwhere(sk))) - set(map(tuple, np.argwhere(visited)))
    for sy, sx in rem:
        if visited[sy, sx]:
            continue
        visited[sy, sx] = True
        un = [p for p in neigh(sy, sx) if not visited[p]]
        if un:
            finalize(trace(sy, sx, *un[0]))
    return polylines
