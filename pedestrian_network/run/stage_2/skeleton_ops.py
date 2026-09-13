"""Skeleton primitives shared by stage 2 (library module).

OBJECTIVE
  Low-level operations on 1-px-wide binary skeletons: degree maps, removal of
  redundant pixels, spur pruning, tracing into paths between junctions, path
  length, Douglas-Peucker simplification, and typing a path from the class map.

INPUT / OUTPUT
  boolean skeleton arrays and lists of (row, col) paths; no file I/O.

FUNCTIONS
  skeleton_degrees(skel)            8-neighbour count per skeleton pixel
  remove_redundant_pixels(skel)     drop pixels whose removal keeps
                                    connectivity (cleans staircase corners)
  prune_spurs(skel, min_px)         iteratively delete end branches shorter
                                    than min_px (= --min-spur-m / res)
  trace_skeleton(skel)              walk the skeleton -> list of paths, each
                                    from a junction / endpoint to the next
  path_length_px(path)              polyline length in pixels
  simplify_path(path, tol_px=2.0)   Douglas-Peucker
  classify_link(path, cls)          majority class along the path -> link type
                                    via CLS_TO_TYPE (1, 2 -> sidewalk;
                                    3 -> midblock; 4 -> crosswalk) + class shares

TUNING
  CLS_TO_TYPE maps the 5 classes to link types: visible sidewalk and
  tree-occluded sidewalk both become 'sidewalk'. Pruning length and the
  simplification tolerance are chosen by the callers (build_network.py flags).
"""
import math

import numpy as np

NBR8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

# 5-class map -> link-type code: 0 none (background), 1 sidewalk (visible OR
# tree-occluded), 2 midblock (entrance), 3 crosswalk
CLS_TO_TYPE = np.array([0, 1, 1, 2, 3])
LINK_TYPE_NAMES = {1: 'sidewalk', 2: 'midblock', 3: 'crosswalk'}


def skeleton_degrees(skel):
    from scipy.ndimage import convolve
    k = np.ones((3, 3), np.uint8); k[1, 1] = 0
    nbr = convolve(skel.astype(np.uint8), k, mode='constant')
    return np.where(skel, nbr, 0)


def remove_redundant_pixels(skel, iters=3):
    """Remove skeleton pixels whose neighbors are mutually 8-connected without
    them (removal cannot disconnect anything).  Cleans the one-pixel stubs left
    by spur pruning and thick junction centers, which would otherwise act as
    fake junctions splitting straight edges in two."""
    skel = skel.copy()
    H, W = skel.shape
    for _ in range(iters):
        removed = False
        for r, c in np.argwhere(skel):
            nbrs = [(r + dr, c + dc) for dr, dc in NBR8
                    if 0 <= r + dr < H and 0 <= c + dc < W and skel[r + dr, c + dc]]
            if len(nbrs) < 2:
                continue                     # endpoints are never removed
            if len(nbrs) == 2:               # fast path: straight-line middle
                (r0, c0), (r1, c1) = nbrs
                if max(abs(r0 - r1), abs(c0 - c1)) > 1:
                    continue
                skel[r, c] = False
                removed = True
                continue
            # BFS over the neighbor cluster (adjacency = Chebyshev distance 1)
            seen, stack = {nbrs[0]}, [nbrs[0]]
            while stack:
                p = stack.pop()
                for q in nbrs:
                    if q not in seen and max(abs(p[0] - q[0]), abs(p[1] - q[1])) <= 1:
                        seen.add(q)
                        stack.append(q)
            if len(seen) == len(nbrs):
                skel[r, c] = False
                removed = True
        if not removed:
            break
    return skel


def prune_spurs(skel, min_px, iters=3):
    """Iteratively remove leaf branches shorter than min_px."""
    skel = skel.copy()
    for _ in range(iters):
        deg = skeleton_degrees(skel)
        ends = np.argwhere(skel & (deg == 1))
        removed_any = False
        for r, c in ends:
            path = [(r, c)]
            prev = None
            cur = (int(r), int(c))
            while len(path) <= min_px:
                nxt = [(cur[0] + dr, cur[1] + dc) for dr, dc in NBR8
                       if 0 <= cur[0] + dr < skel.shape[0]
                       and 0 <= cur[1] + dc < skel.shape[1]
                       and skel[cur[0] + dr, cur[1] + dc]
                       and (cur[0] + dr, cur[1] + dc) != prev]
                if len(nxt) != 1:
                    break                     # reached a junction (or isolated dot)
                prev, cur = cur, nxt[0]
                if deg[cur] != 2:
                    if deg[cur] == 1:         # far end of an isolated short segment
                        path.append(cur)      # -> removable together with the rest
                    break                     # junction itself is never removed
                path.append(cur)
            else:
                continue                      # branch longer than min_px -> keep
            if len(path) < min_px:
                for p in path:
                    skel[tuple(p)] = False
                removed_any = True
        skel = remove_redundant_pixels(skel)
        if not removed_any:
            break
    return skel


def trace_skeleton(skel):
    """Skeleton (bool HxW) -> list of pixel paths [(r,c), ...] between nodes.
    Nodes = endpoints / junctions (degree != 2); pure loops handled separately."""
    deg = skeleton_degrees(skel)
    H, W = skel.shape
    is_node = skel & (deg != 2)
    node_set = {tuple(p) for p in np.argwhere(is_node)}
    visited = np.zeros_like(skel, bool)     # corridor (deg==2) pixels only
    paths, edge_seen = [], set()

    def neighbors(p):
        r, c = p
        for dr, dc in NBR8:
            rr, cc = r + dr, c + dc
            if 0 <= rr < H and 0 <= cc < W and skel[rr, cc]:
                yield (rr, cc)

    for start in node_set:
        for nb in neighbors(start):
            if nb in node_set:
                key = (min(start, nb), max(start, nb))
                if key not in edge_seen:
                    edge_seen.add(key)
                    paths.append([start, nb])
                continue
            if visited[nb]:
                continue
            path, prev, cur = [start, nb], start, nb
            visited[nb] = True
            while True:
                nxt = [q for q in neighbors(cur)
                       if q != prev and (q in node_set or not visited[q])]
                if not nxt:
                    break
                q = nxt[0]
                path.append(q)
                if q in node_set:
                    break
                visited[q] = True
                prev, cur = cur, q
            paths.append(path)

    # closed loops with no node pixel at all
    for r, c in np.argwhere(skel & (deg == 2) & ~visited):
        if visited[r, c]:
            continue
        start = (int(r), int(c))
        path, prev, cur = [start], None, start
        visited[r, c] = True
        while True:
            nxt = [q for q in neighbors(cur) if q != prev and not visited[q]]
            if not nxt:
                path.append(start)          # close the ring
                break
            q = nxt[0]
            path.append(q)
            visited[q] = True
            prev, cur = cur, q
        paths.append(path)
    return paths


def path_length_px(path):
    return sum(math.hypot(b[0] - a[0], b[1] - a[1])
               for a, b in zip(path[:-1], path[1:]))


def simplify_path(path, tol_px=2.0):
    """Iterative Douglas-Peucker on pixel coords."""
    if len(path) < 3:
        return path
    pts = np.asarray(path, float)
    keep = np.zeros(len(pts), bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(pts) - 1)]
    while stack:
        i0, i1 = stack.pop()
        if i1 <= i0 + 1:
            continue
        seg = pts[i1] - pts[i0]
        L = np.hypot(*seg)
        if L == 0:
            d = np.hypot(*(pts[i0 + 1:i1] - pts[i0]).T)
        else:
            d = np.abs(np.cross(seg, pts[i0] - pts[i0 + 1:i1])) / L
        j = int(d.argmax())
        if d[j] > tol_px:
            k = i0 + 1 + j
            keep[k] = True
            stack.extend([(i0, k), (k, i1)])
    return [path[i] for i in np.nonzero(keep)[0]]


def classify_link(path, cls):
    """Majority link type along the centerline pixels + class-share fractions.

    Classes 1+2 vote 'sidewalk' (frac_occluded reports the tree share), 3 votes
    'midblock', 4 votes 'crosswalk'. Centerline pixels on background (areas in
    the mask only via the confidence threshold / gap closing) don't vote; a
    link with no votes at all defaults to 'sidewalk' with zero fractions.
    Ties go to the more specific type (crosswalk > midblock > sidewalk)."""
    rs = np.fromiter((p[0] for p in path), np.int64, len(path))
    cs = np.fromiter((p[1] for p in path), np.int64, len(path))
    c = np.bincount(cls[rs, cs], minlength=5)
    tot = float(c[1:].sum())
    if tot == 0:
        return 'sidewalk', {'frac_sidewalk': 0.0, 'frac_occluded': 0.0,
                            'frac_midblock': 0.0, 'frac_crosswalk': 0.0}
    votes = {'crosswalk': int(c[4]), 'midblock': int(c[3]),
             'sidewalk': int(c[1] + c[2])}
    link_type = max(votes, key=votes.get)
    return link_type, {'frac_sidewalk': round(float(c[1] + c[2]) / tot, 3),
                       'frac_occluded': round(float(c[2]) / tot, 3),
                       'frac_midblock': round(float(c[3]) / tot, 3),
                       'frac_crosswalk': round(float(c[4]) / tot, 3)}
