"""Typed pedestrian-network graph (links + nodes) and typed polygons — stage 2 library.

OBJECTIVE
  From the connected network mask and the cleaned class map, build the
  centerline graph: sidewalk skeleton -> dangling-branch removal -> crosswalk
  axes and pseudo connectors joined in -> links split at type changes and
  merged through pass-through nodes -> gentle straightening -> typed nodes;
  plus per-link widths and the typed polygons.

INPUT / OUTPUT
  numpy arrays (masks, class maps, skeletons) and Python lists/dicts:
  links {'id', 'type', 'geom' [(row, col), ...], 'length_m', 'from_node',
  'to_node', 'avg_width_m', 'min_width_m'}, nodes {'id', 'rc', 'node_type',
  'degree', 'link_types'}, polygons as shapely geometries. build_network.py
  converts them to GeoJSON. No file I/O here.

FUNCTIONS (in call order)
  build_walk_skeleton(net, cls_snap, cw_mask, res, min_spur_m=3.0,
                      on_cw_frac=0.6)   skeletonize the mask with crosswalk
      pixels removed, prune spurs < min_spur_m, drop skeleton links lying
      mostly (> on_cw_frac) on crosswalk pixels; also returns the walkable
      reference mask used for proximity tests and widths.
  repair_skeleton_after_erase(skel, gone, res, min_spur_m)   cut the skeleton
      where midblock false positives were erased; remove orphan branches.
  remove_dangles(skel, keep_zone, res, dangle_max_m=8.0)   delete dangling
      branches shorter than dangle_max_m unless their free end lies inside
      keep_zone (near a crosswalk, an entrance, or the coverage edge).
  build_links(skel_net, segments, res, max_pseudo_m=4.0)   trace the skeleton
      into links; add each crossing's axis as a crosswalk link; add pseudo
      links (<= max_pseudo_m) from crosswalk ends to the nearest skeleton pixel.
  finalize_links(links, cls_snap, res, smooth_sigma_m=1.0, dp_tol_m=0.5,
                 dp_tol_min_m=0.15, min_run_m=1.0)   type each link from the
      class map, split at type changes (runs >= min_run_m), merge same-type
      links through degree-2 nodes, smooth sidewalk geometry (Gaussian sigma +
      straightness-adaptive Douglas-Peucker), split sidewalks at midblock
      entrances, rebuild typed nodes (endpoint / junction / type_change).
  attach_widths(links, walk_mask, cw_union, res, end_trim_m=1.5)   avg / min
      width per link from the distance transform (2 x distance to the edge),
      ignoring end_trim_m at both ends.
  build_typed_polygons(net, cls_snap, ..., smooth_m=0.20, simplify_m=0.30)
      vectorize sidewalk / midblock areas from the class map and one polygon
      per unique crossing; smooth by buffering in/out, then simplify.

TUNING — parameters that matter
  min_spur_m, dangle_max_m, max_pseudo_m, smooth_m, simplify_m are exposed as
  build_network.py flags. Internal: on_cw_frac=0.6; smooth_sigma_m=1.0 (larger
  = straighter sidewalks); dp_tol_m=0.5 / dp_tol_min_m=0.15 (simplification
  tolerance for straight / curved runs); min_run_m=1.0 (shorter type runs are
  absorbed into their neighbours); end_trim_m=1.5 (width measurement ignores
  the link ends, where the mask flares at junctions).
"""
from collections import Counter, defaultdict

import numpy as np
from scipy.ndimage import (binary_closing, binary_dilation, binary_erosion,
                           label as cc_label, distance_transform_edt)

from skeleton_ops import (prune_spurs, remove_redundant_pixels, trace_skeleton,
                          path_length_px, simplify_path, classify_link, CLS_TO_TYPE)
from crosswalks import STRUCT8

TYPE_CODE = {'sidewalk': 1, 'midblock': 2, 'crosswalk': 3, 'pseudo': 4}
CODE_TYPE = {v: k for k, v in TYPE_CODE.items()}


# =============================================================================
# 3b-B/C. Sidewalk skeleton (crosswalk removed), stray links dropped
# =============================================================================
def type_walk_paths(paths, cls_snap):
    typed = []
    for p in paths:
        t, frac = classify_link(p, cls_snap)
        typed.append((p, 'sidewalk' if t == 'crosswalk' else t, frac))
    return typed


def build_walk_skeleton(net, cls_snap, cw_mask, res, min_spur_m=3.0, on_cw_frac=0.6):
    """Skeleton of the non-crosswalk network; links lying mostly on crosswalk
    pixels removed. Returns (skel_walk, typed_walk, sidewalk_ref)."""
    from skimage.morphology import skeletonize
    cw_snap = cls_snap == 4
    walk_mask = binary_closing(net & ~cw_snap, structure=STRUCT8, iterations=2) & net
    skel_walk = skeletonize(walk_mask)
    skel_walk = prune_spurs(skel_walk, max(int(min_spur_m / res), 1))
    typed_walk = type_walk_paths(trace_skeleton(skel_walk), cls_snap)

    cw_dil = binary_dilation(cw_mask, structure=STRUCT8, iterations=2)
    skel_c = skel_walk.copy()
    n_removed = 0
    for p, t, frac in typed_walk:
        if np.mean([cw_dil[r, c] for r, c in p]) >= on_cw_frac:
            n_removed += 1
            for r, c in p[1:-1]:               # keep end nodes shared with neighbours
                skel_c[r, c] = False
    skel_walk = prune_spurs(remove_redundant_pixels(skel_c),
                            max(int(min_spur_m / res), 1))
    typed_walk = type_walk_paths(trace_skeleton(skel_walk), cls_snap)

    sidewalk_ref = np.zeros_like(cw_mask, dtype=bool)
    for p, t, _ in typed_walk:
        if t == 'sidewalk':
            for r, c in p:
                sidewalk_ref[r, c] = True
    return skel_walk, typed_walk, n_removed, sidewalk_ref, walk_mask


# =============================================================================
# 3d (skeleton part). Repair after erasing midblock-on-crosswalk components
# =============================================================================
def repair_skeleton_after_erase(skel_walk, gone, res, min_spur_m=3.0,
                                orphan_max_m=15.0):
    if not gone.any():
        return skel_walk, 0, 0
    gone_near = binary_dilation(gone, STRUCT8, iterations=2)
    n_cut = int((skel_walk & gone_near).sum())
    skel_walk = remove_redundant_pixels(skel_walk & ~gone_near)
    tip_zone = binary_dilation(gone, STRUCT8, iterations=6)
    n_branch = 0
    for _ in range(3):
        paths = trace_skeleton(skel_walk)
        ends = Counter()
        for p in paths:
            ends[p[0]] += 1; ends[p[-1]] += 1
        removed_any = False
        for p in paths:
            free = [q for q in (p[0], p[-1]) if ends[q] == 1]
            if free and any(tip_zone[q] for q in free) \
                    and path_length_px(p) * res <= orphan_max_m:
                for q in p[1:-1]:
                    skel_walk[q] = False
                for q in free:
                    skel_walk[q] = False
                n_branch += 1; removed_any = True
        skel_walk = remove_redundant_pixels(skel_walk)
        if not removed_any:
            break
    skel_walk = prune_spurs(skel_walk, max(int(min_spur_m / res), 1))
    return skel_walk, n_cut, n_branch


# =============================================================================
# Section 6 (1). Dangling-branch removal
# =============================================================================
def remove_dangles(skel, keep_zone, res, dangle_max_m=8.0, iters=3):
    """A dangling link <= dangle_max_m whose free end is NOT in keep_zone
    (near crosswalk / midblock / coverage edge) is a branch artefact."""
    skel = skel.copy()
    n_removed = 0
    for _ in range(iters):
        paths = trace_skeleton(skel)
        ends = Counter()
        for p in paths:
            ends[p[0]] += 1; ends[p[-1]] += 1
        removed_any = False
        for p in paths:
            free = [q for q in (p[0], p[-1]) if ends[q] == 1]
            if not free or path_length_px(p) * res > dangle_max_m:
                continue
            if any(keep_zone[q] for q in free):
                continue
            for r, c in p[1:-1]:
                skel[r, c] = False
            for q in free:
                skel[q] = False
            n_removed += 1; removed_any = True
        skel = remove_redundant_pixels(skel)
        if not removed_any:
            break
    return skel, n_removed


# =============================================================================
# Section 6 (3)+(4). Combined typed skeleton -> links
# =============================================================================
def _nearest_skel_pixel(skel, r, c, max_px):
    """Nearest True pixel of skel within a (2*max_px+1) window, or None."""
    H, W = skel.shape
    r0 = max(r - max_px, 0); r1 = min(r + max_px + 1, H)
    c0 = max(c - max_px, 0); c1 = min(c + max_px + 1, W)
    ys, xs = np.nonzero(skel[r0:r1, c0:c1])
    if not ys.size:
        return None
    d2 = (ys + r0 - r) ** 2 + (xs + c0 - c) ** 2
    i = int(d2.argmin())
    if d2[i] > max_px ** 2:
        return None
    return int(ys[i] + r0), int(xs[i] + c0)


def build_links(skel_net, segments, res, max_pseudo_m=4.0):
    """Crosswalk axes + pseudo connectors rasterized into one typed skeleton,
    traced, split at type changes, same-type links merged through pass-through
    nodes. Returns (links, n_pseudo, n_unattached)."""
    from skimage.draw import line as draw_line
    H, W = skel_net.shape
    max_px = int(max_pseudo_m / res)
    type_r = np.zeros((H, W), np.uint8)
    type_r[skel_net] = 1
    skel_all = skel_net.copy()
    n_pseudo = n_unattached = 0
    for sid, s in segments.items():
        (ra, ca), (rb, cb) = [(int(round(r)), int(round(c))) for r, c in s['axis_rc']]
        ra = int(np.clip(ra, 0, H - 1)); ca = int(np.clip(ca, 0, W - 1))
        rb = int(np.clip(rb, 0, H - 1)); cb = int(np.clip(cb, 0, W - 1))
        rr, cc = draw_line(ra, ca, rb, cb)
        skel_all[rr, cc] = True; type_r[rr, cc] = 3
        for (r, c) in ((ra, ca), (rb, cb)):
            tgt = _nearest_skel_pixel(skel_net, r, c, max_px)
            if tgt is None:
                n_unattached += 1
                continue
            pr, pc = draw_line(r, c, tgt[0], tgt[1])
            newp = ~skel_all[pr, pc]               # never overwrite crosswalk / sidewalk
            skel_all[pr[newp], pc[newp]] = True
            type_r[pr[newp], pc[newp]] = 4
            n_pseudo += 1
    skel_all = remove_redundant_pixels(skel_all)
    type_r[~skel_all] = 0

    # trace and split at type changes (crosswalk -> pseudo -> sidewalk)
    links = []
    for p in trace_skeleton(skel_all):
        tps = [int(type_r[r, c]) for r, c in p]
        cuts = [0] + [i for i in range(1, len(p))
                      if tps[i] != tps[i - 1] and tps[i] and tps[i - 1]] + [len(p) - 1]
        for a, b in zip(cuts, cuts[1:]):
            seg = p[a:b + 1]
            if len(seg) < 2:
                continue
            t_vals = [tps[i] for i in range(a, b + 1) if tps[i]]
            t = Counter(t_vals).most_common(1)[0][0] if t_vals else 1
            links.append({'path': seg, 'type': CODE_TYPE[t]})

    # merge consecutive same-type links through degree-2 nodes
    # (endpoint-indexed: equivalent to the notebook's pairwise scan, but O(n))
    while True:
        ends = defaultdict(list)
        for i, l in enumerate(links):
            ends[l['path'][0]].append(i)
            ends[l['path'][-1]].append(i)
        merged_any = False
        used = set()
        for q, ids in ends.items():
            if len(ids) != 2:
                continue
            i, j = ids
            if i == j or i in used or j in used:
                continue
            if links[i]['type'] != links[j]['type']:
                continue
            pa = links[i]['path']
            pb = links[j]['path']
            if pa[0] == q:
                pa = pa[::-1]
            if pb[-1] == q:
                pb = pb[::-1]
            if pa[-1] != q or pb[0] != q:
                continue
            links[i] = {'path': pa + pb[1:], 'type': links[i]['type']}
            links[j] = None
            used.update((i, j))
            merged_any = True
        links = [l for l in links if l is not None]
        if not merged_any:
            break
    return links, n_pseudo, n_unattached


# =============================================================================
# Section 7. Gentle straightening + sidewalk/midblock split -> final links/nodes
# =============================================================================
def _poly_len_px(G):
    G = np.asarray(G, float)
    return float(np.sum(np.hypot(*np.diff(G, axis=0).T)))


def _smooth_link(path, res, sigma_m, dp_tol_m, dp_tol_min_m):
    """pixel chain -> float polyline (r, c). Endpoints (nodes) kept exactly."""
    from scipy.ndimage import gaussian_filter1d
    P = np.array(path, float)
    if len(P) < 6:
        return P
    sig = sigma_m / res
    S = np.c_[gaussian_filter1d(P[:, 0], sig, mode='nearest'),
              gaussian_filter1d(P[:, 1], sig, mode='nearest')]
    S[0], S[-1] = P[0], P[-1]
    chord = np.hypot(*(P[-1] - P[0]))
    length = np.sum(np.hypot(*np.diff(P, axis=0).T))
    straightness = float(chord / max(length, 1e-9))
    tol_m = dp_tol_min_m + (dp_tol_m - dp_tol_min_m) * np.clip((straightness - 0.7) / 0.3, 0, 1)
    simp = simplify_path([tuple(q) for q in S], tol_px=tol_m / res)
    return np.array(simp, float)


def _densify(G, step_px=1.0):
    G = np.asarray(G, float)
    out = [G[0]]
    for a, b in zip(G[:-1], G[1:]):
        n = max(int(np.ceil(np.hypot(*(b - a)) / step_px)), 1)
        out += [a + (b - a) * t for t in np.linspace(0, 1, n + 1)[1:]]
    return np.array(out)


def _smooth_runs(tps, min_px):
    """runs shorter than min_px take the type of the longer neighbouring run."""
    tps = list(tps)
    changed = True
    while changed:
        changed = False
        runs = []
        i = 0
        while i < len(tps):
            j = i
            while j + 1 < len(tps) and tps[j + 1] == tps[i]:
                j += 1
            runs.append([i, j, tps[i]])
            i = j + 1
        for k, (a, b, t) in enumerate(runs):
            if b - a + 1 >= min_px or len(runs) == 1:
                continue
            nb = [runs[k - 1] if k > 0 else None,
                  runs[k + 1] if k + 1 < len(runs) else None]
            nb = [x for x in nb if x]
            new = nb[0][2] if len(nb) == 1 or nb[0][2] == nb[1][2] else \
                (nb[0][2] if (nb[0][1] - nb[0][0]) >= (nb[1][1] - nb[1][0]) else nb[1][2])
            for i in range(a, b + 1):
                tps[i] = new
            changed = True
            break
    return tps


def finalize_links(links, cls_snap, res, smooth_sigma_m=1.0, dp_tol_m=0.5,
                   dp_tol_min_m=0.15, min_run_m=1.0):
    """Smooth sidewalk geometry; split sidewalk links at midblock-entrance
    runs; rebuild nodes. Returns (links_final, nodes_final)."""
    H, W = cls_snap.shape
    mb_mask = cls_snap == 3
    geoms = []
    for l in links:
        if l['type'] == 'sidewalk':
            G = _smooth_link(l['path'], res, smooth_sigma_m, dp_tol_m, dp_tol_min_m)
        else:
            G = np.array([l['path'][0], l['path'][-1]], float)
        geoms.append((l['type'], G))

    links_final = []
    min_run_px = max(int(min_run_m / res), 2)
    for t, G in geoms:
        if t != 'sidewalk' or not mb_mask.any():
            links_final.append({'type': t, 'geom': G})
            continue
        D = _densify(G)
        rr = np.clip(np.round(D[:, 0]).astype(int), 0, H - 1)
        cc = np.clip(np.round(D[:, 1]).astype(int), 0, W - 1)
        tps = _smooth_runs([2 if mb_mask[r, c] else 1 for r, c in zip(rr, cc)],
                           min_run_px)
        cuts = [0] + [i for i in range(1, len(D)) if tps[i] != tps[i - 1]] + [len(D) - 1]
        if len(cuts) == 2:
            links_final.append({'type': 'midblock' if tps[0] == 2 else 'sidewalk',
                                'geom': G})
            continue
        for a, b in zip(cuts, cuts[1:]):
            piece = D[a:b + 1]
            if len(piece) < 2:
                continue
            piece = np.array(simplify_path([tuple(q) for q in piece], tol_px=0.5), float)
            links_final.append({'type': 'midblock' if tps[a] == 2 else 'sidewalk',
                                'geom': piece})

    node_of, nodes_final = {}, []

    def _key(q):
        return (round(float(q[0]), 2), round(float(q[1]), 2))

    def _node(q):
        k = _key(q)
        if k not in node_of:
            node_of[k] = len(nodes_final)
            nodes_final.append({'id': len(nodes_final), 'rc': k, 'links': []})
        return node_of[k]

    for i, l in enumerate(links_final):
        l['id'] = i
        l['length_m'] = round(_poly_len_px(l['geom']) * res, 2)
        l['from_node'] = _node(l['geom'][0])
        l['to_node'] = _node(l['geom'][-1])
        nodes_final[l['from_node']]['links'].append(i)
        nodes_final[l['to_node']]['links'].append(i)
    for n in nodes_final:
        types = {links_final[i]['type'] for i in n['links']}
        n['degree'] = len(n['links'])
        n['link_types'] = sorted(types)
        n['node_type'] = ('endpoint' if n['degree'] == 1 else
                          'type_change' if len(types) > 1 else 'junction')
    return links_final, nodes_final


# =============================================================================
# Link widths from the surface masks (sidewalk + crosswalk links only)
# =============================================================================
def attach_widths(links_final, walk_mask, cw_union, res, end_trim_m=1.5):
    """avg_width_m / min_width_m per link, from the distance transform of the
    surface the link's centerline runs through: width = 2 * distance to the
    nearest mask edge, sampled along the link. The first/last ~end_trim_m are
    excluded from the statistics (at junctions and tips the centerline
    necessarily approaches the boundary, which would fake a small minimum).
    Midblock and pseudo links get None."""
    w_walk = distance_transform_edt(walk_mask).astype(np.float32)
    w_walk *= 2.0 * res
    w_cw = None
    if cw_union is not None and cw_union.any():
        w_cw = distance_transform_edt(cw_union).astype(np.float32)
        w_cw *= 2.0 * res
    H, W = walk_mask.shape
    trim = max(int(end_trim_m / res), 1)
    for l in links_final:
        if l['type'] not in ('sidewalk', 'crosswalk') or \
                (l['type'] == 'crosswalk' and w_cw is None):
            l['avg_width_m'] = l['min_width_m'] = None
            continue
        raster = w_walk if l['type'] == 'sidewalk' else w_cw
        D = _densify(l['geom'])
        rr = np.clip(np.round(D[:, 0]).astype(int), 0, H - 1)
        cc = np.clip(np.round(D[:, 1]).astype(int), 0, W - 1)
        w = raster[rr, cc]
        k = min(len(w) // 4, trim)
        core = w[k:len(w) - k] if len(w) - 2 * k >= 3 else w
        core = core[core > 0]              # samples off the mask don't count
        if not core.size:
            core = w[w > 0]
        if not core.size:
            l['avg_width_m'] = l['min_width_m'] = None
            continue
        l['avg_width_m'] = round(float(core.mean()), 2)
        l['min_width_m'] = round(float(core.min()), 2)
    return links_final


# =============================================================================
# Sections 4-5. Typed polygons (crosswalks one per segment) + simplification
# =============================================================================
def _smooth_buffer(g, r):
    out = g.buffer(r, join_style='mitre').buffer(-2 * r, join_style='mitre') \
           .buffer(r, join_style='mitre')
    return g if out.is_empty else out


def build_typed_polygons(net, cls_snap, lab_cw, dropped_far, cw_mask, cw_seg,
                         segments, transform, res, smooth_m=0.20, simplify_m=0.30):
    """Typed, simplified polygons. Sidewalk / midblock from the class map;
    crosswalk polygons one per split segment (carved gaps stay empty).
    Returns a list of {'poly_type', 'area_m2', 'segment_id', 'geometry'(shapely)}."""
    from rasterio import features as rio_features
    from rasterio.transform import Affine
    from shapely.geometry import shape

    net_poly = net & ~np.isin(lab_cw, dropped_far) if dropped_far else net
    tmap = CLS_TO_TYPE[cls_snap]
    tmap = np.where(net_poly, tmap, 0).astype(np.uint8)
    tmap[tmap == 3] = 0                       # discard the merged crosswalk class...
    tmap[cw_seg > 0] = 3                      # ...use the separated segments instead
    gap = net_poly & cw_mask & (cw_seg == 0)  # carved separation gaps stay empty

    unfilled = net_poly & (tmap == 0) & ~gap
    if unfilled.any() and (tmap > 0).any():
        _, (iy, ix) = distance_transform_edt(tmap == 0, return_indices=True)
        tmap = np.where(unfilled, tmap[iy, ix], tmap).astype(np.uint8)

    def vectorize(mask, t):
        return [shape(g) for g, _ in rio_features.shapes(mask.astype(np.uint8),
                                                         mask=mask, transform=t)]

    def refine(g):
        out = _smooth_buffer(g, smooth_m) if smooth_m > 0 else g
        return out.simplify(simplify_m, preserve_topology=True) if simplify_m > 0 else out

    polys = []
    for code, name in ((1, 'sidewalk'), (2, 'midblock')):
        for g in vectorize(tmap == code, transform):
            g = refine(g)
            if not g.is_empty:
                polys.append({'poly_type': name, 'segment_id': None,
                              'area_m2': round(g.area, 2), 'geometry': g})
    for sid in sorted(segments):
        rs, cs = segments[sid]['slice']
        t_local = transform * Affine.translation(cs.start, rs.start)
        for g in vectorize(cw_seg[rs, cs] == sid, t_local):
            g = refine(g)
            if not g.is_empty:
                polys.append({'poly_type': 'crosswalk', 'segment_id': sid,
                              'area_m2': round(g.area, 2), 'geometry': g})
    return polys
