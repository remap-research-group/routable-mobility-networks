"""Crosswalk treatment for stage 2 (library module used by build_network.py).

OBJECTIVE
  Turn the model's crosswalk pixels (class 4) into clean, individually
  separated crossings with a reliable axis, and remove crosswalk-related false
  positives, so that stage 2 can draw one link and one polygon per crossing.

INPUT / OUTPUT
  numpy arrays only: uint8 5-class maps, boolean masks, labelled regions;
  `res` is the pixel size in metres. Nothing is read or written here.

FUNCTIONS (in the order build_network.py calls them)
  clean_crosswalks_v2(cls, res, ...)   per chunk: morphological close then open
      of the crosswalk class, fill enclosed holes < max_hole_m2, reassign specks
      smaller than min_area_m2 (per class) to the surrounding class, regularize
      crosswalk blobs to their bounding rectangle when they fill at least
      rect_min_fill of it. Returns (map, stats).
  seal_crosswalk_seams(cls, max_gap_px=6)   fill the thin background band the
      model leaves between a crosswalk and the adjacent walkable surface.
  build_crosswalk_mask(cls_raw, res, max_hole_m2=1.5, min_area_m2=2.0)
      crosswalk regions from the RAW (un-snapped) class map -> (mask, labels, n).
  filter_by_sidewalk(lab_cw, n_cw, sidewalk_ref, res, buffer_m=5.0)
      keep only regions within buffer_m of a sidewalk centerline -> (kept ids,
      dropped ids). Isolated blobs in the roadway are dropped.
  split_crosswalk_regions(lab_cw, near_ids, res, params=DEFAULT_SPLIT)
      one region can hold several crossings meeting at a corner: adaptive
      erosion finds persistent seeds, a watershed grows them back, a 2-px gap
      is forced between segments; recursive up to max_depth. Returns
      (segment map, {id: {'slice', 'axis', ...}}, n_splits).
  correct_axes(segments, cls_snap, res, edge_sw_m=2.0, min_edge_m=1.5,
               align_tol_deg=12.0)   the PCA axis of a short, wide crossing can
      be wrong; re-derive the heading from the segment's own road edges (where
      it meets walkable surface) when they disagree by more than align_tol_deg.
  erase_midblock_on_crosswalk(cls_snap, net, cw_mask, touch_px=3)
      midblock-entrance components touching a crosswalk are false positives:
      erased from the class map and the network mask.

TUNING — parameters that matter
  clean_crosswalks_v2: cw_close_px=4, cw_open_px=2 (morphology radii in px),
      max_hole_m2=1.5, min_area_m2 per class {1: 1.5, 2: 1.0, 3: 1.5, 4: 2.5},
      rect_min_fill=0.80 (regularize only near-rectangular blobs)
  seal_crosswalk_seams: max_gap_px=6 (widest seam bridged, ~0.5 m)
  filter_by_sidewalk: buffer_m (exposed as --buffer-m in build_network.py)
  DEFAULT_SPLIT: erosion_step_m=0.10, persist_m=0.20 (a seed must survive
      0.2 m of erosion to count as its own crossing), min_seed_area_m2=0.75,
      separation_px=2, min_segment_m2=1.0, straight_fill=0.78, max_depth=8
  correct_axes: align_tol_deg=12 (smaller = correct more axes)
  All metric defaults assume 0.08 m/px imagery.
"""
import numpy as np
from scipy.ndimage import (label as cc_label, generate_binary_structure,
                           binary_fill_holes, binary_dilation, binary_erosion,
                           binary_opening, binary_closing, distance_transform_edt)

STRUCT8 = generate_binary_structure(2, 2)


# =============================================================================
# 1a. Crosswalk clean-up v2 (per tile, BEFORE connect_crosswalks)
# =============================================================================
def _reassign_small_components(out, classes, min_area_px, ring_px=2, stats=None):
    """Every component of `classes` smaller than min_area_px[c] takes the majority
    class of the ring around it (crosswalk edge specks -> crosswalk, isolated
    specks -> background)."""
    for c in classes:
        m = out == c
        lab, n = cc_label(m, structure=STRUCT8)
        if not n:
            continue
        sizes = np.bincount(lab.ravel())
        small_ids = np.nonzero(sizes < min_area_px[c])[0]
        small_ids = small_ids[small_ids > 0]
        for k in small_ids:
            comp = lab == k
            ring = binary_dilation(comp, STRUCT8, iterations=ring_px) & ~comp
            vals = out[ring]
            vals = vals[vals != c]                       # never vote for itself
            new = np.bincount(vals, minlength=5).argmax() if vals.size else 0
            out[comp] = new
            if stats is not None:
                stats[f'speck_{c}->{int(new)}'] = stats.get(f'speck_{c}->{int(new)}', 0) + 1
    return out


def _regularize_rectangles(out, cw_class=4, min_fill=0.80, min_area_px=0, stats=None):
    """Replace each crosswalk component that fills >= min_fill of its minimum
    rotated rectangle with that rectangle (painted over background and
    small-class pixels only)."""
    from skimage.draw import polygon as draw_polygon
    from shapely.geometry import MultiPoint
    cw = out == cw_class
    lab, n = cc_label(cw, structure=STRUCT8)
    for k in range(1, n + 1):
        comp = lab == k
        area = int(comp.sum())
        if area < min_area_px:
            continue
        edge = comp & ~binary_erosion(comp, STRUCT8)   # outline is enough for the envelope
        ys, xs = np.nonzero(edge)
        rect = MultiPoint(np.c_[xs, ys]).oriented_envelope
        if rect.area <= 0 or area / rect.area < min_fill:
            continue
        rx, ry = rect.exterior.xy
        rr, cc = draw_polygon(np.asarray(ry), np.asarray(rx), shape=out.shape)
        paint = np.zeros_like(cw); paint[rr, cc] = True
        paint &= ~np.isin(out, (1, 2, 3))              # never overwrite real walkable surface
        out[paint] = cw_class
        if stats is not None:
            stats['rect_regularized'] = stats.get('rect_regularized', 0) + 1
    return out


def clean_crosswalks_v2(cls, res,
                        cw_close_px=4,          # join fragments / fill notches up to ~2*r px
                        cw_open_px=2,           # shave protrusions thinner than ~2*r px
                        max_hole_m2=1.5,        # enclosed holes smaller than this -> crosswalk
                        min_area_m2={1: 1.5, 2: 1.0, 3: 1.5, 4: 2.5},
                        ring_px=2,
                        regularize=True, rect_min_fill=0.80,
                        cw_class=4):
    """Returns (new_cls, stats)."""
    out = cls.copy()
    st = {}
    px = 1.0 / (res * res)
    min_area_px = {c: int(a * px) for c, a in min_area_m2.items()}

    # (1) crosswalk shape: close (join fragments, fill notches) then open.
    #     Pixels ADDED by closing only overwrite background / tree; pixels
    #     REMOVED by opening -> background.
    cw = out == cw_class
    closed = binary_closing(cw, STRUCT8, iterations=cw_close_px) if cw_close_px else cw
    add = closed & ~cw & np.isin(out, (0, 2))
    out[add] = cw_class
    cw = out == cw_class
    opened = binary_opening(cw, STRUCT8, iterations=cw_open_px) if cw_open_px else cw
    out[cw & ~opened] = 0
    st['cw_px_added_by_close'] = int(add.sum())
    st['cw_px_removed_by_open'] = int((cw & ~opened).sum())

    # (2) fill small enclosed holes (any class) inside crosswalk components
    cw = out == cw_class
    holes = binary_fill_holes(cw, structure=STRUCT8) & ~cw
    hl, nh = cc_label(holes, structure=STRUCT8)
    if nh:
        sizes = np.bincount(hl.ravel())
        keep = sizes <= max_hole_m2 * px; keep[0] = False
        out[keep[hl]] = cw_class
        st['holes_filled'] = int(keep[1:].sum())

    # (3) small components of ANY class -> majority of surrounding ring
    out = _reassign_small_components(out, [cw_class, 1, 3, 2], min_area_px, ring_px, st)

    # (4) optional: snap near-rectangular crosswalks to their rotated rectangle
    if regularize:
        out = _regularize_rectangles(out, cw_class, rect_min_fill, min_area_px[cw_class], st)
    return out, st


def seal_crosswalk_seams(cls, max_gap_px=6, cw_class=4, walk_classes=(1, 2, 3), stats=None):
    """Fill the background band between a crosswalk and adjacent walkable
    surface. Each gap pixel takes the class of the nearer side. Gaps wider than
    max_gap_px are left for connect_crosswalks() to bridge with a connector."""
    out  = cls.copy()
    cw   = out == cw_class
    walk = np.isin(out, walk_classes)
    if not cw.any() or not walk.any():
        return out, stats
    d_cw            = distance_transform_edt(~cw)
    d_w, (wy, wx)   = distance_transform_edt(~walk, return_indices=True)
    # "between" test: close to both sides AND the two distances roughly add up
    band = (out == 0) & (d_cw <= max_gap_px) & (d_w <= max_gap_px) \
                      & (d_cw + d_w <= max_gap_px + 1.5)
    to_cw   = band & (d_cw <= d_w)
    to_walk = band & ~to_cw
    out[to_cw]   = cw_class
    out[to_walk] = out[wy[to_walk], wx[to_walk]]
    if stats is not None:
        stats['seam_px_to_crosswalk'] = int(to_cw.sum())
        stats['seam_px_to_walkable']  = int(to_walk.sum())
    return out, stats


# =============================================================================
# 3b-A. Crosswalk mask from the RAW (seam-free) class map
# =============================================================================
def build_crosswalk_mask(cls_raw, res, max_hole_m2=1.5, min_area_m2=2.0):
    """Light cleaning only (holes filled, specks dropped) — deliberately NOT the
    snapped map, whose closing merges corner-touching crossings."""
    px_per_m2 = 1.0 / (res * res)
    cw_raw = cls_raw == 4
    holes = binary_fill_holes(cw_raw, structure=STRUCT8) & ~cw_raw
    hl, nh = cc_label(holes, structure=STRUCT8)
    if nh:
        hsz = np.bincount(hl.ravel())
        small = hsz <= max_hole_m2 * px_per_m2
        small[0] = False
        cw_raw = cw_raw | small[hl]
    lab_cw, n_cw = cc_label(cw_raw, structure=STRUCT8)
    sizes = np.bincount(lab_cw.ravel())
    keep = sizes >= min_area_m2 * px_per_m2
    keep[0] = False
    cw_mask = keep[lab_cw]
    lab_cw, n_cw = cc_label(cw_mask, structure=STRUCT8)
    return cw_mask, lab_cw, n_cw


# =============================================================================
# 3b-D. Keep crosswalk regions near a retained sidewalk centerline
# =============================================================================
def filter_by_sidewalk(lab_cw, n_cw, sidewalk_ref, res, buffer_m=5.0):
    """Bounding-box-local version of the notebook's proximity filter: a region
    is kept iff a sidewalk_ref pixel lies within buffer_m of it."""
    H, W = lab_cw.shape
    buf_px = int(buffer_m / res)
    objects = _find_objects(lab_cw, n_cw)
    near_ids, dropped_far = [], []
    for k in range(1, n_cw + 1):
        sl = objects[k - 1]
        if sl is None:
            continue
        r0 = max(sl[0].start - buf_px, 0); r1 = min(sl[0].stop + buf_px, H)
        c0 = max(sl[1].start - buf_px, 0); c1 = min(sl[1].stop + buf_px, W)
        win = (slice(r0, r1), slice(c0, c1))
        if not sidewalk_ref[win].any():
            dropped_far.append(k)
            continue
        reg = lab_cw[win] == k
        near = binary_dilation(reg, STRUCT8, iterations=buf_px) & sidewalk_ref[win]
        (near_ids if near.any() else dropped_far).append(k)
    return near_ids, dropped_far


def _find_objects(lab, n):
    from scipy.ndimage import find_objects
    return find_objects(lab, max_label=n)


# =============================================================================
# 3c. Adaptive erosion + watershed + forced separation
# =============================================================================
def _interior_distance(mask, res):
    """Explicit padding handles regions touching the crop boundary."""
    return distance_transform_edt(np.pad(mask, 1), sampling=(res, res))[1:-1, 1:-1]


def _pca_box_fill(mask, res):
    """Area / PCA-oriented bounding-box area; a shape heuristic."""
    points = np.column_stack(np.nonzero(mask)).astype(float) * res
    if len(points) < 3:
        return 1.0
    _, vectors = np.linalg.eigh(np.cov(points, rowvar=False))
    projected = (points - points.mean(axis=0)) @ vectors
    box_area = np.prod(np.ptp(projected, axis=0) + res)
    return float(mask.sum() * res * res / max(box_area, res * res))


def _adaptive_markers(mask, res, p):
    """Recursively split thick interiors without specifying a crossing count."""
    split_log = []

    def visit(piece, depth=0):
        if depth >= p['max_depth'] or _pca_box_fill(piece, res) >= p['straight_fill']:
            return [piece]                               # rectangular enough
        distance = _interior_distance(piece, res)
        minimum_area = max(p['min_seed_area_m2'],
                           p['min_seed_fraction'] * piece.sum() * res * res)
        for radius in np.arange(p['erosion_step_m'], float(distance.max()),
                                p['erosion_step_m']):
            labels, n = cc_label(piece & (distance > radius), structure=STRUCT8)
            if n < 2:
                continue
            areas = np.bincount(labels.ravel()) * res * res
            ids = [k for k in range(1, n + 1) if areas[k] >= minimum_area]
            if len(ids) < 2:
                continue
            children = [labels == k for k in ids]
            retained = sum(child.sum() for child in children)
            if retained < p['min_retained_fraction'] * piece.sum():
                continue
            persistent = all(
                np.count_nonzero(child & (distance > radius + p['persist_m'])) * res * res >= 0.5
                for child in children)
            if not persistent:
                continue
            split_log.append({'depth': depth, 'additional_erosion_m': float(radius),
                              'children': len(children)})
            leaves = []
            for child in children:
                leaves.extend(visit(child, depth + 1))
            return leaves
        return [piece]                                   # no supported split

    leaves = visit(mask)
    markers = np.zeros(mask.shape, dtype=np.int32)
    for seed_id, leaf in enumerate(leaves, start=1):
        markers[leaf] = seed_id
    return markers, split_log


def _estimate_axis(seed, segment):
    """Heading from the seed (clean direction), extent from the segment."""
    points = np.column_stack(np.nonzero(seed)).astype(float)
    if len(points) < 3:
        points = np.column_stack(np.nonzero(segment)).astype(float)
    center = points.mean(axis=0)
    if len(points) >= 3:
        _, vectors = np.linalg.eigh(np.cov(points, rowvar=False))
        direction = vectors[:, -1]
    else:
        direction = np.array([0.0, 1.0])
    heading = float(np.degrees(np.arctan2(-direction[0], direction[1])) % 180)
    full_points = np.column_stack(np.nonzero(segment)).astype(float)
    projection = (full_points - center) @ direction
    lo, hi = np.percentile(projection, [2, 98])
    axis = np.vstack([center + lo * direction, center + hi * direction])
    return axis, heading


DEFAULT_SPLIT = dict(erosion_step_m=0.10, persist_m=0.20, min_seed_area_m2=0.75,
                     min_seed_fraction=0.05, min_retained_fraction=0.20,
                     straight_fill=0.78, max_depth=8,
                     separation_px=2, min_segment_m2=1.0, pad_m=1.0)


def split_crosswalk_regions(lab_cw, near_ids, res, params=None):
    """Every retained crosswalk region -> unique crossing segments.

    Returns (cw_seg, segments) where cw_seg is a full-size int32 label map
    (0 = none; carved separation gaps stay 0) and segments maps
    segment id -> {region_id, slice, mask (local bool within slice),
    area_m2, heading_deg, axis_rc (global float 2x2 [[r,c],[r,c]])}.
    """
    from skimage.segmentation import watershed
    p = dict(DEFAULT_SPLIT); p.update(params or {})
    H, W = lab_cw.shape
    pad = max(int(np.ceil(p['pad_m'] / res)), 1)
    cw_seg = np.zeros_like(lab_cw, dtype=np.int32)
    segments = {}
    next_id = 1
    n_split = 0
    objects = _find_objects(lab_cw, int(lab_cw.max()))

    for region_id in sorted(near_ids):
        sl = objects[region_id - 1]
        if sl is None:
            continue
        region_slice = (slice(sl[0].start, sl[0].stop), slice(sl[1].start, sl[1].stop))
        r0, c0 = region_slice[0].start, region_slice[1].start
        mask = np.pad(lab_cw[region_slice] == region_id, pad_width=pad)

        markers, split_log = _adaptive_markers(mask, res, p)
        n_split += len(split_log)
        distance_m = _interior_distance(mask, res)
        seg_lab = watershed(-distance_m, markers=markers, mask=mask,
                            connectivity=np.ones((3, 3), bool), watershed_line=False)

        for local_id in range(1, int(markers.max()) + 1):
            segment = seg_lab == local_id
            seed = markers == local_id
            axis, heading = _estimate_axis(seed, segment)
            segment_id = next_id; next_id += 1
            unpadded = segment[pad:-pad, pad:-pad]
            cw_seg[region_slice][unpadded] = segment_id
            segments[segment_id] = {
                'region_id': region_id, 'slice': region_slice,
                'heading_deg': heading,
                'axis_rc': axis + np.array([r0 - pad, c0 - pad], float),
            }

    # ---- force the segments apart: carve separation_px along every boundary ----
    sep = p['separation_px']
    for sid, s in segments.items():
        rs, cs = s['slice']
        q = sep + 1
        r0 = max(rs.start - q, 0); r1 = min(rs.stop + q, H)
        c0 = max(cs.start - q, 0); c1 = min(cs.stop + q, W)
        win = (slice(r0, r1), slice(c0, c1))
        loc = cw_seg[win]
        me = loc == sid
        others = (loc > 0) & ~me
        if not others.any():
            continue
        near_other = binary_dilation(others, STRUCT8, iterations=sep)
        loc[me & near_other] = 0

    # ---- drop fragments left by the carving; store local masks ----
    min_seg_px = p['min_segment_m2'] / (res * res)
    for sid in list(segments):
        rs, cs = segments[sid]['slice']
        loc = cw_seg[rs, cs]
        m = loc == sid
        lab_m, n_m = cc_label(m, structure=STRUCT8)
        if n_m == 0:
            del segments[sid]; continue
        sizes = np.bincount(lab_m.ravel())
        small = sizes < min_seg_px; small[0] = False
        loc[small[lab_m] & m] = 0
        m = loc == sid
        if not m.any():
            del segments[sid]; continue
        segments[sid]['mask'] = m
        segments[sid]['area_m2'] = float(m.sum() * res * res)
    return cw_seg, segments, n_split


# =============================================================================
# 3c-2. Axis correction from the crosswalk's own edges
# =============================================================================
def _run_direction(pixels):
    P = pixels.astype(float); ctr = P.mean(0)
    if len(P) < 3:
        return None, len(P)
    _, vecs = np.linalg.eigh(np.cov((P - ctr).T))
    return vecs[:, 1], len(P)


def _mean_direction(dirs, weights):
    """weighted mean of undirected directions (mod 180°) via doubled angles."""
    ang = np.array([np.arctan2(d[0], d[1]) for d in dirs]); w = np.array(weights, float)
    z = np.sum(w * np.exp(2j * ang)); a = np.angle(z) / 2.0
    return np.array([np.sin(a), np.cos(a)])


def correct_axes(segments, cls_snap, res,
                 edge_sw_m=2.0, min_edge_m=1.5, align_tol_deg=12.0):
    """Landing pixels (near walkable) vs road edges; replace the PCA axis when
    it deviates from the road-edge heading by more than align_tol_deg."""
    H, W = cls_snap.shape
    sw_ref = np.isin(cls_snap, (1, 2, 3))
    pad = int(edge_sw_m / res) + 2
    n_corr = 0
    for sid, s in segments.items():
        rs, cs = s['slice']
        m = s['mask']
        r0 = max(rs.start - pad, 0); r1 = min(rs.stop + pad, H)
        c0 = max(cs.start - pad, 0); c1 = min(cs.stop + pad, W)
        loc = np.zeros((r1 - r0, c1 - c0), bool)
        loc[rs.start - r0:rs.stop - r0, cs.start - c0:cs.stop - c0] = m
        outline = loc & ~binary_erosion(loc, STRUCT8)
        # landing = outline within edge_sw_m of walkable (local dilation test)
        near_walk = binary_dilation(sw_ref[r0:r1, c0:c1], STRUCT8,
                                    iterations=int(edge_sw_m / res))
        landing = outline & near_walk
        road = outline & ~landing
        lr, nr = cc_label(road, structure=STRUCT8)
        runs = sorted(((int((lr == j).sum()), j) for j in range(1, nr + 1)), reverse=True)
        runs = [(n, j) for n, j in runs if n * res >= min_edge_m][:2]
        s['axis_corrected'] = False
        if not runs:
            continue
        dirs, wts = [], []
        for _, j in runs:
            d, n = _run_direction(np.c_[np.nonzero(lr == j)])
            if d is not None:
                dirs.append(d); wts.append(n)
        if not dirs:
            continue
        d_edge = _mean_direction(dirs, wts)
        heading_edge = float(np.degrees(np.arctan2(-d_edge[0], d_edge[1])) % 180)
        dev = abs((s['heading_deg'] - heading_edge + 90) % 180 - 90)
        if dev <= align_tol_deg:
            continue
        P = np.c_[np.nonzero(loc)].astype(float); ctr = P.mean(0)
        n_vec = np.array([-d_edge[1], d_edge[0]])
        offs = [float(((np.c_[np.nonzero(lr == j)] - ctr) @ n_vec).mean()) for _, j in runs]
        ctr = ctr + (np.mean(offs) if len(offs) == 2 else 0.0) * n_vec
        proj = (P - ctr) @ d_edge; lo, hi = np.percentile(proj, [2, 98])
        axis = np.vstack([ctr + lo * d_edge, ctr + hi * d_edge]) + np.array([r0, c0], float)
        s['axis_rc'] = axis; s['heading_deg'] = heading_edge; s['axis_corrected'] = True
        n_corr += 1
    return n_corr


# =============================================================================
# 3d. Midblock entrances attached to a crosswalk are false positives
# =============================================================================
def erase_midblock_on_crosswalk(cls_snap, net, cw_mask, touch_px=3):
    """Erase midblock components within touch_px of the crosswalk mask from the
    class map AND the network mask. Returns (cls_snap, net, gone, n_erased,
    n_kept); the caller repairs the sidewalk skeleton with `gone`."""
    cw_touch = binary_dilation(cw_mask, STRUCT8, iterations=touch_px)
    mb_lab, n_mb = cc_label(cls_snap == 3, structure=STRUCT8)
    reassigned, kept = [], []
    objects = _find_objects(mb_lab, n_mb)
    for k in range(1, n_mb + 1):
        sl = objects[k - 1]
        if sl is None:
            continue
        comp = mb_lab[sl] == k
        (reassigned if (comp & cw_touch[sl]).any() else kept).append(k)
    gone = np.isin(mb_lab, reassigned) if reassigned else np.zeros_like(cw_mask)
    if reassigned:
        cls_snap = cls_snap.copy()
        cls_snap[gone] = 0
        net = net & ~gone
    return cls_snap, net, gone, len(reassigned), len(kept)
