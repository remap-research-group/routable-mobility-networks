"""Crosswalk snapping and gap closing (stage 2 library; numpy/scipy only).

OBJECTIVE
  Enforce two physical rules on a predicted class map: (1) a crosswalk must
  attach to the walkable network, and (2) small gaps between adjacent objects
  (an artefact of polygon labelling) must not break connectivity.

INPUT / OUTPUT
  uint8 5-class maps and boolean masks; no file I/O.

FUNCTIONS
  connect_crosswalks(pred_cls, max_snap_px=25, connector_width_px=3,
                     drop_isolated=True)   for every crosswalk component:
      touching walkable surface -> keep; nearest walkable pixel within
      max_snap_px -> paint a straight crosswalk connector of connector_width_px
      between the closest pixel pair; farther -> stranded false positive,
      removed. Returns (map, stats). Work is bounding-box-local per component.
  close_network_gaps(mask, radius=3)   binary closing (dilate then erode,
      8-connected) that fills gaps up to ~2 * radius px.
  extract_network(pred_cls, gap_close_px=3)   classes 1-4 -> closed mask ->
      (mask, component count); a convenience for quick checks.

TUNING — parameters that matter
  max_snap_px=25 (2 m at 0.08 m/px): larger snaps crosswalks from farther away
      but risks joining unrelated features; connector_width_px=3.
  radius / gap_close_px: too large merges parallel sidewalks across narrow
      medians; training used GAP_CLOSE_PX = 2 for the same reason.
  Exposed in build_network.py as --max-snap-px, --connector-width-px,
  --gap-close-px.
"""
import numpy as np


def _line_pixels(y0, x0, y1, x1):
    n = int(max(abs(y1 - y0), abs(x1 - x0))) + 1
    ys = np.linspace(y0, y1, n).round().astype(int)
    xs = np.linspace(x0, x1, n).round().astype(int)
    return ys, xs


def close_network_gaps(net_mask, radius=3):
    """Binary closing (8-conn) of a HxW bool/0-1 network mask."""
    from scipy.ndimage import binary_dilation, binary_erosion, generate_binary_structure
    if radius <= 0:
        return net_mask.astype(bool)
    struct = generate_binary_structure(2, 2)
    m = binary_dilation(net_mask.astype(bool), structure=struct, iterations=radius)
    m = binary_erosion(m, structure=struct, iterations=radius, border_value=1)
    return m


def connect_crosswalks(pred_cls, max_snap_px=25, connector_width_px=3,
                       drop_isolated=True, walk_classes=(1, 2, 3), cw_class=4):
    """Enforce 'crosswalk must attach to the sidewalk network' on a 5-class
    prediction map.

    For every connected crosswalk component:
      - already touching walkable surface           -> keep as is
      - nearest walkable pixel within max_snap_px   -> paint a straight
        connector (width connector_width_px, crosswalk class, background
        pixels only) between the closest pixel pair
      - farther than max_snap_px                    -> stranded false positive;
        removed if drop_isolated (25 px = 2 m at 0.08 m/px)

    Returns (new_pred_cls, stats_dict).

    Per-component work is restricted to the component's bounding box, so the
    cost scales with component size rather than image size — same results,
    but it stays fast when called on a whole large image
    (semantic_segmentation_run) instead of a 1024-px tile.
    """
    from scipy.ndimage import (label as cc_label, distance_transform_edt,
                               binary_dilation, generate_binary_structure,
                               find_objects)
    out   = pred_cls.copy()
    walk  = np.isin(pred_cls, walk_classes)
    cw    = pred_cls == cw_class
    stats = {'components': 0, 'already_touching': 0, 'connected': 0, 'dropped': 0}
    if not cw.any() or not walk.any():
        return out, stats

    dist, (iy, ix) = distance_transform_edt(~walk, return_indices=True)
    struct = generate_binary_structure(2, 2)
    lab, n = cc_label(cw, structure=struct)
    stats['components'] = n
    boxes = find_objects(lab)

    for comp in range(1, n + 1):
        sl = boxes[comp - 1]
        if sl is None:
            continue
        in_comp = lab[sl] == comp
        ys, xs = np.nonzero(in_comp)
        ys = ys + sl[0].start
        xs = xs + sl[1].start
        d = dist[ys, xs]
        i = int(d.argmin())
        if d[i] <= 1.5:                      # 8-neighbour contact
            stats['already_touching'] += 1
            continue
        if d[i] > max_snap_px:
            if drop_isolated:
                out[sl][in_comp] = 0
                stats['dropped'] += 1
            continue
        y0, x0 = int(ys[i]), int(xs[i])
        y1, x1 = int(iy[y0, x0]), int(ix[y0, x0])
        ly, lx = _line_pixels(y0, x0, y1, x1)
        # paint the connector inside a local window around the line only
        m = connector_width_px // 2 + 1
        ry0 = max(int(ly.min()) - m, 0); ry1 = min(int(ly.max()) + m + 1, out.shape[0])
        rx0 = max(int(lx.min()) - m, 0); rx1 = min(int(lx.max()) + m + 1, out.shape[1])
        connector = np.zeros((ry1 - ry0, rx1 - rx0), bool)
        connector[ly - ry0, lx - rx0] = True
        if connector_width_px > 1:
            connector = binary_dilation(connector, structure=struct,
                                        iterations=connector_width_px // 2)
        region = (slice(ry0, ry1), slice(rx0, rx1))
        out[region][connector & (pred_cls[region] == 0)] = cw_class
        stats['connected'] += 1
    return out, stats


def extract_network(pred_cls, gap_close_px=3, network_classes=(1, 2, 3, 4)):
    """Final connected pedestrian-network mask from a 5-class prediction.
    Returns (bool network mask, number of connected components)."""
    from scipy.ndimage import label as cc_label, generate_binary_structure
    net = np.isin(pred_cls, network_classes)
    net = close_network_gaps(net, gap_close_px)
    _, n = cc_label(net, structure=generate_binary_structure(2, 2))
    return net, n


