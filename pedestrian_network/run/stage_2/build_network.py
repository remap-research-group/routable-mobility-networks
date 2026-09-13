"""Stage 2 — rasters -> polygons -> centerline network (links + nodes). CPU only.

OBJECTIVE
  Turn the class maps and network-confidence rasters written by stage 1 into
  the deliverable pedestrian network: geo-referenced typed polygons (widths
  measurable) and a topologically connected centerline graph whose junctions
  and endpoints are the routing nodes. All rasters that share a CRS and pixel
  size are mosaicked into ONE array first, so links connect across tiles.

INPUT  (--input <stage-1 output folder>)
  classes/<stem>.tif                uint8 5-class map per tile
  confidence/<stem>_network.tif     calibrated network confidence per tile

OUTPUT  (--out, default <input>/network/; one subfolder per group when the
         inputs span several CRSs or pixel sizes)
  links.geojson             LineStrings: link_id, link_type (sidewalk |
                            midblock | crosswalk | pseudo), length_m,
                            avg_width_m, min_width_m, from_node, to_node
  nodes.geojson             Points: node_id, node_type (endpoint | junction |
                            type_change), degree, link_types
  network_polygons.geojson  Polygons: poly_type (sidewalk | midblock |
                            crosswalk), area_m2, segment_id (unique crossing)
  *_wgs84.geojson           the same three layers in WGS84 lon/lat (RFC 7946)
  network_mask.tif          uint8 connected-network mosaic
  network_stats.json        counts, km and mean widths by type, parameters

HOW IT WORKS (functions)
  discover()   list tiles that have both rasters; group by (CRS, pixel size)
  mosaic()     place each tile in one array by its transform (read_confidence
               undoes the uint8 scaling); `coverage` marks the data extent
  build()      the pipeline, in order:
    1. per --chunk-px chunk: crosswalks.clean_crosswalks_v2 (close/open, hole
       fill, speck reassignment, rectangle regularization),
       crosswalks.seal_crosswalk_seams (fill the background band between a
       crosswalk and walkable surface), postprocess.connect_crosswalks (snap
       crosswalks within --max-snap-px to the walkable surface; drop stranded)
    2. network mask = (confidence >= --thr) OR classes 1-4
       -> postprocess.close_network_gaps (--gap-close-px)
       -> fill holes < --max-hole-m2 -> drop components < --min-component-m2
    3. netgraph.build_walk_skeleton (skeleton of the mask with crosswalk
       removed, spurs < --min-spur-m pruned); crosswalks.build_crosswalk_mask
       from the RAW class map -> filter_by_sidewalk (--buffer-m)
       -> split_crosswalk_regions (adaptive erosion + watershed: one segment
       per unique crossing) -> correct_axes (heading from the road edges)
       -> erase_midblock_on_crosswalk; netgraph.remove_dangles (< --dangle-max-m
       unless near a crosswalk / entrance / the coverage edge)
    4. netgraph.build_links (crosswalk axes + pseudo connectors <=
       --max-pseudo-m joined to the skeleton) -> finalize_links (split at type
       changes, merge pass-through nodes, smooth, split at entrances, typed
       nodes) -> attach_widths -> build_typed_polygons (--poly-smooth-m,
       --poly-simplify-m) -> GeoJSON via common.write_geojson_pair

TUNING — parameters that matter (defaults assume 0.08 m/px)
  --thr 0.50              confidence threshold for mask inclusion; lower = more
                          (fainter) network and more false positives
  --gap-close-px 2        closing radius after mosaicking (bridges ~2*r px gaps)
  --max-hole-m2 5.0       holes smaller than this are filled (bigger = medians)
  --min-component-m2 100  drop network components smaller than this
  --max-snap-px 25        crosswalk snap distance (2 m); --connector-width-px 3
  --buffer-m 5.0          crosswalk regions farther than this from a sidewalk
                          centerline are dropped
  --min-spur-m 3.0        prune skeleton dead-ends shorter than this
  --dangle-max-m 8.0      remove dangling branches shorter than this unless
                          within --dangle-keep-m 2.0 of a crosswalk / entrance
                          or at the coverage edge
  --max-pseudo-m 4.0      crosswalk ends join the skeleton within this distance
  --poly-smooth-m 0.20 / --poly-simplify-m 0.30   polygon smoothing / DP (0 = raw)
  --chunk-px 1024         chunk size of the per-chunk crosswalk clean-up
  --crs / --res           CRS override / m/px for non-georeferenced rasters

USAGE
  python build_network.py --input ../../example/output/dc_2023
  python build_network.py --input /path/to/stage1_out --thr 0.4 --poly-simplify-m 0
"""
import sys
import json
import argparse
from pathlib import Path
from collections import Counter

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))            # run/common.py

from common import (ensure_proj, raster_meta, write_raster, read_confidence,
                    px_to_xy, write_geojson_pair, crs_name)
from postprocess import connect_crosswalks, close_network_gaps
import crosswalks as cwops
import netgraph


# =============================================================================
# Input discovery + mosaicking
# =============================================================================
def discover(stage1_dir, crs_override, res_fallback):
    """[(stem, cls_path, net_path, transform, crs, shape)] for every class
    raster that has a matching network-confidence raster."""
    cls_dir, conf_dir = stage1_dir / 'classes', stage1_dir / 'confidence'
    assert cls_dir.is_dir(), f'{cls_dir} not found — run stage_1/segmentation.py first'
    from rasterio.transform import Affine
    items = []
    for cls_path in sorted(cls_dir.glob('*.tif')):
        stem = cls_path.stem
        net_path = conf_dir / f'{stem}_network.tif'
        if not net_path.exists():
            print(f'WARN: {net_path.name} missing — skipping {stem}')
            continue
        t, crs, shape = raster_meta(cls_path, crs_override)
        if t is None:                                   # not georeferenced
            t, crs = Affine(res_fallback, 0, 0, 0, -res_fallback, 0), None
            key = f'image_{stem}'
        else:
            key = f'{crs_name(crs) if crs is not None else "nocrs"}|{abs(t.a):.6f}'
        items.append({'stem': stem, 'cls': cls_path, 'net': net_path,
                      'transform': t, 'crs': crs, 'shape': shape, 'key': key})
    assert items, f'no class rasters in {cls_dir}'
    return items


def mosaic(group):
    """Place every raster of a group in one array by its georeferencing.
    -> p_net float32, cls_raw uint8, coverage bool, mosaic transform, res, crs"""
    import rasterio
    from rasterio.transform import from_origin
    res = abs(group[0]['transform'].a)
    crs = group[0]['crs']
    xmin = min(g['transform'].c for g in group)
    ymax = max(g['transform'].f for g in group)
    xmax = max(g['transform'].c + g['shape'][1] * res for g in group)
    ymin = min(g['transform'].f - g['shape'][0] * res for g in group)
    W = int(round((xmax - xmin) / res))
    H = int(round((ymax - ymin) / res))
    transform = from_origin(xmin, ymax, res, res)
    print(f'mosaic: {H} x {W} px from {len(group)} raster(s) @ {res:.3f} m/px')
    if H * W > 600_000_000:
        print('WARN: very large mosaic — expect high memory use')

    p_net    = np.zeros((H, W), np.float32)
    cls_raw  = np.zeros((H, W), np.uint8)
    coverage = np.zeros((H, W), bool)
    for g in group:
        t, (h, w) = g['transform'], g['shape']
        r0 = int(round((ymax - t.f) / res))
        c0 = int(round((t.c - xmin) / res))
        sl = (slice(r0, r0 + h), slice(c0, c0 + w))
        p_net[sl] = read_confidence(g['net'])
        with rasterio.open(g['cls']) as ds:
            cls_raw[sl] = ds.read(1)
        coverage[sl] = True
    return p_net, cls_raw, coverage, transform, res, crs


# =============================================================================
# Network construction (the stage-2 pipeline)
# =============================================================================
def build(p_net, cls_raw, coverage, transform, res, crs, out_dir, args):
    from scipy.ndimage import (label as cc_label, binary_fill_holes,
                               binary_dilation, binary_erosion)
    H, W = cls_raw.shape
    px_per_m2 = 1.0 / (res * res)
    STRUCT8 = cwops.STRUCT8
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- per chunk: clean-up -> seam sealing -> connect_crosswalks -----------
    clean_stats = Counter()
    cls_snap = np.zeros_like(cls_raw)
    C = args.chunk_px
    for r0 in range(0, H, C):
        for c0 in range(0, W, C):
            sl = (slice(r0, min(r0 + C, H)), slice(c0, min(c0 + C, W)))
            if not coverage[sl].any():
                continue
            out, st = cwops.clean_crosswalks_v2(cls_raw[sl], res)
            out, st = cwops.seal_crosswalk_seams(out, max_gap_px=6, stats=st)
            out, snap_st = connect_crosswalks(out, max_snap_px=args.max_snap_px,
                                              connector_width_px=args.connector_width_px)
            clean_stats.update(st); clean_stats.update(snap_st)
            cls_snap[sl] = out

    # ---- network mask ---------------------------------------------------------
    net = (p_net >= args.thr) | np.isin(cls_snap, (1, 2, 3, 4))
    net = close_network_gaps(net, args.gap_close_px)
    holes = binary_fill_holes(net, structure=STRUCT8) & ~net
    hl, nh = cc_label(holes, structure=STRUCT8)
    if nh:
        hsz = np.bincount(hl.ravel())
        fill = hsz <= int(args.max_hole_m2 * px_per_m2); fill[0] = False
        net = net | fill[hl]
    lab, n0 = cc_label(net, structure=STRUCT8)
    sizes = np.bincount(lab.ravel())
    keep = sizes >= int(args.min_component_m2 * px_per_m2); keep[0] = False
    net = keep[lab]
    del lab, holes, hl
    print(f'mask: {net.sum() * res * res:,.0f} m² | components {n0} -> {int(keep[1:].sum())}')

    # ---- crosswalk mask from the RAW map + sidewalk skeleton ------------------
    cw_mask, lab_cw, n_cw = cwops.build_crosswalk_mask(cls_raw, res)
    skel_walk, typed_walk, n_stray, sidewalk_ref, walk_mask = netgraph.build_walk_skeleton(
        net, cls_snap, cw_mask, res, min_spur_m=args.min_spur_m)
    print(f'crosswalk regions: {n_cw} | sidewalk skeleton: {len(typed_walk)} links '
          f'| stray-on-crosswalk removed: {n_stray}')

    near_ids, dropped_far = cwops.filter_by_sidewalk(
        lab_cw, n_cw, sidewalk_ref, res, buffer_m=args.buffer_m)
    cw_seg, segments, n_splits = cwops.split_crosswalk_regions(lab_cw, near_ids, res)
    n_corr = cwops.correct_axes(segments, cls_snap, res)
    print(f'unique crossings: {len(segments)} ({len(near_ids)} regions kept, '
          f'{len(dropped_far)} dropped, {n_splits} splits, {n_corr} axes corrected)')

    # ---- midblock-on-crosswalk false positives --------------------------------
    cls_snap, net, gone, n_mb_erased, n_mb_kept = cwops.erase_midblock_on_crosswalk(
        cls_snap, net, cw_mask)
    skel_walk, _, n_orphan = netgraph.repair_skeleton_after_erase(
        skel_walk, gone, res, min_spur_m=args.min_spur_m)

    # ---- dangling branches (kept near crosswalk / midblock / coverage edge) ---
    keep_px = int(args.dangle_keep_m / res)
    border_px = max(int(1.0 / res), 1)
    edge_zone = coverage & ~binary_erosion(coverage, STRUCT8, iterations=border_px,
                                           border_value=0)
    keep_zone = (binary_dilation(cw_mask, STRUCT8, iterations=keep_px)
                 | binary_dilation(cls_snap == 3, STRUCT8, iterations=keep_px)
                 | edge_zone)
    skel_net, n_dangle = netgraph.remove_dangles(skel_walk, keep_zone, res,
                                                 dangle_max_m=args.dangle_max_m)
    print(f'midblock erased: {n_mb_erased} ({n_mb_kept} kept) | orphans: {n_orphan} '
          f'| dangles removed: {n_dangle}')

    # ---- typed graph: links + nodes -------------------------------------------
    links, n_pseudo, n_unatt = netgraph.build_links(skel_net, segments, res,
                                                    max_pseudo_m=args.max_pseudo_m)
    links_final, nodes_final = netgraph.finalize_links(links, cls_snap, res,
                                                       min_run_m=1.0)
    netgraph.attach_widths(links_final, walk_mask, cw_seg > 0, res)
    print(f'links: {len(links_final)} (pseudo {n_pseudo}, unattached crosswalk ends '
          f'{n_unatt}) | nodes: {len(nodes_final)}')

    # ---- polygons -------------------------------------------------------------
    polys = netgraph.build_typed_polygons(net, cls_snap, lab_cw, dropped_far,
                                          cw_mask, cw_seg, segments, transform, res,
                                          smooth_m=args.poly_smooth_m,
                                          simplify_m=args.poly_simplify_m)

    # ---- write ----------------------------------------------------------------
    write_raster(out_dir / 'network_mask.tif', net.astype(np.uint8), transform, crs, 'uint8')
    written = ['network_mask.tif']

    link_feats = [{'type': 'Feature',
                   'properties': {'link_id': l['id'], 'link_type': l['type'],
                                  'length_m': l['length_m'],
                                  'avg_width_m': l['avg_width_m'],
                                  'min_width_m': l['min_width_m'],
                                  'from_node': l['from_node'], 'to_node': l['to_node']},
                   'geometry': {'type': 'LineString',
                                'coordinates': px_to_xy(l['geom'], transform)}}
                  for l in links_final]
    written += write_geojson_pair(out_dir, 'links', link_feats, crs)

    node_feats = [{'type': 'Feature',
                   'properties': {'node_id': n['id'], 'node_type': n['node_type'],
                                  'degree': n['degree'],
                                  'link_types': '+'.join(n['link_types'])},
                   'geometry': {'type': 'Point',
                                'coordinates': px_to_xy([n['rc']], transform)[0]}}
                  for n in nodes_final]
    written += write_geojson_pair(out_dir, 'nodes', node_feats, crs)

    from shapely.geometry import mapping
    poly_feats = [{'type': 'Feature',
                   'properties': {'poly_type': p['poly_type'], 'area_m2': p['area_m2'],
                                  'segment_id': p['segment_id']},
                   'geometry': json.loads(json.dumps(mapping(p['geometry'])))}
                  for p in polys]
    written += write_geojson_pair(out_dir, 'network_polygons', poly_feats, crs)

    by_type = Counter(l['type'] for l in links_final)
    km = Counter()
    for l in links_final:
        km[l['type']] += l['length_m'] / 1000.0
    poly_by = Counter(p['poly_type'] for p in polys)
    stats = {
        'mosaic_px': [H, W], 'resolution_m_per_px': round(res, 4),
        'crs': crs_name(crs) if crs is not None else None,
        'params': {k: v for k, v in vars(args).items() if k not in ('input', 'out')},
        'network_area_m2': round(float(net.sum()) * res * res, 1),
        'crosswalk_clean_snap': dict(clean_stats),
        'crosswalk_regions': n_cw, 'crosswalk_regions_kept': len(near_ids),
        'unique_crossings': len(segments), 'axes_corrected': n_corr,
        'midblock_erased': n_mb_erased, 'dangles_removed': n_dangle,
        'links': {t: {'n': by_type[t], 'km': round(km[t], 3),
                      **({'avg_width_m': round(float(np.mean(ws)), 2),
                          'min_width_m': round(float(np.min(ws)), 2)}
                         if (ws := [l['avg_width_m'] for l in links_final
                                    if l['type'] == t and l['avg_width_m'] is not None])
                         else {})}
                  for t in sorted(by_type)},
        'nodes': dict(Counter(n['node_type'] for n in nodes_final)),
        'unattached_crosswalk_ends': n_unatt,
        'polygons': {t: poly_by[t] for t in sorted(poly_by)},
        'files': written + ['network_stats.json'],
    }
    (out_dir / 'network_stats.json').write_text(json.dumps(stats, indent=2))
    print(f'-> {out_dir}: {len(link_feats)} links, {len(node_feats)} nodes, '
          f'{len(poly_feats)} polygons')
    return stats


# =============================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--input', type=str, required=True,
                    help='stage-1 output folder (contains classes/ and confidence/)')
    ap.add_argument('--out',   type=str, default=None,
                    help='output folder (default: <input>/network)')
    ap.add_argument('--crs',   type=str, default=None,
                    help='CRS to assign to rasters whose embedded CRS is unusable')
    ap.add_argument('--res',   type=float, default=0.08,
                    help='m/px assumed for rasters without georeferencing')
    ap.add_argument('--chunk-px', type=int, default=1024,
                    help='chunk size for the per-chunk crosswalk clean-up + snapping')
    # mask
    ap.add_argument('--thr',              type=float, default=0.50,
                    help='network-confidence threshold for mask inclusion')
    ap.add_argument('--gap-close-px',     type=int,   default=2)
    ap.add_argument('--max-hole-m2',      type=float, default=5.0,
                    help='enclosed holes smaller than this are filled')
    ap.add_argument('--min-component-m2', type=float, default=100.0,
                    help='drop network components smaller than this')
    # crosswalk treatment
    ap.add_argument('--max-snap-px',      type=int,   default=25,
                    help='crosswalk snap distance (25 px = 2 m at 0.08 m/px)')
    ap.add_argument('--connector-width-px', type=int, default=3)
    ap.add_argument('--buffer-m',         type=float, default=5.0,
                    help='crosswalk regions farther than this from a sidewalk centerline are dropped')
    # graph
    ap.add_argument('--min-spur-m',       type=float, default=3.0,
                    help='prune skeleton dead-ends shorter than this')
    ap.add_argument('--dangle-max-m',     type=float, default=8.0,
                    help='dangling sidewalk branch shorter than this is removed '
                         '(unless near crosswalk / midblock / coverage edge)')
    ap.add_argument('--dangle-keep-m',    type=float, default=2.0)
    ap.add_argument('--max-pseudo-m',     type=float, default=4.0,
                    help='crosswalk ends join the sidewalk skeleton within this distance')
    # polygons
    ap.add_argument('--poly-smooth-m',    type=float, default=0.20)
    ap.add_argument('--poly-simplify-m',  type=float, default=0.30,
                    help='Douglas-Peucker tolerance for polygons (0 = raw)')
    args = ap.parse_args()

    ensure_proj()
    stage1_dir = Path(args.input)
    out_root = Path(args.out) if args.out else stage1_dir / 'network'
    items = discover(stage1_dir, args.crs, args.res)
    groups = {}
    for it in items:
        groups.setdefault(it['key'], []).append(it)
    print(f'{len(items)} raster(s) in {len(groups)} mosaic group(s) -> {out_root}')

    for i, (key, group) in enumerate(groups.items()):
        out_dir = out_root if len(groups) == 1 else out_root / (
            group[0]['stem'] if key.startswith('image_') else f'mosaic_{i}')
        print(f'\n== group {key}: {len(group)} raster(s) ==')
        p_net, cls_raw, coverage, transform, res, crs = mosaic(group)
        build(p_net, cls_raw, coverage, transform, res, crs, out_dir, args)
    print('\nStage 2 done.')


if __name__ == '__main__':
    main()
