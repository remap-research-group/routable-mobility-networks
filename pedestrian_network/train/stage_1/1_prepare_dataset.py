"""Step 1 — CVAT export -> training-ready dataset (+ QA report).

OBJECTIVE
  Convert a CVAT "Segmentation mask 1.1" export into the exact layout
  2_train.py reads: one uint8 class-index GeoTIFF per tile, a fixed
  train/val/test split, and class weights. Only needed to REBUILD or EXTEND
  the dataset — train/v0_2026sep_image/ already holds the finalized data.

INPUT  (under train/v0_2026sep_image/ = DATA_ROOT in config.py)
  cvat_exports/<NAME>.zip or <NAME>/   CVAT export: SegmentationClass/*.png +
                                       labelmap.txt (colour -> class is read
                                       from labelmap.txt by NAME, never by
                                       CVAT's index order)
  imagery/dc_<year>/images/*.tif       source aerial GeoTIFF tiles (RGB,
                                       1024 px, 0.08 m/px, tile_<TX>_<TY>)
  imagery/dc_<year>/splits.json        optional split to preserve
  The year -> export-name mapping is CVAT_EXPORTS in config.py.

OUTPUT
  dc_<year>/images/*.tif        copied imagery
  dc_<year>/masks/*.tif         uint8 class indices 0-4, one band,
                                georeferenced from the tile grid
  dc_<year>/splits.json         preserved, or a fresh 8:1:1 split (seed 42)
  dc_<year>/dataset_stats.json  pixel counts per class
  combined_dataset_stats.json   inverse-frequency class weights over all years
                                (read by 2_train.py)

HOW IT WORKS (functions)
  discover_exports()       find the export per year (locate_export,
                           parse_labelmap)
  convert()                PNG mask -> class indices (load_export_mask)
                           -> GeoTIFF (tile_transform, write_geotiff); copy
                           imagery; write or preserve splits
  verify()                 every image has a mask, values in 0-4, sizes match
  stats()                  class pixel counts -> class weights
  qa_gap_analysis()        --qa: measures the background slivers CVAT leaves
                           between adjacent polygons (motivates GAP_CLOSE_PX)
  qa_crosswalk_adjacency() --qa: flags crosswalks farther than flag_px from
                           any walkable label (likely labelling errors)

TUNING — parameters that matter
  --qa, --qa-sample 120    run the two QA checks (on a sample of tiles)
  Class names / colours (PALETTE), the tile grid (EPSG, RES_M_PER_PX, TILE_PX,
  GRID_XMIN, GRID_YMAX) and CVAT_EXPORTS live in config.py. Unknown colours
  map to background and are counted. Keep splits.json to stay comparable with
  the released model.

USAGE
  python 1_prepare_dataset.py            # convert + verify + stats
  python 1_prepare_dataset.py --qa       # ... + gap analysis + crosswalk check
"""
import json
import random
import shutil
import zipfile
import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import rasterio
from rasterio.transform import from_bounds
from scipy.ndimage import (binary_dilation, binary_erosion, distance_transform_edt,
                           generate_binary_structure, label as cc_label)

from config import (DATA_ROOT, CVAT_EXPORTS, PALETTE, EPSG, RES_M_PER_PX,
                    TILE_PX, TILE_M, GRID_XMIN, GRID_YMAX)

EXPORT_DIR   = DATA_ROOT / 'cvat_exports'
IMAGERY_ROOT = DATA_ROOT / 'imagery'

N_CLASSES   = len(PALETTE)
CLASS_NAMES = [PALETTE[c][0] for c in range(N_CLASSES)]
NAME_TO_ID  = {PALETTE[c][0]: c for c in range(N_CLASSES)}
STRUCT8     = generate_binary_structure(2, 2)


# =============================================================================
# Tile grid helpers
# =============================================================================
def bbox_from_stem(stem):
    tx, ty = map(int, stem.replace('tile_', '').split('_'))
    xmin = GRID_XMIN + tx * TILE_M
    ymax = GRID_YMAX - ty * TILE_M
    return xmin, ymax - TILE_M, xmin + TILE_M, ymax


def tile_transform(stem):
    return from_bounds(*bbox_from_stem(stem), TILE_PX, TILE_PX)


def write_geotiff(path, array, transform, is_rgb=False):
    if is_rgb:
        h, w = array.shape[:2]; count = 3
        data = np.transpose(array, (2, 0, 1))
    else:
        h, w = array.shape; count = 1
        data = array[None, :, :]
    profile = {'driver': 'GTiff', 'height': h, 'width': w, 'count': count,
               'dtype': 'uint8', 'compress': 'deflate',
               'transform': transform, 'crs': f'EPSG:{EPSG}'}
    with rasterio.open(path, 'w', **profile) as dst:
        dst.write(data)


# =============================================================================
# CVAT export discovery + parsing
# =============================================================================
def locate_export(yr, name):
    """Extracted export root for one year (extracts the zip if needed), or None."""
    folder = EXPORT_DIR / name
    zpath  = EXPORT_DIR / f'{name}.zip'
    if folder.exists() and any(folder.rglob('SegmentationClass')):
        return folder
    extract_dir = EXPORT_DIR / f'cvat_export_{yr}'
    if zpath.exists():
        if not any(extract_dir.rglob('SegmentationClass')):
            extract_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zpath) as zf:
                zf.extractall(extract_dir)
            print(f'{yr}: extracted {zpath.name} -> {extract_dir}')
        return extract_dir
    return None


def parse_labelmap(path):
    """labelmap.txt lines look like  "visible_sidewalk:255,140,0::"  -> {name: (r,g,b)}"""
    out = {}
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        parts = line.split(':')
        if len(parts) < 2 or not parts[1]:
            continue
        rgb = tuple(int(v) for v in parts[1].split(','))
        out[parts[0]] = rgb
    return out


def load_export_mask(png_path, color_to_class):
    """Color/P-mode PNG -> (class-index uint8 array, n unknown-color px)."""
    im = Image.open(png_path)
    if im.mode == 'P':
        idx = np.array(im)
        pal = np.array(im.getpalette(), dtype=np.uint8).reshape(-1, 3)
        rgb = pal[idx]
    else:
        rgb = np.array(im.convert('RGB'))
    out = np.full(rgb.shape[:2], 255, np.uint8)
    for (r, g, b), cid in color_to_class.items():
        out[(rgb[..., 0] == r) & (rgb[..., 1] == g) & (rgb[..., 2] == b)] = cid
    n_unknown = int((out == 255).sum())
    out[out == 255] = 0
    return out, n_unknown


def find_export_image(jpeg_dir, stem):
    if jpeg_dir is None:
        return None
    for ext in ('.png', '.jpg', '.jpeg', '.tif', '.PNG', '.JPG'):
        p = jpeg_dir / f'{stem}{ext}'
        if p.exists():
            return p
    return None


def discover_exports():
    export_info = {}
    for yr, name in CVAT_EXPORTS.items():
        root = locate_export(yr, name)
        if root is None:
            print(f'{yr}: {name}(.zip) not in {EXPORT_DIR} — SKIPPING this year. '
                  f'Re-run once the export is placed there.\n')
            continue
        seg_dir   = next(root.rglob('SegmentationClass'))
        lm_path   = next(root.rglob('labelmap.txt'))
        jpeg_dirs = list(root.rglob('JPEGImages'))
        labelmap  = parse_labelmap(lm_path)

        color_to_class = {}
        for lbl, rgb in labelmap.items():
            if lbl not in NAME_TO_ID:
                print(f'  WARN {yr}: unexpected label "{lbl}" ({rgb}) in labelmap — ignored')
                continue
            color_to_class[rgb] = NAME_TO_ID[lbl]
            expected = PALETTE[NAME_TO_ID[lbl]][1]
            flag = '' if rgb == expected else f'  <-- color changed in CVAT (expected {expected})'
            print(f'  {yr}: {lbl:18s} {str(rgb):15s} -> class {NAME_TO_ID[lbl]}{flag}')

        missing = set(NAME_TO_ID) - set(labelmap)
        if missing:
            print(f'  WARN {yr}: labels missing from export labelmap: {missing}')

        pngs = sorted(seg_dir.glob('*.png'))
        export_info[yr] = {'root': root, 'seg_dir': seg_dir,
                           'jpeg_dir': jpeg_dirs[0] if jpeg_dirs else None,
                           'color_to_class': color_to_class, 'pngs': pngs}
        print(f'{yr}: {len(pngs)} mask PNGs\n')
    assert export_info, f'No CVAT exports found in {EXPORT_DIR} — nothing to do.'
    return export_info


# =============================================================================
# Convert: masks -> class-index GeoTIFFs, pair with imagery, preserve splits
# =============================================================================
def convert(export_info):
    written = {}
    for yr, info in export_info.items():
        out_ds     = DATA_ROOT / f'dc_{yr}'
        out_images = out_ds / 'images'
        out_masks  = out_ds / 'masks'
        out_images.mkdir(parents=True, exist_ok=True)
        out_masks.mkdir(parents=True, exist_ok=True)
        src_img_dir = IMAGERY_ROOT / f'dc_{yr}' / 'images'   # primary aerial source

        stems, unknown_total, img_from_export, missing_img = [], 0, 0, []
        for i, png in enumerate(info['pngs']):
            stem = png.stem
            mask, n_unk = load_export_mask(png, info['color_to_class'])
            unknown_total += n_unk
            write_geotiff(out_masks / f'{stem}.tif', mask, tile_transform(stem))

            src_tif = src_img_dir / f'{stem}.tif'
            exp_img = find_export_image(info['jpeg_dir'], stem)
            if src_tif.exists():
                shutil.copyfile(src_tif, out_images / f'{stem}.tif')
            elif exp_img is not None:
                img = np.array(Image.open(exp_img).convert('RGB'))
                write_geotiff(out_images / f'{stem}.tif', img, tile_transform(stem), is_rgb=True)
                img_from_export += 1
            else:
                missing_img.append(stem)
                continue
            stems.append(stem)
            if (i + 1) % 100 == 0 or (i + 1) == len(info['pngs']):
                print(f'  [{yr}] {i + 1}/{len(info["pngs"])} tiles')

        # ---- splits.json: keep the SAME split as all previous runs ----
        src_split = IMAGERY_ROOT / f'dc_{yr}' / 'splits.json'
        stems_set = set(stems)
        if src_split.exists():
            splits = json.loads(src_split.read_text())
            filtered = {k: [s for s in v if s in stems_set] for k, v in splits.items()}
            n_lost = sum(len(v) for v in splits.values()) - sum(len(v) for v in filtered.values())
            if n_lost:
                print(f'  WARN {yr}: {n_lost} split stems missing from the CVAT export')
            (out_ds / 'splits.json').write_text(json.dumps(filtered, indent=2))
        else:
            rng = random.Random(42)
            pool = sorted(stems_set); rng.shuffle(pool)
            n = len(pool); n_tr = int(0.8 * n); n_va = int(0.1 * n)
            (out_ds / 'splits.json').write_text(json.dumps(
                {'train': pool[:n_tr], 'val': pool[n_tr:n_tr + n_va],
                 'test': pool[n_tr + n_va:]}, indent=2))
            print(f'  WARN {yr}: no source splits.json — generated a fresh 8:1:1 split (seed 42)')

        written[yr] = stems
        print(f'{yr}: {len(stems)} tiles written | unknown-color px: {unknown_total} | '
              f'export-image fallbacks: {img_from_export} | missing images: {len(missing_img)}')
        if missing_img:
            print(f'   WARN — no aerial image found for: '
                  f'{missing_img[:10]}{" ..." if len(missing_img) > 10 else ""}')
    return written


# =============================================================================
# Verify + stats
# =============================================================================
def verify(export_info):
    valid_vals = set(range(N_CLASSES))
    all_ok = True
    for yr in export_info:
        ds = DATA_ROOT / f'dc_{yr}'
        imgs  = sorted((ds / 'images').glob('*.tif'))
        masks = sorted((ds / 'masks').glob('*.tif'))
        same  = [p.stem for p in imgs] == [p.stem for p in masks]
        print(f'{yr}: {len(imgs)} images | {len(masks)} masks | stems match: {same}')
        all_ok &= same and len(imgs) > 0

        bad_vals, bad_size = [], []
        for mp in masks:
            arr = np.array(Image.open(mp))
            if arr.ndim == 3: arr = arr[..., 0]
            vals = set(np.unique(arr).tolist())
            if not vals <= valid_vals: bad_vals.append((mp.stem, sorted(vals - valid_vals)))
            if arr.shape != (TILE_PX, TILE_PX): bad_size.append((mp.stem, arr.shape))
        print(f'   invalid mask values: {len(bad_vals)} | wrong size: {len(bad_size)}')
        if bad_vals: print('   ', bad_vals[:5]); all_ok = False
        if bad_size: print('   ', bad_size[:5]); all_ok = False

        splits = json.loads((ds / 'splits.json').read_text())
        tr, va, te = (set(splits[k]) for k in ('train', 'val', 'test'))
        on_disk = {p.stem for p in (ds / 'masks').glob('*.tif')}
        print(f'   splits: train={len(tr)} val={len(va)} test={len(te)} | '
              f'overlap={len(tr & va) + len(tr & te) + len(va & te)} | '
              f'not on disk={len((tr | va | te) - on_disk)} | '
              f'unassigned={len(on_disk - (tr | va | te))}')
    print('ALL OK' if all_ok else 'ISSUES FOUND — fix before training!')
    return all_ok


def stats(export_info, written):
    per_year_totals, combined_totals = {}, {c: 0 for c in range(N_CLASSES)}
    for yr in export_info:
        totals = {c: 0 for c in range(N_CLASSES)}
        for mp in (DATA_ROOT / f'dc_{yr}' / 'masks').glob('*.tif'):
            arr = np.array(Image.open(mp))
            if arr.ndim == 3: arr = arr[..., 0]
            cnt = np.bincount(arr.ravel(), minlength=N_CLASSES)
            for c in range(N_CLASSES): totals[c] += int(cnt[c])
        per_year_totals[yr] = totals
        for c in range(N_CLASSES): combined_totals[c] += totals[c]

    grand = sum(combined_totals.values())
    pct = lambda x: 100.0 * x / grand if grand else 0.0
    wt  = lambda x: float(np.clip(grand / (N_CLASSES * x), 0.5, 50.0)) if x > 0 else 0.0
    combined_weights = [wt(combined_totals[c]) for c in range(N_CLASSES)]

    if len(per_year_totals) < 2:
        print(f'NOTE: only {sorted(per_year_totals)} processed — "combined" stats cover '
              f'that year alone. Re-run when the other export arrives.\n')
    print(f'Combined pixel distribution ({" + ".join(str(y) for y in sorted(per_year_totals))}):')
    for c, nme in enumerate(CLASS_NAMES):
        print(f'  {nme:18s} {combined_totals[c]:>16,}  {pct(combined_totals[c]):6.2f}%  '
              f'-> weight={combined_weights[c]:.2f}')

    out = {
        'combined_class_weights': combined_weights,
        'combined_pixels':  {CLASS_NAMES[c]: combined_totals[c] for c in range(N_CLASSES)},
        'combined_percent': {CLASS_NAMES[c]: pct(combined_totals[c]) for c in range(N_CLASSES)},
        'per_year': {str(yr): {CLASS_NAMES[c]: per_year_totals[yr][c] for c in range(N_CLASSES)}
                     for yr in per_year_totals},
    }
    (DATA_ROOT / 'combined_dataset_stats.json').write_text(json.dumps(out, indent=2))
    for yr in export_info:
        yr_grand = sum(per_year_totals[yr].values())
        (DATA_ROOT / f'dc_{yr}' / 'dataset_stats.json').write_text(json.dumps({
            'n_tiles': len(written[yr]),
            'pixels':  {CLASS_NAMES[c]: per_year_totals[yr][c] for c in range(N_CLASSES)},
            'percent': {CLASS_NAMES[c]: 100.0 * per_year_totals[yr][c] / yr_grand
                        for c in range(N_CLASSES)},
        }, indent=2))
    print(f"Saved {DATA_ROOT / 'combined_dataset_stats.json'}")


# =============================================================================
# Optional QA (--qa): gap analysis + crosswalk adjacency check
# =============================================================================
def close_bin(m, r):
    if r <= 0: return m
    x = binary_dilation(m, STRUCT8, iterations=r)
    return binary_erosion(x, STRUCT8, iterations=r, border_value=1)


def qa_gap_analysis(export_info, n_sample=120, radii=(1, 2, 3, 4)):
    """How much does closing radius r reconnect the label union? Informs
    cfg.GAP_CLOSE_PX (pick the smallest r where the mean merges plateau)."""
    rng = random.Random(0)
    for yr in export_info:
        masks = sorted((DATA_ROOT / f'dc_{yr}' / 'masks').glob('*.tif'))
        if n_sample and len(masks) > n_sample:
            masks = rng.sample(masks, n_sample)
        comp0, comps, filled = [], {r: [] for r in radii}, {r: 0 for r in radii}
        for mp in masks:
            arr = np.array(Image.open(mp))
            if arr.ndim == 3: arr = arr[..., 0]
            union = arr > 0
            if not union.any(): continue
            _, n0 = cc_label(union, structure=STRUCT8)
            comp0.append(n0)
            for r in radii:
                closed = close_bin(union, r)
                _, nr = cc_label(closed, structure=STRUCT8)
                comps[r].append(nr)
                filled[r] += int((closed & ~union).sum())
        print(f'{yr} gap analysis (n={len(comp0)} tiles with labels):')
        print(f'   mean components  r=0: {np.mean(comp0):5.2f}')
        for r in radii:
            merges = np.mean(comp0) - np.mean(comps[r])
            print(f'   mean components  r={r}: {np.mean(comps[r]):5.2f}  '
                  f'(merges +{merges:4.2f}/tile | gap px filled {filled[r]:>9,})')


def qa_crosswalk_adjacency(export_info, flag_px=25):
    """Distance of every labeled crosswalk component to the nearest walkable
    pixel (classes 1-3). Components farther than flag_px (2 m) are likely
    annotation misses — fix in CVAT before training."""
    records = []
    for yr in export_info:
        for mp in sorted((DATA_ROOT / f'dc_{yr}' / 'masks').glob('*.tif')):
            arr = np.array(Image.open(mp))
            if arr.ndim == 3: arr = arr[..., 0]
            cw = arr == 4
            if not cw.any(): continue
            walk = np.isin(arr, (1, 2, 3))
            dist = distance_transform_edt(~walk) if walk.any() else np.full(arr.shape, np.inf)
            lab, n = cc_label(cw, structure=STRUCT8)
            for comp in range(1, n + 1):
                sel = lab == comp
                records.append((yr, mp.stem, comp, float(dist[sel].min()), int(sel.sum())))

    if not records:
        print('No crosswalk labels found.')
        return
    d_all = np.array([r[3] for r in records])
    print(f'{len(records)} crosswalk components across processed years')
    print(f'  touching walkable (<1.5 px): {(d_all <= 1.5).sum()} '
          f'({100 * (d_all <= 1.5).mean():.1f}%)')
    print(f'  within gap range  (<=4 px) : {(d_all <= 4).sum()} '
          f'({100 * (d_all <= 4).mean():.1f}%)')
    print(f'  flagged  (> {flag_px} px = {flag_px * RES_M_PER_PX:.1f} m): {(d_all > flag_px).sum()}')
    flagged = sorted([r for r in records if r[3] > flag_px], key=lambda r: -r[3])
    if flagged:
        print('Worst offenders (fix in CVAT if the sidewalk is actually there):')
        for yr, stem, comp, d, area in flagged[:12]:
            print(f'  {yr} {stem} comp{comp}: {d:6.1f} px ({d * RES_M_PER_PX:5.2f} m), {area} px')
    else:
        print('No stranded crosswalks — labeling looks consistent.')


# =============================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--qa', action='store_true',
                    help='also run gap analysis + crosswalk adjacency check')
    ap.add_argument('--qa-sample', type=int, default=120,
                    help='tiles per year for the gap analysis (0 = all)')
    args = ap.parse_args()

    export_info = discover_exports()
    written = convert(export_info)
    ok = verify(export_info)
    stats(export_info, written)
    if args.qa:
        qa_gap_analysis(export_info, n_sample=args.qa_sample or None)
        qa_crosswalk_adjacency(export_info)
    print('\nNext: python 2_train.py --name v4_unified   (run inside tmux)')
    if not ok:
        raise SystemExit('Dataset issues found — fix before training.')


if __name__ == '__main__':
    main()
