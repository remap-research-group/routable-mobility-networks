"""Stage 1 — semantic segmentation of aerial imagery with calibrated confidence.

OBJECTIVE
  Apply the pretrained DBSwinT_v4 model (run/src/) to aerial image(s) and write,
  per image, the raster products that stage 2 turns into the pedestrian
  network: a 5-class map and three calibrated confidence maps. Before anything
  runs, every input is inspected (format, size, bands, CRS, resolution,
  coverage, tiling advice) and the loaded model is described.

INPUT
  One or more images (file paths and/or --dir <folder>):
    * RGB GeoTIFF at 0.08 m/px, georeferenced (transform + CRS) = model-ready.
      Any size: 1024-px tiles, or a large image covered by the sliding window;
      images smaller than 512 px are reflect-padded.
    * Other GDAL rasters (.jp2, .img, .vrt) are inspected too; if the
      resolution is off by more than 5 % the run stops and points at
      prepare_image.py (--force runs anyway).
    * PNG/JPG work, but their outputs carry no georeferencing.
  Weights: run/src/best.pt + temperature.json + train_config.json (or --weights).

OUTPUT  (<out>/, default run/output/stage_1/)
  classes/<stem>.tif                uint8 5-class argmax: 0 background,
                                    1 visible_sidewalk, 2 tree (over pedestrian
                                    infrastructure), 3 entrance, 4 crosswalk
  confidence/<stem>_network.tif     calibrated probability that the pixel is
                                    pedestrian network (classes 1-4) — the
                                    primary confidence, p = sigmoid(logit / T)
  confidence/<stem>_crosswalk.tif   calibrated crosswalk probability
  confidence/<stem>_entrance.tif    calibrated midblock-entrance probability
                                    (uint8 with band scale 1/255 by default;
                                    --confidence-format float32 for raw floats)
  overlay/<stem>.jpg                imagery blended with the class colours (QA)
  stage1_summary.json               per-image class shares, mean confidence,
                                    parameters, weights provenance
  All rasters inherit the input's transform and CRS.

HOW IT WORKS (functions)
  main()                   parse args -> inspect inputs (common.describe_image)
                           -> load model (src/loader.load_model, prints the
                           model report) -> process each image -> summary json
  build_index()            georeference index of all inputs (transform, CRS,
                           size), used to find the neighbours of each tile
  with_context()           pads a tile with --context-px of pixels taken from
                           the neighbouring inputs (reflect padding where there
                           is none) so adjacent tiles agree along shared edges
  infer()                  reflect-pads images smaller than the window, calls
                           sliding_window_logits(), crops the logits back
  sliding_window_logits()  512-px windows with --overlap, batched through the
                           model; the four heads' logits are Hann-weighted and
                           averaged where windows overlap
  process_image()          read image -> context -> logits -> argmax and
                           calibrated sigmoid per head -> write rasters, overlay
  class_rgb() / outputs_for()   overlay colours / output file paths

TUNING — parameters that matter
  --context-px 192    imagery from neighbouring tiles padded around each tile
                      before inference (~15 m). 0 = per-tile (seams appear).
  --overlap 128       sliding-window overlap in px; more = smoother blending,
                      more compute. The window size (512) comes from the
                      training config (IMG_SIZE) and must not be changed.
  --batch 8           windows per GPU batch (~4 GB VRAM at 8).
  --crs EPSG:xxxx     assign a CRS to inputs whose tag is missing/unresolved.
  --confidence-format uint8|float32, --overlay-px 2048 (downscale cap),
  --no-overlay, --overwrite (redo cached images), --inspect (report only),
  --force (run despite NEEDS PREPARE).
  The class map is a hard argmax; the confidence threshold is applied in
  stage 2 (--thr), so stage 1 never needs re-running to tune the network.

USAGE
  python segmentation.py --inspect --dir ../../example/input/dc_2023
  python segmentation.py --dir ../../example/input/dc_2023 --out ../../example/output/dc_2023
  python segmentation.py a.tif b.tif --out /path/out --crs EPSG:26985
"""
import sys
import json
import time
import argparse
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))            # run/common.py
sys.path.insert(0, str(HERE.parent / 'src'))    # run/src/loader.py + model.py

from common import read_image, raster_meta, write_raster, write_confidence, describe_image
from loader import (load_model, resolve_weights, sigmoid, normalize,
                    CLASS_NAMES, PALETTE)

DEFAULT_OUT = HERE.parent / 'output' / 'stage_1'
IMG_EXT = ('.tif', '.tiff', '.jp2', '.img', '.vrt', '.png', '.jpg', '.jpeg')


# =============================================================================
# Sliding-window inference
# =============================================================================
def sliding_window_logits(img_u8, model, device, win, overlap=128, batch=8):
    """(H,W,3) uint8 -> Hann-blended logits (cond [5,H,W], net, en, cw)."""
    import torch
    H, W = img_u8.shape[:2]
    norm = normalize(img_u8)

    hann  = np.hanning(win).astype(np.float32)
    hann2 = np.clip(np.outer(hann, hann), 1e-3, None)
    step  = win - overlap
    ys = sorted({min(y, H - win) for y in range(0, H, step)})
    xs = sorted({min(x, W - win) for x in range(0, W, step)})
    coords = [(y, x) for y in ys for x in xs]

    nc = model.head_cond.out_channels
    cond_sum = np.zeros((nc, H, W), np.float32)
    net_sum  = np.zeros((H, W), np.float32)
    en_sum   = np.zeros((H, W), np.float32)
    cw_sum   = np.zeros((H, W), np.float32)
    wt_sum   = np.zeros((H, W), np.float32)

    n_batches = (len(coords) + batch - 1) // batch
    with torch.no_grad():
        for i in range(0, len(coords), batch):
            if n_batches > 50 and (i // batch) % 50 == 0:
                print(f'    windows {i // batch + 1}/{n_batches} batches')
            chunk = coords[i:i + batch]
            inp = torch.stack([
                torch.from_numpy(norm[y:y + win, x:x + win]).permute(2, 0, 1)
                for y, x in chunk]).float().to(device)
            lc, ln, le, lw = model(inp)
            lc = lc.float().cpu().numpy(); ln = ln[:, 0].float().cpu().numpy()
            le = le[:, 0].float().cpu().numpy(); lw = lw[:, 0].float().cpu().numpy()
            for j, (y, x) in enumerate(chunk):
                cond_sum[:, y:y + win, x:x + win] += lc[j] * hann2
                net_sum[y:y + win, x:x + win]     += ln[j] * hann2
                en_sum[y:y + win, x:x + win]      += le[j] * hann2
                cw_sum[y:y + win, x:x + win]      += lw[j] * hann2
                wt_sum[y:y + win, x:x + win]      += hann2

    d = np.maximum(wt_sum, 1e-6)
    return cond_sum / d, net_sum / d, en_sum / d, cw_sum / d


def infer(rgb, model, device, win, overlap, batch):
    """Pad images smaller than the window (reflect), crop the logits back."""
    H, W = rgb.shape[:2]
    ph, pw = max(win - H, 0), max(win - W, 0)
    if ph or pw:
        mode = 'reflect' if (ph < H and pw < W) else 'edge'
        rgb = np.pad(rgb, ((0, ph), (0, pw), (0, 0)), mode=mode)
    lg = sliding_window_logits(rgb, model, device, win, overlap, batch)
    if ph or pw:
        lg = tuple(a[..., :H, :W] for a in lg)
    return lg


# =============================================================================
# Context from neighbouring images (seam-free tiling)
# =============================================================================
def build_index(paths, crs_override):
    """{path: (transform, crs, shape)} for georeferenced GeoTIFF inputs."""
    index = {}
    for p in paths:
        if p.suffix.lower() not in ('.png', '.jpg', '.jpeg'):
            t, crs, shape = raster_meta(p, crs_override)
            if t is not None:
                index[p] = (t, crs, shape)
    return index


def _same_frame(a, b, tol=1e-6):
    (ta, ca, _), (tb, cb, _) = a, b
    if abs(ta.a - tb.a) > tol or abs(ta.e - tb.e) > tol:
        return False
    return (ca is None and cb is None) or (ca is not None and cb is not None
                                           and ca.to_wkt() == cb.to_wkt())


def with_context(path, rgb, index, ctx, crs_override):
    """Canvas (H+2c, W+2c, 3) with `rgb` in the centre and pixels from the
    neighbouring input images in the margins; margins with no neighbour are
    reflect-padded. Returns (canvas, (row0, col0))."""
    H, W = rgb.shape[:2]
    if ctx <= 0:
        return rgb, (0, 0)
    canvas = np.pad(rgb, ((ctx, ctx), (ctx, ctx), (0, 0)),
                    mode='reflect' if (ctx < H and ctx < W) else 'edge')
    me = index.get(path)
    if me is None:
        return canvas, (ctx, ctx)
    t, _, _ = me
    r = abs(t.a)
    S_r, S_c = H + 2 * ctx, W + 2 * ctx
    for q, meta in index.items():
        if q == path or not _same_frame(me, meta):
            continue
        tq, _, (h, w) = meta
        # where image q lands in the canvas (pixel offsets from georeferencing)
        row0 = int(round((t.f - tq.f) / r)) + ctx
        col0 = int(round((tq.c - t.c) / r)) + ctx
        rs, re_ = max(row0, 0), min(row0 + h, S_r)
        cs, ce  = max(col0, 0), min(col0 + w, S_c)
        if rs >= re_ or cs >= ce:
            continue
        nb, _, _ = read_image(q, crs_override)
        canvas[rs:re_, cs:ce] = nb[rs - row0:re_ - row0, cs - col0:ce - col0]
    return canvas, (ctx, ctx)


# =============================================================================
# Per-image processing
# =============================================================================
def class_rgb(mask):
    rgb = np.zeros((*mask.shape, 3), np.uint8)
    for c, col in PALETTE.items():
        rgb[mask == c] = col
    return rgb


def outputs_for(out_root, stem):
    return {'classes':   out_root / 'classes' / f'{stem}.tif',
            'network':   out_root / 'confidence' / f'{stem}_network.tif',
            'crosswalk': out_root / 'confidence' / f'{stem}_crosswalk.tif',
            'entrance':  out_root / 'confidence' / f'{stem}_entrance.tif'}


def process_image(path, model, cfg, temps, device, index, out_root, args):
    t0 = time.time()
    rgb, transform, crs = read_image(path, args.crs)
    H, W = rgb.shape[:2]
    res = abs(transform.a) if transform is not None else None
    print(f'{path.name}: {H}x{W} px | '
          + (f'{res:.3f} m/px, {crs if crs is not None else "no CRS"}'
             if transform is not None else 'not georeferenced'))

    canvas, (oy, ox) = with_context(path, rgb, index, args.context_px, args.crs)
    lg_cond, lg_net, lg_en, lg_cw = infer(canvas, model, device, cfg.IMG_SIZE,
                                          args.overlap, args.batch)
    lg_cond = lg_cond[:, oy:oy + H, ox:ox + W]
    lg_net, lg_en, lg_cw = (a[oy:oy + H, ox:ox + W] for a in (lg_net, lg_en, lg_cw))

    cls  = lg_cond.argmax(0).astype(np.uint8)
    p_net = sigmoid(lg_net / temps['T_network'])
    p_cw  = sigmoid(lg_cw  / temps['T_crosswalk'])
    p_en  = sigmoid(lg_en  / temps['T_entrance'])

    outs = outputs_for(out_root, path.stem)
    write_raster(outs['classes'], cls, transform, crs, 'uint8')
    for key, p in (('network', p_net), ('crosswalk', p_cw), ('entrance', p_en)):
        write_confidence(outs[key], p, transform, crs, args.confidence_format)

    if not args.no_overlay:
        from PIL import Image
        ov = (class_rgb(cls).astype(np.float32) * 0.45
              + rgb.astype(np.float32) * 0.55).astype(np.uint8)
        ov = Image.fromarray(ov)
        if max(ov.size) > args.overlay_px:
            f = args.overlay_px / max(ov.size)
            ov = ov.resize((int(ov.size[0] * f), int(ov.size[1] * f)), Image.BILINEAR)
        (out_root / 'overlay').mkdir(parents=True, exist_ok=True)
        ov.save(out_root / 'overlay' / f'{path.stem}.jpg', quality=88)

    n_px = cls.size
    summary = {
        'image': path.name, 'size_px': [H, W],
        'georeferenced': transform is not None,
        'crs': str(crs) if crs is not None else None,
        'resolution_m_per_px': round(res, 4) if res else None,
        'class_percent': {CLASS_NAMES[c]: round(100.0 * float((cls == c).sum()) / n_px, 3)
                          for c in range(len(CLASS_NAMES))},
        'mean_network_confidence': round(float(p_net.mean()), 4),
        'network_pixels_at_0.5': int((p_net >= 0.5).sum()),
        'seconds': round(time.time() - t0, 1),
    }
    print(f'  -> {summary["class_percent"]} ({summary["seconds"]} s)')
    return summary


# =============================================================================
def main():
    import torch
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('images', nargs='*', help='image file(s): GeoTIFF / PNG / JPG')
    ap.add_argument('--dir', type=str, default=None,
                    help='process every image in this folder')
    ap.add_argument('--out', type=str, default=None,
                    help=f'output folder (default: {DEFAULT_OUT})')
    ap.add_argument('--weights', type=str, default=None,
                    help='folder with best.pt + temperature.json (default: run/src/)')
    ap.add_argument('--crs', type=str, default=None,
                    help='CRS to assign to inputs whose embedded CRS is missing or '
                         'unusable, e.g. EPSG:26985')
    ap.add_argument('--context-px', type=int, default=192,
                    help='imagery context from neighbouring inputs padded around each '
                         'image before inference (0 = per-image only)')
    ap.add_argument('--overlap', type=int, default=128,
                    help='sliding-window overlap in px (window = 512)')
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--overwrite', action='store_true',
                    help='re-run images whose rasters already exist')
    ap.add_argument('--no-overlay', action='store_true')
    ap.add_argument('--overlay-px', type=int, default=2048,
                    help='overlays larger than this are downscaled (quick-look only)')
    ap.add_argument('--inspect', action='store_true',
                    help='print the input inspection report and exit (no model run)')
    ap.add_argument('--force', action='store_true',
                    help='run even if the inspection says an input needs preparing')
    ap.add_argument('--confidence-format', type=str, default='uint8',
                    choices=['uint8', 'float32'],
                    help='uint8 (scaled, ~10x smaller files) or float32 confidence rasters')
    args = ap.parse_args()

    paths = [Path(p) for p in args.images]
    if args.dir:
        paths += sorted(p for p in Path(args.dir).iterdir()
                        if p.suffix.lower() in IMG_EXT)
    assert paths, 'no input images — pass file paths and/or --dir'
    missing = [p for p in paths if not p.exists()]
    assert not missing, f'not found: {missing}'

    # ---- inspect every input first: format, size, CRS, resolution, coverage ----
    print(f'Inspecting {len(paths)} input image(s)...')
    FULL = 3 if len(paths) > 6 else len(paths)       # full report for the first few,
    infos = []                                        # one line each for the rest
    for k, p in enumerate(paths):
        info = describe_image(p, args.crs, quiet=k >= FULL)
        if k >= FULL:
            print('  ' + info['summary'])
        infos.append(info)
    not_ready = [i['name'] for i in infos if not i['ready']]
    if args.inspect:
        print(f'\n{len(paths) - len(not_ready)} ready, {len(not_ready)} need preparing.')
        return
    if not_ready and not args.force:
        raise SystemExit(f'\n{len(not_ready)} input(s) need preparing first (see the status '
                         f'lines above): {not_ready[:5]}{" ..." if len(not_ready) > 5 else ""}\n'
                         f'Run prepare_image.py on them, or pass --force to run anyway.')
    print()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weights_dir = resolve_weights(args.weights)
    print('Loading the model...')
    model, cfg, temps = load_model(weights_dir, device)

    out_root = Path(args.out) if args.out else DEFAULT_OUT
    out_root.mkdir(parents=True, exist_ok=True)
    index = build_index(paths, args.crs)
    print('── run ' + '─' * 57)
    print(f'  inputs      {len(paths)} image(s), {len(index)} georeferenced')
    print(f'  inference   {cfg.IMG_SIZE}-px windows, overlap {args.overlap} px, '
          f'context {args.context_px} px from neighbouring tiles, batch {args.batch}')
    print(f'  outputs     {out_root}  (classes/, confidence/ [{args.confidence_format}], '
          f'{"overlay/, " if not args.no_overlay else ""}stage1_summary.json)')
    print()

    summaries, n_skip = [], 0
    for p in paths:
        outs = outputs_for(out_root, p.stem)
        if not args.overwrite and all(o.exists() for o in outs.values()):
            n_skip += 1
            continue
        summaries.append(process_image(p, model, cfg, temps, device, index, out_root, args))

    summary_path = out_root / 'stage1_summary.json'
    prev = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    images = {s['image']: s for s in prev.get('images', [])}
    for s in summaries:
        images[s['image']] = s
    summary_path.write_text(json.dumps({
        'weights': str(weights_dir), 'temperatures': temps,
        'params': {'window': cfg.IMG_SIZE, 'overlap': args.overlap,
                   'context_px': args.context_px, 'crs_override': args.crs,
                   'confidence_format': args.confidence_format},
        'confidence': 'p = sigmoid(logit / T_head); uint8 rasters store p*255 '
                      'with band scale 1/255',
        'classes': dict(enumerate(CLASS_NAMES)),
        'images': list(images.values()),
    }, indent=2))
    print(f'\nStage 1 done: {len(summaries)} image(s) processed, {n_skip} already '
          f'cached -> {out_root}')


if __name__ == '__main__':
    main()
