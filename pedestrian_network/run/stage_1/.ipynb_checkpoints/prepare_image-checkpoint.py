"""Prepare imagery for stage 1 — any raster -> RGB GeoTIFF tiles at 0.08 m/px.

OBJECTIVE
  Bring imagery that is not model-ready into the exact form stage 1 expects:
  3-band uint8 GeoTIFF at the model's native 0.08 m/px, with a usable CRS,
  split into tiles small enough for stage 1's memory. Run it whenever the
  inspection report says NEEDS PREPARE (JPEG2000 orthos, 4-band RGB+NIR,
  coarser or finer resolution, one huge ortho).

INPUT
  One or more rasters (file paths and/or --dir <folder>): anything GDAL reads
  (.jp2, .tif, .img, .vrt, ...), georeferenced in a metric CRS (imagery in
  degrees must be reprojected first). Bands beyond RGB are dropped, single-band
  imagery is replicated to RGB, non-uint8 pixels are rescaled to uint8.

OUTPUT  (--out, default run/output/prepared/)
  <stem>_r<row>_c<col>.tif   georeferenced, deflate-compressed RGB tiles of
                             about --tile-px pixels (nearly equal splits), or
  <stem>.tif                 one image when --tile-px 0 or the image is small
  Feed the output folder to segmentation.py --dir.

HOW IT WORKS (functions)
  main()          inspection report per input (common.describe_image), then
                  prepare() each; --inspect stops after the reports
  prepare()       read bands -> to_uint8() -> rasterio.warp.reproject from the
                  source pixel size to --res in the same CRS (a pure rescale)
                  -> assign --crs if the file's tag is unusable
                  -> even_splits() -> write tiles
  even_splits()   splits n pixels into ceil(n / tile_px) nearly equal pieces,
                  so there are no sliver tiles at the edge
  to_uint8()      max-value rescale for 16-bit imagery

TUNING — parameters that matter
  --res 0.08          target m/px. Keep the model's native resolution: the
                      model (version 0) has only seen 0.08 m/px imagery.
  --resampling cubic  cubic | bilinear | nearest
  --tile-px 2048      tile size. Only bounds memory downstream (stage 1 pads
                      tiles with their neighbours, so predictions do not depend
                      on it). 0 = keep one image (~50 B/px RAM in stage 1).
  --crs EPSG:xxxx     CRS to write when the source tag cannot be resolved
  Upsampling coarse imagery adds no real detail: expect softer predictions
  than on native 0.08 m/px imagery. The model was trained on DC imagery only.

USAGE
  python prepare_image.py --inspect --dir ../../example/input/mass_2025
  python prepare_image.py --dir ../../example/input/mass_2025 --out ../../example/output/mass_2025/prepared
  python prepare_image.py ortho.jp2 --out prepared/ --tile-px 4096 --crs EPSG:6348
"""
import sys
import math
import argparse
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))            # run/common.py
from common import ensure_proj, describe_image, crs_is_usable, MODEL_RES_M

DEFAULT_OUT = HERE.parent / 'output' / 'prepared'
RASTER_EXT = ('.jp2', '.tif', '.tiff', '.img', '.vrt', '.png', '.jpg', '.jpeg')


def to_uint8(band):
    if band.dtype == np.uint8:
        return band
    b = band.astype(np.float32)
    return (b / max(float(b.max()), 1.0) * 255.0).astype(np.uint8)


def even_splits(n, tile_px):
    """Split n pixels into ceil(n / tile_px) nearly equal pieces."""
    k = max(math.ceil(n / tile_px), 1)
    base, extra = divmod(n, k)
    out, start = [], 0
    for i in range(k):
        size = base + (1 if i < extra else 0)
        out.append((start, size))
        start += size
    return out


def prepare(path, out_dir, res, resampling, tile_px, crs_override):
    import rasterio
    from rasterio.crs import CRS
    from rasterio.warp import reproject
    from rasterio.transform import Affine

    with rasterio.open(path) as ds:
        src_t, src_crs = ds.transform, ds.crs
        src_res = float(abs(src_t.a))
        n_bands = min(ds.count, 3)
        src = np.stack([to_uint8(ds.read(b)) for b in range(1, n_bands + 1)])
        if n_bands < 3:
            src = np.repeat(src[:1], 3, axis=0)
        if crs_override:
            crs = CRS.from_user_input(crs_override)
        elif crs_is_usable(src_crs):
            crs = src_crs
        else:
            crs = None
            print(f'  WARN: {path.name} has no usable CRS and no --crs given — tiles will '
                  f'carry the transform only (no lat/lon export later)')
        if crs is not None and crs.is_geographic:
            raise SystemExit(f'{path.name} is in a geographic CRS (degrees); reproject it '
                             f'to a metric CRS (e.g. the local UTM zone) before preparing.')
        if abs(src_res - res) < 1e-9:
            dst, dst_t = src, src_t
        else:
            H = round(ds.height * src_res / res)
            W = round(ds.width * src_res / res)
            dst_t = Affine(res, 0.0, src_t.c, 0.0, -res, src_t.f)
            dst = np.zeros((3, H, W), np.uint8)
            work_crs = crs if crs is not None else CRS.from_epsg(3857)   # any metric CRS works for a pure rescale
            for b in range(3):
                reproject(source=src[b], destination=dst[b],
                          src_transform=src_t, src_crs=work_crs,
                          dst_transform=dst_t, dst_crs=work_crs,
                          resampling=resampling)
    _, H, W = dst.shape

    def write(arr, t, name):
        out = out_dir / name
        with rasterio.open(out, 'w', driver='GTiff', height=arr.shape[1], width=arr.shape[2],
                           count=3, dtype='uint8', compress='deflate', predictor=2,
                           tiled=True, blockxsize=512, blockysize=512, BIGTIFF='IF_SAFER',
                           transform=t, crs=crs) as w:
            w.write(arr)
        return out

    written = []
    if tile_px and max(H, W) > tile_px:
        rows, cols = even_splits(H, tile_px), even_splits(W, tile_px)
        for i, (r0, rh) in enumerate(rows):
            for j, (c0, cw) in enumerate(cols):
                t = dst_t * Affine.translation(c0, r0)
                written.append(write(dst[:, r0:r0 + rh, c0:c0 + cw], t,
                                     f'{path.stem}_r{i:02d}_c{j:02d}.tif'))
        print(f'  -> {len(rows)} x {len(cols)} = {len(written)} tiles of ~{rows[0][1]} x {cols[0][1]} px '
              f'@ {res} m/px, CRS {crs if crs is not None else "none"} -> {out_dir}')
    else:
        written.append(write(dst, dst_t, f'{path.stem}.tif'))
        print(f'  -> {W} x {H} px @ {res} m/px, CRS {crs if crs is not None else "none"} -> {written[0]}')
    return written


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('images', nargs='*', help='raster file(s) (.jp2 / .tif / ...)')
    ap.add_argument('--dir', type=str, default=None,
                    help='prepare every raster in this folder')
    ap.add_argument('--out', type=str, default=None,
                    help=f'output folder (default: {DEFAULT_OUT})')
    ap.add_argument('--res', type=float, default=MODEL_RES_M,
                    help=f'target resolution in m/px (model native: {MODEL_RES_M})')
    ap.add_argument('--resampling', type=str, default='cubic',
                    choices=['cubic', 'bilinear', 'nearest'])
    ap.add_argument('--tile-px', type=int, default=2048,
                    help='split outputs into tiles of about this size (0 = one image)')
    ap.add_argument('--crs', type=str, default=None,
                    help='CRS to assign when the file\'s CRS tag is unusable, e.g. EPSG:6348')
    ap.add_argument('--inspect', action='store_true',
                    help='only print the inspection report, write nothing')
    args = ap.parse_args()

    paths = [Path(p) for p in args.images]
    if args.dir:
        paths += sorted(p for p in Path(args.dir).iterdir()
                        if p.suffix.lower() in RASTER_EXT)
    assert paths, 'no input images — pass file paths and/or --dir'
    missing = [p for p in paths if not p.exists()]
    assert not missing, f'not found: {missing}'

    ensure_proj()
    infos = [describe_image(p, args.crs, tile_px=args.tile_px or 2048) for p in paths]
    if args.inspect:
        return
    bad = [i['name'] for i in infos if i.get('error')]
    assert not bad, f'cannot read: {bad}'

    from rasterio.warp import Resampling
    resampling = getattr(Resampling, args.resampling)
    out_dir = Path(args.out) if args.out else DEFAULT_OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    print()
    n = 0
    for p in paths:
        print(f'{p.name}: preparing...')
        n += len(prepare(p, out_dir, args.res, resampling, args.tile_px, args.crs))
    print(f'\nDone: {len(paths)} image(s) -> {n} file(s) in {out_dir}')
    print(f'Next:  python segmentation.py --dir {out_dir} --out <output folder>')


if __name__ == '__main__':
    main()
