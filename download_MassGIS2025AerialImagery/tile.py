"""tile.py — mosaic (VRT or GeoTIFF) → 1024 px JPG tiles (25 % overlap) + Tile_Mappings.csv

OBJECTIVE
  Cut the mosaic from fetch.py into the tiles the analysis tools consume:
  tile_px<X>_py<Y>.jpg with X/Y the global mosaic pixel offset, and a CSV
  mapping each tile to the CRS coordinate of its top-left corner. Every later
  stage relies on this file and naming; imagery_info.json records the CRS and
  pixel size so the analysis never has to guess them.

HOW IT WORKS
  A window of TILE_PX × TILE_PX is read every stride = TILE_PX·(1−overlap)
  pixels. Tiles with ≥ EMPTY_THRESHOLD nodata are dropped (0.5; the earlier
  value 0 silently dropped 70 % of Boston because every tile touching the
  boundary has some nodata). Resumable: Tile_Mappings.csv (kept tiles) and
  tiles_dropped.txt (nodata tiles) are appended to, and tiles listed in either
  are skipped on re-run. Each worker opens the mosaic once (a VRT over ~2,000
  JP2s is expensive to open) with a small GDAL cache, ~0.5 GB per worker.

INPUT
  --city/--state or --boundary/--name, --out     the same as fetch.py
  --workers N      parallel processes (default min(cpu, 16))
  --cleanup        after tiling completes, delete ZipFiles/, Images/, the VRT (+ list) and
                   the BigTIFF — hundreds of GB for a region. tiles/ + Tile_Mappings.csv are
                   all the analysis needs; re-run fetch.py to get the sources back.
  --check          report the mosaic and what has been tiled so far, cut nothing

OUTPUT   (<out>/<REGION>/)
  tiles/tile_px<X>_py<Y>.jpg     BGR JPG, TILE_PX square
  Tile_Mappings.csv              image_name, CRS_X, CRS_Y, Pixel_X, Pixel_Y  (top-left corner)
  imagery_info.json              region, crs, resolution_m, tile_px, tile_overlap, mosaic size …
  tiles_dropped.txt              "x,y" per dropped (nodata) window

USAGE
  python tile.py --city Lexington --state MA --check
  python tile.py --city Lexington --state MA [--workers 16] [--cleanup]
  → next: in bicyclist_network/, set the same area in bikelane.yaml and run `bikelane predict`
Expected: several thousand tiles per town; the dropped counter is printed.
"""
from __future__ import annotations

import argparse
import csv
import json
import multiprocessing
import os
import time
from pathlib import Path

import numpy as np

from common import EMPTY_THRESHOLD, RESOLUTION_M, TILE_OVERLAP, TILE_PX, Area, add_area_args

_W = {}


def _init(path: str):
    os.environ.setdefault("GDAL_CACHEMAX", "256")          # MB, per process
    os.environ.setdefault("GDAL_MAX_DATASET_POOL_SIZE", "64")
    import rasterio
    _W["src"] = rasterio.open(path)


def _tile(args):
    import cv2
    from rasterio.windows import Window
    out_dir, x, y, size, empty_th = args
    src = _W["src"]
    t = src.read(window=Window(x, y, size, size))
    ch = src.count
    sx, sy = src.transform * (x, y)
    empty = np.sum(t[3] == 0) if ch == 4 else np.sum(np.all(t[:3] == 0, axis=0))
    if empty / (size * size) >= empty_th:
        return None, x, y
    hwc = np.transpose(t, (1, 2, 0))
    bgr = (cv2.cvtColor(hwc, cv2.COLOR_RGBA2BGR) if ch == 4 else
           cv2.cvtColor(hwc, cv2.COLOR_RGB2BGR) if ch >= 3 else hwc)
    name = f"tile_px{x}_py{y}.jpg"
    cv2.imwrite(str(Path(out_dir) / name), bgr)
    return [name, sx, sy, x, y], x, y


def _mosaic_of(area: Area) -> Path:
    if area.mosaic.exists():
        return area.mosaic
    if area.vrt.exists():
        return area.vrt
    raise SystemExit(f"no mosaic for {area.region}: run fetch.py first ({area.vrt})")


def _write_info(area: Area, src, size, stride, mosaic: Path):
    info = {
        "region": area.region, "area": area.label,
        "crs": src.crs.to_string() if src.crs else None,
        "epsg": src.crs.to_epsg() if src.crs else None,
        "resolution_m": float(src.transform.a),
        "tile_px": size, "tile_overlap": TILE_OVERLAP, "stride_px": stride,
        "empty_threshold": EMPTY_THRESHOLD,
        "mosaic": str(mosaic), "mosaic_width_px": src.width, "mosaic_height_px": src.height,
        "origin_x": src.transform.c, "origin_y": src.transform.f,
        "bands": src.count,
        "source": "MassGIS 2025 orthophotos (COQ2025)",
        "written": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    area.info_json.write_text(json.dumps(info, indent=1))
    return info


def _cleanup_sources(area: Area, mosaic: Path):
    """Delete ZipFiles/, Images/ (JP2), the VRT and its file list, and the BigTIFF."""
    import shutil
    freed = 0
    for sub in ("ZipFiles", "Images"):
        d = area.region_dir / sub
        if d.exists():
            freed += sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
            shutil.rmtree(d, ignore_errors=True)
    for f in (area.vrt, area.vrt.with_suffix(".txt"), area.mosaic):
        if f.exists():
            freed += f.stat().st_size
            f.unlink()
    print(f"  cleanup: removed source imagery under {area.region_dir} ({freed/1e9:.0f} GB freed)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_area_args(ap)
    ap.add_argument("--workers", type=int, default=None,
                    help="parallel processes (default min(cpu, 16); ~0.5 GB RAM each)")
    ap.add_argument("--cleanup", action="store_true",
                    help="delete the downloaded zips, JP2s, VRT and BigTIFF once tiling is complete")
    ap.add_argument("--check", action="store_true", help="report inputs, cut nothing")
    a = ap.parse_args(argv)
    area = Area(a)
    out_dir, csv_path = area.tiles_dir, area.mappings_csv
    dropped_path = area.region_dir / "tiles_dropped.txt"
    print(f"tile  {area}\n  tiles → {out_dir}\n  mapping → {csv_path}")
    if a.check:
        m = area.mosaic if area.mosaic.exists() else area.vrt
        print(f"  {'ok ' if m.exists() else 'MISSING'} {m}")
        n = (len([1 for _ in open(csv_path)]) - 1) if csv_path.exists() else 0
        print(f"  {n} tiles already in Tile_Mappings.csv; "
              f"{'ok ' if area.info_json.exists() else 'no '} {area.info_json}")
        return
    mosaic = _mosaic_of(area)
    if area.boundary:
        print("  note: tiles outside the boundary polygon but inside its bbox are also cut "
              "(only nodata tiles are dropped); clip outputs to the boundary afterwards if needed")
    import rasterio
    size = TILE_PX
    stride = int(size * (1 - TILE_OVERLAP))
    out_dir.mkdir(parents=True, exist_ok=True)
    with rasterio.open(mosaic) as src:
        w, h, tr = src.width, src.height, src.transform
        if abs(tr.a - RESOLUTION_M) > 1e-3:
            print(f"  WARNING: mosaic pixel size {tr.a} ≠ expected {RESOLUTION_M} m — "
                  "set resolution_m in the analysis config to match")
        if src.crs is None:
            print("  WARNING: mosaic has no CRS; the analysis will have to be told the CRS explicitly")
        info = _write_info(area, src, size, stride, mosaic)
    print(f"  crs {info['crs']}  {info['resolution_m']} m/px  → {area.info_json.name}")

    # resume: skip anything already decided
    done = set()
    if csv_path.exists():
        with open(csv_path) as f:
            for r in csv.DictReader(f):
                done.add((int(r["Pixel_X"]), int(r["Pixel_Y"])))
    if dropped_path.exists():
        for line in dropped_path.read_text().split():
            x, y = line.split(",")
            done.add((int(x), int(y)))
    tasks = [(str(out_dir), x, y, size, EMPTY_THRESHOLD)
             for y in range(0, h - size + 1, stride) for x in range(0, w - size + 1, stride)
             if (x, y) not in done]
    n_all = len(tasks) + len(done)
    workers = a.workers or min(multiprocessing.cpu_count(), 16)
    workers = max(1, min(int(workers), len(tasks) or 1))
    print(f"  {w}×{h} px, stride {stride} → {n_all} candidate tiles, {len(done)} already done, "
          f"{len(tasks)} to do, {workers} workers")
    if not tasks:
        if a.cleanup:
            _cleanup_sources(area, mosaic)
        print("→ next: bicyclist_network — bikelane predict")
        return

    from concurrent.futures import ProcessPoolExecutor
    from concurrent.futures.process import BrokenProcessPool
    saved = dropped = 0
    new_csv = not csv_path.exists() or csv_path.stat().st_size == 0
    ctx = multiprocessing.get_context("fork")
    BATCH = 2000                       # futures in flight; keeps the parent small
    restarts = 0
    with open(csv_path, "a", newline="") as f, open(dropped_path, "a") as fd:
        wr = csv.writer(f)
        if new_csv:
            wr.writerow(["image_name", "CRS_X", "CRS_Y", "Pixel_X", "Pixel_Y"])
        pending = list(tasks)
        done_n = 0
        while pending:
            batch, pending = pending[:BATCH], pending[BATCH:]
            try:
                with ProcessPoolExecutor(workers, mp_context=ctx, initializer=_init,
                                         initargs=(str(mosaic),)) as ex:
                    for r, x, y in ex.map(_tile, batch, chunksize=8):
                        if r is None:
                            dropped += 1
                            fd.write(f"{x},{y}\n")
                        else:
                            wr.writerow(r)
                            saved += 1
                        done_n += 1
                        if done_n % 1000 == 0:
                            f.flush(); fd.flush()
                            print(f"  {done_n}/{len(tasks)} (saved {saved}, dropped {dropped})", flush=True)
            except BrokenProcessPool:
                # a worker was killed (usually OOM while decoding a JP2). Everything written so
                # far is on disk; redo this batch with a fresh pool and fewer workers.
                f.flush(); fd.flush()
                restarts += 1
                done_set = set()
                if csv_path.exists():
                    with open(csv_path) as fr:
                        done_set |= {(int(r["Pixel_X"]), int(r["Pixel_Y"])) for r in csv.DictReader(fr)}
                for line in dropped_path.read_text().split():
                    x_, y_ = line.split(",")
                    done_set.add((int(x_), int(y_)))
                batch = [t for t in batch if (t[1], t[2]) not in done_set]
                pending = batch + pending
                workers = max(2, workers * 2 // 3)
                print(f"  worker pool died (restart {restarts}); continuing with {workers} workers, "
                      f"{len(pending)} tiles left", flush=True)
                if restarts > 20:
                    raise SystemExit("tile: too many worker crashes — check dmesg for OOM, "
                                     "re-run with --workers 4")
        f.flush(); fd.flush()
    print(f"  saved {saved} tiles, dropped {dropped} (≥{EMPTY_THRESHOLD:.0%} empty); "
          f"total kept {len([1 for _ in open(csv_path)]) - 1}")
    if a.cleanup:
        _cleanup_sources(area, mosaic)
    print("→ next: bicyclist_network — set the same area in bikelane.yaml, then bikelane predict")


if __name__ == "__main__":
    main()
