"""make_input.py — cut a small, self-contained example input out of a full imagery run.

OBJECTIVE
  A whole town is thousands of tiles (Lexington: 7,331 ≈ 2 GB), too much to
  ship as an example. This script copies a block of tiles from a full
  download_MassGIS2025AerialImagery region into example/input/<REGION>/ together
  with a Tile_Mappings.csv restricted to those tiles and the imagery_info.json,
  so the block is a complete, runnable imagery_root of its own.

  Pick a block that contains bike facilities (open the town's
  main/<region>_bike_facilities.geojson over the tiles in QGIS and note the
  tile names at a good spot). One 6 × 6 block (= one centerline chunk,
  ≈ 690 m square, ≈ 36 tiles, ≈ 10 MB) is enough to exercise every stage;
  2 × 2 blocks (12 × 12 tiles) give the gap and intersection stages more to do.

INPUT
  --src DIR        the full region folder: <imagery_root>/<REGION>/ (tiles/, Tile_Mappings.csv, imagery_info.json)
  --tile NAME      a tile at the top-left of the block, e.g. tile_px29184_py15360.jpg
  --n N            block size in tiles per side (default 6)
  --out DIR        example/input/<REGION>/ (default: next to this script, same REGION as --src)
  --bbox X0 Y0 X1 Y1   alternative to --tile/--n: map coordinates (imagery CRS) of the block

OUTPUT
  <out>/tiles/*.jpg, <out>/Tile_Mappings.csv, <out>/imagery_info.json  (+ example_input.json: what was cut)

USAGE  (from bicyclist_network/)
  python example/make_input.py --src ../download_MassGIS2025AerialImagery/output/LEXINGTON --tile tile_px29184_py15360.jpg --n 6
  python example/make_input.py --src ../download_MassGIS2025AerialImagery/output/BOSTON --bbox 328500 4690000 329500 4691000
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
_PXPY = re.compile(r"px(\d+)_py(\d+)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", required=True, help="<imagery_root>/<REGION>/ of a full run")
    ap.add_argument("--tile", help="top-left tile of the block, e.g. tile_px29184_py15360.jpg")
    ap.add_argument("--n", type=int, default=6, help="tiles per side (default 6)")
    ap.add_argument("--bbox", nargs=4, type=float, metavar=("X0", "Y0", "X1", "Y1"),
                    help="map-coordinate box instead of --tile/--n")
    ap.add_argument("--out", default=None, help="default example/input/<REGION>/")
    a = ap.parse_args(argv)
    src = Path(a.src).expanduser().resolve()
    csv_in, info_in = src / "Tile_Mappings.csv", src / "imagery_info.json"
    if not csv_in.exists():
        raise SystemExit(f"{csv_in} not found — --src must be a region folder written by tile.py")
    info = json.loads(info_in.read_text()) if info_in.exists() else {}
    region = info.get("region") or src.name
    out = Path(a.out).expanduser() if a.out else HERE / "input" / region
    rows = list(csv.DictReader(open(csv_in)))
    tile_px = int(info.get("tile_px", 1024))
    stride = int(info.get("stride_px", tile_px * 0.75))
    res = float(info.get("resolution_m", 0.15))

    if a.bbox:
        x0, y0, x1, y1 = a.bbox
        keep = [r for r in rows
                if x0 <= float(r["CRS_X"]) <= x1 - tile_px * res
                and y0 + tile_px * res <= float(r["CRS_Y"]) <= y1]
    else:
        if not a.tile:
            raise SystemExit("give --tile <top-left tile> [--n] or --bbox")
        m = _PXPY.search(a.tile)
        if not m:
            raise SystemExit(f"{a.tile}: expected tile_px<X>_py<Y>.jpg")
        px0, py0 = int(m.group(1)), int(m.group(2))
        px1, py1 = px0 + a.n * stride, py0 + a.n * stride
        keep = [r for r in rows if px0 <= int(r["Pixel_X"]) < px1 and py0 <= int(r["Pixel_Y"]) < py1]
    if not keep:
        raise SystemExit("no tiles in that block — check the tile name / bbox against Tile_Mappings.csv")

    (out / "tiles").mkdir(parents=True, exist_ok=True)
    missing = 0
    for r in keep:
        f = src / "tiles" / r["image_name"]
        if f.exists():
            shutil.copyfile(f, out / "tiles" / f.name)
        else:
            missing += 1
    with open(out / "Tile_Mappings.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image_name", "CRS_X", "CRS_Y", "Pixel_X", "Pixel_Y"])
        w.writeheader()
        w.writerows({k: r[k] for k in w.fieldnames} for r in keep)
    if info:
        info = dict(info, example_of=str(src), mosaic=None)
        (out / "imagery_info.json").write_text(json.dumps(info, indent=1))
    xs = [float(r["CRS_X"]) for r in keep]
    ys = [float(r["CRS_Y"]) for r in keep]
    meta = dict(region=region, source=str(src), n_tiles=len(keep),
                extent=[min(xs), min(ys) - tile_px * res, max(xs) + tile_px * res, max(ys)],
                tiles=sorted(r["image_name"] for r in keep))
    (out / "example_input.json").write_text(json.dumps(meta, indent=1))
    size = sum(f.stat().st_size for f in (out / "tiles").glob("*.jpg")) / 1e6
    print(f"{region}: {len(keep)} tiles → {out}  ({size:.0f} MB"
          + (f", {missing} tile files missing in --src" if missing else "") + ")")
    print(f"  extent {meta['extent'][0]:.0f},{meta['extent'][1]:.0f} – {meta['extent'][2]:.0f},{meta['extent'][3]:.0f}"
          f"  ({(meta['extent'][2]-meta['extent'][0]):.0f} × {(meta['extent'][3]-meta['extent'][1]):.0f} m)")
    print(f"→ bikelane config -c example/{region.lower()}.yaml")


if __name__ == "__main__":
    main()
