"""fetch.py — MassGIS 2025 orthophoto tiles for an area → one mosaic (VRT, optionally BigTIFF).

OBJECTIVE
  Get the 15 cm MassGIS/MassDOT 2025 aerial imagery covering an area onto disk
  as a single georeferenced mosaic, ready for tile.py. Nothing is placed by hand.

HOW IT WORKS
  1. The MassGIS orthophoto index shapefile (COQ2025INDEX_POLY) is downloaded
     on first use into <out>/massdot_index/.
  2. The area is either a town (--city/--state: polygon from the Census county
     subdivisions via pygris; the tiles intersecting its bounding box are taken)
     or any polygon file (--boundary: the tiles intersecting the polygon itself).
  3. The zips of those index tiles are downloaded (resumable: existing zips are
     skipped), the .jp2 files extracted, and a GDAL VRT built over them.
  4. --build-tif additionally writes one LZW BigTIFF (sensible for a single
     town; for a whole region the VRT is enough — tile.py reads it directly —
     and a BigTIFF would be hundreds of GB).

INPUT
  --city "Lexington" --state MA          or          --boundary area.geojson [--name boston_mpo]
  --out DIR        output root (default ./output)
  --build-tif      also write Merged_<REGION>.tif
  --check          report what exists and whether gdal is available, download nothing

OUTPUT   (<out>/<REGION>/, see common.py for the full layout)
  ZipFiles/*.zip, Images/*.jp2, Merged_<REGION>.vrt (+ .txt file list) [, Merged_<REGION>.tif]

USAGE
  pip install -r requirements.txt          # + gdal-bin on PATH
  python fetch.py --city Lexington --state MA --check
  python fetch.py --city Lexington --state MA               # (long) ~44 zips, ~7 GB
  python fetch.py --boundary Boundaries.geojson --name boston_mpo   # (very long) run in tmux
  → next: python tile.py  (same area arguments)
"""
from __future__ import annotations

import argparse
import os
import subprocess
import zipfile

from common import INDEX_URL, TOWNS_YEAR, Area, add_area_args


def _download_index(url: str, shp):
    """Fetch the MassGIS index zip and extract it next to the expected .shp."""
    import io
    import requests
    shp.parent.mkdir(parents=True, exist_ok=True)
    print(f"  downloading index {url}")
    r = requests.get(url, timeout=300)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        z.extractall(shp.parent)
    if not shp.exists():                    # zip may nest a folder; find the .shp
        found = next(shp.parent.rglob(shp.name), None)
        if found is None:
            raise SystemExit(f"index zip did not contain {shp.name}")
        for f in found.parent.iterdir():
            f.rename(shp.parent / f.name)
    print(f"  index ready: {shp}")


def _gdal_ok(tool: str) -> bool:
    return subprocess.run(f"which {tool}", shell=True, capture_output=True).returncode == 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_area_args(ap)
    ap.add_argument("--build-tif", action="store_true", help="also write one LZW BigTIFF mosaic")
    ap.add_argument("--check", action="store_true", help="report inputs, download nothing")
    a = ap.parse_args(argv)
    area = Area(a)
    print(f"fetch  {area}\n  index: {area.index_shp}\n  mosaic → {area.vrt}")
    if a.check:
        print(f"  {'ok ' if area.index_shp.exists() else 'will download'} {area.index_shp}")
        for t in ("gdalbuildvrt", "gdal_translate"):
            print(f"  {t}: {'ok' if _gdal_ok(t) else 'MISSING (install gdal-bin)'}")
        n_zip = len(list((area.region_dir / "ZipFiles").glob("*.zip"))) if area.region_dir.exists() else 0
        print(f"  {n_zip} zips already downloaded; {'ok ' if area.vrt.exists() else 'no '} {area.vrt}")
        return
    if not _gdal_ok("gdalbuildvrt"):
        raise SystemExit("gdalbuildvrt not on PATH — install the GDAL command-line tools (gdal-bin)")
    import geopandas as gpd
    import requests
    from tqdm import tqdm

    if not area.index_shp.exists():
        _download_index(INDEX_URL, area.index_shp)
    idx = gpd.read_file(area.index_shp)
    if area.boundary:
        aoi = gpd.read_file(area.boundary).to_crs(idx.crs)
        geom = aoi.geometry.union_all() if hasattr(aoi.geometry, "union_all") else aoi.geometry.unary_union
        overlap = idx[idx.intersects(geom)]              # the polygon itself, not its bbox
        print(f"  {len(overlap)} index tiles intersect the boundary ({len(aoi)} polygons)")
    else:
        import pygris
        from shapely.geometry import box
        towns = pygris.county_subdivisions(state=area.state, year=TOWNS_YEAR)
        town = towns[towns["NAME"].str.lower() == area.city.lower()].to_crs(idx.crs)
        if town.empty:
            raise SystemExit(f"town '{area.city}' not found in {area.state} county subdivisions")
        overlap = idx[idx.intersects(box(*town.total_bounds))]
        print(f"  {len(overlap)} index tiles intersect the town bbox")
    gb = len(overlap) * 0.15
    print(f"  ≈ {gb:.0f} GB of JP2 to download" + (" — run in tmux" if gb > 20 else ""))

    zip_dir, img_dir = area.region_dir / "ZipFiles", area.region_dir / "Images"
    zip_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)
    for url in tqdm(overlap["URL"], desc="  download"):
        dst = zip_dir / url.split("/")[-1]
        if dst.exists():
            continue
        try:
            r = requests.get(url, stream=True, timeout=120)
            r.raise_for_status()
            with open(dst, "wb") as f:
                for chunk in r.iter_content(8192):
                    f.write(chunk)
        except Exception as e:
            print(f"\n  [error] {url}: {e}")
    for zp in tqdm(sorted(zip_dir.glob("*.zip")), desc="  extract"):
        try:
            with zipfile.ZipFile(zp) as z:
                for info in z.infolist():
                    if info.filename.lower().endswith(".jp2"):
                        dst = img_dir / os.path.basename(info.filename)
                        if not dst.exists():
                            with z.open(info) as s, open(dst, "wb") as d:
                                d.write(s.read())
        except zipfile.BadZipFile:
            print(f"\n  [error] corrupt zip: {zp.name}")

    jp2 = sorted(str(f) for f in img_dir.glob("*.jp2"))
    if not jp2:
        raise SystemExit("no .jp2 extracted — check the download errors above")
    lst = area.vrt.with_suffix(".txt")                  # file list: avoids "argument list too long"
    lst.write_text("\n".join(jp2) + "\n")
    subprocess.run(["gdalbuildvrt", "-input_file_list", str(lst), str(area.vrt)], check=True)
    if a.build_tif:
        subprocess.run(["gdal_translate", str(area.vrt), str(area.mosaic), "-co", "COMPRESS=LZW",
                        "-co", "TILED=YES", "-co", "BIGTIFF=YES", "-co", "NUM_THREADS=ALL_CPUS"],
                       check=True)
        print(f"  mosaic written: {area.mosaic}")
    else:
        print(f"  mosaic VRT written: {area.vrt}  ({len(jp2)} JP2; tile.py reads it directly — "
              "--build-tif for a single BigTIFF)")
    print("→ next: python tile.py  (same area arguments)")


if __name__ == "__main__":
    main()
