"""Shared by fetch.py and tile.py: the area, its REGION tag, and the output layout.

Output layout (everything under --out, default ./output next to these scripts):

  output/
    massdot_index/COQ2025INDEX_POLY.shp     MassGIS 2025 orthophoto index (downloaded once)
    <REGION>/
      ZipFiles/*.zip                        downloaded JP2 zips        ┐ sources; deleted by
      Images/*.jp2                          extracted JPEG2000 tiles   │ `tile.py --cleanup`
      Merged_<REGION>.vrt (+ .txt list)     GDAL virtual mosaic        ┘
      Merged_<REGION>.tif                   BigTIFF mosaic, only with fetch.py --build-tif
      tiles/tile_px<X>_py<Y>.jpg            1024 px JPG tiles, 25 % overlap      ┐ what the
      Tile_Mappings.csv                     image_name, CRS_X, CRS_Y, Pixel_X, Pixel_Y │ analysis
      imagery_info.json                     region, crs, resolution_m, tile_px, … ┘ reads
      tiles_dropped.txt                     nodata tiles that were skipped (resume bookkeeping)

<REGION> is the area name upper-cased with non-alphanumerics → "_"
(`--city "Fall River"` → FALL_RIVER; `--boundary x.geojson --name boston_mpo` →
BOSTON_MPO). bicyclist_network/ derives the same tag from its bikelane.yaml, so
give both tools the same area and they meet in output/<REGION>/.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE / "output"
INDEX_URL = ("https://s3.dualstack.us-east-1.amazonaws.com/download.massgis.digital.mass.gov/"
             "images/coq2025_jp2/COQ2025INDEX_POLY.zip")
INDEX_SHP = "massdot_index/COQ2025INDEX_POLY.shp"
TOWNS_YEAR = 2025          # pygris county_subdivisions year (city/state mode)
RESOLUTION_M = 0.15        # MassGIS 2025 orthophotos: 15 cm ground sampling distance
TILE_PX = 1024
TILE_OVERLAP = 0.25
EMPTY_THRESHOLD = 0.50     # drop tiles with >= this fraction of nodata pixels


def region_slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").upper()


def add_area_args(ap: argparse.ArgumentParser) -> None:
    g = ap.add_argument_group("area (give city+state, or boundary [+name])")
    g.add_argument("--city", help='town name as in the Census county subdivisions, e.g. "Fall River"')
    g.add_argument("--state", default="MA", help="two-letter state code (default MA)")
    g.add_argument("--boundary", help="any polygon file geopandas can read (GeoJSON, shapefile, GPKG …)")
    g.add_argument("--name", help="region name for a --boundary run (default: the file's stem)")
    ap.add_argument("--out", default=str(DEFAULT_OUT),
                    help=f"output root (default {DEFAULT_OUT}); the analysis reads <out>/<REGION>/")


class Area:
    def __init__(self, a: argparse.Namespace):
        if a.boundary:
            self.boundary = Path(a.boundary).expanduser().resolve()
            self.city = self.state = None
            self.region = region_slug(a.name or self.boundary.stem)
            self.label = f"boundary {self.boundary}"
        elif a.city:
            self.boundary = None
            self.city, self.state = a.city, a.state.upper()
            self.region = region_slug(a.city)
            self.label = f"{self.city}, {self.state}"
        else:
            raise SystemExit("give --city <Town> [--state MA]  or  --boundary <polygon file> [--name <name>]")
        self.out = Path(a.out).expanduser().resolve()
        self.region_dir = self.out / self.region
        self.index_shp = self.out / INDEX_SHP
        self.mosaic = self.region_dir / f"Merged_{self.region}.tif"
        self.vrt = self.mosaic.with_suffix(".vrt")
        self.tiles_dir = self.region_dir / "tiles"
        self.mappings_csv = self.region_dir / "Tile_Mappings.csv"
        self.info_json = self.region_dir / "imagery_info.json"

    def __repr__(self):
        return f"{self.region} ({self.label}) → {self.region_dir}"
