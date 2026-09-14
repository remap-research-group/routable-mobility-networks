"""File I/O shared across stages.

- GeoJSON LineString/Point read/write in the project CRS, with automatic
  timestamped backup instead of overwriting (every notebook did this by hand).
- TileGrid: the mapping between tile filenames (tile_px<X>_py<Y>.jpg), global
  mosaic pixels and UTM, derived from Tile_Mappings.csv and cross-checked
  across several tiles so a silent mismatch cannot produce wrong coordinates.
"""
from __future__ import annotations

import csv
import json
import re
import time
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .geom import dedup_pts, plen


# ───────────────────────── GeoJSON ─────────────────────────

def crs_block(crs: str) -> dict:
    epsg = crs.split(":")[-1]
    return {"type": "name", "properties": {"name": f"urn:ogc:def:crs:EPSG::{epsg}"}}


def backup_if_exists(path: Path) -> Path | None:
    """Rename an existing file to <stem>.<unixtime><suffix>; return the backup path."""
    path = Path(path)
    if not path.exists():
        return None
    bak = path.with_name(f"{path.stem}.{int(time.time())}{path.suffix}")
    path.rename(bak)
    print(f"  backed up existing file: {bak.name}")
    return bak


def write_lines(path: Path, lines: Sequence[np.ndarray], props: Sequence[dict] | None,
                crs: str, add_length: bool = True) -> Path:
    """Write polylines as a LineString FeatureCollection. Backs up an existing file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    backup_if_exists(path)
    feats = []
    for i, ln in enumerate(lines):
        p = dict(props[i]) if props else {}
        p.setdefault("id", i)
        if add_length:
            p["length_m"] = round(plen(ln), 1)
        for k, v in list(p.items()):
            if isinstance(v, dict):          # e.g. classes counter
                p[k] = json.dumps(v)
            elif isinstance(v, (np.floating, np.integer)):
                p[k] = v.item()
        feats.append({"type": "Feature", "properties": p,
                      "geometry": {"type": "LineString",
                                   "coordinates": [[float(x), float(y)] for x, y in ln]}})
    with open(path, "w") as f:
        json.dump({"type": "FeatureCollection", "crs": crs_block(crs), "features": feats}, f)
    print(f"  saved {path}  ({len(feats)} lines)")
    return path


def write_points(path: Path, xy: Sequence, props: Sequence[dict], crs: str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    backup_if_exists(path)
    feats = []
    for (x, y), p in zip(xy, props):
        p = {k: (v.item() if isinstance(v, (np.floating, np.integer)) else v) for k, v in p.items()}
        feats.append({"type": "Feature", "properties": p,
                      "geometry": {"type": "Point", "coordinates": [float(x), float(y)]}})
    with open(path, "w") as f:
        json.dump({"type": "FeatureCollection", "crs": crs_block(crs), "features": feats}, f)
    print(f"  saved {path}  ({len(feats)} points)")
    return path


def read_lines(path: Path, parse_json_props: Iterable[str] = ("classes",)):
    """Read a LineString FeatureCollection → (lines, props).

    Consecutive duplicate vertices are removed and degenerate features
    dropped. Properties listed in `parse_json_props` are decoded from JSON
    strings back to dicts.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"missing input: {path}")
    lines, props = [], []
    for ft in json.load(open(path))["features"]:
        g = dedup_pts(ft["geometry"]["coordinates"])
        if len(g) < 2:
            continue
        p = dict(ft.get("properties") or {})
        for k in parse_json_props:
            if isinstance(p.get(k), str):
                try:
                    p[k] = json.loads(p[k])
                except json.JSONDecodeError:
                    p[k] = {}
        lines.append(g)
        props.append(p)
    return lines, props


def read_points(path: Path):
    """Read a Point FeatureCollection → (xy (N,2), props)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"missing input: {path}")
    xy, props = [], []
    for ft in json.load(open(path))["features"]:
        xy.append(ft["geometry"]["coordinates"][:2])
        props.append(dict(ft.get("properties") or {}))
    return np.asarray(xy, float).reshape(-1, 2), props


# ───────────────────────── tiles ↔ UTM ─────────────────────────

_PXPY = re.compile(r"px(\d+)_py(\d+)")


def stem_pxpy(stem: str):
    m = _PXPY.search(stem)
    return (int(m.group(1)), int(m.group(2))) if m else None


class TileGrid:
    """Tile filename ↔ global mosaic pixel ↔ UTM.

    X_utm = origin_x + gpx * res
    Y_utm = origin_y - gpy * res      (image rows grow downward)

    The origin comes from Tile_Mappings.csv (Pixel_X/Y ↔ CRS_X/Y) and is
    cross-checked over up to 20 rows so an inconsistent CSV fails loudly.
    `tiles_dir` is optional: stages that only read predictions pass None.
    """

    def __init__(self, mappings_csv: Path, res_m: float, tile_px: int,
                 tiles_dir: Path | None = None):
        self.res = float(res_m)
        self.tile_px = int(tile_px)
        self.geo: dict[str, tuple[int, int, float, float]] = {}
        with open(mappings_csv) as f:
            for r in csv.DictReader(f):
                self.geo[r["image_name"]] = (int(r["Pixel_X"]), int(r["Pixel_Y"]),
                                             float(r["CRS_X"]), float(r["CRS_Y"]))
        if not self.geo:
            raise ValueError(f"empty Tile_Mappings: {mappings_csv}")
        self.origin_x, self.origin_y = self._derive_origin()
        self.tiles_dir = Path(tiles_dir) if tiles_dir else None
        self.tiles = sorted(self.tiles_dir.glob("*.jpg")) if self.tiles_dir else []
        if self.tiles:
            self._check_filenames()

    def _derive_origin(self):
        ox = oy = None
        for n, (name, (px, py, cx, cy)) in enumerate(self.geo.items()):
            pp = stem_pxpy(name)
            if pp is not None and pp != (px, py):
                raise ValueError(f"{name}: filename px/py differs from CSV Pixel_X/Y {px},{py}")
            x0, y0 = cx - px * self.res, cy + py * self.res
            if ox is None:
                ox, oy = x0, y0
            elif abs(x0 - ox) > 1 or abs(y0 - oy) > 1:
                raise ValueError(f"{name}: inconsistent mosaic origin ({x0:.1f},{y0:.1f}) "
                                 f"vs ({ox:.1f},{oy:.1f})")
            if n >= 20:
                break
        return ox, oy

    def _check_filenames(self):
        matched = sum(1 for t in self.tiles if t.name in self.geo)
        if matched == 0:
            raise ValueError(f"no tile in {self.tiles_dir} matches Tile_Mappings.csv")

    def to_utm(self, gpx, gpy):
        return self.origin_x + np.asarray(gpx) * self.res, self.origin_y - np.asarray(gpy) * self.res

    def to_pixel(self, x, y):
        return (np.asarray(x) - self.origin_x) / self.res, (self.origin_y - np.asarray(y)) / self.res

    def matched_tiles(self):
        """Tiles on disk that are also in the CSV, with their (gpx, gpy)."""
        return [(t, stem_pxpy(t.stem)) for t in self.tiles
                if t.name in self.geo and stem_pxpy(t.stem) is not None]

    def __repr__(self):
        return (f"TileGrid({len(self.geo)} mapped, {len(self.tiles)} on disk, "
                f"origin=({self.origin_x:.1f},{self.origin_y:.1f}), res={self.res} m)")


# ───────────────────────── sign CSV ─────────────────────────

SIGN_FIELDS = ["tile", "cls", "cls_name", "conf", "gpx", "gpy", "utm_x", "utm_y", "bbox_local"]


def write_signs_csv(path: Path, rows: list[dict], extra=()) -> None:
    fields = list(SIGN_FIELDS) + list(extra)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for d in rows:
            row = {k: d.get(k, "") for k in fields}
            row["bbox_local"] = json.dumps(list(d["bbox_local"]))
            w.writerow(row)


def read_signs_csv(path: Path) -> list[dict]:
    import ast
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        r["conf"] = float(r["conf"])
        r["cls"] = int(r["cls"])
        r["gpx"], r["gpy"] = float(r["gpx"]), float(r["gpy"])
        r["utm_x"], r["utm_y"] = float(r["utm_x"]), float(r["utm_y"])
        r["bbox_local"] = ast.literal_eval(r["bbox_local"])
        for k in ("n_nb", "isolated"):
            if r.get(k) not in (None, ""):
                r[k] = int(r[k])
    return rows
