"""Shared raster / vector I/O helpers for run/stage_1 and run/stage_2.

numpy + rasterio only (no torch), so stage 2 can run on a CPU-only machine.
"""
import os
import re
import sys
import json
from pathlib import Path

import numpy as np

# DC tile grid of the training data (train/README.md). Used ONLY as a fallback
# for files named tile_<TX>_<TY>.tif whose embedded CRS is unusable — the DC
# tiles carry a bare LOCAL_CS["NAD83 / Maryland"] tag without an EPSG code.
DC_GRID = {'epsg': 26985, 'res': 0.08, 'tile_px': 1024,
           'xmin': 395852.57, 'ymax': 140914.34}

_PROJ_CHECKED = False


def ensure_proj():
    """Make EPSG lookups work even when the shell exports PROJ_DATA/PROJ_LIB
    from another, incompatible PROJ installation (this breaks CRS tagging and
    any reprojection): probe once and, on failure, repoint PROJ at this
    environment's own database."""
    global _PROJ_CHECKED
    if _PROJ_CHECKED:
        return
    import rasterio
    from rasterio.crs import CRS
    try:
        CRS.from_epsg(4326)
    except Exception:
        from rasterio._env import set_proj_data_search_path
        last = None
        for cand in (Path(rasterio.__file__).parent / 'proj_data',
                     Path(sys.prefix) / 'share' / 'proj'):
            if not (cand / 'proj.db').exists():
                continue
            os.environ['PROJ_DATA'] = os.environ['PROJ_LIB'] = str(cand)
            set_proj_data_search_path(str(cand))
            try:
                CRS.from_epsg(4326)
                break
            except Exception as e:
                last = e
        else:
            raise last or RuntimeError('no usable proj.db found')
    _PROJ_CHECKED = True


# =============================================================================
# CRS helpers
# =============================================================================
def crs_is_usable(crs):
    """True if the CRS can be reprojected (has an EPSG code, or is a real
    projected / geographic CRS). A bare LOCAL_CS is not usable."""
    if crs is None:
        return False
    try:
        if crs.to_wkt().lstrip().upper().startswith('LOCAL_CS'):
            return False                    # named but datum-less (some GDAL builds call it projected)
    except Exception:
        pass
    try:
        if crs.to_epsg() is not None:
            return True
    except Exception:
        pass
    try:
        return bool(crs.is_projected or crs.is_geographic)
    except Exception:
        return False


def crs_name(crs):
    """'EPSG:xxxx' when possible, else the WKT string."""
    try:
        e = crs.to_epsg()
        if e is not None:
            return f'EPSG:{e}'
    except Exception:
        pass
    return crs.to_wkt()


def dc_grid_transform(stem, transform=None, tol_m=0.5):
    """Affine transform of a DC training tile from its stem, or None if the stem
    is not tile_<TX>_<TY> or (when a file transform is given) the file's origin
    does not sit on the DC grid."""
    m = re.match(r'^tile_(\d+)_(\d+)$', stem)
    if not m:
        return None
    from rasterio.transform import from_origin
    tx, ty = int(m.group(1)), int(m.group(2))
    g = DC_GRID
    tile_m = g['tile_px'] * g['res']
    xmin = g['xmin'] + tx * tile_m
    ymax = g['ymax'] - ty * tile_m
    if transform is not None and (abs(transform.c - xmin) > tol_m
                                  or abs(transform.f - ymax) > tol_m):
        return None
    return from_origin(xmin, ymax, g['res'], g['res'])


# =============================================================================
# Raster I/O
# =============================================================================
def resolve_georef(path, ds_transform, ds_crs, shape, crs_override=None, quiet=False):
    """Decide the (transform, crs) a raster's outputs will carry.

    Priority: --crs override > usable embedded CRS > DC grid fallback for
    tile_<TX>_<TY> files > georeferenced-without-CRS > not georeferenced."""
    from rasterio.crs import CRS
    path = Path(path)
    transform = ds_transform
    if transform is not None and transform.is_identity:
        transform = None
    if crs_override:
        ensure_proj()
        crs = CRS.from_user_input(crs_override)
        if transform is None:
            transform = dc_grid_transform(path.stem)
        return transform, crs
    if transform is not None and crs_is_usable(ds_crs):
        return transform, ds_crs
    # fallback: DC training-tile naming + grid origin
    t = dc_grid_transform(path.stem, transform)
    if t is not None and tuple(shape[:2]) == (DC_GRID['tile_px'], DC_GRID['tile_px']):
        ensure_proj()
        if not quiet:
            print(f'  {path.name}: no usable CRS in the file — DC tile grid '
                  f'recognised, tagging EPSG:{DC_GRID["epsg"]}')
        return (transform or t), CRS.from_epsg(DC_GRID['epsg'])
    if transform is not None:
        if not quiet:
            print(f'  {path.name}: georeferenced but its CRS is unusable '
                  f'({ds_crs}) — outputs keep the transform but no CRS; '
                  f'pass --crs EPSG:xxxx to fix')
        return transform, None
    return None, None


def read_image(path, crs_override=None):
    """-> (rgb uint8 HxWx3, transform or None, crs or None)."""
    path = Path(path)
    if path.suffix.lower() not in ('.png', '.jpg', '.jpeg'):   # any GDAL raster
        import rasterio
        ensure_proj()
        with rasterio.open(path) as ds:
            data = ds.read()[:3]                          # first 3 bands
            ds_transform, ds_crs = ds.transform, ds.crs
        if data.shape[0] == 1:
            data = np.repeat(data, 3, axis=0)
        rgb = np.ascontiguousarray(np.transpose(data, (1, 2, 0)))
        if rgb.dtype != np.uint8:                         # e.g. uint16 imagery
            rgb = (rgb.astype(np.float32) / max(float(rgb.max()), 1.0) * 255).astype(np.uint8)
        transform, crs = resolve_georef(path, ds_transform, ds_crs, rgb.shape, crs_override)
        return rgb, transform, crs
    from PIL import Image
    rgb = np.array(Image.open(path).convert('RGB'))
    return rgb, None, None


def raster_meta(path, crs_override=None, quiet=True):
    """(transform, crs, (H, W)) without reading the pixels."""
    import rasterio
    ensure_proj()
    with rasterio.open(path) as ds:
        shape = (ds.height, ds.width)
        ds_transform, ds_crs = ds.transform, ds.crs
    transform, crs = resolve_georef(path, ds_transform, ds_crs, shape, crs_override, quiet=quiet)
    return transform, crs, shape


def write_raster(path, array, transform, crs, dtype, scale=None, tags=None):
    """Single-band deflate-compressed GeoTIFF (predictor picked per dtype).
    `scale` is stored as the band's scale factor (GeoTIFF/GDAL convention:
    physical value = stored value * scale) — used for uint8 confidence."""
    import rasterio
    from rasterio.transform import Affine
    ensure_proj()
    profile = {'driver': 'GTiff', 'height': array.shape[0], 'width': array.shape[1],
               'count': 1, 'dtype': dtype, 'compress': 'deflate',
               'predictor': 3 if dtype.startswith('float') else 2,
               'tiled': True, 'blockxsize': 512, 'blockysize': 512,
               'transform': transform if transform is not None else Affine.identity(),
               'crs': crs}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, 'w', **profile) as dst:
        dst.write(array.astype(dtype)[None])
        if scale is not None:
            dst.scales = [float(scale)]
        if tags:
            dst.update_tags(**tags)


def write_confidence(path, p, transform, crs, fmt='uint8'):
    """Calibrated probability map p in [0,1] -> GeoTIFF. 'uint8' stores
    round(p*255) with band scale 1/255 (~0.4 % steps, ~10x smaller files);
    'float32' stores p as is. read_confidence() undoes either."""
    if fmt == 'float32':
        write_raster(path, p.astype(np.float32), transform, crs, 'float32',
                     tags={'CONFIDENCE': 'calibrated probability, p = sigmoid(logit / T)'})
    else:
        q = np.clip(np.rint(p * 255.0), 0, 255).astype(np.uint8)
        write_raster(path, q, transform, crs, 'uint8', scale=1.0 / 255.0,
                     tags={'CONFIDENCE': 'calibrated probability = value * scale '
                                         '(scale = 1/255), p = sigmoid(logit / T)'})


def read_confidence(path):
    """GeoTIFF written by write_confidence() -> float32 probability in [0,1]."""
    import rasterio
    with rasterio.open(path) as ds:
        a = ds.read(1)
        scale = ds.scales[0] if ds.scales else 1.0
    if a.dtype != np.float32 or (scale is not None and scale != 1.0):
        a = a.astype(np.float32) * float(scale if scale else 1.0)
    return a


# =============================================================================
# Vector I/O
# =============================================================================
def px_to_xy(rc_pairs, transform):
    """Pixel-centre (row, col) pairs -> [x, y] in the transform's CRS."""
    return [[round(transform.c + (c + 0.5) * transform.a, 3),
             round(transform.f + (r + 0.5) * transform.e, 3)] for r, c in rc_pairs]


def reproject_features(feats, src_crs, dst_crs='EPSG:4326', ndigits=7):
    """Copy of a GeoJSON feature list with every vertex reprojected (default:
    -> WGS84 lon/lat). One bulk transform; geometry nesting is kept.
    7 decimal degrees ~ 1 cm."""
    ensure_proj()
    from rasterio.warp import transform as warp_transform

    xs, ys = [], []

    def collect(coords):
        if coords and isinstance(coords[0], (int, float)):
            xs.append(coords[0]); ys.append(coords[1])
        else:
            for c in coords:
                collect(c)

    for f in feats:
        collect(f['geometry']['coordinates'])
    if not xs:
        return [json.loads(json.dumps(f)) for f in feats]

    tx, ty = warp_transform(src_crs, dst_crs, xs, ys)
    pts = iter(zip(tx, ty))

    def rebuild(coords):
        if coords and isinstance(coords[0], (int, float)):
            lon, lat = next(pts)
            return [round(lon, ndigits), round(lat, ndigits)]
        return [rebuild(c) for c in coords]

    return [{**f, 'geometry': {**f['geometry'],
                               'coordinates': rebuild(f['geometry']['coordinates'])}}
            for f in feats]


def write_geojson_pair(out_dir, name, feats, crs):
    """<name>.geojson in the native CRS (legacy 'crs' member) and, when the CRS
    is usable and not already geographic, <name>_wgs84.geojson (RFC 7946
    lon/lat). Returns the list of files written."""
    out_dir = Path(out_dir)
    written = []
    fc = {'type': 'FeatureCollection', 'features': feats}
    if crs is not None:
        fc['crs'] = {'type': 'name', 'properties': {'name': crs_name(crs)}}
    (out_dir / f'{name}.geojson').write_text(json.dumps(fc))
    written.append(f'{name}.geojson')
    if crs_is_usable(crs) and not crs.is_geographic:
        (out_dir / f'{name}_wgs84.geojson').write_text(json.dumps(
            {'type': 'FeatureCollection', 'features': reproject_features(feats, crs)}))
        written.append(f'{name}_wgs84.geojson')
    return written


# =============================================================================
# Input inspection (printed before anything is run)
# =============================================================================
MODEL_RES_M = 0.08           # the model's native ground sampling distance
RES_TOL     = 0.05           # accept resolutions within +-5 % of native
BYTES_PER_PX_STAGE1 = 50     # rough RAM need of stage 1 per pixel (logits + rasters)


def guess_epsg_from_name(name):
    """Best-effort EPSG code from a CRS *name* (for LOCAL_CS tags that name a
    system but carry no datum). Only a hint — the user must confirm with --crs."""
    n = (name or '').lower().replace(' ', '_').replace('-', '_')
    m = re.search(r'utm_zone_(\d{1,2})_?([ns])', n)
    if m:
        z, h = int(m.group(1)), m.group(2)
        if 'nad_1983_2011' in n or 'nad83(2011)' in n or 'nad83_2011' in n:
            return 6330 + z if h == 'n' else None
        if 'nad83' in n or 'nad_1983' in n:
            return 26900 + z if h == 'n' else None
        if 'wgs_1984' in n or 'wgs84' in n:
            return (32600 if h == 'n' else 32700) + z
    if 'maryland' in n and ('nad83' in n or 'nad_1983' in n):
        return 26985
    return None


def describe_image(path, crs_override=None, tile_px=2048, quiet=False):
    """Inspect one input image and print a report: format, size, bands,
    georeferencing, resolution vs. the model's native 0.08 m/px, coverage,
    and what has to happen before stage 1 can run (resample / tile / CRS).
    Returns a dict; info['ready'] is True when stage 1 can take the file as is."""
    import math
    path = Path(path)
    info = {'path': str(path), 'name': path.name, 'ready': False, 'actions': [],
            'warnings': [], 'size_mb': round(path.stat().st_size / 2**20, 1)}
    lines = []

    if path.suffix.lower() in ('.png', '.jpg', '.jpeg'):
        from PIL import Image
        with Image.open(path) as im:
            w, h, bands = im.width, im.height, len(im.getbands())
        info.update(driver=path.suffix[1:].upper(), width=w, height=h, bands=bands,
                    dtype='uint8', georef='none', res=None)
        lines += [f'  format      {info["driver"]}, {info["size_mb"]} MB',
                  f'  size        {w} x {h} px | {bands} band(s)',
                  '  georef      none (plain image) -> outputs in pixel coordinates, no lat/lon',
                  f'  resolution  unknown — assumed {MODEL_RES_M} m/px (resample first if it is not)']
        info['warnings'].append('not georeferenced')
        info['ready'] = True
    else:
        import rasterio
        ensure_proj()
        try:
            ds = rasterio.open(path)
        except rasterio.errors.RasterioIOError as e:
            msg = str(e).split('\n')[0]
            hint = (' — this Python\'s GDAL has no JPEG2000 driver; use a rasterio '
                    'build with JP2OpenJPEG (PyPI wheels and conda-forge have it)'
                    if path.suffix.lower() == '.jp2' else '')
            info['error'] = msg
            info['actions'].append('cannot open the file' + hint)
            info['summary'] = f'{path.name}: CANNOT READ ({msg[:60]}...)'
            lines.append(f'  ERROR       cannot open: {msg}{hint}')
            _print_report(path, lines, info, quiet)
            return info
        with ds:
            w, h, bands, dtype = ds.width, ds.height, ds.count, ds.dtypes[0]
            ds_t, ds_crs = ds.transform, ds.crs
            driver, comp = ds.driver, ds.compression
        info.update(driver=driver, width=w, height=h, bands=bands, dtype=dtype)
        band_note = ('' if bands == 3 else ' (first 3 used as RGB)' if bands > 3
                     else ' (grey -> replicated to RGB)')
        lines += [f'  format      {driver}{", " + comp.name if comp else ""}, {info["size_mb"]} MB',
                  f'  size        {w} x {h} px | {bands} band(s) {dtype}{band_note}']
        if dtype != 'uint8':
            info['warnings'].append(f'{dtype} pixels are rescaled to uint8 by max value')

        transform, crs = resolve_georef(path, ds_t, ds_crs, (h, w), crs_override, quiet=True)
        if transform is None:
            info.update(georef='none', res=None)
            lines.append('  georef      none -> outputs in pixel coordinates, no lat/lon')
            info['warnings'].append('not georeferenced')
            res = None
        else:
            res_x, res_y = abs(transform.a), abs(transform.e)
            geographic = crs is not None and crs_is_usable(crs) and crs.is_geographic
            if geographic:                       # degrees -> metres at the image centre
                lat = transform.f + transform.e * h / 2
                res = res_x * 111_320 * math.cos(math.radians(lat))
                unit = f'{res_x:.7f} deg (~{res:.3f} m)'
            else:
                res = res_x
                unit = f'{res:.3f} m'
            info['res'] = res
            if crs is not None and crs_is_usable(crs):
                info['georef'] = 'ok'
                tag = f'CRS {crs_name(crs)}'
                if ds_crs is None or not crs_is_usable(ds_crs):
                    tag += ' (assigned' + (' from --crs' if crs_override else ' from the DC tile grid') + ')'
                lines.append(f'  georef      yes | {tag}')
            else:
                info['georef'] = 'unusable'
                name = None
                try:
                    name = re.search(r'^[A-Z_]+\["([^"]+)"', ds_crs.to_wkt()).group(1)
                except Exception:
                    pass
                guess = guess_epsg_from_name(name)
                info['epsg_guess'] = guess
                lines.append(f'  georef      yes, but the CRS tag "{name}" carries no EPSG code / datum '
                             f'-> lengths OK, no lat/lon export'
                             + (f'; pass --crs EPSG:{guess} (guessed from the name — verify)'
                                if guess else '; pass --crs EPSG:xxxx'))
                info['warnings'].append('CRS unusable (no lat/lon export)')
            ext_w, ext_h = w * res_x, h * res_y
            if geographic:
                ext_w, ext_h = w * res, h * res
            x0, y1 = transform.c, transform.f
            x1, y0 = x0 + w * transform.a, y1 + h * transform.e
            lines.append(f'  extent      {ext_w:,.0f} x {ext_h:,.0f} m ({ext_w * ext_h / 1e4:,.1f} ha) | '
                         f'x {x0:,.2f}..{x1:,.2f}, y {y0:,.2f}..{y1:,.2f}')
            if crs is not None and crs_is_usable(crs):
                try:
                    from rasterio.warp import transform_bounds
                    lon0, lat0, lon1, lat1 = transform_bounds(crs, 'EPSG:4326',
                                                              min(x0, x1), min(y0, y1),
                                                              max(x0, x1), max(y0, y1))
                    info['wgs84_bounds'] = [lon0, lat0, lon1, lat1]
                    lines.append(f'  lat/lon     {lat0:.5f}..{lat1:.5f} N, {lon0:.5f}..{lon1:.5f} E')
                except Exception as e:
                    lines.append(f'  lat/lon     (reprojection failed: {str(e)[:60]})')
            # resolution vs model
            f = res / MODEL_RES_M
            if abs(f - 1.0) <= RES_TOL:
                tw, th = w, h
                lines.append(f'  resolution  {unit}/px | matches the model\'s {MODEL_RES_M} m/px')
            else:
                tw, th = round(w * f), round(h * f)
                info['actions'].append(f'resample from {res:.3f} to {MODEL_RES_M} m/px')
                lines.append(f'  resolution  {unit}/px | model native {MODEL_RES_M} m/px -> RESAMPLE '
                             f'x{f:.3f} to {tw} x {th} px'
                             + (' (upsampling: no real extra detail, expect softer predictions)'
                                if f > 1 else ''))
            info['target_size'] = [th, tw]
            # tiling
            mpx = tw * th / 1e6
            ram_gb = tw * th * BYTES_PER_PX_STAGE1 / 2**30
            if max(tw, th) > 2 * tile_px:
                nc, nr = math.ceil(tw / tile_px), math.ceil(th / tile_px)
                info['tiles'] = [nr, nc]
                info['actions'].append(f'split into {nr} x {nc} tiles of ~{tile_px} px')
                lines.append(f'  tiling      {tw} x {th} px = {mpx:,.0f} Mpx as one image (~{ram_gb:.0f} GB RAM '
                             f'in stage 1) -> recommend --tile-px {tile_px} ({nr} x {nc} = {nr * nc} tiles)')
            else:
                lines.append(f'  tiling      not needed ({mpx:,.1f} Mpx, ~{max(ram_gb, 0.1):.1f} GB RAM)')
        blocking = [a for a in info['actions'] if a.startswith('resample')]
        info['ready'] = not blocking
    info['summary'] = (f'{path.name}: {info.get("width")} x {info.get("height")} px, '
                       f'{info.get("bands")} band(s)'
                       + (f', {info["res"]:.3f} m/px' if info.get('res') else '')
                       + f', georef {info.get("georef")}'
                       + (', ' + '; '.join(info['actions']) if info['actions'] else '')
                       + ' -> ' + ('READY' if info['ready'] else 'NEEDS PREPARE'))
    _print_report(path, lines, info, quiet)
    return info


def _print_report(path, lines, info, quiet):
    if quiet:
        return
    print(f'── {path.name} ' + '─' * max(60 - len(path.name), 4))
    for l in lines:
        print(l)
    if info['actions']:
        acts = [a for a in info['actions'] if not a.startswith('cannot')]
        if any(a.startswith('cannot') for a in info['actions']):
            print('  status      CANNOT READ')
        else:
            crs_hint = f' --crs EPSG:{info["epsg_guess"]}' if info.get('epsg_guess') else ''
            tile_hint = (f' --tile-px {int(acts[-1].split("~")[1].split()[0])}'
                         if any(a.startswith('split') for a in acts) else '')
            print(f'  status      {"NEEDS PREPARE" if not info["ready"] else "READY (tiling recommended)"}: '
                  f'{"; ".join(acts)} -> python run/stage_1/prepare_image.py {path.name}{crs_hint}{tile_hint}')
    else:
        print('  status      READY' + (f' ({"; ".join(info["warnings"])})' if info['warnings'] else ''))
