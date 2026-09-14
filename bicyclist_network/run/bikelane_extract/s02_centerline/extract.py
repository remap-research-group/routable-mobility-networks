"""centerlines extract — predictions → per-chunk polylines (UTM), cached as .npz.

Ported from 08c_run_centerlines.py. Runs chunk by chunk (CHUNK×CHUNK tiles
merged onto one canvas, so lines crossing a tile edge are extracted once)
and caches each chunk, so an interrupted run resumes where it stopped.

Why merge tiles first: predictions come as overlapping 1024 px tiles. If
the centerline were extracted per tile, every overlap would yield the
same line twice, slightly offset. Building a 0/1/2 canvas per chunk and
extracting once removes that. Curb (2) is kept on the canvas because the
Voronoi step uses it as an owner.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

from ..config import Config
from ..io import TileGrid, stem_pxpy
from .skeleton import skeleton_to_polylines
from .voronoi import VoronoiParams, mask_to_centerline

Image.MAX_IMAGE_PIXELS = None


class PredictionGrid:
    """Prediction PNGs of one region, indexed by (gpx, gpy)."""

    def __init__(self, pred_dir: Path, tile_px: int):
        self.pred_dir = Path(pred_dir)
        self.tile_px = tile_px
        self.grid = {}
        for p in self.pred_dir.glob("*_pred.png"):
            pp = stem_pxpy(p.stem)
            if pp is not None:
                self.grid[pp] = p
        if not self.grid:
            raise FileNotFoundError(f"no *_pred.png in {pred_dir}")
        self.pxs = sorted({px for px, _ in self.grid})
        self.pys = sorted({py for _, py in self.grid})
        self.step_px = self.pxs[1] - self.pxs[0] if len(self.pxs) > 1 else tile_px

    def canvas(self, gpx0, gpy0, gpx1, gpy1, use_curb=True):
        """0/1/2 canvas for a global-pixel window. Lane wins where classes overlap."""
        W, H = gpx1 - gpx0, gpy1 - gpy0
        T = self.tile_px
        canvas = np.zeros((H, W), np.uint8)
        used = 0
        for (px, py), path in self.grid.items():
            if px + T <= gpx0 or px >= gpx1 or py + T <= gpy0 or py >= gpy1:
                continue
            pr = np.array(Image.open(path))
            if pr.ndim == 3:
                pr = pr[:, :, 0]
            x0, y0 = px - gpx0, py - gpy0
            cx0, cy0 = max(0, x0), max(0, y0)
            cx1, cy1 = min(W, x0 + T), min(H, y0 + T)
            tx0, ty0 = cx0 - x0, cy0 - y0
            sub = pr[ty0:ty0 + (cy1 - cy0), tx0:tx0 + (cx1 - cx0)]
            dst = canvas[cy0:cy1, cx0:cx1]
            if use_curb:
                dst[(sub == 2) & (dst == 0)] = 2
            dst[sub == 1] = 1
            used += 1
        return canvas, used


def process_chunk(preds: PredictionGrid, tg: TileGrid, gpx0, gpy0, gpx1, gpy1,
                  P: VoronoiParams, simplify_eps=2.0, return_skel=False):
    canvas, _ = preds.canvas(gpx0, gpy0, gpx1, gpy1, P.use_curb)
    if canvas.sum() == 0:
        return ([], None) if return_skel else []
    skel = mask_to_centerline(canvas, P)
    out = []
    for pl in skeleton_to_polylines(skel, simplify_eps):
        x, y = tg.to_utm(pl[:, 0] + gpx0, pl[:, 1] + gpy0)
        out.append(np.column_stack([x, y]))
    return (out, (canvas, skel)) if return_skel else out


# ── chunk cache ──

def chunk_path(chunk_dir: Path, region: str, gx, gy) -> Path:
    return chunk_dir / f"{region}_{gx}_{gy}.npz"


def save_chunk(path: Path, lines):
    if lines:
        cat = np.concatenate(lines).astype(np.float64)
        off = np.cumsum([0] + [len(l) for l in lines]).astype(np.int64)
    else:
        cat, off = np.zeros((0, 2)), np.zeros(1, np.int64)
    np.savez_compressed(path, pts=cat, off=off)


def load_chunk(path: Path):
    d = np.load(path)
    pts, off = d["pts"], d["off"]
    return [pts[off[i]:off[i + 1]] for i in range(len(off) - 1)]


def load_all_chunks(chunk_dir: Path, region: str):
    lines = []
    files = sorted(chunk_dir.glob(f"{region}_*.npz"))
    for f in files:
        lines += [ln for ln in load_chunk(f) if len(ln) >= 2]
    return lines, len(files)


# ── driver ──

_W = {}   # per-worker state (set by _init_worker)


def _init_worker(pred_dir, tile_px, csv_path, res_m, params: dict, simplify_eps):
    import os
    os.environ.setdefault("OMP_NUM_THREADS", "1")     # numpy/scipy: one thread per process
    _W["preds"] = PredictionGrid(pred_dir, tile_px)
    _W["tg"] = TileGrid(csv_path, res_m, tile_px)
    _W["P"] = VoronoiParams(**params)
    _W["eps"] = simplify_eps


def _do_chunk(job):
    gx, gy, gx1, gy1, out = job
    t0 = time.time()
    try:
        lines = process_chunk(_W["preds"], _W["tg"], gx, gy, gx1, gy1, _W["P"], _W["eps"])
        save_chunk(Path(out), lines)
        return gx, gy, len(lines), time.time() - t0, None
    except MemoryError:
        return gx, gy, 0, time.time() - t0, "MemoryError"
    except Exception as e:
        return gx, gy, 0, time.time() - t0, f"{type(e).__name__}: {e}"


def run(cfg: Config, check=False, fresh=False, limit=0, chunk_tiles=None, workers=None):
    sec = cfg.get("centerlines")
    chunk = int(chunk_tiles or sec["chunk_tiles"])
    overlap = int(sec.get("overlap_tiles", 1))
    P = VoronoiParams.from_dict(sec.get("voronoi"))
    P.use_curb = bool(sec.get("use_curb", True))
    pred_dir = cfg.path("predictions_dir")
    csv = cfg.path("tile_mappings_csv")
    chunk_dir = cfg.path("chunks_dir", mkdir=not check)
    log_path = cfg.path("logs_dir", mkdir=not check) / f"extract_{cfg.region}.log"
    print(f"centerlines extract  {cfg.region}\n  predictions: {pred_dir}\n  chunks: {chunk_dir}")
    if check:
        for f in (pred_dir, csv):
            print(f"  {'ok ' if f.exists() else 'MISSING'} {f}")
        return

    logf = open(log_path, "a")

    def log(*a):
        msg = " ".join(str(x) for x in a)
        print(msg, flush=True)
        logf.write(f"{time.strftime('%H:%M:%S')}  {msg}\n")
        logf.flush()

    preds = PredictionGrid(pred_dir, cfg.tile_px)
    tg = TileGrid(csv, cfg.resolution_m, cfg.tile_px)
    T = cfg.tile_px
    log(f"{len(preds.grid)} prediction tiles | px {preds.pxs[0]}~{preds.pxs[-1]}, "
        f"py {preds.pys[0]}~{preds.pys[-1]}, step {preds.step_px} | origin "
        f"({tg.origin_x:.1f}, {tg.origin_y:.1f})")

    if fresh:
        for f in chunk_dir.glob(f"{cfg.region}_*.npz"):
            f.unlink()
        log("chunk cache cleared")

    step = (chunk - overlap) * preds.step_px
    gxs = list(range(preds.pxs[0], preds.pxs[-1] + T, step))
    gys = list(range(preds.pys[0], preds.pys[-1] + T, step))
    jobs = [(gx, gy) for gx in gxs for gy in gys]
    if limit:
        jobs = jobs[:limit]
    done = sum(1 for gx, gy in jobs if chunk_path(chunk_dir, cfg.region, gx, gy).exists())
    log(f"=== {cfg.region} | chunk={chunk} overlap={overlap} use_curb={P.use_curb} | "
        f"{len(jobs)} chunks ({done} done) ===")

    todo = []
    for gx, gy in jobs:
        out = chunk_path(chunk_dir, cfg.region, gx, gy)
        if out.exists():
            continue
        gx1 = gx + chunk * preds.step_px + (T - preds.step_px)
        gy1 = gy + chunk * preds.step_px + (T - preds.step_px)
        todo.append((gx, gy, gx1, gy1, str(out)))

    import os
    from dataclasses import asdict
    n_workers = int(workers or sec.get("workers") or max(1, min(os.cpu_count() or 1, 8)))
    n_workers = max(1, min(n_workers, len(todo) or 1))
    log(f"{len(todo)} chunks to do with {n_workers} worker(s)  "
        f"(~{3.0 * (chunk / 6) ** 2:.1f} GB RAM each at chunk={chunk})")

    t_start = time.time()
    done_n = 0

    def report(gx, gy, n, dt, err):
        nonlocal done_n
        done_n += 1
        if err:
            log(f"[{done_n}/{len(todo)}] ({gx},{gy}) FAILED {err}")
        else:
            log(f"[{done_n}/{len(todo)}] ({gx},{gy}) {n:5d} lines  {dt:5.0f}s  "
                f"elapsed {(time.time()-t_start)/60:.0f} min")

    init_args = (str(pred_dir), cfg.tile_px, str(csv), cfg.resolution_m, asdict(P), 2.0)
    if n_workers == 1:
        _init_worker(*init_args)
        for job in todo:
            report(*_do_chunk(job))
    else:
        import multiprocessing as mp
        ctx = mp.get_context("fork" if hasattr(os, "fork") else "spawn")
        with ctx.Pool(n_workers, initializer=_init_worker, initargs=init_args) as pool:
            for res in pool.imap_unordered(_do_chunk, todo):
                report(*res)

    missing = [j for j in jobs if not chunk_path(chunk_dir, cfg.region, *j).exists()]
    n_lines = sum(len(load_chunk(chunk_path(chunk_dir, cfg.region, *j)))
                  for j in jobs if j not in missing)
    log(f"done: {n_lines} polylines in {len(jobs)-len(missing)} chunks "
        f"({len(missing)} failed)  {(time.time()-t_start)/60:.0f} min")
    if missing:
        log("  failed chunks: " + ", ".join(f"({gx},{gy})" for gx, gy in missing[:20]))
        log("  re-run to retry them; if MemoryError, use --chunk 4 or fewer --workers")
    log("→ next: bikelane centerlines graph")
