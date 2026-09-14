"""predict — seg weights + tiles → predictions/<REGION>/<stem>_pred.png (0/1/2/3)

Ported from 05_apply_boston. Output filename keeps the tile stem so the
px/py coordinates survive into the centerline stage. Already-predicted
tiles are skipped, so the run is resumable.
"""
from __future__ import annotations

import time

import numpy as np
from PIL import Image

from ..config import Config
from ..model import device as dev
from ..model.common import load_checkpoint, make_model, normalize

Image.MAX_IMAGE_PIXELS = None


def predict_dir(model, tiles, out_dir, device, batch=16):
    import torch
    from tqdm import tqdm
    todo = [t for t in tiles if not (out_dir / f"{t.stem}_pred.png").exists()]
    print(f"  {len(tiles)} tiles, {len(todo)} to predict")
    buf, paths = [], []

    @torch.no_grad()
    def flush():
        if not buf:
            return
        x = torch.from_numpy(np.stack(buf)).to(device)
        with dev.autocast(device):
            logits = model(x)
        for pred, p in zip(logits.argmax(1).cpu().numpy().astype(np.uint8), paths):
            Image.fromarray(pred).save(out_dir / f"{p.stem}_pred.png")
        buf.clear()
        paths.clear()

    for t in tqdm(todo, desc="  predict"):
        buf.append(normalize(np.array(Image.open(t).convert("RGB"))))
        paths.append(t)
        if len(buf) >= batch:
            flush()
    flush()


def summarize(out_dir, min_px=50):
    preds = sorted(out_dir.glob("*_pred.png"))
    lane = curb = empty = 0
    for pp in preds:
        m = np.array(Image.open(pp))
        hl, hc = (m == 1).sum() > min_px, (m == 2).sum() > min_px
        lane += hl
        curb += hc
        empty += not hl and not hc
    n = max(len(preds), 1)
    print(f"  {len(preds)} predictions: lane in {lane} ({100*lane/n:.0f}%), "
          f"curb in {curb} ({100*curb/n:.0f}%), empty {empty} ({100*empty/n:.0f}%)")


def run(cfg: Config, argv=None, check: bool = False):
    import argparse
    ap = argparse.ArgumentParser(prog="bikelane predict")
    ap.add_argument("--device", default=None, help="cuda / cuda:1 / mps / cpu (auto if omitted)")
    ap.add_argument("--summary", action="store_true", help="only summarize existing predictions")
    a = ap.parse_args(argv)
    weights = cfg.weights("seg")
    tiles_dir = cfg.path("tiles_dir")
    out_dir = cfg.path("predictions_dir", mkdir=not check)
    print(f"predict  {cfg.region}\n  weights: {weights}\n  tiles: {tiles_dir}\n  out: {out_dir}")
    if check:
        for f in (weights, tiles_dir):
            print(f"  {'ok ' if f.exists() else 'MISSING'} {f}")
        return
    if a.summary:
        return summarize(out_dir)
    device = dev.pick(a.device)
    tr = cfg.get("train")
    model = make_model(tr["encoder"], tr["num_classes"]).to(device)
    meta = load_checkpoint(model, weights, device)
    model.eval()
    print(f"  model loaded on {dev.describe(device)} {meta}")
    if device == "cpu":
        print("  (CPU inference: expect several hours for a whole town; MPS or CUDA is much faster)")
    t0 = time.time()
    predict_dir(model, sorted(tiles_dir.glob("*.jpg")), out_dir, device, cfg.get("predict.batch"))
    print(f"  done in {time.time()-t0:.0f}s")
    summarize(out_dir)
    print("→ next: bikelane centerlines extract")
