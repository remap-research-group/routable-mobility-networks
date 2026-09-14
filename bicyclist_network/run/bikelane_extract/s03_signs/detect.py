"""signs detect — YOLO inference on every tile → <region>_bikesigns.csv (raw).

Ported from 09_bikesign_detect §2. Inference runs once at a LOW confidence
(conf_infer, 0.25) and the raw detections are kept; the acceptance threshold
is applied afterwards by `signs filter`, so re-deciding the threshold never
means re-running hours of inference. Existing raw CSV → skipped unless --fresh.
"""
from __future__ import annotations

import time

from ..config import Config
from ..io import TileGrid, write_signs_csv


def detect(cfg: Config, weights, tg: TileGrid, conf, iou, imgsz, batch, device=None):
    from ultralytics import YOLO
    from tqdm import tqdm
    from ..model import device as dev
    device = dev.pick(device)
    model = YOLO(str(weights))
    print(f"  device: {dev.describe(device)}")
    print(f"  classes: {model.names}")
    tiles = tg.matched_tiles()
    det = []
    for i in tqdm(range(0, len(tiles), batch), desc="  YOLO"):
        chunk = tiles[i:i + batch]
        res = model.predict([str(t) for t, _ in chunk], imgsz=imgsz, conf=conf, iou=iou,
                            device=device, verbose=False)
        for (t, (gpx0, gpy0)), rr in zip(chunk, res):
            if rr.boxes is None or len(rr.boxes) == 0:
                continue
            for b in rr.boxes:
                cls = int(b.cls[0])
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                gcx, gcy = gpx0 + (x1 + x2) / 2, gpy0 + (y1 + y2) / 2
                ux, uy = tg.to_utm(gcx, gcy)
                det.append(dict(tile=t.name, cls=cls, cls_name=model.names[cls],
                                conf=round(float(b.conf[0]), 3), gpx=round(gcx, 1),
                                gpy=round(gcy, 1), utm_x=round(float(ux), 2),
                                utm_y=round(float(uy), 2),
                                bbox_local=[round(v, 1) for v in (x1, y1, x2, y2)]))
    return det


def run(cfg: Config, check=False, fresh=False, device=None):
    p = cfg.get("signs")
    weights = cfg.weights("yolo")
    raw = cfg.path("signs_raw_csv", mkdir=not check)
    print(f"signs detect  {cfg.region}\n  weights: {weights}\n  raw csv: {raw}")
    tiles_dir, csv = cfg.path("tiles_dir"), cfg.path("tile_mappings_csv")
    if check:
        for f in (weights, tiles_dir, csv):
            print(f"  {'ok ' if f.exists() else 'MISSING'} {f}")
        return
    if raw.exists() and not fresh:
        print("  raw CSV exists — skipping inference (use --fresh to redo)")
        return
    tg = TileGrid(csv, cfg.resolution_m, cfg.tile_px, tiles_dir)
    print(f"  {tg}")
    t0 = time.time()
    det = detect(cfg, weights, tg, p["conf_infer"], p["iou"], p["imgsz"], p["batch"], device)
    write_signs_csv(raw, det)
    print(f"  {len(det)} detections at conf ≥ {p['conf_infer']}  ({time.time()-t0:.0f}s)")
    print("→ next: bikelane signs filter  (run `filter --sweep` first for a new region)")
