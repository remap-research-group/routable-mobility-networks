"""2_train.py — fine-tune a YOLO detector on the bike pavement-symbol dataset.

OBJECTIVE
  Train the detector that `bikelane signs detect` (run/) applies to every tile:
  three classes, BikeOnly / Sharrow / OnlyBikeBus (facility.py), on 1024 px
  0.15 m/px tiles. This is a plain ultralytics fine-tune of yolo.base_model
  (default yolo26m.pt, fetched by ultralytics on first use) at imgsz 1024 —
  the recipe the released weights came from; everything else is ultralytics'
  default augmentation and schedule.

INPUT
  <yolo_data_dir>/data.yaml     from 1_prepare_dataset.py
  yolo.* in default.yaml        base_model, imgsz, epochs, batch, patience, seed
                                (override in ./bikelane.yaml or with --epochs / --batch)

OUTPUT   (<yolo_runs_dir>/<name>/, default <data_root>/train/yolo/runs/)
  weights/best.pt, weights/last.pt, results.csv, confusion_matrix.png, val_batch*.jpg …
  → python train/yolo/3_export_weights.py   copies best.pt to run/src/yolo26m_bikesign.pt

USAGE  (from bicyclist_network/; pip install -e ".[signs]" and a CUDA build of torch)
  python train/yolo/2_train.py --check
  python train/yolo/2_train.py                        # (long) run inside tmux
  python train/yolo/2_train.py --name bikesign_v2 --epochs 150 --batch 8
  python train/yolo/2_train.py --resume               # continue the newest run from last.pt
  python train/yolo/2_train.py --device 0,1           # ultralytics device string
What to look at: results.csv (mAP50 per class should climb and plateau) and
val_batch*_pred.jpg (boxes on symbols, not on manholes or arrows). The
acceptance threshold used at run time is a separate, region-sensitive choice —
`bikelane signs filter --sweep` — so training aims for recall at conf 0.25.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from bikelane_extract.config import Config


def newest_run(runs_dir: Path) -> Path | None:
    runs = [d for d in runs_dir.glob("*") if (d / "weights").exists()] if runs_dir.exists() else []
    return max(runs, key=lambda d: d.stat().st_mtime) if runs else None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", "-c", default=None, help="project file (default ./bikelane.yaml)")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--name", default=None, help="run name (default bikesign_<timestamp>)")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--model", default=None, help="base checkpoint (default yolo.base_model)")
    ap.add_argument("--device", default=None, help="ultralytics device: 0 | 0,1 | cpu | mps (auto if omitted)")
    ap.add_argument("--resume", action="store_true", help="continue the newest run from last.pt")
    ap.add_argument("--check", action="store_true", help="show the paths and what exists, run nothing")
    a = ap.parse_args(argv)
    cfg = Config.for_training(a.config, a.data_root)
    y = cfg.get("yolo")
    data = cfg.path("yolo_data_dir") / "data.yaml"
    runs = cfg.path("yolo_runs_dir", mkdir=not a.check)
    base = a.model or y["base_model"]
    print(f"yolo train\n  data: {data}\n  base: {base}\n  runs: {runs}")
    if a.check:
        print(f"  {'ok ' if data.exists() else 'MISSING'} {data}")
        last = newest_run(runs)
        print(f"  newest run: {last or '-'}")
        return
    if not data.exists():
        raise SystemExit(f"{data} not found — run train/yolo/1_prepare_dataset.py first")

    from ultralytics import YOLO
    if a.resume:
        run = newest_run(runs)
        if run is None or not (run / "weights" / "last.pt").exists():
            raise SystemExit(f"nothing to resume under {runs}")
        print(f"  resuming {run}")
        YOLO(str(run / "weights" / "last.pt")).train(resume=True)
        return
    name = a.name or f"bikesign_{time.strftime('%Y%m%d-%H%M')}"
    model = YOLO(base)
    kw = dict(data=str(data), imgsz=int(y["imgsz"]), epochs=int(a.epochs or y["epochs"]),
              batch=int(a.batch or y["batch"]), patience=int(y.get("patience", 30)),
              seed=int(y.get("seed", 42)), project=str(runs), name=name, exist_ok=False)
    if a.device is not None:
        kw["device"] = a.device
    print("  " + " ".join(f"{k}={v}" for k, v in kw.items() if k not in ("project",)))
    model.train(**kw)
    best = runs / name / "weights" / "best.pt"
    print(f"  done → {best}\n→ next: python train/yolo/3_export_weights.py")


if __name__ == "__main__":
    main()
