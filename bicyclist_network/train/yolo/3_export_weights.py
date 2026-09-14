"""3_export_weights.py — copy a trained YOLO checkpoint into run/src/.

OBJECTIVE
  Make a detector trained by 2_train.py the one the run stages use: copy
  best.pt to run/src/yolo26m_bikesign.pt (the name weights.yolo in default.yaml
  expects) and refresh its line in run/src/SHA256SUMS. The class names inside
  the checkpoint are checked against facility.CLASSES first, because the join
  and gap stages key on those names.

INPUT
  --ckpt   checkpoint to export (default: weights/best.pt of the newest run under <yolo_runs_dir>)

OUTPUT
  run/src/yolo26m_bikesign.pt   (an existing file is renamed to <name>.<unixtime>.pt)
  run/src/SHA256SUMS            updated line

USAGE  (from bicyclist_network/)
  python train/yolo/3_export_weights.py
  python train/yolo/3_export_weights.py --ckpt /path/to/best.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from bikelane_extract.config import Config
from bikelane_extract.facility import CLASSES

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from export_common import export_weights  # noqa: E402  (train/export_common.py)


def _names(path: Path):
    from ultralytics import YOLO
    names = YOLO(str(path)).names
    return [names[i] for i in sorted(names)]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", "-c", default=None, help="project file (default ./bikelane.yaml)")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--ckpt", default=None, help="checkpoint to export (default newest run's best.pt)")
    ap.add_argument("--force", action="store_true", help="export even if the class names differ")
    a = ap.parse_args(argv)
    cfg = Config.for_training(a.config, a.data_root)
    if a.ckpt:
        src = Path(a.ckpt)
    else:
        runs = cfg.path("yolo_runs_dir")
        cands = sorted(runs.glob("*/weights/best.pt"), key=lambda p: p.stat().st_mtime) if runs.exists() else []
        if not cands:
            raise SystemExit(f"no */weights/best.pt under {runs} — give --ckpt")
        src = cands[-1]
    try:
        names = _names(src)
        print(f"  classes in checkpoint: {names}")
        if tuple(names) != tuple(CLASSES) and not a.force:
            raise SystemExit(f"class names {names} differ from facility.CLASSES {list(CLASSES)}; "
                             "fix the dataset (1_prepare_dataset.py) or pass --force")
    except ImportError:
        print("  (ultralytics not installed — class names not checked)")
    export_weights(src, cfg.weights("yolo"), describe=lambda p: src.parent.parent.name)


if __name__ == "__main__":
    main()
