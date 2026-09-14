"""3_export_weights.py — copy a trained segmentation checkpoint into run/src/.

OBJECTIVE
  Make a model trained by 2_train.py the one the run stages use: copy best.pt
  to run/src/seg_unet_r34.pt (the name weights.seg in default.yaml expects)
  and refresh its line in run/src/SHA256SUMS so `python download.py` does not
  overwrite it later. Nothing else in run/ has to change.

INPUT
  --ckpt   checkpoint to export (default <ckpt_dir>/best.pt, i.e. the newest training run)

OUTPUT
  run/src/seg_unet_r34.pt      (an existing file is renamed to <name>.<unixtime>.pt)
  run/src/SHA256SUMS           updated line for seg_unet_r34.pt

USAGE  (from bicyclist_network/)
  python train/seg/3_export_weights.py
  python train/seg/3_export_weights.py --ckpt /path/to/best.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from bikelane_extract.config import Config

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from export_common import export_weights  # noqa: E402  (train/export_common.py)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", "-c", default=None, help="project file (default ./bikelane.yaml)")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--ckpt", default=None, help="checkpoint to export (default <ckpt_dir>/best.pt)")
    a = ap.parse_args(argv)
    cfg = Config.for_training(a.config, a.data_root)
    src = Path(a.ckpt) if a.ckpt else cfg.path("ckpt_dir") / "best.pt"
    dst = cfg.weights("seg")
    export_weights(src, dst, describe=_describe)


def _describe(path: Path) -> str:
    try:
        import torch
        ck = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(ck, dict) and "model" in ck:
            return f"epoch {ck.get('epoch')} mIoU {ck.get('miou', float('nan')):.3f}"
    except Exception as e:                      # torch missing or a bare state_dict
        return f"(not inspected: {e})"
    return "bare state_dict"


if __name__ == "__main__":
    main()
