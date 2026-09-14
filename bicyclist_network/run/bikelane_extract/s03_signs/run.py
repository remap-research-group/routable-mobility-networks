"""bikelane signs -c <cfg> {detect|filter} [--fresh] [--sweep]

  detect   YOLO on all tiles at conf_infer → raw CSV (once)          09 §2
  filter   raw → accepted/dropped by conf_keep, isolation flag       09 §3-4
           --sweep: isolated-share table per threshold, to choose conf_keep
"""
from __future__ import annotations

import argparse

from ..config import Config
from . import detect, filter as filt


def run(cfg: Config, argv=None, check: bool = False):
    ap = argparse.ArgumentParser(prog="bikelane signs", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["detect", "filter"])
    ap.add_argument("--fresh", action="store_true", help="(detect) re-run inference")
    ap.add_argument("--sweep", action="store_true", help="(filter) threshold table only")
    ap.add_argument("--device", default=None, help="(detect) cuda / mps / cpu (auto if omitted)")
    a = ap.parse_args(argv)
    if a.step == "detect":
        return detect.run(cfg, check=check, fresh=a.fresh, device=a.device)
    return filt.run(cfg, check=check, sweep=a.sweep)
