"""bikelane join -c <cfg> {match|clean} [--sweep] [--radius R]

  match   signs × centerlines within radius_m → <region>_bikelanes.geojson   10
          --sweep: match rate / lines-per-sign / km for several radii
  clean   hooks + overlaps with evidence merge → <region>_bikelanes_clean     10b
"""
from __future__ import annotations

import argparse

from ..config import Config
from . import clean, spatial_join


def run(cfg: Config, argv=None, check: bool = False):
    ap = argparse.ArgumentParser(prog="bikelane join", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["match", "clean"])
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--radius", type=float, default=None, help="(match) override join.radius_m")
    a = ap.parse_args(argv)
    if a.step == "match":
        return spatial_join.run(cfg, check=check, sweep=a.sweep, radius=a.radius)
    return clean.run(cfg, check=check)
