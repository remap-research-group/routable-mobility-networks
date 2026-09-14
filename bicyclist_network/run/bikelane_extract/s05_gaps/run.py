"""bikelane gaps -c <cfg> {scan|join|intersections} [--sweep]

  scan            candidates only, split into straight / crossing  (11)
  join            apply confirmed straight candidates               (12)
  intersections   remaining gaps; crossings via OSM junction nodes  (16)
  network         facilities + connectors → one edge layer with shared node ids, + node layer

Run them in that order, and look at the scan output before joining.
"""
from __future__ import annotations

import argparse

from ..config import Config
from . import intersections, join, network, scan


def run(cfg: Config, argv=None, check: bool = False):
    ap = argparse.ArgumentParser(prog="bikelane gaps", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["scan", "join", "intersections", "network"])
    ap.add_argument("--sweep", action="store_true",
                    help="(scan) print how candidate counts change with gap ceiling, then exit")
    a = ap.parse_args(argv)
    if a.step == "scan":
        return scan.run(cfg, check=check, sweep=a.sweep)
    if a.step == "join":
        return join.run(cfg, check=check)
    if a.step == "network":
        return network.run(cfg, check=check)
    return intersections.run(cfg, check=check)
