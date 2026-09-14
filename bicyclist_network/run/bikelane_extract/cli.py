"""bikelane <stage> [step] [options]

Reads ./bikelane.yaml (the area, data_root, imagery_root, optional overrides)
and runs one stage. Stages run one at a time; each reads the previous stage's
files from <data_root>/<subdir>/<REGION>/. Nothing runs end to end on purpose —
every stage has a check step (a sweep, a plot, a QGIS look) between it and the
next.

Input imagery (1024 px tiles + Tile_Mappings.csv) is produced by the sibling
tool download_MassGIS2025AerialImagery/ and read from
<imagery_root>/<REGION>/. Model weights live in run/src/ (python download.py).
Training (segmentation model, YOLO detector) is in train/, not in this CLI.

All parameters live in the package's default.yaml; override them in
bikelane.yaml, or for one run with --set key=value. `bikelane config` prints
what is in effect.
  1  predict       tiles + seg weights → predictions/<REGION>/*_pred.png
  2  centerlines   predictions → voronoi centerlines → cleaned centerlines
  3  signs         tiles + YOLO weights → bike-sign points (raw / filtered / dropped)
  4  join          centerlines × signs → bike lanes with type + evidence counts
  5  gaps          scan → join → intersections (via OSM junction nodes) → network
     osm           fetch OSM drive network and junction nodes for the tile extent
     config        print the parameters and paths in effect for an area
"""
from __future__ import annotations

import argparse
import importlib
import sys

from .config import Config, TodoParameter, parse_value

# subcommand → (module, function). Modules are imported lazily so that a
# missing optional dependency (torch, ultralytics, osmnx…) only breaks the
# stage that needs it.
STAGES = {
    "predict":     ("bikelane_extract.s01_predict.predict", "run"),
    "centerlines": ("bikelane_extract.s02_centerline.run", "run"),
    "signs":       ("bikelane_extract.s03_signs.run", "run"),
    "join":        ("bikelane_extract.s04_join.run", "run"),
    "gaps":        ("bikelane_extract.s05_gaps.run", "run"),
    "osm":         ("bikelane_extract.osm", "run"),
}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="bikelane", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=list(STAGES) + ["config"])
    ap.add_argument("--config", "-c", default=None, help="project file (default ./bikelane.yaml)")
    ap.add_argument("--city", default=None, help="override the town in the project file for this run")
    ap.add_argument("--state", default=None)
    ap.add_argument("--data-root", default=None, help="override data_root for this run")
    ap.add_argument("--imagery-root", default=None,
                    help="override imagery_root for this run (folder holding <REGION>/tiles + Tile_Mappings.csv)")
    ap.add_argument("--crs", default=None)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override a parameter for this run, e.g. --set join.radius_m=1.5")
    ap.add_argument("--check", action="store_true",
                    help="load config and validate inputs for this stage, then exit")
    args, rest = ap.parse_known_args(argv)

    overrides = {}
    for kv in args.set:
        if "=" not in kv:
            sys.exit(f"--set expects KEY=VALUE, got {kv}")
        k, v = kv.split("=", 1)
        overrides[k] = parse_value(v)
    try:
        cfg = Config.from_args(args.city, args.state, args.data_root, args.crs, overrides, args.config,
                               imagery_root=args.imagery_root)
    except TodoParameter as e:
        sys.exit(f"config error: {e}")

    if args.stage == "config":
        _config_cmd(cfg, rest)
        return 0

    mod_name, fn_name = STAGES[args.stage]
    try:
        mod = importlib.import_module(mod_name)
    except ModuleNotFoundError as e:
        if e.name and e.name.startswith("bikelane_extract"):
            sys.exit(f"stage '{args.stage}' is not ported yet ({mod_name})")
        sys.exit(f"stage '{args.stage}' needs an optional dependency: {e.name}\n"
                 f"  pip install 'bikelane-extract[{_extra_for(args.stage)}]'")
    fn = getattr(mod, fn_name)
    if not args.check:
        cfg.snapshot(args.stage)
    try:
        fn(cfg, rest, check=args.check)     # stage return values are for tests, not the shell
    except TodoParameter as e:
        sys.exit(f"config error: {e}")
    return 0


def _config_cmd(cfg, rest):
    """bikelane config : print the effective configuration."""
    import yaml
    area = (f"boundary {cfg.get('boundary')}" if cfg.get("boundary", None)
            else f"{cfg.get('city', '?')}, {cfg.get('state', '')}")
    print(f"region: {cfg.region} ({area})  crs: {cfg.crs}")
    print(f"data_root:    {cfg.data_root}\nimagery_root: {cfg.imagery_root}")
    print("paths:")
    for k in ("tiles_dir", "tile_mappings_csv", "predictions_dir", "centerlines_final",
              "signs_filtered", "bikelanes_final", "network_edges", "osm_nodes"):
        p = cfg.path(k)
        print(f"  {k:20s} {p}  [{'ok' if p.exists() else 'missing'}]")
    for w in ("seg", "yolo"):
        p = cfg.weights(w)
        print(f"  weights.{w:12s} {p}  [{'ok' if p.exists() else 'missing'}]")
    print(f"project file: {cfg.source}\nparameters in effect:")
    body = {k: v for k, v in cfg._d.items()
            if k not in ("city", "state", "region", "crs", "data_root", "imagery_root", "paths")}
    print("  " + yaml.safe_dump(body, sort_keys=False, allow_unicode=True).replace("\n", "\n  "))


def _extra_for(stage: str) -> str:
    return {"predict": "seg", "signs": "signs", "osm": "osm"}.get(stage, "all")


if __name__ == "__main__":
    main()
