"""1_prepare_dataset.py — labeled tiles (YOLO txt) → train/val split + data.yaml.

OBJECTIVE
  Turn a folder of labeled tiles into the dataset layout ultralytics trains on.
  The released detector (run/src/yolo26m_bikesign.pt) was trained on bike
  pavement symbols labeled on 1024 px MassDOT tiles — the same tiles the run
  stages use (download_MassGIS2025AerialImagery output) — so label those:
  they are already 0.15 m/px and 1024 px, which is what `signs detect` runs at.
  The labeled tiles are not part of this repository.

INPUT
  --src DIR   a folder with
                images/  *.jpg | *.png           (any tool that exports "YOLO" format:
                labels/  <same stem>.txt          CVAT, Roboflow, Label Studio, labelImg …)
              each label line: <class_id> <cx> <cy> <w> <h>   (normalized 0–1)
              class ids follow yolo.classes in default.yaml: 0 BikeOnly, 1 Sharrow, 2 OnlyBikeBus.
              A tile with no symbols may have an empty .txt (or none): it is kept as a negative.
              If images/ already contains train/ and val/ subfolders the split is kept as is.
  yolo.val_frac, yolo.seed   (default.yaml / bikelane.yaml)   share held out for validation

OUTPUT   (<yolo_data_dir>, default <data_root>/train/yolo/dataset/)
  images/{train,val}/, labels/{train,val}/   copies (or symlinks with --link) of the inputs
  data.yaml                                  path, train, val, names — the file 2_train.py reads
  split.json                                 which stem went where (seeded, reproducible)

USAGE  (from bicyclist_network/)
  python train/yolo/1_prepare_dataset.py --src /path/to/labeled_tiles
  python train/yolo/1_prepare_dataset.py --src /path/to/labeled_tiles --link     # symlink, no copy
  python train/yolo/1_prepare_dataset.py --check                                 # what is there now
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter
from pathlib import Path

import yaml

from bikelane_extract.config import Config
from bikelane_extract.facility import CLASSES

IMG_EXT = (".jpg", ".jpeg", ".png", ".tif", ".tiff")


def _pairs(img_dir: Path, lbl_dir: Path):
    """(image, label-or-None) for every image; a missing label = no symbols."""
    out = []
    for im in sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXT):
        lb = lbl_dir / f"{im.stem}.txt"
        out.append((im, lb if lb.exists() else None))
    return out


def _class_counts(pairs, n_cls):
    c, bad = Counter(), 0
    for _, lb in pairs:
        if lb is None:
            continue
        for line in lb.read_text().split("\n"):
            if not line.strip():
                continue
            try:
                k = int(line.split()[0])
            except ValueError:
                bad += 1
                continue
            if 0 <= k < n_cls:
                c[k] += 1
            else:
                bad += 1
    return c, bad


def _place(pairs, split, out: Path, link: bool):
    (out / "images" / split).mkdir(parents=True, exist_ok=True)
    (out / "labels" / split).mkdir(parents=True, exist_ok=True)
    for im, lb in pairs:
        di = out / "images" / split / im.name
        dl = out / "labels" / split / f"{im.stem}.txt"
        for s, d in ((im, di), (lb, dl)):
            if d.exists() or d.is_symlink():
                d.unlink()
            if s is None:
                d.write_text("")                      # explicit negative
            elif link:
                d.symlink_to(s.resolve())
            else:
                shutil.copyfile(s, d)


def write_data_yaml(out: Path, names) -> Path:
    data = {"path": str(out.resolve()), "train": "images/train", "val": "images/val",
            "names": {i: n for i, n in enumerate(names)}}
    f = out / "data.yaml"
    f.write_text(yaml.safe_dump(data, sort_keys=False))
    return f


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", "-c", default=None, help="project file (default ./bikelane.yaml)")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--src", default=None, help="folder with images/ and labels/")
    ap.add_argument("--link", action="store_true", help="symlink instead of copying")
    ap.add_argument("--check", action="store_true", help="report the dataset folder, run nothing")
    a = ap.parse_args(argv)
    cfg = Config.for_training(a.config, a.data_root)
    y = cfg.get("yolo")
    names = list(y.get("classes", CLASSES))
    if tuple(names) != tuple(CLASSES):
        print(f"  WARNING: yolo.classes {names} differs from facility.CLASSES {list(CLASSES)} — "
              "the join/gap stages expect the latter names")
    out = cfg.path("yolo_data_dir")
    print(f"yolo dataset → {out}\n  classes: {names}")
    if a.check or not a.src:
        for split in ("train", "val"):
            d = out / "images" / split
            n = len(list(d.glob("*"))) if d.exists() else 0
            print(f"  {split:5s} {n:5d} images  {'ok ' if n else 'MISSING'} {d}")
        print(f"  {'ok ' if (out/'data.yaml').exists() else 'MISSING'} {out/'data.yaml'}")
        if not a.src and not a.check:
            print("  give --src <folder with images/ and labels/> to build it")
        return

    src = Path(a.src).expanduser()
    img_root, lbl_root = src / "images", src / "labels"
    if not img_root.exists():
        raise SystemExit(f"{img_root} not found (expected images/ and labels/ under --src)")
    if (img_root / "train").exists():                   # already split
        splits = {s: _pairs(img_root / s, lbl_root / s) for s in ("train", "val")
                  if (img_root / s).exists()}
        print("  keeping the existing train/val split")
    else:
        pairs = _pairs(img_root, lbl_root)
        if not pairs:
            raise SystemExit(f"no images in {img_root}")
        rng = random.Random(int(y.get("seed", 42)))
        rng.shuffle(pairs)
        n_val = max(1, int(round(len(pairs) * float(y.get("val_frac", 0.15)))))
        splits = {"val": pairs[:n_val], "train": pairs[n_val:]}
    out.mkdir(parents=True, exist_ok=True)
    for split, pairs in splits.items():
        _place(pairs, split, out, a.link)
        counts, bad = _class_counts(pairs, len(names))
        neg = sum(1 for _, lb in pairs if lb is None or not lb.read_text().strip())
        per = ", ".join(f"{names[k]} {counts[k]}" for k in range(len(names)))
        print(f"  {split:5s} {len(pairs):5d} tiles ({neg} without symbols) | boxes: {per}"
              + (f" | {bad} bad label lines IGNORED" if bad else ""))
    (out / "split.json").write_text(json.dumps(
        {s: [im.name for im, _ in pairs] for s, pairs in splits.items()}, indent=1))
    f = write_data_yaml(out, names)
    print(f"  data.yaml → {f}\n→ next: python train/yolo/2_train.py")


if __name__ == "__main__":
    main()
