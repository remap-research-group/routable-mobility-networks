"""1_prepare.py — OpenSatMap level-20 download → baked masks → 1024 px image/GT tile pairs.

OBJECTIVE
  Build the training set for the lane-marking segmentation model (2_train.py)
  from the OpenSatMap dataset (level 20, 0.15 m/px; CC BY-NC-SA 4.0 — check the
  licence fits your use). Ported from 01_prepare_data and 03_bake_masks. It
  drives the OpenSatMap `tools-release` scripts (external repo), so it needs:

  <opensatmap_dir>/<zip_prefix>.zip.001, .002, …   split archive of the images
  <opensatmap_dir>/<anno_json>                      line annotations
  <opensatmap_tools_dir>/                           the tools-release checkout

Steps:
  download fetch the level-20 split archive + annotation JSON from the
           Hugging Face dataset (z-hb/OpenSatMap, ~50 GB) and clone the
           OpenSatMap code repository for tools-release
  unzip    concatenate the split parts (they are a plain byte split) and extract
  bake     copy tools-release into <prep_work>, patch it (see below), run the
           mask + direction generation (1–3) and split/tile/clean (4–10)
  images   cut image tiles matching the GT tile names
  verify   image : GT tile counts must be 1:1

Patches applied to tools-release (it targets an old mmcv/mmengine stack and
a POSIX filesystem; our storage is CIFS):
  cut_satellite.py         mmcv.imread/imwrite → cv2 (note reversed arg order),
                           mmengine ProgressBar/mkdir_or_exist → local shims
  train_val_test_*.py      shutil.copy → shutil.copyfile (no permission copy)
The generated shell scripts avoid `set -e` and use `cp … || true` for the same reason.

INPUT / OUTPUT  (paths from run/bikelane_extract/default.yaml, overridable under
`paths:` in ./bikelane.yaml; all under data_root)
  paths.opensatmap_dir         raw download          default <data_root>/train/opensatmap
  paths.opensatmap_tools_dir   tools-release         default <data_root>/train/opensatmap/tools-release
  paths.prep_work_dir          baked masks + tiles   default <data_root>/train/seg/prep_work
  → <prep_work_dir>/picuse20save/final/allpic-cut/{train,val}/*.png            images
    <prep_work_dir>/picuse20save/final/pic20gtsplit-cut/category/{train,val}/*-GT.png  masks 0/1/2/3/255

USAGE  (from bicyclist_network/, the `bikelane` package installed: pip install -e ".[seg]")
  python train/seg/1_prepare.py --check
  python train/seg/1_prepare.py download     # (long) ~50 GB from Hugging Face + git clone of the tools
  python train/seg/1_prepare.py unzip        # concatenate parts and extract
  python train/seg/1_prepare.py bake         # (long) masks + split + 1024 px cut
  python train/seg/1_prepare.py images       # image tiles matching the GT tiles
  python train/seg/1_prepare.py verify       # image : GT tile counts must be 1:1
  python train/seg/1_prepare.py all
Needs POSIX shell tools (cat, unzip, bash) — use WSL on Windows.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

from bikelane_extract.config import Config

_TILE_RE = re.compile(r"^(.*)_(\d+)_(\d+)_(\d+)_(\d+)-GT\.png$")


def _sh(cmd, log=None):
    print(f"  $ {cmd}")
    if log:
        cmd = f"({cmd}) 2>&1 | tee -a '{log}'"
    return subprocess.run(cmd, shell=True).returncode


# ── download ──

def download(cfg: Config, p: dict):
    """OpenSatMap zips + annotations from Hugging Face; tools-release via git."""
    osm_dir = cfg.path("opensatmap_dir", mkdir=True)
    tools = cfg.path("opensatmap_tools_dir")
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise SystemExit("pip install huggingface_hub  (or 'bikelane-extract[seg]')")
    patterns = [f"{p['zip_prefix']}.zip.*", p["anno_json"]]
    print(f"  downloading {patterns} from {p['hf_repo']} → {osm_dir}  (~50 GB, resumable)")
    snapshot_download(repo_id=p["hf_repo"], repo_type="dataset", local_dir=str(osm_dir),
                      allow_patterns=patterns)
    parts = sorted(osm_dir.glob(f"{p['zip_prefix']}.zip.*"))
    print(f"  {len(parts)} zip parts, {sum(q.stat().st_size for q in parts)/1e9:.1f} GB; "
          f"{'ok' if (osm_dir/p['anno_json']).exists() else 'MISSING'} {p['anno_json']}")
    if not (tools / "cut_satellite.py").exists():
        clone = tools.parent / "_opensatmap_code"
        if not clone.exists():
            _sh(f"git clone --depth 1 {p['tools_repo']} '{clone}'")
        src = next(clone.rglob("cut_satellite.py"), None)
        if src is None:
            raise SystemExit(f"cut_satellite.py not found in {p['tools_repo']}; set "
                             f"paths.opensatmap_tools_dir to the tools folder manually")
        tools.parent.mkdir(parents=True, exist_ok=True)
        _sh(f"cp -r '{src.parent}' '{tools}'")
        print(f"  tools-release → {tools}")
    else:
        print(f"  tools already present: {tools}")


# ── unzip ──

def unzip(osm_dir: Path, prefix: str):
    parts = sorted(osm_dir.glob(f"{prefix}.zip.*"))
    if not parts:
        raise FileNotFoundError(f"no {prefix}.zip.* parts in {osm_dir}")
    merged = osm_dir / f"{prefix}_merged.zip"
    out = osm_dir / "extracted" / prefix
    if (out / f"picuse{prefix[:2]}trainvaltest").exists():
        print(f"  already extracted: {out}")
        return out
    head = open(parts[0], "rb").read(4)
    if head != b"PK\x03\x04":
        raise RuntimeError(f"first part does not start with a zip header ({head.hex()}): "
                           "this is a zip multi-volume, not a byte split — use `zip -s 0`")
    print(f"  {len(parts)} parts, {sum(p.stat().st_size for p in parts)/1e9:.1f} GB")
    if not merged.exists():
        _sh(f"cat {' '.join(str(p) for p in parts)} > '{merged}'")
    out.mkdir(parents=True, exist_ok=True)
    if _sh(f"unzip -q -o '{merged}' -d '{out}'"):
        raise RuntimeError("unzip failed")
    return out


# ── bake ──

def patch_tools(dst: Path):
    cut = dst / "cut_satellite.py"
    s = cut.read_text()
    s = s.replace("import mmcv", "import cv2")
    s = s.replace("import cv2\n", "import cv2\ndef _imwrite(img, path):\n    return cv2.imwrite(path, img)\n", 1)
    s = s.replace("mmcv.imread(", "cv2.imread(").replace("mmcv.imwrite(", "_imwrite(")
    s = s.replace("from mmengine.utils import ProgressBar, mkdir_or_exist", '''import os as _os
def mkdir_or_exist(d):
    _os.makedirs(d, exist_ok=True)
class ProgressBar:
    def __init__(self, task_num=0, *a, **k):
        self.task_num = task_num; self.completed = 0
    def update(self, n=1):
        self.completed += n
        print(f"  {self.completed}/{self.task_num}", end="\\r")''')
    cut.write_text(s)
    for f in ("train_val_test_tag.py", "train_val_test_gt.py"):
        p = dst / f
        p.write_text(p.read_text().replace("shutil.copy(", "shutil.copyfile("))
    left = subprocess.run(f"grep -ln 'mmcv\\|mmengine' {dst}/*.py", shell=True,
                          capture_output=True, text=True).stdout.strip()
    print("  tools patched" + (f"; still referencing mm*: {left}" if left else ""))


def write_scripts(work: Path, anno: Path, pic_use: Path, p: dict):
    sf = "picuse20save"
    run1 = work / "run_bake.sh"
    run1.write_text(f'''#!/bin/bash
cd "{work}"
SAVE_FOLDER={sf}
JSON_FILE="{anno}"
PIC_SRC="{pic_use}"
TARGET_FOLDER=${{SAVE_FOLDER}}/pic20gt
echo "=== 1. gather images ==="
rm -rf allpic ${{SAVE_FOLDER}}
mkdir -p allpic/use
cp "$PIC_SRC"/train/* allpic/use/ 2>/dev/null || true
cp "$PIC_SRC"/val/*   allpic/use/ 2>/dev/null || true
cp "$PIC_SRC"/*filenames.txt allpic/ 2>/dev/null || true
echo "=== 2. multiclass GT masks ==="
python tools-release/generate_GT_multiclass_only-multi.py --file_name "$JSON_FILE" \\
    --target_folder $TARGET_FOLDER --source_folder allpic --image_size {p["image_size"]}
echo "=== 3. direction tag GT ==="
python tools-release/generate_GT_direction_tag-multi.py --file_name "$JSON_FILE" \\
    --target_folder $TARGET_FOLDER --source_folder allpic --image_size {p["image_size"]}
echo "=== 1-3 DONE ==="
''')
    run2 = work / "run_split_tile.sh"
    run2.write_text(f'''#!/bin/bash
cd "{work}"
SAVE_FOLDER={sf}
PIC_USE=allpic
TARGET_FOLDER=${{SAVE_FOLDER}}/pic20gt
AFTER=${{SAVE_FOLDER}}/pic20gtsplit
echo "=== 4. split tag GT ==="
python tools-release/train_val_test_tag.py --source_folder $TARGET_FOLDER-mask-tag/use \\
    --target_folder $AFTER --pic_folder $PIC_USE --zoom {p["zoom"]}
echo "=== 5. split other GT ==="
python tools-release/train_val_test_gt.py --source_folder $TARGET_FOLDER \\
    --target_folder $AFTER --pic_folder $PIC_USE --zoom {p["zoom"]}
echo "=== 6. cut into tiles ==="
bash tools-release/run_cut_satellite.sh $AFTER $PIC_USE {p["clip_size"]}
echo "=== 7. check npy ==="
bash tools-release/check_npy.sh $AFTER-cut
echo "=== 8. find black tiles ==="
bash tools-release/run_black_images.sh $AFTER-cut $PIC_USE-cut
echo "=== 9. remove black tiles ==="
bash tools-release/remove_black_image.sh $PIC_USE-cut/txt $AFTER-cut $PIC_USE-cut
echo "=== 10. move final ==="
mkdir -p ${{SAVE_FOLDER}}/final
mv $PIC_USE-cut ${{SAVE_FOLDER}}/final
mv $AFTER-cut ${{SAVE_FOLDER}}/final
echo "=== 4-10 DONE ==="
''')
    return run1, run2


def bake(cfg: Config, extracted: Path):
    p = cfg.get("prepare")
    osm_dir, tools_src = cfg.path("opensatmap_dir"), cfg.path("opensatmap_tools_dir")
    work = cfg.path("prep_work_dir", mkdir=True)
    anno = osm_dir / p["anno_json"]
    pic_use = next(extracted.rglob("picuse*trainvaltest"), None)
    if pic_use is None or not anno.exists():
        raise FileNotFoundError(f"need {anno} and a picuse*trainvaltest folder under {extracted}")
    for sub in ("train", "val", "test"):
        n = len(list((pic_use / sub).glob("*.png"))) if (pic_use / sub).exists() else 0
        print(f"  {sub:5s} {n:5d} png")
    dst = work / "tools-release"
    if dst.exists():
        shutil.rmtree(dst, ignore_errors=True)
    dst.mkdir(parents=True)
    _sh(f"cp '{tools_src}'/*.py '{tools_src}'/*.sh '{dst}'/")
    patch_tools(dst)
    run1, run2 = write_scripts(work, anno, pic_use, p)
    print("  baking masks (long — this is the tmux step)")
    _sh(f"bash '{run1}'", log=work / "bake.log")
    _sh(f"bash '{run2}'", log=work / "split_tile.log")
    return pic_use


# ── image tiles + verify ──

def cut_image_tiles(cfg: Config, src_img_dir: Path):
    """Image tiles named like the GT tiles: <stem>_<x0>_<y0>_<x1>_<y1>.png."""
    import numpy as np
    from PIL import Image
    from tqdm import tqdm
    Image.MAX_IMAGE_PIXELS = None
    final = cfg.path("prep_work_dir") / "picuse20save" / "final"
    out_root, gt_root = final / "allpic-cut", final / "pic20gtsplit-cut" / "category"
    cache = {}
    for split in ("train", "val", "not_use_train"):
        gts = sorted((gt_root / split).glob("*-GT.png")) if (gt_root / split).exists() else []
        if not gts:
            continue
        (out_root / split).mkdir(parents=True, exist_ok=True)
        made = skipped = 0
        for gt in tqdm(gts, desc=f"  tiles {split}"):
            m = _TILE_RE.match(gt.name)
            if not m:
                skipped += 1
                continue
            stem, x0, y0, x1, y1 = m.group(1), *map(int, m.groups()[1:])
            out = out_root / split / f"{stem}_{x0}_{y0}_{x1}_{y1}.png"
            if out.exists():
                made += 1
                continue
            if stem not in cache:
                src = next(src_img_dir.rglob(f"{stem}.png"), None)
                cache[stem] = np.array(Image.open(src).convert("RGB")) if src else None
            if cache[stem] is None:
                skipped += 1
                continue
            Image.fromarray(cache[stem][y0:y1, x0:x1]).save(out)
            made += 1
        cache.clear()
        print(f"  {split}: {made} tiles, {skipped} skipped")


def verify(cfg: Config):
    final = cfg.path("prep_work_dir") / "picuse20save" / "final"
    n_img = len(list((final / "allpic-cut").rglob("*.png")))
    n_gt = len(list((final / "pic20gtsplit-cut" / "category").rglob("*-GT.png")))
    print(f"  image tiles {n_img} | GT tiles {n_gt} | " + ("1:1 ok" if n_img == n_gt else "MISMATCH"))
    return n_img == n_gt


def run(cfg: Config, argv=None, check: bool = False):
    import argparse
    ap = argparse.ArgumentParser(prog="1_prepare.py")
    ap.add_argument("step", choices=["download", "unzip", "bake", "images", "verify", "all"],
                    nargs="?", default="all")
    a = ap.parse_args(argv)
    p = cfg.get("prepare")
    osm_dir, tools = cfg.path("opensatmap_dir"), cfg.path("opensatmap_tools_dir")
    print(f"prepare\n  opensatmap: {osm_dir}\n  tools: {tools}\n  work: {cfg.path('prep_work_dir')}")
    if check:
        for f in (osm_dir / p["anno_json"], tools / "cut_satellite.py"):
            print(f"  {'ok ' if f.exists() else 'MISSING'} {f}")
        print(f"  {len(list(osm_dir.glob(p['zip_prefix']+'.zip.*')))} zip parts")
        return
    extracted = osm_dir / "extracted" / p["zip_prefix"]
    if a.step in ("download", "all"):
        download(cfg, p)
    if a.step in ("unzip", "all"):
        extracted = unzip(osm_dir, p["zip_prefix"])
    pic_use = next(extracted.rglob("picuse*trainvaltest"), extracted)
    if a.step in ("bake", "all"):
        pic_use = bake(cfg, extracted)
    if a.step in ("images", "all"):
        cut_image_tiles(cfg, pic_use)
    if a.step in ("verify", "all"):
        verify(cfg)
    print("→ next: python train/seg/2_train.py")


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--config", "-c", default=None, help="project file (default ./bikelane.yaml)")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--check", action="store_true", help="show the paths and what exists, run nothing")
    a, rest = ap.parse_known_args(argv)
    run(Config.for_training(a.config, a.data_root), rest, check=a.check)


if __name__ == "__main__":
    main()
