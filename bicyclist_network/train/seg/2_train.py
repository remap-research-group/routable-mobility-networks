"""2_train.py — OpenSatMap tile pairs → U-Net (ResNet34) lane-marking segmentation.

OBJECTIVE
  Train the segmentation model that `bikelane predict` (run/) applies to every
  tile. Ported from 04_train_segmentation. Classes: 0 bg, 1 lane line, 2 curb,
  3 virtual line; 255 = ignore. Ground truth is 1 px wide, so masks are
  dilated (train.dilate_px) and the loss is CE (bg weight 0.1) + Dice. The
  architecture is built by run/bikelane_extract/model/common.py — the same
  code predict uses — so a checkpoint from here loads there unchanged.

INPUT  (produced by 1_prepare.py)
  <prep_work>/picuse20save/final/allpic-cut/<split>/<stem>.png
  <prep_work>/picuse20save/final/pic20gtsplit-cut/category/<split>/<stem>-GT.png
  hyper-parameters: the `train:` section of run/bikelane_extract/default.yaml
  (encoder, num_classes, dilate_px, batch, epochs, lr, …; override in ./bikelane.yaml)

OUTPUT
  <ckpt_dir>/last.pt, best.pt   ({'model','epoch','miou','best'}), best by mIoU over
                                classes 1–3. Default <data_root>/train/seg/ckpts/
  → python train/seg/3_export_weights.py   copies best.pt to run/src/seg_unet_r34.pt

USAGE  (from bicyclist_network/; pip install -e ".[seg]" and a CUDA build of torch)
  python train/seg/2_train.py --check
  python train/seg/2_train.py                  # (long) 30 epochs, batch 8 — run inside tmux
  python train/seg/2_train.py --resume         # continue from last.pt
  python train/seg/2_train.py --epochs 5       # smoke test
  python train/seg/2_train.py --device cuda:1
The reference model reached lane IoU ≈ 0.41 — low because the lines are 1–3 px
wide, not because the model misses them.

Negative result worth knowing before changing the loss: adding 0.5×clDice
(30 ep) cut fragmentation 10–17 % but lost 28–31 % coverage — it drops the
faint short segments that residential centerlines are made of. Baseline kept.
"""
from __future__ import annotations

import time

import cv2
import numpy as np
from PIL import Image

from bikelane_extract.config import Config
from bikelane_extract.model import device as dev
from bikelane_extract.model.common import IGNORE, IMAGENET_MEAN, IMAGENET_STD, make_model

Image.MAX_IMAGE_PIXELS = None


def data_roots(cfg: Config):
    final = cfg.path("prep_work_dir") / "picuse20save" / "final"
    return final / "allpic-cut", final / "pic20gtsplit-cut" / "category"


class LaneDataset:
    def __init__(self, img_root, gt_root, split, dilate=3, augment=False):
        import torch
        from torch.utils.data import Dataset
        self.torch = torch
        self.img_dir, self.gt_dir = img_root / split, gt_root / split
        self.imgs = [p for p in sorted(self.img_dir.glob("*.png"))
                     if (self.gt_dir / f"{p.stem}-GT.png").exists()]
        self.augment = augment
        self.kernel = (cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate, dilate))
                       if dilate > 1 else None)

    def __len__(self):
        return len(self.imgs)

    def _dilate(self, m):
        if self.kernel is None:
            return m
        out = m.copy()
        for cls in (1, 2, 3):
            d = cv2.dilate((m == cls).astype(np.uint8), self.kernel)
            out[(d == 1) & (out == 0)] = cls
        return out

    def __getitem__(self, i):
        ip = self.imgs[i]
        img = np.array(Image.open(ip).convert("RGB"), np.float32) / 255.0
        m = np.array(Image.open(self.gt_dir / f"{ip.stem}-GT.png"))
        if m.ndim == 3:
            m = m[:, :, 0]
        m = self._dilate(m.astype(np.int64))
        if self.augment:
            if np.random.rand() < 0.5:
                img, m = img[:, ::-1].copy(), m[:, ::-1].copy()
            if np.random.rand() < 0.5:
                img, m = img[::-1].copy(), m[::-1].copy()
            k = np.random.randint(4)
            if k:
                img, m = np.rot90(img, k).copy(), np.rot90(m, k).copy()
            if np.random.rand() < 0.3:
                img = np.clip(img * np.random.uniform(0.8, 1.2), 0, 1)
        img = (img - IMAGENET_MEAN) / IMAGENET_STD
        return (self.torch.from_numpy(img.transpose(2, 0, 1)).float(),
                self.torch.from_numpy(m).long())


def iou_per_class(pred, target, n_cls, ignore=IGNORE):
    valid = target != ignore
    out = []
    for c in range(n_cls):
        p, t = (pred == c) & valid, (target == c) & valid
        u = (p | t).sum().item()
        out.append((p & t).sum().item() / u if u > 0 else float("nan"))
    return out


def run(cfg: Config, argv=None, check: bool = False):
    import argparse
    ap = argparse.ArgumentParser(prog="2_train.py")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--resume", action="store_true", help="continue from last.pt")
    ap.add_argument("--device", default=None, help="cuda / mps / cpu (auto if omitted)")
    a = ap.parse_args(argv)
    tr = cfg.get("train")
    img_root, gt_root = data_roots(cfg)
    ckpt_dir = cfg.path("ckpt_dir", mkdir=not check)
    print(f"train\n  images: {img_root}\n  gt: {gt_root}\n  ckpts: {ckpt_dir}")
    if check:
        for f in (img_root / "train", gt_root / "train", img_root / "val", gt_root / "val"):
            print(f"  {'ok ' if f.exists() else 'MISSING'} {f}")
        return

    import torch
    import torch.nn as nn
    import segmentation_models_pytorch as smp
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    device = dev.pick(a.device)
    torch.backends.cudnn.benchmark = True
    if device != "cuda":
        print(f"  WARNING: training on {dev.describe(device)} — expect days, not hours. "
              "Use the released weights unless you need to retrain.")
    epochs = a.epochs or tr["epochs"]
    n_cls = tr["num_classes"]

    train_ds = LaneDataset(img_root, gt_root, "train", tr["dilate_px"], augment=True)
    val_ds = LaneDataset(img_root, gt_root, "val", tr["dilate_px"], augment=False)
    nw = tr.get("num_workers", 8) if device == "cuda" else min(tr.get("num_workers", 8), 4)
    train_ld = DataLoader(train_ds, batch_size=tr["batch"], shuffle=True, num_workers=nw,
                          pin_memory=True, drop_last=True)
    val_ld = DataLoader(val_ds, batch_size=tr["batch"], shuffle=False, num_workers=nw,
                        pin_memory=True)
    print(f"  train {len(train_ds)} | val {len(val_ds)} | {len(train_ld)} batches/epoch | {dev.describe(device)}")

    model = make_model(tr["encoder"], n_cls, pretrained=True).to(device)
    w = torch.tensor([tr["bg_weight"]] + [1.0] * (n_cls - 1)).to(device)
    ce = nn.CrossEntropyLoss(ignore_index=IGNORE, weight=w)
    dice = smp.losses.DiceLoss(mode="multiclass", ignore_index=IGNORE)
    opt = torch.optim.AdamW(model.parameters(), lr=tr["lr"], weight_decay=tr["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.amp.GradScaler(enabled=device == "cuda")
    start, best = 0, 0.0
    if a.resume and (ckpt_dir / "last.pt").exists():
        ck = torch.load(ckpt_dir / "last.pt", map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        start, best = ck["epoch"] + 1, ck.get("best", 0.0)
        for _ in range(start):
            sched.step()
        print(f"  resumed at epoch {start+1}")

    def evaluate():
        model.eval()
        acc, cnt = np.zeros(n_cls), np.zeros(n_cls)
        with torch.no_grad():
            for img, m in val_ld:
                img, m = img.to(device), m.to(device)
                with dev.autocast(device):
                    out = model(img)
                for c, v in enumerate(iou_per_class(out.argmax(1), m, n_cls)):
                    if not np.isnan(v):
                        acc[c] += v
                        cnt[c] += 1
        return acc / np.maximum(cnt, 1)

    for ep in range(start, epochs):
        model.train()
        t0, running = time.time(), 0.0
        pbar = tqdm(train_ld, desc=f"  epoch {ep+1}/{epochs}")
        for img, m in pbar:
            img, m = img.to(device), m.to(device)
            opt.zero_grad()
            with dev.autocast(device):
                out = model(img)
                loss = ce(out, m) + dice(out, m)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            running += loss.item()
            pbar.set_postfix(loss=f"{running/(pbar.n+1):.4f}")
        sched.step()
        ious = evaluate()
        miou = float(np.nanmean(ious[1:]))
        print(f"  [{ep+1}] loss {running/len(train_ld):.4f} | IoU lane {ious[1]:.3f} "
              f"curb {ious[2]:.3f} virtual {ious[3]:.3f} | mIoU {miou:.3f} | {time.time()-t0:.0f}s")
        state = {"model": model.state_dict(), "epoch": ep, "miou": miou, "best": best}
        torch.save(state, ckpt_dir / "last.pt")
        if miou > best:
            best = miou
            state["best"] = best
            torch.save(state, ckpt_dir / "best.pt")
            print(f"  ★ new best mIoU {miou:.3f}")
    print(f"  done. best mIoU {best:.3f} → python train/seg/3_export_weights.py  "
          f"(copies {ckpt_dir/'best.pt'} to {cfg.weights('seg')})")


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
