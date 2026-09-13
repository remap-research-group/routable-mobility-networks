"""Step 2 — train DBSwinT_v4 on the prepared dataset.

OBJECTIVE
  Train the dual-branch Swin-T segmentation model with four heads (5-class
  condition-aware map, binary pedestrian-network head, auxiliary entrance and
  crosswalk heads) on the DC 2023 + 2025 tiles, keep the best checkpoint by
  validation network IoU, and calibrate its confidence temperatures.

INPUT  (paths from config.py)
  train/v0_2026sep_image/dc_2023, dc_2025: images/, masks/, splits.json
  train/v0_2026sep_image/combined_dataset_stats.json: class weights
  ImageNet-pretrained Swin-T weights for the local branch (downloaded by timm
  on first use; --no-pretrained trains from scratch)

OUTPUT  (train/output/runs/<timestamp>_<name>/)
  best.pt            {'model_state', 'epoch', 'val_metrics', 'best_metric'}
  last.pt            resumable state (model, optimizer, scheduler)
  config.json        config + CLI snapshot (read by run/src/loader.py)
  history.json       per-epoch losses and metrics
  train.log
  temperature.json   per-head calibration temperatures (unless --no-calibrate)

HOW IT WORKS
  main()  builds datasets (dataset.load_dataset_pairs, SidewalkDataset with
          albumentations: random 512-px crops, flips, rotations, colour
          jitter; --strong-aug adds shift/scale/rotate, brightness/contrast,
          noise) -> model.DBSwinT_v4 -> losses.CombinedLoss -> AdamW with two
          LR groups (utils.build_param_groups: pretrained branch vs. scratch)
          -> warmup + cosine schedule -> engine.train_one_epoch / validate per
          epoch -> best / last checkpoints -> calibration.calibrate_run on the
          val split.
  The loss (losses.py) builds gap-tolerant targets on the GPU per batch: the
  network target is morphologically closed (GAP_CLOSE_PX) and the filled gap
  band is ignored (index 255) by the 5-class CE; weighted CE + Dice + RMI per
  head; a connectivity loss penalizes crosswalk probability farther than
  CONN_RADIUS_PX from predicted walkable surface.

TUNING — parameters that matter (defaults in config.py, all overridable)
  --epochs 120, --batch-size 4, --img-size 512 (crop size; it is also the
      inference window and lands in run/src/train_config.json)
  --lr 2e-4 (scratch), --lr-pretrained 2e-5 (Swin-T branch),
      --weight-decay 1e-3, --warmup-epochs 5
  --gap-close-px 2       closing radius of the network target (0 = off);
                         larger bridges bigger label gaps but blurs true gaps
  --conn-radius 8, --lambda-conn 0.20   crosswalk connectivity loss (px, weight)
  --rmi-net 0.5, --rmi-en 0.10, --rmi-cw 0.10   RMI loss weight per head
  --class-weights a,b,c,d,e   override the inverse-frequency weights
      (clipped at MAX_CLASS_WEIGHT = 25)
  --best-metric iou | f1 | tree_f1 | ent_f1 | cw_f1   checkpoint selection
  --no-entrance-head / --no-crosswalk-head, --no-pretrained, --no-strong-aug,
  --no-calibrate, --ds-2023-only / --ds-2025-only, --resume <run_dir>

USAGE
  python 2_train.py --name v4_unified            # full training (long; use tmux)
  python 2_train.py --resume <run_folder>
  python 2_train.py --ds-2025-only --epochs 60
"""
import json
import math
import time
import random
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from config import cfg
from dataset import SidewalkDataset, build_transforms, load_dataset_pairs
from model import DBSwinT_v4
from losses import CombinedLoss
from engine import train_one_epoch, validate
from utils import make_run_dir, save_run_config, setup_logger, build_param_groups
from calibration import calibrate_run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs',           type=int,   default=cfg.EPOCHS)
    ap.add_argument('--batch-size',       type=int,   default=cfg.BATCH_SIZE)
    ap.add_argument('--lr',               type=float, default=cfg.LR)
    ap.add_argument('--lr-pretrained',    type=float, default=cfg.LR_PRETRAINED)
    ap.add_argument('--weight-decay',     type=float, default=cfg.WEIGHT_DECAY)
    ap.add_argument('--warmup-epochs',    type=int,   default=cfg.WARMUP_EPOCHS)
    ap.add_argument('--img-size',         type=int,   default=cfg.IMG_SIZE)
    ap.add_argument('--rmi-net',          type=float, default=cfg.LAMBDA_RMI_NET)
    ap.add_argument('--rmi-en',           type=float, default=cfg.LAMBDA_RMI_EN)
    ap.add_argument('--rmi-cw',           type=float, default=cfg.LAMBDA_RMI_CW)
    ap.add_argument('--gap-close-px',     type=int,   default=cfg.GAP_CLOSE_PX,
                    help='closing radius for the network target; fills inter-object '
                         'gaps up to ~2*r px (0 disables gap handling)')
    ap.add_argument('--conn-radius',      type=int,   default=cfg.CONN_RADIUS_PX,
                    help='crosswalk-connectivity radius in px')
    ap.add_argument('--lambda-conn',      type=float, default=cfg.LAMBDA_CONN,
                    help='weight of the crosswalk-connectivity loss (0 disables)')
    ap.add_argument('--class-weights',    type=str,   default=None)
    ap.add_argument('--best-metric',      type=str,   default=cfg.BEST_METRIC,
                    choices=['iou', 'f1', 'tree_f1', 'ent_f1', 'cw_f1'])
    ap.add_argument('--no-pretrained',    action='store_true')
    ap.add_argument('--no-entrance-head', action='store_true')
    ap.add_argument('--no-crosswalk-head', action='store_true')
    ap.add_argument('--no-calibrate',     action='store_true',
                    help='skip post-training temperature calibration')
    ap.add_argument('--strong-aug',       action='store_true',  default=cfg.STRONG_AUG)
    ap.add_argument('--no-strong-aug',    dest='strong_aug',    action='store_false')
    ap.add_argument('--name',             type=str,   default='v4_unified')
    ap.add_argument('--resume',           type=str,   default=None)
    ap.add_argument('--ds-2023-only',     action='store_true')
    ap.add_argument('--ds-2025-only',     action='store_true')
    args = ap.parse_args()

    cfg.EPOCHS             = args.epochs
    cfg.BATCH_SIZE         = args.batch_size
    cfg.LR                 = args.lr
    cfg.LR_PRETRAINED      = args.lr_pretrained
    cfg.WEIGHT_DECAY       = args.weight_decay
    cfg.WARMUP_EPOCHS      = args.warmup_epochs
    cfg.IMG_SIZE           = args.img_size
    cfg.LAMBDA_RMI_NET     = args.rmi_net
    cfg.LAMBDA_RMI_EN      = args.rmi_en
    cfg.LAMBDA_RMI_CW      = args.rmi_cw
    cfg.GAP_CLOSE_PX       = args.gap_close_px
    cfg.CONN_RADIUS_PX     = args.conn_radius
    cfg.LAMBDA_CONN        = args.lambda_conn
    cfg.BEST_METRIC        = args.best_metric
    cfg.STRONG_AUG         = args.strong_aug
    cfg.USE_ENTRANCE_HEAD  = not args.no_entrance_head
    cfg.USE_CROSSWALK_HEAD = not args.no_crosswalk_head
    if args.no_pretrained:
        cfg.PRETRAINED_LOCAL = False
    if args.class_weights:
        cfg.CLASS_WEIGHTS = [float(x) for x in args.class_weights.split(',')]
        assert len(cfg.CLASS_WEIGHTS) == cfg.NUM_CLASSES
    elif cfg.COMBINED_STATS.exists():
        combined = json.loads(cfg.COMBINED_STATS.read_text())
        w = combined.get('combined_class_weights', cfg.CLASS_WEIGHTS)
        if len(w) == cfg.NUM_CLASSES:
            cfg.CLASS_WEIGHTS = [float(min(x, cfg.MAX_CLASS_WEIGHT)) for x in w]

    if args.resume:
        cfg.CKPT_DIR = Path(args.resume)
        assert cfg.CKPT_DIR.exists()
    else:
        cfg.CKPT_DIR = make_run_dir(cfg.RUNS_DIR, args.name)

    random.seed(42); np.random.seed(42)
    torch.manual_seed(42); torch.cuda.manual_seed_all(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger, _ = setup_logger(cfg.CKPT_DIR)
    save_run_config(cfg.CKPT_DIR, cfg, args)
    logger.info(f'Run folder: {cfg.CKPT_DIR}')
    logger.info(f'Device: {device}')
    logger.info(f'APPROACH: unified 5-class pedestrian network — '
                f'NETWORK_CLASSES={cfg.NETWORK_CLASSES}  '
                f'gap_close={cfg.GAP_CLOSE_PX}px  '
                f'conn(r={cfg.CONN_RADIUS_PX}px, λ={cfg.LAMBDA_CONN})')

    pairs_23 = load_dataset_pairs(cfg.DS_2023, cfg) if not args.ds_2025_only else {'train':[],'val':[],'test':[]}
    pairs_25 = load_dataset_pairs(cfg.DS_2025, cfg) if not args.ds_2023_only else {'train':[],'val':[],'test':[]}

    tf_train = build_transforms(cfg.IMG_SIZE, True, strong=cfg.STRONG_AUG)
    tf_val   = build_transforms(cfg.IMG_SIZE, False)

    train_ds = ConcatDataset([
        SidewalkDataset(pairs_23['train'], tf_train),
        SidewalkDataset(pairs_25['train'], tf_train),
    ])
    val_ds = ConcatDataset([
        SidewalkDataset(pairs_23['val'], tf_val),
        SidewalkDataset(pairs_25['val'], tf_val),
    ])

    logger.info(f'Train: {len(train_ds)} tiles | Val: {len(val_ds)} tiles')
    logger.info(f'Class weights: {[round(w, 2) for w in cfg.CLASS_WEIGHTS]} | '
                f'Entrance head: {cfg.USE_ENTRANCE_HEAD} | '
                f'Crosswalk head: {cfg.USE_CROSSWALK_HEAD}')

    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,
                              num_workers=cfg.NUM_WORKERS, pin_memory=True,
                              drop_last=True, persistent_workers=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.BATCH_SIZE, shuffle=False,
                              num_workers=cfg.NUM_WORKERS, pin_memory=True,
                              persistent_workers=True)

    model     = DBSwinT_v4(cfg).to(device)
    n_params  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'Model params: {n_params/1e6:.2f}M')
    criterion = CombinedLoss(cfg).to(device)
    groups, _, _ = build_param_groups(model, cfg)
    optimizer = AdamW(groups, weight_decay=cfg.WEIGHT_DECAY)

    warm = max(cfg.WARMUP_EPOCHS, 0)
    def lr_factor(epoch):
        if warm > 0 and epoch < warm:
            return (epoch + 1) / warm
        progress = (epoch - warm) / max(cfg.EPOCHS - warm, 1)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))
    scheduler = LambdaLR(optimizer, lr_lambda=lr_factor)

    start_epoch = 1
    best_score  = 0.0
    last_path   = cfg.CKPT_DIR / 'last.pt'
    if args.resume and last_path.exists():
        ck = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ck['model_state'])
        optimizer.load_state_dict(ck['optimizer_state'])
        scheduler.load_state_dict(ck['scheduler_state'])
        start_epoch = ck['epoch'] + 1
        best_score  = ck.get('best_score', 0.0)
        logger.info(f'Resumed from epoch {ck["epoch"]} (best {best_score:.4f})')

    history_path = cfg.CKPT_DIR / 'history.json'
    history = {
        'train_loss': [], 'val_loss': [], 'val_conn': [],
        'net_IoU': [], 'net_F1': [],   # pedestrian network (classes 1-4, gap-closed)
        'visible_recall': [], 'treeover_recall': [],
        'entrance_recall_cond': [], 'crosswalk_recall_cond': [],
        'ent_IoU': [], 'ent_F1': [], 'ent_recall': [],
        'cw_IoU': [], 'cw_F1': [], 'cw_recall': [],
    }
    if args.resume and history_path.exists():
        history = json.loads(history_path.read_text())

    logger.info(f'Starting v4 training: epochs {start_epoch}..{cfg.EPOCHS}')

    for epoch in range(start_epoch, cfg.EPOCHS + 1):
        t0 = time.time()
        tr = train_one_epoch(model, train_loader, criterion, optimizer, device)
        vl, vm = validate(model, val_loader, criterion, device, cfg)
        scheduler.step()
        dt = time.time() - t0

        history['train_loss'].append(tr['total'])
        history['val_loss'].append(vl['total'])
        history['val_conn'].append(vl['conn'])
        history['net_IoU'].append(vm['net_IoU'])
        history['net_F1'].append(vm['net_F1'])
        history['visible_recall'].append(vm['recall_1'])
        history['treeover_recall'].append(vm['recall_2'])
        history['entrance_recall_cond'].append(vm['recall_3'])
        history['crosswalk_recall_cond'].append(vm['recall_4'])
        history['ent_IoU'].append(vm['ent_IoU'])
        history['ent_F1'].append(vm['ent_F1'])
        history['ent_recall'].append(vm['ent_recall'])
        history['cw_IoU'].append(vm['cw_IoU'])
        history['cw_F1'].append(vm['cw_F1'])
        history['cw_recall'].append(vm['cw_recall'])
        history_path.write_text(json.dumps(history, indent=2))

        logger.info(
            f"[{epoch:3d}/{cfg.EPOCHS}] {dt:5.1f}s | "
            f"train {tr['total']:.4f} | val {vl['total']:.4f} | "
            f"NET_IoU {vm['net_IoU']:.4f} | NET_F1 {vm['net_F1']:.4f} | "
            f"vis_rec {vm['recall_1']:.4f} | tree_rec {vm['recall_2']:.4f} | "
            f"ent_rec {vm['recall_3']:.4f} | cw_rec {vm['recall_4']:.4f} | "
            f"cw_F1(aux) {vm['cw_F1']:.4f} | conn {vl['conn']:.4f}")

        torch.save({
            'epoch': epoch, 'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'best_score': best_score, 'val_metrics': vm,
        }, last_path)

        if cfg.BEST_METRIC == 'f1':
            score = vm['net_F1']
        elif cfg.BEST_METRIC == 'tree_f1':
            f1, tr_rec = vm['net_F1'], vm['recall_2']
            score = 2 * f1 * tr_rec / (f1 + tr_rec + 1e-9)
        elif cfg.BEST_METRIC == 'ent_f1':
            score = vm['ent_F1']
        elif cfg.BEST_METRIC == 'cw_f1':
            score = vm['cw_F1']
        else:
            score = vm['net_IoU']

        if score > best_score:
            best_score = score
            torch.save({
                'epoch': epoch, 'model_state': model.state_dict(),
                'val_metrics': vm, 'best_metric': cfg.BEST_METRIC,
            }, cfg.CKPT_DIR / 'best.pt')
            logger.info(f'  * new best {cfg.BEST_METRIC} {best_score:.4f} (saved best.pt)')

    logger.info(f'Done. Best {cfg.BEST_METRIC}: {best_score:.4f}')

    # ---- Temperature calibration of the BEST checkpoint on the val split ----
    if not args.no_calibrate:
        best_path = cfg.CKPT_DIR / 'best.pt'
        if best_path.exists():
            ck = torch.load(best_path, map_location=device, weights_only=False)
            model.load_state_dict(ck['model_state'])
            calibrate_run(cfg.CKPT_DIR, model, val_loader, criterion, device,
                          logger=logger)
        else:
            logger.info('No best.pt found — skipping calibration.')

    logger.info(f'Checkpoints in {cfg.CKPT_DIR}')
    logger.info('Confidence: p = sigmoid(logit / T) with T from temperature.json.')
    logger.info('Next: 4_evaluate.py (held-out metrics), then 5_export_weights.py '
                'to place the deployable weights in run/src/.')




if __name__ == '__main__':
    main()
