"""Step 4 — evaluate a finished run on a held-out split.

OBJECTIVE
  Report the segmentation quality of a checkpoint on tiles the model did not
  train on, with the same validate() used during training: network IoU / F1 /
  recall (gap-closed union of classes 1-4), per-class recalls of the 5-class
  head, and the auxiliary entrance / crosswalk head metrics.

INPUT
  train/output/runs/<run>/: best.pt + config.json (newest run, or --run)
  --split test (default) | val | train, from dc_2023 + dc_2025

OUTPUT
  train/output/runs/<run>/<split>_metrics.json   net_IoU, net_F1, net_recall,
      IoU_0..4, recall_0..4, ent_IoU / ent_F1 / ent_recall, cw_IoU / cw_F1 /
      cw_recall — also printed to the terminal

HOW IT WORKS (functions)
  resolve_run() / newest_run(), load_run()   as in 3_calibrate.py
  build_split_loader(split)                  centre-crop loader (IMG_SIZE)
  engine.validate()                          forward pass + losses.SegMetrics
                                             accumulated over the split

TUNING / CAVEATS
  Metrics are computed on IMG_SIZE (512 px) centre crops of the 1024-px tiles,
  matching training-time validation — not on full tiles. Full-tile,
  sliding-window quality is what stage 2 consumes; compare networks there.
  Class weights from combined_dataset_stats.json are loaded so the reported
  loss matches training. The confidence heads are scored at 0.5, the 5-class
  head by argmax; no thresholds to tune here.

USAGE
  python 4_evaluate.py                     # newest run, test split
  python 4_evaluate.py --split val --run <name>
"""
import json
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, ConcatDataset

from config import cfg
from dataset import SidewalkDataset, build_transforms, load_dataset_pairs
from model import DBSwinT_v4
from losses import CombinedLoss
from engine import validate


def newest_run(runs_dir):
    cands = sorted(d for d in Path(runs_dir).glob('*') if (d / 'best.pt').exists())
    assert cands, f'no finished runs under {runs_dir}'
    return cands[-1]


def resolve_run(run_arg):
    if run_arg is None:
        return newest_run(cfg.RUNS_DIR)
    run_dir = Path(run_arg)
    if not run_dir.is_absolute():
        run_dir = cfg.RUNS_DIR / run_arg
    assert (run_dir / 'best.pt').exists(), f'{run_dir} has no best.pt'
    return run_dir


def load_run(run_dir, device):
    """Rebuild the model exactly as trained (config.json) and load best.pt."""
    run_cfg = json.loads((run_dir / 'config.json').read_text())
    cfg.IMG_SIZE = int(run_cfg.get('IMG_SIZE', cfg.IMG_SIZE))
    cfg.PRETRAINED_LOCAL = False          # weights come from the checkpoint
    if cfg.COMBINED_STATS.exists():
        w = json.loads(cfg.COMBINED_STATS.read_text()).get(
            'combined_class_weights', cfg.CLASS_WEIGHTS)
        if len(w) == cfg.NUM_CLASSES:
            cfg.CLASS_WEIGHTS = [float(min(x, cfg.MAX_CLASS_WEIGHT)) for x in w]
    model = DBSwinT_v4(cfg).to(device)
    ck = torch.load(run_dir / 'best.pt', map_location=device, weights_only=False)
    model.load_state_dict(ck['model_state'])
    model.eval()
    return model, ck


def build_split_loader(split, batch_size=None):
    tf = build_transforms(cfg.IMG_SIZE, train=False)
    parts = []
    for ds_root in (cfg.DS_2023, cfg.DS_2025):
        if (Path(ds_root) / 'splits.json').exists():
            pairs = load_dataset_pairs(ds_root, cfg)
            parts.append(SidewalkDataset(pairs[split], tf))
    ds = ConcatDataset(parts)
    return DataLoader(ds, batch_size=batch_size or cfg.BATCH_SIZE, shuffle=False,
                      num_workers=cfg.NUM_WORKERS, pin_memory=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run',   type=str, default=None,
                    help='run folder (name under train/output/runs/ or absolute); default: newest')
    ap.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'])
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    run_dir = resolve_run(args.run)
    print(f'run: {run_dir} | split: {args.split} | device: {device}')

    model, ck = load_run(run_dir, device)
    print(f'best checkpoint from epoch {ck.get("epoch", "?")} '
          f'(best_metric={ck.get("best_metric", "?")})')
    criterion = CombinedLoss(cfg).to(device)
    loader = build_split_loader(args.split)
    print(f'{args.split}: {len(loader.dataset)} tiles')

    losses, metrics = validate(model, loader, criterion, device, cfg)
    report = {'run': str(run_dir), 'split': args.split,
              'n_tiles': len(loader.dataset),
              'losses': {k: round(float(v), 5) for k, v in losses.items()},
              'metrics': {k: round(float(v), 5) for k, v in metrics.items()}}
    (run_dir / f'{args.split}_metrics.json').write_text(json.dumps(report, indent=2))

    print(json.dumps(report['metrics'], indent=2))
    print(f'saved -> {run_dir / f"{args.split}_metrics.json"}')


if __name__ == '__main__':
    main()
