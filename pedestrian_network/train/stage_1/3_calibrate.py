"""Step 3 — (re)fit the confidence temperatures of a finished run.

OBJECTIVE
  Make the model's confidence a calibrated probability. One temperature T per
  binary head (network, entrance, crosswalk) is fitted on the validation split
  so that p = sigmoid(logit / T) matches observed frequencies. 2_train.py
  already does this at the end of training; run this step only after an
  interrupted training, a changed validation split, or a replaced best.pt.

INPUT
  train/output/runs/<run>/: best.pt + config.json (newest run, or --run)
  validation split of dc_2023 + dc_2025 (paths from config.py)

OUTPUT
  train/output/runs/<run>/temperature.json   {"T_network", "T_entrance",
                                              "T_crosswalk"}

HOW IT WORKS (functions)
  resolve_run() / newest_run()   pick the run folder
  load_run()                     rebuild DBSwinT_v4 from config.json and load
                                 best.pt (no pretrained download)
  build_split_loader('val')      centre-crop validation loader
  calibration.calibrate_run()    collect logits and targets over the val split
                                 (subsampled per batch), fit each T with LBFGS
                                 on the binary NLL, write temperature.json;
                                 a head with no positives keeps T = 1

TUNING
  --run <name|path>   run folder (default: newest). No other knobs — the
  temperatures are data-driven. T > 1 means the raw logits were over-confident
  (released model: T_network 1.80, T_entrance 1.67, T_crosswalk 1.51).
  Everything downstream (stage 1 confidence rasters, stage 2 --thr) assumes
  temperature.json travels with best.pt (5_export_weights.py copies it).

USAGE
  python 3_calibrate.py
  python 3_calibrate.py --run 20260903_110705_v4_unified
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
from calibration import calibrate_run


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
    ap.add_argument('--run', type=str, default=None,
                    help='run folder (name under train/output/runs/ or absolute); default: newest')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    run_dir = resolve_run(args.run)
    print(f'run: {run_dir} | device: {device}')

    model, _ = load_run(run_dir, device)
    criterion = CombinedLoss(cfg).to(device)
    val_loader = build_split_loader('val')
    temps = calibrate_run(run_dir, model, val_loader, criterion, device)
    print(f'temperatures: {temps} -> {run_dir / "temperature.json"}')


if __name__ == '__main__':
    main()
