"""Run-folder, logging, and optimizer parameter-group helpers.

Extracted verbatim from train_v4_unified.py (Utilities section).
"""
import sys
import json
import time
import logging
from pathlib import Path


def make_run_dir(runs_dir, name=None):
    ts      = time.strftime('%Y%m%d_%H%M%S')
    folder  = ts if not name else f'{ts}_{name}'
    run_dir = Path(runs_dir) / folder
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_run_config(run_dir, cfg, args):
    snap = {k: (str(v) if isinstance(v, Path) else v)
            for k, v in vars(cfg.__class__).items()
            if not k.startswith('__') and not callable(v)}
    snap['CKPT_DIR'] = str(cfg.CKPT_DIR)
    snap['cli_args'] = vars(args)
    (run_dir / 'config.json').write_text(json.dumps(snap, indent=2, default=str))


def setup_logger(run_dir):
    log_path = run_dir / 'train.log'
    logger   = logging.getLogger('train')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter('%(asctime)s  %(message)s', '%H:%M:%S')
    fh  = logging.FileHandler(log_path); fh.setFormatter(fmt); logger.addHandler(fh)
    sh  = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt); logger.addHandler(sh)
    return logger, log_path


def build_param_groups(model, cfg):
    pretrained_params, scratch_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith('local_branch.'):
            pretrained_params.append(p)
        else:
            scratch_params.append(p)
    return [
        {'params': scratch_params,    'lr': cfg.LR},
        {'params': pretrained_params, 'lr': cfg.LR_PRETRAINED},
    ], len(pretrained_params), len(scratch_params)


