"""Post-training temperature calibration of the confidence heads.

Extracted verbatim from train_v4_unified.py. After training, per-head
temperatures are fitted on the validation split with LBFGS and written to
<run>/temperature.json. Downstream confidence is p = sigmoid(logit / T_head).
"""
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


def fit_temperature(logits_list, target_list):
    """LBFGS fit of a single temperature T minimizing BCE(logits/T, targets)."""
    logits  = torch.tensor(np.concatenate(logits_list))
    targets = torch.tensor(np.concatenate(target_list))
    logT    = torch.zeros(1, requires_grad=True)
    opt     = torch.optim.LBFGS([logT], lr=0.1, max_iter=60)
    bce     = nn.BCEWithLogitsLoss()
    def closure():
        opt.zero_grad()
        loss = bce(logits / torch.exp(logT), targets)
        loss.backward()
        return loss
    opt.step(closure)
    # LBFGS can diverge on degenerate logits; T=0 would mean division by zero
    # downstream, so clamp to a sane range (well-trained heads land near 1-3).
    T = float(torch.exp(logT).clamp(0.05, 20.0).item())
    return T if math.isfinite(T) else 1.0


@torch.no_grad()
def calibrate_run(run_dir, model, val_loader, criterion, device,
                  max_px_per_batch=40_000, logger=None):
    """Fit per-head temperatures on the val split -> <run>/temperature.json.

    Confidence downstream:  p = sigmoid(logit / T_<head>).
    The network-head targets are the same gap-CLOSED targets used in training,
    so the calibrated confidence refers to the connected network.
    """
    rng  = np.random.default_rng(42)
    data = {k: ([], []) for k in ('net', 'en', 'cw')}
    model.eval()
    for img, mask in val_loader:
        img, mask = img.to(device), mask.to(device)
        lc, ln, le, lw = model(img)
        _, net_t, en_t, cw_t = criterion.build_targets(mask)
        for key, lg, tg in (('net', ln[:, 0], net_t),
                            ('en',  le[:, 0], en_t),
                            ('cw',  lw[:, 0], cw_t)):
            l = lg.reshape(-1).float().cpu().numpy()
            t = tg.reshape(-1).float().cpu().numpy()
            if l.size > max_px_per_batch:
                idx = rng.choice(l.size, max_px_per_batch, replace=False)
                l, t = l[idx], t[idx]
            data[key][0].append(l)
            data[key][1].append(t)

    temps = {}
    for key, name in (('net', 'T_network'), ('en', 'T_entrance'), ('cw', 'T_crosswalk')):
        lg, tg = data[key]
        if float(np.concatenate(tg).sum()) == 0.0:
            temps[name] = 1.0     # no positives in val -> leave uncalibrated
        else:
            temps[name] = fit_temperature(lg, tg)
    (Path(run_dir) / 'temperature.json').write_text(json.dumps(temps, indent=2))
    if logger:
        logger.info(f'Calibration: {temps} -> saved temperature.json')
    return temps


def load_temperatures(run_dir):
    """Read <run>/temperature.json; returns all-1.0 defaults if absent."""
    p = Path(run_dir) / 'temperature.json'
    if p.exists():
        return json.loads(p.read_text())
    return {'T_network': 1.0, 'T_entrance': 1.0, 'T_crosswalk': 1.0}


