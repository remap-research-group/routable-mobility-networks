"""Train / validate loops. Extracted verbatim from train_v4_unified.py."""
import numpy as np
import torch

from losses import SegMetrics


LOSS_KEYS = ('ce', 'dice_net', 'rmi_net', 'dice_en', 'rmi_en',
             'dice_cw', 'rmi_cw', 'conn', 'total')


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    run = {k: 0.0 for k in LOSS_KEYS}
    n = 0
    for img, mask in loader:
        img, mask = img.to(device, non_blocking=True), mask.to(device, non_blocking=True)
        optimizer.zero_grad()
        lc, ln, le, lw = model(img)
        loss, parts = criterion(lc, ln, le, lw, mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        for k in run:
            run[k] += parts[k] * img.size(0)
        n += img.size(0)
    return {k: run[k] / n for k in run}


@torch.no_grad()
def validate(model, loader, criterion, device, cfg):
    model.eval()
    metrics = SegMetrics(cfg)
    run = {k: 0.0 for k in LOSS_KEYS}
    n = 0
    for img, mask in loader:
        img, mask = img.to(device, non_blocking=True), mask.to(device, non_blocking=True)
        lc, ln, le, lw = model(img)
        loss, parts = criterion(lc, ln, le, lw, mask)
        for k in run:
            run[k] += parts[k] * img.size(0)
        n += img.size(0)

        ce_t, net_t, en_t, cw_t = criterion.build_targets(mask)
        pred_cond = lc.argmax(1).cpu().numpy().ravel()
        pred_net  = (torch.sigmoid(ln[:, 0]) > 0.5).cpu().numpy().ravel().astype(np.int64)
        pred_en   = (torch.sigmoid(le[:, 0]) > 0.5).cpu().numpy().ravel().astype(np.int64)
        pred_cw   = (torch.sigmoid(lw[:, 0]) > 0.5).cpu().numpy().ravel().astype(np.int64)
        metrics.update(pred_cond, pred_net, pred_en, pred_cw,
                       ce_t.cpu().numpy().ravel(),
                       net_t.cpu().numpy().ravel().astype(np.int64),
                       en_t.cpu().numpy().ravel().astype(np.int64),
                       cw_t.cpu().numpy().ravel().astype(np.int64))
    return {k: run[k] / n for k in run}, metrics.compute()


