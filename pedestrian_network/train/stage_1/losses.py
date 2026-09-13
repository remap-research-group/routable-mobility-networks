"""Gap-tolerant target construction, losses (CE + Dice + RMI + crosswalk
connectivity), and segmentation metrics.

Extracted verbatim from train_v4_unified.py (target + loss + metric sections).
Key ideas (see README):
  * build_targets() morphologically CLOSES the binary network target on the GPU
    (radius cfg.GAP_CLOSE_PX) and CE-ignores the filled gap band.
  * CombinedLoss adds a differentiable crosswalk-connectivity penalty
    (crosswalk probability mass farther than cfg.CONN_RADIUS_PX from predicted
    walkable surface).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def morph_close_gpu(t, radius):
    """Binary closing of a (B,H,W) float tensor with a (2r+1) square kernel.

    max_pool2d pads with -inf, so the dilation never grows in from outside the
    tile and the erosion never eats objects touching the border — exactly the
    behaviour we want for tile-wise targets.
    """
    if radius <= 0:
        return t
    k = 2 * radius + 1
    x = t.unsqueeze(1)
    x = F.max_pool2d(x, k, stride=1, padding=radius)     # dilation
    x = -F.max_pool2d(-x, k, stride=1, padding=radius)   # erosion
    return x.squeeze(1)


def build_targets(mask, net_classes, en_classes, cw_classes, gap_px, ignore_index):
    """From an integer label mask (B,H,W) build all training targets.

    Returns:
      ce_t       — 5-class CE target; the gap band filled by the closing is set
                   to ignore_index so the conditional head is not forced to
                   reproduce the 2-3 px background slivers between objects
      net_t      — gap-CLOSED binary pedestrian-network target (classes 1..4)
      en_t, cw_t — raw binary auxiliary targets (not closed)
    """
    net_t = torch.zeros_like(mask, dtype=torch.float)
    for c in net_classes:
        net_t = torch.maximum(net_t, (mask == c).float())
    net_t = (morph_close_gpu(net_t, gap_px) > 0.5).float()

    gap_band = (net_t > 0.5) & (mask == 0)
    ce_t = mask.clone()
    ce_t[gap_band] = ignore_index

    en_t = torch.zeros_like(mask, dtype=torch.float)
    for c in en_classes:
        en_t = torch.maximum(en_t, (mask == c).float())
    cw_t = torch.zeros_like(mask, dtype=torch.float)
    for c in cw_classes:
        cw_t = torch.maximum(cw_t, (mask == c).float())
    return ce_t, net_t, en_t, cw_t


# =============================================================================
# Losses & Metrics
# =============================================================================
class DiceLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logits, target):
        if logits.dim() == 4 and logits.shape[1] == 1:
            logits = logits[:, 0]
        probs  = torch.sigmoid(logits)
        target = target.float()
        inter  = (probs * target).sum((1, 2))
        union  = probs.sum((1, 2)) + target.sum((1, 2))
        return 1 - ((2 * inter + self.eps) / (union + self.eps)).mean()


class RMILoss(nn.Module):
    def __init__(self, radius=3, stride=4, eps=1e-5):
        super().__init__()
        self.radius = radius
        self.stride = stride
        self.eps    = eps

    def _patches(self, x):
        return F.unfold(x, kernel_size=self.radius, stride=self.stride)

    def forward(self, logits, target):
        if logits.dim() == 3: logits = logits.unsqueeze(1)
        if target.dim() == 3: target = target.unsqueeze(1)
        probs  = torch.sigmoid(logits)
        target = target.float()
        bce    = F.binary_cross_entropy_with_logits(logits, target)
        p = self._patches(probs)
        t = self._patches(target)
        p = p - p.mean(2, keepdim=True)
        t = t - t.mean(2, keepdim=True)
        n = p.shape[2]
        cov_pp = (p @ p.transpose(1, 2)) / (n - 1 + self.eps)
        cov_tt = (t @ t.transpose(1, 2)) / (n - 1 + self.eps)
        cov_pt = (p @ t.transpose(1, 2)) / (n - 1 + self.eps)
        I = torch.eye(p.shape[1], device=p.device).unsqueeze(0)
        cov_pp = cov_pp + self.eps * I
        cov_tt = cov_tt + self.eps * I
        try:
            inv_tt = torch.linalg.inv(cov_tt)
            cond   = cov_pp - cov_pt @ inv_tt @ cov_pt.transpose(1, 2)
            ld_pp  = torch.logdet(cov_pp.clamp_min(self.eps))
            ld_c   = torch.logdet(cond.clamp_min(self.eps) + self.eps * I)
            rmi_loss = -0.5 * (ld_pp - ld_c).mean() / p.shape[1]
        except Exception:
            rmi_loss = torch.tensor(0.0, device=logits.device)
        return bce + rmi_loss


class CombinedLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        wt   = torch.tensor(cfg.CLASS_WEIGHTS, dtype=torch.float)
        self.ce   = nn.CrossEntropyLoss(weight=wt, ignore_index=cfg.IGNORE_INDEX)
        self.dice = DiceLoss()
        self.rmi  = RMILoss()
        self.l_ce       = cfg.LAMBDA_CE
        self.l_dice_net = cfg.LAMBDA_DICE_NET
        self.l_rmi_net  = cfg.LAMBDA_RMI_NET
        self.l_dice_en  = cfg.LAMBDA_DICE_EN
        self.l_rmi_en   = cfg.LAMBDA_RMI_EN
        self.l_dice_cw  = cfg.LAMBDA_DICE_CW
        self.l_rmi_cw   = cfg.LAMBDA_RMI_CW
        self.l_conn     = cfg.LAMBDA_CONN
        self.net = cfg.NETWORK_CLASSES
        self.en  = cfg.ENTRANCE_CLASSES
        self.cw  = cfg.CROSSWALK_CLASSES
        self.gap_px       = cfg.GAP_CLOSE_PX
        self.conn_radius  = cfg.CONN_RADIUS_PX
        self.ignore_index = cfg.IGNORE_INDEX
        self.use_en_head  = cfg.USE_ENTRANCE_HEAD
        self.use_cw_head  = cfg.USE_CROSSWALK_HEAD

    def build_targets(self, mask):
        return build_targets(mask, self.net, self.en, self.cw,
                             self.gap_px, self.ignore_index)

    def connectivity_loss(self, logits_cond):
        """Fraction of predicted crosswalk probability mass that lies farther
        than conn_radius px from any predicted walkable pixel (classes 1+2+3).

        Gradients flow both ways: crosswalk mass is pulled toward walkable
        surface AND walkable probability is encouraged in the gap between a
        crosswalk and its sidewalk (that band is CE-ignored, so nothing
        contradicts it)."""
        probs  = F.softmax(logits_cond, dim=1)
        p_walk = probs[:, 1] + probs[:, 2] + probs[:, 3]
        p_cw   = probs[:, 4]
        k = 2 * self.conn_radius + 1
        walk_near = F.max_pool2d(p_walk.unsqueeze(1), k, stride=1,
                                 padding=self.conn_radius).squeeze(1)
        stranded = (p_cw * (1.0 - walk_near)).sum((1, 2))
        mass     = p_cw.sum((1, 2))
        return (stranded / (mass + 1e-6)).mean()

    def forward(self, logits_cond, logits_net, logits_en, logits_cw, mask):
        ce_t, net_t, en_t, cw_t = self.build_targets(mask)

        l_ce       = self.ce(logits_cond, ce_t)
        l_dice_net = self.dice(logits_net, net_t)
        l_rmi_net  = self.rmi(logits_net,  net_t)
        total = (self.l_ce * l_ce + self.l_dice_net * l_dice_net
                 + self.l_rmi_net * l_rmi_net)

        zero = torch.tensor(0.0, device=logits_cond.device)
        l_dice_en = l_rmi_en = l_dice_cw = l_rmi_cw = l_conn = zero
        if self.use_en_head:
            l_dice_en = self.dice(logits_en, en_t)
            l_rmi_en  = self.rmi(logits_en,  en_t)
            total = total + self.l_dice_en * l_dice_en + self.l_rmi_en * l_rmi_en
        if self.use_cw_head:
            l_dice_cw = self.dice(logits_cw, cw_t)
            l_rmi_cw  = self.rmi(logits_cw,  cw_t)
            total = total + self.l_dice_cw * l_dice_cw + self.l_rmi_cw * l_rmi_cw
        if self.l_conn > 0:
            l_conn = self.connectivity_loss(logits_cond)
            total  = total + self.l_conn * l_conn

        return total, {
            'ce': l_ce.item(), 'dice_net': l_dice_net.item(), 'rmi_net': l_rmi_net.item(),
            'dice_en': l_dice_en.item(), 'rmi_en': l_rmi_en.item(),
            'dice_cw': l_dice_cw.item(), 'rmi_cw': l_rmi_cw.item(),
            'conn': l_conn.item(), 'total': total.item(),
        }


class SegMetrics:
    def __init__(self, cfg):
        self.nc = cfg.NUM_CLASSES
        self.reset()

    def reset(self):
        self.cm = np.zeros((self.nc, self.nc), np.int64)
        self.net_tp = self.net_fp = self.net_fn = 0
        self.ent_tp = self.ent_fp = self.ent_fn = 0
        self.cw_tp  = self.cw_fp  = self.cw_fn  = 0

    def update(self, pred_cond, pred_net, pred_en, pred_cw,
               ce_target, net_target, en_target, cw_target):
        # ce_target has IGNORE_INDEX in the gap band -> excluded here
        m   = (ce_target >= 0) & (ce_target < self.nc)
        idx = self.nc * ce_target[m] + pred_cond[m]
        self.cm += np.bincount(idx, minlength=self.nc**2).reshape(self.nc, self.nc)

        for pred, tgt, pref in ((pred_net, net_target, 'net'),
                                (pred_en,  en_target,  'ent'),
                                (pred_cw,  cw_target,  'cw')):
            tp = int(((pred == 1) & (tgt == 1)).sum())
            fp = int(((pred == 1) & (tgt == 0)).sum())
            fn = int(((pred == 0) & (tgt == 1)).sum())
            setattr(self, f'{pref}_tp', getattr(self, f'{pref}_tp') + tp)
            setattr(self, f'{pref}_fp', getattr(self, f'{pref}_fp') + fp)
            setattr(self, f'{pref}_fn', getattr(self, f'{pref}_fn') + fn)

    def compute(self):
        eps = 1e-9
        out = {}
        for c in range(self.nc):
            tp = self.cm[c, c]
            fn = self.cm[c, :].sum() - tp
            fp = self.cm[:, c].sum() - tp
            out[f'IoU_{c}']    = tp / (tp + fp + fn + eps)
            out[f'recall_{c}'] = tp / (tp + fn + eps)
        for pref in ('net', 'ent', 'cw'):
            tp = getattr(self, f'{pref}_tp')
            fp = getattr(self, f'{pref}_fp')
            fn = getattr(self, f'{pref}_fn')
            out[f'{pref}_IoU']    = tp / (tp + fp + fn + eps)
            out[f'{pref}_recall'] = tp / (tp + fn + eps)
            out[f'{pref}_F1']     = 2 * tp / (2 * tp + fp + fn + eps)
        return out


