"""DBSwinT_v4 — dual-branch Swin Transformer with AFF fusion, U-Net decoder,
and four heads (5-class condition-aware, binary network, entrance aux,
crosswalk aux).

Extracted verbatim from train_v4_unified.py (Model section).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


def _gn(c):
    for g in (8, 4, 2, 1):
        if c % g == 0:
            return nn.GroupNorm(g, c)
    return nn.GroupNorm(1, c)


class MSCAM(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        inter = max(channels // reduction, 8)
        self.local_att  = nn.Sequential(
            nn.Conv2d(channels, inter, 1, bias=False), _gn(inter), nn.ReLU(True),
            nn.Conv2d(inter, channels, 1, bias=False), _gn(channels))
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter, 1, bias=False), _gn(inter), nn.ReLU(True),
            nn.Conv2d(inter, channels, 1, bias=False), _gn(channels))

    def forward(self, x):
        return torch.sigmoid(self.local_att(x) + self.global_att(x))


class AFF(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.mscam = MSCAM(channels, reduction)

    def forward(self, x, y):
        if y.shape[-2:] != x.shape[-2:]:
            y = F.interpolate(y, size=x.shape[-2:], mode='bilinear', align_corners=False)
        w = self.mscam(x + y)
        return w * x + (1 - w) * y


class SwinBranch(nn.Module):
    def __init__(self, img_size, patch_size, embed_dim, depths, num_heads,
                 window_size, in_chans=3, pretrained=False,
                 pretrained_model='swin_tiny_patch4_window7_224'):
        super().__init__()
        self.swin = timm.create_model(
            pretrained_model, pretrained=pretrained, features_only=False,
            img_size=img_size, patch_size=patch_size, embed_dim=embed_dim,
            depths=depths, num_heads=num_heads, window_size=window_size,
            in_chans=in_chans, num_classes=0, global_pool='')

    def forward(self, x):
        x = self.swin.patch_embed(x)
        if hasattr(self.swin, 'pos_drop'):
            x = self.swin.pos_drop(x)
        feats = []
        for layer in self.swin.layers:
            x = layer(x)
            feats.append(x)
        out = []
        for f in feats:
            if f.dim() == 4:
                f = f.permute(0, 3, 1, 2).contiguous()
            elif f.dim() == 3:
                B, L, C = f.shape
                s = int(math.sqrt(L))
                f = f.transpose(1, 2).reshape(B, C, s, s)
            out.append(f)
        return out


class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1, bias=False),
            _gn(out_ch), nn.ReLU(True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            _gn(out_ch), nn.ReLU(True))

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class DBSwinT_v4(nn.Module):
    """
    v3 architecture with one extra 1x1 head:
      head_cond      — 5-class conditional head (bg/vis_sw/tree/entrance/crosswalk)
      head_network   — binary pedestrian network (classes 1+2+3+4, gap-closed target)
      head_entrance  — entrance-only auxiliary
      head_crosswalk — crosswalk-only auxiliary (new)
    """
    def __init__(self, cfg):
        super().__init__()
        C = cfg.EMBED_DIM
        self.local_branch  = SwinBranch(cfg.IMG_SIZE, cfg.LOCAL_PATCH,  C, cfg.DEPTHS,
                                        cfg.NUM_HEADS, cfg.WINDOW_SIZE, cfg.IN_CHANNELS,
                                        pretrained=cfg.PRETRAINED_LOCAL,
                                        pretrained_model=cfg.PRETRAINED_MODEL)
        self.global_branch = SwinBranch(cfg.IMG_SIZE, cfg.GLOBAL_PATCH, C, cfg.DEPTHS,
                                        cfg.NUM_HEADS, cfg.WINDOW_SIZE, cfg.IN_CHANNELS,
                                        pretrained=False,
                                        pretrained_model=cfg.PRETRAINED_MODEL)
        stage_ch = [C, 2*C, 4*C, 8*C]
        self.affs    = nn.ModuleList([AFF(c) for c in stage_ch])
        self.up3     = UpBlock(8*C, 4*C, 4*C)
        self.up2     = UpBlock(4*C, 2*C, 2*C)
        self.up1     = UpBlock(2*C, C,   C)
        self.final_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(C, C//2, 3, padding=1, bias=False), _gn(C//2), nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(C//2, C//2, 3, padding=1, bias=False), _gn(C//2), nn.ReLU(True))
        self.head_cond      = nn.Conv2d(C//2, cfg.NUM_CLASSES, 1)
        self.head_network   = nn.Conv2d(C//2, 1, 1)   # pedestrian network (1+2+3+4)
        self.head_entrance  = nn.Conv2d(C//2, 1, 1)   # entrance-only auxiliary
        self.head_crosswalk = nn.Conv2d(C//2, 1, 1)   # crosswalk-only auxiliary
        self.use_entrance_head  = cfg.USE_ENTRANCE_HEAD
        self.use_crosswalk_head = cfg.USE_CROSSWALK_HEAD

    def forward(self, x):
        in_size = x.shape[-2:]
        lf = self.local_branch(x)
        gf = self.global_branch(x)
        fused = [aff(l, g) for aff, l, g in zip(self.affs, lf, gf)]
        f1, f2, f3, f4 = fused
        d3 = self.up3(f4, f3)
        d2 = self.up2(d3, f2)
        d1 = self.up1(d2, f1)
        d0 = self.final_up(d1)
        if d0.shape[-2:] != in_size:
            d0 = F.interpolate(d0, size=in_size, mode='bilinear', align_corners=False)
        return (self.head_cond(d0), self.head_network(d0),
                self.head_entrance(d0), self.head_crosswalk(d0))


