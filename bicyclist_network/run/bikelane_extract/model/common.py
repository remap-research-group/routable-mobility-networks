"""Shared pieces for seg train / predict."""
from __future__ import annotations

import numpy as np

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)
IGNORE = 255
CLASS_NAMES = {0: "background", 1: "lane", 2: "curb", 3: "virtual"}


def make_model(encoder="resnet34", num_classes=4, pretrained=False):
    import segmentation_models_pytorch as smp
    return smp.Unet(encoder_name=encoder, encoder_weights="imagenet" if pretrained else None,
                    in_channels=3, classes=num_classes)


def load_checkpoint(model, path, device):
    """Accepts {'model': state_dict, 'epoch', 'miou'} or a bare state_dict."""
    import torch
    ck = torch.load(path, map_location=device, weights_only=False)
    sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    model.load_state_dict(sd)
    meta = {k: ck[k] for k in ("epoch", "miou") if isinstance(ck, dict) and k in ck}
    return meta


def normalize(img_uint8_rgb):
    img = img_uint8_rgb.astype(np.float32) / 255.0
    return ((img - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1)
