"""Dataset, augmentations, and split loading.

Extracted verbatim from train_v4_unified.py (Dataset section).
"""
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image


class SidewalkDataset(Dataset):
    def __init__(self, pairs, transform=None):
        self.pairs = pairs
        self.transform = transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        ip, mp = self.pairs[idx]
        img  = np.array(Image.open(ip).convert('RGB'))
        mask = np.array(Image.open(mp))
        if mask.ndim == 3:
            mask = mask[..., 0]
        mask = mask.astype(np.int64)
        if self.transform is not None:
            out  = self.transform(image=img, mask=mask)
            img, mask = out['image'], out['mask'].long()
        else:
            img  = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            mask = torch.from_numpy(mask).long()
        return img, mask


def build_transforms(img_size, train, strong=False):
    norm = A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    if not train:
        return A.Compose([A.CenterCrop(img_size, img_size), norm, ToTensorV2()])
    aug = [
        A.RandomCrop(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ColorJitter(0.2, 0.2, 0.2, 0.05, p=0.5),
    ]
    if strong:
        aug += [
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.15,
                               rotate_limit=15, border_mode=0, p=0.5),
            A.RandomBrightnessContrast(0.2, 0.2, p=0.4),
            A.GaussNoise(p=0.2),
        ]
    aug += [norm, ToTensorV2()]
    return A.Compose(aug)


def load_dataset_pairs(ds_root, cfg):
    splits_path = Path(ds_root) / 'splits.json'
    if not splits_path.exists():
        raise FileNotFoundError(f'splits.json not found in {ds_root}')
    splits = json.loads(splits_path.read_text())
    img_dir  = Path(ds_root) / 'images'
    mask_dir = Path(ds_root) / 'masks'
    result = {}
    for split, stems in splits.items():
        pairs = []
        for s in stems:
            ip = img_dir  / f'{s}{cfg.IMG_EXT}'
            mp = mask_dir / f'{s}{cfg.IMG_EXT}'
            if ip.exists() and mp.exists():
                pairs.append((ip, mp))
        result[split] = pairs
    return result

