"""Load the deployable DBSwinT_v4 weights kept in this folder (run/src/).

    best.pt              checkpoint: {'model_state', 'epoch', 'val_metrics', ...}
    temperature.json     per-head calibration temperatures (T_network,
                         T_crosswalk, T_entrance)
    train_config.json    architecture / input snapshot of the training run
    model_card.json      classes, normalization, confidence formula
    model.py             the DBSwinT_v4 architecture (identical copy of
                         train/stage_1/model.py)

Typical use (see run/stage_1/segmentation.py):

    from loader import load_model, sigmoid
    model, cfg, temps = load_model()               # weights from run/src/
    cond, net, en, cw = model(x)                   # x: [B,3,512,512] normalized
    p_net = sigmoid(net / temps['T_network'])      # calibrated confidence
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

# ImageNet normalization used at training time
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD  = np.array([0.229, 0.224, 0.225], np.float32)

CLASS_NAMES = ['background', 'visible_sidewalk', 'tree', 'entrance', 'crosswalk']
PALETTE = {0: (0, 0, 0), 1: (255, 140, 0), 2: (34, 139, 34),
           3: (70, 130, 180), 4: (220, 20, 60)}          # class id -> RGB (overlays)


class ModelConfig:
    """The subset of the training Config that model.py needs. Defaults match
    train/stage_1/config.py; train_config.json overrides them."""
    NUM_CLASSES        = 5
    IN_CHANNELS        = 3
    IMG_SIZE           = 512
    EMBED_DIM          = 96
    DEPTHS             = [2, 2, 6, 2]
    NUM_HEADS          = [3, 6, 12, 24]
    WINDOW_SIZE        = 7
    LOCAL_PATCH        = 4
    GLOBAL_PATCH       = 8
    PRETRAINED_LOCAL   = False           # weights come from best.pt, never downloaded
    PRETRAINED_MODEL   = 'swin_tiny_patch4_window7_224'
    USE_ENTRANCE_HEAD  = True
    USE_CROSSWALK_HEAD = True

    def __init__(self, overrides=None):
        for k, v in (overrides or {}).items():
            if hasattr(self, k) and k != 'PRETRAINED_LOCAL':
                setattr(self, k, v)


def resolve_weights(weights_arg=None):
    """Folder holding best.pt. Default: this folder (run/src/)."""
    d = Path(weights_arg) if weights_arg else HERE
    if not d.is_absolute():
        d = (Path.cwd() / d).resolve()
    assert (d / 'best.pt').exists(), (
        f'{d} has no best.pt — run `python download.py` in the pedestrian_network '
        f'folder to fetch the released weights, or export a training run with '
        f'train/stage_1/5_export_weights.py')
    return d


def load_temperatures(weights_dir):
    p = Path(weights_dir) / 'temperature.json'
    if p.exists():
        return json.loads(p.read_text())
    print(f'WARN: {p} missing — confidence will be uncalibrated (T = 1)')
    return {'T_network': 1.0, 'T_entrance': 1.0, 'T_crosswalk': 1.0}


def load_model(weights_dir=None, device=None, verbose=True):
    """-> (model in eval mode on `device`, ModelConfig, temperatures dict).
    verbose: print a short model report (weights, source run, validation
    metrics of the checkpoint, classes, input contract, temperatures)."""
    import torch
    from model import DBSwinT_v4
    weights_dir = resolve_weights(weights_dir)
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # train_config.json (exported by 5_export_weights.py) or config.json (a raw
    # training run folder under train/output/runs/) — both carry the same keys
    overrides = {}
    for name in ('train_config.json', 'config.json'):
        if (weights_dir / name).exists():
            overrides = json.loads((weights_dir / name).read_text())
            break
    else:
        print(f'WARN: no train_config.json / config.json in {weights_dir} — '
              f'using the default architecture')
    cfg = ModelConfig(overrides)
    model = DBSwinT_v4(cfg).to(device)
    ck = torch.load(weights_dir / 'best.pt', map_location=device, weights_only=False)
    model.load_state_dict(ck['model_state'])
    model.eval()
    temps = load_temperatures(weights_dir)
    if verbose:
        describe_model(model, cfg, temps, weights_dir, ck, device)
    return model, cfg, temps


def describe_model(model, cfg, temps, weights_dir, ck, device):
    """Print what was loaded — the terminal report shown before inference."""
    import torch
    card = {}
    if (weights_dir / 'model_card.json').exists():
        card = json.loads((weights_dir / 'model_card.json').read_text())
    n_params = sum(p.numel() for p in model.parameters())
    size_mb = (weights_dir / 'best.pt').stat().st_size / 2**20
    vm = ck.get('val_metrics', {}) or {}
    dev = str(device)
    if dev.startswith('cuda') and torch.cuda.is_available():
        dev += f' ({torch.cuda.get_device_name(0)})'
    names = getattr(cfg, 'CLASS_NAMES', CLASS_NAMES)
    print('── model ' + '─' * 55)
    print(f'  model       {card.get("model", "DBSwinT_v4")} | {n_params / 1e6:.1f} M parameters')
    print(f'  weights     {weights_dir / "best.pt"} ({size_mb:.0f} MB)'
          + (f' | source run {card["source_run"]}' if card.get('source_run') else ''))
    if vm:
        print(f'  checkpoint  epoch {ck.get("epoch", "?")}, best by validation '
              f'{ck.get("best_metric", "?")} | val network IoU {vm.get("net_IoU", float("nan")):.3f}, '
              f'F1 {vm.get("net_F1", float("nan")):.3f} | crosswalk F1 {vm.get("cw_F1", float("nan")):.3f} '
              f'| entrance F1 {vm.get("ent_F1", float("nan")):.3f}')
    print('  classes     ' + ', '.join(f'{i} {n}' for i, n in enumerate(names))
          + f' | network = classes {getattr(cfg, "NETWORK_CLASSES", [1, 2, 3, 4])}')
    print(f'  input       RGB uint8 at 0.08 m/px, {cfg.IMG_SIZE}-px windows, ImageNet normalization')
    print('  confidence  p = sigmoid(logit / T): '
          + ', '.join(f'{k} {v:.3f}' for k, v in temps.items()))
    print(f'  device      {dev}')



def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def normalize(img_u8):
    """(H,W,3) uint8 -> (H,W,3) float32 normalized like the training data."""
    return (img_u8.astype(np.float32) / 255.0 - MEAN) / STD
