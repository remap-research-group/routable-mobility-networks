"""Central configuration for the training pipeline (train/stage_1/).

All paths are resolved relative to the repository root, so the repo can be
cloned anywhere:
    train/v0_2026sep_image/   finalized training data (dc_2023/, dc_2025/)
    train/output/runs/        training runs (checkpoints, logs, history)
    run/src/                  deployable weights (written by 5_export_weights.py)
Edit hyperparameters here, not in the step scripts.
"""
from pathlib import Path

# ---- Repository layout ----
REPO_ROOT = Path(__file__).resolve().parents[2]            # <repo>/
DATA_ROOT = REPO_ROOT / 'train' / 'v0_2026sep_image'       # see train/README.md
OUT_ROOT  = REPO_ROOT / 'train' / 'output'
SRC_DIR   = REPO_ROOT / 'run' / 'src'                      # deployable weights

# ---- DC tile grid (tile_<TX>_<TY> stems; see train/README.md) ----
EPSG         = 26985
RES_M_PER_PX = 0.08
TILE_PX      = 1024
TILE_M       = TILE_PX * RES_M_PER_PX      # 81.92 m
GRID_XMIN    = 395852.57
GRID_YMAX    = 140914.34

# ---- Canonical 5-class scheme (id -> (name, CVAT labelmap RGB)) ----
PALETTE = {
    0: ('background',       (  0,   0,   0)),
    1: ('visible_sidewalk', (255, 140,   0)),
    2: ('tree',             ( 34, 139,  34)),
    3: ('entrance',         ( 70, 130, 180)),
    4: ('crosswalk',        (220,  20,  60)),
}

# ---- CVAT exports expected in train/v0_2026sep_image/cvat_exports/ ("Segmentation mask 1.1") ----
CVAT_EXPORTS = {2023: 'DC_2023', 2025: 'DC_2025'}   # year -> export name (.zip or folder)


class Config:
    DATA_ROOT = DATA_ROOT
    OUT_ROOT  = OUT_ROOT

    DS_2023  = DATA_ROOT / 'dc_2023'
    DS_2025  = DATA_ROOT / 'dc_2025'
    IMG_EXT  = '.tif'

    COMBINED_STATS = DATA_ROOT / 'combined_dataset_stats.json'

    # ---- Labels ----
    NUM_CLASSES      = 5
    CLASS_NAMES      = ['background', 'visible_sidewalk', 'tree', 'entrance', 'crosswalk']

    # Full pedestrian network = everything walkable, crosswalk included.
    NETWORK_CLASSES   = [1, 2, 3, 4]
    ENTRANCE_CLASSES  = [3]      # auxiliary head
    CROSSWALK_CLASSES = [4]      # auxiliary head

    CLASS_WEIGHTS = [0.50, 2.50, 3.50, 5.00, 3.00]   # overridden by COMBINED_STATS
    MAX_CLASS_WEIGHT = 25.0      # stats-file weights are clipped here for stability
    IGNORE_INDEX  = 255

    USE_ENTRANCE_HEAD  = True
    USE_CROSSWALK_HEAD = True

    # ---- Gap handling / connectivity (see README) ----
    GAP_CLOSE_PX   = 2    # closing radius: fills inter-object gaps up to ~2*r px
    CONN_RADIUS_PX = 8    # crosswalk mass must lie within this many px (0.64 m) of walkable
    LAMBDA_CONN    = 0.20

    IMG_SIZE    = 512
    IN_CHANNELS = 3

    EMBED_DIM    = 96
    DEPTHS       = [2, 2, 6, 2]
    NUM_HEADS    = [3, 6, 12, 24]
    WINDOW_SIZE  = 7
    LOCAL_PATCH  = 4
    GLOBAL_PATCH = 8

    PRETRAINED_LOCAL = True
    PRETRAINED_MODEL = 'swin_tiny_patch4_window7_224'

    BATCH_SIZE    = 4
    NUM_WORKERS   = 4
    EPOCHS        = 120
    LR            = 2e-4
    LR_PRETRAINED = 2e-5
    WEIGHT_DECAY  = 1e-3
    WARMUP_EPOCHS = 5

    LAMBDA_CE       = 1.0
    LAMBDA_DICE_NET = 1.0
    LAMBDA_RMI_NET  = 0.5
    # Aux heads: classes 3 and 4 are already inside the network head; these only
    # provide a "what is specifically entrance / crosswalk" signal.
    LAMBDA_DICE_EN  = 0.25
    LAMBDA_RMI_EN   = 0.10
    LAMBDA_DICE_CW  = 0.25
    LAMBDA_RMI_CW   = 0.10

    BEST_METRIC = 'iou'   # net IoU = full (gap-closed) pedestrian-network coverage
    STRONG_AUG  = True

    RUNS_DIR = OUT_ROOT / 'runs'
    SRC_DIR  = SRC_DIR             # deployable weights land here (5_export_weights.py)
    CKPT_DIR = None


cfg = Config()
