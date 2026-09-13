"""Step 5 — export the deployable artifacts to run/src/.

OBJECTIVE
  Publish a finished training run as the model that run/ ships, so inference
  needs neither the training code paths nor the training data. After this
  step run/stage_1/segmentation.py uses the new model automatically.

INPUT
  train/output/runs/<run>/: best.pt, temperature.json, config.json
  (newest run with a best.pt, or --run <name|path>)

OUTPUT  (run/src/, existing files overwritten)
  best.pt              model checkpoint (state dict + epoch + val metrics)
  temperature.json     per-head calibration temperatures
  train_config.json    architecture / input / training snapshot (ARCH_KEYS of
                       config.json; absolute paths and CLI args stripped) —
                       run/src/loader.py rebuilds the model from it
  model_card.json      classes, network classes, normalization, confidence
                       formula, source run, export date

HOW IT WORKS (functions)
  resolve_run()   newest run under train/output/runs/, or the given one
  main()          copy best.pt and temperature.json, filter config.json to
                  ARCH_KEYS, write the model card

TUNING / CAVEATS
  ARCH_KEYS lists what the loader needs (NUM_CLASSES, IMG_SIZE, EMBED_DIM,
  DEPTHS, NUM_HEADS, WINDOW_SIZE, LOCAL_PATCH, GLOBAL_PATCH, head switches,
  ...) plus a few training facts for provenance. If you change the
  architecture itself in train/stage_1/model.py, copy that file over
  run/src/model.py as well — the two are identical copies. A missing
  temperature.json (interrupted training) is reported: run 3_calibrate.py.

USAGE
  python 5_export_weights.py                # newest run
  python 5_export_weights.py --run <name>
"""
import json
import shutil
import argparse
from datetime import datetime, timezone
from pathlib import Path

from config import cfg

# keys of the run's config.json that run/src/loader.py needs to rebuild the model
ARCH_KEYS = ['NUM_CLASSES', 'CLASS_NAMES', 'NETWORK_CLASSES', 'ENTRANCE_CLASSES',
             'CROSSWALK_CLASSES', 'USE_ENTRANCE_HEAD', 'USE_CROSSWALK_HEAD',
             'IMG_SIZE', 'IN_CHANNELS', 'EMBED_DIM', 'DEPTHS', 'NUM_HEADS',
             'WINDOW_SIZE', 'LOCAL_PATCH', 'GLOBAL_PATCH', 'PRETRAINED_MODEL',
             'GAP_CLOSE_PX', 'CONN_RADIUS_PX', 'LAMBDA_CONN', 'EPOCHS', 'BATCH_SIZE',
             'LR', 'LR_PRETRAINED', 'WEIGHT_DECAY', 'WARMUP_EPOCHS', 'BEST_METRIC',
             'STRONG_AUG', 'CLASS_WEIGHTS']


def resolve_run(run_arg):
    if run_arg is None:
        cands = sorted(d for d in cfg.RUNS_DIR.glob('*') if (d / 'best.pt').exists())
        assert cands, f'no finished runs under {cfg.RUNS_DIR}'
        return cands[-1]
    run_dir = Path(run_arg)
    if not run_dir.is_absolute():
        run_dir = cfg.RUNS_DIR / run_arg
    assert (run_dir / 'best.pt').exists(), f'{run_dir} has no best.pt'
    return run_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=str, default=None,
                    help='run folder (name under train/output/runs/ or absolute); default: newest')
    args = ap.parse_args()

    run_dir = resolve_run(args.run)
    src = cfg.SRC_DIR
    src.mkdir(parents=True, exist_ok=True)

    shutil.copyfile(run_dir / 'best.pt', src / 'best.pt')
    copied = ['best.pt']

    if (run_dir / 'temperature.json').exists():
        shutil.copyfile(run_dir / 'temperature.json', src / 'temperature.json')
        copied.append('temperature.json')
    else:
        print(f'WARN: {run_dir / "temperature.json"} missing — run 3_calibrate.py first')

    if (run_dir / 'config.json').exists():
        run_cfg = json.loads((run_dir / 'config.json').read_text())
        snap = {k: run_cfg[k] for k in ARCH_KEYS if k in run_cfg}
        snap['source_run'] = run_dir.name
        (src / 'train_config.json').write_text(json.dumps(snap, indent=2))
        copied.append('train_config.json')
    else:
        print(f'WARN: {run_dir / "config.json"} missing (no config snapshot)')

    card = {
        'model': 'DBSwinT_v4 (dual-branch Swin-T, AFF fusion, U-Net decoder, 4 heads)',
        'source_run': run_dir.name,
        'exported_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'classes': dict(enumerate(cfg.CLASS_NAMES)),
        'network_classes': cfg.NETWORK_CLASSES,
        'input': {'img_size': cfg.IMG_SIZE, 'channels': 3,
                  'resolution_m_per_px': 0.08,
                  'normalize_mean': [0.485, 0.456, 0.406],
                  'normalize_std': [0.229, 0.224, 0.225]},
        'confidence': 'p = sigmoid(logit / T_head), T from temperature.json',
        'consumers': ['run/stage_1/segmentation.py'],
    }
    (src / 'model_card.json').write_text(json.dumps(card, indent=2))
    copied.append('model_card.json')

    print(f'exported from {run_dir.name} -> {src}:')
    for c in copied:
        print(f'  {c}')


if __name__ == '__main__':
    main()
