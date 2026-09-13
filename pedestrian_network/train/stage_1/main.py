"""Run the semantic segmentation pipeline end to end (steps 1 -> 5).

Each step is an ordinary script and can be run on its own; this runner just
chains them for a fresh full pass. Extra arguments after `--` are forwarded to
2_train.py (e.g. `python main.py -- --epochs 60 --ds-2025-only`).

    python main.py                       # 1,2,3,4,5
    python main.py --steps 1             # dataset prep only
    python main.py --steps 4,5 --run X   # evaluate + export an existing run
"""
import sys
import argparse
import subprocess
from pathlib import Path

HERE  = Path(__file__).resolve().parent
STEPS = {
    '1': ['1_prepare_dataset.py', '--qa'],
    '2': ['2_train.py'],
    '3': ['3_calibrate.py'],
    '4': ['4_evaluate.py'],
    '5': ['5_export_weights.py'],
}


def main():
    argv = sys.argv[1:]
    extra = []
    if '--' in argv:
        i = argv.index('--')
        argv, extra = argv[:i], argv[i + 1:]

    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=str, default='1,2,3,4,5',
                    help='comma-separated subset of 1..5, in order')
    ap.add_argument('--run',   type=str, default=None,
                    help='forwarded to steps 3-5 (default: newest run)')
    args = ap.parse_args(argv)

    for s in args.steps.split(','):
        s = s.strip()
        cmd = [sys.executable, str(HERE / STEPS[s][0])] + STEPS[s][1:]
        if s == '2':
            cmd += extra
        if s in ('3', '4', '5') and args.run:
            cmd += ['--run', args.run]
        print('\n' + '=' * 70 + f'\nSTEP {s}: {" ".join(cmd[1:])}\n' + '=' * 70)
        subprocess.run(cmd, check=True, cwd=HERE)


if __name__ == '__main__':
    main()
