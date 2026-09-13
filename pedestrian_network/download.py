"""Download the released model weights (and optionally the training dataset).

OBJECTIVE
  The pretrained checkpoint (~240 MB) and the training imagery (~2.4 GB) are
  too large for git, so they are attached to a GitHub Release. This script
  fetches them into the places the code expects, verifies the SHA-256 sums,
  and unpacks the dataset. Python standard library only — no dependencies.

INPUT
  --weights   (default) best.pt                    -> run/src/best.pt
  --dataset   dc_2023.zip + dc_2025.zip            -> train/v0_2026sep_image/
  --tag       release tag to fetch from (default: RELEASE_TAG below)
  --force     re-download even if the file already exists

OUTPUT
  run/src/best.pt                                the model; run/ is ready
  train/v0_2026sep_image/dc_2023/{images,masks}  the training tiles (with
  train/v0_2026sep_image/dc_2025/{images,masks}   --dataset only)

USAGE  (from the pedestrian_network/ folder)
  python download.py                 # weights only — enough to run the tool
  python download.py --dataset       # also the training imagery (for train/)
"""
import sys
import json
import hashlib
import argparse
import zipfile
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = 'remap-research-group/routable-mobility-networks'
RELEASE_TAG = 'pednet-v0.1'

ASSETS = {
    'weights': [('best.pt', HERE / 'run' / 'src' / 'best.pt')],
    'dataset': [('dc_2023.zip', HERE / 'train' / 'v0_2026sep_image' / 'dc_2023.zip'),
                ('dc_2025.zip', HERE / 'train' / 'v0_2026sep_image' / 'dc_2025.zip')],
}
SUMS_NAME = 'SHA256SUMS.txt'


def url_for(tag, name):
    return f'https://github.com/{REPO}/releases/download/{tag}/{name}'


def fetch(url, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + '.part')
    req = urllib.request.Request(url, headers={'User-Agent': 'pednet-download'})
    with urllib.request.urlopen(req) as r, open(tmp, 'wb') as f:
        total = int(r.headers.get('Content-Length') or 0)
        done = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f'\r  {dest.name}: {done / 2**20:,.0f} / {total / 2**20:,.0f} MB', end='')
    print()
    tmp.replace(dest)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def load_sums(tag):
    try:
        with urllib.request.urlopen(url_for(tag, SUMS_NAME)) as r:
            text = r.read().decode()
    except Exception as e:
        print(f'  (no {SUMS_NAME} in the release: {e} — skipping checksum verification)')
        return {}
    return {line.split()[1].lstrip('*'): line.split()[0] for line in text.splitlines() if line.strip()}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--weights', action='store_true', help='model weights (default if nothing else is given)')
    ap.add_argument('--dataset', action='store_true', help='training imagery + masks (~2.4 GB)')
    ap.add_argument('--tag', default=RELEASE_TAG, help=f'release tag (default {RELEASE_TAG})')
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()
    if not args.dataset:
        args.weights = True

    sums = load_sums(args.tag)
    groups = (['weights'] if args.weights else []) + (['dataset'] if args.dataset else [])
    for g in groups:
        for name, dest in ASSETS[g]:
            if dest.exists() and not args.force:
                print(f'{dest.relative_to(HERE)} already exists — skipping (use --force to re-download)')
            else:
                print(f'downloading {url_for(args.tag, name)}')
                fetch(url_for(args.tag, name), dest)
            if name in sums:
                ok = sha256(dest) == sums[name]
                print(f'  sha256 {"OK" if ok else "MISMATCH"}')
                if not ok:
                    raise SystemExit(f'{dest} is corrupt — delete it and run again')
            if dest.suffix == '.zip':
                print(f'  unpacking into {dest.parent.relative_to(HERE)}/')
                with zipfile.ZipFile(dest) as z:
                    z.extractall(dest.parent)
                dest.unlink()
    print('done')


if __name__ == '__main__':
    main()
