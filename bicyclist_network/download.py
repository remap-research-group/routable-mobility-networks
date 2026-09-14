"""Download the released model weights into run/src/.

OBJECTIVE
  The two trained models (segmentation U-Net ~94 MB, YOLO bike-sign detector
  ~45 MB) are too large for git, so they are attached to a GitHub Release.
  This script fetches them into run/src/ — where the run stages look for them —
  and verifies the SHA-256 sums listed in run/src/SHA256SUMS. Python standard
  library only, no dependencies.

INPUT
  --tag     release tag to fetch from (default: RELEASE_TAG below; "latest" = newest release)
  --force   re-download even if the file already exists
  --only    seg | yolo   fetch just one of the two

OUTPUT
  run/src/seg_unet_r34.pt        U-Net (ResNet34), 4 classes — stage 1 predict
  run/src/yolo26m_bikesign.pt    YOLO, 3 classes (BikeOnly / Sharrow / OnlyBikeBus) — stage 3 signs

USAGE  (from the bicyclist_network/ folder)
  python download.py
  python download.py --tag v0.1.0

Trained your own model? train/seg/3_export_weights.py and
train/yolo/3_export_weights.py copy a checkpoint into run/src/ under these
names (and refresh SHA256SUMS), so nothing else has to change.
"""
import argparse
import hashlib
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "run" / "src"
# TODO: point at the routable-mobility-networks release once the weights are attached there.
REPO = "Synn-Arch/bikelane-extract"
RELEASE_TAG = "latest"

ASSETS = {
    "seg":  "seg_unet_r34.pt",
    "yolo": "yolo26m_bikesign.pt",
}
SUMS = SRC / "SHA256SUMS"


def url_for(tag, name):
    if tag == "latest":
        return f"https://github.com/{REPO}/releases/latest/download/{name}"
    return f"https://github.com/{REPO}/releases/download/{tag}/{name}"


def fetch(url, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "bikelane-download"})
    try:
        r = urllib.request.urlopen(req)
    except urllib.error.HTTPError as e:
        raise SystemExit(
            f"\nERROR: {url} -> HTTP {e.code}.\n"
            f"  The release asset was not found. Check that the release tag exists (--tag), that the\n"
            f"  repository is public (or you are logged in to GitHub), and that the asset is named {dest.name}.\n"
            f"  Manual alternative: download {dest.name} from\n"
            f"  https://github.com/{REPO}/releases and put it at {dest}") from None
    except urllib.error.URLError as e:
        raise SystemExit(f"\nERROR: cannot reach {url}: {e.reason}") from None
    with r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {dest.name}: {done / 2**20:,.0f} / {total / 2**20:,.0f} MB", end="")
    print()
    tmp.replace(dest)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_sums():
    """run/src/SHA256SUMS: '<hex>  <filename>' per line (paths are stripped to the basename)."""
    if not SUMS.exists():
        print(f"  (no {SUMS.relative_to(HERE)} — skipping checksum verification)")
        return {}
    out = {}
    for line in SUMS.read_text().splitlines():
        parts = line.split()
        if len(parts) == 2:
            out[Path(parts[1].lstrip("*")).name] = parts[0]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tag", default=RELEASE_TAG, help=f"release tag (default {RELEASE_TAG})")
    ap.add_argument("--only", choices=sorted(ASSETS), help="fetch only this model")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    sums = load_sums()
    for key, name in ASSETS.items():
        if args.only and key != args.only:
            continue
        dest = SRC / name
        if dest.exists() and not args.force:
            print(f"{dest.relative_to(HERE)} already exists — skipping (use --force to re-download)")
        else:
            print(f"downloading {url_for(args.tag, name)}")
            fetch(url_for(args.tag, name), dest)
        if name in sums:
            ok = sha256(dest) == sums[name]
            print(f"  sha256 {'OK' if ok else 'MISMATCH'}")
            if not ok:
                raise SystemExit(f"{dest} is corrupt — delete it and run again")
    print("done — run/src/ is ready")


if __name__ == "__main__":
    main()
