"""Download the released model weights into run/src/, and optionally the example imagery.

OBJECTIVE
  The two trained models (segmentation U-Net ~94 MB, YOLO bike-sign detector
  ~45 MB) and the example imagery (all MassGIS 2025 tiles of the two reference
  towns, several GB) are too large for git, so they are attached to the GitHub
  Release RELEASE_TAG. This script fetches them into the places the code
  expects and verifies SHA-256 sums (run/src/SHA256SUMS for the weights,
  SHA256SUMS.txt from the release for the tiles). Standard library only.

INPUT
  (nothing)          the weights  -> run/src/seg_unet_r34.pt, run/src/yolo26m_bikesign.pt
  --tiles REGION     the tiles of a reference town (BOSTON or LEXINGTON; repeatable)
                     -> example/input/<REGION>/tiles/   (the release ships them as
                        <REGION>_tiles_part01.zip, part02.zip, … each < 2 GB; every part is
                        downloaded, verified, unpacked and deleted)
  --only seg|yolo    fetch just one of the two weights
  --tag TAG          release tag (default RELEASE_TAG)
  --force            re-download even if the file exists
  --keep-zips        keep the tile zips next to the unpacked tiles

OUTPUT
  run/src/seg_unet_r34.pt, run/src/yolo26m_bikesign.pt
  example/input/<REGION>/tiles/tile_px<X>_py<Y>.jpg   (Tile_Mappings.csv and imagery_info.json
                                                       for the town are already in the repository)

USAGE  (from the bicyclist_network/ folder)
  python download.py                        # weights — enough to run the tool on your own imagery
  python download.py --tiles BOSTON         # + Boston example imagery (~4 GB), then: bikelane predict -c example/boston.yaml
  python download.py --tiles LEXINGTON      # + Lexington (~14 GB)

Trained your own model? train/seg/3_export_weights.py and
train/yolo/3_export_weights.py copy a checkpoint into run/src/ under these
names (and refresh SHA256SUMS), so nothing else has to change.
"""
import argparse
import hashlib
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "run" / "src"
EXAMPLE_INPUT = HERE / "example" / "input"
REPO = "remap-research-group/routable-mobility-networks"
RELEASE_TAG = "bikenet-v0.1"

ASSETS = {
    "seg":  "seg_unet_r34.pt",
    "yolo": "yolo26m_bikesign.pt",
}
SUMS = SRC / "SHA256SUMS"                 # weights (in the repository)
RELEASE_SUMS = "SHA256SUMS.txt"           # tile zips (release asset)
TILE_REGIONS = ("BOSTON", "LEXINGTON")


def url_for(tag, name):
    if tag == "latest":
        return f"https://github.com/{REPO}/releases/latest/download/{name}"
    return f"https://github.com/{REPO}/releases/download/{tag}/{name}"


def _open(url):
    req = urllib.request.Request(url, headers={"User-Agent": "bikelane-download"})
    return urllib.request.urlopen(req)


def exists(url):
    """True if the release asset exists (HEAD-like probe; GitHub answers 302/200, 404 otherwise)."""
    try:
        with _open(url):
            return True
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return False
        raise


def fetch(url, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        r = _open(url)
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


def parse_sums(text):
    out = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 2:
            out[Path(parts[1].lstrip("*")).name] = parts[0]
    return out


def load_sums():
    """run/src/SHA256SUMS: '<hex>  <filename>' per line (paths are stripped to the basename)."""
    if not SUMS.exists():
        print(f"  (no {SUMS.relative_to(HERE)} — skipping checksum verification)")
        return {}
    return parse_sums(SUMS.read_text())


def load_release_sums(tag):
    try:
        with _open(url_for(tag, RELEASE_SUMS)) as r:
            return parse_sums(r.read().decode())
    except Exception as e:
        print(f"  (no {RELEASE_SUMS} in the release: {e} — tile zips are not checksum-verified)")
        return {}


def verify(dest, sums):
    if dest.name in sums:
        ok = sha256(dest) == sums[dest.name]
        print(f"  sha256 {'OK' if ok else 'MISMATCH'}")
        if not ok:
            raise SystemExit(f"{dest} is corrupt — delete it and run again")


def get_weights(args, sums):
    for key, name in ASSETS.items():
        if args.only and key != args.only:
            continue
        dest = SRC / name
        if dest.exists() and not args.force:
            print(f"{dest.relative_to(HERE)} already exists — skipping (use --force to re-download)")
        else:
            print(f"downloading {url_for(args.tag, name)}")
            fetch(url_for(args.tag, name), dest)
        verify(dest, sums)


def get_tiles(region, args, sums):
    region = region.upper()
    if region not in TILE_REGIONS:
        raise SystemExit(f"--tiles: unknown region {region}; the release has {', '.join(TILE_REGIONS)}")
    out = EXAMPLE_INPUT / region
    tiles = out / "tiles"
    have = len(list(tiles.glob("*.jpg"))) if tiles.exists() else 0
    if have and not args.force:
        print(f"{tiles.relative_to(HERE)} already has {have} tiles — skipping (use --force to re-download)")
        return
    if not (out / "Tile_Mappings.csv").exists():
        print(f"  WARNING: {out / 'Tile_Mappings.csv'} is missing — the repository should ship it; "
              f"the tiles alone cannot be georeferenced")
    part = 0
    while True:
        part += 1
        name = f"{region}_tiles_part{part:02d}.zip"
        url = url_for(args.tag, name)
        if not exists(url):
            break
        dest = out / name
        if not (dest.exists() and not args.force):
            print(f"downloading {url}")
            fetch(url, dest)
        verify(dest, sums)
        print(f"  unpacking {name} → {tiles.relative_to(HERE)}/")
        with zipfile.ZipFile(dest) as z:
            z.extractall(out)                       # zip members are tiles/<name>.jpg
        if not args.keep_zips:
            dest.unlink()
    if part == 1:
        raise SystemExit(f"no {region}_tiles_part01.zip in release {args.tag} — check the release page:\n"
                         f"  https://github.com/{REPO}/releases/tag/{args.tag}")
    n = len(list(tiles.glob("*.jpg")))
    print(f"{region}: {part - 1} part(s), {n} tiles in {tiles.relative_to(HERE)}/"
          f"\n→ bikelane config -c example/{region.lower()}.yaml")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tag", default=RELEASE_TAG, help=f"release tag (default {RELEASE_TAG})")
    ap.add_argument("--only", choices=sorted(ASSETS), help="fetch only this model")
    ap.add_argument("--tiles", action="append", default=[], metavar="REGION",
                    help="also fetch the example tiles of this town into example/input/<REGION>/ (BOSTON | LEXINGTON)")
    ap.add_argument("--keep-zips", action="store_true", help="keep the tile zips after unpacking")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    get_weights(args, load_sums())
    if args.tiles:
        rel = load_release_sums(args.tag)
        for region in args.tiles:
            get_tiles(region, args, rel)
    print("done — run/src/ is ready" + (" and example/input/ has the tiles" if args.tiles else ""))


if __name__ == "__main__":
    main()
