"""Shared by train/seg/3_export_weights.py and train/yolo/3_export_weights.py:
copy a checkpoint into run/src/ under the released name and update SHA256SUMS."""
from __future__ import annotations

import hashlib
import shutil
import time
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def update_sums(sums_file: Path, name: str, digest: str) -> None:
    lines = sums_file.read_text().splitlines() if sums_file.exists() else []
    lines = [ln for ln in lines if not ln.split()[-1:] == [name]]
    lines.append(f"{digest}  {name}")
    sums_file.write_text("\n".join(lines) + "\n")


def export_weights(src: Path, dst: Path, describe=None) -> Path:
    src, dst = Path(src), Path(dst)
    if not src.exists():
        raise SystemExit(f"checkpoint not found: {src}")
    print(f"export  {src}" + (f"  [{describe(src)}]" if describe else ""))
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        bak = dst.with_name(f"{dst.stem}.{int(time.time())}{dst.suffix}")
        dst.rename(bak)
        print(f"  previous weights kept as {bak.name}")
    shutil.copyfile(src, dst)
    digest = sha256(dst)
    update_sums(dst.parent / "SHA256SUMS", dst.name, digest)
    print(f"  → {dst} ({dst.stat().st_size/1e6:.0f} MB)\n  sha256 {digest}  (SHA256SUMS updated)")
    return dst
