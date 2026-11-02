"""Extract OSSDD .tar shards into browsable folders and render PNG previews for each shard.

  data/raw/ossdd/extracted/<split>/<shard>/   <- vv.tif, vh.tif, aabb.txt, ... per chip
  results/figures/ossdd_preview/<shard>/      <- viewable PNGs with ship boxes

Shards that are incomplete (size not a multiple of the 512-byte tar block, or unreadable) are
skipped, so this is safe to re-run after every download.

Usage:
  python -m src.extract_ossdd --shards data/raw/ossdd/webdataset --n-previews 8
"""
import argparse
import subprocess
import sys
import tarfile
from pathlib import Path


def is_complete(tar_path: Path) -> bool:
    if tar_path.stat().st_size % 512 != 0:
        return False
    try:
        with tarfile.open(tar_path) as t:
            t.getmembers()
        return True
    except (tarfile.TarError, EOFError, OSError):
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", default="data/raw/ossdd/webdataset")
    ap.add_argument("--out", default="data/raw/ossdd/extracted")
    ap.add_argument("--n-previews", type=int, default=8)
    a = ap.parse_args()

    for tar_path in sorted(Path(a.shards).glob("*/*.tar")):
        split, shard = tar_path.parent.name, tar_path.stem
        if not is_complete(tar_path):
            print(f"skip (incomplete): {tar_path.name}")
            continue
        dest = Path(a.out) / split / shard
        if not dest.exists():
            dest.mkdir(parents=True)
            with tarfile.open(tar_path) as t:
                t.extractall(dest, filter="data")  # 'data' filter blocks paths escaping dest
            print(f"extracted {tar_path.name} -> {dest} ({len(list(dest.iterdir()))} files)")
        preview_dir = Path("results/figures/ossdd_preview") / shard
        if not preview_dir.exists():
            subprocess.run(
                [sys.executable, "-W", "ignore", "-m", "src.preview_ossdd", "--tar", str(tar_path),
                 "--n", str(a.n_previews), "--out", str(preview_dir)],
                check=True,
            )


if __name__ == "__main__":
    main()
