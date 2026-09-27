"""Download the X-Ego-CS demos used for development (FACEIT matches, MIT licence).

Usage:
    python -m cspredict.fetch_xego            # every MAP_NAME demo (46 on Mirage, ~17 GB) into data/raw/xego/
    python -m cspredict.fetch_xego --limit 3  # just a few, to try the pipeline
"""

from __future__ import annotations

import argparse
import csv
import shutil
import urllib.request
from pathlib import Path

from loguru import logger

from cspredict.config import MAP_NAME, RAW_DIR

BASE_URL = "https://huggingface.co/datasets/wangyz1999/X-EGO-CS/resolve/main"


def _download(url: str, dest: Path) -> None:
    """Stream to a .part file first so an interrupted download is never mistaken for a demo."""
    tmp = dest.with_name(dest.name + ".part")
    with urllib.request.urlopen(url) as resp, tmp.open("wb") as out:
        shutil.copyfileobj(resp, out, length=1 << 20)
    tmp.rename(dest)


def main() -> None:
    ap = argparse.ArgumentParser(description="Download X-Ego-CS demos for MAP_NAME.")
    ap.add_argument("--out", type=Path, default=RAW_DIR / "xego")
    ap.add_argument("--limit", type=int, default=None, help="download at most this many demos")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    manifest = args.out / "matches.csv"  # also holds the dataset's train/val/test splits
    if not manifest.exists():
        _download(f"{BASE_URL}/manifest/matches.csv", manifest)
    with manifest.open() as fh:
        ids = [row["match_id"] for row in csv.DictReader(fh) if row["map_name"] == MAP_NAME][: args.limit]
    for match_id in ids:
        dest = args.out / f"{match_id}.dem"
        if not dest.exists():
            logger.info(f"Downloading {match_id}")
            _download(f"{BASE_URL}/demo/{match_id}.dem", dest)
    logger.info(f"{len(ids)} {MAP_NAME} demos in {args.out}")


if __name__ == "__main__":
    main()
