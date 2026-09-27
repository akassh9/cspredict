"""Find parsed demos and assign train/val/test splits."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from cspredict.config import PARSED_DIR, RAW_DIR


@dataclass(frozen=True)
class DemoRef:
    """A parsed demo on disk."""

    source: str
    demo_id: str
    path: Path
    split: str

    def table(self, name: str) -> pl.DataFrame:
        return pl.read_parquet(self.path / f"{name}.parquet")

    @property
    def meta(self) -> dict:
        return json.loads((self.path / "meta.json").read_text())


def _read_splits(source: str) -> dict[str, str]:
    """Explicit splits: X-Ego's matches.csv, or data/raw/<source>/splits.csv (demo_id,split)."""
    splits: dict[str, str] = {}
    for name, id_col in (("matches.csv", "match_id"), ("splits.csv", "demo_id")):
        f = RAW_DIR / source / name
        if f.exists():
            with f.open() as fh:
                splits.update({row[id_col]: row["split"] for row in csv.DictReader(fh)})
    return splits


def _hash_split(demo_id: str, test_frac: float = 0.2, val_frac: float = 0.1) -> str:
    """Stable pseudo-random split so a demo never moves between splits."""
    u = int(hashlib.sha1(demo_id.encode()).hexdigest(), 16) % 10_000 / 10_000
    return "test" if u < test_frac else "val" if u < test_frac + val_frac else "train"


def list_demos(sources: list[str] | None = None, splits: list[str] | None = None) -> list[DemoRef]:
    """Parsed demos, optionally filtered by source (e.g. ["hltv"]) and split (e.g. ["train"])."""
    refs = []
    for source_dir in sorted(p for p in PARSED_DIR.glob("*") if p.is_dir()):
        source = source_dir.name
        if sources and source not in sources:
            continue
        explicit = _read_splits(source)
        for demo_dir in sorted(p for p in source_dir.iterdir() if (p / "meta.json").exists()):
            split = explicit.get(demo_dir.name, _hash_split(demo_dir.name))
            if splits and split not in splits:
                continue
            refs.append(DemoRef(source, demo_dir.name, demo_dir, split))
    return refs


def load_ticks(refs: list[DemoRef], columns: list[str] | None = None) -> pl.DataFrame:
    """Concatenate tick tables from several demos, tagged with demo_id."""
    frames = []
    for ref in refs:
        t = pl.read_parquet(ref.path / "ticks.parquet", columns=columns)
        frames.append(t.with_columns(pl.lit(ref.demo_id).alias("demo_id")))
    return pl.concat(frames, how="vertical_relaxed") if frames else pl.DataFrame()
