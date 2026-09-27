"""Score the belief filters on held-out demos.

A sample is an alive enemy who is off the radar at a filter step. Samples where the enemy was
seen earlier in the round are the mid/late-round case this project targets; the rest (never
seen yet) are reported separately. Per sample:

    nll        -log P(true node)
    top1/5/20  true node among the k most likely nodes
    mass300    probability within 300 units walking distance of the true position
    exp_dist   expected walking distance from the belief to the true position
    p_place    probability of the true callout (e.g. "Apartments"); place_top1 its top-1 hit

Usage:
    python -m cspredict.evaluate --train-sources xego --split test
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
import polars as pl
from loguru import logger

from cspredict.build import load_models, model_dir
from cspredict.config import OUTPUT_DIR
from cspredict.dataset import DemoRef, list_demos
from cspredict.filters import DEFAULT_CONFIGS, FilterConfig, gather_evidence, run_filter
from cspredict.infostate import Episode, episodes

NEAR = 300.0
SINCE_BINS = [0, 2, 5, 10, 20, 40, np.inf]
SINCE_LABELS = ["0-2s", "2-5s", "5-10s", "10-20s", "20-40s", "40s+"]


def _sighting_history(ep: Episode) -> tuple[np.ndarray, np.ndarray]:
    """(S, E) whether each enemy was seen at an earlier step, and seconds since last seen."""
    seen = ep.enemy_seen
    ever = np.zeros_like(seen)
    since = np.full(seen.shape, np.nan, dtype=np.float32)
    last = np.full(seen.shape[1], np.nan)
    for s in range(ep.n_steps):
        ever[s] = np.isfinite(last)
        since[s] = ep.t_rel[s] - last
        last = np.where(seen[s], ep.t_rel[s], last)
    return ever, since


def score_episode(ep: Episode, models, configs: tuple[FilterConfig, ...]) -> pl.DataFrame:
    grid = models.grid
    ev = gather_evidence(ep, grid, models.spot)
    ever, since = _sighting_history(ep)
    place_onehot = np.zeros((grid.n, len(grid.places)))
    place_onehot[np.arange(grid.n), grid.node_place_idx] = 1.0
    n_friend = ep.obs_alive.sum(axis=1)
    n_enemy = ep.enemy_alive.sum(axis=1)

    cols: dict[str, list] = {k: [] for k in (
        "model", "step", "enemy", "t_rel", "seen_before", "since", "n_friend", "n_enemy", "planted",
        "p_true", "rank", "mass300", "exp_dist", "p_place", "place_top1",
    )}  # fmt: skip
    for cfg in configs:
        for s, b in run_filter(ep, grid, models.motion, cfg, ev, models.library):
            es = np.flatnonzero(ep.enemy_alive[s] & ~ep.enemy_seen[s] & (ep.enemy_node[s] >= 0))
            if len(es) == 0:
                continue
            true = ep.enemy_node[s, es]
            be = b[es]
            p_true = be[np.arange(len(es)), true]
            d = grid.dist[true]
            pp = be @ place_onehot
            place_true = grid.node_place_idx[true]
            k = len(es)
            cols["model"] += [cfg.name] * k
            cols["step"].append(np.full(k, s))
            cols["enemy"].append(es)
            cols["t_rel"].append(np.full(k, ep.t_rel[s]))
            cols["seen_before"].append(ever[s, es])
            cols["since"].append(since[s, es])
            cols["n_friend"].append(np.full(k, n_friend[s]))
            cols["n_enemy"].append(np.full(k, n_enemy[s]))
            cols["planted"].append(np.full(k, ep.planted[s]))
            cols["p_true"].append(p_true)
            cols["rank"].append((be >= p_true[:, None]).sum(axis=1) - 1)  # ties count against the model
            cols["mass300"].append((be * (d <= NEAR)).sum(axis=1))
            cols["exp_dist"].append((be * d).sum(axis=1))
            cols["p_place"].append(pp[np.arange(k), place_true])
            cols["place_top1"].append(pp.argmax(axis=1) == place_true)
    if not cols["model"]:
        return pl.DataFrame()
    df = pl.DataFrame({k: (v if k == "model" else np.concatenate(v)) for k, v in cols.items()})
    return df.with_columns(
        pl.lit(ep.demo_id).alias("demo"), pl.lit(ep.round_num).alias("round"), pl.lit(ep.friendly).alias("friendly")
    )


def score_demo(args: tuple[DemoRef, Path, tuple[FilterConfig, ...]]) -> pl.DataFrame:
    ref, models_path, configs = args
    models = load_models(models_path)
    frames = [
        score_episode(ep, models, configs)
        for friendly in ("ct", "t")
        for ep in episodes(models.grid, ref, friendly)
    ]
    frames = [f for f in frames if f.height]
    return pl.concat(frames) if frames else pl.DataFrame()


def summarize(df: pl.DataFrame, by: list[str] | None = None) -> pl.DataFrame:
    by = by or []
    return (
        df.group_by(["model", *by])
        .agg(
            pl.len().alias("n"),
            (-pl.col("p_true").log()).mean().alias("nll"),
            (pl.col("rank") < 1).mean().alias("top1"),
            (pl.col("rank") < 5).mean().alias("top5"),
            (pl.col("rank") < 20).mean().alias("top20"),
            pl.col("mass300").mean(),
            pl.col("exp_dist").mean(),
            pl.col("p_place").mean(),
            (-pl.col("p_place").clip(1e-9, 1.0).log()).mean().alias("place_nll"),
            pl.col("place_top1").mean(),
        )
        .sort([*by, "nll"])
    )


def with_slices(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        pl.col("since").cut(SINCE_BINS[1:-1], labels=SINCE_LABELS, left_closed=True).alias("since_bin"),
        pl.format("{}v{}", pl.col("n_friend"), pl.col("n_enemy")).alias("situation"),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Score belief filters on held-out demos.")
    ap.add_argument("--train-sources", nargs="+", default=["xego"], help="which fitted models to load")
    ap.add_argument("--sources", nargs="+", default=None, help="demo sources to score (default: train sources)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--gamma", type=float, nargs="+", default=[1.0], help="negative-information tempering")
    ap.add_argument("--models", nargs="+", default=None, help="only these filter configs (default: all)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", type=Path, default=OUTPUT_DIR / "eval")
    args = ap.parse_args()

    models_path = model_dir(args.train_sources)
    refs = list_demos(args.sources or args.train_sources, [args.split])
    configs = [c for c in DEFAULT_CONFIGS if not c.negative]
    for g in args.gamma:
        configs += [replace(c, name=f"{c.name}_g{g}" if len(args.gamma) > 1 else c.name, gamma=g)
                    for c in DEFAULT_CONFIGS if c.negative]  # fmt: skip
    if args.models:
        configs = [c for c in configs if c.name.split("_g")[0] in args.models]
    logger.info(f"Scoring {len(refs)} {args.split} demos with {[c.name for c in configs]}")

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        frames = list(pool.map(score_demo, [(r, models_path, tuple(configs)) for r in refs]))
    df = with_slices(pl.concat([f for f in frames if f.height]))
    args.out.mkdir(parents=True, exist_ok=True)
    df.write_parquet(args.out / f"samples_{args.split}.parquet")

    pl.Config.set_tbl_rows(100)
    pl.Config.set_tbl_cols(20)
    pl.Config.set_tbl_width_chars(200)
    pl.Config.set_float_precision(3)
    mid = df.filter(pl.col("seen_before"))
    print(f"\n== Mid/late round: enemy seen earlier this round, currently off radar ({mid.height:,} samples) ==")
    print(summarize(mid))
    print("\n== By time since last seen ==")
    print(summarize(mid, ["since_bin"]).filter(pl.col("model").is_in(["last_seen", "prior_neg", "hmm", "pf", "ens", "ens_nokill"])))
    print("\n== Clutches: one friendly player left ==")
    print(summarize(mid.filter(pl.col("n_friend") == 1), ["n_enemy"]).filter(pl.col("model").is_in(["last_seen", "prior_neg", "hmm", "ens"])))
    print("\n== Never seen yet this round ==")
    print(summarize(df.filter(~pl.col("seen_before"))))
    summarize(mid).write_csv(args.out / f"summary_{args.split}.csv")


if __name__ == "__main__":
    main()
