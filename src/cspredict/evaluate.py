"""Score the belief filters on held-out demos.

A sample is an alive enemy who is off the radar at a filter step. Samples where the enemy was
seen earlier in the round are the mid/late-round case this project targets; the rest (never
seen yet) are reported separately. Per sample:

    nll          -log P(true node) (cell log-loss)
    top1/5/20    true node among the k most likely nodes
    mass300      probability within 300 units walking distance of the true position
    exp_dist     expected walking distance from the belief to the true position
    p_place      probability of the true callout (e.g. "Apartments"); place_nll = -log p_place
    place_top1/3 true callout among the 1 or 3 most likely callouts (ties count against the model)

Intervals are 95% bootstrap intervals that resample whole rounds, because moments within a round
are correlated. Differences between models are paired: every model is scored on the same samples,
so the interval of "model minus reference" is much tighter than the two separate intervals.

Usage:
    python -m cspredict.evaluate --train-sources hltv xego --sources hltv --split val
    python -m cspredict.evaluate ... --models ens ens_weapon --set particles.N_PARTICLES=256
    python -m cspredict.evaluate --report outputs/eval --split val --ref ens   # re-print a saved run
"""

from __future__ import annotations

import argparse
import importlib
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
import polars as pl
from loguru import logger

from cspredict.build import load_models, model_dir
from cspredict.config import OUTPUT_DIR
from cspredict.dataset import DemoRef, list_demos
from cspredict.filters import ALL_CONFIGS, DEFAULT_CONFIGS, FilterConfig, gather_evidence, run_filter
from cspredict.infostate import WEAPON_CLASSES, Episode, episodes

NEAR = 300.0
SINCE_BINS = [0, 2, 5, 10, 20, 40, np.inf]
SINCE_LABELS = ["0-2s", "2-5s", "5-10s", "10-20s", "20-40s", "40s+"]
CALIB_EDGES = (0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
HEADLINE = ("place_top1", "place_top3", "place_nll", "nll")
PROB_FLOOR = 1e-9  # callout probabilities are clipped here before taking logs


def _sighting_history(ep: Episode) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(S, E) whether each enemy was seen at an earlier step, seconds since last seen, and the
    weapon class in their hands then (-1 if not seen yet)."""
    seen = ep.enemy_seen
    ever = np.zeros_like(seen)
    since = np.full(seen.shape, np.nan, dtype=np.float32)
    weapon = np.full(seen.shape, -1, dtype=np.int8)
    last = np.full(seen.shape[1], np.nan)
    last_w = np.full(seen.shape[1], -1, dtype=np.int8)
    for s in range(ep.n_steps):
        ever[s] = np.isfinite(last)
        since[s] = ep.t_rel[s] - last
        weapon[s] = last_w
        last = np.where(seen[s], ep.t_rel[s], last)
        last_w = np.where(seen[s], ep.enemy_seen_weapon[s], last_w)
    return ever, since, weapon


def score_episode(ep: Episode, models, configs: tuple[FilterConfig, ...]) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Per-sample scores, and the full callout distributions of the mid/late-round samples."""
    grid = models.grid
    ev = gather_evidence(ep, grid, models.spot)
    ever, since, last_weapon = _sighting_history(ep)
    place_onehot = np.zeros((grid.n, len(grid.places)))
    place_onehot[np.arange(grid.n), grid.node_place_idx] = 1.0
    n_friend = ep.obs_alive.sum(axis=1)
    n_enemy = ep.enemy_alive.sum(axis=1)

    cols: dict[str, list] = {k: [] for k in (
        "model", "step", "enemy", "t_rel", "seen_before", "since", "last_weapon", "n_friend", "n_enemy", "planted",
        "p_true", "rank", "mass300", "exp_dist", "p_place", "place_true", "place_rank",
    )}  # fmt: skip
    probs: list[np.ndarray] = []
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
            p_place = pp[np.arange(len(es)), place_true]
            k = len(es)
            cols["model"] += [cfg.name] * k
            cols["step"].append(np.full(k, s))
            cols["enemy"].append(es)
            cols["t_rel"].append(np.full(k, ep.t_rel[s]))
            cols["seen_before"].append(ever[s, es])
            cols["since"].append(since[s, es])
            cols["last_weapon"].append(last_weapon[s, es])
            cols["n_friend"].append(np.full(k, n_friend[s]))
            cols["n_enemy"].append(np.full(k, n_enemy[s]))
            cols["planted"].append(np.full(k, ep.planted[s]))
            cols["p_true"].append(p_true)
            cols["rank"].append((be >= p_true[:, None]).sum(axis=1) - 1)  # ties count against the model
            cols["mass300"].append((be * (d <= NEAR)).sum(axis=1))
            cols["exp_dist"].append((be * d).sum(axis=1))
            cols["p_place"].append(p_place)
            cols["place_true"].append(place_true)
            cols["place_rank"].append((pp >= p_place[:, None]).sum(axis=1) - 1)
            probs.append(pp[ever[s, es]].astype(np.float32))
    if not cols["model"]:
        return pl.DataFrame(), pl.DataFrame()
    df = pl.DataFrame({k: (v if k == "model" else np.concatenate(v)) for k, v in cols.items()})
    df = df.with_columns(
        pl.lit(ep.demo_id).alias("demo"), pl.lit(ep.round_num).alias("round"), pl.lit(ep.friendly).alias("friendly")
    )
    mid = df.filter(pl.col("seen_before")).select("model", "demo", "round", "friendly", "step", "enemy", "place_true")
    pp = np.concatenate(probs)
    return df, mid.with_columns(pl.Series("place_probs", pp, dtype=pl.Array(pl.Float32, pp.shape[1])))


def apply_overrides(overrides: tuple[str, ...]) -> None:
    """Set module constants, e.g. "particles.N_PARTICLES=256", for quick variants."""
    for item in overrides:
        target, value = item.split("=", 1)
        module, name = target.rsplit(".", 1)
        mod = importlib.import_module(f"cspredict.{module}")
        if not hasattr(mod, name):
            raise AttributeError(f"cspredict.{module} has no constant {name}")
        setattr(mod, name, type(getattr(mod, name))(float(value) if value.lower() != "inf" else np.inf))


def score_demo(args: tuple[DemoRef, Path, tuple[FilterConfig, ...], tuple[str, ...]]) -> tuple[pl.DataFrame, pl.DataFrame]:
    ref, models_path, configs, overrides = args
    apply_overrides(overrides)
    models = load_models(models_path)
    out = [
        score_episode(ep, models, configs)
        for friendly in ("ct", "t")
        for ep in episodes(models.grid, ref, friendly)
    ]
    out = [o for o in out if o[0].height]
    if not out:
        return pl.DataFrame(), pl.DataFrame()
    return pl.concat([o[0] for o in out]), pl.concat([o[1] for o in out if o[1].height])


# ---------------------------------------------------------------------- reporting
def summarize(df: pl.DataFrame, by: list[str] | None = None) -> pl.DataFrame:
    by = by or []
    return (
        df.group_by(["model", *by])
        .agg(
            pl.len().alias("n"),
            (pl.col("place_rank") < 1).mean().alias("place_top1"),
            (pl.col("place_rank") < 3).mean().alias("place_top3"),
            (-pl.col("p_place").clip(PROB_FLOOR, 1.0).log()).mean().alias("place_nll"),
            (-pl.col("p_true").log()).mean().alias("nll"),
            (pl.col("rank") < 1).mean().alias("top1"),
            (pl.col("rank") < 20).mean().alias("top20"),
            pl.col("mass300").mean(),
            pl.col("exp_dist").mean(),
        )
        .sort([*by, "place_nll"])
    )


def with_slices(df: pl.DataFrame) -> pl.DataFrame:
    weapons = pl.col("last_weapon").replace_strict(dict(enumerate(WEAPON_CLASSES)), default="not seen", return_dtype=pl.String)
    return df.with_columns(
        pl.col("since").cut(SINCE_BINS[1:-1], labels=SINCE_LABELS, left_closed=True).alias("since_bin"),
        pl.format("{}v{}", pl.col("n_friend"), pl.col("n_enemy")).alias("situation"),
        weapons.alias("weapon_seen"),
    )


def _metric_columns(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        (pl.col("place_rank") < 1).cast(pl.Float64).alias("place_top1"),
        (pl.col("place_rank") < 3).cast(pl.Float64).alias("place_top3"),
        (-pl.col("p_place").clip(PROB_FLOOR, 1.0).log()).alias("place_nll"),
        (-pl.col("p_true").log()).alias("nll"),
        pl.format("{}|{}", pl.col("demo"), pl.col("round")).alias("cluster"),
    )


def intervals(
    df: pl.DataFrame, ref: str | None = None, metrics: tuple[str, ...] = HEADLINE, reps: int = 2000, seed: int = 0
) -> pl.DataFrame:
    """Mean and round-bootstrap 95% interval of each metric per model, plus the paired difference
    to `ref` (same resampled rounds for every model). Both views of a round share a cluster."""
    d = _metric_columns(df)
    clusters = d["cluster"].unique().sort()
    idx = {c: i for i, c in enumerate(clusters.to_list())}
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(clusters), size=(reps, len(clusters)))
    times = np.zeros((reps, len(clusters)))
    np.add.at(times, (np.repeat(np.arange(reps), len(clusters)), draws.ravel()), 1.0)

    per_model = {}
    for (model,), g in d.group_by(["model"]):
        agg = g.group_by("cluster").agg(pl.len().alias("n"), *[pl.col(m).sum() for m in metrics])
        ci = np.array([idx[c] for c in agg["cluster"].to_list()])
        n = np.zeros(len(clusters))
        n[ci] = agg["n"].to_numpy()
        sums = np.zeros((len(metrics), len(clusters)))
        for j, m in enumerate(metrics):
            sums[j, ci] = agg[m].to_numpy()
        per_model[model] = (n, sums)

    rows = []
    for model, (n, sums) in per_model.items():
        boot = (times @ sums.T) / (times @ n)[:, None]  # (reps, metrics)
        row = {"model": model, "n": int(n.sum())}
        for j, m in enumerate(metrics):
            lo, hi = np.percentile(boot[:, j], [2.5, 97.5])
            row |= {m: sums[j].sum() / n.sum(), f"{m}_lo": lo, f"{m}_hi": hi}
        if ref is not None and ref in per_model and model != ref:
            rn, rsums = per_model[ref]
            rboot = (times @ rsums.T) / (times @ rn)[:, None]
            for j, m in enumerate(metrics):
                diff = boot[:, j] - rboot[:, j]
                lo, hi = np.percentile(diff, [2.5, 97.5])
                row |= {f"d_{m}": sums[j].sum() / n.sum() - rsums[j].sum() / rn.sum(), f"d_{m}_lo": lo, f"d_{m}_hi": hi}
        rows.append(row)
    return pl.DataFrame(rows).sort("place_nll")


def format_intervals(table: pl.DataFrame, metrics: tuple[str, ...] = HEADLINE) -> str:
    """Plain-text table: value (interval), and the paired difference to the reference if present."""
    lines = []
    head = f"{'model':<24}{'n':>9}" + "".join(f"{m:>26}" for m in metrics)
    lines.append(head)
    for r in table.iter_rows(named=True):
        line = f"{r['model']:<24}{r['n']:>9,}"
        for m in metrics:
            line += f"{r[m]:>9.3f} ({r[m + '_lo']:.3f}-{r[m + '_hi']:.3f})"
        lines.append(line)
        if r.get(f"d_{metrics[0]}") is not None:
            diff = f"{'':<24}{'vs ref':>9}"
            for m in metrics:
                diff += f"{r['d_' + m]:>+9.3f} ({r['d_' + m + '_lo']:+.3f},{r['d_' + m + '_hi']:+.3f})"
            lines.append(diff)
    return "\n".join(lines)


def calibration(probs: pl.DataFrame, model: str, edges: tuple[float, ...] = CALIB_EDGES, top_only: bool = False) -> pl.DataFrame:
    """Reliability table: every predicted callout probability of `model` (or only each sample's
    top callout) binned by its value, against how often that callout held the enemy."""
    sub = probs.filter(pl.col("model") == model)
    p = sub["place_probs"].to_numpy()
    y = np.zeros_like(p, dtype=bool)
    y[np.arange(len(p)), sub["place_true"].to_numpy()] = True
    if top_only:
        pick = p.argmax(axis=1)
        p, y = p[np.arange(len(p)), pick][:, None], y[np.arange(len(p)), pick][:, None]
    p, y = p.ravel(), y.ravel()
    b = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    n = np.bincount(b, minlength=len(edges) - 1)
    said = np.bincount(b, p, len(edges) - 1) / np.maximum(n, 1)
    there = np.bincount(b, y, len(edges) - 1) / np.maximum(n, 1)
    labels = [f"{lo:.0%}-{hi:.0%}" for lo, hi in zip(edges[:-1], edges[1:])]
    return pl.DataFrame({"says": labels, "n": n, "mean_said": said, "actually_there": there}).filter(pl.col("n") > 0)


def calibration_error(table: pl.DataFrame) -> float:
    """Expected calibration error: bin-count-weighted mean |said - actually there|."""
    w = table["n"].to_numpy() / table["n"].sum()
    return float((w * np.abs(table["mean_said"].to_numpy() - table["actually_there"].to_numpy())).sum())


def sliced_differences(df: pl.DataFrame, by: str, ref: str, models: list[str]) -> str:
    """Per value of `by`: each model's headline metrics and paired difference to `ref`."""
    out = []
    for (value,), g in df.filter(pl.col("model").is_in([ref, *models])).group_by([by]):
        t = intervals(g, ref)
        out.append(f"-- {by} = {value} ({g.height // t.height:,} samples)\n{format_intervals(t)}")
    return "\n".join(sorted(out))


def report(df: pl.DataFrame, probs: pl.DataFrame, ref: str | None, focus: list[str], by: list[str] | None = None) -> None:
    pl.Config.set_tbl_rows(100)
    pl.Config.set_tbl_cols(20)
    pl.Config.set_tbl_width_chars(250)
    pl.Config.set_float_precision(3)
    mid = df.filter(pl.col("seen_before"))
    focus = [m for m in focus if m in df["model"].unique().to_list()]
    print(f"\n== Mid/late round: enemy seen earlier this round, currently off radar ({mid.height // df['model'].n_unique():,} samples) ==")
    print(summarize(mid))
    print("\n== Headline metrics with round-bootstrap 95% intervals" + (f" (differences vs {ref})" if ref else "") + " ==")
    print(format_intervals(intervals(mid, ref)))
    print("\n== By time since last seen ==")
    print(summarize(mid, ["since_bin"]).filter(pl.col("model").is_in(focus)).sort("since_bin", "model"))
    print("\n== Clutches: one friendly player left ==")
    print(summarize(mid.filter(pl.col("n_friend") == 1), ["n_enemy"]).filter(pl.col("model").is_in(focus)).sort("n_enemy", "model"))
    for col in by or []:
        print(f"\n== By {col} (paired differences vs {ref}) ==")
        print(sliced_differences(mid, col, ref, [m for m in df["model"].unique().to_list() if m != ref]))
    print("\n== Never seen yet this round ==")
    print(summarize(df.filter(~pl.col("seen_before"))))
    if probs.height:
        for model in focus:
            table = calibration(probs, model)
            print(f"\n== Calibration of {model}: every callout probability (ECE {calibration_error(table):.4f}) ==")
            print(table)


def load_samples(out: Path, split: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    probs = out / f"probs_{split}.parquet"
    return pl.read_parquet(out / f"samples_{split}.parquet"), pl.read_parquet(probs) if probs.exists() else pl.DataFrame()


def main() -> None:
    names = [c.name for c in ALL_CONFIGS]
    ap = argparse.ArgumentParser(description="Score belief filters on held-out demos.")
    ap.add_argument("--train-sources", nargs="+", default=["xego"], help="which fitted models to load")
    ap.add_argument("--sources", nargs="+", default=None, help="demo sources to score (default: train sources)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--gamma", type=float, nargs="+", default=[1.0], help="negative-information tempering")
    ap.add_argument("--models", nargs="+", default=None, help=f"filter configs (default: {[c.name for c in DEFAULT_CONFIGS]}; all: {names})")
    ap.add_argument("--set", nargs="+", default=[], metavar="MODULE.CONST=VALUE", help="override module constants in the workers")
    ap.add_argument("--ref", default="ens", help="reference model for paired differences")
    ap.add_argument("--focus", nargs="+", default=["last_seen", "prior_neg", "hmm", "pf", "ens"], help="models in the breakdowns")
    ap.add_argument("--by", nargs="+", default=[], help="extra slices with paired differences, e.g. weapon_seen since_bin")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", type=Path, default=OUTPUT_DIR / "eval")
    ap.add_argument("--report", type=Path, default=None, help="re-print the report of a saved run in this folder")
    args = ap.parse_args()

    if args.report is not None:
        report(*load_samples(args.report, args.split), args.ref, args.focus, args.by)
        return

    models_path = model_dir(args.train_sources)
    refs = list_demos(args.sources or args.train_sources, [args.split])
    chosen = [c for c in ALL_CONFIGS if c.name in args.models] if args.models else list(DEFAULT_CONFIGS)
    if args.models and len(chosen) != len(args.models):
        raise SystemExit(f"Unknown models {set(args.models) - {c.name for c in chosen}}; choose from {names}")
    configs = [c for c in chosen if not c.negative]
    for g in args.gamma:
        configs += [replace(c, name=f"{c.name}@g{g}" if len(args.gamma) > 1 else c.name, gamma=g)
                    for c in chosen if c.negative]  # fmt: skip
    logger.info(f"Scoring {len(refs)} {args.split} demos with {[c.name for c in configs]} {args.set or ''}")

    for var in ("VECLIB_MAXIMUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "POLARS_MAX_THREADS"):
        os.environ.setdefault(var, "1")  # one thread per worker; the workers are the parallelism
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        out = list(pool.map(score_demo, [(r, models_path, tuple(configs), tuple(args.set)) for r in refs]))
    df = with_slices(pl.concat([o[0] for o in out if o[0].height]))
    probs = pl.concat([o[1] for o in out if o[1].height])
    args.out.mkdir(parents=True, exist_ok=True)
    df.write_parquet(args.out / f"samples_{args.split}.parquet")
    probs.write_parquet(args.out / f"probs_{args.split}.parquet")
    summarize(df.filter(pl.col("seen_before"))).write_csv(args.out / f"summary_{args.split}.csv")
    report(df, probs, args.ref, args.focus, args.by)


if __name__ == "__main__":
    main()
