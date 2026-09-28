"""Honest percentages: a post-hoc correction of the callout probabilities.

A filter's callout probabilities p (one per callout, summing to 1) are mapped to

    q_c = exp(g(log p_c)) / sum_k exp(g(log p_k))

where g is increasing, so the order of the callouts never changes: only how confident the model
is. Candidate maps, fitted by maximum likelihood on validation samples:

    power      g(x) = a x                      one parameter (temperature scaling)
    platt      q_c ~ sigmoid(a logit p_c + b)  two parameters
    piecewise  g piecewise linear in log p with its own slope between KNOTS, so the map can sharpen
               near-certain predictions while softening mid-range ones

How wrong the raw percentages are depends on how long ago the enemy was seen: shortly after a
sighting the filters are too unsure, long after it too sure. So each map can also be fitted
separately per group of seconds since last seen (SINCE_GROUPS). Enemies not seen yet this round
are left as they are (no validation samples were fitted for them).

`select` compares the candidates by cross-validation over whole rounds, and the winner is refitted
on all validation samples. It is stored next to the models (calibration.json) and applied by the
filters to the final model's beliefs: each cell keeps its share of its callout.

Usage:
    python -m cspredict.calibrate --run outputs/eval_team2_val --model ens_team_p100c
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import polars as pl
from scipy.optimize import minimize

KNOTS = (0.02, 0.2, 0.6)  # callout probabilities where the piecewise map may change slope
SINCE_GROUPS = (5.0, 20.0)  # seconds since last seen: separate maps for 0-5 s, 5-20 s and 20 s+
P_FLOOR = 1e-6
KINDS = ("power", "platt", "piecewise")


@dataclass
class Calibration:
    kind: str
    params: list[list[float]]  # one parameter list per group of seconds since last seen
    since_edges: list[float] = field(default_factory=list)  # group boundaries; empty = one map for all
    model: str = ""  # filter config it was fitted for
    fitted_on: str = ""
    n: int = 0

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: Path) -> Calibration:
        return cls(**json.loads(path.read_text()))

    def apply(self, p: np.ndarray, since: np.ndarray | None = None) -> np.ndarray:
        """(n, C) callout probabilities -> calibrated (n, C). `since` (n,) is the seconds since
        each enemy was last seen (inf or NaN if never); unseen enemies are left unchanged."""
        if not self.since_edges:
            return _transform(self.kind, np.asarray(self.params[0]), p)
        if since is None:
            raise ValueError("this calibration depends on the time since last seen")
        out = np.array(p, dtype=np.float64, copy=True)
        group = _groups(since, self.since_edges)
        for k, theta in enumerate(self.params):
            sel = group == k
            if sel.any():
                out[sel] = _transform(self.kind, np.asarray(theta), p[sel])
        return out


def _groups(since: np.ndarray, edges: list[float]) -> np.ndarray:
    """Group index per sample; -1 for enemies not seen yet."""
    since = np.asarray(since, dtype=np.float64)
    return np.where(np.isfinite(since), np.searchsorted(np.asarray(edges), since, side="right"), -1)


def _g(kind: str, theta: np.ndarray, p: np.ndarray) -> np.ndarray:
    x = np.log(np.clip(p, P_FLOOR, 1.0))
    if kind == "power":
        return theta[0] * x
    if kind == "platt":
        lp = x - np.log1p(-np.clip(p, P_FLOOR, 1 - 1e-9))  # logit p
        return -np.logaddexp(0.0, -(theta[0] * lp + theta[1]))  # log sigmoid
    if kind == "piecewise":
        knots = np.log(np.asarray(KNOTS))
        g = theta[0] * np.minimum(x, knots[0])
        for k in range(len(knots)):
            hi = knots[k + 1] if k + 1 < len(knots) else 0.0
            g = g + theta[k + 1] * np.clip(x - knots[k], 0.0, hi - knots[k])
        return g
    raise ValueError(kind)


def _transform(kind: str, theta: np.ndarray, p: np.ndarray) -> np.ndarray:
    g = _g(kind, theta, p)
    g = g - g.max(axis=1, keepdims=True)
    q = np.exp(g)
    return q / q.sum(axis=1, keepdims=True)


def _start(kind: str) -> tuple[np.ndarray, list[tuple[float, float]]]:
    if kind == "power":
        return np.array([1.0]), [(0.05, 20.0)]
    if kind == "platt":
        return np.array([1.0, 0.0]), [(0.05, 20.0), (-20.0, 20.0)]
    return np.ones(len(KNOTS) + 1), [(0.0, 20.0)] * (len(KNOTS) + 1)


def log_loss(q: np.ndarray, truth: np.ndarray) -> np.ndarray:
    return -np.log(np.clip(q[np.arange(len(truth)), truth], 1e-9, 1.0))


def fit_params(kind: str, p: np.ndarray, truth: np.ndarray) -> np.ndarray:
    theta0, bounds = _start(kind)
    res = minimize(lambda th: log_loss(_transform(kind, th, p), truth).mean(), theta0, method="L-BFGS-B", bounds=bounds)
    return res.x


def fit(kind: str, p: np.ndarray, truth: np.ndarray, since: np.ndarray, by_since: bool) -> Calibration:
    if not by_since:
        return Calibration(kind, [fit_params(kind, p, truth).tolist()])
    group = _groups(since, list(SINCE_GROUPS))
    params = [fit_params(kind, p[group == k], truth[group == k]).tolist() for k in range(len(SINCE_GROUPS) + 1)]
    return Calibration(kind, params, list(SINCE_GROUPS))


def select(p: np.ndarray, truth: np.ndarray, since: np.ndarray, clusters: np.ndarray, folds: int = 4, seed: int = 0) -> pl.DataFrame:
    """Held-out log-loss of each candidate map, folds made of whole rounds."""
    ids = np.unique(clusters)
    fold_of = dict(zip(ids, np.random.default_rng(seed).permutation(len(ids)) % folds))
    fold = np.array([fold_of[c] for c in clusters])
    rows = [{"kind": "none", "by_since": False, "cv_log_loss": float(log_loss(p, truth).mean())}]
    for kind in KINDS:
        for by_since in (False, True):
            held = np.empty(len(truth))
            for f in range(folds):
                test = fold == f
                cal = fit(kind, p[~test], truth[~test], since[~test], by_since)
                held[test] = log_loss(cal.apply(p[test], since[test]), truth[test])
            rows.append({"kind": kind, "by_since": by_since, "cv_log_loss": float(held.mean())})
    return pl.DataFrame(rows).sort("cv_log_loss")


def arrays(run: Path, split: str, model: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Callout probabilities, true callouts, seconds since last seen and round ids of a saved run."""
    key = ["demo", "round", "friendly", "step", "enemy"]
    probs = pl.read_parquet(run / f"probs_{split}.parquet").filter(pl.col("model") == model)
    since = pl.read_parquet(run / f"samples_{split}.parquet", columns=["model", *key, "since"]).filter(pl.col("model") == model)
    sub = probs.join(since.drop("model"), on=key)
    clusters = sub.select(pl.format("{}|{}", pl.col("demo"), pl.col("round")))[:, 0].to_numpy()
    return sub["place_probs"].to_numpy().astype(np.float64), sub["place_true"].to_numpy(), sub["since"].to_numpy(), clusters


def main() -> None:
    from cspredict.build import model_dir  # build imports this module for Calibration

    ap = argparse.ArgumentParser(description="Fit the callout-probability calibration on a validation run.")
    ap.add_argument("--run", type=Path, required=True, help="evaluate --out folder of a validation run")
    ap.add_argument("--model", required=True, help="filter config to calibrate")
    ap.add_argument("--split", default="val")
    ap.add_argument("--train-sources", nargs="+", default=["hltv", "xego"])
    args = ap.parse_args()

    p, truth, since, clusters = arrays(args.run, args.split, args.model)
    table = select(p, truth, since, clusters)
    print(f"Cross-validated callout log-loss on {len(truth):,} {args.split} samples ({len(np.unique(clusters))} rounds):")
    print(table)
    best = table.filter(pl.col("kind") != "none").row(0, named=True)
    cal = fit(best["kind"], p, truth, since, best["by_since"])
    cal.model, cal.fitted_on, cal.n = args.model, f"{args.run.name} {args.split}", len(truth)
    path = model_dir(args.train_sources) / "calibration.json"
    cal.save(path)
    print(f"{cal.kind} (by time since seen: {bool(cal.since_edges)}) parameters {np.round(cal.params, 3).tolist()} -> {path}")


if __name__ == "__main__":
    main()
