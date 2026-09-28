"""Benchmark a text-based model (TypeSafe's Jev) against the belief filters.

A random sample of held-out moments is drawn from the same population the filters are scored on:
an enemy seen earlier this round who is now off the radar. Each moment is described in words
from the friendly team's point of view (describe.py), and Jev is asked which of Mirage's callouts
that enemy is in. Its probabilities over the callouts are scored exactly like the filters'
beliefs, on the same moments.

Usage:
    python -m cspredict.typesafe_bench --n 3000              # needs TYPESAFE_API_KEY (env or .env)
    python -m cspredict.typesafe_bench --n 3000 --dry-run    # example request and cost, no API calls

Answers are cached in outputs/typesafe/answers.jsonl; re-running only asks what is missing.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import polars as pl
from loguru import logger

from cspredict.build import load_models, model_dir
from cspredict.config import OUTPUT_DIR, ROOT
from cspredict.dataset import DemoRef, list_demos
from cspredict.describe import (
    add_reachable,
    callout_name,
    callout_neighbours,
    callout_run_times,
    callouts,
    describe,
    enemy_labels,
)
from cspredict.filters import DEFAULT_CONFIGS, gather_evidence, run_filter
from cspredict.infostate import episodes

BASELINES = ("last_seen", "prior", "hmm", "pf", "ens")
POPULATION = 102_486  # mid/late-round moments in the HLTV test split (evaluate.py)
PRICE_PER_MTOK = 0.042  # USD per million input tokens (docs.typesafe.ai/models)
EPS = 1e-6

INSTRUCTIONS = (
    "Which callout is enemy {label} in right now? Use where {label} was last seen and how long ago, "
    "what they were doing, the bomb situation, the kill feed, and the callouts the team is watching "
    "(an enemy standing there would probably have been spotted)."
)
# Variant "reach" also lists, per enemy, the callouts they could have run to since last seen:
# a calculation done in code, as TypeSafe's guidance recommends, since Jev is not a calculator.
INSTRUCTIONS_REACH = INSTRUCTIONS[:-1] + (
    ", and the callouts {label} could have run to since then (`could_have_reached` in the `enemies` "
    "entry with id {label})."
)
# Variant "reach_walk" gives walking and running times: in pro clutches the last player alive
# walks (silently, at about half running speed) for about two thirds of their movement.
INSTRUCTIONS_WALK = INSTRUCTIONS[:-1] + (
    ", and how long {label} would need to reach each callout (`could_have_reached` in the `enemies` "
    "entry with id {label}). Players usually shift-walk, which is silent and takes the `walking_s` "
    "time; running takes `running_s` but makes audible footsteps."
)


# ---------------------------------------------------------------------- sampling moments
def _u(seed: int, *key) -> float:
    digest = hashlib.blake2b("|".join(map(str, (seed, *key))).encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") / 2**64


def collect_demo(args: tuple[DemoRef, Path, float, int]) -> list[dict]:
    """Kept moments of one demo with the filters' callout probabilities and a description."""
    ref, models_path, keep, seed = args
    models = load_models(models_path)
    g = models.grid
    onehot = np.zeros((g.n, len(g.places)))
    onehot[np.arange(g.n), g.node_place_idx] = 1.0
    cfgs = [c for c in DEFAULT_CONFIGS if c.name in BASELINES]
    out = []
    for friendly in ("ct", "t"):
        for ep in episodes(g, ref, friendly):
            ever = np.zeros(len(ep.enemy_ids), dtype=bool)
            picks = []
            for s in range(ep.n_steps):
                for e in range(len(ep.enemy_ids)):
                    hidden = ep.enemy_alive[s, e] and not ep.enemy_seen[s, e] and ep.enemy_node[s, e] >= 0
                    if hidden and ever[e] and _u(seed, ref.demo_id, friendly, ep.round_num, s, e) < keep:
                        picks.append((s, e))
                ever |= ep.enemy_seen[s]
            if not picks:
                continue
            ev = gather_evidence(ep, g, models.spot, models.fires)
            wanted = {s for s, _ in picks}
            beliefs = {c.name: {} for c in cfgs}
            for c in cfgs:
                for s, b in run_filter(ep, g, models.motion, c, ev, models.library, models.teams):
                    if s in wanted:
                        beliefs[c.name][s] = b @ onehot
            labels = enemy_labels(ep)
            states = {s: describe(ep, s, models, ev.unseen[s]) for s in wanted}
            for s, e in picks:
                out.append({
                    "demo": ref.demo_id, "friendly": friendly, "round": ep.round_num, "step": s, "enemy": e,
                    "label": labels[e], "u": _u(seed, ref.demo_id, friendly, ep.round_num, s, e),
                    "t_rel": float(ep.t_rel[s]), "n_friend": int(ep.obs_alive[s].sum()),
                    "n_enemy": int(ep.enemy_alive[s].sum()),
                    "truth": int(g.node_place_idx[ep.enemy_node[s, e]]),
                    "probs": {name: beliefs[name][s][e].tolist() for name in beliefs},
                    "state": states[s],
                })  # fmt: skip
    return out


def sample_moments(refs: list[DemoRef], models_path: Path, n: int, seed: int, workers: int) -> list[dict]:
    """Exactly n moments, uniform over the population, reproducible for a given seed."""
    keep = min(1.0, 1.25 * n / POPULATION)  # oversample a little, then keep the n smallest draws
    with ProcessPoolExecutor(max_workers=workers) as pool:
        moments = [m for part in pool.map(collect_demo, [(r, models_path, keep, seed) for r in refs]) for m in part]
    moments.sort(key=lambda m: m["u"])
    if len(moments) < n:
        logger.warning(f"Only {len(moments)} moments available (asked for {n})")
    return moments[:n]


# ---------------------------------------------------------------------- asking the model
def group_key(m: dict) -> str:
    return f"{m['demo']}|{m['friendly']}|{m['round']}|{m['step']}"


def build_requests(moments: list[dict], grid, variant: str = "base") -> dict[str, dict]:
    """One request per moment in time, with one Choice question per enemy asked about."""
    neighbours = callout_neighbours(grid)
    criteria = {c: f"Connects to: {', '.join(neighbours.get(c, [])) or 'nothing listed'}." for c in callouts(grid)}
    instructions = {"reach": INSTRUCTIONS_REACH, "reach_walk": INSTRUCTIONS_WALK}.get(variant, INSTRUCTIONS)
    names, seconds = callout_run_times(grid) if variant != "base" else (None, None)
    requests: dict[str, dict] = {}
    for m in moments:
        if group_key(m) not in requests:
            state = json.loads(json.dumps(m["state"]))  # copy, so variants never alter the saved moments
            if variant != "base":
                add_reachable(state, names, seconds, walking=variant == "reach_walk")
            requests[group_key(m)] = {"state": state, "questions": {}}
        requests[group_key(m)]["questions"][m["label"]] = {
            "type": "choice", "instructions": instructions.format(label=m["label"]), "criteria": criteria,
        }  # fmt: skip
    return requests


def _api_key() -> str | None:
    if os.environ.get("TYPESAFE_API_KEY"):
        return os.environ["TYPESAFE_API_KEY"]
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            name, _, value = line.partition("=")
            if name.strip() == "TYPESAFE_API_KEY":
                return value.strip().strip("'\"")
    return None


def _load_cache(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    return {row["key"]: row for row in map(json.loads, path.read_text().splitlines()) if row.get("answers")}


async def _ask(requests: dict[str, dict], cache_path: Path, concurrency: int, per_second: float) -> None:
    from typesafe_sdk import AsyncTypeSafeClient, Choice, RetryPolicy

    todo = [k for k in requests if k not in _load_cache(cache_path)]
    logger.info(f"{len(requests) - len(todo)} requests cached, {len(todo)} to send")
    if not todo:
        return
    gate = asyncio.Semaphore(concurrency)
    spacing, next_start = 1.0 / per_second, [time.monotonic()]
    done, failed = 0, 0
    async with AsyncTypeSafeClient(api_key=_api_key(), retry=RetryPolicy(max_retries=5), timeout=60.0) as client:
        with cache_path.open("a") as cache:

            async def one(key: str) -> None:
                nonlocal done, failed
                async with gate:
                    wait = next_start[0] - time.monotonic()
                    next_start[0] = max(next_start[0], time.monotonic()) + spacing
                    if wait > 0:
                        await asyncio.sleep(wait)
                    req = requests[key]
                    questions = {
                        label: Choice(instructions=q["instructions"], criteria=q["criteria"])
                        for label, q in req["questions"].items()
                    }
                    try:
                        resp = await client.system_one(req["state"], questions)
                    except Exception as exc:
                        if done == 0 and failed == 0 and key == todo[0]:
                            raise  # the first request failing means the setup is wrong; stop here
                        failed += 1  # later failures are skipped and retried on the next run
                        logger.warning(f"Request failed ({type(exc).__name__}: {exc}); will retry next run")
                        return
                    answers = {
                        label: {"probabilities": a.probabilities, "choice": a.choice, "confidence": a.confidence}
                        for label, a in resp.choices.items()
                    }
                    usage = {"input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens}
                    cache.write(json.dumps({"key": key, "model": resp.model, "answers": answers, "usage": usage}) + "\n")
                    cache.flush()
                    done += 1
                    if done % 250 == 0:
                        logger.info(f"{done}/{len(todo)} answered")

            await one(todo[0])  # alone first, so a bad key or request shape fails fast
            await asyncio.gather(*(one(k) for k in todo[1:]))
    logger.info(f"Answered {done}, failed {failed}")


# ---------------------------------------------------------------------- scoring
def _metrics(p: np.ndarray, truth: np.ndarray) -> dict[str, np.ndarray]:
    """Per-moment top-1 / top-3 hits and log-loss for callout distributions p (n, K)."""
    p = np.clip(p, EPS, None)
    p = p / p.sum(axis=1, keepdims=True)
    pt = p[np.arange(len(truth)), truth]
    rank = (p >= pt[:, None]).sum(axis=1) - 1  # ties count against the model
    return {"top1": (rank < 1).astype(float), "top3": (rank < 3).astype(float), "logloss": -np.log(pt)}


def _cluster_ci(values: np.ndarray, clusters: np.ndarray, reps: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95% bootstrap interval of the mean, resampling whole rounds (moments within a round are correlated)."""
    rng = np.random.default_rng(seed)
    ids, inv = np.unique(clusters, return_inverse=True)
    sums, counts = np.bincount(inv, weights=values), np.bincount(inv).astype(float)
    draws = rng.integers(0, len(ids), size=(reps, len(ids)))
    means = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def score(moments: list[dict], cache: dict[str, dict], places: list[str]) -> tuple[pl.DataFrame, dict]:
    answered = [m for m in moments if m["label"] in cache.get(group_key(m), {}).get("answers", {})]
    readable = [callout_name(p) for p in places]
    truth = np.array([m["truth"] for m in answered])
    clusters = np.array([f"{m['demo']}|{m['friendly']}|{m['round']}" for m in answered])
    probs = {name: np.array([m["probs"][name] for m in answered]) for name in BASELINES}
    ts = cache  # answers keyed by moment group
    probs["typesafe"] = np.array(
        [[ts[group_key(m)]["answers"][m["label"]]["probabilities"].get(c, 0.0) for c in readable] for m in answered]
    )
    probs["ens+typesafe"] = 0.5 * probs["ens"] + 0.5 * probs["typesafe"] / probs["typesafe"].sum(axis=1, keepdims=True)

    rows, per_moment = [], {}
    for name, p in probs.items():
        met = _metrics(p, truth)
        per_moment[name] = met
        row = {"model": name, "n": len(truth)}
        for k, v in met.items():
            lo, hi = _cluster_ci(v, clusters)
            row |= {k: float(v.mean()), f"{k}_lo": lo, f"{k}_hi": hi}
        rows.append(row)
    table = pl.DataFrame(rows).sort("logloss")
    diff = per_moment["ens"]["top1"] - per_moment["typesafe"]["top1"]
    extras = {"ens_minus_typesafe_top1": (float(diff.mean()), *_cluster_ci(diff, clusters))}

    p = probs["typesafe"] / probs["typesafe"].sum(axis=1, keepdims=True)
    y = np.zeros_like(p)
    y[np.arange(len(truth)), truth] = 1
    edges = [0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0001]
    calib = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (p >= lo) & (p < hi)
        if sel.sum():
            calib.append({"says": f"{lo:.0%}-{min(hi, 1):.0%}", "n": int(sel.sum()),
                          "actually_there": float(y[sel].mean()), "mean_said": float(p[sel].mean())})  # fmt: skip
    extras["typesafe_calibration"] = calib
    extras["tokens"] = int(sum(r["usage"]["input_tokens"] or 0 for r in cache.values()))
    return table, extras


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark TypeSafe Jev against the belief filters.")
    ap.add_argument("--n", type=int, default=3000, help="moments to ask about")
    ap.add_argument("--train-sources", nargs="+", default=["hltv", "xego"])
    ap.add_argument("--sources", nargs="+", default=["hltv"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--concurrency", type=int, default=8, help="requests in flight")
    ap.add_argument("--per-second", type=float, default=15.0, help="request starts per second (limit: 20)")
    ap.add_argument("--variant", default="base", choices=["base", "reach", "reach_walk"],
                    help="reach: also list the callouts each enemy could have run to since last seen; "
                         "reach_walk: the same with walking and running times")
    ap.add_argument("--dry-run", action="store_true", help="show an example request and the cost; send nothing")
    ap.add_argument("--out", type=Path, default=OUTPUT_DIR / "typesafe")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    moments_path = args.out / f"moments_{args.split}_seed{args.seed}_n{args.n}.json"
    models_path = model_dir(args.train_sources)
    if moments_path.exists():
        moments = json.loads(moments_path.read_text())
    else:
        refs = list_demos(args.sources, [args.split])
        logger.info(f"Sampling {args.n} moments from {len(refs)} {args.split} demos and running the filters")
        moments = sample_moments(refs, models_path, args.n, args.seed, args.workers)
        moments_path.write_text(json.dumps(moments))
    grid = load_models(models_path).grid
    requests = build_requests(moments, grid, args.variant)
    chars = sum(len(json.dumps(r)) for r in requests.values())
    logger.info(
        f"{len(moments)} moments in {len(requests)} requests, ~{chars / 4 / 1e6:.2f}M input tokens "
        f"(~${chars / 4 / 1e6 * PRICE_PER_MTOK:.2f})"
    )

    if args.dry_run:
        key, req = next(iter(requests.items()))
        print(json.dumps({"state": req["state"], "questions": req["questions"]}, indent=1)[:6000])
        return
    if _api_key() is None:
        raise SystemExit("Set TYPESAFE_API_KEY (environment or a line in .env) to call the API")

    suffix = "" if args.variant == "base" else f"_{args.variant}"
    cache_path = args.out / f"answers{suffix}.jsonl"
    asyncio.run(_ask(requests, cache_path, args.concurrency, args.per_second))
    table, extras = score(moments, _load_cache(cache_path), list(grid.places))

    pl.Config.set_tbl_rows(20)
    pl.Config.set_tbl_cols(12)
    pl.Config.set_tbl_width_chars(200)
    pl.Config.set_float_precision(3)
    print(f"\n== {table['n'][0]} held-out moments; 95% intervals resample whole rounds ==")
    print(table.select("model", "top1", "top1_lo", "top1_hi", "top3", "logloss", "logloss_lo", "logloss_hi"))
    d, lo, hi = extras["ens_minus_typesafe_top1"]
    print(f"\nens minus typesafe, callout top-1: {d:+.3f} (95% {lo:+.3f} to {hi:+.3f})")
    print("\nTypeSafe honesty: when it says X, how often is the enemy there?")
    print(pl.DataFrame(extras["typesafe_calibration"]))
    print(f"\nInput tokens used: {extras['tokens']:,} (~${extras['tokens'] / 1e6 * PRICE_PER_MTOK:.2f})")
    table.write_csv(args.out / f"results{suffix}.csv")
    (args.out / f"extras{suffix}.json").write_text(json.dumps(extras, indent=2))


if __name__ == "__main__":
    main()
