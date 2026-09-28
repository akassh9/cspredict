"""Jev against OpenAI's GPT-6 Luna, asked the same questions (protocol: docs/frontier_protocol.md).

The requests are the ones Jev answered in the Jev lab (jev_lab.build). Each is shown to GPT with a
fixed wrapper that restates TypeSafe's definitions of its two question types, and GPT answers in a
strict JSON schema with a probability for every option. GPT's answers are cached in the same shape
as Jev's, so jev_lab.jev_probs turns both into callout probabilities, and both are calibrated,
blended and scored the same way. Ties get split credit (docs/frontier_protocol.md#scoring).

Usage (OPENAI_API_KEY in the environment or in .env; the Jev timing also needs TYPESAFE_API_KEY):
    python -m cspredict.frontier_bench --dry-run                  # the wrapper, an example request and schema
    python -m cspredict.frontier_bench --pilot 100                # first 100 validation moments: cost, tokens, speed
    python -m cspredict.frontier_bench --split val --budget 20    # full runs, each with a spending cap in USD
    python -m cspredict.frontier_bench --split test --budget 30
    python -m cspredict.frontier_bench --split test --no-ask      # only score what is cached

Answers are cached in outputs/frontier/answers_<variant>.jsonl, failures included (they are retried
on the next run); re-running only sends what is missing.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path

import numpy as np
import polars as pl
from loguru import logger

from cspredict.build import load_models, model_dir
from cspredict.calibrate import Calibration, fit
from cspredict.config import OUTPUT_DIR, ROOT
from cspredict.describe import callouts
from cspredict.jev_lab import DEFAULT_N, HORIZONS, _softmax, build, features, jev_probs, load_moments, movement_rates, request_key
from cspredict.jev_lab import MODEL as JEV_MODEL
from cspredict.jev_lab import OUT as JEV_OUT
from cspredict.jev_lab import cache_path as jev_cache_path
from cspredict.typesafe_bench import BASELINES, EPS, PRICE_PER_MTOK, _api_key, _cluster_ci, _load_cache, _metrics

OUT = OUTPUT_DIR / "frontier"
MODEL = "gpt-6-luna"
EFFORT = "medium"  # OpenAI's default for GPT-6, set explicitly so a change of default can't alter the run
VARIANTS = ("base", "split_rates")
MAX_OUTPUT_TOKENS = 25_000  # OpenAI: reserve at least 25k tokens for reasoning and the answer
# USD per million tokens, standard tier, short context (developers.openai.com/api/docs/pricing,
# 2026-09-28). Reasoning tokens are billed as output. Flex and Batch are half price.
PRICES = {"gpt-6-luna": {"input": 0.10, "cached": 0.01, "cache_write": 0.125, "output": 0.50}}
TIER_FACTOR = {"default": 1.0, "flex": 0.5}
BLEND_REF = {"base": "evidence formula (no Jev)", "split_rates": "evidence formula + movement rates (no Jev)"}

# TypeSafe's definitions of its question types (docs.typesafe.ai/primitives/choice and /noul).
WRAPPER = """You answer typed questions about a state. The input is JSON with a `state` and `questions`, a map from each question's name to the question. Each question has a `type`:
- "choice": `instructions` is the question you answer. `criteria` holds the answer options, as a map: each key is an option name and each value is a description of that option. Give the probability that each option is the right answer. The probabilities sum to 1.
- "noul": `instructions` is the yes/no question you answer, or a statement for you to judge. `criteria`, if present, describes what counts as true and as false. Give the probability that the answer is yes, where 0 means no and 1 means yes.

Answer every question in the JSON format requested."""


# ---------------------------------------------------------------------- the wrapper
def gpt_input(req: dict) -> str:
    """The request exactly as Jev receives it: the state and the typed questions."""
    return json.dumps({"state": req["state"], "questions": req["questions"]}, ensure_ascii=False)


def answer_schema(questions: dict) -> dict:
    """Strict JSON schema: a probability per option for each choice question, one for each yes/no."""
    props = {}
    for name, q in questions.items():
        if q["type"] == "noul":
            props[name] = {"type": "number", "description": "Probability that the answer is yes: 0 means no, 1 means yes."}
        else:
            options = list(q["criteria"])
            props[name] = {
                "type": "object",
                "description": "Probability that each option is the right answer; they sum to 1.",
                "properties": {o: {"type": "number"} for o in options},
                "required": options,
                "additionalProperties": False,
            }  # fmt: skip
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


def parse_answers(text: str, questions: dict) -> dict:
    """GPT's JSON -> answers in the shape Jev's are cached in: negatives become 0, each choice is
    rescaled to sum to 1 and each yes/no clipped to 0-1. Raises ValueError if unusable."""
    raw = json.loads(text)
    out = {}
    for name, q in questions.items():
        if q["type"] == "noul":
            yes = float(raw[name])
            if not np.isfinite(yes):
                raise ValueError(f"no probability given for question {name}")
            out[name] = {"noul": float(np.clip(yes, 0.0, 1.0))}
            continue
        p = {o: max(float(raw[name][o]), 0.0) for o in q["criteria"]}
        total = sum(p.values())
        if not np.isfinite(total) or total <= 0:
            raise ValueError(f"no probability given for question {name}")
        out[name] = {"probabilities": {o: v / total for o, v in p.items()}}
    return out


def usage_of(resp) -> dict:
    u = resp.usage
    din, dout = u.input_tokens_details, u.output_tokens_details
    return {
        "input_tokens": u.input_tokens,
        "cached_tokens": getattr(din, "cached_tokens", 0) or 0,
        "cache_write_tokens": getattr(din, "cache_write_tokens", 0) or 0,
        "output_tokens": u.output_tokens,
        "reasoning_tokens": getattr(dout, "reasoning_tokens", 0) or 0,
    }


def cost_usd(usage: dict, model: str = MODEL, tier: str = "default") -> float:
    """List price of one request from its usage counts (cached and cache-write tokens are part of the input)."""
    pr = PRICES[model]
    cached, write = usage.get("cached_tokens", 0), usage.get("cache_write_tokens", 0)
    fresh = max(usage["input_tokens"] - cached - write, 0)
    usd = fresh * pr["input"] + cached * pr["cached"] + write * pr["cache_write"] + usage["output_tokens"] * pr["output"]
    return TIER_FACTOR.get(tier, 1.0) * usd / 1e6


def _env_key(name: str) -> str | None:
    if os.environ.get(name):
        return os.environ[name]
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            key, _, value = line.partition("=")
            if key.strip() == name:
                return value.strip().strip("'\"") or None
    return None


def answers_path(variant: str) -> Path:
    return OUT / f"answers_{variant}.jsonl"


def _rows(path: Path) -> list[dict]:
    """Every cached row, failures included."""
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


# ---------------------------------------------------------------------- asking the models
def _refusal(resp) -> str | None:
    for item in resp.output or []:
        for part in getattr(item, "content", None) or []:
            if getattr(part, "type", None) == "refusal":
                return part.refusal
    return None


async def ask_gpt(requests: dict[str, dict], path: Path, *, model: str = MODEL, effort: str = EFFORT,
                  tier: str = "default", concurrency: int = 8, budget_usd: float = 2.0) -> float:
    """Send the requests not answered yet and append GPT's answers (or failures) to `path`. Stops
    starting new requests once this run has spent `budget_usd`. Returns what this run spent."""
    from openai import APIError, AsyncOpenAI

    todo = [k for k in requests if k not in _load_cache(path)]
    logger.info(f"{path.stem}: {len(requests) - len(todo)} requests answered already, {len(todo)} to send")
    if not todo:
        return 0.0
    gate = asyncio.Semaphore(concurrency)
    spent, done, failed, skipped = 0.0, 0, 0, 0
    async with AsyncOpenAI(api_key=_env_key("OPENAI_API_KEY"), max_retries=8, timeout=600.0) as client:
        with path.open("a") as out:

            async def one(key: str) -> None:
                nonlocal spent, done, failed, skipped
                async with gate:
                    if spent >= budget_usd:
                        skipped += 1
                        return
                    req = requests[key]
                    t0 = time.monotonic()
                    try:
                        resp = await client.responses.create(
                            model=model, instructions=WRAPPER, input=gpt_input(req), reasoning={"effort": effort},
                            text={"format": {"type": "json_schema", "name": "answers", "strict": True,
                                             "schema": answer_schema(req["questions"])}},
                            max_output_tokens=MAX_OUTPUT_TOKENS, store=False, service_tier=tier,
                            prompt_cache_options={"mode": "explicit"},  # no breakpoints: no caching, no cache-write fee
                        )  # fmt: skip
                    except APIError as exc:
                        if done == 0 and failed == 0 and key == todo[0]:
                            raise  # the first request failing means the setup is wrong; stop here
                        failed += 1  # nothing is written, so it is retried on the next run
                        logger.warning(f"Request failed ({type(exc).__name__}: {exc}); will retry next run")
                        return
                    usage = usage_of(resp)
                    row = {"key": key, "model": resp.model, "effort": effort, "service_tier": resp.service_tier,
                           "usage": usage, "cost_usd": cost_usd(usage, model, resp.service_tier or tier),
                           "latency_s": round(time.monotonic() - t0, 3)}  # fmt: skip
                    spent += row["cost_usd"]
                    try:
                        if (refusal := _refusal(resp)) is not None:
                            raise ValueError(f"refusal: {refusal}")
                        if resp.status != "completed":
                            raise ValueError(f"status {resp.status} ({getattr(resp.incomplete_details, 'reason', None)})")
                        row["answers"] = parse_answers(resp.output_text, req["questions"])
                        done += 1
                    except (ValueError, KeyError, TypeError) as exc:
                        row["error"] = f"{type(exc).__name__}: {exc}"[:300]
                        failed += 1
                    out.write(json.dumps(row) + "\n")
                    out.flush()
                    if (done + failed) % 250 == 0:
                        logger.info(f"{done + failed}/{len(todo)} answered, ${spent:.2f} spent")

            await one(todo[0])  # alone first, so a bad key or request shape fails fast
            await asyncio.gather(*(one(k) for k in todo[1:]))
    logger.info(f"{path.stem}: answered {done}, failed {failed}, not sent (budget) {skipped}; this run spent ${spent:.3f}")
    return spent


async def time_jev(requests: dict[str, dict], concurrency: int = 8, per_second: float = 15.0) -> list[float]:
    """Seconds per request for Jev on the same requests. Only the timing is kept: Jev's cached
    answers from the Jev lab stand."""
    from typesafe_sdk import AsyncTypeSafeClient, Choice, Noul, RetryPolicy

    gate = asyncio.Semaphore(concurrency)
    spacing, next_start, seconds = 1.0 / per_second, [time.monotonic()], []
    async with AsyncTypeSafeClient(api_key=_api_key(), retry=RetryPolicy(max_retries=5), timeout=60.0) as client:

        async def one(req: dict) -> None:
            questions = {label: (Noul if q.get("type") == "noul" else Choice)(instructions=q["instructions"], criteria=q["criteria"])
                         for label, q in req["questions"].items()}  # fmt: skip
            async with gate:
                wait = next_start[0] - time.monotonic()
                next_start[0] = max(next_start[0], time.monotonic()) + spacing
                if wait > 0:
                    await asyncio.sleep(wait)
                t0 = time.monotonic()
                await client.system_one(req["state"], questions, model=JEV_MODEL)
                seconds.append(time.monotonic() - t0)

        await asyncio.gather(*(one(r) for r in requests.values()))
    return seconds


# ---------------------------------------------------------------------- answers -> probabilities, scoring
def gpt_probs(moments: list[dict], cache: dict[str, dict], variant: str, names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """(n, C) GPT's callout probabilities, and which moments had no usable answer (scored as uniform)."""
    p = jev_probs(moments, cache, variant, names)
    missing = np.isnan(p).any(axis=1)
    p[missing] = 1.0 / len(names)
    return p, missing


def calibrated_gpt(moments: list[dict], grid, variant: str) -> np.ndarray | None:
    """(n, C) GPT's calibrated callout probabilities for these moments, or None before the
    validation run has fitted the calibration."""
    cal_path = OUT / f"calibration_{variant}.json"
    if not cal_path.exists() or not answers_path(variant).exists():
        return None
    since = np.array([m["evidence"]["seconds_since_seen"] for m in moments], dtype=float)
    p, _ = gpt_probs(moments, _load_cache(answers_path(variant)), variant, callouts(grid))
    return Calibration.load(cal_path).apply(np.clip(p, EPS, None), since)


metrics = _metrics  # ties get split credit by default (typesafe_bench._metrics)


def score(named: dict[str, np.ndarray], truth: np.ndarray, clusters: np.ndarray, last_idx: np.ndarray, ties: str) -> pl.DataFrame:
    rows = []
    for name, p in named.items():
        met = metrics(p, truth, ties)
        row = {"model": name, "n": len(truth), "picks_last_seen": float((p.argmax(axis=1) == last_idx).mean())}
        for k, v in met.items():
            lo, hi = _cluster_ci(v, clusters)
            row |= {k: float(v.mean()), f"{k}_lo": lo, f"{k}_hi": hi}
        rows.append(row)
    return pl.DataFrame(rows).sort("logloss")


def _cost_summary(rows: list[dict]) -> dict:
    """Totals over rows (failures included, since they were paid for)."""
    if not rows:
        return {}
    u = [r["usage"] for r in rows]
    reasoning = np.array([x["reasoning_tokens"] for x in u], dtype=float)
    latency = np.array([r["latency_s"] for r in rows if r.get("service_tier") in (None, "default")], dtype=float)
    return {
        "requests": len(rows),
        "failures": sum("answers" not in r for r in rows),
        "cost_usd": float(sum(r["cost_usd"] for r in rows)),
        "input_tokens_mean": float(np.mean([x["input_tokens"] for x in u])),
        "output_tokens_mean": float(np.mean([x["output_tokens"] for x in u])),
        "reasoning_tokens_mean": float(reasoning.mean()),
        "reasoning_tokens_p90": float(np.percentile(reasoning, 90)),
        "reasoning_tokens_max": float(reasoning.max()),
        "cached_tokens": int(sum(x["cached_tokens"] for x in u)),
        "cache_write_tokens": int(sum(x["cache_write_tokens"] for x in u)),
        "latency_median_s": float(np.median(latency)) if len(latency) else None,
        "latency_p90_s": float(np.percentile(latency, 90)) if len(latency) else None,
    }


# ---------------------------------------------------------------------- pilot
def pilot(n: int, budget: float, concurrency: int, workers: int) -> None:
    """The first n validation moments (a uniform random subset), both variants: tokens, cost,
    speed and failures, and the full run's projected cost. Accuracy on n moments is a sanity check only."""
    grid = load_models(model_dir(["hltv", "xego"])).grid
    names = callouts(grid)
    full = {s: load_moments(s, DEFAULT_N[s], 0, workers) for s in ("val", "test")}
    moments = full["val"][:n]
    OUT.mkdir(parents=True, exist_ok=True)
    summary, spent = {"moments": n, "model": MODEL, "effort": EFFORT, "variants": {}}, 0.0
    for v in VARIANTS:
        # Built from all validation moments, then subset, so a pilot request equals the full run's.
        every = build(full["val"], grid, v)
        keys = list(dict.fromkeys(request_key(m, v) for m in moments))
        reqs = {k: every[k] for k in keys}
        spent += asyncio.run(ask_gpt(reqs, answers_path(v), concurrency=concurrency, budget_usd=budget - spent))
        rows = [r for r in _rows(answers_path(v)) if r["key"] in reqs]
        if not rows:
            logger.warning(f"{v}: no GPT answers within the budget; skipped")
            continue
        s = _cost_summary(rows)
        jev_rows = _load_cache(jev_cache_path(v))
        jev_seconds = asyncio.run(time_jev(reqs, concurrency))
        s["jev_latency_median_s"], s["jev_latency_p90_s"] = float(np.median(jev_seconds)), float(np.percentile(jev_seconds, 90))
        s["jev_cost_usd"] = sum(jev_rows[k]["usage"]["input_tokens"] for k in reqs) * PRICE_PER_MTOK / 1e6
        n_full = len(every) + len(build(full["test"], grid, v))
        per_request = s["cost_usd"] / max(s["requests"], 1)
        s["full_run_requests"] = n_full
        s["full_run_cost_usd"] = {"standard": per_request * n_full, "flex_or_batch": 0.5 * per_request * n_full}
        s["full_run_jev_cost_usd"] = s["jev_cost_usd"] / len(reqs) * n_full

        # Sanity only: are the answers usable and not absurd?
        truth = np.array([m["truth"] for m in moments])
        last_idx = np.array([names.index(m["evidence"]["last_seen_at"]) for m in moments])
        gp, missing = gpt_probs(moments, _load_cache(answers_path(v)), v, names)
        jp = jev_probs(moments, jev_rows, v, names)
        s["sanity"] = {
            "unusable": int(missing.sum()),
            "gpt_top1": float(metrics(gp, truth)["top1"].mean()), "jev_top1": float(metrics(jp, truth)["top1"].mean()),
            "gpt_picks_last_seen": float((gp.argmax(axis=1) == last_idx).mean()),
            "jev_picks_last_seen": float((jp.argmax(axis=1) == last_idx).mean()),
            "gpt_top_tied": float(np.mean([np.sort(q)[-1] == np.sort(q)[-2] for q in gp])),
        }  # fmt: skip
        summary["variants"][v] = s
    summary["pilot_cost_usd"] = sum(s["cost_usd"] for s in summary["variants"].values())
    summary["full_run_cost_usd"] = {
        t: sum(s["full_run_cost_usd"][t] for s in summary["variants"].values()) for t in ("standard", "flex_or_batch")
    }
    (OUT / "pilot_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


# ---------------------------------------------------------------------- full runs and the report
def report(split: str, variants: list[str], workers: int) -> None:
    moments = load_moments(split, DEFAULT_N[split], 0, workers)
    grid = load_models(model_dir(["hltv", "xego"])).grid
    names = callouts(grid)
    truth = np.array([m["truth"] for m in moments])
    clusters = np.array([f"{m['demo']}|{m['round']}" for m in moments])
    since = np.array([m["evidence"]["seconds_since_seen"] for m in moments], dtype=float)
    last_idx = np.array([names.index(m["evidence"]["last_seen_at"]) for m in moments])

    named = {b: np.array([m["probs"][b] for m in moments]) for b in BASELINES}
    w = np.array(list(json.loads((JEV_OUT / "formula_weights.json").read_text()).values()))
    wr = np.array(list(json.loads((JEV_OUT / "formula_rates_weights.json").read_text()).values()))
    named[BLEND_REF["base"]] = _softmax(features(moments, grid), w)
    named[BLEND_REF["split_rates"]] = _softmax(features(moments, grid, movement_rates(grid)), wr)
    blend_path, costs, unusable = OUT / "blend_weights.json", {}, {}
    blends = json.loads(blend_path.read_text()) if blend_path.exists() else {}
    for v in variants:
        jp = jev_probs(moments, _load_cache(jev_cache_path(v)), v, names)
        named[f"jev {v}"] = jp
        named[f"jev {v} + calibration"] = Calibration.load(JEV_OUT / f"calibration_{v}.json").apply(np.clip(jp, EPS, None), since)
        gp, missing = gpt_probs(moments, _load_cache(answers_path(v)), v, names)
        unusable[v] = int(missing.sum())
        named[f"gpt {v}"] = gp
        cal_path = OUT / f"calibration_{v}.json"
        if split == "val":  # fitted where nothing is tuned but these maps; applied on test
            fit("platt", np.clip(gp, EPS, None), truth, since, by_since=True).save(cal_path)
        q = Calibration.load(cal_path).apply(np.clip(gp, EPS, None), since)
        named[f"gpt {v} + calibration"] = q
        ref = named[BLEND_REF[v]]
        if split == "val":
            grid_w = np.linspace(0, 1, 21)
            ll = [-np.log(np.clip((x * q + (1 - x) * ref)[np.arange(len(truth)), truth], 1e-9, 1)).mean() for x in grid_w]
            blends[v] = float(grid_w[int(np.argmin(ll))])
        if v in blends:
            named[f"gpt {v} + calibration, blended ({blends[v]:.2f}) with formula"] = blends[v] * q + (1 - blends[v]) * ref
        keys = {request_key(m, v) for m in moments}
        costs[v] = _cost_summary([r for r in _rows(answers_path(v)) if r["key"] in keys])
    if split == "val":
        blend_path.write_text(json.dumps(blends, indent=2))

    pl.Config.set_tbl_rows(40)
    pl.Config.set_tbl_width_chars(220)
    pl.Config.set_float_precision(3)
    cols = ("model", "top1", "top1_lo", "top1_hi", "top3", "logloss", "logloss_lo", "logloss_hi", "picks_last_seen")
    table = score(named, truth, clusters, last_idx, "split")
    print(f"\n== {split}: {len(moments):,} moments; ties get split credit; 95% intervals resample whole rounds ==")
    print(table.select(cols))
    table.write_csv(OUT / f"results_{split}.csv")
    old = score(named, truth, clusters, last_idx, "miss")
    old.write_csv(OUT / f"results_{split}_ties_miss.csv")
    print("\nWith ties counted as misses (the rule of evaluate.py), to show how much the rule matters:")
    print(old.select("model", "top1", "top3", "logloss"))

    print("\nPaired differences (same moments; ties get split credit; 95% intervals resample whole rounds):")
    pairs = []
    for v in variants:
        g = f"gpt {v} + calibration"
        pairs += [(g, f"jev {v} + calibration"), (g, BLEND_REF[v]), (g, "last_seen"), (g, "ens")]
        blended = [k for k in named if k.startswith(f"{g}, blended")]
        pairs += [(blended[0], BLEND_REF[v])] if blended else []
    for a, b in pairs:
        ma, mb = metrics(named[a], truth), metrics(named[b], truth)
        for k in ("top1", "logloss"):
            d = ma[k] - mb[k]
            lo, hi = _cluster_ci(d, clusters)
            print(f"  {a}  minus  {b}: {k} {d.mean():+.3f} ({lo:+.3f} to {hi:+.3f})")

    print("\nRight callout first, by seconds since last seen (calibrated; ties get split credit):")
    rows = []
    for lo, hi, label in HORIZONS:
        sel = (since >= lo) & (since < hi)
        row = {"since": label, "n": int(sel.sum())}
        for name in ("ens", "last_seen", *(f"{who} {v} + calibration" for v in variants for who in ("gpt", "jev"))):
            row[name] = float(metrics(named[name][sel], truth[sel])["top1"].mean())
        rows.append(row)
    print(pl.DataFrame(rows))

    print(f"\nUnusable GPT answers (scored as uniform): {unusable}")
    for v, c in costs.items():
        if c:
            print(f"{v}: {c['requests']} GPT requests, ${c['cost_usd']:.2f}, reasoning {c['reasoning_tokens_mean']:.0f} tokens "
                  f"per request (p90 {c['reasoning_tokens_p90']:.0f}), median {c['latency_median_s']}s")
    (OUT / f"extras_{split}.json").write_text(json.dumps({"unusable": unusable, "costs": costs, "blend_weights": blends}, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description="Jev against GPT-6 Luna, asked the same questions (docs/frontier_protocol.md).")
    ap.add_argument("--pilot", type=int, default=None, help="only the first N validation moments: cost, tokens, speed")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=VARIANTS)
    ap.add_argument("--budget", type=float, default=2.0, help="USD this run may spend on GPT (stops starting requests)")
    ap.add_argument("--tier", default="default", choices=["default", "flex"], help="flex: half price, slower (answers unchanged)")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true", help="print the wrapper, an example request and its schema; send nothing")
    ap.add_argument("--no-ask", action="store_true", help="only score what is cached")
    args = ap.parse_args()

    if not args.dry_run and not args.no_ask and _env_key("OPENAI_API_KEY") is None:
        raise SystemExit("Add a line OPENAI_API_KEY=... to .env (git-ignored), or set it in the environment")
    if args.pilot:
        pilot(args.pilot, args.budget, args.concurrency, args.workers)
        return
    moments = load_moments(args.split, DEFAULT_N[args.split], 0, args.workers)
    grid = load_models(model_dir(["hltv", "xego"])).grid
    OUT.mkdir(parents=True, exist_ok=True)
    spent = 0.0
    for v in args.variants:
        requests = build(moments, grid, v)
        chars = sum(len(gpt_input(r)) for r in requests.values()) + len(WRAPPER) * len(requests)
        logger.info(f"{v}: {len(requests)} requests, ~{chars / 4 / 1e6:.2f}M input tokens (~${chars / 4 / 1e6 * PRICES[MODEL]['input']:.2f} before output)")
        if args.dry_run:
            key, req = next(iter(requests.items()))
            print(f"--- instructions\n{WRAPPER}\n--- input (first 3000 characters)\n{gpt_input(req)[:3000]}")
            print(f"--- schema\n{json.dumps(answer_schema(req['questions']))[:1500]}")
            continue
        if not args.no_ask:
            spent += asyncio.run(ask_gpt(requests, answers_path(v), tier=args.tier, concurrency=args.concurrency,
                                         budget_usd=args.budget - spent))  # fmt: skip
    if not args.dry_run:
        report(args.split, args.variants, args.workers)


if __name__ == "__main__":
    main()
