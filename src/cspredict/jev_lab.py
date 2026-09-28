"""How far can Jev be pushed, honestly? Prompt experiments for the Jev write-up.

Rules, so that the numbers mean what they say:
- Jev only sees what the friendly team knew at that moment (describe.py): no later events, no
  true positions, and none of our filters' beliefs. Facts computed for it in code (travel times,
  how likely the team was to spot someone in each callout, and how often players of that side
  are in each callout at this point of a round in the *training* demos) are labelled as such.
- Variants are compared on validation moments. Only the finalists run on the test moments, once.
  Calibration and the evidence-only formula are fitted on validation and applied to test.
- Every variant tried is reported, with its cost.

Variants (one Jev call per moment for base / reach_walk, per enemy for the others):
    base        the plain JSON description of the moment (as in typesafe_bench)
    reach_walk  + walking and running times to each callout (typesafe_bench)
    focus       only the context of the enemy asked about, and every callout option carries the
                facts computed in code, following TypeSafe's advice to keep arithmetic and
                spatial reasoning in code and to send only relevant state
    split       focus asked as two atomic questions, combined in code: "still where last seen?"
                (yes/no) and "if not, which callout?"
    split_rates split plus historical movement rates from the training demos: how often players
                last seen in that callout that long ago are still there, and where the ones who
                left are now

Baselines scored on the same moments: last_seen, prior, our filters (hmm, pf, ens), and the
evidence-only formula: a 7-weight softmax over the same per-callout facts, with no Jev.

Usage:
    python -m cspredict.jev_lab --split val --variants base reach_walk focus split split_rates
    python -m cspredict.jev_lab --split test --variants base reach_walk focus split split_rates
    python -m cspredict.jev_lab --split val --variants focus --dry-run      # example request, cost
    python -m cspredict.jev_lab --split test --chart                        # docs/jev_by_horizon.png
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import numpy as np
import polars as pl
from loguru import logger
from scipy.optimize import minimize

from cspredict.build import load_models, model_dir
from cspredict.calibrate import Calibration, fit
from cspredict.config import OUTPUT_DIR
from cspredict.dataset import list_demos
from cspredict.describe import RUN_SPEED, WALK_SPEED, callout_name, callout_neighbours, callout_run_times, callouts
from cspredict.typesafe_bench import (
    BASELINES,
    EPS,
    PRICE_PER_MTOK,
    _api_key,
    _ask,
    _cluster_ci,
    _load_cache,
    _metrics,
    build_requests,
    group_key,
    sample_moments,
)

OUT = OUTPUT_DIR / "jev"
MODEL = "jev-1.13.0"  # pinned, so reruns hit the same model
VARIANTS = ("base", "reach_walk", "focus", "split", "split_rates")
RATE_SINCE_EDGES = (2.0, 5.0, 10.0, 20.0, 40.0)  # seconds since last seen, for the movement rates
RATE_PSEUDO = 5.0  # pseudo-counts pulling a sparse row of the movement table towards its side's average
DEFAULT_N = {"val": 2000, "test": 3000}
SEE_NOW = (0.25, 0.10)  # spot-chance levels for "yes" / "partly"
USUAL = (0.12, 0.04)  # share of that side's players in a callout for "often" / "sometimes" (uniform ~0.04)

FOCUS_INSTRUCTIONS = (
    "Which callout is enemy {label} in right now? Every option lists facts worked out from what "
    "your team knew: whether {label} could have reached it since last seen (players usually "
    "shift-walk, which is silent), whether your team can see it now or watched it since {label} was "
    "last seen (an enemy standing there would probably have been spotted), and how often players of "
    "{label}'s side are there at this point of a round in pro matches."
)
STAY_INSTRUCTIONS = (
    "Is enemy {label} still in {last}, the callout where they were last seen {since} s ago? Use their "
    "recent sightings, the time since then, whether your team can see {last} now, the bomb and the "
    "kill feed."
)
MOVED_INSTRUCTIONS = (
    "Enemy {label} has left {last}, where they were last seen {since} s ago. Which callout are they "
    "in now? Every option lists facts worked out from what your team knew: whether {label} could have "
    "reached it (players usually shift-walk, which is silent), whether your team can see it now or "
    "watched it since {label} was last seen, and how often players of {label}'s side are there at "
    "this point of a round in pro matches."
)


# ---------------------------------------------------------------------- facts and requests
def _travel(grid) -> tuple[dict[str, int], np.ndarray]:
    names, run_s = callout_run_times(grid)
    return {n: i for i, n in enumerate(names)}, run_s


def callout_facts(m: dict, callout: str, index: dict[str, int], run_s: np.ndarray) -> dict:
    """Plain-language facts about one callout for the enemy of moment m, all computed in code."""
    ev = m["evidence"]
    last, since = ev["last_seen_at"], ev["seconds_since_seen"]
    facts: dict[str, str] = {}
    if callout == last:
        facts["last_seen_here"] = f"yes, {since} s ago"
    elif last in index and callout in index:
        run = float(run_s[index[last], index[callout]])
        walk = run * RUN_SPEED / WALK_SPEED
        facts["reachable_since_last_seen"] = (
            "yes, even at a silent walk" if walk <= since + 1
            else "only by running (audible footsteps)" if run <= since + 1
            else "no, too far"
        )  # fmt: skip
    now = ev["spot_chance_now"].get(callout, 0.0)
    mean = ev["spot_chance_since_seen"].get(callout, 0.0)
    share = ev["usual_share"].get(callout, 0.0)
    facts["your_team_can_see_it_now"] = "yes" if now >= SEE_NOW[0] else "partly" if now >= SEE_NOW[1] else "no"
    facts["watched_since_last_seen"] = "mostly" if mean >= SEE_NOW[0] else "partly" if mean >= SEE_NOW[1] else "hardly"
    facts["how_often_that_side_is_here_now"] = "often" if share >= USUAL[0] else "sometimes" if share >= USUAL[1] else "rarely"
    return facts


def movement_rates(grid) -> np.ndarray:
    """(2 sides, 3 bomb phases, since bins, C, C) from the training demos only: of the players of a
    side last seen in callout a that many seconds ago (and off the radar now), the share now in
    callout b. Rows are shrunk towards the side's average row for that phase and time."""
    path = OUT / "movement_rates.npz"
    if path.exists():
        return np.load(path)["rates"]
    from cspredict.motion import phase_expr

    C = len(grid.places)
    counts = np.zeros((2, 3, len(RATE_SINCE_EDGES) + 1, C, C))
    for ref in list_demos(["hltv", "xego"], ["train"]):
        t = ref.table("ticks").filter(pl.col("is_alive")).join(
            ref.table("rounds").select("round_num", "bomb_plant", "bomb_site"), on="round_num"
        ).with_columns(phase_expr()).sort("round_num", "steamid", "tick")
        place = grid.node_place_idx[grid.locate(t["X"].to_numpy(), t["Y"].to_numpy(), t["Z"].to_numpy())]
        t = t.with_columns(pl.Series("place", place)).with_columns(
            pl.when(pl.col("spotted")).then(pl.col("place")).forward_fill().over("round_num", "steamid").alias("last_place"),
            pl.when(pl.col("spotted")).then(pl.col("tick")).forward_fill().over("round_num", "steamid").alias("last_tick"),
        ).filter(~pl.col("spotted") & pl.col("last_tick").is_not_null())
        since = (t["tick"].to_numpy() - t["last_tick"].to_numpy()) / 64.0
        side = (t["side"] == "ct").to_numpy().astype(int)
        np.add.at(counts, (side, t["phase"].to_numpy(), np.searchsorted(RATE_SINCE_EDGES, since, side="right"),
                           t["last_place"].to_numpy(), t["place"].to_numpy()), 1.0)  # fmt: skip
    average = counts.sum(axis=3, keepdims=True)
    average = average / np.maximum(average.sum(axis=4, keepdims=True), 1.0)
    rates = (counts + RATE_PSEUDO * average) / (counts.sum(axis=4, keepdims=True) + RATE_PSEUDO)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, rates=rates)
    return rates


def _rates_row(m: dict, grid, rates: np.ndarray) -> np.ndarray:
    """(C,) the training-demo shares for this moment's enemy: side, bomb phase, time since seen."""
    side = 0 if m["friendly"] == "ct" else 1  # the enemy side: T if we are CT
    bomb = m["state"]["bomb"]
    phase = 1 if "on A site" in bomb else 2 if "on B site" in bomb else 0
    b = int(np.searchsorted(RATE_SINCE_EDGES, m["evidence"]["seconds_since_seen"], side="right"))
    last = [callout_name(p) for p in grid.places].index(m["evidence"]["last_seen_at"])
    return rates[side, phase, b, last]


def focused_state(m: dict) -> dict:
    """Only the context of the enemy asked about (TypeSafe: accuracy falls as unrelated state grows)."""
    st, ev, label = m["state"], m["evidence"], m["label"]
    me = next(e for e in st["enemies"] if e["id"] == label)
    kills = [k for k in st["recent_kill_feed"] if isinstance(k, dict) and k.get("killer") == label]
    others = [{k: v for k, v in e.items() if k in ("id", "status", "at", "last_seen_at", "seconds_since_seen")}
              for e in st["enemies"] if e["id"] != label]  # fmt: skip
    return {
        "game": st["game"],
        "your_team": st["your_team"],
        "seconds_into_round": st["seconds_into_round"],
        "bomb": st["bomb"],
        "players_alive": st["players_alive"],
        "enemy_team_buy": st["enemy_team_buy"],
        "enemy_asked_about": {
            "id": label,
            "last_seen_at": ev["last_seen_at"],
            "seconds_since_seen": ev["seconds_since_seen"],
            "when_last_seen": me.get("when_last_seen", "unknown"),
            "recent_sightings": ev["history"],
            "kills_in_the_last_30_s": kills or "none",
        },
        "its_teammates": others,
        "smokes_active_at": st["smokes_active_at"],
        "molotovs_burning_at": st.get("molotovs_burning_at", ["none"]),
    }


def request_key(m: dict, variant: str) -> str:
    return group_key(m) if variant in ("base", "reach_walk") else f"{group_key(m)}|{m['label']}"


def build(moments: list[dict], grid, variant: str) -> dict[str, dict]:
    if variant in ("base", "reach_walk"):
        return build_requests(moments, grid, variant)
    index, run_s = _travel(grid)
    neighbours = callout_neighbours(grid)
    names = callouts(grid)
    rates = movement_rates(grid) if variant == "split_rates" else None
    requests = {}
    for m in moments:
        label, ev = m["label"], m["evidence"]
        last, since = ev["last_seen_at"], ev["seconds_since_seen"]
        row = _rates_row(m, grid, rates) if rates is not None else None
        stay = float(row[names.index(last)]) if row is not None else None

        def option(c: str) -> dict:
            facts = {"connects_to": ", ".join(neighbours.get(c, [])) or "nothing listed", **callout_facts(m, c, index, run_s)}
            if row is not None and c != last:
                share = float(row[names.index(c)]) / max(1.0 - stay, 1e-9)
                facts["pro_players_who_left_now_here"] = f"{share:.0%} of those who left {last}"
            return facts

        if variant == "focus":
            questions = {"where": {"type": "choice", "instructions": FOCUS_INSTRUCTIONS.format(label=label),
                                   "criteria": {c: option(c) for c in names}}}  # fmt: skip
        else:
            questions = {
                "stay": {"type": "noul", "instructions": STAY_INSTRUCTIONS.format(label=label, last=last, since=since),
                         "criteria": {"true": {"what": f"Still in {last}: holding there or moving within it.",
                                               f"facts_about_{last}": callout_facts(m, last, index, run_s)
                                               | ({"pro_players_last_seen_here_this_long_ago_still_here": f"{stay:.0%}"}
                                                  if stay is not None else {})},
                                      "false": f"Has left {last} for another callout."}},
                "where": {"type": "choice", "instructions": MOVED_INSTRUCTIONS.format(label=label, last=last, since=since),
                          "criteria": {c: option(c) for c in names if c != last}},
            }  # fmt: skip
        requests[request_key(m, variant)] = {"state": focused_state(m), "questions": questions}
    return requests


# ---------------------------------------------------------------------- answers -> probabilities
def jev_probs(moments: list[dict], cache: dict[str, dict], variant: str, names: list[str]) -> np.ndarray:
    """(n, C) Jev's callout probabilities per moment (rows of NaN where unanswered)."""
    col = {c: i for i, c in enumerate(names)}
    out = np.full((len(moments), len(names)), np.nan)
    for i, m in enumerate(moments):
        row = cache.get(request_key(m, variant))
        if row is None:
            continue
        a = row["answers"]
        if variant in ("base", "reach_walk"):
            dist = a.get(m["label"], {}).get("probabilities")
            if dist is None:
                continue
            p = np.array([dist.get(c, 0.0) for c in names])
        elif variant == "focus":
            p = np.array([a["where"]["probabilities"].get(c, 0.0) for c in names])  # split and split_rates below
        else:
            stay = float(a["stay"]["noul"])
            where = np.array([a["where"]["probabilities"].get(c, 0.0) for c in names])
            where = where / max(where.sum(), 1e-12)
            p = (1.0 - stay) * where
            p[col[m["evidence"]["last_seen_at"]]] = stay
        out[i] = p / max(p.sum(), 1e-12)
    return out


# ---------------------------------------------------------------------- evidence-only formula
FEATURES = ("log usual share", "reachable at a walk", "reachable only running", "seen now", "watched since",
            "last seen here", "last seen here x log(1 + s since)")  # fmt: skip
RATE_FEATURES = ("last seen here x log(historical stay rate)", "elsewhere x log(historical share of leavers)")


def features(moments: list[dict], grid, rates: np.ndarray | None = None) -> np.ndarray:
    """(n, C, F) the per-callout facts behind the focus variant, as numbers (plus the historical
    movement rates behind split_rates when `rates` is given)."""
    index, run_s = _travel(grid)
    names = callouts(grid)
    X = np.zeros((len(moments), len(names), len(FEATURES) + (len(RATE_FEATURES) if rates is not None else 0)))
    for i, m in enumerate(moments):
        if rates is not None:
            row = _rates_row(m, grid, rates)
            j_last = names.index(m["evidence"]["last_seen_at"])
            stay = row[j_last]
            X[i, :, 8] = np.log(row / max(1 - stay, 1e-9) + 1e-4)
            X[i, j_last, 8] = 0.0
            X[i, j_last, 7] = np.log(stay + 1e-4)
        ev = m["evidence"]
        last, since = ev["last_seen_at"], ev["seconds_since_seen"]
        for j, c in enumerate(names):
            X[i, j, 0] = np.log(ev["usual_share"].get(c, 0.0) + 1e-3)
            if c == last:
                X[i, j, 5], X[i, j, 6] = 1.0, np.log1p(since)
            elif last in index and c in index:
                run = float(run_s[index[last], index[c]])
                X[i, j, 1] = float(run * RUN_SPEED / WALK_SPEED <= since + 1)
                X[i, j, 2] = float(run <= since + 1) - X[i, j, 1]
            X[i, j, 3] = ev["spot_chance_now"].get(c, 0.0)
            X[i, j, 4] = ev["spot_chance_since_seen"].get(c, 0.0)
    return X


def _softmax(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    z = X @ w
    z = z - z.max(axis=1, keepdims=True)
    q = np.exp(z)
    return q / q.sum(axis=1, keepdims=True)


def fit_formula(X: np.ndarray, truth: np.ndarray) -> np.ndarray:
    def nll(w):
        q = _softmax(X, w)
        return -np.log(np.clip(q[np.arange(len(truth)), truth], 1e-9, 1)).mean()

    return minimize(nll, np.zeros(X.shape[2]), method="L-BFGS-B").x


# ---------------------------------------------------------------------- scoring
def score(named: dict[str, np.ndarray], truth: np.ndarray, clusters: np.ndarray, last_idx: np.ndarray) -> pl.DataFrame:
    rows = []
    for name, p in named.items():
        ok = ~np.isnan(p).any(axis=1)
        met = _metrics(p[ok], truth[ok])
        row = {"model": name, "n": int(ok.sum()), "picks_last_seen": float((p[ok].argmax(axis=1) == last_idx[ok]).mean())}
        for k, v in met.items():
            lo, hi = _cluster_ci(v, clusters[ok])
            row |= {k: float(v.mean()), f"{k}_lo": lo, f"{k}_hi": hi}
        rows.append(row)
    return pl.DataFrame(rows).sort("logloss")


def cross_validate(named: dict[str, np.ndarray], X: np.ndarray, truth: np.ndarray, clusters: np.ndarray,
                   since: np.ndarray, folds: int = 2, seed: int = 0) -> pl.DataFrame:
    """Held-out scores on validation, folds of whole rounds: the evidence formula, each Jev variant
    calibrated, and each calibrated variant blended with the formula (weight fitted on the other
    folds). If the blend beats the formula alone, Jev adds something the formula's facts don't."""
    ids = np.unique(clusters)
    fold_of = dict(zip(ids, np.random.default_rng(seed).permutation(len(ids)) % folds))
    fold = np.array([fold_of[c] for c in clusters])
    held: dict[str, np.ndarray] = {"evidence formula (no Jev)": np.zeros((len(truth), X.shape[1]))}
    weights: dict[str, list[float]] = {}
    for f in range(folds):
        tr, te = fold != f, fold == f
        formula_tr = _softmax(X[tr], fit_formula(X[tr], truth[tr]))
        formula_te = _softmax(X[te], fit_formula(X[tr], truth[tr]))
        held["evidence formula (no Jev)"][te] = formula_te
        for name, p in named.items():
            if not name.startswith("jev ") or name.endswith("calibration") or np.isnan(p).any():
                continue
            cal = fit("platt", np.clip(p[tr], EPS, None), truth[tr], since[tr], by_since=True)
            q_tr, q_te = cal.apply(np.clip(p[tr], EPS, None), since[tr]), cal.apply(np.clip(p[te], EPS, None), since[te])
            grid_w = np.linspace(0, 1, 21)
            ll = [-np.log(np.clip((w * q_tr + (1 - w) * formula_tr)[np.arange(tr.sum()), truth[tr]], 1e-9, 1)).mean() for w in grid_w]
            w = float(grid_w[int(np.argmin(ll))])
            weights.setdefault(name, []).append(w)
            held.setdefault(f"{name} + calibration", np.zeros_like(p))[te] = q_te
            held.setdefault(f"{name} blended with formula", np.zeros_like(p))[te] = w * q_te + (1 - w) * formula_te
    rows = []
    for name, p in held.items():
        met = _metrics(p, truth)
        row = {"model": name, "top1": float(met["top1"].mean()), "logloss": float(met["logloss"].mean())}
        lo, hi = _cluster_ci(met["logloss"], clusters)
        row |= {"logloss_lo": lo, "logloss_hi": hi}
        base = name.replace(" blended with formula", "")
        row["jev_weight"] = float(np.mean(weights[base])) if name.endswith("blended with formula") else None
        rows.append(row)
    return pl.DataFrame(rows).sort("logloss")


HORIZONS = ((0, 5, "0–5 s"), (5, 10, "5–10 s"), (10, 20, "10–20 s"), (20, 40, "20–40 s"), (40, np.inf, "40 s+"))


def plot_by_horizon(moments: list[dict], grid, path: Path) -> None:
    """Right callout by seconds since last seen: our model, the formula, pushed and plain Jev and
    "last seen", on the cached answers (validation-fitted calibration and formula). The colours are
    the first five categorical slots of the validated default palette (dataviz skill)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = callouts(grid)
    truth = np.array([m["truth"] for m in moments])
    since = np.array([m["evidence"]["seconds_since_seen"] for m in moments], dtype=float)

    def jev(v: str) -> np.ndarray:
        p = jev_probs(moments, _load_cache(cache_path(v)), v, names)
        return Calibration.load(OUT / f"calibration_{v}.json").apply(np.clip(p, EPS, None), since)

    wr = np.array(list(json.loads((OUT / "formula_rates_weights.json").read_text()).values()))
    series = {
        "cspredict (my model)": ("#2a78d6", np.array([m["probs"]["ens"] for m in moments])),
        "Jev, pushed (best honest setup)": ("#eb6834", jev("split_rates")),
        "9-number formula, same facts, no AI": ("#1baf7a", _softmax(features(moments, grid, movement_rates(grid)), wr)),
        "Jev, plain description": ("#eda100", jev("base")),
        '"where I last saw them"': ("#e87ba4", np.array([m["probs"]["last_seen"] for m in moments])),
    }
    surface, ink, ink2, grid_c = "#fcfcfb", "#0b0b0b", "#52514e", "#e7e6e2"
    fig, ax = plt.subplots(figsize=(8.6, 5.4), dpi=200, facecolor=surface)
    ax.set_facecolor(surface)
    x = np.arange(len(HORIZONS))
    counts = [int(((since >= lo) & (since < hi)).sum()) for lo, hi, _ in HORIZONS]
    for label, (color, p) in series.items():
        top1 = _metrics(p, truth)["top1"]
        ys = [100 * top1[(since >= lo) & (since < hi)].mean() for lo, hi, _ in HORIZONS]
        ax.plot(x, ys, color=color, lw=2.0, marker="o", ms=6.5, mec=surface, mew=1.5, label=label, zorder=3)
    ax.set_xticks(x, [h[2] for h in HORIZONS], color=ink2, fontsize=9)
    ax.set_xlabel("seconds since the enemy was last seen", color=ink2, fontsize=9, labelpad=8)
    ax.set_ylabel("right callout named first (of 23)", color=ink2, fontsize=9, labelpad=8)
    ax.set_ylim(0, 85)
    ax.set_xlim(-0.3, len(x) - 0.7)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.tick_params(axis="y", colors=ink2, labelsize=8.5, length=0)
    ax.tick_params(axis="x", length=0)
    ax.grid(axis="y", color=grid_c, lw=0.8, zorder=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right", labelcolor=ink, handlelength=2.2)
    fig.suptitle("Where is the enemy you can't see?", x=0.075, y=0.975, ha="left", fontsize=13, color=ink, weight="bold")
    fig.text(0.075, 0.905, f"{len(moments):,} held-out pro moments on Mirage (CS2). Every model sees only what the team knew.",
             ha="left", fontsize=9, color=ink2)
    fig.text(0.075, 0.02, "Pushed Jev: one focused question per enemy, facts computed in code, and pro movement rates from "
             "training rounds; calibrated on validation rounds.\nFormula: 9 weights fitted on validation rounds, using the same "
             f"facts. Each point is {min(counts):,}–{max(counts):,} moments, so gaps of a few points are within noise.",
             ha="left", fontsize=7, color=ink2, linespacing=1.4)
    fig.subplots_adjust(left=0.1, right=0.97, top=0.86, bottom=0.2)
    fig.savefig(path, facecolor=surface)
    plt.close(fig)


def load_moments(split: str, n: int, seed: int, workers: int) -> list[dict]:
    path = OUT / f"moments_{split}_seed{seed}_n{n}.json"
    if path.exists():
        return json.loads(path.read_text())
    refs = list_demos(["hltv"], [split])
    logger.info(f"Sampling {n} {split} moments from {len(refs)} demos (runs the filters once)")
    moments = sample_moments(refs, model_dir(["hltv", "xego"]), n, seed, workers, split)
    OUT.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(moments))
    return moments


def cache_path(variant: str) -> Path:
    return OUT / f"answers_{variant}.jsonl"


def main() -> None:
    ap = argparse.ArgumentParser(description="Honest prompt experiments with TypeSafe Jev.")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--n", type=int, default=None, help="moments (default: 2000 val, 3000 test)")
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=VARIANTS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--per-second", type=float, default=15.0, help="request starts per second (limit: 20)")
    ap.add_argument("--dry-run", action="store_true", help="print an example request and the cost; send nothing")
    ap.add_argument("--no-ask", action="store_true", help="only score what is cached")
    ap.add_argument("--chart", action="store_true", help="draw docs/jev_by_horizon.png from the cached answers and exit")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    moments = load_moments(args.split, args.n or DEFAULT_N[args.split], args.seed, args.workers)
    grid = load_models(model_dir(["hltv", "xego"])).grid
    if args.chart:
        plot_by_horizon(moments, grid, OUTPUT_DIR.parent / "docs" / "jev_by_horizon.png")
        return
    names = callouts(grid)
    truth = np.array([m["truth"] for m in moments])
    clusters = np.array([f"{m['demo']}|{m['round']}" for m in moments])
    since = np.array([m["evidence"]["seconds_since_seen"] for m in moments], dtype=float)
    last_idx = np.array([names.index(m["evidence"]["last_seen_at"]) for m in moments])

    for variant in args.variants:
        requests = build(moments, grid, variant)
        tokens = sum(len(json.dumps(r)) for r in requests.values()) / 4
        logger.info(f"{variant}: {len(requests)} requests, ~{tokens / 1e6:.2f}M input tokens (~${tokens / 1e6 * PRICE_PER_MTOK:.2f})")
        if args.dry_run:
            print(json.dumps(next(iter(requests.values())), indent=1)[:5000])
            continue
        if not args.no_ask:
            if _api_key() is None:
                raise SystemExit("Set TYPESAFE_API_KEY (environment or .env)")
            asyncio.run(_ask(requests, cache_path(variant), args.concurrency, args.per_second, model=MODEL))
    if args.dry_run:
        return

    named = {b: np.array([m["probs"][b] for m in moments]) for b in BASELINES}
    X = features(moments, grid)
    if args.split == "val":
        w = fit_formula(X, truth)
        (OUT / "formula_weights.json").write_text(json.dumps(dict(zip(FEATURES, w.tolist())), indent=2))
    else:
        w = np.array(list(json.loads((OUT / "formula_weights.json").read_text()).values()))
    named["evidence formula (no Jev)"] = _softmax(X, w)
    Xr = features(moments, grid, movement_rates(grid))
    if args.split == "val":
        wr = fit_formula(Xr, truth)
        (OUT / "formula_rates_weights.json").write_text(json.dumps(dict(zip(FEATURES + RATE_FEATURES, wr.tolist())), indent=2))
    else:
        wr = np.array(list(json.loads((OUT / "formula_rates_weights.json").read_text()).values()))
    named["evidence formula + movement rates (no Jev)"] = _softmax(Xr, wr)
    tokens_used = {}
    for variant in args.variants:
        cache = _load_cache(cache_path(variant))
        p = jev_probs(moments, cache, variant, names)
        named[f"jev {variant}"] = p
        used = {request_key(m, variant) for m in moments}
        tokens_used[variant] = sum(r["usage"]["input_tokens"] or 0 for k, r in cache.items() if k in used)
        cal_path = OUT / f"calibration_{variant}.json"
        ok = ~np.isnan(p).any(axis=1)
        if args.split == "val":  # fit where the variant was developed, apply on test
            fit("platt", np.clip(p[ok], EPS, None), truth[ok], since[ok], by_since=True).save(cal_path)
        if cal_path.exists():
            cal = Calibration.load(cal_path)
            q = np.full_like(p, np.nan)
            q[ok] = cal.apply(np.clip(p[ok], EPS, None), since[ok])
            named[f"jev {variant} + calibration"] = q

    # Jev blended with the formula that has the same facts; the weight is fitted on validation.
    blend_path = OUT / "blend_weights.json"
    blend_ref = {"split_rates": "evidence formula + movement rates (no Jev)"}
    if args.split == "val":
        weights = {}
        for v in args.variants:
            q, ref = named.get(f"jev {v} + calibration"), named[blend_ref.get(v, "evidence formula (no Jev)")]
            if q is None or np.isnan(q).any():
                continue
            grid_w = np.linspace(0, 1, 21)
            ll = [-np.log(np.clip((w_ * q + (1 - w_) * ref)[np.arange(len(truth)), truth], 1e-9, 1)).mean() for w_ in grid_w]
            weights[v] = float(grid_w[int(np.argmin(ll))])
        blend_path.write_text(json.dumps(weights, indent=2))
    for v, w_ in (json.loads(blend_path.read_text()) if blend_path.exists() else {}).items():
        q, ref = named.get(f"jev {v} + calibration"), named[blend_ref.get(v, "evidence formula (no Jev)")]
        if q is not None and v in args.variants:
            named[f"jev {v} + calibration, blended ({w_:.2f}) with formula"] = w_ * q + (1 - w_) * ref

    table = score(named, truth, clusters, last_idx)
    pl.Config.set_tbl_rows(40)
    pl.Config.set_tbl_width_chars(220)
    pl.Config.set_float_precision(3)
    print(f"\n== {args.split}: {len(moments):,} moments; 95% intervals resample whole rounds ==")
    print(table.select("model", "n", "top1", "top1_lo", "top1_hi", "top3", "logloss", "logloss_lo", "logloss_hi", "picks_last_seen"))
    if args.split == "val":
        print("\n(calibration above is fitted in-sample; held out, over folds of whole validation rounds:)")
        print(cross_validate(named, X, truth, clusters, since))
        print("\nThe same with the movement rates in the formula (compare with split_rates):")
        print(cross_validate({k: v for k, v in named.items() if k == "jev split_rates"}, Xr, truth, clusters, since))
    for v, t in tokens_used.items():
        print(f"{v}: {t:,} input tokens (~${t / 1e6 * PRICE_PER_MTOK:.2f})")
    table.write_csv(OUT / f"results_{args.split}.csv")
    print("\nPaired differences (same moments; 95% intervals resample whole rounds):")
    pairs = [("jev split_rates + calibration", "last_seen"), ("jev split_rates + calibration", "jev base + calibration"),
             ("jev split_rates + calibration", "evidence formula + movement rates (no Jev)"),
             ("jev split_rates + calibration", "ens")]
    blended = [k for k in named if k.startswith("jev split_rates + calibration, blended")]
    pairs += [(blended[0], "evidence formula + movement rates (no Jev)")] if blended else []
    for a_, b_ in pairs:
        if a_ not in named or b_ not in named or np.isnan(named[a_]).any():
            continue
        ma, mb = _metrics(named[a_], truth), _metrics(named[b_], truth)
        for k in ("top1", "logloss"):
            d = ma[k] - mb[k]
            lo, hi = _cluster_ci(d, clusters)
            print(f"  {a_}  minus  {b_}: {k} {d.mean():+.3f} ({lo:+.3f} to {hi:+.3f})")


if __name__ == "__main__":
    main()
