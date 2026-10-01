"""Jev as a feature extractor on top of cspredict (the TypeSafe "autoresearch feature discovery" pattern).

Instead of asking Jev for the final answer (which of 23 callouts?), Jev answers small questions
about each moment (is this enemy still where they were seen? heading for A or B? lurking?), and a
small model learns how much each answer is worth on top of cspredict's own probabilities. Jev's
overconfidence then does not matter: the stacked model learns how far to trust each answer.

Rules, as in jev_lab:
- Jev only sees what the friendly team knew (describe.py, via jev_lab.focused_state plus the
  team's own positions). It never sees cspredict's beliefs, later events or true positions.
- Everything is fitted and scored on validation moments, held out over folds of whole rounds.
  No test moment is used.
- A no-Jev control gets the same stacked model with only facts computed in code, so a Jev gain
  is not just the effect of refitting.

The stacked model is a conditional logit over the 23 callouts: score(c) = w . x(c), softmax over
c. x(c) holds log cspredict(c) (its weight starts at 1), the 9 per-callout facts of jev_lab's
formula, travel times computed in code, and Jev's answers times per-callout facts (an answer that
is the same for every callout cancels in the softmax, so it has to be paired with one that is
not, e.g. "heading for A" x "seconds from this callout to A").

Round 1 questions were proposed by Claude before any answer was seen (the cookbook's proposer).

Usage:
    python -m cspredict.jev_features --n 300 --dry-run    # example request and cost
    python -m cspredict.jev_features --n 300              # pilot: ask, then score
    python -m cspredict.jev_features --n 300 --no-ask     # score the cached answers only
    python -m cspredict.jev_features --roleplay           # Jev plays the enemy (all 2,000 val moments)
    python -m cspredict.jev_features --context            # role-play + CS basics, score, the player's match

Role-play: the same request as jev_lab's focus variant (same state, same 23 options with the same
facts), with only the instructions changed, from "which callout is enemy T1 in?" to "you are T1,
think like a T player: where would you be?". Compared with the cached focus answers, so the
difference measures the framing alone. Jev's pick enters the stacked model as one more input.

Context (--context), on top of the role-play request, in two steps so each can be judged:
    ctx         + the basics of the game, the match score before this round and the last five
                results, and the player's match so far (kills, deaths, opening duels, main weapon)
    ctx_habits  + where the friendly team has spotted this player in earlier rounds on this side
All of it was known to the friendly team at the time: earlier rounds' results and kill feed, and
its own sightings. The habit counts also go to the code-only model, so a Jev gain has to come
from more than the raw counts.
"""

from __future__ import annotations

import argparse
import asyncio
import json

from collections import Counter

import numpy as np
import polars as pl
from loguru import logger
from scipy.optimize import minimize

from cspredict.build import load_models, model_dir
from cspredict.config import OUTPUT_DIR
from cspredict.dataset import list_demos
from cspredict.describe import callouts
from cspredict.infostate import half_start, match_rounds
from cspredict.jev_lab import FEATURES, MODEL, RATE_FEATURES, _travel, features, focused_state, load_moments, movement_rates
from cspredict.jev_lab import build as build_lab
from cspredict.jev_lab import jev_probs
from cspredict.typesafe_bench import EPS, PRICE_PER_MTOK, _api_key, _ask, _cluster_ci, _load_cache, _metrics, group_key

OUT = OUTPUT_DIR / "jev_features"
LAMBDA = 0.01  # L2 pull towards "cspredict alone", on standardised features; fixed before any fit
FOLDS, FOLD_SEEDS = 5, 5

HEADING = {
    "holding": "Holding a position in or near the callout where they were last seen; not going anywhere.",
    "to_a": "Moving toward Bombsite A: to take it, retake it or defend it.",
    "to_b": "Moving toward Bombsite B: to take it, retake it or defend it.",
    "to_your_team": "Moving toward your team's players, to fight or flank them.",
    "away": "Falling back, away from the fighting: saving, hiding or playing for time.",
}
AGGRESSION = [
    "Passive: hiding, saving the weapon or waiting for the other team to come.",
    "Holding an angle and waiting for a fight.",
    "Actively repositioning or rotating.",
    "Aggressive: pushing to find a kill or take ground.",
]
DISTANCE = [
    "Has not moved: still in the callout where they were last seen.",
    "Moved to a neighbouring callout.",
    "Moved a medium distance, two or three callouts away.",
    "Moved across the map.",
]


# ---------------------------------------------------------------------- round 1 questions
def state_for(m: dict) -> dict:
    """jev_lab's focused state plus where the friendly team is (all of it known to the team)."""
    st = m["state"]
    return focused_state(m) | {
        "your_teammates": st["your_teammates"],
        "callouts_your_team_is_watching": st["callouts_your_team_is_watching"],
        "recent_kill_feed": st["recent_kill_feed"],
    }


def build(moments: list[dict]) -> dict[str, dict]:
    requests = {}
    for m in moments:
        label, ev = m["label"], m["evidence"]
        last, since = ev["last_seen_at"], ev["seconds_since_seen"]
        questions = {
            "stay": {"type": "noul", "instructions": f"Is enemy {label} still in {last}, where they were last seen {since} s ago?",
                     "criteria": {"true": f"Still in {last}.", "false": f"Has left {last}."}},
            "heading": {"type": "choice", "criteria": HEADING,
                        "instructions": f"What is enemy {label} most likely doing since they were last seen in {last} {since} s ago?"},
            "aggression": {"type": "score", "criteria": AGGRESSION,
                           "instructions": f"How aggressively is enemy {label} likely playing right now?"},
            "lurking": {"type": "noul", "instructions": f"Is enemy {label} playing alone, away from their teammates (lurking or flanking)?",
                        "criteria": {"true": "Alone, away from their teammates.", "false": "With or near their teammates."}},
            "distance": {"type": "score", "criteria": DISTANCE,
                         "instructions": f"How far has enemy {label} probably moved since they were last seen in {last} {since} s ago?"},
        }  # fmt: skip
        requests[f"{group_key(m)}|{label}"] = {"state": state_for(m), "questions": questions}
    return requests


# ---------------------------------------------------------------------- features
CODE_EXTRA = ("s from last seen", "s to A", "s to B", "s to nearest teammate of ours", "s to its nearest known teammate")
OLD_JEV = ("split_rates: logit(still there) x last seen here", "split_rates: log P(where | left) x elsewhere")
NEW_JEV = (
    "logit(still there) x last seen here", "holding x last seen here", "to A x -(s to A)", "to B x -(s to B)",
    "to our team x -(s to our team)", "away x (s to our team)", "aggression x last seen here",
    "aggression x -(s to our team)", "lurking x (s to its teammates)", "distance x (s from last seen)",
    "distance x last seen here",
)  # fmt: skip


def _logit(p: float) -> float:
    p = min(max(p, 0.01), 0.99)
    return float(np.log(p / (1 - p)))


def _mean_score(a: dict, levels: int) -> float:
    return float(a["score"]) / (levels - 1)


def code_features(moments: list[dict], grid) -> np.ndarray:
    """(n, C, F) log cspredict, jev_lab's 9 formula facts and travel times, all from code."""
    names = callouts(grid)
    index, run_s = _travel(grid)
    order = np.array([index[c] for c in names])
    R = run_s[np.ix_(order, order)] / 10.0  # tens of seconds, in callouts() order
    col = {c: i for i, c in enumerate(names)}
    A, B = col["Bombsite A"], col["Bombsite B"]
    base = np.log(np.clip(np.array([m["probs"]["ens"] for m in moments]), EPS, None))[:, :, None]
    facts = features(moments, grid, movement_rates(grid))
    extra = np.zeros((len(moments), len(names), len(CODE_EXTRA)))
    for i, m in enumerate(moments):
        st, last = m["state"], col[m["evidence"]["last_seen_at"]]
        extra[i, :, 0] = R[last]
        extra[i, :, 1], extra[i, :, 2] = R[:, A], R[:, B]
        ours = [col[t["at"]] for t in st["your_teammates"] if t.get("at") in col]
        if ours:
            extra[i, :, 3] = R[:, ours].min(axis=1)
        mates = [col[e.get("at") or e.get("last_seen_at")] for e in st["enemies"]
                 if e["id"] != m["label"] and (e.get("at") or e.get("last_seen_at")) in col]  # fmt: skip
        if mates:
            extra[i, :, 4] = R[:, mates].min(axis=1)
    return np.concatenate([base, facts, extra], axis=2)


def old_jev_features(moments: list[dict], grid) -> np.ndarray | None:
    """(n, C, 2) from the split_rates answers already cached by jev_lab (no new requests)."""
    cache = _load_cache(OUTPUT_DIR / "jev" / "answers_split_rates.jsonl")
    names = callouts(grid)
    X = np.zeros((len(moments), len(names), len(OLD_JEV)))
    for i, m in enumerate(moments):
        row = cache.get(f"{group_key(m)}|{m['label']}")
        if row is None:
            return None
        a, last = row["answers"], names.index(m["evidence"]["last_seen_at"])
        X[i, last, 0] = _logit(float(a["stay"]["noul"]))
        where = np.array([a["where"]["probabilities"].get(c, 0.0) for c in names])
        X[i, :, 1] = np.log(where / max(where.sum(), 1e-12) + 0.01)
        X[i, last, 1] = 0.0
    return X


def new_jev_features(moments: list[dict], cache: dict[str, dict], code: np.ndarray) -> np.ndarray:
    """(n, C, 11) round 1 answers times per-callout facts; NaN rows where unanswered."""
    last_here = code[:, :, 1 + FEATURES.index("last seen here")]
    t_last, t_a, t_b, t_ours, t_mates = (code[:, :, 1 + len(FEATURES) + len(RATE_FEATURES) + k] for k in range(5))
    X = np.full(code.shape[:2] + (len(NEW_JEV),), np.nan)
    for i, m in enumerate(moments):
        row = cache.get(f"{group_key(m)}|{m['label']}")
        if row is None:
            continue
        a = row["answers"]
        h = a["heading"]["probabilities"]
        aggr = _mean_score(a["aggression"], len(AGGRESSION)) - 0.5
        dist = _mean_score(a["distance"], len(DISTANCE))
        lurk = float(a["lurking"]["noul"]) - 0.5
        L = last_here[i]
        X[i] = np.stack([
            _logit(float(a["stay"]["noul"])) * L, h.get("holding", 0.0) * L, -h.get("to_a", 0.0) * t_a[i],
            -h.get("to_b", 0.0) * t_b[i], -h.get("to_your_team", 0.0) * t_ours[i], h.get("away", 0.0) * t_ours[i],
            aggr * L, -aggr * t_ours[i], lurk * t_mates[i], dist * t_last[i], dist * L,
        ], axis=1)  # fmt: skip
    return X


# ---------------------------------------------------------------------- the stacked model
def _fit(X: np.ndarray, y: np.ndarray, lam: float) -> tuple[np.ndarray, np.ndarray]:
    """Penalised conditional logit; column 0 is log cspredict. Returns weights and feature scales."""
    scale = X.reshape(-1, X.shape[2]).std(axis=0) + 1e-9
    scale[0] = 1.0
    Z = X / scale
    w0 = np.zeros(X.shape[2])
    w0[0] = 1.0
    rows = np.arange(len(y))

    def loss(w):
        z = Z @ w
        z = z - z.max(axis=1, keepdims=True)
        q = np.exp(z)
        q /= q.sum(axis=1, keepdims=True)
        onehot = np.zeros_like(q)
        onehot[rows, y] = 1.0
        nll = -np.log(np.clip(q[rows, y], 1e-12, None)).mean()
        grad = np.einsum("ncf,nc->f", Z, q - onehot) / len(y)
        return nll + lam * ((w - w0) ** 2).sum(), grad + 2 * lam * (w - w0)

    w = minimize(loss, w0, jac=True, method="L-BFGS-B").x
    return w, scale


def _predict(X: np.ndarray, w: np.ndarray, scale: np.ndarray) -> np.ndarray:
    z = (X / scale) @ w
    z = z - z.max(axis=1, keepdims=True)
    q = np.exp(z)
    return q / q.sum(axis=1, keepdims=True)


def held_out(X: np.ndarray, y: np.ndarray, rounds: np.ndarray, lam: float = LAMBDA) -> np.ndarray:
    """Held-out probabilities over folds of whole rounds, averaged over several fold draws."""
    ids = np.unique(rounds)
    out = np.zeros(X.shape[:2])
    for seed in range(FOLD_SEEDS):
        fold_of = dict(zip(ids, np.random.default_rng(seed).permutation(len(ids)) % FOLDS))
        fold = np.array([fold_of[r] for r in rounds])
        for f in range(FOLDS):
            tr, te = fold != f, fold == f
            w, scale = _fit(X[tr], y[tr], lam)
            out[te] += _predict(X[te], w, scale) / FOLD_SEEDS
    return out


def report(named: dict[str, np.ndarray], y: np.ndarray, rounds: np.ndarray, pairs: list[tuple[str, str]], title: str) -> pl.DataFrame:
    rows = []
    for name, p in named.items():
        met = _metrics(p, y)
        lo, hi = _cluster_ci(met["logloss"], rounds)
        rows.append({"model": name, "top1": met["top1"].mean(), "top3": met["top3"].mean(),
                     "logloss": met["logloss"].mean(), "ll_lo": lo, "ll_hi": hi})  # fmt: skip
    table = pl.DataFrame(rows)
    print(f"\n== {title} ==")
    print(table)
    print("Paired differences (95% intervals resample whole rounds; negative log-loss = better):")
    for a, b in pairs:
        ma, mb = _metrics(named[a], y), _metrics(named[b], y)
        parts = []
        for k in ("top1", "top3", "logloss"):
            d = ma[k] - mb[k]
            lo, hi = _cluster_ci(d, rounds)
            parts.append(f"{k} {d.mean():+.3f} ({lo:+.3f} to {hi:+.3f})")
        print(f"  {a}  minus  {b}: " + ", ".join(parts))
    return table


# ---------------------------------------------------------------------- role-play: Jev plays the enemy
ROLEPLAY_INSTRUCTIONS = (
    "Put yourself in the shoes of {label}, a professional {side} player on Mirage. The state is what the "
    "opposing {friendly} team knew about you and your team: where you were last seen {since} s ago, the "
    "bomb, the time left, the players alive, the kill feed, smokes and molotovs. Thinking like a {side} "
    "player in this situation, where would you be right now? Pick the callout you would most likely be "
    "in. Every option lists facts worked out in code: whether you could have reached it since you were "
    "last seen (pros usually shift-walk, which is silent), whether the {friendly} team can see it now or "
    "watched it since, and how often {side} players are there at this point of a round in pro matches."
)


def build_roleplay(moments: list[dict], grid) -> dict[str, dict]:
    """jev_lab's focus requests with only the instructions changed."""
    requests = build_lab(moments, grid, "focus")
    for m in moments:
        ev = m["evidence"]
        req = requests[f"{group_key(m)}|{m['label']}"]
        req["questions"]["where"]["instructions"] = ROLEPLAY_INSTRUCTIONS.format(
            label=m["label"], side="T" if m["friendly"] == "ct" else "CT", friendly=m["friendly"].upper(),
            since=ev["seconds_since_seen"],
        )  # fmt: skip
    return requests


def pick_features(p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Jev's answer as stacked inputs: (n, C, 1) "this is Jev's pick" (ties share it), and
    (n, C, 1) log of Jev's probability."""
    top = (p == p.max(axis=1, keepdims=True)).astype(float)
    return (top / top.sum(axis=1, keepdims=True))[:, :, None], np.log(p + 0.01)[:, :, None]


def roleplay_main(args) -> None:
    moments = load_moments("val", 2000, 0, 8)
    grid = load_models(model_dir(["hltv", "xego"])).grid
    names = callouts(grid)
    requests = build_roleplay(moments, grid)
    tokens = sum(len(json.dumps(r)) for r in requests.values()) / 4
    logger.info(f"role-play: {len(requests)} requests, ~{tokens / 1e6:.2f}M input tokens (~${tokens / 1e6 * PRICE_PER_MTOK:.3f})")
    if args.dry_run:
        print(json.dumps(next(iter(requests.values()))["questions"]["where"]["instructions"], indent=1))
        return
    cache_path = OUT / "answers_roleplay.jsonl"
    if not args.no_ask:
        if _api_key() is None:
            raise SystemExit("Set TYPESAFE_API_KEY (environment or .env)")
        asyncio.run(_ask(requests, cache_path, args.concurrency, args.per_second, model=MODEL))
    cache = _load_cache(cache_path)
    used = sum(cache[k]["usage"]["input_tokens"] or 0 for k in requests if k in cache)
    logger.info(f"{sum(k in cache for k in requests)}/{len(requests)} answered, {used:,} input tokens (~${used / 1e6 * PRICE_PER_MTOK:.3f})")

    p_role = jev_probs(moments, cache, "focus", names)  # same answer format as focus
    p_focus = jev_probs(moments, _load_cache(OUTPUT_DIR / "jev" / "answers_focus.jsonl"), "focus", names)
    ok = ~np.isnan(p_role).any(axis=1) & ~np.isnan(p_focus).any(axis=1)
    moments = [m for m, k in zip(moments, ok) if k]
    p_role, p_focus = p_role[ok], p_focus[ok]
    y = np.array([m["truth"] for m in moments])
    rounds = np.array([f"{m['demo']}|{m['round']}" for m in moments])
    last = np.array([names.index(m["evidence"]["last_seen_at"]) for m in moments])
    code = code_features(moments, grid)

    print(f"\n== Jev on its own, {len(y):,} validation moments (raw answers, ties share credit) ==")
    for name, p in (("focus: which callout is enemy T1 in?", p_focus), ("role-play: you are T1, where would you be?", p_role)):
        met = _metrics(p, y)
        lo, hi = _cluster_ci(met["top1"], rounds)
        picks_last = (p[np.arange(len(y)), last] == p.max(axis=1)).mean()
        print(f"  {name:45s} top-1 {met['top1'].mean():.3f} ({lo:.3f}-{hi:.3f}), top-3 {met['top3'].mean():.3f}, "
              f"raw log-loss {met['logloss'].mean():.2f}, picks the last-seen callout {picks_last:.0%}")  # fmt: skip
    d = _metrics(p_role, y)["top1"] - _metrics(p_focus, y)["top1"]
    lo, hi = _cluster_ci(d, rounds)
    print(f"  role-play minus focus, top-1: {d.mean():+.3f} ({lo:+.3f} to {hi:+.3f})")
    print(f"  the two pick the same callout in {(p_role.argmax(1) == p_focus.argmax(1)).mean():.0%} of moments; "
          f"truly still in the last-seen callout: {(y == last).mean():.0%}")

    pick_role, logp_role = pick_features(p_role)
    pick_focus, logp_focus = pick_features(p_focus)
    variants = {
        "cspredict": np.exp(code[:, :, 0]),
        "stack: code facts (no Jev)": held_out(code, y, rounds),
        "stack + focus pick": held_out(np.concatenate([code, pick_focus], axis=2), y, rounds),
        "stack + focus log-probability": held_out(np.concatenate([code, logp_focus], axis=2), y, rounds),
        "stack + role-play pick": held_out(np.concatenate([code, pick_role], axis=2), y, rounds),
        "stack + role-play log-probability": held_out(np.concatenate([code, logp_role], axis=2), y, rounds),
        "cspredict + role-play pick only": held_out(np.concatenate([code[:, :, :1], pick_role], axis=2), y, rounds),
    }
    base = "stack: code facts (no Jev)"
    pairs = [("stack + role-play pick", base), ("stack + role-play log-probability", base),
             ("stack + role-play pick", "stack + focus pick"), ("cspredict + role-play pick only", "cspredict")]  # fmt: skip
    report(variants, y, rounds, pairs, f"stacked on cspredict, {len(y):,} validation moments in {len(np.unique(rounds))} rounds, held out")

    # How much weight Jev's pick gets: its odds are multiplied by exp(weight); 95% interval over rounds.
    def pick_weight(X: np.ndarray, idx: np.ndarray) -> float:
        w, scale = _fit(X[idx], y[idx], LAMBDA)
        return float(w[-1] / scale[-1])

    ids, inv = np.unique(rounds, return_inverse=True)
    members = [np.flatnonzero(inv == k) for k in range(len(ids))]
    rng = np.random.default_rng(0)
    print("\nWeight on Jev's pick (its odds are multiplied by this; 1 = ignored), 95% interval resampling rounds:")
    for name, X in (("on cspredict alone", np.concatenate([code[:, :, :1], pick_role], axis=2)),
                    ("on cspredict + code facts", np.concatenate([code, pick_role], axis=2)),
                    ("focus pick, on cspredict + code facts", np.concatenate([code, pick_focus], axis=2))):  # fmt: skip
        full = pick_weight(X, np.arange(len(y)))
        boot = [pick_weight(X, np.concatenate([members[k] for k in rng.integers(0, len(ids), len(ids))])) for _ in range(200)]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        print(f"  {name:40s} x{np.exp(full):.2f} ({np.exp(lo):.2f} to {np.exp(hi):.2f})")


# ---------------------------------------------------------------------- context: the match so far
GAME_BASICS = {
    "objectives": "Ts win a round by planting the bomb on Bombsite A or B and keeping it from being defused "
                  "until it explodes (40 s after the plant), or by killing every CT. CTs win by killing every T "
                  "before a plant, by defusing the bomb, or when the round time (1:55) runs out with no plant.",
    "typical_play": "Before a plant, Ts gather information and then commit to one site, often with a lurker "
                    "playing alone elsewhere; late in the round they must commit. CTs hold the sites and mid, "
                    "then rotate once the Ts show where they are going. After a plant, Ts hold angles around "
                    "the site and CTs retake it. A player left alone against several often hides, saves the "
                    "weapon or plays for time. Teams short of money play more passively.",
    "match_format": "First to 13 rounds wins; teams swap sides after 12 rounds.",
}  # fmt: skip


def _demo_context(ref, grid, wanted: list[dict]) -> dict[str, dict]:
    """Context per moment of one demo, from rounds before the moment's round (and nothing later)."""
    names = callouts(grid)
    ticks = ref.table("ticks").select("round_num", "tick", "steamid", "side", "is_alive", "spotted", "X", "Y", "Z")
    rounds, kills = ref.table("rounds"), ref.table("kills")
    number = match_rounds(rounds, kills)
    winner = dict(rounds.select("round_num", "winner").iter_rows())
    playing = ticks.filter(pl.col("is_alive"))
    side_of = {(int(r), int(sid)): side for r, sid, side in
               playing.group_by("round_num", "steamid").agg(pl.col("side").mode().first()).iter_rows()}  # fmt: skip
    seen = playing.filter(pl.col("spotted")).sort("tick")
    seen = seen.with_columns(pl.Series("callout", [names[i] for i in grid.node_place_idx[
        grid.locate(seen["X"].to_numpy(), seen["Y"].to_numpy(), seen["Z"].to_numpy())]]))  # fmt: skip
    seen_rounds = seen.partition_by("round_num", "steamid", as_dict=True)
    kills = kills.sort("tick")
    first_kill = {int(r): g.row(0, named=True) for (r,), g in kills.partition_by("round_num", as_dict=True).items()}
    out = {}
    for m in wanted:
        R = int(m["round"])
        enemy = "t" if m["friendly"] == "ct" else "ct"
        ids = np.sort(playing.filter((pl.col("round_num") == R) & (pl.col("side") == enemy))["steamid"].unique().to_numpy())
        sid, team = int(ids[m["enemy"]]), [int(x) for x in ids]
        earlier = sorted((r for r in number if number[r] < number.get(R, 0)), key=number.get)
        results = []  # from the enemy team's point of view
        team_side = {}
        for r in number:
            sides = [side_of[(r, x)] for x in team if (r, x) in side_of]
            if sides:
                team_side[r] = Counter(sides).most_common(1)[0][0]
        for r in earlier:
            side = team_side.get(r) or next(  # no ticks for r (late recording): sides are fixed within a half
                (team_side[q] for q in sorted(team_side, key=lambda q: abs(number[q] - number[r]))
                 if half_start(number[q]) == half_start(number[r])), None)  # fmt: skip
            if side is not None:
                results.append(winner.get(r) == side)
        prior = kills.filter(pl.col("round_num").is_in(earlier))
        mine = prior.filter(pl.col("attacker_steamid") == sid)
        weapons = Counter(w for w in mine["weapon"].to_list() if w and w != "world")
        opening = [first_kill[r] for r in earlier if r in first_kill]
        same_side = [r for r in earlier if side_of.get((r, sid)) == enemy]  # rounds with no ticks have no sightings either
        any_at, first_at, never = Counter(), Counter(), 0
        for r in same_side:
            g = seen_rounds.get((r, sid))
            if g is None or g.height == 0:
                never += 1
                continue
            any_at.update(set(g["callout"].to_list()))
            first_at[g["callout"][0]] += 1
        label, side = m["label"], enemy.upper()
        out[f"{group_key(m)}|{label}"] = {
            "score": {"you": sum(results), "them": len(results) - sum(results)},
            "last_five": ["won" if w else "lost" for w in results[-5:]][::-1],
            "you_this_match": {
                "rounds_played_before_this_one": len(earlier),
                "kills": mine.height,
                "deaths": prior.filter(pl.col("victim_steamid") == sid).height,
                "opening_kills": sum(o["attacker_steamid"] == sid for o in opening),
                "opening_deaths": sum(o["victim_steamid"] == sid for o in opening),
                "main_weapon": weapons.most_common(1)[0][0] if weapons else "no kills yet",
            },
            "habits": {
                "earlier_rounds_on_this_side": len(same_side),
                "rounds_not_seen_at_all": never,
                "first_seen_at": dict(first_at.most_common(6)),
                "seen_at_some_point_in": dict(any_at.most_common(8)),
            },
            "side": side,
        }
    return out


def match_context(moments: list[dict], grid) -> dict[str, dict]:
    path = OUT / "context_val.json"
    if path.exists():
        return json.loads(path.read_text())
    refs = {r.demo_id: r for r in list_demos(["hltv"], ["val"])}
    by_demo: dict[str, list[dict]] = {}
    for m in moments:
        by_demo.setdefault(m["demo"], []).append(m)
    ctx = {}
    for demo, ms in by_demo.items():
        logger.info(f"context: {demo[:60]} ({len(ms)} moments)")
        ctx |= _demo_context(refs[demo], grid, ms)
    path.write_text(json.dumps(ctx))
    return ctx


CONTEXT_SENTENCE = {
    "ctx": " Use also the basics of the game, the match score and how you have played this match so far.",
    "ctx_habits": " Use also the basics of the game, the match score, how you have played this match so far, "
                  "and where the {friendly} team has spotted you in earlier rounds on this side (players "
                  "often repeat their positions).",
}  # fmt: skip


def build_context(moments: list[dict], grid, ctx: dict[str, dict], variant: str) -> dict[str, dict]:
    """The role-play requests plus the match context (variant "ctx" or "ctx_habits")."""
    requests = build_roleplay(moments, grid)
    for m in moments:
        key = f"{group_key(m)}|{m['label']}"
        c, label, friendly = ctx[key], m["label"], m["friendly"].upper()
        req = requests[key]
        req["state"] = {
            "game_basics": GAME_BASICS,
            **req["state"],
            "match_score_before_this_round": {f"{label}'s team ({c['side']})": c["score"]["you"], f"{friendly} team": c["score"]["them"]},
            f"{label}'s_team_last_five_rounds_newest_first": c["last_five"] or ["none yet"],
            f"{label}_this_match": c["you_this_match"],
        }  # fmt: skip
        if variant == "ctx_habits":
            h = c["habits"]
            req["state"][f"where_the_{friendly}_team_has_seen_{label}_in_earlier_rounds_on_this_side"] = {
                "earlier_rounds_on_this_side": h["earlier_rounds_on_this_side"],
                "rounds_not_seen_at_all": h["rounds_not_seen_at_all"],
                "first_seen_at (rounds)": h["first_seen_at"] or "none",
                "seen_at_some_point_in (rounds)": h["seen_at_some_point_in"] or "none",
            }
        req["questions"]["where"]["instructions"] += CONTEXT_SENTENCE[variant].format(friendly=friendly)
    return requests


def habit_features(moments: list[dict], ctx: dict[str, dict], names: list[str]) -> np.ndarray:
    """(n, C, 2) code-side habits: the share of earlier same-side rounds this player was first seen
    in / seen at some point in each callout (0 when there are no earlier rounds)."""
    col = {c: i for i, c in enumerate(names)}
    X = np.zeros((len(moments), len(names), 2))
    for i, m in enumerate(moments):
        h = ctx[f"{group_key(m)}|{m['label']}"]["habits"]
        n = h["earlier_rounds_on_this_side"]
        for k, field in enumerate(("first_seen_at", "seen_at_some_point_in")):
            for c, cnt in h[field].items():
                X[i, col[c], k] = cnt / max(n, 1)
    return X


def context_main(args) -> None:
    moments = load_moments("val", 2000, 0, 8)
    grid = load_models(model_dir(["hltv", "xego"])).grid
    names = callouts(grid)
    ctx = match_context(moments, grid)
    variants = ("ctx", "ctx_habits")
    for v in variants:
        requests = build_context(moments, grid, ctx, v)
        tokens = sum(len(json.dumps(r)) for r in requests.values()) / 4
        logger.info(f"{v}: {len(requests)} requests, ~{tokens / 1e6:.2f}M input tokens (~${tokens / 1e6 * PRICE_PER_MTOK:.3f})")
        if args.dry_run:
            r = next(iter(requests.values()))
            print(json.dumps({k: x for k, x in r["state"].items() if k not in ("enemy_asked_about", "its_teammates")}, indent=1))
            print(r["questions"]["where"]["instructions"])
            continue
        if not args.no_ask:
            if _api_key() is None:
                raise SystemExit("Set TYPESAFE_API_KEY (environment or .env)")
            asyncio.run(_ask(requests, OUT / f"answers_{v}.jsonl", args.concurrency, args.per_second, model=MODEL))
        cache = _load_cache(OUT / f"answers_{v}.jsonl")
        used = sum(cache[k]["usage"]["input_tokens"] or 0 for k in requests if k in cache)
        logger.info(f"{v}: {sum(k in cache for k in requests)}/{len(requests)} answered, {used:,} input tokens (~${used / 1e6 * PRICE_PER_MTOK:.3f})")
    if args.dry_run:
        return

    probs = {"role-play": jev_probs(moments, _load_cache(OUT / "answers_roleplay.jsonl"), "focus", names)}
    for v in variants:
        probs[v] = jev_probs(moments, _load_cache(OUT / f"answers_{v}.jsonl"), "focus", names)
    ok = np.all([~np.isnan(p).any(axis=1) for p in probs.values()], axis=0)
    moments = [m for m, k in zip(moments, ok) if k]
    probs = {k: p[ok] for k, p in probs.items()}
    y = np.array([m["truth"] for m in moments])
    rounds = np.array([f"{m['demo']}|{m['round']}" for m in moments])
    last = np.array([names.index(m["evidence"]["last_seen_at"]) for m in moments])
    code = code_features(moments, grid)
    habits = habit_features(moments, ctx, names)
    n_earlier = np.array([ctx[f"{group_key(m)}|{m['label']}"]["habits"]["earlier_rounds_on_this_side"] for m in moments])

    print(f"\n== Jev on its own, {len(y):,} validation moments (raw answers, ties share credit) ==")
    for name, p in probs.items():
        met = _metrics(p, y)
        lo, hi = _cluster_ci(met["top1"], rounds)
        picks_last = (p[np.arange(len(y)), last] == p.max(axis=1)).mean()
        print(f"  {name:12s} top-1 {met['top1'].mean():.3f} ({lo:.3f}-{hi:.3f}), top-3 {met['top3'].mean():.3f}, "
              f"raw log-loss {met['logloss'].mean():.2f}, picks the last-seen callout {picks_last:.0%}")  # fmt: skip
    for v in variants:
        d = _metrics(probs[v], y)["top1"] - _metrics(probs["role-play"], y)["top1"]
        lo, hi = _cluster_ci(d, rounds)
        print(f"  {v} minus role-play, top-1: {d.mean():+.3f} ({lo:+.3f} to {hi:+.3f})")
    late = n_earlier >= 4
    d = _metrics(probs["ctx_habits"], y)["top1"] - _metrics(probs["ctx"], y)["top1"]
    lo, hi = _cluster_ci(d[late], rounds[late])
    print(f"  ctx_habits minus ctx, top-1, moments with 4+ earlier rounds on this side ({late.mean():.0%}): {d[late].mean():+.3f} ({lo:+.3f} to {hi:+.3f})")
    habit_top = habits[:, :, 0] + 0.01 * habits[:, :, 1]
    has = habit_top.max(axis=1) > 0
    pick = habit_top.argmax(axis=1)
    print(f"  code only, \"where this player was first seen most often\": right in {(pick[has] == y[has]).mean():.1%} "
          f"of the {has.mean():.0%} moments with a history")

    pick_of = {k: pick_features(p)[0] for k, p in probs.items()}
    stack = np.concatenate([code, habits], axis=2)
    variants_ = {
        "cspredict": np.exp(code[:, :, 0]),
        "stack: code facts (no Jev)": held_out(code, y, rounds),
        "stack: code facts + habit counts (no Jev)": held_out(stack, y, rounds),
        "  + role-play pick": held_out(np.concatenate([stack, pick_of["role-play"]], axis=2), y, rounds),
        "  + ctx pick": held_out(np.concatenate([stack, pick_of["ctx"]], axis=2), y, rounds),
        "  + ctx_habits pick": held_out(np.concatenate([stack, pick_of["ctx_habits"]], axis=2), y, rounds),
        "cspredict + ctx_habits pick only": held_out(np.concatenate([code[:, :, :1], pick_of["ctx_habits"]], axis=2), y, rounds),
    }
    base = "stack: code facts + habit counts (no Jev)"
    pairs = [(base, "stack: code facts (no Jev)"), ("  + ctx pick", base), ("  + ctx_habits pick", base),
             ("  + ctx_habits pick", "  + role-play pick"), ("cspredict + ctx_habits pick only", "cspredict")]  # fmt: skip
    table = report(variants_, y, rounds, pairs, f"stacked on cspredict, {len(y):,} validation moments in {len(np.unique(rounds))} rounds, held out")
    table.write_csv(OUT / "context_results.csv")

    def pick_weight(X: np.ndarray, idx: np.ndarray) -> float:
        w, scale = _fit(X[idx], y[idx], LAMBDA)
        return float(w[-1] / scale[-1])

    ids, inv = np.unique(rounds, return_inverse=True)
    members = [np.flatnonzero(inv == k) for k in range(len(ids))]
    rng = np.random.default_rng(0)
    print("\nWeight on Jev's pick (its odds are multiplied by this; 1 = ignored), 95% interval resampling rounds:")
    for name, X in (("ctx_habits, on cspredict alone", np.concatenate([code[:, :, :1], pick_of["ctx_habits"]], axis=2)),
                    ("role-play, on code facts + habits", np.concatenate([stack, pick_of["role-play"]], axis=2)),
                    ("ctx, on code facts + habits", np.concatenate([stack, pick_of["ctx"]], axis=2)),
                    ("ctx_habits, on code facts + habits", np.concatenate([stack, pick_of["ctx_habits"]], axis=2))):  # fmt: skip
        full = pick_weight(X, np.arange(len(y)))
        boot = [pick_weight(X, np.concatenate([members[k] for k in rng.integers(0, len(ids), len(ids))])) for _ in range(200)]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        print(f"  {name:38s} x{np.exp(full):.2f} ({np.exp(lo):.2f} to {np.exp(hi):.2f})")


def main() -> None:
    ap = argparse.ArgumentParser(description="Jev answers as features on top of cspredict (validation only).")
    ap.add_argument("--n", type=int, default=300, help="validation moments to ask about (the first n of jev_lab's 2,000)")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--per-second", type=float, default=15.0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-ask", action="store_true")
    ap.add_argument("--roleplay", action="store_true", help="Jev plays the enemy, on all 2,000 validation moments")
    ap.add_argument("--context", action="store_true", help="role-play plus the match context, on all 2,000 validation moments")
    args = ap.parse_args()
    if args.roleplay or args.context:
        OUT.mkdir(parents=True, exist_ok=True)
        return roleplay_main(args) if args.roleplay else context_main(args)

    OUT.mkdir(parents=True, exist_ok=True)
    everything = load_moments("val", 2000, 0, 8)  # jev_lab's validation moments, uniform draws in u order
    moments = everything[: args.n]
    grid = load_models(model_dir(["hltv", "xego"])).grid
    requests = build(moments)
    tokens = sum(len(json.dumps(r)) for r in requests.values()) / 4
    logger.info(f"round 1: {len(requests)} requests, ~{tokens / 1e6:.2f}M input tokens (~${tokens / 1e6 * PRICE_PER_MTOK:.3f})")
    if args.dry_run:
        print(json.dumps(next(iter(requests.values())), indent=1)[:6000])
        return
    cache_path = OUT / "answers_round1.jsonl"
    if not args.no_ask:
        if _api_key() is None:
            raise SystemExit("Set TYPESAFE_API_KEY (environment or .env)")
        asyncio.run(_ask(requests, cache_path, args.concurrency, args.per_second, model=MODEL))
    cache = _load_cache(cache_path)
    used = [cache[k]["usage"]["input_tokens"] or 0 for k in requests if k in cache]
    logger.info(f"{len(used)}/{len(requests)} answered, {sum(used):,} input tokens (~${sum(used) / 1e6 * PRICE_PER_MTOK:.3f})")

    # All 2,000 validation moments: does Jev's existing split_rates answer add to cspredict?
    y_all = np.array([m["truth"] for m in everything])
    rounds_all = np.array([f"{m['demo']}|{m['round']}" for m in everything])
    code_all = code_features(everything, grid)
    old_all = old_jev_features(everything, grid)
    named = {"cspredict": np.exp(code_all[:, :, 0]), "stack: code facts (no Jev)": held_out(code_all, y_all, rounds_all)}
    if old_all is not None:
        named["stack: code facts + old Jev answers"] = held_out(np.concatenate([code_all, old_all], axis=2), y_all, rounds_all)
    report(named, y_all, rounds_all, [("stack: code facts (no Jev)", "cspredict")]
           + ([("stack: code facts + old Jev answers", "stack: code facts (no Jev)")] if old_all is not None else []),
           f"all {len(everything):,} validation moments, no new requests")  # fmt: skip

    # The pilot moments: do the round 1 answers add to cspredict?
    code = code_all[: args.n]
    new = new_jev_features(moments, cache, code)
    ok = ~np.isnan(new).any(axis=(1, 2))
    y, rounds = y_all[: args.n][ok], rounds_all[: args.n][ok]
    code, new = code[ok], new[ok]
    old = old_all[: args.n][ok] if old_all is not None else np.zeros(code.shape[:2] + (0,))
    variants = {
        "cspredict": np.exp(code[:, :, 0]),
        "stack: code facts (no Jev)": held_out(code, y, rounds),
        "stack: code facts + old Jev answers": held_out(np.concatenate([code, old], axis=2), y, rounds),
        "stack: code facts + round 1 Jev answers": held_out(np.concatenate([code, new], axis=2), y, rounds),
        "stack: code facts + old + round 1": held_out(np.concatenate([code, old, new], axis=2), y, rounds),
    }
    base, r1 = "stack: code facts (no Jev)", "stack: code facts + round 1 Jev answers"
    table = report(variants, y, rounds, [(base, "cspredict"), (r1, base), ("stack: code facts + old + round 1", base),
                                         ("stack: code facts + old + round 1", "stack: code facts + old Jev answers"),
                                         (r1, "cspredict")],
                   f"pilot: {ok.sum()} validation moments in {len(np.unique(rounds))} rounds, held out over folds of whole rounds")  # fmt: skip
    table.write_csv(OUT / "pilot_results.csv")

    print(f"\nSensitivity to the penalty (fixed at {LAMBDA} before fitting), log-loss of round 1 minus no-Jev:")
    for lam in (0.001, 0.1):
        d = _metrics(held_out(np.concatenate([code, new], axis=2), y, rounds, lam), y)["logloss"] \
            - _metrics(held_out(code, y, rounds, lam), y)["logloss"]  # fmt: skip
        lo, hi = _cluster_ci(d, rounds)
        print(f"  lambda {lam}: {d.mean():+.3f} ({lo:+.3f} to {hi:+.3f})")

    # Feature importance for the next round's proposer: weights fitted on all pilot moments.
    X = np.concatenate([code, new], axis=2)
    w, scale = _fit(X, y, LAMBDA)
    names = ("log cspredict", *FEATURES, *RATE_FEATURES, *CODE_EXTRA, *NEW_JEV)
    imp = pl.DataFrame({"feature": names, "weight (standardised)": w}).with_columns(
        pl.col("weight (standardised)").abs().alias("abs")).sort("abs", descending=True).drop("abs")
    pl.Config.set_tbl_rows(40)
    print("\nWeights fitted on all pilot moments (standardised; log cspredict starts at 1, the rest at 0):")
    print(imp)
    imp.write_csv(OUT / "pilot_weights.csv")
    answers = [cache[k]["answers"] for k in requests if k in cache]
    print("\nAnswer spread (round 1):")
    print(f"  still there: mean {np.mean([a['stay']['noul'] for a in answers]):.2f}, "
          f"true rate {np.mean([m['truth'] == callouts(grid).index(m['evidence']['last_seen_at']) for m in moments]):.2f}")
    for k in HEADING:
        print(f"  heading {k}: mean {np.mean([a['heading']['probabilities'].get(k, 0.0) for a in answers]):.2f}")
    print(f"  aggression score: mean {np.mean([a['aggression']['score'] for a in answers]):.2f} of 0-3, sd {np.std([a['aggression']['score'] for a in answers]):.2f}")
    print(f"  lurking: mean {np.mean([a['lurking']['noul'] for a in answers]):.2f}")
    print(f"  distance score: mean {np.mean([a['distance']['score'] for a in answers]):.2f} of 0-3, sd {np.std([a['distance']['score'] for a in answers]):.2f}")


if __name__ == "__main__":
    main()
