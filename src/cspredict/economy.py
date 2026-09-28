"""The enemy team's buy (eco, force or full), as the friendly team can infer it.

Nobody sees the other team's money or equipment values. Two things are public: the round history
(the scoreboard: which rounds of this half each team won) and the guns of the enemies the team has
seen, on the radar or in the kill feed. The estimate combines them like a naive Bayes classifier:

    P(buy | what the team knows)  ~  P(buy | situation) * prod_e P(best gun enemy e has shown | buy)^TEMPER

- situation (history_key): the pistol round, overtime, the round after winning or losing the pistol
  round, round 3 after losing round 2, after a win, or after a loss with a loss-bonus count of 1, 2,
  3 or 4+. P(buy | situation) is how often teams in that situation bought eco, force or full in the
  training rounds, smoothed towards the overall rates. After a win teams full-buy 99% of the time;
  after losses it is a three-way guess.
- best gun (best_guns): the dearest class an enemy has shown so far this round, pistol < SMG <
  rifle < sniper. P(gun | buy) is counted over every step of every training round.
- TEMPER < 1 because teammates' guns are not independent given the buy (a team buys together). It
  was chosen on the training rounds.

On the professional validation maps the most likely class is right for 77% of rounds when freeze
time ends (the history alone) and at 92% of the scored moments (an enemy seen earlier, now hidden),
by when guns have been seen. Most of the remaining errors are force buys taken for full buys: a
force buy with cheap rifles looks like a full buy.

Recorded rounds in the trajectory and team libraries keep their true class (infostate.buy_type of
the team's equipment values): there everything is known. Only the round being tracked uses this
estimate, and matching then weights a recorded round by its expected similarity (buy_weights).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from cspredict.dataset import DemoRef
from cspredict.grid import NavGrid
from cspredict.infostate import MAX_LOSSES, BuyHistory, Episode, episodes

BUY_CLASSES = ("eco", "force", "full")
GUNS = ("pistol", "smg", "rifle", "sniper")  # cheapest to dearest; best_guns numbers them 1 to 4
_GUN_OF_CLASS = np.array([4, 3, 2, 1, 0], dtype=np.int8)  # WEAPON_CLASSES (sniper, rifle, smg, pistol, other) -> best_guns code
PRIOR_PSEUDO = 3.0  # rounds at the overall rates added to every situation's counts
TEMPER = 0.7  # exponent on the gun likelihoods; 0.7 fitted the training rounds best (0.5-1.0 are within 0.02 in log-loss)


def history_key(h: BuyHistory | None) -> str:
    """The economic situation a round history puts a team in."""
    if h is None:
        return "unknown"
    if h.overtime:
        return "overtime"
    if h.half_round == 1:
        return "pistol round"
    if h.half_round == 2:
        return "after winning the pistol round" if h.won_last else "after losing the pistol round"
    if h.won_last:
        return "after a win"
    if h.half_round == 3:
        return "lost round 2 after winning the pistol round" if h.won_pistol else "lost the pistol round and round 2"
    return f"after a loss, loss count {min(h.losses, MAX_LOSSES)}"


def best_guns(ep: Episode) -> np.ndarray:
    """(S, E) dearest gun each enemy has shown so far this round, on the radar or in the kill feed:
    0 none yet, then 1 to 4 for GUNS."""

    def code(weapon: np.ndarray) -> np.ndarray:
        return np.where(weapon >= 0, _GUN_OF_CLASS[np.maximum(weapon, 0)], 0)

    shown = np.maximum(code(ep.enemy_seen_weapon), code(ep.enemy_kill_weapon))
    return np.maximum.accumulate(shown, axis=0)


def buy_weights(buy: int | np.ndarray, mismatch: float) -> np.ndarray:
    """(3,) weight of a recorded round of each buy class when matching a round whose buy is `buy`,
    either a known class or P(eco, force, full). The weight falls by `mismatch` per class of
    difference; for an uncertain buy it is the expected weight."""
    p = np.eye(len(BUY_CLASSES))[buy] if isinstance(buy, (int, np.integer)) else np.asarray(buy, dtype=np.float64)
    c = np.arange(len(BUY_CLASSES))
    return (mismatch ** np.abs(c[:, None] - c[None, :])) @ p


@dataclass
class BuyModel:
    prior: dict[str, list[float]]  # situation -> P(eco, force, full)
    base: list[float]  # overall P(eco, force, full), for situations not met in training
    guns: list[list[float]]  # (3, len(GUNS)) P(best gun shown | buy)
    temper: float = TEMPER
    n_rounds: int = 0  # (round, side) pairs counted

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: Path) -> BuyModel:
        return cls(**json.loads(path.read_text()))

    def posterior(self, ep: Episode, weapons: bool = True) -> np.ndarray:
        """(S, 3) P(eco, force, full) of the enemy team at every step: from the round history, and with
        `weapons` also from the guns the enemies have shown up to and including that step."""
        p = np.asarray(self.prior.get(history_key(ep.enemy_history), self.base))
        logp = np.tile(np.log(p), (ep.n_steps, 1))
        if weapons:
            best = best_guns(ep)
            shown = np.stack([(best == k + 1).sum(axis=1) for k in range(len(GUNS))], axis=1)  # (S, 4) enemies per best gun
            logp = logp + self.temper * shown @ np.log(np.asarray(self.guns)).T
        q = np.exp(logp - logp.max(axis=1, keepdims=True))
        return q / q.sum(axis=1, keepdims=True)


def fit_buy_model(grid: NavGrid, refs: list[DemoRef]) -> BuyModel:
    """Count situations and guns shown over the training rounds, from both sides' view."""
    counts: dict[str, np.ndarray] = {}
    guns = np.ones((len(BUY_CLASSES), len(GUNS)))  # add-one smoothing
    for ref in refs:
        for friendly in ("ct", "t"):
            for ep in episodes(grid, ref, friendly):
                counts.setdefault(history_key(ep.enemy_history), np.zeros(len(BUY_CLASSES)))[ep.enemy_buy_true] += 1
                best = best_guns(ep)
                guns[ep.enemy_buy_true] += np.bincount(best[best > 0] - 1, minlength=len(GUNS))
    total = np.sum(list(counts.values()), axis=0)
    base = total / total.sum()
    prior = {k: ((c + PRIOR_PSEUDO * base) / (c.sum() + PRIOR_PSEUDO)).tolist() for k, c in sorted(counts.items())}
    return BuyModel(prior=prior, base=base.tolist(), guns=(guns / guns.sum(axis=1, keepdims=True)).tolist(), n_rounds=int(total.sum()))
