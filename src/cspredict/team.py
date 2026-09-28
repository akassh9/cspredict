"""Team coordination: whole recorded teams as a joint prior over where the enemies are.

The per-enemy filters track every enemy on their own, so the team heatmap is a sum of independent
beliefs: seeing three Ts on B says nothing about the other two. Real teams move together. They
stack a site, send one lurker, and CTs hold two-one-two. This module corrects each enemy's callout
probabilities with recorded team snapshots:

1. Candidates. Snapshots of the enemy side's alive players (their callouts) from training rounds,
   taken every SNAP_EVERY_S seconds, in a similar situation: the same bomb phase, a similar clock
   (round time before a plant, time since the plant after one), softly the same buy (the friendly
   team's estimate of the enemy's, economy.py), and at least as many players alive as enemies alive
   now.
2. Pairing. Our enemies are paired with a snapshot's players. A pairing is scored by how well each
   recorded player's callout fits the matching enemy's own belief, as a likelihood ratio
   L_e(c) = (P_e(c) / pi(c))^beta, where P_e is the per-enemy filter's callout belief and pi the
   candidates' usual occupancy. With at most five enemies all pairings (at most 120) are summed:
   the exact version of assigning players by Hungarian matching.
3. Correction. The weighted snapshots give each enemy a team-aware callout distribution Q_e.
   Each enemy's cell belief is rescaled so its callout totals follow Q_e, relative to what the same
   weighting gives when teammates are independent. So the correction is exactly 1 when recorded
   teams carry no information about how positions are correlated. Q_e is shrunk towards that
   independent answer when only a few recorded teams fit what was seen (a small effective sample).

Chosen on validation (filters.ENS): shrinkage of 100 effective recorded teams, and no correction
before the first enemy has been seen (with no sightings every enemy's belief is the same, and
multiplying their identical likelihood ratios only exaggerates the shared negative evidence).
Keeping only each snapshot's best pairing (as Hungarian matching would) was clearly worse than
summing over all pairings.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import polars as pl

from cspredict.config import STEP_TICKS, TICKRATE
from cspredict.dataset import DemoRef
from cspredict.economy import buy_weights
from cspredict.grid import NavGrid
from cspredict.infostate import buy_type
from cspredict.motion import phase_expr

SIDE_ID = {"t": 0, "ct": 1}
MAX_PLAYERS = 5
SNAP_EVERY_S = 2.0  # spacing of stored snapshots within a round (about 2,000 rounds per side anyway)
CLOCK_SIGMA = 2.0  # seconds; how closely the clock must match
WINDOW = 1.5  # candidates within this many CLOCK_SIGMA
BUY_MISMATCH = 0.7  # weight per step of buy difference (as for particles)
EXTRA_ALIVE = 0.5  # weight per recorded player more than the enemies alive now
BETA = 1.0  # tempering of the per-enemy likelihood ratios
PSEUDO = 20.0  # shrinkage: effective recorded teams needed for the correction to count half (ens uses 100)
MAX_CANDIDATES = 5000  # clock-nearest candidates kept per step
PRIOR_FLOOR = 1e-3  # floor on callout probabilities in the likelihood ratios


@dataclass
class TeamLibrary:
    place: np.ndarray  # (Q, MAX_PLAYERS) callout index of each alive player, -1 for none
    n_alive: np.ndarray  # (Q,) players alive
    clock: np.ndarray  # (Q,) round time before the plant, time since the plant after
    buy: np.ndarray  # (Q,) the team's buy (0 eco, 1 force, 2 full)
    ptr: np.ndarray  # (2 * 3 + 1,) rows of key side * 3 + phase, each block sorted by clock

    def save(self, path: Path) -> None:
        np.savez_compressed(path, **self.__dict__)

    @classmethod
    def load(cls, path: Path) -> TeamLibrary:
        with np.load(path) as z:
            return cls(**{k: z[k] for k in cls.__dataclass_fields__})

    def block(self, side: int, phase: int) -> slice:
        key = side * 3 + phase
        return slice(int(self.ptr[key]), int(self.ptr[key + 1]))


def build_team_library(grid: NavGrid, refs: list[DemoRef]) -> TeamLibrary:
    every = int(round(SNAP_EVERY_S * TICKRATE / STEP_TICKS)) * STEP_TICKS
    frames = []
    for ref in refs:
        ticks = ref.table("ticks")
        rounds = ref.table("rounds").select("round_num", "freeze_end", "bomb_plant", "bomb_site")
        first = ticks.group_by("round_num").agg(pl.col("tick").min().alias("first"))
        buys = (
            ticks.join(first, on="round_num").filter(pl.col("tick") == pl.col("first"))
            .group_by("round_num", "side").agg(pl.col("equip").mean().alias("mean_equip"))
            .with_columns(pl.col("mean_equip").map_elements(buy_type, return_dtype=pl.Int8).alias("buy"))
        )  # fmt: skip
        t = ticks.join(rounds, on="round_num").filter((pl.col("tick") % every) == 0)
        t = t.filter(pl.col("is_alive")).with_columns(phase_expr())
        node = grid.locate(t["X"].to_numpy(), t["Y"].to_numpy(), t["Z"].to_numpy())
        t = t.with_columns(pl.Series("place", grid.node_place_idx[node].astype(np.int8)))
        clock = (
            pl.when(pl.col("phase") > 0).then((pl.col("tick") - pl.col("bomb_plant")) / TICKRATE)
            .otherwise((pl.col("tick") - pl.col("freeze_end")) / TICKRATE)
        )  # fmt: skip
        snaps = (
            t.with_columns(clock.cast(pl.Float32).alias("clock"))
            .group_by("round_num", "tick", "side")
            .agg(pl.col("place").sort().head(MAX_PLAYERS), pl.len().alias("n_alive"), pl.col("phase").first(), pl.col("clock").first())
            .join(buys.select("round_num", "side", "buy"), on=["round_num", "side"], how="left")
        )
        frames.append(snaps.with_columns(pl.lit(ref.demo_id).alias("demo")))
    df = pl.concat(frames).with_columns(
        pl.col("side").replace_strict(SIDE_ID, return_dtype=pl.Int8).alias("side_id"),
        pl.col("n_alive").clip(0, MAX_PLAYERS).cast(pl.Int8),
        pl.col("buy").fill_null(2).cast(pl.Int8),
    )
    df = df.with_columns((pl.col("side_id").cast(pl.Int32) * 3 + pl.col("phase").cast(pl.Int32)).alias("key")).sort("key", "clock")
    place = np.full((df.height, MAX_PLAYERS), -1, dtype=np.int8)
    for i, row in enumerate(df["place"].to_list()):
        place[i, : len(row)] = row
    key = df["key"].to_numpy()
    return TeamLibrary(
        place=place, n_alive=df["n_alive"].to_numpy(), clock=df["clock"].to_numpy(), buy=df["buy"].to_numpy(),
        ptr=np.searchsorted(key, np.arange(2 * 3 + 1)).astype(np.int64),
    )  # fmt: skip


@lru_cache(maxsize=None)
def pairing_maps(k: int, n: int) -> np.ndarray:
    """(M, k) every way to pair k enemies with distinct slots among n recorded players."""
    return np.array(list(itertools.permutations(range(n), k)), dtype=np.int64).reshape(-1, k)


@lru_cache(maxsize=None)
def pairings(k: int, n: int) -> np.ndarray:
    """(k * n, M) indicator of every way to pair k enemies with distinct slots among n recorded
    players: entry (e * n + j, map) is 1 when that pairing puts enemy e with slot j."""
    maps = pairing_maps(k, n)
    ind = np.zeros((k * n, len(maps)))
    for e in range(k):
        ind[e * n + maps[:, e], np.arange(len(maps))] = 1.0
    return ind


@dataclass(frozen=True)
class TeamParams:
    beta: float = BETA
    pseudo: float = PSEUDO
    clock_sigma: float = CLOCK_SIGMA
    extra_alive: float = EXTRA_ALIVE
    best_pairing: bool = False  # keep only each snapshot's best pairing (as Hungarian matching would)
    after_contact: bool = False  # only correct once some enemy has been seen this round


def team_callouts(
    lib: TeamLibrary, side: int, phase: int, clock: float, buy: int | np.ndarray | None, P: np.ndarray,
    params: TeamParams = TeamParams(),
) -> np.ndarray | None:
    """Correction factors R (k, C) for the callout beliefs P (k, C) of the k alive enemies, or
    None when no recorded team fits. Multiply enemy e's cell belief in callout c by R[e, c]. `buy` is
    the enemy team's buy, a class or P(eco, force, full) (None: no buy preference)."""
    k, n_places = P.shape
    blk = lib.block(side, phase)
    clk = lib.clock[blk]
    lo, hi = np.searchsorted(clk, [clock - WINDOW * params.clock_sigma, clock + WINDOW * params.clock_sigma])
    rows = np.arange(blk.start + lo, blk.start + hi)
    rows = rows[lib.n_alive[rows] >= k]
    if len(rows) == 0:
        return None
    if len(rows) > MAX_CANDIDATES:  # keep the clock-nearest
        rows = rows[np.argsort(np.abs(lib.clock[rows] - clock), kind="stable")[:MAX_CANDIDATES]]
    ctx = np.exp(-0.5 * ((lib.clock[rows] - clock) / params.clock_sigma) ** 2)
    if buy is not None:
        ctx *= buy_weights(buy, BUY_MISMATCH)[lib.buy[rows]]
    ctx *= params.extra_alive ** (lib.n_alive[rows].astype(np.int64) - k)

    # Usual occupancy of a recorded player in this situation.
    places = lib.place[rows]
    alive = places >= 0
    pi = np.bincount(places[alive], np.broadcast_to(ctx[:, None], places.shape)[alive], n_places)
    pi = np.maximum(pi / pi.sum(), PRIOR_FLOOR)
    Pf = np.maximum(P, PRIOR_FLOOR)
    ratio = (Pf / pi) ** params.beta  # (k, C)
    z = (pi * ratio).sum(axis=1)  # (k,) mean ratio of an independent player
    ratio = ratio / z[:, None]  # mean 1 for independent teammates

    log_ratio = np.log(ratio)
    num = np.zeros((k, n_places))
    den, sq = 0.0, 0.0
    for n in range(k, MAX_PLAYERS + 1):
        sel = lib.n_alive[rows] == n
        if not sel.any():
            continue
        pc = places[sel, :n]  # (m, n) callouts of the recorded players
        ind = pairings(k, n)  # (k * n, M)
        # log weight of each pairing: sum over enemies of the log ratio for their partner's callout
        lr = log_ratio[:, pc].transpose(1, 0, 2).reshape(len(pc), k * n)  # (m, k * n)
        w = np.exp(lr @ ind) * (ctx[sel] / ind.shape[1])[:, None]  # (m, M)
        if params.best_pairing:  # the single most likely pairing per snapshot
            best = w.argmax(axis=1)
            w = np.where(np.arange(w.shape[1])[None, :] == best[:, None], w, 0.0) * ind.shape[1]
        per_team = w.sum(axis=1)
        den += per_team.sum()
        sq += (per_team**2).sum()
        by_slot = (w @ ind.T).reshape(len(pc), k, n)  # weight of pairings putting enemy e with slot j
        for e in range(k):
            num[e] += np.bincount(pc.ravel(), by_slot[:, e, :].ravel(), n_places)
    if den <= 0:
        return None
    # With independent teammates enemy e's callout would follow pi * ratio_e (= P_e when beta = 1).
    indep = pi[None, :] * ratio
    ess = den**2 / max(sq, 1e-300)  # effective number of recorded teams behind the answer
    q = (ess * num / den + params.pseudo * indep) / (ess + params.pseudo)
    return q / np.maximum(indep, 1e-12)
