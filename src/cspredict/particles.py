"""Trajectory-library particle filter.

A first-order Markov chain on the grid spreads belief like diffusion, but real players run
along routes: 10 s after a sighting they are often 2,000+ units away, far outside a diffused
belief. Here every particle instead follows a real trajectory from the training demos that
passed through a similar state: same side, same or neighbouring node, similar heading, round
time, bomb phase, time since that player was last spotted (a player who was just seen
usually holds or falls back rather than running on), man advantage (a 1v3 plays differently
from a 5v5), team buy (recorded rounds store their true buy; the enemy's is the friendly team's
estimate from economy.py), and preferably the same player. Particles are weighted by negative
information and the kill feed like the grid filters, and are re-matched to a fresh trajectory
when theirs ends (death or round end) and when resampling duplicates them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
from scipy import sparse

from cspredict.config import STEP_TICKS, TICKRATE
from cspredict.dataset import DemoRef
from cspredict.economy import buy_weights
from cspredict.grid import NavGrid
from cspredict.infostate import OTHER_WEAPON, WEAPON_CLASSES, Episode, buy_type
from cspredict.motion import phase_expr, with_since_spotted

SIDE_ID = {"t": 0, "ct": 1}
MIN_CANDIDATES = 40  # widen to neighbouring nodes when a node has fewer recorded states
TIME_SIGMA = 15.0  # seconds; how strongly round time must match
PHASE_MISMATCH = 0.02  # weight for states from the other bomb phase
HEADING_KAPPA = 3.0  # von Mises concentration on heading agreement
MOVING = 20.0  # units per step (80 u/s) above which a heading is meaningful
SINCE_SIGMA = 0.5  # log(1 + seconds); how strongly time since last spotted must match
SINCE_MISMATCH = 0.05  # weight for seen-before vs never-seen mismatches
# Situation matching, tuned on validation (FACEIT data): a soft buy preference helped; matching
# the numbers alive (even softly) and preferring the same player did not, so both are off.
# Re-test them on pro demos, where teams are coordinated and the same players recur.
# Matching the weapon in hand at the last sighting (all classes, guns only, or AWP vs not) was
# within Monte Carlo noise on pro validation demos, so it is off too.
BUY_MISMATCH = 0.7  # weight per step of buy difference (eco / force / full)
ALIVE_SIGMA = np.inf  # players; how closely the numbers alive on each side must match
SAME_PLAYER = 1.0  # weight boost for the same player's own recorded trajectories
# Weight for recorded states whose weapon at their last sighting differs in class (sniper / rifle /
# smg / pistol / other) from the enemy's weapon when last seen. 1.0 turns weapon matching off; the
# filter configs set their own value (FilterConfig.weapon).
WEAPON_MISMATCH = 1.0
N_PARTICLES = 512
SNIPER = WEAPON_CLASSES.index("sniper")


@dataclass(frozen=True)
class Situation:
    """What the friendly team knows about one enemy's circumstances at a step."""

    t: float  # round time
    phase: int  # bomb phase
    since: float  # seconds since this enemy was last seen, inf if not yet this round
    n_team: int  # players alive on the enemy's side
    n_opp: int  # players alive on the friendly side
    buy: np.ndarray | None  # the enemy team's buy as P(eco, force, full); None for no buy preference
    player: int  # the enemy's steamid
    weapon: int = -1  # WEAPON_CLASSES index of the weapon in hand when last seen, -1 if never seen


@dataclass
class TrajectoryLibrary:
    node: np.ndarray  # (L,) node of each recorded state
    t: np.ndarray  # (L,) round time
    since: np.ndarray  # (L,) seconds since that player was last spotted, inf if not yet this round
    phase: np.ndarray  # (L,) 0 bomb not planted, 1 planted on A, 2 planted on B
    n_team: np.ndarray  # (L,) players alive on that player's side
    n_opp: np.ndarray  # (L,) players alive on the other side
    buy: np.ndarray  # (L,) that player's team buy this round
    player: np.ndarray  # (L,) steamid
    weapon: np.ndarray  # (L,) weapon class in hand at that player's last sighting, -1 if not yet seen
    side: np.ndarray  # (L,) 0 = T, 1 = CT
    nxt: np.ndarray  # (L,) next state of the same trajectory, -1 at its end
    vel: np.ndarray  # (L, 2) displacement over the previous step
    starts: np.ndarray  # indices of first states (freeze-time end)
    bucket_ptr: np.ndarray  # (2N + 1,) CSR pointers for key side * N + node
    bucket_idx: np.ndarray  # (L,) state indices sorted by key

    def save(self, path: Path) -> None:
        np.savez_compressed(path, **self.__dict__)

    @classmethod
    def load(cls, path: Path) -> TrajectoryLibrary:
        with np.load(path) as z:
            fields = {k: z[k] for k in cls.__dataclass_fields__ if k in z.files}
        fields.setdefault("weapon", np.full(len(fields["node"]), -1, dtype=np.int8))  # libraries built before weapons
        return cls(**fields)

    @property
    def n_nodes(self) -> int:
        return (len(self.bucket_ptr) - 1) // 2

    def bucket(self, side: int, node: int) -> np.ndarray:
        key = side * self.n_nodes + node
        return self.bucket_idx[self.bucket_ptr[key] : self.bucket_ptr[key + 1]]

    def candidates(self, grid: NavGrid, side: int, node: int) -> np.ndarray:
        cand = self.bucket(side, node)
        if len(cand) >= MIN_CANDIDATES:
            return cand
        ring = [node]
        seen = {node}
        for _ in range(3):  # widen up to three rings of neighbours
            ring = [j for i in ring for j in grid.adj_indices[grid.adj_indptr[i] : grid.adj_indptr[i + 1]] if j not in seen]
            seen.update(ring)
            cand = np.concatenate([cand, *[self.bucket(side, j) for j in ring]]) if ring else cand
            if len(cand) >= MIN_CANDIDATES:
                break
        return cand

    def sample(
        self,
        grid: NavGrid,
        rng: np.random.Generator,
        side: int,
        node: int,
        sit: Situation,
        k: int,
        heading: np.ndarray | None = None,
        weapon_mismatch: float = WEAPON_MISMATCH,
        weapon_mode: str = "all",
    ) -> np.ndarray:
        """k recorded states resembling the enemy's node, heading and situation."""
        cand = self.candidates(grid, side, node)
        if len(cand) == 0:
            cand = self.starts[self.side[self.starts] == side]
        w = np.exp(-0.5 * ((self.t[cand] - sit.t) / TIME_SIGMA) ** 2)
        w *= np.where(self.phase[cand] == sit.phase, 1.0, PHASE_MISMATCH)
        w *= _since_weight(self.since[cand], sit.since)
        alive_gap = (self.n_team[cand] - sit.n_team) ** 2 + (self.n_opp[cand] - sit.n_opp) ** 2
        w *= np.exp(-0.5 * alive_gap / ALIVE_SIGMA**2)
        if sit.buy is not None:
            w *= buy_weights(sit.buy, BUY_MISMATCH)[self.buy[cand]]
        w *= np.where(self.player[cand] == sit.player, SAME_PLAYER, 1.0)
        if sit.weapon >= 0 and weapon_mismatch != 1.0:
            w *= _weapon_weight(self.weapon[cand], sit.weapon, weapon_mismatch, weapon_mode)
        if heading is not None:
            speed = np.hypot(*heading)
            v = self.vel[cand]
            vs = np.hypot(v[:, 0], v[:, 1])
            if speed > MOVING:
                cos = (v @ heading) / np.maximum(vs * speed, 1e-6)
                w *= np.where(vs > MOVING, np.exp(HEADING_KAPPA * (cos - 1.0)), 0.3)
            else:
                w *= np.exp(-vs / 15.0)
        w = w + 1e-12
        return cand[rng.choice(len(cand), size=k, p=w / w.sum())]

    def sample_starts(self, rng: np.random.Generator, side: int, k: int) -> np.ndarray:
        """k trajectories starting when freeze time ends."""
        pool = self.starts[(self.side[self.starts] == side) & (self.t[self.starts] < 3.0)]
        return pool[rng.integers(0, len(pool), size=k)]


def _since_weight(lib_since: np.ndarray, since: float) -> np.ndarray:
    lib_never = ~np.isfinite(lib_since)
    if not np.isfinite(since):
        return np.where(lib_never, 1.0, SINCE_MISMATCH)
    gap = (np.log1p(np.where(lib_never, 0.0, lib_since)) - np.log1p(since)) / SINCE_SIGMA
    return np.where(lib_never, SINCE_MISMATCH, np.exp(-0.5 * gap**2))


def _weapon_weight(lib_weapon: np.ndarray, weapon: int, mismatch: float, mode: str = "all") -> np.ndarray:
    """1 for recorded states whose last-seen weapon class matches (or is unknown), else `mismatch`.
    mode "all" compares every class; "guns" treats a knife, grenade or bomb in hand as unknown;
    "awp" only compares sniper against not-sniper (also ignoring knife, grenade or bomb)."""
    if mode == "all":
        return np.where((lib_weapon < 0) | (lib_weapon == weapon), 1.0, mismatch)
    if weapon == OTHER_WEAPON:
        return np.ones(len(lib_weapon))
    unknown = (lib_weapon < 0) | (lib_weapon == OTHER_WEAPON)
    same = (lib_weapon == SNIPER) == (weapon == SNIPER) if mode == "awp" else lib_weapon == weapon
    return np.where(unknown | same, 1.0, mismatch)


def team_context(ticks: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Players alive per (round, tick, side), and each side's buy per round (team mean
    equipment value at the first in-play sample)."""
    alive = ticks.filter(pl.col("is_alive")).group_by("round_num", "tick", "side").len("n_alive")
    first = ticks.group_by("round_num").agg(pl.col("tick").min().alias("first"))
    buys = (
        ticks.join(first, on="round_num")
        .filter(pl.col("tick") == pl.col("first"))
        .group_by("round_num", "side")
        .agg(pl.col("equip").mean().alias("mean_equip"))
        .with_columns(pl.col("mean_equip").map_elements(buy_type, return_dtype=pl.Int8).alias("buy"))
    )
    return alive, buys.select("round_num", "side", "buy")


def build_library(grid: NavGrid, refs: list[DemoRef]) -> TrajectoryLibrary:
    frames = []
    for ref in refs:
        ticks = ref.table("ticks")
        alive, buys = team_context(ticks)
        other = alive.with_columns(pl.when(pl.col("side") == "t").then(pl.lit("ct")).otherwise(pl.lit("t")).alias("side"))
        t = with_since_spotted(ticks.filter(pl.col("is_alive")))
        t = t.filter(pl.col("tick") % STEP_TICKS == 0)
        t = (
            t.join(alive.rename({"n_alive": "n_team"}), on=["round_num", "tick", "side"], how="left")
            .join(other.rename({"n_alive": "n_opp"}), on=["round_num", "tick", "side"], how="left")
            .join(buys, on=["round_num", "side"], how="left")
        )
        r = ref.table("rounds").select("round_num", "freeze_end", "bomb_plant", "bomb_site")
        frames.append(
            t.join(r, on="round_num").select(
                pl.lit(ref.demo_id).alias("demo"), "round_num", "steamid", "tick", "side", "X", "Y", "Z",
                ((pl.col("tick") - pl.col("freeze_end")) / TICKRATE).cast(pl.Float32).alias("t"),
                pl.col("since").fill_null(np.inf),
                phase_expr(),
                pl.col("n_team").fill_null(0).cast(pl.Int8), pl.col("n_opp").fill_null(0).cast(pl.Int8),
                pl.col("buy").fill_null(2).cast(pl.Int8), pl.col("seen_weapon").cast(pl.Int8),
            )  # fmt: skip
        )
    df = pl.concat(frames).sort("demo", "round_num", "steamid", "tick")
    same_next = (
        (pl.col("demo").shift(-1) == pl.col("demo"))
        & (pl.col("round_num").shift(-1) == pl.col("round_num"))
        & (pl.col("steamid").shift(-1) == pl.col("steamid"))
        & (pl.col("tick").shift(-1) - pl.col("tick") == STEP_TICKS)
    ).fill_null(False)
    df = df.with_columns(same_next.alias("has_next"))
    has_next = df["has_next"].to_numpy()
    has_prev = np.r_[False, has_next[:-1]]
    x, y = df["X"].to_numpy().astype(np.float32), df["Y"].to_numpy().astype(np.float32)
    vel = np.zeros((df.height, 2), dtype=np.float32)
    vel[1:, 0] = np.where(has_prev[1:], x[1:] - x[:-1], 0.0)
    vel[1:, 1] = np.where(has_prev[1:], y[1:] - y[:-1], 0.0)
    nxt = np.where(has_next, np.arange(df.height) + 1, -1).astype(np.int32)
    node = grid.locate(x, y, df["Z"].to_numpy())
    side = np.array([SIDE_ID[s] for s in df["side"].to_list()], dtype=np.int8)

    key = side.astype(np.int64) * grid.n + node
    order = np.argsort(key, kind="stable").astype(np.int32)
    ptr = np.searchsorted(key[order], np.arange(2 * grid.n + 1)).astype(np.int64)
    return TrajectoryLibrary(
        node=node, t=df["t"].to_numpy(), since=df["since"].to_numpy().astype(np.float32),
        phase=df["phase"].to_numpy(), n_team=df["n_team"].to_numpy(), n_opp=df["n_opp"].to_numpy(),
        buy=df["buy"].to_numpy(), player=df["steamid"].to_numpy(), weapon=df["seen_weapon"].to_numpy(),
        side=side, nxt=nxt, vel=vel,
        starts=np.flatnonzero(~has_prev).astype(np.int32), bucket_ptr=ptr, bucket_idx=order,
    )  # fmt: skip


def _systematic(w: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    m = len(w)
    pos = (rng.random() + np.arange(m)) / m
    return np.minimum(np.searchsorted(np.cumsum(w), pos), m - 1)


def run_particles(
    ep: Episode,
    grid: NavGrid,
    lib: TrajectoryLibrary,
    unseen: np.ndarray | None,
    gamma: float,
    n_particles: int | None = None,
    seed: int = 0,
    kills: dict[int, list[tuple[int, np.ndarray]]] | None = None,
    weapon_mismatch: float = WEAPON_MISMATCH,
    weapon_mode: str = "all",
    fire_step: dict[int, np.ndarray] | None = None,
    fire_weight: dict[int, np.ndarray] | None = None,
    buy: np.ndarray | None = None,
):
    """Yield (step, beliefs (E, N)) like filters.run_filter. `kills` maps a step to
    (enemy, likelihood over nodes) pairs from the kill feed; `weapon_mismatch` weights recorded
    states whose last-seen weapon class differs from the enemy's. `buy` (S, 3) is the enemy team's
    buy at each step as P(eco, force, full), for the soft buy preference (None: no preference).
    `fire_step` maps a step to per-node move multipliers near burning fires: a particle whose
    recorded path steps into fire waits at the edge (it keeps its place in the recording) with
    probability 1 - m(next) / m(current). `fire_weight` maps a step to per-node occupancy factors
    near fires that multiply the particle weights like evidence, so particles standing in fire are
    resampled away."""
    rng = np.random.default_rng(seed)
    fire_rng = np.random.default_rng(seed + 1)  # separate stream: runs without fires stay identical
    side = SIDE_ID[ep.enemy]
    n_e, m = len(ep.enemy_ids), n_particles or N_PARTICLES  # read at call time so overrides apply
    idx = np.stack([lib.sample_starts(rng, side, m) for _ in range(n_e)])
    w = np.full((n_e, m), 1.0 / m)
    last_seen_step = np.full(n_e, -10)
    last_seen_t = np.full(n_e, np.inf)
    last_seen_xy = np.full((n_e, 2), np.nan)
    last_seen_weapon = np.full(n_e, -1)
    spread = sparse.diags(1.0 / np.maximum(np.diff(grid.adj_indptr), 1)) @ grid.adjacency

    def situation(e: int, s: int, since: float) -> Situation:
        return Situation(
            t=float(ep.t_rel[s]), phase=int(ep.phase[s]), since=since,
            n_team=int(ep.enemy_alive[s].sum()), n_opp=int(ep.obs_alive[s].sum()),
            buy=None if buy is None else buy[s], player=int(ep.enemy_ids[e]), weapon=int(last_seen_weapon[e]),
        )  # fmt: skip

    def rematch(e: int, which: np.ndarray, s: int) -> None:
        sit = situation(e, s, float(ep.t_rel[s] - last_seen_t[e]))
        for node in np.unique(lib.node[idx[e, which]]):
            sel = which[lib.node[idx[e, which]] == node]
            idx[e, sel] = lib.sample(grid, rng, side, int(node), sit, len(sel), lib.vel[idx[e, sel[0]]], weapon_mismatch, weapon_mode)

    for s in range(ep.n_steps):
        if s > 0:
            nxt = lib.nxt[idx]
            ended = nxt < 0
            if fire_step is not None and s in fire_step:
                mult = fire_step[s]
                go = mult[lib.node[np.maximum(nxt, 0)]] / mult[lib.node[idx]]
                nxt = np.where(~ended & (fire_rng.random(idx.shape) >= go), idx, nxt)  # wait outside the fire
            idx = np.where(ended, idx, nxt)
            for e in np.flatnonzero(ended.any(axis=1)):
                rematch(e, np.flatnonzero(ended[e]), s)
        if unseen is not None:
            w = w * unseen[s][lib.node[idx]] ** gamma
            tot = w.sum(axis=1, keepdims=True)
            w = np.where(tot > 1e-300, w / np.maximum(tot, 1e-300), 1.0 / m)
        if fire_weight is not None and s in fire_weight:
            w = w * fire_weight[s][lib.node[idx]]
            tot = w.sum(axis=1, keepdims=True)
            w = np.where(tot > 1e-300, w / np.maximum(tot, 1e-300), 1.0 / m)
        for e, like in (kills or {}).get(s, []):
            post = w[e] * like[lib.node[idx[e]]]
            w[e] = post / post.sum() if post.sum() > 1e-300 else 1.0 / m
        for e in np.flatnonzero(ep.enemy_seen_node[s] >= 0):
            xy = ep.enemy_seen_xyz[s, e, :2]
            heading = (xy - last_seen_xy[e]) / (s - last_seen_step[e]) if s - last_seen_step[e] <= 2 else None
            last_seen_weapon[e] = ep.enemy_seen_weapon[s, e]
            sit = situation(e, s, 0.0)
            idx[e] = lib.sample(grid, rng, side, int(ep.enemy_seen_node[s, e]), sit, m, heading, weapon_mismatch, weapon_mode)
            w[e] = 1.0 / m
            last_seen_step[e], last_seen_t[e], last_seen_xy[e] = s, ep.t_rel[s], xy
        ess = 1.0 / (w**2).sum(axis=1)
        for e in np.flatnonzero(ess < m / 3):
            pick = _systematic(w[e], rng)
            idx[e] = idx[e, pick]
            dup = np.flatnonzero(np.r_[False, pick[1:] == pick[:-1]])
            if len(dup):
                rematch(e, dup, s)  # give copies their own futures
            w[e] = 1.0 / m

        b = np.zeros((n_e, grid.n))
        for e in range(n_e):
            b[e] = np.bincount(lib.node[idx[e]], weights=w[e], minlength=grid.n)
        # Kernel smoothing over two rings of neighbours: particles are a sparse sample.
        one = np.asarray(b @ spread)
        yield s, 0.4 * b + 0.4 * one + 0.2 * np.asarray(one @ spread)
