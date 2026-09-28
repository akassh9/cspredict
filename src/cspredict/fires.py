"""Molotovs and incendiaries: places an enemy will not stand in or walk through while they burn.

Assumption: every active fire is known to both teams. The thrower's team saw where it landed, and
the other team sees the flames and hears them. A fire is a disc around the point where the grenade
burst, on that floor only; the real flames spread along the ground, so the edge is soft.

Two profiles over the distance r from a burning fire's centre are learned from the training demos:

    occupancy(r)  how densely players stand at distance r while the fire burns, relative to the
                  same spots in the seconds after it has burnt out (1 = unaffected). This is the
                  likelihood of "standing here" given the fire, and gives the fire radius: players
                  stand inside it at a fraction of the usual density.
    step(r)       per filter step, a multiplier on moving into (or staying in) a cell at distance r,
                  fitted by maximum likelihood on the moves players really made next to fires, given
                  the grid motion model. The blocked probability goes to the other moves from the
                  same cell, so players wait at the edge or go around.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
from loguru import logger
from scipy.optimize import minimize

from cspredict.config import STEP_TICKS, TICKRATE
from cspredict.dataset import DemoRef
from cspredict.grid import NavGrid

EDGES = np.arange(0.0, 300.0, 20.0)  # distance bins from a fire's centre; unaffected beyond the last edge
FLOOR_Z = 100.0  # a fire only affects nodes within this height of it
SPREAD_S = 0.5  # the flames need about half a second to spread
CONTROL_GAP_S = 1.0  # control window for the occupancy profile starts this long after the fire ends
MIN_FACTOR = 0.01


@dataclass
class FireModel:
    edges: np.ndarray  # (B + 1,) distance bin edges
    occupancy: np.ndarray  # (B,) relative density of players at that distance while a fire burns
    step: np.ndarray  # (B,) per-step multiplier on moving into / staying at that distance
    radius: float  # distance where the occupancy first rises to half the usual density

    def save(self, path: Path) -> None:
        np.savez_compressed(path, edges=self.edges, occupancy=self.occupancy, step=self.step, radius=self.radius)

    @classmethod
    def load(cls, path: Path) -> FireModel:
        with np.load(path) as z:
            return cls(edges=z["edges"], occupancy=z["occupancy"], step=z["step"], radius=float(z["radius"]))

    def node_factors(self, grid: NavGrid, fires_xyz: np.ndarray, profile: str) -> np.ndarray:
        """(N,) the profile ("occupancy" or "step") at every node for these active fires; the
        smallest factor wins where fires overlap, 1 away from fires."""
        values = getattr(self, profile)
        out = np.ones(grid.n)
        for x, y, z in np.atleast_2d(fires_xyz):
            r = np.hypot(grid.node_xy[:, 0] - x, grid.node_xy[:, 1] - y)
            near = (r < self.edges[-1]) & (np.abs(grid.node_z - z) < FLOOR_Z)
            b = np.searchsorted(self.edges, r[near], side="right") - 1
            out[near] = np.minimum(out[near], values[b])
        return out


def active_fires(infernos: pl.DataFrame, ticks: np.ndarray) -> list[np.ndarray]:
    """Per tick, the (k, 3) centres of fires burning then (after they have spread)."""
    x, y, z = (infernos[c].to_numpy() for c in ("X", "Y", "Z"))
    start = infernos["start_tick"].to_numpy() + int(SPREAD_S * TICKRATE)
    end = infernos["end_tick"].to_numpy()
    xyz = np.column_stack([x, y, z]) if len(x) else np.zeros((0, 3))
    return [xyz[(start <= t) & (t < end)] for t in ticks]


def _occupancy_profile(refs: list[DemoRef]) -> np.ndarray:
    """Density of alive players by distance from burning fires, relative to the same distances in
    an equally long window starting CONTROL_GAP_S after each fire ends."""
    during = np.zeros(len(EDGES) - 1)
    after = np.zeros(len(EDGES) - 1)
    for ref in refs:
        ticks = ref.table("ticks").filter(pl.col("is_alive")).select("round_num", "tick", "X", "Y", "Z")
        for f in ref.table("infernos").iter_rows(named=True):
            rt = ticks.filter(pl.col("round_num") == f["round_num"])
            t = rt["tick"].to_numpy()
            r = np.hypot(rt["X"].to_numpy() - f["X"], rt["Y"].to_numpy() - f["Y"])
            same_floor = np.abs(rt["Z"].to_numpy() - f["Z"]) < FLOOR_Z
            start, end = f["start_tick"] + int(SPREAD_S * TICKRATE), f["end_tick"]
            gap = int(CONTROL_GAP_S * TICKRATE)
            burning = same_floor & (t >= start) & (t < end)
            control = same_floor & (t >= end + gap) & (t < end + gap + (end - start))
            during += np.histogram(r[burning], EDGES)[0]
            after += np.histogram(r[control], EDGES)[0]
    return during / np.maximum(after, 1.0)


def _moves_near_fires(grid: NavGrid, refs: list[DemoRef]) -> list[tuple[str, int, int, int, int, np.ndarray]]:
    """One-step moves (side, phase, since bin, node, next node, active fire centres) of alive
    players that started within reach of a burning fire."""
    from cspredict.motion import phase_expr, since_bin, with_since_spotted

    reach = EDGES[-1] + 2 * 64.0  # a move ends at most about two cells from where it starts
    out = []
    for ref in refs:
        inf = ref.table("infernos")
        if inf.height == 0:
            continue
        rounds = ref.table("rounds").select("round_num", "bomb_plant", "bomb_site")
        t = with_since_spotted(ref.table("ticks").filter(pl.col("is_alive")))
        t = t.filter(pl.col("tick") % STEP_TICKS == 0).join(rounds, on="round_num").with_columns(phase_expr())
        t = t.sort("round_num", "steamid", "tick").with_columns(
            pl.Series("node", grid.locate(t["X"].to_numpy(), t["Y"].to_numpy(), t["Z"].to_numpy()))
        ).with_columns(
            pl.col("node").shift(-1).over("round_num", "steamid").alias("next"),
            (pl.col("tick").shift(-1).over("round_num", "steamid") - pl.col("tick")).alias("dt"),
        ).filter(pl.col("next").is_not_null() & (pl.col("dt") == STEP_TICKS))
        ticks = np.unique(t["tick"].to_numpy())
        burning = dict(zip(ticks.tolist(), active_fires(inf, ticks + STEP_TICKS)))  # fires where the move ends
        node, nxt = t["node"].to_numpy(), t["next"].to_numpy().astype(np.int64)
        for i, (tick, side, phase, since) in enumerate(zip(t["tick"].to_list(), t["side"].to_list(), t["phase"].to_list(), t["since"].to_list())):
            fires = burning[tick]
            if not len(fires):
                continue
            d = np.hypot(fires[:, 0] - grid.node_xy[node[i], 0], fires[:, 1] - grid.node_xy[node[i], 1])
            near = (d < reach) & (np.abs(fires[:, 2] - grid.node_z[node[i]]) < FLOOR_Z)
            if near.any():
                out.append((side, int(phase), since_bin(np.inf if since is None else since), int(node[i]), int(nxt[i]), fires[near]))
    return out


def _fit_step_profile(grid: NavGrid, motion, moves: list, edges: np.ndarray) -> np.ndarray:
    """Maximum-likelihood per-bin multipliers m on the grid-motion transitions near fires:
    P(i -> j) = T_ij m(r_j) / sum_k T_ik m(r_k), with m = 1 beyond the last edge."""
    n_bins = len(edges) - 1
    cand_rows, cand_bins, cand_p, chosen_bins = [], [], [], []
    row = 0
    for side, phase, sbin, node, nxt, fires in moves:
        t = motion.trans[motion.key(side, phase, sbin)]
        lo, hi = t.indptr[node], t.indptr[node + 1]
        cols, p = t.indices[lo:hi], t.data[lo:hi]
        where = np.flatnonzero(cols == nxt)
        if len(where) == 0:
            continue
        r = np.hypot(grid.node_xy[cols, 0, None] - fires[:, 0], grid.node_xy[cols, 1, None] - fires[:, 1])
        r = np.where(np.abs(grid.node_z[cols, None] - fires[:, 2]) < FLOOR_Z, r, np.inf).min(axis=1)
        b = np.where(r < edges[-1], np.searchsorted(edges, r, side="right") - 1, n_bins)  # n_bins = unaffected
        if (b == n_bins).all():
            continue
        cand_rows.append(np.full(len(cols), row))
        cand_bins.append(b)
        cand_p.append(p)
        chosen_bins.append(int(b[where[0]]))
        row += 1
    rows_, bins_, p_ = np.concatenate(cand_rows), np.concatenate(cand_bins), np.concatenate(cand_p)
    chosen = np.bincount(np.array(chosen_bins), minlength=n_bins + 1)[:n_bins]

    def loss(theta: np.ndarray) -> tuple[float, np.ndarray]:
        logm = np.append(theta, 0.0)
        wp = p_ * np.exp(logm[bins_])
        z = np.bincount(rows_, wp, row)
        share = wp / z[rows_]  # each candidate move's share of its row
        nll = -(chosen * theta).sum() + np.log(z).sum()
        grad = -chosen + np.bincount(bins_, share, n_bins + 1)[:n_bins]
        return nll + 0.5e-3 * (theta @ theta), grad + 1e-3 * theta

    bounds = [(np.log(MIN_FACTOR), 0.0)] * n_bins
    res = minimize(loss, np.zeros(n_bins), jac=True, method="L-BFGS-B", bounds=bounds)
    logger.info(f"Fire step profile fitted on {row:,} moves next to burning fires")
    return np.exp(res.x)


def fit_fire_model(grid: NavGrid, motion, refs: list[DemoRef]) -> FireModel:
    occupancy = np.clip(_occupancy_profile(refs), MIN_FACTOR, 1.0)
    # Beyond the first bin back at normal density the fire has no effect (the busier ring just
    # outside it is where players wait, which the step profile produces).
    normal = np.flatnonzero(occupancy >= 0.95)
    if len(normal):
        occupancy[normal[0]:] = 1.0
    half = np.flatnonzero(occupancy >= 0.5)
    radius = float(EDGES[half[0]]) if len(half) else float(EDGES[-1])
    step = _fit_step_profile(grid, motion, _moves_near_fires(grid, refs), EDGES)
    logger.info(f"Fire radius (half the usual density) {radius:.0f} units; occupancy {np.round(occupancy, 2)}; step {np.round(step, 2)}")
    return FireModel(edges=EDGES.astype(np.float32), occupancy=occupancy, step=step, radius=radius)
