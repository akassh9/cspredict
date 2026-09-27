"""Motion models: how an unseen enemy moves between grid nodes in one filter step (0.25 s).

Learned from demo trajectories as first-order Markov transition matrices. The context is
side x bomb phase (not planted / planted on A / on B) x time since the player was last
spotted: players who were just seen mostly hold or fall back, while players unseen for a
while rotate along routes. Each
since-spotted matrix is shrunk towards the pooled side-and-phase matrix, and that one towards
a lazy random walk, so rarely visited nodes still move sensibly. Also learns where each side
spawns and, for the population-prior baseline, where each side tends to be by round time.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
from scipy import sparse

from cspredict.config import STEP_TICKS, TICKRATE
from cspredict.dataset import DemoRef
from cspredict.grid import NavGrid

SIDES = ("t", "ct")
PHASES = ("pre", "a", "b")  # bomb not planted / planted on A / planted on B
SINCE_EDGES = (2.0, 5.0, 10.0, 20.0)  # seconds since last spotted; last bin = never spotted this round
N_SINCE_BINS = len(SINCE_EDGES) + 2
SMOOTH = 0.5  # pseudo-counts spread over a node and its neighbours (pooled matrices)
SHRINK = 4.0  # pseudo-counts pulling a since-spotted matrix towards its pooled matrix
TIME_BIN_S = 5.0
N_TIME_BINS = 36  # 0..180 s


def context_key(side: str, phase: int) -> str:
    return f"{side}_{PHASES[phase]}"


def phase_expr() -> pl.Expr:
    """0 before the plant, then 1 or 2 for a bomb planted on A or B (needs tick, bomb_plant, bomb_site)."""
    planted = pl.col("bomb_plant").is_not_null() & (pl.col("tick") >= pl.col("bomb_plant"))
    site = pl.when(pl.col("bomb_site") == "bombsite_b").then(2).otherwise(1)
    return pl.when(planted).then(site).otherwise(0).cast(pl.Int8).alias("phase")


def since_bin(since: float) -> int:
    """Bin of seconds since last spotted; NaN/inf means never spotted this round."""
    if not np.isfinite(since):
        return N_SINCE_BINS - 1
    return int(np.searchsorted(SINCE_EDGES, since, side="right"))


def trans_key(side: str, phase: int, sbin: int) -> str:
    return f"{context_key(side, phase)}_s{sbin}"


@dataclass
class MotionModel:
    trans: dict[str, sparse.csr_matrix]  # trans_key -> (N, N) row-stochastic
    random_walk: sparse.csr_matrix  # (N, N) lazy random walk, the untrained baseline
    spawn: dict[str, np.ndarray]  # side -> (N,) distribution when freeze time ends
    occupancy: dict[str, np.ndarray]  # context_key -> (N_TIME_BINS, N) where that side is by round time

    def save(self, path: Path) -> None:
        arrays = {}
        for key, m in {**self.trans, "random_walk": self.random_walk}.items():
            m = m.tocsr()
            arrays |= {f"{key}__data": m.data, f"{key}__indices": m.indices, f"{key}__indptr": m.indptr}
        arrays |= {f"spawn__{k}": v for k, v in self.spawn.items()}
        arrays |= {f"occ__{k}": v for k, v in self.occupancy.items()}
        arrays["trans_keys"] = np.array(sorted(self.trans))
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path: Path) -> MotionModel:
        with np.load(path) as z:
            n = len(z["random_walk__indptr"]) - 1

            def csr(key: str) -> sparse.csr_matrix:
                return sparse.csr_matrix((z[f"{key}__data"], z[f"{key}__indices"], z[f"{key}__indptr"]), shape=(n, n))

            contexts = [context_key(s, p) for s in SIDES for p in range(len(PHASES))]
            return cls(
                trans={str(k): csr(str(k)) for k in z["trans_keys"]},
                random_walk=csr("random_walk"),
                spawn={s: z[f"spawn__{s}"] for s in SIDES},
                occupancy={k: z[f"occ__{k}"] for k in contexts},
            )


def _normalize_rows(m: sparse.spmatrix) -> sparse.csr_matrix:
    m = sparse.csr_matrix(m, dtype=np.float64)
    rows = np.asarray(m.sum(axis=1)).ravel()
    return sparse.csr_matrix(sparse.diags(1.0 / np.maximum(rows, 1e-12)) @ m)


def with_since_spotted(ticks: pl.DataFrame) -> pl.DataFrame:
    """Add `since` = seconds since the player was last on the enemy radar this round (null if never)."""
    return (
        ticks.sort("round_num", "steamid", "tick")
        .with_columns(
            pl.when(pl.col("spotted")).then(pl.col("tick")).otherwise(None).forward_fill().over("round_num", "steamid")
            .alias("last_spot")
        )  # fmt: skip
        .with_columns(((pl.col("tick") - pl.col("last_spot")) / TICKRATE).cast(pl.Float32).alias("since"))
        .drop("last_spot")
    )


def _step_samples(grid: NavGrid, ref: DemoRef) -> pl.DataFrame:
    """Alive players at every filter step with node, bomb phase and time since last spotted."""
    ticks = with_since_spotted(ref.table("ticks").filter(pl.col("is_alive")))
    ticks = ticks.filter(pl.col("tick") % STEP_TICKS == 0)
    rounds = ref.table("rounds").select("round_num", "freeze_end", "bomb_plant", "bomb_site")
    ticks = ticks.join(rounds, on="round_num").with_columns(
        phase_expr(), ((pl.col("tick") - pl.col("freeze_end")) / TICKRATE).alias("t_rel")
    )
    node = grid.locate(ticks["X"].to_numpy(), ticks["Y"].to_numpy(), ticks["Z"].to_numpy())
    sbin = np.array([since_bin(v) for v in ticks["since"].fill_null(np.inf).to_numpy()], dtype=np.int8)
    return ticks.select("round_num", "tick", "steamid", "side", "phase", "t_rel").with_columns(
        pl.Series("node", node), pl.Series("sbin", sbin)
    )


def fit_motion_model(grid: NavGrid, refs: list[DemoRef]) -> MotionModel:
    samples = pl.concat([_step_samples(grid, ref).with_columns(pl.lit(ref.demo_id).alias("demo")) for ref in refs])
    samples = samples.sort("demo", "round_num", "steamid", "tick").with_columns(
        pl.col("node").shift(-1).over("demo", "round_num", "steamid").alias("next"),
        (pl.col("tick").shift(-1).over("demo", "round_num", "steamid") - pl.col("tick")).alias("dt"),
    )
    moves = samples.filter(pl.col("next").is_not_null() & (pl.col("dt") == STEP_TICKS))

    n = grid.n
    stay = float((moves["node"] == moves["next"]).mean())
    deg = np.asarray(grid.adjacency.sum(axis=1)).ravel()
    random_walk = sparse.diags(np.full(n, stay)) + sparse.diags((1 - stay) / np.maximum(deg, 1)) @ grid.adjacency
    smooth = _normalize_rows(sparse.identity(n, format="csr") + grid.adjacency) * SMOOTH

    def counts(m: pl.DataFrame) -> sparse.csr_matrix:
        return sparse.csr_matrix(
            (np.ones(m.height), (m["node"].to_numpy(), m["next"].to_numpy().astype(np.int64))), shape=(n, n)
        )

    trans, occupancy = {}, {}
    for side in SIDES:
        for phase in range(len(PHASES)):
            ctx = moves.filter((pl.col("side") == side) & (pl.col("phase") == phase))
            pooled = _normalize_rows(counts(ctx) + smooth)
            for sbin in range(N_SINCE_BINS):
                c = counts(ctx.filter(pl.col("sbin") == sbin))
                trans[trans_key(side, phase, sbin)] = _normalize_rows(c + SHRINK * pooled)

            s = samples.filter((pl.col("side") == side) & (pl.col("phase") == phase))
            tbin = np.minimum((s["t_rel"].to_numpy() // TIME_BIN_S).astype(np.int64), N_TIME_BINS - 1)
            occ = np.zeros((N_TIME_BINS, n))
            np.add.at(occ, (tbin, s["node"].to_numpy()), 1.0)
            occ += 0.01  # every node keeps a sliver of mass
            occupancy[context_key(side, phase)] = (occ / occ.sum(axis=1, keepdims=True)).astype(np.float32)

    spawn = {}
    first = samples.group_by("demo", "round_num", "steamid").agg(pl.all().sort_by("tick").first())
    for side in SIDES:
        nodes = first.filter(pl.col("side") == side)["node"].to_numpy()
        c = np.bincount(nodes, minlength=n).astype(np.float64) + 1e-3
        spawn[side] = (c / c.sum()).astype(np.float32)
    return MotionModel(trans=trans, random_walk=_normalize_rows(random_walk), spawn=spawn, occupancy=occupancy)
