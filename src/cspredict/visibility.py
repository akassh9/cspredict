"""Spotting model: how likely an enemy standing in a node is to appear on the friendly radar.

For one observer o and an enemy standing in node c:

    logit P(o spots c) = a[sight-line class, distance bin] + b[view offset bin] + c[vertical offset bin]
    P(o spots c)      *= per-node-pair correction learned from the demos (observed / expected)

* The sight-line class comes from two ray casts between node eye points through the occupancy
  raster: "walk" treats every pixel nobody ever stood on as a wall (precise but misses open
  areas), "carve" also opens pixels that real spotting rays passed through (catches ~95% of
  real spottings but over-opens). The model learns how far to trust each.
* View offsets are the horizontal and vertical angles between where the observer looks and
  the target. Spotting falls off steeply with both.
* The pair correction is an empirical-Bayes observed/expected ratio, so node pairs never seen
  together keep the geometric estimate.
* Active smokes block sight lines; blinded observers see nothing.

An enemy who is alive but not on the radar is weighted by prod_o (1 - P(o spots c)): the
negative-information update of the belief filters.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
from loguru import logger
from scipy.optimize import minimize

from cspredict.config import EYE_HEIGHT, FINE_CELL, TICKRATE
from cspredict.dataset import DemoRef
from cspredict.grid import FLOOR_GAP, NavGrid

RAY_STEP = 8.0  # spacing of ray-cast samples
MAX_RANGE = 4500.0  # longer sight lines are treated as blocked
CHEST_HEIGHT = 48.0  # spotting rays aim at the target's chest
SMOKE_RADIUS = 130.0  # a smoke blocks rays passing closer than this (2D)
P_MAX = 0.9  # never treat a cell as certainly seen (top predictions run ~20% hot)
PAIR_PRIOR = 3.0  # pseudo-counts shrinking the per-pair correction towards 1
FACTOR_RANGE = (0.02, 50.0)

DIST_EDGES = np.array([250, 500, 750, 1000, 1500, 2000, 3000], dtype=np.float32)
ANGLE_BIN = 5.0
N_OFFSET_BINS = 36  # 0..180 degrees
N_VOFFSET_BINS = 10  # 0..45+ degrees
N_CLASSES = 4  # 2 * walk + carve


@dataclass
class Transparency:
    """Height band through which real spotting rays crossed each FINE_CELL pixel."""

    zmin: np.ndarray  # (fine_ny, fine_nx), inf where no spotting ray ever crossed
    zmax: np.ndarray  # (fine_ny, fine_nx), -inf where no spotting ray ever crossed


@dataclass
class SpotModel:
    walk: np.ndarray  # (N, N) bool ray cast through walked pixels only
    carve: np.ndarray  # (N, N) bool ray cast that also passes carved pixels
    a: np.ndarray  # (N_CLASSES, n_dist_bins) logit terms
    b: np.ndarray  # (N_OFFSET_BINS,)
    c: np.ndarray  # (N_VOFFSET_BINS,)
    factor: np.ndarray  # (N, N) float16 observed/expected correction per node pair

    def save(self, path: Path) -> None:
        np.savez_compressed(path, **self.__dict__)

    @classmethod
    def load(cls, path: Path) -> SpotModel:
        with np.load(path) as z:
            return cls(**{k: z[k] for k in cls.__dataclass_fields__})

    def logit(self, cls_: np.ndarray, dist: np.ndarray, offset: np.ndarray, voffset: np.ndarray) -> np.ndarray:
        return (
            self.a[cls_, np.searchsorted(DIST_EDGES, dist)]
            + self.b[np.minimum(offset // ANGLE_BIN, N_OFFSET_BINS - 1).astype(np.int64)]
            + self.c[np.minimum(voffset // ANGLE_BIN, N_VOFFSET_BINS - 1).astype(np.int64)]
        )

    def spot_prob(
        self,
        grid: NavGrid,
        obs_xyz: np.ndarray,
        obs_yaw: np.ndarray,
        obs_pitch: np.ndarray,
        smokes_xy: np.ndarray | None = None,
    ) -> np.ndarray:
        """(k, N) probability that each observer would spot an enemy standing in each node."""
        obs_xyz = np.atleast_2d(obs_xyz).astype(np.float64)
        obs_nodes = grid.locate(obs_xyz[:, 0], obs_xyz[:, 1], obs_xyz[:, 2])
        tgt = grid.node_xy.astype(np.float64)
        tgt_z = grid.node_z.astype(np.float64) + CHEST_HEIGHT
        out = np.empty((len(obs_xyz), grid.n), dtype=np.float32)
        for o in range(len(obs_xyz)):
            geo = view_geometry(
                obs_xyz[o, 0], obs_xyz[o, 1], obs_xyz[o, 2], obs_yaw[o], obs_pitch[o], tgt[:, 0], tgt[:, 1], tgt_z
            )
            on = obs_nodes[o]
            cls_ = 2 * self.walk[on].astype(np.int64) + self.carve[on].astype(np.int64)
            p = _sigmoid(self.logit(cls_, *geo)) * self.factor[on].astype(np.float32)
            if smokes_xy is not None:
                for s in np.atleast_2d(smokes_xy):
                    p[_segment_point_dist(obs_xyz[o, :2], tgt, s) < SMOKE_RADIUS] = 0.0
            out[o] = np.minimum(p, P_MAX)
        return out

    def shooter_likelihood(self, grid: NavGrid, victim_xyz: np.ndarray, obstructed: bool) -> np.ndarray:
        """(N,) relative likelihood of the killer standing in each node, given where the victim
        died: the killer needed a sight line to the victim, unless the kill feed shows a
        wallbang or a kill through smoke, in which case only range is used."""
        vx, vy, vz = (float(v) for v in victim_xyz)
        v_node = grid.locate(np.array([vx]), np.array([vy]), np.array([vz]))[0]
        dist = np.hypot(grid.node_xy[:, 0] - vx, grid.node_xy[:, 1] - vy)
        if obstructed:
            return np.where(dist <= 2000.0, 1.0, 0.05)
        cls_ = 2 * self.walk[v_node].astype(np.int64) + self.carve[v_node].astype(np.int64)
        p = _sigmoid(self.a[cls_, np.searchsorted(DIST_EDGES, dist)] + self.b[0] + self.c[0])
        p = p * self.factor[v_node].astype(np.float64)
        return 0.05 + 0.95 * p / max(p.max(), 1e-9)

    def unseen_likelihood(self, grid: NavGrid, obs_xyz, obs_yaw, obs_pitch, smokes_xy=None) -> np.ndarray:
        """(N,) likelihood of "not on the radar" for an enemy standing in each node."""
        if len(obs_xyz) == 0:
            return np.ones(grid.n)
        p = self.spot_prob(grid, obs_xyz, obs_yaw, obs_pitch, smokes_xy).astype(np.float64)
        return np.prod(1.0 - p, axis=0)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def view_geometry(ox, oy, oz, yaw, pitch, tx, ty, tz_chest):
    """Distance, horizontal and vertical view offsets (degrees) from an observer to targets.
    Source-engine pitch is positive when looking down."""
    dx, dy = tx - ox, ty - oy
    dist = np.hypot(dx, dy)
    offset = np.abs((np.degrees(np.arctan2(dy, dx)) - yaw + 180.0) % 360.0 - 180.0)
    elev = np.degrees(np.arctan2(tz_chest - (oz + EYE_HEIGHT), np.maximum(dist, 1.0)))
    voffset = np.abs(elev + pitch)
    return dist, offset, voffset


def _segment_point_dist(a: np.ndarray, b: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Distance from point s to each segment a[i] -> b[i] (2D); a may be a single point."""
    ab = b - a
    t = np.clip(((s - a) * ab).sum(axis=1) / np.maximum((ab * ab).sum(axis=1), 1e-6), 0.0, 1.0)
    closest = a + t[:, None] * ab
    return np.hypot(closest[:, 0] - s[0], closest[:, 1] - s[1])


# ---------------------------------------------------------------------- ray casting
def _ray_samples(grid: NavGrid, p: np.ndarray, q: np.ndarray):
    """Sample points every RAY_STEP along p[i] -> q[i]; returns pixel coords, heights, masks."""
    d = q - p
    length = np.hypot(d[:, 0], d[:, 1])
    n = np.maximum(np.ceil(length / RAY_STEP).astype(np.int64), 1)
    t = np.arange(1, max(int(n.max()), 2))[None, :] / n[:, None]
    along = t * length[:, None]
    valid = (t < 1.0) & (along > 20.0) & (length[:, None] - along > 20.0)  # skip the endpoints' own pixels
    xs = p[:, 0, None] + t * d[:, 0, None]
    ys = p[:, 1, None] + t * d[:, 1, None]
    zs = p[:, 2, None] + t * d[:, 2, None]
    fx = np.floor((xs - grid.x0) / FINE_CELL).astype(np.int64)
    fy = np.floor((ys - grid.y0) / FINE_CELL).astype(np.int64)
    inside = (fx >= 0) & (fx < grid.fine_nx) & (fy >= 0) & (fy < grid.fine_ny)
    return np.clip(fx, 0, grid.fine_nx - 1), np.clip(fy, 0, grid.fine_ny - 1), zs, valid, inside


def ray_clear(grid: NavGrid, p: np.ndarray, q: np.ndarray, transp: Transparency | None = None) -> np.ndarray:
    """True where the segment p[i] -> q[i] (3D) never crosses an unwalked pixel, rising
    terrain or the slab of an upper floor, unless (with transp) a real spotting ray crossed
    that pixel at a similar height."""
    fx, fy, zs, valid, inside = _ray_samples(grid, p, q)
    walked = inside & (grid.fine_count[fy, fx] > 0)
    zlo, zhi = grid.fine_zlo[fy, fx], grid.fine_zhi[fy, fx]
    with np.errstate(invalid="ignore"):
        terrain = zlo > zs - 8.0
        slab = (zhi - zlo > FLOOR_GAP) & (zs > zhi - 40.0) & (zs < zhi + 4.0)
    solid = ~walked | terrain | slab
    if transp is not None:
        # Seen through at this height; allow some headroom above the highest observed ray.
        solid &= ~((transp.zmin[fy, fx] <= zs + 16.0) & (zs <= transp.zmax[fy, fx] + 96.0))
    return ~(valid & solid).any(axis=1)


def raycast_matrix(grid: NavGrid, transp: Transparency | None = None) -> np.ndarray:
    """(N, N) symmetric ray-cast visibility between node eye points."""
    eye = grid.node_xyz.astype(np.float64) + np.array([0.0, 0.0, EYE_HEIGHT])
    out = np.zeros((grid.n, grid.n), dtype=bool)
    for i in range(grid.n - 1):
        j = np.arange(i + 1, grid.n)
        j = j[np.hypot(*(eye[j, :2] - eye[i, :2]).T) <= MAX_RANGE]
        if len(j):
            clear = ray_clear(grid, np.broadcast_to(eye[i], (len(j), 3)), eye[j], transp)
            out[i, j[clear]] = True
    out |= out.T
    np.fill_diagonal(out, True)
    return out


def carve_transparency(grid: NavGrid, refs: list[DemoRef]) -> Transparency:
    """Every real spotting proves its sight line was open: record the heights at which
    observer-eye -> target-chest rays crossed each pixel."""
    zmin = np.full(grid.fine_ny * grid.fine_nx, np.inf, dtype=np.float32)
    zmax = np.full(grid.fine_ny * grid.fine_nx, -np.inf, dtype=np.float32)
    for ref in refs:
        hits = observer_pairs(grid, ref).filter(pl.col("hit"))
        p = np.column_stack([hits["ox"], hits["oy"], hits["oz"] + EYE_HEIGHT]).astype(np.float64)
        q = np.column_stack([hits["tx"], hits["ty"], hits["tz"] + CHEST_HEIGHT]).astype(np.float64)
        for s in range(0, len(p), 20_000):
            fx, fy, zs, valid, inside = _ray_samples(grid, p[s : s + 20_000], q[s : s + 20_000])
            keep = valid & inside
            pix = (fy * grid.fine_nx + fx)[keep]
            np.minimum.at(zmin, pix, zs[keep].astype(np.float32))
            np.maximum.at(zmax, pix, zs[keep].astype(np.float32))
    shape = (grid.fine_ny, grid.fine_nx)
    return Transparency(zmin.reshape(shape), zmax.reshape(shape))


# ---------------------------------------------------------------------- learning from demos
def observer_pairs(grid: NavGrid, ref: DemoRef) -> pl.DataFrame:
    """Every (observer, enemy) pair per sampled tick: positions, view angles, whether the enemy's
    spotted-by list contains the observer, and whether a smoke or a flash hid the enemy."""
    ticks = ref.table("ticks").filter(pl.col("is_alive"))
    ticks = ticks.with_columns(
        pl.Series("node", grid.locate(ticks["X"].to_numpy(), ticks["Y"].to_numpy(), ticks["Z"].to_numpy()))
    )
    obs = ticks.select(
        "round_num", "tick", pl.col("steamid").alias("oid"), pl.col("side").alias("oside"),
        pl.col("X").alias("ox"), pl.col("Y").alias("oy"), pl.col("Z").alias("oz"),
        pl.col("yaw").alias("oyaw"), pl.col("pitch").alias("opitch"), pl.col("node").alias("on"),
    )  # fmt: skip
    tgt = ticks.select(
        "tick", pl.col("side").alias("tside"), pl.col("X").alias("tx"), pl.col("Y").alias("ty"),
        pl.col("Z").alias("tz"), pl.col("node").alias("tn"), "spotted_by",
    )  # fmt: skip
    pairs = (
        obs.join(tgt, on="tick")
        .filter(pl.col("oside") != pl.col("tside"))
        .with_columns(pl.col("spotted_by").list.contains(pl.col("oid")).alias("hit"))
        .drop("spotted_by", "tside")
    )
    ox, oy, tx, ty = (pairs[c].to_numpy().astype(np.float64) for c in ("ox", "oy", "tx", "ty"))
    tick = pairs["tick"].to_numpy()

    hidden = np.zeros(pairs.height, dtype=bool)
    for s in ref.table("smokes").iter_rows(named=True):
        idx = np.flatnonzero((tick >= s["start_tick"]) & (tick < s["end_tick"]))
        if len(idx):
            dist = _segment_point_dist(
                np.column_stack([ox[idx], oy[idx]]), np.column_stack([tx[idx], ty[idx]]), np.array([s["X"], s["Y"]])
            )
            hidden[idx[dist < SMOKE_RADIUS]] = True
    oid = pairs["oid"].to_numpy()
    for b in ref.table("blinds").iter_rows(named=True):
        if (b.get("blind_duration") or 0) > 0.3 and b.get("user_steamid"):
            until = b["tick"] + b["blind_duration"] * TICKRATE
            hidden |= (oid == np.uint64(int(b["user_steamid"]))) & (tick >= b["tick"]) & (tick <= until)
    return pairs.with_columns(pl.Series("hidden", hidden))


def _pair_features(grid: NavGrid, walk: np.ndarray, carve: np.ndarray, pairs: pl.DataFrame):
    """Features of observer -> target-node pairs, measured to the target node's representative
    point exactly as at inference time."""
    on, tn = pairs["on"].to_numpy(), pairs["tn"].to_numpy()
    tgt = grid.node_xy[tn].astype(np.float64)
    dist, offset, voffset = view_geometry(
        pairs["ox"].to_numpy().astype(np.float64), pairs["oy"].to_numpy().astype(np.float64),
        pairs["oz"].to_numpy().astype(np.float64), pairs["oyaw"].to_numpy().astype(np.float64),
        pairs["opitch"].to_numpy().astype(np.float64), tgt[:, 0], tgt[:, 1], grid.node_z[tn] + CHEST_HEIGHT,
    )  # fmt: skip
    cls_ = 2 * walk[on, tn].astype(np.int64) + carve[on, tn].astype(np.int64)
    return on, tn, cls_, dist, offset, voffset


def _fit_logit_tables(cls_, dbin, obin, vbin, n, k) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Additive logistic model on binned features, fitted to aggregated binomial counts."""
    na, nb, nc = N_CLASSES * (len(DIST_EDGES) + 1), N_OFFSET_BINS, N_VOFFSET_BINS
    ia = cls_ * (len(DIST_EDGES) + 1) + dbin

    def loss(w):
        z = w[:na][ia] + w[na : na + nb][obin] + w[na + nb :][vbin]
        p = _sigmoid(z)
        nll = -(k * np.log(p + 1e-12) + (n - k) * np.log(1 - p + 1e-12)).sum()
        g = n * p - k
        grad = np.concatenate(
            [np.bincount(ia, g, na), np.bincount(obin, g, nb), np.bincount(vbin, g, nc)]
        ) + 1e-2 * w
        return nll + 0.5e-2 * (w @ w), grad

    w0 = np.full(na + nb + nc, -2.0 / 3.0)
    res = minimize(loss, w0, jac=True, method="L-BFGS-B", options={"maxiter": 2000})
    w = res.x
    return w[:na].reshape(N_CLASSES, -1), w[na : na + nb], w[na + nb :]


def fit_spot_model(grid: NavGrid, refs: list[DemoRef]) -> SpotModel:
    logger.info("Carving see-through space from real spottings")
    transp = carve_transparency(grid, refs)
    logger.info("Ray casting node-to-node visibility")
    walk, carve = raycast_matrix(grid), raycast_matrix(grid, transp)

    feats = []
    for ref in refs:
        pairs = observer_pairs(grid, ref).filter(~pl.col("hidden"))
        on, tn, cls_, dist, offset, voffset = _pair_features(grid, walk, carve, pairs)
        feats.append(
            pl.DataFrame({
                "on": on, "tn": tn, "cls": cls_,
                "dbin": np.searchsorted(DIST_EDGES, dist),
                "obin": np.minimum(offset // ANGLE_BIN, N_OFFSET_BINS - 1).astype(np.int64),
                "vbin": np.minimum(voffset // ANGLE_BIN, N_VOFFSET_BINS - 1).astype(np.int64),
                "hit": pairs["hit"].to_numpy(),
            })  # fmt: skip
        )
    feats = pl.concat(feats)
    agg = (
        feats.group_by("cls", "dbin", "obin", "vbin").agg(pl.len().alias("n"), pl.col("hit").sum().alias("k"))
        .sort("cls", "dbin", "obin", "vbin")  # group order is random; a fixed order makes the fit reproducible
    )  # fmt: skip
    a, b, c = _fit_logit_tables(*(agg[col].to_numpy() for col in ("cls", "dbin", "obin", "vbin", "n", "k")))
    model = SpotModel(walk=walk, carve=carve, a=a, b=b, c=c, factor=np.ones((grid.n, grid.n), np.float16))

    # Per node pair: observed hits vs the hits the geometric model expects, shrunk towards 1.
    expected = _sigmoid(
        a[feats["cls"].to_numpy(), feats["dbin"].to_numpy()] + b[feats["obin"].to_numpy()] + c[feats["vbin"].to_numpy()]
    )
    per_pair = (
        feats.select("on", "tn", "hit")
        .with_columns(pl.Series("e", expected))
        .group_by("on", "tn")
        .agg(pl.col("hit").sum().alias("k"), pl.col("e").sum())
    )
    kk = np.zeros((grid.n, grid.n), np.float32)
    ee = np.zeros((grid.n, grid.n), np.float32)
    kk[per_pair["on"].to_numpy(), per_pair["tn"].to_numpy()] = per_pair["k"].to_numpy()
    ee[per_pair["on"].to_numpy(), per_pair["tn"].to_numpy()] = per_pair["e"].to_numpy()
    kk, ee = kk + kk.T, ee + ee.T  # sight lines are symmetric
    model.factor = np.clip((kk + PAIR_PRIOR) / (ee + PAIR_PRIOR), *FACTOR_RANGE).astype(np.float16)
    logger.info(f"Spot model fitted on {feats.height:,} observer-target pairs ({int(feats['hit'].sum()):,} spotted)")
    return model
