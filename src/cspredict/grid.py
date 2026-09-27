"""Data-driven navigation grid for one map.

awpy's nav-mesh and map-geometry downloads are unavailable, so the walkable area is learned
from where players actually stand in the demos. This also keeps the grid in sync with
whatever map version the demos were recorded on.

* The map is cut into CELL x CELL columns. A column can hold several floors (e.g. Mirage's
  underpass beneath the catwalk), and each (column, floor) pair is a node. Nodes are the
  state space of the belief filters.
* Edges join nodes that players were seen moving between, plus flat 8-neighbours.
* A finer FINE_CELL occupancy raster with per-pixel floor heights is kept for line-of-sight
  ray casting (see visibility.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np
import polars as pl
from scipy import sparse
from scipy.sparse import csgraph
from scipy.spatial import cKDTree

from cspredict.config import CELL, FINE_CELL

MAX_FLOORS = 3  # floors kept per column
FLOOR_GAP = 110.0  # vertical gap separating two floors in one column (a jump reaches ~66)
MIN_NODE_SAMPLES = 8  # samples at 8 Hz = one second of presence in total
STEP_HEIGHT = 48.0  # floor difference still walkable between neighbouring columns
MARGIN = 256.0


@dataclass
class NavGrid:
    x0: float
    y0: float
    nx: int
    ny: int
    node_ix: np.ndarray  # (N,) column index along x
    node_iy: np.ndarray  # (N,) column index along y
    node_z: np.ndarray  # (N,) floor height
    node_count: np.ndarray  # (N,) training samples seen in the node
    node_place: np.ndarray  # (N,) most common callout name
    node_rep: np.ndarray  # (N, 2) typical standing spot: mean position, snapped onto walked pixels
    col_nodes: np.ndarray  # (nx*ny, MAX_FLOORS) node ids per column, -1 if none, sorted by height
    adj_indptr: np.ndarray  # undirected adjacency (CSR, no self loops)
    adj_indices: np.ndarray
    fine_nx: int
    fine_ny: int
    fine_count: np.ndarray  # (fine_ny, fine_nx) samples per FINE_CELL pixel
    fine_zlo: np.ndarray  # lowest floor per pixel (nan if never occupied)
    fine_zhi: np.ndarray  # highest floor per pixel (== zlo on single-floor pixels)

    # ------------------------------------------------------------------ geometry
    @property
    def n(self) -> int:
        return len(self.node_z)

    @cached_property
    def node_xy(self) -> np.ndarray:
        """(N, 2) representative position of each node."""
        return self.node_rep.astype(np.float32)

    @cached_property
    def node_xyz(self) -> np.ndarray:
        return np.column_stack([self.node_xy, self.node_z]).astype(np.float32)

    @cached_property
    def column_xy(self) -> np.ndarray:
        """(N, 2) centre of each node's column (for drawing)."""
        return np.stack(
            [self.x0 + (self.node_ix + 0.5) * CELL, self.y0 + (self.node_iy + 0.5) * CELL], axis=1
        ).astype(np.float32)

    @cached_property
    def adjacency(self) -> sparse.csr_matrix:
        data = np.ones(len(self.adj_indices), dtype=np.float32)
        return sparse.csr_matrix((data, self.adj_indices, self.adj_indptr), shape=(self.n, self.n))

    @cached_property
    def dist(self) -> np.ndarray:
        """(N, N) shortest walking distance between nodes, in game units."""
        a = self.adjacency.tocoo()
        w = np.linalg.norm(self.node_xyz[a.row] - self.node_xyz[a.col], axis=1)
        graph = sparse.csr_matrix((w, (a.row, a.col)), shape=a.shape)
        d = csgraph.shortest_path(graph, method="D", directed=False)
        return np.where(np.isfinite(d), d, 1e5).astype(np.float32)

    @cached_property
    def places(self) -> list[str]:
        return sorted(set(self.node_place.tolist()))

    @cached_property
    def node_place_idx(self) -> np.ndarray:
        lookup = {p: i for i, p in enumerate(self.places)}
        return np.array([lookup[p] for p in self.node_place], dtype=np.int32)

    @cached_property
    def _kdtree(self) -> cKDTree:
        # Height counts double so a nearby column on another floor ranks behind.
        return cKDTree(np.column_stack([self.column_xy, self.node_z * 2.0]))

    def locate(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Node id for each position: the highest floor at or below the feet in its column,
        falling back to the nearest node when the column was never walked on."""
        x, y, z = (np.asarray(v, dtype=np.float64) for v in (x, y, z))
        ix = np.floor((x - self.x0) / CELL).astype(np.int64)
        iy = np.floor((y - self.y0) / CELL).astype(np.int64)
        inside = (ix >= 0) & (ix < self.nx) & (iy >= 0) & (iy < self.ny)
        col = np.where(inside, iy * self.nx + ix, 0)
        cand = np.where(inside[:, None], self.col_nodes[col], -1)
        cz = np.where(cand >= 0, self.node_z[np.maximum(cand, 0)], np.nan)
        below = (cand >= 0) & (cz <= z[:, None] + 32.0)
        # Highest floor below the feet; otherwise the lowest floor of the column.
        score = np.where(below, cz, -np.inf)
        best = np.argmax(score, axis=1)
        out = cand[np.arange(len(x)), best]
        none_below = ~below.any(axis=1)
        out = np.where(none_below, cand[:, 0], out)
        missing = out < 0
        if missing.any():
            q = np.column_stack([x[missing], y[missing], z[missing] * 2.0])
            out[missing] = self._kdtree.query(q)[1]
        return out.astype(np.int32)

    # ------------------------------------------------------------------ io
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fields = {k: v for k, v in self.__dict__.items() if not k.startswith("_") and k in self.__dataclass_fields__}
        fields["node_place"] = np.array(self.node_place, dtype="U32")
        np.savez_compressed(path, **fields)

    @classmethod
    def load(cls, path: Path) -> NavGrid:
        with np.load(path) as z:
            kw = {k: z[k] for k in cls.__dataclass_fields__}
        for k in ("x0", "y0"):
            kw[k] = float(kw[k])
        for k in ("nx", "ny", "fine_nx", "fine_ny"):
            kw[k] = int(kw[k])
        return cls(**kw)


def _floors(zs: np.ndarray) -> list[tuple[float, int]]:
    """Split one column's height samples into floors: [(floor_z, n_samples), ...]."""
    zs = np.sort(zs)
    breaks = np.flatnonzero(np.diff(zs) > FLOOR_GAP) + 1
    floors = []
    for part in np.split(zs, breaks):
        if len(part) >= MIN_NODE_SAMPLES:
            floors.append((float(np.percentile(part, 10)), len(part)))
    return floors


def build_grid(ticks: pl.DataFrame) -> NavGrid:
    """Build the grid from tick rows (needs demo_id, round_num, tick, steamid, X, Y, Z,
    is_alive, place)."""
    pos = ticks.filter(pl.col("is_alive")).select("demo_id", "round_num", "tick", "steamid", "X", "Y", "Z", "place")
    x, y, z = (pos[c].to_numpy().astype(np.float64) for c in ("X", "Y", "Z"))
    x0 = float(np.floor((x.min() - MARGIN) / CELL) * CELL)
    y0 = float(np.floor((y.min() - MARGIN) / CELL) * CELL)
    nx = int(np.ceil((x.max() + MARGIN - x0) / CELL))
    ny = int(np.ceil((y.max() + MARGIN - y0) / CELL))
    ix = ((x - x0) // CELL).astype(np.int64)
    iy = ((y - y0) // CELL).astype(np.int64)
    col = iy * nx + ix

    # Floors per column.
    order = np.argsort(col, kind="stable")
    col_sorted, z_sorted = col[order], z[order]
    starts = np.flatnonzero(np.r_[True, np.diff(col_sorted) != 0])
    ends = np.r_[starts[1:], len(col_sorted)]
    node_ix, node_iy, node_z, node_count = [], [], [], []
    col_nodes = np.full((nx * ny, MAX_FLOORS), -1, dtype=np.int32)
    for s, e in zip(starts, ends):
        c = int(col_sorted[s])
        floors = _floors(z_sorted[s:e])[:MAX_FLOORS]
        for k, (fz, cnt) in enumerate(sorted(floors)):
            col_nodes[c, k] = len(node_z)
            node_ix.append(c % nx)
            node_iy.append(c // nx)
            node_z.append(fz)
            node_count.append(cnt)
    grid = NavGrid(
        x0=x0, y0=y0, nx=nx, ny=ny,
        node_ix=np.array(node_ix, dtype=np.int32), node_iy=np.array(node_iy, dtype=np.int32),
        node_z=np.array(node_z, dtype=np.float32), node_count=np.array(node_count, dtype=np.int32),
        node_place=np.array([""] * len(node_z), dtype="U32"), node_rep=np.zeros((len(node_z), 2), np.float32),
        col_nodes=col_nodes,
        adj_indptr=np.zeros(len(node_z) + 1, dtype=np.int32), adj_indices=np.zeros(0, dtype=np.int32),
        fine_nx=0, fine_ny=0, fine_count=np.zeros((0, 0), np.int32),
        fine_zlo=np.zeros((0, 0), np.float32), fine_zhi=np.zeros((0, 0), np.float32),
    )  # fmt: skip

    node = grid.locate(x, y, z)
    grid.node_place = _majority_place(node, pos["place"].to_numpy(), grid.n)
    rows, cols = _edges(grid, pos, node)
    keep = _largest_component(grid.n, rows, cols)
    grid = _subset(grid, keep, rows, cols)
    _fill_fine_raster(grid, x, y, z)
    grid.node_rep = _representative_points(grid, x, y, z)
    return grid


def _representative_points(grid: NavGrid, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Mean standing position per node, moved to the nearest walked pixel inside its column
    (the mean of an L-shaped area can land inside a wall)."""
    node = grid.locate(x, y, z)
    cnt = np.maximum(np.bincount(node, minlength=grid.n), 1)
    rep = np.column_stack([np.bincount(node, x, grid.n) / cnt, np.bincount(node, y, grid.n) / cnt])
    per_col = int(CELL // FINE_CELL)
    offs = (np.arange(per_col) + 0.5) * FINE_CELL
    for i in range(grid.n):
        px0, py0 = grid.node_ix[i] * per_col, grid.node_iy[i] * per_col
        block = grid.fine_count[py0 : py0 + per_col, px0 : px0 + per_col]
        yy, xx = np.nonzero(block)
        if len(xx) == 0:
            continue
        cx = grid.x0 + grid.node_ix[i] * CELL + offs[xx]
        cy = grid.y0 + grid.node_iy[i] * CELL + offs[yy]
        j = np.argmin((cx - rep[i, 0]) ** 2 + (cy - rep[i, 1]) ** 2)
        rep[i] = cx[j], cy[j]
    return rep.astype(np.float32)


def _majority_place(node: np.ndarray, place: np.ndarray, n: int) -> np.ndarray:
    top = (
        pl.DataFrame({"node": node, "place": place})
        .filter(pl.col("place").is_not_null() & (pl.col("place") != ""))
        .group_by("node", "place")
        .len()
        .sort("len", descending=True)
        .group_by("node")
        .first()
    )
    out = np.full(n, "Unknown", dtype="U32")
    out[top["node"].to_numpy()] = top["place"].to_numpy()
    return out


def _edges(grid: NavGrid, pos: pl.DataFrame, node: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Undirected edges: observed moves between nodes plus walkable 8-neighbours."""
    moves = (
        pos.select("demo_id", "round_num", "steamid", "tick")
        .with_columns(pl.Series("node", node))
        .sort("demo_id", "round_num", "steamid", "tick")
        .with_columns(
            pl.col("node").shift(-1).over("demo_id", "round_num", "steamid").alias("next"),
            (pl.col("tick").shift(-1).over("demo_id", "round_num", "steamid") - pl.col("tick")).alias("dt"),
        )
        .filter(pl.col("next").is_not_null() & (pl.col("next") != pl.col("node")) & (pl.col("dt") <= 16))
        .group_by("node", "next")
        .len()
        .filter(pl.col("len") >= 2)
    )
    a, b = moves["node"].to_numpy(), moves["next"].to_numpy().astype(np.int32)
    gap = np.abs(grid.node_ix[a] - grid.node_ix[b]) + np.abs(grid.node_iy[a] - grid.node_iy[b])
    ok = gap <= 3  # drop teleports and parsing glitches
    rows, cols = [a[ok]], [b[ok]]

    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            jx, jy = grid.node_ix + dx, grid.node_iy + dy
            inside = (jx >= 0) & (jx < grid.nx) & (jy >= 0) & (jy < grid.ny)
            src = np.flatnonzero(inside)
            nbr = grid.col_nodes[jy[src] * grid.nx + jx[src]]  # (k, MAX_FLOORS)
            for f in range(MAX_FLOORS):
                cand = nbr[:, f]
                ok = (cand >= 0) & (np.abs(grid.node_z[np.maximum(cand, 0)] - grid.node_z[src]) <= STEP_HEIGHT)
                rows.append(src[ok])
                cols.append(cand[ok])
    r, c = np.concatenate(rows), np.concatenate(cols)
    return np.concatenate([r, c]).astype(np.int32), np.concatenate([c, r]).astype(np.int32)


def _largest_component(n: int, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    g = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    _, labels = csgraph.connected_components(g, directed=False)
    return labels == np.bincount(labels).argmax()


def _subset(grid: NavGrid, keep: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> NavGrid:
    new_id = np.full(grid.n, -1, dtype=np.int32)
    new_id[keep] = np.arange(keep.sum(), dtype=np.int32)
    col_nodes = np.where(grid.col_nodes >= 0, new_id[np.maximum(grid.col_nodes, 0)], -1)
    # Re-pack each column so existing floors come first, still sorted by height.
    col_nodes = np.sort(np.where(col_nodes < 0, np.iinfo(np.int32).max, col_nodes), axis=1)
    col_nodes = np.where(col_nodes == np.iinfo(np.int32).max, -1, col_nodes).astype(np.int32)
    ok = keep[rows] & keep[cols]
    adj = sparse.csr_matrix(
        (np.ones(ok.sum(), dtype=np.float32), (new_id[rows[ok]], new_id[cols[ok]])), shape=(keep.sum(),) * 2
    )
    adj.sum_duplicates()
    adj.setdiag(0)
    adj.eliminate_zeros()
    return NavGrid(
        x0=grid.x0, y0=grid.y0, nx=grid.nx, ny=grid.ny,
        node_ix=grid.node_ix[keep], node_iy=grid.node_iy[keep], node_z=grid.node_z[keep],
        node_count=grid.node_count[keep], node_place=grid.node_place[keep], node_rep=grid.node_rep[keep],
        col_nodes=col_nodes,
        adj_indptr=adj.indptr.astype(np.int32), adj_indices=adj.indices.astype(np.int32),
        fine_nx=0, fine_ny=0, fine_count=grid.fine_count, fine_zlo=grid.fine_zlo, fine_zhi=grid.fine_zhi,
    )  # fmt: skip


def _fill_fine_raster(grid: NavGrid, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> None:
    """Per-pixel sample counts and floor heights (lowest and highest floor)."""
    fnx = int(np.ceil(grid.nx * CELL / FINE_CELL))
    fny = int(np.ceil(grid.ny * CELL / FINE_CELL))
    fx = np.clip(((x - grid.x0) // FINE_CELL).astype(np.int64), 0, fnx - 1)
    fy = np.clip(((y - grid.y0) // FINE_CELL).astype(np.int64), 0, fny - 1)
    pix = fy * fnx + fx
    count = np.bincount(pix, minlength=fnx * fny)
    zlo = np.full(fnx * fny, np.inf)
    np.minimum.at(zlo, pix, z)
    upper = z > zlo[pix] + FLOOR_GAP
    zhi = zlo.copy()
    if upper.any():
        up = np.full(fnx * fny, np.inf)
        np.minimum.at(up, pix[upper], z[upper])
        has_up = np.isfinite(up)
        zhi[has_up] = up[has_up]
    empty = count == 0
    zlo[empty] = np.nan
    zhi[empty] = np.nan
    grid.fine_nx, grid.fine_ny = fnx, fny
    grid.fine_count = count.reshape(fny, fnx).astype(np.int32)
    grid.fine_zlo = zlo.reshape(fny, fnx).astype(np.float32)
    grid.fine_zhi = zhi.reshape(fny, fnx).astype(np.float32)
