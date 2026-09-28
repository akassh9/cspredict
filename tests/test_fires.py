"""Molotov and incendiary fires: profile lookup, active fires, and their effect on the filters."""

import numpy as np
import polars as pl


class _Grid:
    """The two NavGrid attributes FireModel.node_factors reads."""

    def __init__(self, xy, z):
        self.node_xy, self.node_z = np.asarray(xy, dtype=np.float32), np.asarray(z, dtype=np.float32)
        self.n = len(z)


def test_fire_node_factors():
    from cspredict.fires import FireModel

    fm = FireModel(edges=np.array([0.0, 100.0, 200.0]), occupancy=np.array([0.1, 0.5]), step=np.array([0.2, 0.6]), radius=150.0)
    grid = _Grid([[0, 0], [150, 0], [250, 0], [0, 0]], [0, 0, 0, 300])  # last node: same spot, floor above
    np.testing.assert_allclose(fm.node_factors(grid, np.array([[0.0, 0.0, 0.0]]), "occupancy"), [0.1, 0.5, 1.0, 1.0])
    # Overlapping fires: the smaller factor wins; no fires: no effect.
    two = np.array([[0.0, 0.0, 0.0], [400.0, 0.0, 0.0]])  # the second fire is 150 units from node 2
    np.testing.assert_allclose(fm.node_factors(grid, two, "step"), [0.2, 0.6, 0.6, 1.0])
    np.testing.assert_allclose(fm.node_factors(grid, np.zeros((0, 3)), "step"), 1.0)


def test_active_fires():
    from cspredict.fires import SPREAD_S, active_fires
    from cspredict.config import TICKRATE

    inf = pl.DataFrame({"X": [1.0], "Y": [2.0], "Z": [3.0], "start_tick": [1000], "end_tick": [1400]})
    spread = int(SPREAD_S * TICKRATE)
    got = active_fires(inf, np.array([999, 1000 + spread, 1399, 1400]))
    assert [len(g) for g in got] == [0, 1, 1, 0]


def test_masked_predict_keeps_probability_and_redirects_blocked_moves():
    from scipy import sparse

    from cspredict.filters import predict

    # Three cells in a row; from cell 0 half the players stay and half move on to cell 1.
    t = sparse.csr_matrix(np.array([[0.5, 0.5, 0.0], [0.0, 0.5, 0.5], [0.0, 0.0, 1.0]]))
    b = np.array([[1.0, 0.0, 0.0]])
    np.testing.assert_allclose(predict(b, t), [[0.5, 0.5, 0.0]])
    # Cell 1 is burning (factor 0.1): most of the move is blocked and those players wait in cell 0.
    out = predict(b, t, np.array([1.0, 0.1, 1.0]))
    np.testing.assert_allclose(out.sum(), 1.0)
    np.testing.assert_allclose(out, [[0.5 / 0.55, 0.05 / 0.55, 0.0]])
