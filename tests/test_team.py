"""Team coordination: recorded team snapshots as a joint prior over enemy callouts."""

import numpy as np


def _team_library(places: np.ndarray, clock: float = 20.0):
    from cspredict.team import TeamLibrary

    q = len(places)
    ptr = np.zeros(7, dtype=np.int64)
    ptr[1:] = q  # every snapshot in block side 0 (T), phase 0 (before the plant)
    return TeamLibrary(
        place=places.astype(np.int8), n_alive=(places >= 0).sum(axis=1).astype(np.int8),
        clock=np.full(q, clock, dtype=np.float32), buy=np.full(q, 2, dtype=np.int8), ptr=ptr,
    )  # fmt: skip


def test_team_correction_is_one_for_independent_teammates():
    from cspredict.team import TeamParams, team_callouts

    # Every combination of callouts for 2 players, each callout equally common: positions independent.
    grid = np.array([(a, b) for a in range(4) for b in range(4)])
    lib = _team_library(grid)
    P = np.array([[0.7, 0.1, 0.1, 0.1], [0.25, 0.25, 0.25, 0.25]])
    R = team_callouts(lib, 0, 0, 20.0, 2, P, TeamParams(pseudo=0.0))
    np.testing.assert_allclose(R, 1.0, atol=1e-9)


def test_team_correction_follows_stacks():
    from cspredict.team import TeamParams, team_callouts

    # Recorded teams always stack: both players in the same callout.
    lib = _team_library(np.array([(c, c) for c in range(4)] * 5))
    seen = np.array([1e-6, 1.0, 1e-6, 1e-6])  # enemy 0 is on the radar in callout 1
    hidden = np.full(4, 0.25)  # enemy 1: no information of its own
    R = team_callouts(lib, 0, 0, 20.0, 2, np.array([seen / seen.sum(), hidden]), TeamParams(pseudo=0.0))
    q = hidden * R[1]
    q /= q.sum()
    assert q[1] > 0.99




def test_best_pairing_keeps_one_pairing_per_snapshot():
    from cspredict.team import TeamParams, team_callouts

    # One recorded team: players in callouts 0 and 1. Enemy 0 is seen in callout 0, so the best
    # pairing puts enemy 1 in callout 1; summing over pairings keeps some weight on callout 0.
    lib = _team_library(np.array([(0, 1)] * 10))
    P = np.array([[0.9, 0.05, 0.05], [0.4, 0.4, 0.2]])
    R_best = team_callouts(lib, 0, 0, 20.0, 2, P, TeamParams(pseudo=0.0, best_pairing=True))
    R_sum = team_callouts(lib, 0, 0, 20.0, 2, P, TeamParams(pseudo=0.0))
    q_best, q_sum = P[1] * R_best[1], P[1] * R_sum[1]
    assert q_best[1] / q_best.sum() > 0.99
    assert 0.5 < q_sum[1] / q_sum.sum() < 0.99
