"""Post-hoc calibration of callout probabilities (calibrate.py) and how the filters apply it."""

import numpy as np

from cspredict.calibrate import KNOTS, Calibration, fit_params, log_loss


def _random_p(rng, n=500, k=6):
    p = rng.dirichlet(np.full(k, 0.5), size=n)
    return p / p.sum(axis=1, keepdims=True)


def test_maps_are_monotone_distributions_and_identity_at_slope_one():
    rng = np.random.default_rng(0)
    p = _random_p(rng)
    for kind, params in (("power", [1.3]), ("platt", [1.2, -0.4]), ("piecewise", [1.1, 0.8, 0.9, 2.5])):
        q = Calibration(kind, [params]).apply(p)
        np.testing.assert_allclose(q.sum(axis=1), 1.0)
        assert (np.argsort(q, axis=1) == np.argsort(p, axis=1)).all()  # order of callouts unchanged
    # Identity up to the 1e-6 floor on vanishing probabilities.
    np.testing.assert_allclose(Calibration("power", [[1.0]]).apply(p), p, atol=1e-5)
    np.testing.assert_allclose(Calibration("piecewise", [[1.0] * (len(KNOTS) + 1)]).apply(p), p, atol=1e-5)


def test_power_fit_recovers_known_underconfidence():
    rng = np.random.default_rng(1)
    q = _random_p(rng, n=20000)
    truth = np.array([rng.choice(q.shape[1], p=row) for row in q])
    p = q**0.5
    p /= p.sum(axis=1, keepdims=True)  # a model that is too unsure: the right fix is power 2
    (alpha,) = fit_params("power", p, truth)
    assert 1.8 < alpha < 2.2
    fixed = Calibration("power", [[alpha]]).apply(p)
    assert log_loss(fixed, truth).mean() < log_loss(p, truth).mean()


def test_filter_output_keeps_cell_shares_within_callouts():
    from cspredict.filters import FilterConfig, _calibrated_output, _Post

    class Grid:
        places = ["A", "B"]
        node_place_idx = np.array([0, 0, 1])

    b = np.array([[0.2, 0.2, 0.6], [0.5, 0.25, 0.25]])
    cal = Calibration("power", [[2.0]])
    post = _Post(ep=None, grid=Grid(), ev=None, teams=None, plant_t=None, calibration=cal, since=np.array([[3.0, 30.0]]))
    out = _calibrated_output(b, FilterConfig("x", "none", negative=False, calibrated=True), post, 0)
    callouts = np.stack([out[:, :2].sum(axis=1), out[:, 2]], axis=1)
    np.testing.assert_allclose(callouts, cal.apply(np.array([[0.4, 0.6], [0.75, 0.25]])))
    np.testing.assert_allclose(out[:, 0] / out[:, 1], b[:, 0] / b[:, 1])  # shares inside callout A kept


def test_calibration_by_time_since_seen_leaves_unseen_enemies_alone():
    rng = np.random.default_rng(2)
    p = _random_p(rng, n=4)
    cal = Calibration("power", [[2.0], [1.0], [0.5]], since_edges=[5.0, 20.0])
    q = cal.apply(p, np.array([1.0, 10.0, 60.0, np.inf]))
    np.testing.assert_allclose(q[0], Calibration("power", [[2.0]]).apply(p[:1])[0])  # seen 1 s ago
    np.testing.assert_allclose(q[1], p[1], atol=1e-5)  # the 5-20 s map here is the identity
    np.testing.assert_allclose(q[2], Calibration("power", [[0.5]]).apply(p[2:3])[0])  # seen a minute ago
    np.testing.assert_allclose(q[3], p[3])  # never seen: unchanged
