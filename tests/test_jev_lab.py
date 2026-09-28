"""Jev prompt experiments (jev_lab.py): facts computed for Jev, and how answers become probabilities."""

import numpy as np

from cspredict.jev_lab import _softmax, callout_facts, fit_formula, jev_probs, request_key

NAMES = ["A", "B", "C"]


def _moment(last="A", since=4):
    return {
        "demo": "d", "friendly": "ct", "round": 3, "step": 40, "label": "T2",
        "evidence": {
            "last_seen_at": last, "seconds_since_seen": since,
            "spot_chance_now": {"A": 0.30, "B": 0.12, "C": 0.0},
            "spot_chance_since_seen": {"A": 0.05, "B": 0.30, "C": 0.11},
            "usual_share": {"A": 0.20, "B": 0.05, "C": 0.01},
        },
    }  # fmt: skip


def test_callout_facts_use_fixed_thresholds_and_travel_times():
    index = {"A": 0, "B": 1, "C": 2}
    run_s = np.array([[0.0, 1.5, 6.0], [1.5, 0.0, 3.0], [6.0, 3.0, 0.0]])  # running seconds between callouts
    m = _moment(last="A", since=4)
    a, b, c = (callout_facts(m, x, index, run_s) for x in NAMES)
    assert a["last_seen_here"] == "yes, 4 s ago" and "reachable_since_last_seen" not in a
    assert b["reachable_since_last_seen"] == "yes, even at a silent walk"  # 1.5 s running ~ 3.1 s walking <= 5
    assert c["reachable_since_last_seen"] == "no, too far"  # 6 s running > 4 + 1
    assert (a["your_team_can_see_it_now"], b["your_team_can_see_it_now"], c["your_team_can_see_it_now"]) == ("yes", "partly", "no")
    assert (a["watched_since_last_seen"], b["watched_since_last_seen"], c["watched_since_last_seen"]) == ("hardly", "mostly", "partly")
    assert (a["how_often_that_side_is_here_now"], b["how_often_that_side_is_here_now"], c["how_often_that_side_is_here_now"]) == ("often", "sometimes", "rarely")


def test_split_answers_combine_stay_and_where():
    m = _moment(last="B")
    cache = {request_key(m, "split"): {"answers": {"stay": {"noul": 0.4}, "where": {"probabilities": {"A": 0.75, "C": 0.25}}}}}
    np.testing.assert_allclose(jev_probs([m], cache, "split", NAMES)[0], [0.6 * 0.75, 0.4, 0.6 * 0.25])
    assert np.isnan(jev_probs([_moment(last="A")] , {}, "focus", NAMES)).all()  # unanswered stays missing


def test_evidence_formula_recovers_a_known_weight():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(4000, 5, 2))
    w_true = np.array([1.5, -0.5])
    truth = np.array([rng.choice(5, p=q) for q in _softmax(X, w_true)])
    np.testing.assert_allclose(fit_formula(X, truth), w_true, atol=0.15)
