"""Jev against GPT-6 Luna (frontier_bench.py): the wrapper, answer clean-up, cost and tie credit."""

import json

import numpy as np
import pytest

from cspredict.frontier_bench import answer_schema, cost_usd, gpt_input, gpt_probs, metrics, parse_answers
from cspredict.jev_lab import request_key

QUESTIONS = {
    "stay": {"type": "noul", "instructions": "Is T2 still in B?", "criteria": {"true": "Still in B.", "false": "Left B."}},
    "where": {"type": "choice", "instructions": "Where is T2?", "criteria": {"Bombsite A": {"connects_to": "B"}, "C": "x"}},
}


def test_input_is_the_request_jev_gets():
    req = {"state": {"bomb": "not planted", "enemies": [{"id": "T2"}]}, "questions": QUESTIONS}
    assert json.loads(gpt_input(req)) == req


def test_schema_asks_for_every_option_and_nothing_else():
    s = answer_schema(QUESTIONS)
    assert s["required"] == ["stay", "where"] and s["additionalProperties"] is False
    assert s["properties"]["stay"]["type"] == "number"
    where = s["properties"]["where"]
    assert where["required"] == ["Bombsite A", "C"] and where["additionalProperties"] is False


def test_answers_are_cleaned_like_jevs():
    a = parse_answers(json.dumps({"stay": 1.2, "where": {"Bombsite A": 3, "C": -1}}), QUESTIONS)
    assert a == {"stay": {"noul": 1.0}, "where": {"probabilities": {"Bombsite A": 1.0, "C": 0.0}}}
    with pytest.raises(ValueError):
        parse_answers(json.dumps({"stay": 0.5, "where": {"Bombsite A": 0, "C": 0}}), QUESTIONS)
    with pytest.raises(ValueError):
        parse_answers("not json", QUESTIONS)


def test_cost_counts_cached_and_written_input_at_their_own_prices():
    usage = {"input_tokens": 1_000_000, "cached_tokens": 200_000, "cache_write_tokens": 100_000, "output_tokens": 2_000_000}
    expected = 0.7 * 0.10 + 0.2 * 0.01 + 0.1 * 0.125 + 2 * 0.50
    assert cost_usd(usage) == pytest.approx(expected)
    assert cost_usd(usage, tier="flex") == pytest.approx(expected / 2)


def test_ties_get_split_credit():
    p = np.array([[0.4, 0.4, 0.2], [0.5, 0.3, 0.2], [0.2, 0.2, 0.2]])
    m = metrics(p, np.array([0, 1, 2]))
    np.testing.assert_allclose(m["top1"], [0.5, 0.0, 1 / 3])
    np.testing.assert_allclose(metrics(p[:1], np.array([0]), ties="miss")["top1"], [0.0])  # the rule of evaluate.py
    q = np.array([[0.1, 0.3, 0.3, 0.3]])  # truth tied with two others for places 1-3
    assert metrics(q, np.array([1]))["top1"][0] == pytest.approx(1 / 3)
    assert metrics(q, np.array([1]))["top3"][0] == pytest.approx(1.0)


def test_unusable_answers_are_scored_as_uniform():
    m = {"demo": "d", "friendly": "ct", "round": 3, "step": 40, "label": "T2", "evidence": {"last_seen_at": "B"}}
    ok = dict(m, step=41)
    cache = {request_key(ok, "split"): {"answers": {"stay": {"noul": 0.4}, "where": {"probabilities": {"A": 1.0, "C": 0.0}}}}}
    p, missing = gpt_probs([m, ok], cache, "split", ["A", "B", "C"])
    assert missing.tolist() == [True, False]
    np.testing.assert_allclose(p, [[1 / 3] * 3, [0.6, 0.4, 0.0]])
