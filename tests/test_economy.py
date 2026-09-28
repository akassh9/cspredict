"""The enemy buy as the friendly team can infer it: round history and guns seen (economy.py)."""

from types import SimpleNamespace

import numpy as np
import polars as pl

from cspredict.economy import BuyModel, best_guns, buy_weights, history_key
from cspredict.infostate import OTHER_WEAPON, WEAPON_CLASSES, BuyHistory, buy_histories, match_rounds


def _rounds(winners: list[str], first_freeze_end: int | None = 100) -> pl.DataFrame:
    n = len(winners)
    return pl.DataFrame({
        "round_num": pl.Series(range(1, n + 1), dtype=pl.UInt32),
        "freeze_end": [first_freeze_end] + [1000 * r for r in range(2, n + 1)],
        "winner": winners,
    })  # fmt: skip


def _kills(round_num: int, weapons: list[str]) -> pl.DataFrame:
    return pl.DataFrame({"round_num": pl.Series([round_num] * len(weapons), dtype=pl.UInt32), "weapon": weapons})


def test_knife_round_is_not_a_match_round():
    rounds = _rounds(["ct", "t", "t"], first_freeze_end=None)
    assert match_rounds(rounds, _kills(1, ["knife_karambit", "knife_t", "world"])) == {2: 1, 3: 2}
    # A first round without freeze time but with gun kills is a pistol round recorded late.
    assert match_rounds(rounds, _kills(1, ["glock", "usp_silencer"])) == {1: 1, 2: 2, 3: 3}


def test_round_history():
    # The T side loses the pistol round, wins round 2, then loses three: its loss-bonus count goes
    # 1, 0, 1, 2, 3 (the CT side's 0, 1, 0, 0, 0). Sides swap after 12 rounds, so round 13 is a
    # fresh pistol round.
    winners = ["ct", "t", "ct", "ct", "ct"] + ["t"] * 7 + ["ct", "t"]
    h = buy_histories(_rounds(winners), _kills(1, ["glock"]))
    assert h[(1, "t")] == BuyHistory(half_round=1, overtime=False, won_last=None, won_pistol=None, losses=0)
    assert h[(2, "t")] == BuyHistory(half_round=2, overtime=False, won_last=False, won_pistol=False, losses=1)
    assert h[(3, "t")].losses == 0 and h[(3, "t")].won_last
    assert h[(6, "t")].losses == 3 and h[(6, "ct")].losses == 0
    assert h[(13, "t")].half_round == 1 and h[(13, "t")].losses == 0
    assert h[(14, "ct")] == BuyHistory(half_round=2, overtime=False, won_last=True, won_pistol=True, losses=0)


def test_overtime_halves():
    h = buy_histories(_rounds(["t", "ct"] * 12 + ["ct"] * 4), _kills(1, ["glock"]))
    assert [h[(r, "t")].half_round for r in (25, 27, 28)] == [1, 3, 1]
    assert h[(26, "t")].overtime and h[(26, "t")].won_pistol is None and h[(26, "t")].losses == 1


def test_history_keys():
    usual = {"half_round": 5, "overtime": False, "won_last": False, "won_pistol": True, "losses": 2}
    key = lambda **kw: history_key(BuyHistory(**usual | kw))  # noqa: E731
    assert key(half_round=1, won_last=None) == "pistol round"
    assert key(half_round=2, won_last=True) == "after winning the pistol round"
    assert key(half_round=3, won_pistol=False) == "lost the pistol round and round 2"
    assert key(won_last=True) == "after a win"
    assert key(losses=7) == "after a loss, loss count 4"
    assert key(overtime=True) == "overtime"
    assert history_key(None) == "unknown"


def test_best_guns_keep_the_dearest_gun_shown():
    sniper, rifle, smg, pistol = (WEAPON_CLASSES.index(c) for c in ("sniper", "rifle", "smg", "pistol"))
    seen = np.array([[pistol, -1], [OTHER_WEAPON, -1], [rifle, -1], [pistol, smg]])
    kill = np.array([[-1, -1], [-1, sniper], [-1, -1], [-1, -1]])
    got = best_guns(SimpleNamespace(enemy_seen_weapon=seen, enemy_kill_weapon=kill))
    np.testing.assert_array_equal(got, [[1, 0], [1, 4], [3, 4], [3, 4]])


def test_buy_weights():
    # A known class reproduces the old per-class weights; an uncertain buy gives the expected weight.
    np.testing.assert_allclose(buy_weights(2, 0.7), [0.49, 0.7, 1.0])
    np.testing.assert_allclose(buy_weights(np.array([0.0, 0.0, 1.0]), 0.7), [0.49, 0.7, 1.0])
    np.testing.assert_allclose(buy_weights(np.array([0.5, 0.0, 0.5]), 0.7), [0.745, 0.7, 0.745])


def test_posterior_follows_history_then_guns():
    model = BuyModel(
        prior={"pistol round": [0.98, 0.01, 0.01], "after a loss, loss count 2": [0.4, 0.3, 0.3]},
        base=[0.2, 0.2, 0.6], guns=[[0.9, 0.05, 0.04, 0.01], [0.3, 0.2, 0.45, 0.05], [0.01, 0.04, 0.8, 0.15]],
    )  # fmt: skip
    rifle = WEAPON_CLASSES.index("rifle")
    ep = SimpleNamespace(
        n_steps=3, enemy_history=BuyHistory(half_round=6, overtime=False, won_last=False, won_pistol=True, losses=2),
        enemy_seen_weapon=np.array([[-1, -1], [rifle, -1], [-1, rifle]]), enemy_kill_weapon=np.full((3, 2), -1),
    )  # fmt: skip
    q = model.posterior(ep)
    np.testing.assert_allclose(q.sum(axis=1), 1.0)
    np.testing.assert_allclose(q[0], [0.4, 0.3, 0.3])  # nothing seen yet: the round history alone
    assert q[1, 2] > q[0, 2] and q[2, 2] > q[1, 2] > 0.5  # every rifle seen makes a full buy likelier
    np.testing.assert_allclose(model.posterior(ep, weapons=False), np.tile([0.4, 0.3, 0.3], (3, 1)))
    ep.enemy_history = None  # history unknown: the overall rates
    np.testing.assert_allclose(model.posterior(ep, weapons=False)[0], [0.2, 0.2, 0.6])
