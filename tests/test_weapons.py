"""Weapon class at the last sighting (matching is implemented but off by default)."""

import numpy as np
import polars as pl

from cspredict.infostate import OTHER_WEAPON, WEAPON_CLASSES, weapon_class_expr
from cspredict.motion import trans_key
from cspredict.particles import _weapon_weight


def test_weapon_classes():
    names = ["AWP", "SSG 08", "AK-47", "M4A1-S", "MP9", "MAG-7", "Glock-18", "Desert Eagle", "Karambit", "Smoke Grenade", "C4 Explosive", None]
    got = pl.DataFrame({"weapon": names}).select(weapon_class_expr())["weapon"].to_list()
    want = ["sniper", "sniper", "rifle", "rifle", "smg", "smg", "pistol", "pistol", "other", "other", "other", "other"]
    assert [WEAPON_CLASSES[g] if g is not None else "other" for g in got] == want
    assert WEAPON_CLASSES[OTHER_WEAPON] == "other"


def test_weapon_weight():
    lib = np.array([0, 1, -1, 1, 3, 4])  # sniper, rifle, never seen, rifle, pistol, knife
    np.testing.assert_allclose(_weapon_weight(lib, 1, 0.5), [0.5, 1, 1, 1, 0.5, 0.5])  # unknown is not penalised
    np.testing.assert_allclose(_weapon_weight(lib, 1, 0.5, "guns"), [0.5, 1, 1, 1, 0.5, 1])  # knife in hand: unknown
    np.testing.assert_allclose(_weapon_weight(lib, 1, 0.5, "awp"), [0.5, 1, 1, 1, 1, 1])  # only sniper vs not
    np.testing.assert_allclose(_weapon_weight(lib, 0, 0.5, "awp"), [1, 0.5, 1, 0.5, 0.5, 1])
    np.testing.assert_allclose(_weapon_weight(lib, 4, 0.5, "awp"), 1.0)  # enemy seen with a knife: no information


def test_weapon_transition_keys():
    assert trans_key("t", 0, 2) == "t_pre_s2"
    assert trans_key("t", 0, 2, 0) == "t_pre_s2_w0"
    assert trans_key("ct", 1, 3, -1) == "ct_a_s3"
