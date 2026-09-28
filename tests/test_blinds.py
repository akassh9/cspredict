"""Flashed players rebuilt from every player's flash duration (parse._flash_blinds)."""

from pathlib import Path

import numpy as np
import polars as pl
import pytest


def test_flash_blinds():
    from cspredict.parse import _flash_blinds

    # Player 1 is flashed at tick 11, flashed again while still blind at 13, recovers at 15 and is
    # flashed at 16 by a grenade without a detonation event; player 2 is never flashed.
    flash = pl.DataFrame({
        "tick": pl.Series(list(range(10, 17)) * 2, dtype=pl.Int32),
        "steamid": pl.Series([1] * 7 + [2] * 7, dtype=pl.UInt64),
        "name": ["a"] * 7 + ["b"] * 7,
        "flash_duration": pl.Series([0.0, 3.2, 3.2, 3.4, 3.4, 0.0, 0.6] + [0.0] * 7, dtype=pl.Float32),
    }).sample(fraction=1.0, shuffle=True, seed=0)  # rows in any order
    det = pl.DataFrame({"entityid": [5, 6], "tick": [11, 13], "user_name": ["c", "d"], "user_steamid": ["3", "4"]})
    got = _flash_blinds(flash, det).sort("tick")
    assert got["tick"].to_list() == [11, 13, 16]
    assert got["user_steamid"].to_list() == ["1", "1", "1"]
    np.testing.assert_allclose(got["blind_duration"], [3.2, 3.4, 0.6], rtol=1e-6)
    assert got["attacker_steamid"].to_list() == ["3", "4", None]
    assert got["entityid"].to_list() == [5, 6, None]


def test_flash_blinds_match_player_blind_events():
    """On a demo whose server records player_blind, the rebuilt rows are exactly the event's."""
    from demoparser2 import DemoParser

    from cspredict.dataset import list_demos
    from cspredict.parse import _blinds

    files = [r.meta["file"] for r in list_demos(["xego"]) if Path(r.meta["file"]).exists()]
    if not files:
        pytest.skip("no raw X-Ego demo on disk; run `python -m cspredict.fetch_xego`")
    parser = DemoParser(files[0])
    events = pl.from_pandas(parser.parse_event("player_blind")).filter(pl.col("user_steamid").is_not_null())
    got = _blinds(parser)
    key = ["tick", "user_steamid", "attacker_steamid", "entityid"]
    assert events.join(got, on=key, how="anti").height == 0
    assert got.join(events, on=key, how="anti").height == 0
    pair = events.join(got, on=key)
    np.testing.assert_allclose(pair["blind_duration"], pair["blind_duration_right"], atol=1e-4)
