"""Parse CS2 demos into compact parquet tables for modeling.

Each demo becomes one directory, data/parsed/<source>/<demo_id>/, holding meta.json plus
rounds, ticks (8 Hz, in-play only), kills, bomb, smokes, infernos, shots, footsteps and
blinds tables. <source> is the first folder under data/raw (e.g. "hltv", "xego").

HLTV serves demos as .rar archives holding every map of a series; archives dropped into
data/raw/<source>/ are unpacked first, and only demos recorded on MAP_NAME are parsed.

Usage:
    python -m cspredict.parse           # parse every new demo under data/raw
    python -m cspredict.parse --force   # re-parse everything
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

import polars as pl
from awpy import Demo
from demoparser2 import DemoParser
from loguru import logger

from cspredict.config import MAP_NAME, PARSED_DIR, RAW_DIR, SAMPLE_EVERY, TICKRATE

PLAYER_PROPS = [
    "yaw",
    "pitch",
    "is_alive",
    "team_num",
    "spotted",
    "approximate_spotted_by",
    "active_weapon_name",
    "current_equip_value",
    "round_start_equip_value",
    "balance",
    "armor_value",
    "has_helmet",
    "has_defuser",
]

# Fallback lifetimes when a grenade's expiry event is missing.
SMOKE_SECONDS = 18.0
INFERNO_SECONDS = 7.0


def extract_archives(raw_dir: Path) -> None:
    """Unpack .rar/.zip archives into a sibling folder named after the archive."""
    archives = sorted([*raw_dir.rglob("*.rar"), *raw_dir.rglob("*.zip")])
    if archives and shutil.which("bsdtar") is None:
        raise RuntimeError("bsdtar is needed to unpack demo archives (install libarchive)")
    for archive in archives:
        dest = archive.with_suffix("")
        if dest.exists():
            continue
        dest.mkdir(parents=True)
        subprocess.run(["bsdtar", "-xf", str(archive), "-C", str(dest)], check=True)
        logger.info(f"Extracted {archive.name}")


def _existing(df: pl.DataFrame, cols: list[str]) -> list[str]:
    return [c for c in cols if c in df.columns]


def _event(parser: DemoParser, name: str, **kwargs) -> pl.DataFrame:
    """One game event as a DataFrame; demoparser2 returns an empty list if it never fires."""
    out = parser.parse_event(name, **kwargs)
    return pl.from_pandas(out) if hasattr(out, "columns") else pl.DataFrame({"tick": pl.Series([], dtype=pl.Int32)})


def _in_play_ticks(ticks: pl.DataFrame, rounds: pl.DataFrame) -> pl.DataFrame:
    """Keep every SAMPLE_EVERY-th tick between the end of freeze time and the round end."""
    return (
        ticks.filter(pl.col("tick") % SAMPLE_EVERY == 0)
        .join(rounds.select("round_num", "freeze_end", "end"), on="round_num")
        .filter(pl.col("tick").is_between(pl.col("freeze_end"), pl.col("end")))
        .select(
            "round_num",
            "tick",
            pl.col("steamid").cast(pl.UInt64),
            "name",
            "side",
            pl.col("X").cast(pl.Float32),
            pl.col("Y").cast(pl.Float32),
            pl.col("Z").cast(pl.Float32),
            pl.col("yaw").cast(pl.Float32),
            pl.col("pitch").cast(pl.Float32),
            "is_alive",
            pl.col("health").cast(pl.Int16),
            pl.col("armor").cast(pl.Int16),
            "has_helmet",
            "has_defuser",
            "spotted",
            pl.col("approximate_spotted_by").cast(pl.List(pl.UInt64)).alias("spotted_by"),
            "place",
            pl.col("active_weapon_name").alias("weapon"),
            pl.col("current_equip_value").cast(pl.Int32).alias("equip"),
            pl.col("round_start_equip_value").cast(pl.Int32).alias("round_start_equip"),
            pl.col("balance").cast(pl.Int32),
        )
        .sort("tick", "steamid")
    )


def _with_round(events: pl.DataFrame, rounds: pl.DataFrame, tick_col: str = "tick") -> pl.DataFrame:
    """Attach round_num to events that fall between freeze end and round end."""
    spans = rounds.select("round_num", "freeze_end", "end").sort("freeze_end")
    return (
        events.sort(tick_col)
        .join_asof(spans, left_on=tick_col, right_on="freeze_end", strategy="backward")
        .filter(pl.col(tick_col) <= pl.col("end"))
        .drop("freeze_end", "end")
    )


def _fix_bomb_sites(rounds: pl.DataFrame, bomb: pl.DataFrame, ticks: pl.DataFrame) -> pl.DataFrame:
    """Label each plant with the nearer bombsite. awpy's own label needs map data that may be
    missing, so compare the plant position with where players stood inside each site (the
    game's own "BombsiteA"/"BombsiteB" place names)."""
    centres = {}
    for site in ("A", "B"):
        inside = ticks.filter(pl.col("place") == f"Bombsite{site}")
        if inside.height:
            centres[site] = (inside["X"].mean(), inside["Y"].mean())
    plants = bomb.filter(pl.col("event") == "plant").group_by("round_num").agg(pl.col("X").first(), pl.col("Y").first())
    if len(centres) < 2 or plants.height == 0:
        return rounds
    site = {
        r["round_num"]: "bombsite_" + min(centres, key=lambda s: (r["X"] - centres[s][0]) ** 2 + (r["Y"] - centres[s][1]) ** 2).lower()
        for r in plants.iter_rows(named=True)
    }
    return rounds.with_columns(
        pl.when(pl.col("bomb_plant").is_not_null())
        .then(pl.col("round_num").replace_strict(site, default=None, return_dtype=pl.String))
        .otherwise(pl.lit("not_planted"))
        .alias("bomb_site")
    )


def _grenades(dem: Demo, parser: DemoParser, kind: str) -> pl.DataFrame:
    """Smokes or infernos with an end tick taken from the expiry event when available."""
    table, expire_event, seconds = {
        "smokes": (dem.smokes, "smokegrenade_expired", SMOKE_SECONDS),
        "infernos": (dem.infernos, "inferno_expire", INFERNO_SECONDS),
    }[kind]
    table = table.select(_existing(table, ["entity_id", "start_tick", "X", "Y", "Z", "round_num", "thrower_side"]))
    expired = _event(parser, expire_event)
    if len(expired) and "entityid" in expired.columns:
        ends = expired.group_by("entityid").agg(pl.col("tick").min().alias("end_tick"))
        table = table.join(
            ends.rename({"entityid": "entity_id"}).with_columns(pl.col("entity_id").cast(table["entity_id"].dtype)),
            on="entity_id",
            how="left",
        )
    else:
        table = table.with_columns(pl.lit(None, dtype=pl.Int64).alias("end_tick"))
    # An entity id can be reused later in the match, so only accept expiries after the start.
    fallback = pl.col("start_tick") + int(seconds * TICKRATE)
    valid = pl.col("end_tick").is_not_null() & (pl.col("end_tick") > pl.col("start_tick"))
    valid = valid & (pl.col("end_tick") - pl.col("start_tick") < 2 * seconds * TICKRATE)
    return table.with_columns(pl.when(valid).then(pl.col("end_tick")).otherwise(fallback).alias("end_tick"))


def parse_demo(dem_path: Path, out_dir: Path, source: str, demo_id: str) -> dict | None:
    """Parse one demo into out_dir. Returns its metadata, or None if it is not on MAP_NAME."""
    parser = DemoParser(str(dem_path))
    header = parser.parse_header()
    if header.get("map_name") != MAP_NAME:
        return None

    start = time.perf_counter()
    dem = Demo(dem_path, tickrate=TICKRATE)
    dem.parse(player_props=PLAYER_PROPS)
    ticks = _in_play_ticks(dem.ticks, dem.rounds)
    rounds = _fix_bomb_sites(dem.rounds, dem.bomb, ticks)

    kills = dem.kills.select(
        _existing(
            dem.kills,
            [
                "tick", "round_num",
                "attacker_steamid", "attacker_side", "attacker_X", "attacker_Y", "attacker_Z", "attacker_spotted",
                "victim_steamid", "victim_side", "victim_X", "victim_Y", "victim_Z",
                "weapon", "headshot", "thrusmoke", "penetrated", "attackerblind",
            ],
        )
    )
    shots = dem.shots.select(
        _existing(
            dem.shots,
            ["tick", "round_num", "player_steamid", "player_side", "player_X", "player_Y", "player_Z",
             "player_active_weapon_name", "silenced"],
        )
    ).rename({"player_active_weapon_name": "weapon"}, strict=False)
    footsteps = _with_round(
        _event(parser, "player_footstep", player=["X", "Y", "Z"]).rename(
            {"user_steamid": "steamid", "user_X": "X", "user_Y": "Y", "user_Z": "Z", "user_team_num": "team_num"},
            strict=False,
        ),
        rounds,
    )
    blinds = _with_round(_event(parser, "player_blind"), rounds)

    tables = {
        "rounds": rounds,
        "ticks": ticks,
        "kills": kills,
        "bomb": dem.bomb,
        "smokes": _grenades(dem, parser, "smokes"),
        "infernos": _grenades(dem, parser, "infernos"),
        "shots": shots,
        "footsteps": footsteps,
        "blinds": blinds,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.write_parquet(out_dir / f"{name}.parquet")

    meta = {
        "demo_id": demo_id,
        "source": source,
        "file": str(dem_path),
        "map_name": header.get("map_name"),
        "server_name": header.get("server_name"),
        "patch_version": header.get("patch_version"),
        "n_rounds": rounds.height,
        "tickrate": TICKRATE,
        "sample_every": SAMPLE_EVERY,
        "parse_seconds": round(time.perf_counter() - start, 1),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description="Parse CS2 demos into parquet tables.")
    ap.add_argument("--raw", type=Path, default=RAW_DIR, help="folder holding <source>/<demos>")
    ap.add_argument("--force", action="store_true", help="re-parse demos that were already parsed")
    args = ap.parse_args()

    extract_archives(args.raw)
    for dem_path in sorted(args.raw.rglob("*.dem")):
        rel = dem_path.relative_to(args.raw)
        if len(rel.parts) < 2:
            logger.warning(f"Skipping {rel}: put demos in a source folder such as data/raw/hltv/")
            continue
        source, demo_id = rel.parts[0], "__".join(rel.with_suffix("").parts[1:])
        out_dir = PARSED_DIR / source / demo_id
        if (out_dir / "meta.json").exists() and not args.force:
            continue
        try:
            meta = parse_demo(dem_path, out_dir, source, demo_id)
        except Exception as exc:  # a truncated or corrupt demo should not stop the batch
            logger.error(f"Failed to parse {rel}: {exc}")
            continue
        if meta is None:
            logger.info(f"Skipped {rel} (not {MAP_NAME})")
        else:
            logger.info(f"Parsed {rel}: {meta['n_rounds']} rounds in {meta['parse_seconds']} s")


if __name__ == "__main__":
    main()
