"""Fit the map grid, spotting model, motion models and the other fitted parts on the training demos.

Usage:
    python -m cspredict.build --sources xego          # writes data/maps/de_mirage/xego/
    python -m cspredict.build --sources hltv xego     # pool several sources
    python -m cspredict.build --sources hltv xego --only library motion   # refit some parts, keep the rest
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from cspredict.calibrate import Calibration
from cspredict.config import MAP_NAME, MAPS_DIR
from cspredict.dataset import list_demos, load_ticks
from cspredict.economy import BuyModel, fit_buy_model
from cspredict.fires import FireModel, fit_fire_model
from cspredict.grid import NavGrid, build_grid
from cspredict.motion import MotionModel, fit_motion_model
from cspredict.particles import TrajectoryLibrary, build_library
from cspredict.team import TeamLibrary, build_team_library
from cspredict.visibility import SpotModel, fit_spot_model


@dataclass
class Models:
    grid: NavGrid
    spot: SpotModel
    motion: MotionModel
    library: TrajectoryLibrary
    fires: FireModel | None = None
    teams: TeamLibrary | None = None
    calibration: Calibration | None = None
    buy: BuyModel | None = None


PARTS = ("grid", "spot", "motion", "library", "fires", "team", "buy")


def model_dir(sources: list[str]) -> Path:
    return MAPS_DIR / MAP_NAME / "+".join(sorted(sources))


def load_models(path: Path) -> Models:
    return Models(
        grid=NavGrid.load(path / "grid.npz"),
        spot=SpotModel.load(path / "spot.npz"),
        motion=MotionModel.load(path / "motion.npz"),
        library=TrajectoryLibrary.load(path / "library.npz"),
        fires=FireModel.load(path / "fires.npz") if (path / "fires.npz").exists() else None,
        teams=TeamLibrary.load(path / "team.npz") if (path / "team.npz").exists() else None,
        calibration=Calibration.load(path / "calibration.json") if (path / "calibration.json").exists() else None,
        buy=BuyModel.load(path / "buy.json") if (path / "buy.json").exists() else None,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Fit grid, spotting and motion models.")
    ap.add_argument("--sources", nargs="+", default=["xego"], help="parsed data sources to train on")
    ap.add_argument("--only", nargs="+", choices=PARTS, default=list(PARTS), help="refit only these parts")
    args = ap.parse_args()

    train = list_demos(args.sources, ["train"])
    if not train:
        raise SystemExit(f"No parsed training demos for {args.sources}; run `python -m cspredict.parse` first")
    out = model_dir(args.sources)
    out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()

    if "grid" in args.only:
        logger.info(f"Building grid from {len(train)} training demos")
        grid = build_grid(load_ticks(train, ["round_num", "tick", "steamid", "X", "Y", "Z", "is_alive", "place"]))
        grid.save(out / "grid.npz")
        logger.info(f"Grid: {grid.n} nodes, {len(grid.places)} callouts")
    else:
        grid = NavGrid.load(out / "grid.npz")
    if "spot" in args.only:
        fit_spot_model(grid, train).save(out / "spot.npz")
    if "motion" in args.only:
        fit_motion_model(grid, train).save(out / "motion.npz")
    if "library" in args.only:
        build_library(grid, train).save(out / "library.npz")
    if "fires" in args.only:
        fit_fire_model(grid, MotionModel.load(out / "motion.npz"), train).save(out / "fires.npz")
    if "team" in args.only:
        build_team_library(grid, train).save(out / "team.npz")
    if "buy" in args.only:
        fit_buy_model(grid, train).save(out / "buy.json")
    (out / "train_demos.json").write_text(json.dumps([r.demo_id for r in train], indent=2))
    logger.info(f"Models written to {out} in {time.perf_counter() - start:.0f} s")


if __name__ == "__main__":
    main()
