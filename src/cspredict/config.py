"""Project-wide constants and paths."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PARSED_DIR = DATA_DIR / "parsed"
MAPS_DIR = DATA_DIR / "maps"
OUTPUT_DIR = ROOT / "outputs"

MAP_NAME = "de_mirage"

TICKRATE = 64  # CS2 demos record 64 ticks per second
SAMPLE_EVERY = 8  # stored tick data keeps every 8th tick (8 Hz)
STEP_TICKS = 16  # the filter advances in 16-tick steps (0.25 s)

CELL = 64.0  # nav-grid cell size in game units (a player is 32 units wide)
FINE_CELL = 16.0  # occupancy raster used for line-of-sight ray casting
EYE_HEIGHT = 64.0  # standing eye height above the floor
