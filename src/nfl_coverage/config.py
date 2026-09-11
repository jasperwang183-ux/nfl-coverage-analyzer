"""Shared paths and constants."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"

BDB2021 = DATA_RAW / "bdb2021"  # 2018 season, all pass plays, pre-snap frames; labels via nflverse
BDB2026 = DATA_RAW / "bdb2026" / "114239_nfl_competition_files_published_analytics_final"

MODELS_DIR = ROOT / "models"  # trained model bundles (gitignored — derived from competition data)

# Field dimensions in yards (x includes both end zones)
FIELD_LENGTH = 120.0
FIELD_WIDTH = 53.3

# Tracking data sample rate
FRAMES_PER_SECOND = 10

# Canonical coverage labels (BDB 2026 enums; nflverse values map onto these via
# data/nflverse.py COVERAGE_MAP)
COVERAGE_CLASSES = [
    "COVER_0_MAN",
    "COVER_1_MAN",
    "COVER_2_MAN",
    "COVER_2_ZONE",
    "COVER_3_ZONE",
    "COVER_4_ZONE",
    "COVER_6_ZONE",
    "PREVENT",
]
MAN_ZONE_CLASSES = ["MAN_COVERAGE", "ZONE_COVERAGE"]
