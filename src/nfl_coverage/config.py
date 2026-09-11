"""Shared paths and constants."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"

BDB2021 = DATA_RAW / "bdb2021"  # 2018 season, all pass plays, pre-snap frames; labels via nflverse
BDB2026 = DATA_RAW / "bdb2026" / "114239_nfl_competition_files_published_analytics_final"

# Field dimensions in yards (x includes both end zones)
FIELD_LENGTH = 120.0
FIELD_WIDTH = 53.3

# Tracking data sample rate
FRAMES_PER_SECOND = 10

# Coverage label sets (from PFF labels in BDB 2025 plays.csv)
COVERAGE_CLASSES = [
    "Cover-0",
    "Cover-1",
    "Cover-2",
    "Cover-3",
    "Cover-4",
    "Cover-6",
    "2-Man",
    "Prevent",
]
MAN_ZONE_CLASSES = ["Man", "Zone"]
