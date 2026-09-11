"""Download Big Data Bowl datasets via the Kaggle API and unzip them.

Prereqs (see data/README.md):
  - `pip install kaggle`
  - kaggle.json token in ~/.kaggle/
  - Competition rules accepted on kaggle.com for each competition
"""

import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# bdb2021 (2018 season, all pass plays, pre-snap frames) gets coverage labels from
# nflverse participation data (src/nfl_coverage/data/nflverse.py); bdb2026 ships its own.
COMPETITIONS = {
    "nfl-big-data-bowl-2026-analytics": ROOT / "data" / "raw" / "bdb2026",
    "nfl-big-data-bowl-2021": ROOT / "data" / "raw" / "bdb2021",
}


def download(competition: str, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    print(f"\n=== {competition} -> {dest} ===")
    result = subprocess.run(
        ["kaggle", "competitions", "download", "-c", competition, "-p", str(dest)],
    )
    if result.returncode != 0:
        print(
            f"Download failed for {competition}. Make sure you've accepted the "
            "competition rules on kaggle.com and that ~/.kaggle/kaggle.json exists.",
            file=sys.stderr,
        )
        return

    for archive in dest.glob("*.zip"):
        print(f"Unzipping {archive.name} ...")
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(dest)
        archive.unlink()
    print(f"Done: {competition}")


if __name__ == "__main__":
    for comp, dest in COMPETITIONS.items():
        download(comp, dest)
