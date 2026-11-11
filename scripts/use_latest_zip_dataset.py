"""Utility to find and extract the latest dataset zip export.

Usage examples:
  python -m scripts.use_latest_zip_dataset
  python -m scripts.use_latest_zip_dataset --source datasets/crop_non_crop/crops_data --dest datasets/crop_non_crop/current_dataset --force
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
import zipfile
import shutil
import datetime


def find_latest_zip(source: Path) -> Path | None:
    zips = sorted(source.glob("*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
    return zips[0] if zips else None


def extract_zip(zip_path: Path, dest: Path, force: bool = False) -> None:
    if dest.exists():
        if not force:
            raise FileExistsError(f"Destination {dest} already exists (use --force to overwrite)")
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Find and extract the latest dataset zip export")
    p.add_argument("--source", default="datasets/crop_non_crop/crops_data", help="Directory containing dataset zip exports")
    p.add_argument("--dest", default="datasets/crop_non_crop/current_dataset", help="Destination directory to extract the latest zip into")
    p.add_argument("--force", action="store_true", help="Overwrite destination if it exists")
    p.add_argument("--dry-run", action="store_true", help="Show which zip would be extracted without extracting")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    source = Path(args.source)
    dest = Path(args.dest)

    if not source.exists() or not source.is_dir():
        print(f"Source directory does not exist: {source}")
        return 2

    latest = find_latest_zip(source)
    if not latest:
        print(f"No .zip files found in {source}")
        return 1

    mtime = datetime.datetime.fromtimestamp(latest.stat().st_mtime)
    print(f"Latest zip: {latest} (modified {mtime})")

    if args.dry_run:
        print(f"Dry run: would extract to {dest}")
        return 0

    try:
        extract_zip(latest, dest, force=args.force)
    except FileExistsError as e:
        print(str(e))
        return 3
    except zipfile.BadZipFile:
        print(f"Bad zip file: {latest}")
        return 4

    print(f"Extracted {latest.name} -> {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
