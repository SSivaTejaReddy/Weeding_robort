"""Strip Roboflow-style random suffixes from image filenames and rename matching label files.

Examples:
  python -m scripts.strip_roboflow_suffixes --paths datasets/crop_non_crop/current_dataset/train/images datasets/crop_non_crop/current_dataset/valid/images

This script will:
 - Rename files like `onion_mov-0089_jpg.rf.be72ff15...jpg` -> `onion_mov-0089.jpg`
 - If multiple files map to the same target name, append `_1`, `_2` to avoid collisions.
 - Rename corresponding label files in sibling `labels` directories when present.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import shutil
from typing import Dict


def canonical_base(name: str) -> str:
    # Remove the _rf.* part and the preceding image format token like _jpg
    # Example: onion_mov-0089_jpg.rf.be72...jpg -> onion_mov-0089
    # Work on stem (without extension)
    stem = name
    # Find _rf. occurrence
    idx = stem.find("_rf.")
    if idx != -1:
        base = stem[:idx]
    else:
        # fallback: remove trailing tokens like _jpg or _png if present
        base = re.sub(r'_(?:jpg|jpeg|png|bmp|webp)$', '', stem, flags=re.IGNORECASE)
    # Also strip trailing format markers like _jpg if still present
    base = re.sub(r'_(?:jpg|jpeg|png|bmp|webp)$', '', base, flags=re.IGNORECASE)
    return base


def unique_target(target_dir: Path, base: str, ext: str, used: Dict[str, int]) -> Path:
    candidate = f"{base}{ext}"
    i = used.get(candidate, 0)
    while (target_dir / candidate).exists() or i > 0:
        i += 1
        candidate = f"{base}_{i}{ext}"
    used[candidate] = i
    return target_dir / candidate


def rename_in_pair(images_dir: Path, labels_dir: Path | None) -> None:
    images = sorted(p for p in images_dir.iterdir() if p.is_file())
    used: Dict[str, int] = {}
    for img in images:
        stem = img.stem
        ext = img.suffix
        base = canonical_base(stem)
        target_path = unique_target(images_dir, base, ext, used)
        if target_path.name == img.name:
            continue
        # perform move
        img.rename(target_path)
        # rename label if exists
        if labels_dir:
            # original label name assumed to be img.stem + .txt
            old_label = labels_dir / f"{stem}.txt"
            if old_label.exists():
                new_label = labels_dir / f"{target_path.stem}.txt"
                old_label.rename(new_label)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--paths", nargs="+", help="Image directories to process")
    args = p.parse_args()

    if not args.paths:
        print("No paths provided. Use --paths to list directories to normalize.")
        return 2

    for path_str in args.paths:
        images_dir = Path(path_str)
        if not images_dir.exists() or not images_dir.is_dir():
            print(f"Skipping missing directory: {images_dir}")
            continue
        # find sibling labels dir if present
        labels_dir = None
        if images_dir.parent.name == "images":
            maybe_labels = images_dir.parent.parent / "labels" / images_dir.name
            if maybe_labels.exists():
                labels_dir = maybe_labels
        print(f"Processing images in {images_dir} (labels: {labels_dir})")
        rename_in_pair(images_dir, labels_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
