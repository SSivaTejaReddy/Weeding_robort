"""Prepare a weakly labeled selected-crop YOLO segmentation dataset."""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

import cv2
import numpy as np


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("datasets/crop_non_crop/crops_data"),
        help="Folder of selected-crop images.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("datasets/crop_non_crop"),
        help="YOLO dataset output root.",
    )
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=26)
    parser.add_argument(
        "--min-area-ratio",
        type=float,
        default=0.0015,
        help="Drop small green regions below this image-area ratio.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 0 < args.val_fraction < 1:
        raise SystemExit("--val-fraction must be between 0 and 1.")

    images = sorted(
        path for path in args.source.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES
    )
    if len(images) < 2:
        raise SystemExit(f"Need at least two source images in {args.source}.")

    split_images = split_paths(images, args.val_fraction, args.seed)
    summary = {"train": 0, "val": 0, "labels": 0, "regions": 0}
    for split, paths in split_images.items():
        image_dir = args.dataset / "images" / split
        label_dir = args.dataset / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        for image_path in paths:
            frame = cv2.imread(str(image_path))
            if frame is None:
                print(f"skip unreadable image: {image_path}")
                continue
            label_rows = green_crop_segments(frame, args.min_area_ratio)
            shutil.copy2(image_path, image_dir / image_path.name)
            (label_dir / f"{image_path.stem}.txt").write_text(
                "\n".join(label_rows),
                encoding="utf-8",
            )
            summary[split] += 1
            summary["labels"] += 1
            summary["regions"] += len(label_rows)

    print(
        "prepared weak selected-crop dataset: "
        f"{summary['train']} train, {summary['val']} val, "
        f"{summary['regions']} green crop-region polygons"
    )
    print("Review generated labels before trusting extraction behavior.")
    return 0


def split_paths(paths: list[Path], val_fraction: float, seed: int) -> dict[str, list[Path]]:
    shuffled = paths.copy()
    random.Random(seed).shuffle(shuffled)
    val_count = max(1, round(len(shuffled) * val_fraction))
    return {"train": shuffled[val_count:], "val": shuffled[:val_count]}


def green_crop_segments(frame: np.ndarray, min_area_ratio: float) -> list[str]:
    """Create safety-biased crop polygons from strong green vegetation pixels."""

    height, width = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, np.array([28, 28, 18]), np.array([105, 255, 255]))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    green = cv2.morphologyEx(green, cv2.MORPH_OPEN, kernel)
    green = cv2.morphologyEx(green, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_area = height * width
    min_area = max(12, image_area * min_area_ratio)
    rows: list[str] = []
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue
        epsilon = max(1.0, 0.004 * cv2.arcLength(contour, True))
        polygon = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
        if len(polygon) < 3:
            continue
        values = ["0"]
        for x, y in polygon:
            values.extend((f"{x / width:.6f}", f"{y / height:.6f}"))
        rows.append(" ".join(values))
    return rows


if __name__ == "__main__":
    raise SystemExit(main())

