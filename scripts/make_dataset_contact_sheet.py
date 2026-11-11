"""Create contact-sheet previews for selected-crop images and YOLO labels."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("datasets/crop_non_crop"))
    parser.add_argument("--source", type=Path, default=Path("datasets/crop_non_crop/crops_data"))
    parser.add_argument("--output", type=Path, default=Path("runs/dataset_review"))
    parser.add_argument("--thumb-width", type=int, default=240)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for split in ("all", "train", "val"):
        images = image_paths(args.source if split == "all" else args.dataset / "images" / split)
        if not images:
            continue
        previews = [
            make_preview(path, find_label(args.dataset, path), args.thumb_width)
            for path in images
        ]
        sheet = tile(previews, columns=5)
        output_path = args.output / f"{split}_contact_sheet.jpg"
        cv2.imwrite(str(output_path), sheet)
        print(output_path)
    return 0


def image_paths(path: Path) -> list[Path]:
    suffixes = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    return sorted(item for item in path.glob("*") if item.suffix.lower() in suffixes)


def find_label(dataset: Path, image: Path) -> Path | None:
    for split in ("train", "val", "test"):
        label = dataset / "labels" / split / f"{image.stem}.txt"
        if label.exists():
            return label
    return None


def make_preview(image_path: Path, label_path: Path | None, thumb_width: int) -> np.ndarray:
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Could not read {image_path}")
    overlay = image.copy()
    height, width = image.shape[:2]
    if label_path:
        for points in read_polygons(label_path, width, height):
            cv2.polylines(overlay, [points], True, (0, 0, 255), 2)
            cv2.fillPoly(overlay, [points], (0, 90, 255))
        image = cv2.addWeighted(overlay, 0.42, image, 0.58, 0)

    scale = thumb_width / width
    thumb_height = max(1, round(height * scale))
    image = cv2.resize(image, (thumb_width, thumb_height), interpolation=cv2.INTER_AREA)
    label = image_path.stem
    cv2.rectangle(image, (0, 0), (thumb_width, 24), (0, 0, 0), -1)
    cv2.putText(
        image,
        label,
        (5, 17),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return image


def read_polygons(label_path: Path, width: int, height: int) -> list[np.ndarray]:
    polygons: list[np.ndarray] = []
    for row in label_path.read_text(encoding="utf-8").splitlines():
        values = row.split()
        if len(values) < 7:
            continue
        coords = [float(value) for value in values[1:]]
        points = []
        for x, y in zip(coords[0::2], coords[1::2]):
            points.append([round(x * width), round(y * height)])
        polygons.append(np.array(points, dtype=np.int32))
    return polygons


def tile(images: list[np.ndarray], columns: int) -> np.ndarray:
    max_width = max(image.shape[1] for image in images)
    max_height = max(image.shape[0] for image in images)
    rows = math.ceil(len(images) / columns)
    canvas = np.full((rows * max_height, columns * max_width, 3), 245, dtype=np.uint8)
    for index, image in enumerate(images):
        row = index // columns
        column = index % columns
        y = row * max_height
        x = column * max_width
        canvas[y : y + image.shape[0], x : x + image.shape[1]] = image
    return canvas


if __name__ == "__main__":
    raise SystemExit(main())

