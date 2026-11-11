"""Check image and label coverage before training a YOLO dataset."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import yaml


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("datasets/crop_non_crop"))
    parser.add_argument("--data", type=Path, help="Optional YOLO data.yaml file.")
    parser.add_argument("--classes", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset, splits = load_dataset_layout(args)
    failed = False
    for split, image_dir in splits.items():
        stats, errors = audit_split(dataset, split, image_dir, args.classes)
        print(
            f"{split}: {stats['images']} images, {stats['labels']} label files, "
            f"{stats['empty']} empty label files, classes {dict(stats['classes'])}"
        )
        for error in errors:
            print(f"  error: {error}")
        failed = failed or bool(errors)

    if failed:
        print("Dataset audit failed. Fix label coverage or malformed YOLO rows.")
        return 1
    return 0


def audit_split(dataset: Path, split: str, image_dir: Path, class_count: int):
    label_dir = infer_label_dir(dataset, image_dir)
    images = sorted(
        path for path in image_dir.glob("*") if path.suffix.lower() in IMAGE_SUFFIXES
    )
    stats = {"images": len(images), "labels": 0, "empty": 0, "classes": Counter()}
    errors: list[str] = []
    if split in {"train", "val"} and not images:
        errors.append(f"{image_dir.relative_to(dataset)} has no images")

    for image in images:
        label = label_dir / f"{image.stem}.txt"
        if not label.exists():
            errors.append(f"missing label for {image.relative_to(dataset)}")
            continue
        stats["labels"] += 1
        rows = [row.strip() for row in label.read_text(encoding="utf-8").splitlines()]
        rows = [row for row in rows if row]
        if not rows:
            stats["empty"] += 1
            continue
        for row_number, row in enumerate(rows, start=1):
            values = row.split()
            if len(values) < 5:
                errors.append(f"{label.relative_to(dataset)}:{row_number} has <5 values")
                continue
            class_id = parse_class_id(values[0])
            if class_id is None or class_id < 0 or class_id >= class_count:
                errors.append(
                    f"{label.relative_to(dataset)}:{row_number} has invalid class {values[0]}"
                )
                continue
            stats["classes"][class_id] += 1

    return stats, errors


def load_dataset_layout(args: argparse.Namespace) -> tuple[Path, dict[str, Path]]:
    if not args.data:
        dataset = args.dataset
        return (
            dataset,
            {
                "train": dataset / "images" / "train",
                "val": dataset / "images" / "val",
                "test": dataset / "images" / "test",
            },
        )

    data = yaml.safe_load(args.data.read_text(encoding="utf-8"))
    dataset = Path(data.get("path", args.data.parent))
    if not dataset.is_absolute():
        dataset = (Path.cwd() / dataset).resolve()
    splits = {}
    for split, key in (("train", "train"), ("val", "val"), ("test", "test")):
        value = data.get(key)
        if value:
            splits[split] = (dataset / value).resolve()
        else:
            splits[split] = dataset / "images" / split
    return dataset, splits


def infer_label_dir(dataset: Path, image_dir: Path) -> Path:
    parts = list(image_dir.parts)
    if "images" in parts:
        index = len(parts) - 1 - parts[::-1].index("images")
        parts[index] = "labels"
        return Path(*parts)
    return dataset / "labels" / image_dir.name


def parse_class_id(value: str) -> int | None:
    try:
        parsed = int(value)
    except ValueError:
        return None
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
