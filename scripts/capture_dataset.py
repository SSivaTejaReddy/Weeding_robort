"""Capture field images and quality metadata for annotation."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path
import time

import cv2

from ai_validation.vision_quality import measure_frame_quality


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("datasets/raw"))
    parser.add_argument("--source", default="0", help="Camera index or video path.")
    parser.add_argument("--crop", choices=("onion", "groundnut"), required=True)
    parser.add_argument(
        "--condition",
        choices=("sunlight", "shadow", "mixed", "indoor"),
        default="mixed",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0,
        help="Optional seconds between automatic captures.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    image_dir = args.output / args.crop / args.condition
    image_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = args.output / "captures.csv"
    source = int(args.source) if args.source.isdigit() else args.source
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise SystemExit(f"Could not open capture source: {args.source}")

    last_capture_at = 0.0
    print("Press SPACE to save a frame, q to quit.")
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            quality = measure_frame_quality(frame)
            preview = frame.copy()
            message = (
                f"{quality.lighting_tag} brightness={quality.brightness:.0f} "
                f"blur={quality.blur_score:.0f}"
            )
            cv2.putText(
                preview,
                message,
                (14, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.72,
                (40, 245, 40),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("AI dataset capture", preview)

            now = time.monotonic()
            key = cv2.waitKey(1) & 0xFF
            automatic = args.interval > 0 and now - last_capture_at >= args.interval
            if key == ord("q"):
                break
            if key == ord(" ") or automatic:
                filename = save_frame(image_dir, frame)
                append_metadata(metadata_path, filename, args.crop, args.condition, quality)
                last_capture_at = now
                print(f"saved {filename}")
    finally:
        capture.release()
        cv2.destroyAllWindows()
    return 0


def save_frame(image_dir: Path, frame) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    filename = image_dir / f"{stamp}.jpg"
    if not cv2.imwrite(str(filename), frame):
        raise RuntimeError(f"Could not save {filename}")
    return filename


def append_metadata(metadata_path: Path, filename: Path, crop: str, condition: str, quality) -> None:
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not metadata_path.exists()
    with metadata_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if is_new:
            writer.writerow(
                [
                    "image",
                    "crop",
                    "capture_condition",
                    "lighting_tag",
                    "brightness",
                    "shadow_ratio",
                    "highlight_ratio",
                    "blur_score",
                ]
            )
        writer.writerow(
            [
                filename.as_posix(),
                crop,
                condition,
                quality.lighting_tag,
                f"{quality.brightness:.3f}",
                f"{quality.shadow_ratio:.5f}",
                f"{quality.highlight_ratio:.5f}",
                f"{quality.blur_score:.3f}",
            ]
        )


if __name__ == "__main__":
    raise SystemExit(main())

