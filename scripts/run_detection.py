"""Run a selected-crop model and show crop-complement validation overlays."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2

from ai_validation.targeting import (
    Detection,
    TargetingConfig,
    assign_crop_complement_targets,
)
from ai_validation.vision_quality import measure_frame_quality


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--source", default="0", help="Camera index, video, or image path.")
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Crop protection confidence threshold. Lower values reduce false negatives.",
    )
    parser.add_argument("--zones", type=int, default=4)
    parser.add_argument("--roi-top", type=float, default=0.35)
    parser.add_argument(
        "--guard-px",
        type=float,
        default=32.0,
        help="Pixel margin around crop detections that blocks extraction points.",
    )
    parser.add_argument("--save-dir", type=Path, help="Write annotated frames here.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    set_workspace_ultralytics_config()
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Install requirements-ai.txt before inference.") from exc

    model = YOLO(str(args.model))
    source = int(args.source) if args.source.isdigit() else args.source
    if is_image_source(args.source):
        frame = cv2.imread(args.source)
        if frame is None:
            raise SystemExit(f"Could not read image: {args.source}")
        process_frame(model, frame, args, frame_index=0)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return 0

    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise SystemExit(f"Could not open source: {args.source}")

    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            process_frame(model, frame, args, frame_index)
            frame_index += 1
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()
    return 0


def process_frame(model, frame, args: argparse.Namespace, frame_index: int) -> None:
    result = model(frame, conf=args.conf, verbose=False)[0]
    annotated = result.plot()
    detections = result_to_detections(result)
    config = TargetingConfig(
        zone_count=args.zones,
        crop_min_confidence=args.conf,
        working_roi_top=args.roi_top,
        crop_guard_px=args.guard_px,
    )
    targets = assign_crop_complement_targets(
        detections,
        frame_width=frame.shape[1],
        frame_height=frame.shape[0],
        config=config,
    )
    quality = measure_frame_quality(frame)
    draw_zones(annotated, config)
    for target in targets:
        point = (round(target.center_x), round(target.center_y))
        cv2.drawMarker(
            annotated,
            point,
            (0, 0, 255),
            cv2.MARKER_CROSS,
            22,
            3,
        )
        cv2.putText(
            annotated,
            f"~crop Z{target.zone_index + 1}",
            (point[0] + 8, max(point[1] - 8, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    status = (
        f"{quality.lighting_tag} bright={quality.brightness:.0f} "
        f"shadow={quality.shadow_ratio:.2f} blur={quality.blur_score:.0f}"
    )
    cv2.putText(
        annotated,
        status,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.64,
        (20, 245, 245),
        2,
        cv2.LINE_AA,
    )
    print(
        json.dumps(
            {
                "frame": frame_index,
                "lighting": quality.lighting_tag,
                "zones": [target.zone_index + 1 for target in targets],
                "targets": [
                    {
                        "label": target.detection.label,
                        "confidence": round(target.detection.confidence, 4),
                        "x": round(target.center_x, 1),
                        "y": round(target.center_y, 1),
                    }
                    for target in targets
                ],
            }
        )
    )
    cv2.imshow("AI selected crop validation", annotated)
    if args.save_dir:
        args.save_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.save_dir / f"frame_{frame_index:06d}.jpg"), annotated)


def result_to_detections(result) -> list[Detection]:
    names = result.names
    output: list[Detection] = []
    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        class_id = int(box.cls[0].item())
        output.append(
            Detection(
                label=str(names[class_id]),
                confidence=float(box.conf[0].item()),
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
            )
        )
    return output


def draw_zones(frame, config: TargetingConfig) -> None:
    height, width = frame.shape[:2]
    roi_y = round(height * config.working_roi_top)
    cv2.line(frame, (0, roi_y), (width, roi_y), (255, 220, 40), 2)
    for index in range(1, config.zone_count):
        x = round(width * index / config.zone_count)
        cv2.line(frame, (x, roi_y), (x, height), (255, 220, 40), 1)


def is_image_source(source: str) -> bool:
    return Path(source).suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def set_workspace_ultralytics_config() -> None:
    os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(".ultralytics").resolve()))
    os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib").resolve()))


if __name__ == "__main__":
    raise SystemExit(main())
