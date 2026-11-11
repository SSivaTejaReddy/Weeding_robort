"""Train a lightweight YOLO model for selected-crop protection."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("configs/selected_crop_dataset.yaml"),
    )
    parser.add_argument("--task", choices=("detect", "segment"), default="detect")
    parser.add_argument("--model", help="Ultralytics checkpoint override.")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default=None, help="Examples: cpu, 0, 0,1.")
    parser.add_argument("--name", default=None)
    parser.add_argument(
        "--patience",
        type=int,
        default=30,
        help="Early-stopping patience. Larger values help recall continue improving.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Data loader workers. Zero is safest on Windows.",
    )
    parser.add_argument(
        "--close-mosaic",
        type=int,
        default=10,
        help="Disable mosaic this many final epochs so validation-like crops are learned.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    set_workspace_ultralytics_config()
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Install requirements-ai.txt before training.") from exc

    default_model = "yolo26n-seg.pt" if args.task == "segment" else "yolo26n.pt"
    model = YOLO(args.model or default_model)
    run_name = args.name or (
        "selected_crop_segmentation" if args.task == "segment" else "selected_crop_detection"
    )
    model.train(
        data=str(args.data),
        task=args.task,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        patience=args.patience,
        workers=args.workers,
        close_mosaic=args.close_mosaic,
        project=str(Path("runs/ai_validation").resolve()),
        name=run_name,
    )
    return 0


def set_workspace_ultralytics_config() -> None:
    os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(".ultralytics").resolve()))
    os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib").resolve()))


if __name__ == "__main__":
    raise SystemExit(main())
