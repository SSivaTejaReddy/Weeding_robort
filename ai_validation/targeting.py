"""Map crop detections and crop-complement points to extraction modules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class Detection:
    """A pixel-space bounding box emitted by the vision model."""

    label: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)


@dataclass(frozen=True)
class TargetingConfig:
    """Geometry and safety policy for the camera-to-extractor handoff."""

    zone_count: int = 4
    min_confidence: float = 0.55
    crop_min_confidence: float = 0.25
    working_roi_top: float = 0.35
    crop_guard_px: float = 32.0
    complement_target_y: float = 0.78
    extraction_labels: tuple[str, ...] = ("not_crop",)
    crop_labels: tuple[str, ...] = ("crop", "onion")


@dataclass(frozen=True)
class ZoneTarget:
    """A model candidate that is safe enough to queue for a module."""

    zone_index: int
    detection: Detection
    center_x: float
    center_y: float


def assign_extraction_targets(
    detections: Iterable[Detection],
    frame_width: int,
    frame_height: int,
    config: TargetingConfig | None = None,
) -> list[ZoneTarget]:
    """Return one highest-confidence candidate per extraction zone.

    Targets are limited to the lower camera region where the implement can
    physically act and are suppressed near crop detections.
    """

    policy = config or TargetingConfig()
    _validate_frame_and_policy(frame_width, frame_height, policy)

    items = list(detections)
    crop_boxes = [
        item
        for item in items
        if item.label in policy.crop_labels
        and item.confidence >= policy.crop_min_confidence
    ]
    candidates: dict[int, ZoneTarget] = {}
    zone_width = frame_width / policy.zone_count
    roi_top_px = frame_height * policy.working_roi_top

    for item in items:
        center_x, center_y = item.center
        if item.label not in policy.extraction_labels:
            continue
        if item.confidence < policy.min_confidence:
            continue
        if center_y < roi_top_px:
            continue
        if not (0 <= center_x < frame_width and 0 <= center_y < frame_height):
            continue
        if _near_crop(center_x, center_y, crop_boxes, policy.crop_guard_px):
            continue

        zone_index = min(int(center_x / zone_width), policy.zone_count - 1)
        target = ZoneTarget(zone_index, item, center_x, center_y)
        current = candidates.get(zone_index)
        if current is None or item.confidence > current.detection.confidence:
            candidates[zone_index] = target

    return [candidates[index] for index in sorted(candidates)]


def assign_crop_complement_targets(
    detections: Iterable[Detection],
    frame_width: int,
    frame_height: int,
    config: TargetingConfig | None = None,
) -> list[ZoneTarget]:
    """Expose one `not_crop` working point per zone outside crop detections.

    A crop-only model cannot name every possible non-crop object. This helper
    turns the inverse decision into explicit validation points in the implement
    working region: target points outside selected-crop guard regions are
    reported as `not_crop`.
    """

    policy = config or TargetingConfig()
    _validate_frame_and_policy(frame_width, frame_height, policy)
    if not policy.working_roi_top <= policy.complement_target_y < 1:
        raise ValueError("complement_target_y must be inside the working ROI")

    crop_boxes = [
        item
        for item in detections
        if item.label in policy.crop_labels
        and item.confidence >= policy.crop_min_confidence
    ]
    zone_width = frame_width / policy.zone_count
    target_y = frame_height * policy.complement_target_y
    targets: list[ZoneTarget] = []

    for zone_index in range(policy.zone_count):
        target_x = (zone_index + 0.5) * zone_width
        if _near_crop(target_x, target_y, crop_boxes, policy.crop_guard_px):
            continue
        detection = Detection(
            label="not_crop",
            confidence=1.0,
            x1=target_x,
            y1=target_y,
            x2=target_x,
            y2=target_y,
        )
        targets.append(ZoneTarget(zone_index, detection, target_x, target_y))

    return targets


def _validate_frame_and_policy(
    frame_width: int,
    frame_height: int,
    policy: TargetingConfig,
) -> None:
    if frame_width <= 0 or frame_height <= 0:
        raise ValueError("frame dimensions must be positive")
    if policy.zone_count <= 0:
        raise ValueError("zone_count must be positive")
    if not 0 <= policy.working_roi_top < 1:
        raise ValueError("working_roi_top must be in [0, 1)")


def _near_crop(
    x: float,
    y: float,
    crops: Iterable[Detection],
    guard_px: float,
) -> bool:
    for crop in crops:
        if (
            crop.x1 - guard_px <= x <= crop.x2 + guard_px
            and crop.y1 - guard_px <= y <= crop.y2 + guard_px
        ):
            return True
    return False
