"""AI validation helpers for the precision weed extraction rover."""

from .targeting import (
    Detection,
    TargetingConfig,
    ZoneTarget,
    assign_crop_complement_targets,
    assign_extraction_targets,
)

__all__ = [
    "Detection",
    "TargetingConfig",
    "ZoneTarget",
    "assign_crop_complement_targets",
    "assign_extraction_targets",
]
