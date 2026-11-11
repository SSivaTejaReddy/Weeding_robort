"""Small OpenCV metrics used while collecting and field-testing images."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class FrameQuality:
    brightness: float
    shadow_ratio: float
    highlight_ratio: float
    blur_score: float

    @property
    def lighting_tag(self) -> str:
        if self.highlight_ratio >= 0.18:
            return "harsh_sunlight"
        if self.shadow_ratio >= 0.32:
            return "deep_shadow"
        if self.brightness < 65:
            return "dim"
        return "usable"


def measure_frame_quality(frame: np.ndarray) -> FrameQuality:
    """Estimate useful annotation metadata from a BGR or grayscale frame."""

    if frame.size == 0:
        raise ValueError("frame must not be empty")

    if len(frame.shape) == 2:
        gray = frame
    else:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    return FrameQuality(
        brightness=float(np.mean(gray)),
        shadow_ratio=float(np.mean(gray <= 45)),
        highlight_ratio=float(np.mean(gray >= 235)),
        blur_score=float(cv2.Laplacian(gray, cv2.CV_64F).var()),
    )

