"""Camera-relative direction estimation.

The frame is split into three vertical zones (left / centre / right) and the
horizontal centre of a bounding box decides the zone.

Important: this is a *2-D framing* estimate only. "Ahead" means "in the middle
third of the picture", not "close" and not "in the path". Nothing here measures
distance or collision risk.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Direction(str, Enum):
    """Where an object sits inside the camera frame."""

    LEFT = "left"
    CENTER = "center"
    RIGHT = "right"

    @property
    def spoken(self) -> str:
        """Phrase used in a spoken sentence."""
        return {
            Direction.LEFT: "on your left",
            Direction.CENTER: "ahead",
            Direction.RIGHT: "on your right",
        }[self]

    @property
    def short(self) -> str:
        """Compact label for the on-screen overlay."""
        return {
            Direction.LEFT: "LEFT",
            Direction.CENTER: "AHEAD",
            Direction.RIGHT: "RIGHT",
        }[self]

    @property
    def arrow(self) -> str:
        """ASCII arrow, kept ASCII so OpenCV can draw it with the default font."""
        return {
            Direction.LEFT: "<--",
            Direction.CENTER: "^",
            Direction.RIGHT: "-->",
        }[self]


@dataclass(frozen=True)
class Box:
    """Axis-aligned box in frame pixels."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def center_x(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def center_y(self) -> float:
        return (self.y1 + self.y2) / 2.0

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)


class DirectionEstimator:
    """Assign LEFT / CENTER / RIGHT from the horizontal centre of a box.

    ``left_ratio`` and ``right_ratio`` are fractions of the frame width, so the
    middle zone defaults to the middle third. Widen the middle zone if the
    camera is mounted off-centre.
    """

    def __init__(self, left_ratio: float = 1.0 / 3.0, right_ratio: float = 2.0 / 3.0) -> None:
        if not 0.0 < left_ratio < right_ratio < 1.0:
            raise ValueError(
                f"need 0 < left_ratio < right_ratio < 1, got {left_ratio}, {right_ratio}"
            )
        self.left_ratio = float(left_ratio)
        self.right_ratio = float(right_ratio)

    def boundaries(self, frame_width: int) -> tuple[float, float]:
        """Pixel x positions of the two zone edges."""
        return self.left_ratio * frame_width, self.right_ratio * frame_width

    def estimate(self, box: Box, frame_width: int) -> Direction:
        """Zone of the box centre."""
        if frame_width <= 0:
            raise ValueError("frame_width must be positive")
        cx = box.center_x
        left_edge, right_edge = self.boundaries(frame_width)
        if cx < left_edge:
            return Direction.LEFT
        if cx > right_edge:
            return Direction.RIGHT
        return Direction.CENTER

    def estimate_xyxy(
        self, xyxy: tuple[float, float, float, float], frame_width: int
    ) -> Direction:
        return self.estimate(Box(*xyxy), frame_width)

    @staticmethod
    def normalized_offset(box: Box, frame_width: int) -> float:
        """Horizontal offset of the box centre: -1.0 (far left) .. +1.0 (far right).

        Kept for later work (audio panning, distance estimation). It is not a
        distance and is not used for any safety decision.
        """
        if frame_width <= 0:
            raise ValueError("frame_width must be positive")
        return max(-1.0, min(1.0, 2.0 * box.center_x / frame_width - 1.0))

    def zone_mask(self, frame_width: int, frame_height: int) -> list[tuple[int, int]]:
        """Pixel x positions of the two dividers, for drawing."""
        left_edge, right_edge = self.boundaries(frame_width)
        return [(int(left_edge), frame_height), (int(right_edge), frame_height)]


class DirectionSmoother:
    """Suppress zone flicker when an object straddles a divider.

    A raw zone change is only accepted after ``stable_frames`` consecutive
    frames agree. Without this, a box sitting on a boundary flips between
    "ahead" and "on your left" every frame and the speaker never shuts up.
    """

    def __init__(self, stable_frames: int = 3) -> None:
        if stable_frames < 1:
            raise ValueError("stable_frames must be >= 1")
        self.stable_frames = int(stable_frames)
        self._stable: dict[int, Direction] = {}
        self._candidate: dict[int, tuple[Direction, int]] = {}

    def update(self, key: int, raw: Direction) -> Direction:
        """Return the smoothed direction for this object key."""
        current = self._stable.get(key)
        if current is None:
            self._stable[key] = raw
            self._candidate[key] = (raw, 0)
            return raw
        if raw == current:
            self._candidate[key] = (raw, 0)
            return current
        candidate, count = self._candidate.get(key, (raw, 0))
        count = count + 1 if candidate == raw else 1
        if count >= self.stable_frames:
            self._stable[key] = raw
            self._candidate[key] = (raw, 0)
            return raw
        self._candidate[key] = (raw, count)
        return current

    def forget(self, key: int) -> None:
        self._stable.pop(key, None)
        self._candidate.pop(key, None)

    def reset(self) -> None:
        self._stable.clear()
        self._candidate.clear()
