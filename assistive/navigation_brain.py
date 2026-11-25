#!/usr/bin/env python3
"""
navigation_brain.py
===================

Turn raw YOLO detections into short, spoken navigation guidance.

This is a deterministic demo brain, not an LLM. It deliberately does not guess
distances or motion: for the demo those values come from command-line options
(``--distance`` and ``--motion-mode``). A production system should replace them
with a metric depth sensor and a real object tracker.

Example output::

    Person, 20 metres ahead, moving left.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from time import monotonic


@dataclass(frozen=True)
class Detection:
    """One detection passed to the brain."""

    name: str
    conf: float
    box: tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels
    depth_m: float | None = None
    track_id: int | None = None


class NavigationBrain:
    """Convert detections into a single concise guidance sentence."""

    def __init__(
        self,
        frame_width: int,
        *,
        distance_m: float | None = 20.0,
        motion_mode: str = "auto",
        position_mode: str = "auto",
        min_conf: float = 0.45,
        cooldown_s: float = 3.0,
        priority_names: tuple[str, ...] = ("person", "person (upper body)"),
        left_frac: float = 0.40,
        right_frac: float = 0.60,
    ) -> None:
        self.frame_width = frame_width
        self.distance_m = distance_m
        self.motion_mode = motion_mode
        self.position_mode = position_mode
        self.min_conf = min_conf
        self.cooldown_s = cooldown_s
        self.priority_names = priority_names
        self.left_frac = left_frac
        self.right_frac = right_frac

        self.history: dict[int, deque[tuple[float, float]]] = defaultdict(
            lambda: deque(maxlen=15)
        )
        self.last_text = ""
        self.last_time = 0.0

    def _position(self, cx: float) -> str:
        if self.position_mode in {"ahead", "left", "right"}:
            return self.position_mode
        if cx < self.left_frac * self.frame_width:
            return "left"
        if cx > self.right_frac * self.frame_width:
            return "right"
        return "ahead"

    def _tracked_motion(self, track_id: int | None) -> str | None:
        if track_id is None:
            return None

        points = list(self.history[track_id])
        if len(points) < 4:
            return None

        dx = points[-1][0] - points[0][0]
        if abs(dx) < 12:
            return None
        return "left" if dx < 0 else "right"

    def _motion(self, track_id: int | None) -> str | None:
        if self.motion_mode == "none":
            return None
        if self.motion_mode in {"left", "right"}:
            return self.motion_mode
        return self._tracked_motion(track_id)

    def _pick(self, detections: list[Detection]) -> Detection | None:
        if not detections:
            return None

        # Priority objects first, then whichever is nearest.
        def key(det: Detection) -> tuple[int, float]:
            priority = 0 if det.name.lower() in self.priority_names else 1
            distance = det.depth_m if det.depth_m is not None else float("inf")
            return priority, distance

        return min(detections, key=key)

    def update(
        self,
        detections: list[Detection],
        *,
        now: float | None = None,
    ) -> str | None:
        """Return a sentence to speak, or ``None`` when nothing should be said."""

        now = now if now is not None else monotonic()
        usable = [d for d in detections if d.conf >= self.min_conf]

        for det in usable:
            if det.track_id is not None:
                x1, _y1, x2, _y2 = det.box
                cx = (x1 + x2) / 2.0
                self.history[det.track_id].append((cx, now))

        chosen = self._pick(usable)
        if chosen is None:
            return None

        x1, _y1, x2, _y2 = chosen.box
        cx = (x1 + x2) / 2.0
        position = None if self.position_mode == "none" else self._position(cx)
        motion = self._motion(chosen.track_id)

        distance = self.distance_m if chosen.depth_m is None else chosen.depth_m

        if distance is None:
            text = f"{chosen.name}"
            if position:
                text += f", {position}"
        else:
            text = f"{chosen.name}, {max(1, round(distance))} metres"
            if position:
                text += f" {position}"

        if motion:
            text += f", moving {motion}"
        text += "."

        if self.cooldown_s > 0:
            if now - self.last_time < self.cooldown_s:
                return None
            if text == self.last_text:
                return None

        self.last_text = text
        self.last_time = now
        return text
