"""Approximate distance from a bounding box's height.

This is a *rough, monocular estimate*, not a measurement. It assumes a typical
real-world height for each object class and a camera vertical field of view;
the box height in pixels is then converted with a pinhole-camera model:

    distance = (real height * focal length in pixels) / box height in pixels

Distance is spoken with "about" so it is never mistaken for a measured range.
It must not be used for navigation, collision avoidance or any safety decision.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HEIGHTS_PATH = PROJECT_ROOT / "config" / "object_heights.json"

DEFAULT_VERTICAL_FOV_DEG = 60.0
DEFAULT_HEIGHT_M = 0.8
MIN_DISTANCE_M = 0.3
MAX_DISTANCE_M = 30.0

# Fallbacks used when config/object_heights.json is missing or incomplete.
# ``None`` means the class has no useful vertical size (flat signs, ground
# markings), so no distance should be spoken for it.
FALLBACK_HEIGHTS_M: dict[int, float | None] = {
    0: 1.70,    # person
    1: 0.90,    # chair
    2: 0.15,    # toothbrush
    3: 0.20,    # knife
    4: 0.25,    # bottle
    5: 0.10,    # cup
    6: 0.15,    # spoon
    7: 0.45,    # bench
    8: 1.80,    # refrigerator
    9: 0.18,    # fork
    10: 3.20,   # bus
    11: 0.75,   # toilet
    12: 1.10,   # bicycle
    13: 4.00,   # airplane
    14: 3.00,   # truck
    15: 1.20,   # motorcycle
    16: 0.90,   # oven
    17: 0.50,   # dog
    18: 0.55,   # bed
    19: 0.30,   # cat
    20: 3.00,   # traffic light
    21: 0.24,   # book
    22: 0.65,   # person (upper body)
    23: 2.20,   # stop sign
    24: 1.50,   # car
    25: 1.00,   # barrier
    26: None,   # manhole cover (flat on the ground)
    27: None,   # stairs (box height is not an object height)
    28: 3.50,   # train
    29: 0.90,   # trash can
    30: 0.90,   # walking stick
    31: None,   # men's restroom sign (flat)
    32: 0.15,   # mobile phone
    33: None,   # women's restroom sign (flat)
    34: 0.80,   # sink
}


def spoken_distance(metres: float | None) -> str:
    """Natural phrase for an estimate, e.g. ``"about 2 metres"``."""
    if metres is None or not math.isfinite(float(metres)):
        return ""
    distance = max(0.0, float(metres))
    if distance < 0.5:
        return "less than half a metre"
    rounded = max(0.5, round(distance * 2) / 2)  # half-metre steps are stable
    noun = "metre" if rounded == 1 else "metres"
    return f"about {rounded:g} {noun}"


@dataclass(frozen=True)
class DistanceEstimator:
    """Convert box heights into approximate distances in metres.

    Parameters
    ----------
    heights:
        Typical real-world heights, keyed by numeric class ID or by display
        name. ``None`` disables the estimate for that class.
    vertical_fov_deg:
        Vertical field of view used to derive the focal length. A typical phone
        webcam is around 60 degrees; measure it per camera for better estimates.
    """

    heights: Mapping[int | str, float | None]
    vertical_fov_deg: float = DEFAULT_VERTICAL_FOV_DEG
    default_height_m: float = DEFAULT_HEIGHT_M

    @classmethod
    def load(
        cls,
        path: str | Path = DEFAULT_HEIGHTS_PATH,
        vertical_fov_deg: float = DEFAULT_VERTICAL_FOV_DEG,
    ) -> "DistanceEstimator":
        """Load the height table, falling back to built-in values on error."""
        heights: dict[int | str, float | None] = dict(FALLBACK_HEIGHTS_M)
        config_path = Path(path)
        if config_path.is_file():
            try:
                raw = json.loads(config_path.read_text(encoding="utf-8-sig"))
                for key, value in raw.items():
                    if str(key).startswith("_"):
                        continue
                    try:
                        parsed_key: int | str = int(key)
                    except (TypeError, ValueError):
                        parsed_key = str(key).strip()
                    if value is None:
                        heights[parsed_key] = None
                    elif isinstance(value, (int, float)) and not isinstance(value, bool):
                        if float(value) > 0:
                            heights[parsed_key] = float(value)
            except (OSError, ValueError) as exc:
                print(f"[distance] could not read {config_path}: {exc}; using defaults", flush=True)
        else:
            print(f"[distance] {config_path} not found; using built-in heights", flush=True)
        return cls(heights=heights, vertical_fov_deg=vertical_fov_deg)

    def height_for(self, class_id: int, name: str | None = None) -> float | None:
        """Return the assumed real height for a class, if one is usable.

        A display-name key overrides a numeric ID (useful when tuning one
        object without changing the ID table). An explicit ``None`` disables
        the estimate; only a *missing* key uses ``default_height_m``.
        """
        missing = object()
        value = self.heights.get(int(class_id), missing)
        if name and str(name) in self.heights:
            value = self.heights[str(name)]
        if value is missing:
            return self.default_height_m
        if value is None:
            return None
        return float(value) if value > 0 else None

    def focal_px(self, frame_height: int) -> float:
        """Focal length in pixels for the configured vertical field of view."""
        if frame_height <= 0:
            raise ValueError("frame_height must be positive")
        fov = max(1.0, min(179.0, float(self.vertical_fov_deg)))
        return frame_height / (2.0 * math.tan(math.radians(fov) / 2.0))

    def estimate(
        self,
        class_id: int,
        box,
        frame_height: int,
        name: str | None = None,
    ) -> float | None:
        """Approximate distance in metres, or ``None`` when unavailable.

        ``box`` only needs a ``height`` property in frame pixels.
        """
        height_m = self.height_for(class_id, name)
        if height_m is None:
            return None
        box_height = float(getattr(box, "height", 0.0))
        if box_height <= 0 or frame_height <= 0:
            return None
        distance = height_m * self.focal_px(frame_height) / box_height
        return round(min(MAX_DISTANCE_M, max(MIN_DISTANCE_M, distance)), 1)
