"""Build a small scenario video from real dataset images, for integration tests.

The clip deliberately exercises:

* a single object;
* several objects at once, one per zone;
* an object that disappears and later reappears;
* an object that slides left -> centre -> right, producing direction changes.

Objects are cropped from real test-set images (nothing in the dataset is
modified) and pasted into separate horizontal lanes so they never cover each
other. The chair slides across the frame; the book and phone stay in place.

Run::

    python checks/make_test_video.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT.parent / "model_training" / "dataset"
OUT = ROOT / "checks" / "scenario.mp4"

import cv2
import numpy as np

W, H, FPS = 1280, 720, 30
LANE_Y = {"chair": 20, "book": 250, "phone": 480}
SLOT_X = {"left": 40, "center": 470, "right": 900}
MAX_W, MAX_H = 340, 210


def find_object(class_id: int, limit: int = 400) -> np.ndarray:
    """Crop the largest instance of ``class_id`` from the test split."""
    best: tuple[float, np.ndarray, tuple[float, float, float, float]] | None = None
    labels = DATASET / "labels" / "test"
    for label_path in sorted(labels.glob("*.txt"))[:limit]:
        image_path = DATASET / "images" / "test" / f"{label_path.stem}.jpg"
        if not image_path.is_file():
            continue
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        for line in label_path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) != 5 or int(float(parts[0])) != class_id:
                continue
            cx, cy, bw, bh = (float(p) for p in parts[1:])
            area = bw * bh
            if best is None or area > best[0]:
                best = (area, image, (cx, cy, bw, bh))
    if best is None:
        raise SystemExit(f"no test image with class {class_id}")

    _, image, (cx, cy, bw, bh) = best
    height, width = image.shape[:2]
    x1 = int((cx - bw / 2) * width)
    x2 = int((cx + bw / 2) * width)
    y1 = int((cy - bh / 2) * height)
    y2 = int((cy + bh / 2) * height)
    pad_x = max(4, int((x2 - x1) * 0.18))
    pad_y = max(4, int((y2 - y1) * 0.18))
    crop = image[
        max(0, y1 - pad_y) : min(height, y2 + pad_y),
        max(0, x1 - pad_x) : min(width, x2 + pad_x),
    ]
    if crop.size == 0:
        raise SystemExit(f"empty crop for class {class_id}")
    return crop


def fit_tile(tile: np.ndarray) -> np.ndarray:
    """Scale a crop down so it fits one lane without distortion."""
    scale = min(MAX_W / tile.shape[1], MAX_H / tile.shape[0])
    size = (max(1, int(tile.shape[1] * scale)), max(1, int(tile.shape[0] * scale)))
    return cv2.resize(tile, size, interpolation=cv2.INTER_AREA)


def paste(canvas: np.ndarray, tile: np.ndarray, x: int, y: int) -> None:
    x = max(0, min(canvas.shape[1] - tile.shape[1], int(x)))
    y = max(0, min(canvas.shape[0] - tile.shape[0], int(y)))
    canvas[y : y + tile.shape[0], x : x + tile.shape[1]] = tile


def main() -> int:
    tiles = {
        "chair": fit_tile(find_object(1)),
        "book": fit_tile(find_object(21)),
        "phone": fit_tile(find_object(32)),
    }
    print("tiles:", {k: v.shape[:2] for k, v in tiles.items()})

    writer = cv2.VideoWriter(str(OUT), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    if not writer.isOpened():
        raise SystemExit("could not open the video writer")

    # (frames, chair x, book x, phone x) - None means "not visible"
    script: list[tuple[int, int | None, int | None, int | None]] = []

    # A: one object, left zone.
    script += [(30, SLOT_X["left"], None, None)]
    # B: three objects, one per zone.
    script += [(30, SLOT_X["left"], SLOT_X["center"], SLOT_X["right"])]
    # C: the phone disappears.
    script += [(30, SLOT_X["left"], SLOT_X["center"], None)]
    # D: the chair slides from the left zone into the centre zone.
    script += [
        (
            1,
            int(round(SLOT_X["left"] + (SLOT_X["center"] - SLOT_X["left"]) * i / 29)),
            SLOT_X["center"],
            SLOT_X["right"],
        )
        for i in range(30)
    ]
    # E: the chair slides from the centre zone into the right zone.
    script += [
        (
            1,
            int(round(SLOT_X["center"] + (SLOT_X["right"] - SLOT_X["center"]) * i / 29)),
            SLOT_X["center"],
            SLOT_X["right"],
        )
        for i in range(30)
    ]

    frame_no = 0
    for count, chair_x, book_x, phone_x in script:
        for _ in range(count):
            canvas = np.full((H, W, 3), 118, dtype=np.uint8)
            if chair_x is not None:
                paste(canvas, tiles["chair"], chair_x, LANE_Y["chair"])
            if book_x is not None:
                paste(canvas, tiles["book"], book_x, LANE_Y["book"])
            if phone_x is not None:
                paste(canvas, tiles["phone"], phone_x, LANE_Y["phone"])
            cv2.putText(
                canvas, f"frame {frame_no}", (20, 44),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (40, 40, 40), 2, cv2.LINE_AA,
            )
            writer.write(canvas)
            frame_no += 1

    writer.release()
    print(f"wrote {OUT} ({frame_no} frames, {frame_no / FPS:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



# cd Assistive-Navigation
# conda run --no-capture-output -n yolo python src/main.py --camera droidcam