"""Tests for name loading, camera streaming and detection helpers."""

import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from detection import (  # noqa: E402
    CameraStream,
    Detection,
    Detector,
    detections_to_rows,
    load_class_names,
    placeholder,
)
from direction import Box, Direction  # noqa: E402

NAMES = ROOT / "config" / "class_names.json"


class ClassNameTests(unittest.TestCase):
    def test_real_config(self) -> None:
        names = load_class_names(NAMES)
        self.assertEqual(len(names), 35)
        self.assertEqual(names[0], "person")
        self.assertEqual(names[1], "chair")
        self.assertEqual(names[22], "person (upper body)")
        self.assertEqual(names[34], "sink")

    def test_missing_file(self) -> None:
        with self.assertRaises(FileNotFoundError):
            load_class_names(ROOT / "config" / "does_not_exist.json")

    def test_empty_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "names.json"
            path.write_text('{"_comment": "nothing here"}', encoding="utf-8")
            with self.assertRaises(ValueError):
                load_class_names(path)

    def test_placeholder(self) -> None:
        self.assertEqual(placeholder(7), "class_07")

    def test_config_matches_training_copy_if_present(self) -> None:
        source = ROOT.parent / "model_training" / "class_names.json"
        if not source.is_file():
            self.skipTest("training copy not present")
        self.assertEqual(load_class_names(source), load_class_names(NAMES))


class DetectionRowTests(unittest.TestCase):
    def test_rows(self) -> None:
        det = Detection(
            class_id=1,
            name="chair",
            confidence=0.87654,
            box=Box(1.234, 2.345, 30.0, 40.0),
            track_id=5,
            direction=Direction.LEFT,
            direction_stable=Direction.CENTER,
        )
        rows = detections_to_rows([det], frame_index=7)
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["frame"], 7)
        self.assertEqual(row["track_id"], 5)
        self.assertEqual(row["name"], "chair")
        self.assertEqual(row["confidence"], 0.8765)
        self.assertEqual(row["direction"], "center")  # smoothed value wins
        self.assertEqual(row["x1"], 1.2)

    def test_display_direction_falls_back_to_raw(self) -> None:
        det = Detection(1, "chair", 0.9, Box(0, 0, 1, 1), direction=Direction.RIGHT)
        self.assertEqual(det.display_direction, Direction.RIGHT)
        self.assertIn("RIGHT", det.label)


class FakeCapture:
    """Minimal stand-in for ``cv2.VideoCapture``."""

    def __init__(self, frames, fail_after: int) -> None:
        self._frames = list(frames)
        self._index = 0
        self._fail_after = fail_after
        self.released = False

    def isOpened(self) -> bool:  # noqa: N802 - OpenCV naming
        return True

    def read(self):
        if self._index >= self._fail_after or self._index >= len(self._frames):
            return False, None
        frame = self._frames[self._index]
        self._index += 1
        return True, frame

    def release(self) -> None:
        self.released = True

    def set(self, *_args) -> bool:
        return True


class CameraStreamTests(unittest.TestCase):
    def test_sequential_video_keeps_every_frame(self) -> None:
        values = [30, 70, 110, 150, 190]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "clip.mp4"
            writer = cv2.VideoWriter(
                str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 64)
            )
            self.assertTrue(writer.isOpened(), "mp4v writer unavailable")
            for value in values:
                writer.write(np.full((64, 64, 3), value, np.uint8))
            writer.release()

            stream = CameraStream(source=str(path), realtime=False)
            stream.start(timeout=5.0)
            try:
                got = []
                while True:
                    frame = stream.read(timeout=1.0)
                    if frame is None:
                        break
                    got.append(int(round(frame.mean())))
            finally:
                stream.stop()

        self.assertEqual(len(got), len(values))
        self.assertEqual(got, sorted(got), "frames must arrive in order")
        self.assertTrue(stream.ended)

    def test_missing_video_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            stream = CameraStream(source=str(Path(tmp) / "nope.mp4"))
            with self.assertRaises(FileNotFoundError):
                stream.start(timeout=1.0)

    def test_camera_reconnect_after_failure(self) -> None:
        frames = [np.full((8, 8, 3), value, np.uint8) for value in (10, 20, 30, 40)]
        captures = [FakeCapture(frames[:2], fail_after=2), FakeCapture(frames[2:], fail_after=2)]

        def open_fn():
            return captures.pop(0) if captures else FakeCapture([], fail_after=0)

        stream = CameraStream(
            source=0,
            realtime=False,
            open_fn=open_fn,
            retry_delay=0.01,
            queue_size=4,
        )
        stream.start(timeout=5.0)
        got = []
        try:
            deadline = time.monotonic() + 10.0
            while len(got) < 4 and time.monotonic() < deadline:
                frame = stream.read(timeout=1.0)
                if frame is not None:
                    got.append(int(frame[0, 0, 0]))
        finally:
            stream.stop()

        self.assertEqual(got, [10, 20, 30, 40])
        self.assertGreaterEqual(stream.reconnects, 1)

    def test_realtime_mode_returns_something(self) -> None:
        frames = [np.full((8, 8, 3), value, np.uint8) for value in (10, 20, 30)]
        stream = CameraStream(
            source=0,
            realtime=True,
            open_fn=lambda: FakeCapture(frames, fail_after=3),
            retry_delay=0.01,
        )
        stream.start(timeout=5.0)
        try:
            self.assertIsNotNone(stream.read(timeout=2.0))
        finally:
            stream.stop()


class DetectorErrorTests(unittest.TestCase):
    def test_missing_weights(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                Detector(weights=Path(tmp) / "best.pt", verbose=False)


if __name__ == "__main__":
    unittest.main()
