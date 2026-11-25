"""Tests for the approximate, box-height distance estimate."""

import math
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from direction import Box  # noqa: E402
from distance import (  # noqa: E402
    FALLBACK_HEIGHTS_M,
    DistanceEstimator,
    spoken_distance,
)


class EstimateTests(unittest.TestCase):
    def test_known_height_matches_pinhole_model(self) -> None:
        estimator = DistanceEstimator(heights={1: 1.0}, vertical_fov_deg=60.0)
        frame_height = 720
        focal = frame_height / (2 * math.tan(math.radians(30)))
        # A 1 m object filling `focal` pixels is exactly 1 m away in this model.
        self.assertAlmostEqual(
            estimator.estimate(1, Box(0, 0, 10, focal), frame_height), 1.0, delta=0.1
        )

    def test_larger_box_is_closer(self) -> None:
        estimator = DistanceEstimator(heights={1: 1.0})
        near = estimator.estimate(1, Box(0, 0, 10, 400), 720)
        far = estimator.estimate(1, Box(0, 0, 10, 100), 720)
        self.assertIsNotNone(near)
        self.assertIsNotNone(far)
        self.assertLess(near, far)

    def test_name_key_overrides_id(self) -> None:
        estimator = DistanceEstimator(heights={1: 1.0, "chair": 2.0})
        self.assertAlmostEqual(
            estimator.estimate(1, Box(0, 0, 10, 100), 720, "chair"),
            2 * estimator.focal_px(720) / 100,
            delta=0.1,
        )

    def test_unknown_class_uses_default_height(self) -> None:
        estimator = DistanceEstimator(heights={}, default_height_m=2.0)
        self.assertAlmostEqual(
            estimator.estimate(99, Box(0, 0, 10, 100), 720),
            2 * estimator.focal_px(720) / 100,
            delta=0.1,
        )

    def test_disabled_class_has_no_estimate(self) -> None:
        estimator = DistanceEstimator(heights=FALLBACK_HEIGHTS_M)
        self.assertIsNone(estimator.estimate(26, Box(0, 0, 10, 100), 720))

    def test_invalid_geometry_has_no_estimate(self) -> None:
        estimator = DistanceEstimator(heights={1: 1.0})
        self.assertIsNone(estimator.estimate(1, Box(0, 0, 10, 0), 720))
        self.assertIsNone(estimator.estimate(1, Box(0, 0, 10, 100), 0))

    def test_extremes_are_clamped(self) -> None:
        estimator = DistanceEstimator(heights={1: 1.0})
        self.assertEqual(estimator.estimate(1, Box(0, 0, 10, 100000), 720), 0.3)
        self.assertEqual(estimator.estimate(1, Box(0, 0, 10, 0.001), 720), 30.0)

    def test_load_reads_json_and_ignores_comments(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "heights.json"
            path.write_text(
                '{"_comment": "ignored", "1": 2.0, "chair": 1.5, "bench": null}',
                encoding="utf-8",
            )
            estimator = DistanceEstimator.load(path)
        self.assertEqual(estimator.height_for(1), 2.0)
        self.assertEqual(estimator.height_for(0, "chair"), 1.5)
        self.assertIsNone(estimator.height_for(7, "bench"))

    def test_missing_file_falls_back_to_builtins(self) -> None:
        estimator = DistanceEstimator.load(Path("does-not-exist.json"))
        self.assertEqual(estimator.height_for(0), 1.70)


class SpokenDistanceTests(unittest.TestCase):
    def test_phrases(self) -> None:
        self.assertEqual(spoken_distance(None), "")
        self.assertEqual(spoken_distance(float("nan")), "")
        self.assertEqual(spoken_distance(0.3), "less than half a metre")
        self.assertEqual(spoken_distance(0.8), "about 1 metre")
        self.assertEqual(spoken_distance(2.4), "about 2.5 metres")
        self.assertEqual(spoken_distance(3.0), "about 3 metres")


if __name__ == "__main__":
    unittest.main()
