"""Tests for camera-relative direction estimation."""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from direction import Box, Direction, DirectionEstimator, DirectionSmoother  # noqa: E402


class DirectionEstimatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.est = DirectionEstimator()

    def test_three_zones(self) -> None:
        width = 640
        self.assertEqual(self.est.estimate(Box(0, 0, 100, 100), width), Direction.LEFT)
        self.assertEqual(self.est.estimate(Box(280, 0, 360, 100), width), Direction.CENTER)
        self.assertEqual(self.est.estimate(Box(540, 0, 640, 100), width), Direction.RIGHT)

    def test_edges_belong_to_center(self) -> None:
        width = 900  # edges at 300 and 600
        self.assertEqual(self.est.estimate(Box(300, 0, 300, 10), width), Direction.CENTER)
        self.assertEqual(self.est.estimate(Box(600, 0, 600, 10), width), Direction.CENTER)
        self.assertEqual(self.est.estimate(Box(299, 0, 299, 10), width), Direction.LEFT)
        self.assertEqual(self.est.estimate(Box(601, 0, 601, 10), width), Direction.RIGHT)

    def test_custom_ratios(self) -> None:
        est = DirectionEstimator(left_ratio=0.25, right_ratio=0.75)
        width = 400  # edges at 100 and 300
        self.assertEqual(est.estimate(Box(0, 0, 180, 10), width), Direction.LEFT)
        self.assertEqual(est.estimate(Box(120, 0, 260, 10), width), Direction.CENTER)
        self.assertEqual(est.estimate(Box(240, 0, 380, 10), width), Direction.RIGHT)

    def test_invalid_ratios(self) -> None:
        with self.assertRaises(ValueError):
            DirectionEstimator(0.6, 0.4)
        with self.assertRaises(ValueError):
            DirectionEstimator(0.0, 0.5)

    def test_zero_width_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.est.estimate(Box(0, 0, 1, 1), 0)

    def test_normalized_offset(self) -> None:
        width = 100
        self.assertAlmostEqual(DirectionEstimator.normalized_offset(Box(0, 0, 0, 10), width), -1.0)
        self.assertAlmostEqual(DirectionEstimator.normalized_offset(Box(50, 0, 50, 10), width), 0.0)
        self.assertAlmostEqual(DirectionEstimator.normalized_offset(Box(100, 0, 100, 10), width), 1.0)

    def test_zone_mask(self) -> None:
        self.assertEqual(self.est.zone_mask(600, 400), [(200, 400), (400, 400)])

    def test_spoken_labels(self) -> None:
        self.assertEqual(Direction.LEFT.spoken, "on your left")
        self.assertEqual(Direction.CENTER.spoken, "ahead")
        self.assertEqual(Direction.RIGHT.spoken, "on your right")
        self.assertEqual(Direction.LEFT.short, "LEFT")
        self.assertEqual(Direction.RIGHT.arrow, "-->")


class DirectionSmootherTests(unittest.TestCase):
    def test_first_value_is_accepted(self) -> None:
        smoother = DirectionSmoother(stable_frames=3)
        self.assertEqual(smoother.update(1, Direction.LEFT), Direction.LEFT)

    def test_change_needs_stable_frames(self) -> None:
        smoother = DirectionSmoother(stable_frames=3)
        smoother.update(1, Direction.CENTER)
        self.assertEqual(smoother.update(1, Direction.LEFT), Direction.CENTER)
        self.assertEqual(smoother.update(1, Direction.LEFT), Direction.CENTER)
        self.assertEqual(smoother.update(1, Direction.LEFT), Direction.LEFT)

    def test_flicker_is_ignored(self) -> None:
        smoother = DirectionSmoother(stable_frames=3)
        smoother.update(1, Direction.CENTER)
        for direction in (Direction.LEFT, Direction.CENTER, Direction.LEFT, Direction.CENTER):
            self.assertEqual(smoother.update(1, direction), Direction.CENTER)

    def test_keys_are_independent(self) -> None:
        smoother = DirectionSmoother(stable_frames=2)
        smoother.update(1, Direction.LEFT)
        smoother.update(2, Direction.RIGHT)
        self.assertEqual(smoother.update(1, Direction.CENTER), Direction.LEFT)
        self.assertEqual(smoother.update(2, Direction.CENTER), Direction.RIGHT)

    def test_forget_and_reset(self) -> None:
        smoother = DirectionSmoother(stable_frames=2)
        smoother.update(1, Direction.LEFT)
        smoother.forget(1)
        self.assertEqual(smoother.update(1, Direction.RIGHT), Direction.RIGHT)
        smoother.reset()
        self.assertEqual(smoother.update(1, Direction.CENTER), Direction.CENTER)

    def test_invalid_stable_frames(self) -> None:
        with self.assertRaises(ValueError):
            DirectionSmoother(stable_frames=0)


if __name__ == "__main__":
    unittest.main()
