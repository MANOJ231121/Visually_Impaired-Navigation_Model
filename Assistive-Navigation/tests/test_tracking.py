"""Tests for track identity, lifecycle events and direction changes."""

import sys
import unittest
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from direction import Box, Direction  # noqa: E402
from tracking import TrackManager, iou  # noqa: E402


@dataclass
class FakeDetection:
    class_id: int
    name: str
    confidence: float
    box: Box
    track_id: int | None = None
    direction: Direction | None = None
    direction_stable: Direction | None = None


def make_det(
    class_id: int = 1,
    name: str = "chair",
    box: Box | None = None,
    track_id: int | None = None,
    direction: Direction | None = None,
) -> FakeDetection:
    return FakeDetection(
        class_id=class_id,
        name=name,
        confidence=0.9,
        box=box or Box(100, 100, 200, 300),
        track_id=track_id,
        direction=direction,
    )


class IoUTests(unittest.TestCase):
    def test_identical_boxes(self) -> None:
        self.assertAlmostEqual(iou(Box(0, 0, 10, 10), Box(0, 0, 10, 10)), 1.0)

    def test_disjoint_boxes(self) -> None:
        self.assertEqual(iou(Box(0, 0, 10, 10), Box(20, 20, 30, 30)), 0.0)

    def test_partial_overlap(self) -> None:
        # intersection 5x5=25, union 100+100-25=175
        self.assertAlmostEqual(iou(Box(0, 0, 10, 10), Box(5, 5, 15, 15)), 25 / 175)


class TrackLifecycleTests(unittest.TestCase):
    def test_appear_persist_disappear(self) -> None:
        tracker = TrackManager(max_missing=2)
        det = make_det()

        events = tracker.update([det], 1)
        self.assertEqual([e.kind for e in events], ["appeared"])
        self.assertEqual(det.track_id, 1)

        events = tracker.update([det], 2)
        self.assertEqual(events, [])
        self.assertEqual(tracker.get(1).hits, 2)

        self.assertEqual(tracker.update([], 3), [])
        self.assertEqual(tracker.update([], 4), [])
        events = tracker.update([], 5)
        self.assertEqual([e.kind for e in events], ["disappeared"])
        self.assertIsNone(tracker.get(1))

    def test_reappearance_gets_a_new_track(self) -> None:
        tracker = TrackManager(max_missing=1)
        det = make_det()
        tracker.update([det], 1)
        tracker.update([], 2)
        tracker.update([], 3)  # dropped here

        again = make_det()
        events = tracker.update([again], 4)
        self.assertEqual([e.kind for e in events], ["appeared"])
        self.assertEqual(again.track_id, 2)

    def test_class_aware_matching(self) -> None:
        tracker = TrackManager(max_missing=2)
        chair = make_det(class_id=1, name="chair")
        tracker.update([chair], 1)

        # Same box, different class -> must not inherit the chair's identity.
        book = make_det(class_id=21, name="book")
        events = tracker.update([book], 2)
        self.assertEqual([e.kind for e in events], ["appeared"])
        self.assertNotEqual(book.track_id, chair.track_id)

    def test_iou_fallback_matches_moving_box(self) -> None:
        tracker = TrackManager(max_missing=2, iou_threshold=0.3)
        first = make_det(box=Box(100, 100, 200, 300))
        tracker.update([first], 1)
        moved = make_det(box=Box(110, 105, 210, 305))
        tracker.update([moved], 2)
        self.assertEqual(moved.track_id, first.track_id)

    def test_external_track_id_is_respected(self) -> None:
        tracker = TrackManager()
        det = make_det(track_id=42)
        events = tracker.update([det], 1)
        self.assertEqual([e.kind for e in events], ["appeared"])
        self.assertEqual(det.track_id, 42)
        self.assertIsNotNone(tracker.get(42))

    def test_two_objects_get_distinct_ids(self) -> None:
        tracker = TrackManager()
        a = make_det(box=Box(0, 0, 50, 50))
        b = make_det(box=Box(300, 0, 350, 50), class_id=21, name="book")
        events = tracker.update([a, b], 1)
        self.assertEqual(len(events), 2)
        self.assertNotEqual(a.track_id, b.track_id)

    def test_invalid_max_missing(self) -> None:
        with self.assertRaises(ValueError):
            TrackManager(max_missing=0)

    def test_reset(self) -> None:
        tracker = TrackManager()
        det = make_det()
        tracker.update([det], 1)
        tracker.reset()
        self.assertEqual(tracker.tracks, {})
        fresh = make_det()
        tracker.update([fresh], 1)
        self.assertEqual(fresh.track_id, 1)


class DirectionChangeTests(unittest.TestCase):
    def test_change_is_emitted_once(self) -> None:
        tracker = TrackManager()
        det = make_det(direction=Direction.LEFT)
        tracker.update([det], 1)
        self.assertEqual(tracker.note_directions([det]), [])

        det.direction = Direction.CENTER
        events = tracker.note_directions([det])
        self.assertEqual([e.kind for e in events], ["direction_change"])
        self.assertEqual(events[0].previous_direction, Direction.LEFT)
        self.assertEqual(events[0].direction, Direction.CENTER)

        # Same direction again -> no repeated sentence.
        self.assertEqual(tracker.note_directions([det]), [])

    def test_smoothed_direction_is_preferred(self) -> None:
        tracker = TrackManager()
        det = make_det(direction=Direction.LEFT)
        tracker.update([det], 1)
        tracker.note_directions([det])

        det.direction = Direction.RIGHT
        det.direction_stable = Direction.CENTER
        events = tracker.note_directions([det])
        self.assertEqual(events[0].direction, Direction.CENTER)

    def test_unknown_track_produces_nothing(self) -> None:
        tracker = TrackManager()
        det = make_det(track_id=99, direction=Direction.LEFT)
        self.assertEqual(tracker.note_directions([det]), [])

    def test_mark_spoken(self) -> None:
        tracker = TrackManager()
        det = make_det(direction=Direction.LEFT)
        tracker.update([det], 1)
        tracker.mark_spoken(det.track_id, Direction.LEFT)
        track = tracker.get(det.track_id)
        self.assertEqual(track.spoken_count, 1)
        self.assertEqual(track.announced_direction, Direction.LEFT)


if __name__ == "__main__":
    unittest.main()
