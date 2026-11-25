"""Tests for sentence building, cooldowns and the speech queue."""

import sys
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from direction import Box, Direction  # noqa: E402
from tracking import TrackEvent  # noqa: E402
from voice import (  # noqa: E402
    NullBackend,
    VoiceFeedback,
    announcement_priority,
    build_announcement,
    build_direction_change,
    make_backend,
    spoken_name,
)


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def event(
    kind: str, direction: Direction | None, track_id: int = 1, area: float = 100.0
) -> TrackEvent:
    box = Box(0, 0, area ** 0.5, area ** 0.5)
    return TrackEvent(
        kind=kind,
        track_id=track_id,
        class_id=1,
        name="chair",
        confidence=0.9,
        direction=direction,
        previous_direction=Direction.LEFT if kind == "direction_change" else None,
        box=box,
    )


class SentenceTests(unittest.TestCase):
    def test_spoken_name(self) -> None:
        self.assertEqual(spoken_name("chair"), "chair")
        self.assertEqual(spoken_name("person (upper body)"), "person, upper body")

    def test_build_announcement(self) -> None:
        self.assertEqual(build_announcement("chair", Direction.LEFT), "chair detected on your left.")
        self.assertEqual(build_announcement("person", Direction.CENTER), "person detected ahead.")
        self.assertEqual(build_announcement("table", Direction.RIGHT), "table detected on your right.")
        self.assertEqual(build_announcement("chair", None), "chair detected.")

    def test_build_announcement_with_approximate_distance(self) -> None:
        self.assertEqual(
            build_announcement("chair", Direction.LEFT, 2.4),
            "chair detected about 2.5 metres on your left.",
        )
        self.assertEqual(
            build_announcement("person", Direction.CENTER, 1.2),
            "person detected about 1 metre ahead.",
        )
        self.assertEqual(
            build_announcement("chair", None, 0.3),
            "chair detected less than half a metre away.",
        )

    def test_event_distance_is_spoken(self) -> None:
        voice = VoiceFeedback(enabled=True, backend="console", clock=FakeClock())
        ev = event("appeared", Direction.CENTER)
        ev.distance_m = 2.0
        self.assertTrue(voice.announce(ev))
        self.assertEqual(voice.last_message, "chair detected about 2 metres ahead.")

    def test_build_direction_change(self) -> None:
        self.assertEqual(
            build_direction_change("chair", Direction.RIGHT, Direction.CENTER),
            "chair moved from ahead to on your right.",
        )
        self.assertEqual(
            build_direction_change("chair", Direction.LEFT, None),
            "chair is now on your left.",
        )
        self.assertEqual(
            build_direction_change("chair", Direction.RIGHT, Direction.CENTER, 2.0),
            "chair moved from ahead to on your right, about 2 metres away.",
        )

    def test_priority_prefers_center_then_larger(self) -> None:
        left = event("appeared", Direction.LEFT, track_id=1, area=400)
        center = event("appeared", Direction.CENTER, track_id=2, area=100)
        self.assertLess(announcement_priority(center), announcement_priority(left))

        small = event("appeared", Direction.CENTER, track_id=3, area=100)
        large = event("appeared", Direction.CENTER, track_id=4, area=900)
        self.assertLess(announcement_priority(large), announcement_priority(small))


class VoiceCooldownTests(unittest.TestCase):
    def make_voice(self, **kwargs) -> VoiceFeedback:
        params = dict(
            enabled=True,
            backend="console",
            cooldown_s=10.0,
            min_gap_s=0.0,
            change_cooldown_s=3.0,
            clock=FakeClock(),
        )
        params.update(kwargs)
        return VoiceFeedback(**params)

    def test_same_track_is_not_repeated_within_cooldown(self) -> None:
        voice = self.make_voice()
        ev = event("appeared", Direction.LEFT)
        self.assertTrue(voice.announce(ev))
        self.assertFalse(voice.announce(ev))
        self.assertEqual(voice.dropped_count, 1)

    def test_repeat_allowed_after_cooldown(self) -> None:
        voice = self.make_voice()
        ev = event("appeared", Direction.LEFT)
        self.assertTrue(voice.announce(ev))
        voice._clock.advance(10.1)
        self.assertTrue(voice.announce(ev))

    def test_different_tracks_are_independent(self) -> None:
        voice = self.make_voice()
        self.assertTrue(voice.announce(event("appeared", Direction.LEFT, track_id=1)))
        self.assertTrue(voice.announce(event("appeared", Direction.RIGHT, track_id=2)))

    def test_direction_change_uses_change_cooldown(self) -> None:
        voice = self.make_voice()
        self.assertTrue(voice.announce(event("appeared", Direction.LEFT)))
        # A change immediately after the appearance is inside the change cooldown.
        self.assertFalse(voice.announce(event("direction_change", Direction.CENTER)))
        voice._clock.advance(3.1)
        self.assertTrue(voice.announce(event("direction_change", Direction.CENTER)))

    def test_min_gap_blocks_back_to_back_speech(self) -> None:
        voice = self.make_voice(min_gap_s=5.0)
        voice.spoken_count = 1
        voice._last_speak_time = voice._clock()
        self.assertFalse(voice.announce(event("appeared", Direction.LEFT)))
        voice._clock.advance(5.1)
        self.assertTrue(voice.announce(event("appeared", Direction.LEFT)))

    def test_disabled_voice_queues_nothing(self) -> None:
        voice = self.make_voice(enabled=False)
        self.assertFalse(voice.announce(event("appeared", Direction.LEFT)))
        self.assertFalse(voice.say("hello"))

    def test_disappearance_is_silent(self) -> None:
        voice = self.make_voice()
        self.assertFalse(voice.announce(event("disappeared", Direction.LEFT)))
        self.assertEqual(voice.dropped_count, 0)

    def test_toggle(self) -> None:
        voice = self.make_voice(enabled=True)
        self.assertFalse(voice.toggle())
        self.assertFalse(voice.enabled)
        self.assertTrue(voice.toggle())
        self.assertTrue(voice.enabled)

    def test_rate_and_volume_are_clamped(self) -> None:
        voice = self.make_voice()
        voice.set_rate(10)
        self.assertEqual(voice.rate, 80)
        voice.set_rate(1000)
        self.assertEqual(voice.rate, 400)
        voice.set_volume(-1)
        self.assertEqual(voice.volume, 0.0)
        voice.set_volume(2)
        self.assertEqual(voice.volume, 1.0)

    def test_queue_overflow_is_counted(self) -> None:
        voice = self.make_voice(queue_size=1, cooldown_s=0.0)
        self.assertTrue(voice.announce(event("appeared", Direction.LEFT, track_id=1)))
        # The worker is not running, so the single queue slot stays full.
        self.assertFalse(voice.announce(event("appeared", Direction.RIGHT, track_id=2)))
        self.assertGreaterEqual(voice.dropped_count, 1)


class StartupGapTests(unittest.TestCase):
    """The greeting must not consume the inter-utterance gap.

    Otherwise an object that is already in view when the app starts produces
    its one and only ``appeared`` event inside the greeting's gap, is dropped,
    and is never announced again for as long as it stays in view.
    """

    def _wait_for_speech(self, voice: VoiceFeedback, count: int = 1) -> None:
        deadline = time.monotonic() + 5.0
        while voice.spoken_count < count and time.monotonic() < deadline:
            time.sleep(0.02)

    def test_startup_line_does_not_block_the_first_announcement(self) -> None:
        voice = VoiceFeedback(
            enabled=True, backend="console", cooldown_s=0.0, min_gap_s=60.0
        )
        voice.start()
        try:
            self.assertTrue(
                voice.say("Assistive navigation ready.", counts_as_gap=False)
            )
            self._wait_for_speech(voice)
            self.assertEqual(voice.spoken_count, 1)
            self.assertTrue(
                voice.announce(event("appeared", Direction.CENTER)),
                "an object seen at startup must still be announced",
            )
        finally:
            voice.stop()

    def test_a_real_announcement_still_starts_the_gap(self) -> None:
        voice = VoiceFeedback(
            enabled=True, backend="console", cooldown_s=0.0, min_gap_s=60.0
        )
        voice.start()
        try:
            self.assertTrue(voice.say("chair detected ahead."))
            self._wait_for_speech(voice)
            self.assertFalse(
                voice.announce(event("appeared", Direction.LEFT, track_id=2)),
                "the gap must still protect back-to-back object announcements",
            )
        finally:
            voice.stop()


class WorkerTests(unittest.TestCase):
    def test_console_backend_worker_speaks(self) -> None:
        voice = VoiceFeedback(
            enabled=True, backend="console", cooldown_s=0.0, min_gap_s=0.0
        )
        voice.start()
        try:
            self.assertTrue(voice.say("test utterance"))
            deadline = time.monotonic() + 5.0
            while voice.spoken_count == 0 and time.monotonic() < deadline:
                time.sleep(0.02)
            self.assertEqual(voice.spoken_count, 1)
            self.assertEqual(voice.last_message, "test utterance")
            self.assertEqual(voice.backend_name, "console")
        finally:
            voice.stop()

    def test_make_backend_console(self) -> None:
        backend = make_backend("console")
        self.assertIsInstance(backend, NullBackend)


if __name__ == "__main__":
    unittest.main()
