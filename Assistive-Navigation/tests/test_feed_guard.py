"""Tests for the feed guard.

A camera that is connected but not streaming produces either black frames or a
frozen placeholder. Both look exactly like an empty room: no detections and no
announcements. The guard turns those silent failures into a spoken warning.
"""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from main import (  # noqa: E402
    FEED_WARMUP_FRAMES,
    AssistiveNavApp,
    is_black_frame,
)


class _FakeVoice:
    def __init__(self) -> None:
        self.enabled = True
        self.said: list[str] = []

    def say(self, text: str) -> bool:
        self.said.append(text)
        return True


class _FakeCamera:
    is_video_file = False
    label = "DroidCam Source 3 (index 2)"


def _frame(value: int) -> np.ndarray:
    return np.full((48, 64, 3), value, np.uint8)


def _noisy_frame(seed: int) -> np.ndarray:
    """A frame with per-frame noise, like any real sensor produces."""
    rng = np.random.default_rng(seed)
    return rng.integers(60, 180, size=(48, 64, 3), dtype=np.uint8)


def _make_app(**overrides):
    app = SimpleNamespace(
        _feed_warning_shown=False,
        _brightest_frame=0.0,
        _previous_frame=None,
        _identical_frames=0,
        frame_index=0,
        camera=_FakeCamera(),
        voice=_FakeVoice(),
    )
    for key, value in overrides.items():
        setattr(app, key, value)
    app._warn_feed = lambda problem, spoken, hint: AssistiveNavApp._warn_feed(
        app, problem, spoken, hint
    )
    return app


def _feed(app, frames) -> None:
    """Push frames through the guard, incrementing the frame index like the loop."""
    for frame in frames:
        app.frame_index += 1
        AssistiveNavApp._check_feed(app, frame)


class BlackFrameTests(unittest.TestCase):
    def test_black_frame_is_black(self) -> None:
        self.assertTrue(is_black_frame(_frame(0)))

    def test_near_black_frame_is_black(self) -> None:
        self.assertTrue(is_black_frame(_frame(3)))

    def test_dim_room_is_not_black(self) -> None:
        self.assertFalse(is_black_frame(_frame(20)))

    def test_normal_frame_is_not_black(self) -> None:
        self.assertFalse(is_black_frame(_frame(120)))

    def test_threshold_is_configurable(self) -> None:
        self.assertTrue(is_black_frame(_frame(40), threshold=64.0))
        self.assertFalse(is_black_frame(_frame(40), threshold=16.0))


class BlackFeedGuardTests(unittest.TestCase):
    def test_silent_before_enough_frames(self) -> None:
        app = _make_app()
        _feed(app, [_frame(0)] * (FEED_WARMUP_FRAMES - 1))
        self.assertFalse(app._feed_warning_shown)
        self.assertEqual(app.voice.said, [])

    def test_warns_once_on_persistent_black_feed(self) -> None:
        app = _make_app()
        _feed(app, [_frame(0)] * (FEED_WARMUP_FRAMES + 10))
        self.assertTrue(app._feed_warning_shown)
        self.assertEqual(app.voice.said, ["Camera is not sending video."])

    def test_bright_frame_early_suppresses_black_warning(self) -> None:
        app = _make_app()
        frames = [_frame(0), _frame(90)]
        frames += [_noisy_frame(seed) for seed in range(FEED_WARMUP_FRAMES)]
        _feed(app, frames)
        self.assertFalse(app._feed_warning_shown)
        self.assertEqual(app.voice.said, [])

    def test_dark_first_frames_do_not_warn_once_scene_brightens(self) -> None:
        """Auto-exposure can start dark; only a permanently black feed warns."""
        app = _make_app()
        _feed(app, [_frame(1)] * (FEED_WARMUP_FRAMES - 1) + [_frame(40)])
        self.assertFalse(app._feed_warning_shown)

    def test_video_files_are_not_checked(self) -> None:
        camera = _FakeCamera()
        camera.is_video_file = True
        app = _make_app(camera=camera)
        _feed(app, [_frame(0)] * (FEED_WARMUP_FRAMES + 5))
        self.assertFalse(app._feed_warning_shown)
        self.assertEqual(app.voice.said, [])

    def test_no_speech_when_voice_disabled(self) -> None:
        voice = _FakeVoice()
        voice.enabled = False
        app = _make_app(voice=voice)
        _feed(app, [_frame(0)] * (FEED_WARMUP_FRAMES + 5))
        self.assertTrue(app._feed_warning_shown)
        self.assertEqual(voice.said, [])


class FrozenFeedGuardTests(unittest.TestCase):
    def test_warns_on_a_frozen_placeholder(self) -> None:
        """DroidCam shows a bright still image until connected."""
        app = _make_app()
        _feed(app, [_frame(143)] * (FEED_WARMUP_FRAMES + 10))
        self.assertTrue(app._feed_warning_shown)
        self.assertEqual(app.voice.said, ["Camera video is frozen."])

    def test_live_video_never_warns(self) -> None:
        app = _make_app()
        _feed(app, [_noisy_frame(seed) for seed in range(FEED_WARMUP_FRAMES + 10)])
        self.assertFalse(app._feed_warning_shown)
        self.assertEqual(app.voice.said, [])

    def test_a_moving_scene_does_not_warn(self) -> None:
        app = _make_app()
        frames = [_frame(100 + (i % 2) * 10) for i in range(FEED_WARMUP_FRAMES + 10)]
        _feed(app, frames)
        self.assertFalse(app._feed_warning_shown)

    def test_freeze_detection_resets_when_video_resumes(self) -> None:
        app = _make_app()
        frames = [_frame(143)] * (FEED_WARMUP_FRAMES - 2)
        frames += [_noisy_frame(seed) for seed in range(10)]
        _feed(app, frames)
        self.assertFalse(app._feed_warning_shown)

    def test_black_and_frozen_report_black_first(self) -> None:
        """A black frozen feed is described as black, the more actionable cause."""
        app = _make_app()
        _feed(app, [_frame(0)] * (FEED_WARMUP_FRAMES + 10))
        self.assertEqual(app.voice.said, ["Camera is not sending video."])


if __name__ == "__main__":
    unittest.main()
