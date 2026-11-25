"""Asynchronous speech output with per-object cooldowns.

Design notes
------------
* Speech runs in its own thread behind a bounded queue. The detection loop only
  ever appends a short string, so a slow voice can never stall the camera.
* ``pyttsx3`` is created *inside* the worker thread. On Windows the SAPI5 driver
  must be used from the thread that created it, otherwise ``runAndWait`` either
  blocks forever or raises.
* Cooldowns are keyed by ``track_id``: the same chair may be announced again
  after ``cooldown_s`` seconds, but not once per frame. A global ``min_gap_s``
  keeps two objects from talking over each other.
* If no speech engine is available the class degrades to a logging backend, so
  the whole pipeline still runs and can be tested on a machine without audio.
"""

from __future__ import annotations

import queue
import re
import threading
import time
from dataclasses import dataclass
from typing import Callable, Protocol

from distance import spoken_distance
from direction import Direction
from tracking import TrackEvent

# ---------------------------------------------------------------------------
# Sentence building
# ---------------------------------------------------------------------------

_PAREN_OPEN = re.compile(r"\s*\(\s*")
_PAREN_CLOSE = re.compile(r"\s*\)")
_SPACES = re.compile(r"\s+")
_BEFORE_PUNCT = re.compile(r"\s+([,.;:])")


def spoken_name(name: str) -> str:
    """Make a config name comfortable to hear.

    ``"person (upper body)"`` becomes ``"person, upper body,"`` so the engine
    pauses instead of reading a bracket. No class name is hardcoded here.
    """
    text = _PAREN_OPEN.sub(", ", name)
    text = _PAREN_CLOSE.sub(",", text)
    text = _BEFORE_PUNCT.sub(r"\1", text)
    text = _SPACES.sub(" ", text).strip(" ,")
    return text or name.strip()


def build_announcement(
    name: str, direction: Direction | None, distance_m: float | None = None
) -> str:
    """``"Chair detected about 2 metres on your left."``

    The distance phrase is optional and always hedged with "about": a bounding
    box cannot measure range, so this is a rough framing cue only.
    """
    label = spoken_name(name)
    distance = spoken_distance(distance_m)
    place = direction.spoken if direction is not None else "away"
    if distance:
        return f"{label} detected {distance} {place}."
    if direction is None:
        return f"{label} detected."
    return f"{label} detected {direction.spoken}."


def build_direction_change(
    name: str,
    direction: Direction,
    previous: Direction | None,
    distance_m: float | None = None,
) -> str:
    """``"Chair moved from ahead to on your right."``"""
    label = spoken_name(name)
    if previous is None:
        text = f"{label} is now {direction.spoken}"
    else:
        text = f"{label} moved from {previous.spoken} to {direction.spoken}"
    distance = spoken_distance(distance_m)
    return f"{text}, {distance} away." if distance else f"{text}."


def announcement_priority(event: TrackEvent) -> tuple[int, float]:
    """Order simultaneous events: centre zone first, then the larger object.

    Box area is a framing cue, not a distance measurement, so this only decides
    speaking order - it is never used to claim that something is close.
    """
    zone_rank = 0 if event.direction == Direction.CENTER else 1
    area = event.box.area if event.box is not None else 0.0
    return (zone_rank, -area)


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------


class SpeechBackend(Protocol):
    name: str

    def speak(self, text: str) -> None: ...
    def stop(self) -> None: ...
    def close(self) -> None: ...


class NullBackend:
    """Used when no TTS engine can be loaded. Prints so tests still see output."""

    name = "console"

    def __init__(self, echo: bool = True) -> None:
        self.echo = echo

    def speak(self, text: str) -> None:
        if self.echo:
            print(f"[voice] {text}", flush=True)

    def stop(self) -> None:
        pass

    def close(self) -> None:
        pass


class Pyttsx3Backend:
    """Windows SAPI5 (or NSSpeech/espeak elsewhere) through pyttsx3."""

    name = "pyttsx3"

    def __init__(self, rate: int = 175, volume: float = 1.0, voice_hint: str | None = None) -> None:
        import pyttsx3  # imported here so the rest of the app has no hard dependency

        self._engine = pyttsx3.init()
        self._lock = threading.Lock()
        self._apply(rate, volume)
        if voice_hint:
            self._select_voice(voice_hint)
        self.voice = self._current_voice_name()

    def _apply(self, rate: int, volume: float) -> None:
        self._engine.setProperty("rate", int(rate))
        self._engine.setProperty("volume", max(0.0, min(1.0, float(volume))))

    def _select_voice(self, hint: str) -> None:
        needle = hint.lower()
        for voice in self._engine.getProperty("voices"):
            blob = f"{getattr(voice, 'name', '')} {getattr(voice, 'id', '')}".lower()
            if needle in blob:
                self._engine.setProperty("voice", voice.id)
                return
        print(f"[voice] no voice matched {hint!r}; using the system default", flush=True)

    def _current_voice_name(self) -> str:
        try:
            voice_id = self._engine.getProperty("voice")
            for voice in self._engine.getProperty("voices"):
                if voice.id == voice_id:
                    return str(getattr(voice, "name", voice_id))
        except Exception:  # noqa: BLE001 - cosmetic only
            pass
        return "default"

    def speak(self, text: str) -> None:
        with self._lock:
            self._engine.say(text)
            self._engine.runAndWait()

    def stop(self) -> None:
        try:
            self._engine.stop()
        except Exception:  # noqa: BLE001 - interrupting speech is best effort
            pass

    def close(self) -> None:
        self.stop()


class SapiBackend:
    """Fallback: talk to SAPI directly through COM (needs pywin32)."""

    name = "sapi"

    def __init__(self, rate: int = 175, volume: float = 1.0) -> None:
        import win32com.client  # noqa: PLC0415 - optional dependency

        self._voice = win32com.client.Dispatch("SAPI.SpVoice")
        self._apply(rate, volume)
        self.voice = "SAPI default"

    def _apply(self, rate: int, volume: float) -> None:
        """Map 80-400 wpm / 0-1 volume onto SAPI's -10..10 / 0..100 ranges."""
        self._rate = int(rate)
        self._volume = max(0, min(100, int(volume * 100)))
        self._voice.Rate = max(-10, min(10, round((self._rate - 175) / 25)))
        self._voice.Volume = self._volume

    def speak(self, text: str) -> None:
        self._voice.Speak(text)

    def stop(self) -> None:
        try:
            self._voice.Speak("", 3)  # SVSFPurgeBeforeSpeak
        except Exception:  # noqa: BLE001
            pass

    def close(self) -> None:
        pass


def make_backend(
    preference: str = "auto", rate: int = 175, volume: float = 1.0, voice_hint: str | None = None
) -> SpeechBackend:
    """Build the best available backend, degrading to console output."""
    order = ["pyttsx3", "sapi"] if preference in ("auto", "") else [preference]
    for choice in order:
        try:
            if choice == "pyttsx3":
                return Pyttsx3Backend(rate=rate, volume=volume, voice_hint=voice_hint)
            if choice == "sapi":
                return SapiBackend(rate=rate, volume=volume)
            if choice in ("console", "null"):
                return NullBackend()
        except Exception as exc:  # noqa: BLE001 - try the next backend
            print(f"[voice] {choice} unavailable: {type(exc).__name__}: {exc}", flush=True)
    return NullBackend()


# ---------------------------------------------------------------------------
# Queued speaker
# ---------------------------------------------------------------------------


@dataclass
class _Utterance:
    text: str
    track_id: int | None
    direction: Direction | None
    kind: str
    created_at: float
    # False for housekeeping lines such as "Assistive navigation ready.": they
    # must not start the inter-utterance gap, or the first real object would be
    # silently dropped and never announced again.
    counts_as_gap: bool = True


class VoiceFeedback:
    """Thread-safe speech queue with cooldowns.

    Parameters
    ----------
    enabled:        start muted when False.
    rate, volume:   engine settings; also changeable at runtime.
    cooldown_s:     minimum seconds before the same track is announced again.
    min_gap_s:      minimum seconds between any two utterances.
    change_cooldown_s: cooldown applied to "moved to your ..." messages.
    queue_size:     bounded queue; extras are dropped rather than delayed.
    """

    def __init__(
        self,
        enabled: bool = True,
        rate: int = 175,
        volume: float = 1.0,
        cooldown_s: float = 6.0,
        min_gap_s: float = 1.5,
        change_cooldown_s: float = 3.0,
        queue_size: int = 4,
        backend: str = "auto",
        voice_hint: str | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._enabled = bool(enabled)
        self.rate = int(rate)
        self.volume = float(volume)
        self.cooldown_s = float(cooldown_s)
        self.min_gap_s = float(min_gap_s)
        self.change_cooldown_s = float(change_cooldown_s)
        # Injectable so cooldown behaviour can be tested without sleeping.
        self._clock = clock or time.monotonic

        self._queue: queue.Queue[_Utterance | None] = queue.Queue(maxsize=max(1, queue_size))
        self._lock = threading.Lock()
        self._backend_preference = backend
        self._voice_hint = voice_hint
        self._backend: SpeechBackend | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._stopping = threading.Event()
        self._applied: tuple[int, float] = (self.rate, self.volume)

        self._last_track_time: dict[int, float] = {}
        self._last_text: str = ""
        self._last_text_time: float = 0.0
        self._last_speak_time: float = 0.0

        self.spoken_count = 0
        self.dropped_count = 0
        self.last_message: str = ""
        self.last_spoken_at: float = 0.0
        self.backend_name = "not started"
        self.error: str | None = None

    # -- lifecycle ----------------------------------------------------------
    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def backend(self) -> SpeechBackend | None:
        return self._backend

    def start(self, timeout: float = 10.0) -> None:
        """Start the worker thread and wait until the engine is ready."""
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="voice-worker", daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout):
            self.error = "speech engine did not start in time"
            print(f"[voice] {self.error}; continuing without speech", flush=True)

    def _run(self) -> None:
        try:
            self._backend = make_backend(
                self._backend_preference, self.rate, self.volume, self._voice_hint
            )
            self.backend_name = self._backend.name
        except Exception as exc:  # noqa: BLE001
            self.error = f"{type(exc).__name__}: {exc}"
            self._backend = NullBackend()
            self.backend_name = self._backend.name
        finally:
            self._ready.set()

        assert self._backend is not None
        while not self._stopping.is_set():
            try:
                item = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if item is None:
                break
            if not self._enabled:
                continue
            # Apply rate/volume changes made from the main thread.
            if (self.rate, self.volume) != self._applied and hasattr(self._backend, "_apply"):
                try:
                    self._backend._apply(self.rate, self.volume)  # type: ignore[attr-defined]
                    self._applied = (self.rate, self.volume)
                except Exception:  # noqa: BLE001
                    pass
            try:
                self._backend.speak(item.text)
                with self._lock:
                    self.spoken_count += 1
                    self.last_message = item.text
                    self.last_spoken_at = self._clock()
                    if item.counts_as_gap:
                        self._last_speak_time = self.last_spoken_at
            except Exception as exc:  # noqa: BLE001 - never kill the worker
                self.error = f"{type(exc).__name__}: {exc}"
                print(f"[voice] speak failed: {self.error}", flush=True)

        self._backend.close()

    def stop(self, timeout: float = 3.0) -> None:
        """Stop speaking, drain the queue and join the worker."""
        self._stopping.set()
        self._drain()
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    def _drain(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                return

    # -- controls -----------------------------------------------------------
    def set_enabled(self, enabled: bool) -> None:
        self._enabled = bool(enabled)
        if not enabled:
            self._drain()
            if self._backend is not None:
                self._backend.stop()

    def toggle(self) -> bool:
        self.set_enabled(not self._enabled)
        return self._enabled

    def stop_speaking(self) -> None:
        """Cut the current sentence (bound to the ``m`` key)."""
        self._drain()
        if self._backend is not None:
            self._backend.stop()

    def set_rate(self, rate: int) -> None:
        self.rate = max(80, min(400, int(rate)))

    def set_volume(self, volume: float) -> None:
        self.volume = max(0.0, min(1.0, float(volume)))

    @property
    def queue_depth(self) -> int:
        return self._queue.qsize()

    # -- speech -------------------------------------------------------------
    def say(self, text: str, track_id: int | None = None, counts_as_gap: bool = True) -> bool:
        """Queue a raw sentence, respecting the global gap.

        ``counts_as_gap=False`` is for housekeeping lines (the startup message):
        they are spoken but do not delay the first object announcement.
        """
        if not self._enabled or not text:
            return False
        now = self._clock()
        with self._lock:
            if now - self._last_speak_time < self.min_gap_s and self.spoken_count:
                self.dropped_count += 1
                return False
            if text == self._last_text and now - self._last_text_time < self.cooldown_s:
                self.dropped_count += 1
                return False
            try:
                self._queue.put_nowait(
                    _Utterance(text=text, track_id=track_id, direction=None, kind="say",
                               created_at=now, counts_as_gap=counts_as_gap)
                )
            except queue.Full:
                self.dropped_count += 1
                return False
            self._last_text = text
            self._last_text_time = now
        return True

    def announce(self, event: TrackEvent) -> bool:
        """Speak an event if its cooldown allows it. Returns True if queued."""
        if not self._enabled:
            return False
        if event.kind == "disappeared":
            return False  # deliberately silent; tracked, not narrated

        now = self._clock()
        if event.kind == "appeared":
            text = build_announcement(event.name, event.direction, event.distance_m)
            cooldown = self.cooldown_s
        elif event.kind == "direction_change":
            text = build_direction_change(
                event.name, event.direction, event.previous_direction, event.distance_m
            )
            cooldown = self.change_cooldown_s
        else:
            return False

        with self._lock:
            last = self._last_track_time.get(event.track_id)
            if last is not None and now - last < cooldown:
                self.dropped_count += 1
                return False
            if now - self._last_speak_time < self.min_gap_s and self.spoken_count:
                self.dropped_count += 1
                return False
            try:
                self._queue.put_nowait(
                    _Utterance(
                        text=text,
                        track_id=event.track_id,
                        direction=event.direction,
                        kind=event.kind,
                        created_at=now,
                    )
                )
            except queue.Full:
                self.dropped_count += 1
                return False
            self._last_track_time[event.track_id] = now
            self._last_text = text
            self._last_text_time = now
            self.last_message = text
        return True

    def announce_many(self, events: list[TrackEvent]) -> int:
        """Announce several events in priority order; returns how many queued."""
        queued = 0
        for event in sorted(events, key=announcement_priority):
            if self.announce(event):
                queued += 1
            if self._queue.full():
                break
        return queued
