#!/usr/bin/env python3
"""
voice.py
========

Small background text-to-speech queue for the demo navigation brain.

Speech is spoken on a daemon thread so detection keeps running while the
sentence is played. On Windows, COM is initialised inside that thread for the
SAPI voice used by ``pyttsx3``.

If the speech engine cannot start, the worker falls back to console output so
the detection pipeline can still be tested without audio.
"""

from __future__ import annotations

import queue
import threading

try:  # Windows only; harmless if pywin32 is not installed.
    import pythoncom
except ImportError:  # pragma: no cover - non-Windows fallback
    pythoncom = None


class _ConsoleEngine:
    """Drop-in stand-in used when no TTS engine is available."""

    def setProperty(self, _name: str, _value: object) -> None:
        pass

    def say(self, text: str) -> None:
        print(f"[VOICE] {text}", flush=True)

    def runAndWait(self) -> None:
        pass


class Speaker:
    """Speak queued strings without blocking the detection loop."""

    def __init__(
        self,
        *,
        rate: int = 165,
        enabled: bool = True,
        backend: str = "pyttsx3",
        voice_hint: str | None = None,
    ) -> None:
        self.enabled = enabled
        self.queue: queue.Queue[str] = queue.Queue()
        self.backend = backend
        self.backend_name = "off" if not enabled else "starting"
        self.last_error: str | None = None
        if not enabled:
            return
        threading.Thread(
            target=self._worker,
            kwargs={
                "rate": rate,
                "backend": backend,
                "voice_hint": voice_hint,
            },
            daemon=True,
        ).start()

    def _worker(self, *, rate: int, backend: str, voice_hint: str | None) -> None:
        if pythoncom is not None:
            pythoncom.CoInitialize()

        engine = None
        try:
            if backend == "console":
                engine = _ConsoleEngine()
            else:
                import pyttsx3

                engine = pyttsx3.init()
                engine.setProperty("rate", rate)
                if voice_hint:
                    self._select_voice(engine, voice_hint)
            self.backend_name = "console" if backend == "console" else "pyttsx3"
        except Exception as exc:  # noqa: BLE001 - keep the demo running without audio
            self.last_error = f"{type(exc).__name__}: {exc}"
            print(
                f"[voice] {backend} unavailable ({self.last_error}); "
                "using console output",
                flush=True,
            )
            engine = _ConsoleEngine()
            self.backend_name = "console"

        try:
            while True:
                text = self.queue.get()
                engine.say(text)
                engine.runAndWait()
        finally:
            if pythoncom is not None:
                pythoncom.CoUninitialize()

    @staticmethod
    def _select_voice(engine, hint: str) -> None:
        needle = hint.lower()
        for voice in engine.getProperty("voices"):
            label = f"{getattr(voice, 'name', '')} {getattr(voice, 'id', '')}".lower()
            if needle in label:
                engine.setProperty("voice", voice.id)
                return
        print(f"[voice] no voice matched {hint!r}; using the system default", flush=True)

    def say(self, text: str) -> None:
        if self.enabled and text.strip():
            self.queue.put(text.strip())
