"""Model loading, camera access and per-frame inference.

The trained weights are used read-only. Class names come from
``config/class_names.json`` and are never written back into the model file, so
renaming a class costs nothing and the checkpoint stays exactly as trained.
"""

from __future__ import annotations

import json
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import cv2
import numpy as np

from cameras import open_camera
from direction import Box, Direction

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS = PROJECT_ROOT / "models" / "best.pt"
DEFAULT_NAMES = PROJECT_ROOT / "config" / "class_names.json"


# ---------------------------------------------------------------------------
# Class names
# ---------------------------------------------------------------------------


def load_class_names(path: str | Path = DEFAULT_NAMES) -> dict[int, str]:
    """Read ``class_names.json`` and return ``{class_id: name}``.

    Keys starting with ``_`` (comments) and non-string values are ignored. A
    missing or empty file is an error: showing ``class_07`` to a blind user is
    worse than refusing to start.
    """
    names_path = Path(path)
    if not names_path.is_file():
        raise FileNotFoundError(
            f"Class-name file not found: {names_path}\n"
            "Copy model_training/class_names.json to config/class_names.json."
        )
    raw = json.loads(names_path.read_text(encoding="utf-8-sig"))
    names: dict[int, str] = {}
    for key, value in raw.items():
        if str(key).startswith("_") or not isinstance(value, str):
            continue
        try:
            names[int(key)] = value.strip()
        except ValueError:
            continue
    if not names:
        raise ValueError(f"No usable class names in {names_path}")
    return names


def placeholder(class_id: int) -> str:
    """Fallback label for an ID that the config does not describe."""
    return f"class_{class_id:02d}"


# ---------------------------------------------------------------------------
# Detections
# ---------------------------------------------------------------------------


@dataclass
class Detection:
    """One object in one frame."""

    class_id: int
    name: str
    confidence: float
    box: Box
    track_id: int | None = None
    direction: Direction | None = None
    direction_stable: Direction | None = None
    # Approximate distance in metres, filled by the main loop when enabled.
    distance_m: float | None = None

    @property
    def display_direction(self) -> Direction | None:
        """Smoothed zone if available, else the raw one."""
        return self.direction_stable or self.direction

    @property
    def label(self) -> str:
        parts = [self.name, f"{self.confidence:.2f}"]
        if self.display_direction is not None:
            parts.append(self.display_direction.short)
        return "  ".join(parts)


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------


class CameraStream:
    """Background frame reader with two delivery modes.

    * **realtime** (default for cameras) - only the newest frame matters. Old
      frames are dropped so the consumer always works on fresh video and can
      never fall behind the sensor.
    * **sequential** (default for video files) - every frame is delivered in
      order and the reader waits when the consumer falls behind. A test clip is
      therefore processed exactly once, in order, with no skipped frames.

    Reading in a thread also means an unplugged USB camera cannot freeze the
    detection loop: the consumer simply gets ``None`` until the camera returns.
    """

    def __init__(
        self,
        source: int | str = 0,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        retry_delay: float = 0.5,
        loop_video: bool = False,
        realtime: bool | None = None,
        open_fn: Callable[[], "cv2.VideoCapture | None"] | None = None,
        queue_size: int | None = None,
        backend: str | None = None,
        label: str | None = None,
    ) -> None:
        self.source = source
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.retry_delay = float(retry_delay)
        self.loop_video = bool(loop_video)
        self._open_fn = open_fn
        self.backend = backend
        self.label = label or (
            str(source) if isinstance(source, str) else f"camera {source}"
        )
        # A file has a finite, ordered sequence; a camera only has "now".
        self.realtime = (not isinstance(source, str)) if realtime is None else bool(realtime)
        if queue_size is None:
            queue_size = 1 if self.realtime else 8
        self._queue: queue.Queue = queue.Queue(maxsize=max(1, int(queue_size)))
        self._end_marker = object()

        self._cap: cv2.VideoCapture | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

        self.reconnects = 0
        self.frames_read = 0
        self.ended = False
        self.last_error: str | None = None

    # -- properties ---------------------------------------------------------
    @property
    def is_open(self) -> bool:
        return self._cap is not None and bool(self._cap.isOpened())

    @property
    def is_video_file(self) -> bool:
        return isinstance(self.source, str)

    # -- opening ------------------------------------------------------------
    def _open(self) -> cv2.VideoCapture | None:
        if self._open_fn is not None:
            return self._open_fn()
        if self.is_video_file:
            cap = cv2.VideoCapture(str(self.source))
            if not cap.isOpened():
                cap.release()
                raise FileNotFoundError(f"Could not open video file: {self.source}")
            return cap
        return open_camera(
            int(self.source),
            self.width,
            self.height,
            self.fps,
            backend=self.backend,
            label=self.label,
        )

    def start(self, timeout: float = 8.0) -> None:
        """Open the source synchronously, then keep reading in the background."""
        if self._thread is not None:
            return
        self._cap = self._open()
        self._thread = threading.Thread(target=self._run, name="camera-reader", daemon=True)
        self._thread.start()
        # Wait until the first frame exists, or the file turned out to be empty.
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.frames_read > 0 or self.ended:
                return
            time.sleep(0.02)

    # -- reader thread ------------------------------------------------------
    def _deliver(self, frame: np.ndarray) -> bool:
        """Hand a frame to the consumer.

        Realtime mode keeps the queue at one frame and drops the oldest one, so
        latency stays bounded. Sequential mode blocks instead, so no frame is
        ever lost.
        """
        if self._stop.is_set():
            return False
        if self.realtime:
            try:
                self._queue.put_nowait(frame)
            except queue.Full:
                try:
                    self._queue.get_nowait()  # discard the stale frame
                except queue.Empty:
                    pass
                try:
                    self._queue.put_nowait(frame)
                except queue.Full:
                    return False
            return True

        while not self._stop.is_set():
            try:
                self._queue.put(frame, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def _put_blocking(self, item) -> None:
        while not self._stop.is_set():
            try:
                self._queue.put(item, timeout=0.1)
                return
            except queue.Full:
                continue

    def _run(self) -> None:
        consecutive_failures = 0
        while not self._stop.is_set():
            if self._cap is None:
                if self.ended:
                    self._stop.wait(self.retry_delay)
                    continue
                try:
                    self._cap = self._open()
                    consecutive_failures = 0
                    self.last_error = None
                except Exception as exc:  # noqa: BLE001 - keep retrying
                    self.last_error = str(exc)
                    self._stop.wait(self.retry_delay)
                    continue

            ok, frame = self._cap.read()
            if ok and frame is not None:
                consecutive_failures = 0
                if self._deliver(frame):
                    self.frames_read += 1
                continue

            # Read failed: end of file, or the camera went away.
            if self.is_video_file:
                if self.loop_video and not self._stop.is_set():
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                self.ended = True
                self._put_blocking(self._end_marker)
                return

            consecutive_failures += 1
            if consecutive_failures >= 5:
                self._cap.release()
                self._cap = None
                self.reconnects += 1
                self.last_error = "camera lost, reconnecting"
                print(f"[camera] lost after {self.frames_read} frames; retrying", flush=True)
            self._stop.wait(0.05)

    # -- consumer API -------------------------------------------------------
    def read(self, timeout: float = 2.0) -> np.ndarray | None:
        """Return the next (file) or newest (camera) frame, else ``None``."""
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            try:
                item = self._queue.get(timeout=min(remaining, 0.1))
            except queue.Empty:
                if self._stop.is_set():
                    return None
                continue
            if item is self._end_marker:
                self.ended = True
                return None
            return item

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------


class Detector:
    """Thin wrapper around a trained Ultralytics YOLO model.

    Tracking is on by default: ``model.track`` runs ByteTrack inside the same
    call as detection, which costs almost nothing on top of inference.
    """

    def __init__(
        self,
        weights: str | Path = DEFAULT_WEIGHTS,
        names_path: str | Path = DEFAULT_NAMES,
        conf: float = 0.35,
        iou: float = 0.45,
        imgsz: int = 640,
        device: str | None = None,
        half: bool | None = None,
        use_tracking: bool = True,
        tracker: str = "bytetrack.yaml",
        verbose: bool = True,
    ) -> None:
        self.weights = Path(weights)
        if not self.weights.is_file():
            raise FileNotFoundError(
                f"Weights not found: {self.weights}\n"
                "Copy your trained best.pt to models/best.pt or pass --weights."
            )

        self.conf = float(conf)
        self.iou = float(iou)
        self.imgsz = int(imgsz)
        self.verbose = verbose
        self.tracker = tracker
        self.use_tracking = bool(use_tracking)
        self.tracking_error: str | None = None
        self.last_latency_ms = 0.0

        try:
            from ultralytics import YOLO
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"ultralytics could not be imported ({exc}). Install it with "
                "'pip install ultralytics'."
            ) from exc

        if self.verbose:
            print(f"[model] loading {self.weights}", flush=True)
        self.model = YOLO(str(self.weights))

        # Resolve the device once so every later call agrees with it.
        import torch

        if device is None:
            device = "0" if torch.cuda.is_available() else "cpu"
        self.device = str(device)
        self.using_cuda = self.device != "cpu" and torch.cuda.is_available()
        if half is None:
            half = self.using_cuda
        self.half = bool(half and self.using_cuda)

        # Ultralytics renamed the FP16 switch from ``half`` to ``quantize``.
        # Use whichever this build understands so no deprecation warning is
        # printed on every frame.
        try:
            from ultralytics.cfg import DEFAULT_CFG_DICT

            self._precision_key = "quantize" if "quantize" in DEFAULT_CFG_DICT else "half"
        except Exception:  # noqa: BLE001 - very old or patched builds
            self._precision_key = "half"

        # Names are display metadata only; the checkpoint is not modified.
        self.names = load_class_names(names_path)
        self.names_path = Path(names_path)
        model_names = dict(getattr(self.model.model, "names", {}) or {})
        model_names.update(self.names)
        self.model.model.names = model_names

        self.num_classes = int(getattr(self.model.model, "nc", len(self.names)))
        missing = [i for i in range(self.num_classes) if i not in self.names]
        if missing:
            print(
                f"[model] warning: {len(missing)} class IDs have no name in "
                f"{self.names_path.name}: {missing[:8]}",
                flush=True,
            )
        if self.verbose:
            print(
                f"[model] {self.num_classes} classes, device={self.device}, "
                f"half={self.half}, imgsz={self.imgsz}, "
                f"tracking={'on' if self.use_tracking else 'off'}",
                flush=True,
            )

    # -- helpers ------------------------------------------------------------
    def name_for(self, class_id: int) -> str:
        return self.names.get(int(class_id), placeholder(int(class_id)))

    def set_conf(self, conf: float) -> float:
        self.conf = max(0.01, min(0.99, float(conf)))
        return self.conf

    def warmup(self, frame_size: tuple[int, int] | None = None) -> None:
        """Compile kernels for the real capture size before the loop starts.

        This build of Ultralytics letterboxes to a *rectangular* 640-long-side
        canvas, so every new aspect ratio costs a one-off cuDNN autotune of
        several seconds. Warming up with the actual camera resolution (default
        720p) moves that cost to startup instead of the first live frame.
        """
        height, width = frame_size if frame_size else (720, 1280)
        dummy = np.zeros((int(height), int(width), 3), dtype=np.uint8)
        started = time.perf_counter()
        try:
            # Two calls: the first builds the predictor, the second fills the
            # shape cache for the tracking path.
            self.detect(dummy, use_tracking=False)
            self.detect(dummy, use_tracking=self.use_tracking)
        except Exception as exc:  # noqa: BLE001 - warmup must never be fatal
            print(f"[model] warmup skipped: {exc}", flush=True)
            return
        if self.verbose:
            print(
                f"[model] warmup at {width}x{height} took "
                f"{1000 * (time.perf_counter() - started):.0f} ms",
                flush=True,
            )

    # -- inference ----------------------------------------------------------
    def _kwargs(self) -> dict:
        kwargs = {
            "imgsz": self.imgsz,
            "conf": self.conf,
            "iou": self.iou,
            "device": self.device,
            "verbose": False,
        }
        if self._precision_key == "quantize":
            kwargs["quantize"] = 16 if self.half else None
        else:
            kwargs["half"] = self.half
        return kwargs

    @staticmethod
    def _is_tracking_error(exc: Exception) -> bool:
        text = str(exc).lower()
        return any(word in text for word in ("lap", "tracker", "assignment", "bytetrack"))

    def detect(
        self, frame: np.ndarray, use_tracking: bool | None = None
    ) -> tuple[list[Detection], float]:
        """Run one frame. Returns ``(detections, latency_ms)``."""
        want_tracking = self.use_tracking if use_tracking is None else bool(use_tracking)
        started = time.perf_counter()
        results = None

        if want_tracking:
            try:
                results = self.model.track(
                    source=frame, persist=True, tracker=self.tracker, **self._kwargs()
                )
            except Exception as exc:  # noqa: BLE001
                if not self._is_tracking_error(exc):
                    raise
                self.use_tracking = False
                self.tracking_error = str(exc)
                print(
                    f"[model] ByteTrack unavailable ({exc}); "
                    "falling back to IoU tracking",
                    flush=True,
                )
        if results is None:
            results = self.model.predict(source=frame, **self._kwargs())

        latency_ms = 1000.0 * (time.perf_counter() - started)
        self.last_latency_ms = latency_ms
        return self._to_detections(results), latency_ms

    def _to_detections(self, results) -> list[Detection]:
        if not results:
            return []
        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []

        xyxy = boxes.xyxy.detach().cpu().numpy()
        conf = boxes.conf.detach().cpu().numpy()
        cls = boxes.cls.detach().cpu().numpy().astype(int)
        if getattr(boxes, "id", None) is not None:
            ids = boxes.id.detach().cpu().numpy().astype(int).tolist()
        else:
            ids = [None] * len(xyxy)

        detections: list[Detection] = []
        for (x1, y1, x2, y2), score, class_id, track_id in zip(xyxy, conf, cls, ids):
            detections.append(
                Detection(
                    class_id=int(class_id),
                    name=self.name_for(int(class_id)),
                    confidence=float(score),
                    box=Box(float(x1), float(y1), float(x2), float(y2)),
                    track_id=None if track_id is None else int(track_id),
                )
            )
        return detections

    @property
    def inference_ms(self) -> float:
        """Wall-clock latency of the last :meth:`detect` call, in milliseconds."""
        return self.last_latency_ms


def detections_to_rows(detections: Iterable[Detection], frame_index: int) -> list[dict]:
    """Flat rows, handy for CSV logging or tests."""
    rows = []
    for det in detections:
        rows.append(
            {
                "frame": frame_index,
                "track_id": det.track_id,
                "class_id": det.class_id,
                "name": det.name,
                "confidence": round(det.confidence, 4),
                "direction": det.display_direction.value if det.display_direction else "",
                "x1": round(det.box.x1, 1),
                "y1": round(det.box.y1, 1),
                "x2": round(det.box.x2, 1),
                "y2": round(det.box.y2, 1),
            }
        )
    return rows
