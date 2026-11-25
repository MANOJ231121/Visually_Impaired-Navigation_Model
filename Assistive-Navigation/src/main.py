"""Assistive navigation prototype - entry point.

    webcam -> YOLO11 detect -> boxes -> direction -> tracking -> speech

Run it with::

    python src/main.py

Keyboard: q quit | v voice on/off | +/- confidence | s screenshot
          m stop talking | t tracking on/off

This is a research prototype. Directions are camera-relative: "ahead" means
"the middle third of the picture", not "close" and not "in your path". It is
not a mobility aid and must not be used to navigate.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# Allow "python src/main.py" without installing the project as a package.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cameras import (  # noqa: E402
    DEFAULT_CAMERA_CONFIG,
    CameraError,
    probe_cameras,
    resolve_camera,
)
from detection import (  # noqa: E402
    DEFAULT_NAMES,
    DEFAULT_WEIGHTS,
    PROJECT_ROOT,
    CameraStream,
    Detection,
    Detector,
    detections_to_rows,
    load_class_names,
)
from distance import DEFAULT_HEIGHTS_PATH, DistanceEstimator  # noqa: E402
from direction import Direction, DirectionEstimator, DirectionSmoother  # noqa: E402
from tracking import TrackManager  # noqa: E402
from voice import VoiceFeedback, announcement_priority  # noqa: E402

WINDOW = "Assistive navigation prototype (research only)"

# A camera that is connected but not streaming fails in one of two ways, and
# both look exactly like an empty room: zero detections and therefore zero
# announcements.
#   1. black frames - phone screen off, DroidCam client not running, lens
#      covered, privacy shutter closed;
#   2. a frozen placeholder - a virtual camera (DroidCam) showing its
#      "not connected" still image.
# Anything this dark cannot contain a detectable object, and a frame that never
# changes is not live video.
BLACK_FEED_MEAN = 5.0
FEED_WARMUP_FRAMES = 30

BOX_COLORS = {
    Direction.LEFT: (0, 170, 255),   # orange
    Direction.CENTER: (0, 200, 0),   # green
    Direction.RIGHT: (255, 150, 0),  # blue
}
DEFAULT_COLOR = (190, 190, 190)
WHITE = (255, 255, 255)
DIM = (120, 120, 120)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def is_black_frame(frame, threshold: float = BLACK_FEED_MEAN) -> bool:
    """True when a frame is so dark that no object could be visible."""
    return float(frame.mean()) < float(threshold)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class AppConfig:
    weights: Path = DEFAULT_WEIGHTS
    names: Path = DEFAULT_NAMES
    camera: str | int = "0"
    camera_config: Path | None = DEFAULT_CAMERA_CONFIG
    camera_backend: str | None = "auto"
    camera_scan_limit: int = 6
    source: str | None = None
    conf: float = 0.35
    iou: float = 0.45
    imgsz: int = 640
    device: str | None = None
    camera_width: int = 1280
    camera_height: int = 720
    use_tracking: bool = True
    tracker: str = "bytetrack.yaml"
    max_missing: int = 15
    stable_frames: int = 3
    left_ratio: float = 1.0 / 3.0
    right_ratio: float = 2.0 / 3.0
    distance_enabled: bool = True
    object_heights: Path = DEFAULT_HEIGHTS_PATH
    vertical_fov: float = 60.0
    voice_enabled: bool = True
    voice_backend: str = "auto"
    voice_hint: str | None = None
    rate: int = 175
    volume: float = 1.0
    cooldown: float = 6.0
    min_gap: float = 1.5
    headless: bool = False
    max_frames: int = 0
    camera_timeout: float = 15.0
    loop_video: bool = False
    snapshot_dir: Path = PROJECT_ROOT / "checks" / "screenshots"
    snapshot_every: int = 0
    record: bool = False
    stats_json: Path | None = None
    log_csv: Path | None = None


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------


class AssistiveNavApp:
    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg
        # Resolve the camera first so a bad alias/index fails before the model
        # spends time loading and warming up.
        self.camera = self._make_camera()
        self.detector = Detector(
            weights=cfg.weights,
            names_path=cfg.names,
            conf=cfg.conf,
            iou=cfg.iou,
            imgsz=cfg.imgsz,
            device=cfg.device,
            use_tracking=cfg.use_tracking,
            tracker=cfg.tracker,
        )
        self.estimator = DirectionEstimator(cfg.left_ratio, cfg.right_ratio)
        self.distance = (
            DistanceEstimator.load(cfg.object_heights, vertical_fov_deg=cfg.vertical_fov)
            if cfg.distance_enabled
            else None
        )
        self.smoother = DirectionSmoother(cfg.stable_frames)
        self.tracker = TrackManager(max_missing=cfg.max_missing)
        self.voice = VoiceFeedback(
            enabled=cfg.voice_enabled,
            rate=cfg.rate,
            volume=cfg.volume,
            cooldown_s=cfg.cooldown,
            min_gap_s=cfg.min_gap,
            backend=cfg.voice_backend,
            voice_hint=cfg.voice_hint,
        )

        self.frame_index = 0
        self.fps = 0.0
        self._fps_ema: float | None = None
        self._last_spoken_at = 0.0
        self._writer: cv2.VideoWriter | None = None
        self._csv_file = None
        self._csv_writer = None
        self._seen_track_ids: set[int] = set()
        self._brightest_frame = 0.0
        self._previous_frame = None
        self._identical_frames = 0
        self._feed_warning_shown = False
        self.stats = {
            "frames": 0,
            "detections": 0,
            "unique_tracks": 0,
            "appeared": 0,
            "direction_changes": 0,
            "disappeared": 0,
            "announcements": 0,
            "camera_reconnects": 0,
            "per_class": {},
            "voice_backend": "not started",
            "tracking": "bytetrack" if self.detector.use_tracking else "iou",
            "camera": self.camera.label,
        }

    # -- setup --------------------------------------------------------------
    def _make_camera(self) -> CameraStream:
        cfg = self.cfg
        if cfg.source:
            # Video files are delivered frame-by-frame; cameras keep only the
            # newest frame so the loop can never lag behind live video.
            return CameraStream(source=cfg.source, loop_video=cfg.loop_video, realtime=False)

        selection = resolve_camera(
            cfg.camera,
            config_path=cfg.camera_config,
            backend=cfg.camera_backend,
            width=cfg.camera_width,
            height=cfg.camera_height,
            max_index=cfg.camera_scan_limit,
        )
        print(f"[camera] source: {selection.label}", flush=True)
        return CameraStream(
            source=selection.index,
            width=cfg.camera_width,
            height=cfg.camera_height,
            realtime=True,
            backend=selection.backend,
            label=selection.label,
        )

    def _open_outputs(self) -> None:
        if self.cfg.snapshot_dir:
            self.cfg.snapshot_dir.mkdir(parents=True, exist_ok=True)
        if self.cfg.log_csv:
            self.cfg.log_csv.parent.mkdir(parents=True, exist_ok=True)
            self._csv_file = self.cfg.log_csv.open("w", newline="", encoding="utf-8")
            self._csv_writer = csv.DictWriter(
                self._csv_file,
                fieldnames=[
                    "frame", "track_id", "class_id", "name", "confidence",
                    "direction", "x1", "y1", "x2", "y2",
                ],
            )
            self._csv_writer.writeheader()

    def _start_writer(self, frame_shape: tuple[int, ...]) -> None:
        """Create the recorder once the real frame size is known.

        Using the configured camera size here would break recording whenever a
        video file has a different resolution.
        """
        height, width = frame_shape[:2]
        self.cfg.snapshot_dir.mkdir(parents=True, exist_ok=True)
        path = self.cfg.snapshot_dir / f"session_{datetime.now():%Y%m%d_%H%M%S}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(str(path), fourcc, 20.0, (int(width), int(height)))
        if not self._writer.isOpened():
            print(f"[record] could not open {path}", file=sys.stderr)
            self._writer = None
            return
        print(f"[record] {path} ({width}x{height})", flush=True)

    # -- main loop ----------------------------------------------------------
    def run(self) -> int:
        cfg = self.cfg
        print("[app] starting; press q in the video window to quit", flush=True)
        self.voice.start()
        self.stats["voice_backend"] = self.voice.backend_name
        backend = self.voice.backend
        voice_name = getattr(backend, "voice", None)
        voice_info = f", voice: {voice_name}" if voice_name else ""
        print(
            f"[voice] backend: {self.voice.backend_name}{voice_info} "
            f"(enabled: {self.voice.enabled})",
            flush=True,
        )

        try:
            self.camera.start()
        except Exception as exc:  # noqa: BLE001
            print(f"[camera] {exc}", file=sys.stderr)
            self.voice.stop()
            return 1

        # Warm up on a real frame: this build of Ultralytics letterboxes to a
        # rectangular canvas, so the first frame of a new aspect ratio would
        # otherwise stall the loop for several seconds.
        probe = self.camera.read(timeout=5.0)
        if probe is not None:
            self.detector.warmup(probe.shape[:2])
        else:
            self.detector.warmup((cfg.camera_height, cfg.camera_width))

        if self.voice.enabled:
            # counts_as_gap=False: the greeting must not swallow the first real
            # announcement when an object is already in view at startup.
            self.voice.say("Assistive navigation ready.", counts_as_gap=False)

        self._open_outputs()
        started = time.perf_counter()
        last_frame_at = time.perf_counter()
        exit_code = 0
        pending = probe  # the frame used for warm-up is still a real frame

        try:
            while True:
                frame = pending if pending is not None else self.camera.read(timeout=2.0)
                pending = None
                if frame is None:
                    if self.camera.ended:
                        print("[camera] source finished", flush=True)
                        break
                    if time.perf_counter() - last_frame_at > cfg.camera_timeout:
                        print("[camera] no frames for too long; stopping", file=sys.stderr)
                        exit_code = 1
                        break
                    continue

                last_frame_at = time.perf_counter()
                annotated = self.process_frame(frame)
                self._update_fps()

                if self._writer is not None:
                    self._writer.write(annotated)
                elif cfg.record:
                    self._start_writer(annotated.shape)
                    if self._writer is not None:
                        self._writer.write(annotated)

                if cfg.snapshot_every and self.frame_index % cfg.snapshot_every == 0:
                    self.save_screenshot(annotated, reason="auto")

                if cfg.headless:
                    if cfg.max_frames and self.frame_index >= cfg.max_frames:
                        break
                    continue

                cv2.imshow(WINDOW, annotated)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                self.handle_key(key, annotated)

                if cfg.max_frames and self.frame_index >= cfg.max_frames:
                    break
        except KeyboardInterrupt:
            print("\n[app] interrupted", flush=True)
        finally:
            self.shutdown(started)

        return exit_code

    # -- per-frame pipeline -------------------------------------------------
    def process_frame(self, frame):
        """detect -> direction -> tracking -> speech -> overlay."""
        self.frame_index += 1
        height, width = frame.shape[:2]

        self._check_feed(frame)
        detections, latency_ms = self.detector.detect(frame)

        # 1. raw camera-relative zone for every box
        for det in detections:
            det.direction = self.estimator.estimate(det.box, width)

        # 1b. rough distance from box height ("about N metres", never exact)
        if self.distance is not None:
            for det in detections:
                det.distance_m = self.distance.estimate(
                    det.class_id, det.box, height, det.name
                )

        # 2. identity + lifecycle events (ByteTrack IDs, or IoU fallback IDs)
        events = self.tracker.update(detections, self.frame_index)

        # 3. smoothed zone, so a box on a divider cannot flicker
        for det in detections:
            key = det.track_id if det.track_id is not None else -(det.class_id + 1)
            det.direction_stable = self.smoother.update(key, det.direction)

        # 4. zone changes are only meaningful after the identity exists
        events.extend(self.tracker.note_directions(detections))

        # 4b. attach the same estimate to each event so speech can include it
        if self.distance is not None:
            for event in events:
                if event.box is not None:
                    event.distance_m = self.distance.estimate(
                        event.class_id, event.box, height, event.name
                    )

        # 5. speech, centre zone first
        for event in sorted(events, key=announcement_priority):
            if event.kind == "appeared":
                self.stats["appeared"] += 1
            elif event.kind == "direction_change":
                self.stats["direction_changes"] += 1
            elif event.kind == "disappeared":
                self.stats["disappeared"] += 1
                self.smoother.forget(event.track_id)
            if self.voice.announce(event):
                self.tracker.mark_spoken(event.track_id, event.direction)
                self.stats["announcements"] += 1
                self._last_spoken_at = time.perf_counter()

        # 6. bookkeeping + drawing
        self.stats["frames"] += 1
        self.stats["detections"] += len(detections)
        for det in detections:
            if det.track_id is not None:
                self._seen_track_ids.add(det.track_id)
        self.stats["unique_tracks"] = len(self._seen_track_ids)
        for det in detections:
            self.stats["per_class"][det.name] = self.stats["per_class"].get(det.name, 0) + 1
        if self._csv_writer is not None:
            for row in detections_to_rows(detections, self.frame_index):
                self._csv_writer.writerow(row)

        annotated = self.draw(frame, detections, latency_ms)
        return annotated

    # -- feed sanity --------------------------------------------------------
    def _check_feed(self, frame) -> None:
        """Warn once, out loud, when the camera is not delivering live video.

        Both failure modes - a black feed and a frozen placeholder - produce no
        detections, so without this the user hears the greeting and then
        nothing at all, which is indistinguishable from an empty room.
        """
        if self._feed_warning_shown or self.camera.is_video_file:
            return

        self._brightest_frame = max(self._brightest_frame, float(frame.mean()))
        if (
            self._previous_frame is not None
            and frame.shape == self._previous_frame.shape
            and np.array_equal(frame, self._previous_frame)
        ):
            self._identical_frames += 1
        else:
            self._identical_frames = 0
        # Held by reference: process_frame never mutates the captured frame.
        self._previous_frame = frame

        if self.frame_index < FEED_WARMUP_FRAMES:
            return

        if self._brightest_frame < BLACK_FEED_MEAN:
            self._warn_feed(
                "is sending black frames, so nothing can be detected",
                "Camera is not sending video.",
                "Check that the phone camera is streaming, the lens is not "
                "covered, and the privacy shutter is open. Try --camera laptop "
                "to check the pipeline.",
            )
        elif self._identical_frames >= FEED_WARMUP_FRAMES - 1:
            self._warn_feed(
                "is sending a still image, so the video is not updating",
                "Camera video is frozen.",
                "DroidCam is showing its placeholder. Open the DroidCam Client "
                "and wait for the phone preview, check the Wi-Fi link, or try "
                "--camera laptop.",
            )

    def _warn_feed(self, problem: str, spoken: str, hint: str) -> None:
        self._feed_warning_shown = True
        print(f"[camera] warning: {self.camera.label} {problem}. {hint}", flush=True)
        # Also spoken: a silent failure is the hardest kind to diagnose, and the
        # HUD is useless to someone who cannot see it.
        if self.voice.enabled:
            self.voice.say(spoken)

    # -- drawing ------------------------------------------------------------
    def _panel(self, frame, top_left, bottom_right, alpha: float = 0.55) -> None:
        x1, y1 = top_left
        x2, y2 = bottom_right
        x1, y1 = max(0, x1), max(0, y1)
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)
        if x2 <= x1 or y2 <= y1:
            return
        roi = frame[y1:y2, x1:x2]
        shade = roi.copy()
        shade[:] = (0, 0, 0)
        cv2.addWeighted(shade, alpha, roi, 1 - alpha, 0, roi)

    def draw(self, frame, detections: list[Detection], latency_ms: float):
        out = frame.copy()
        height, width = out.shape[:2]

        # zone dividers, so the user can see what "ahead" means
        for x, _ in self.estimator.zone_mask(width, height):
            cv2.line(out, (x, 0), (x, height), (70, 70, 70), 1)
        for label, x in (
            ("LEFT", int(width * self.estimator.left_ratio / 2)),
            ("AHEAD", int(width * (self.estimator.left_ratio + self.estimator.right_ratio) / 2)),
            ("RIGHT", int(width * (1 + self.estimator.right_ratio) / 2)),
        ):
            cv2.putText(out, label, (x - 22, height - 8), FONT, 0.45, DIM, 1, cv2.LINE_AA)

        for det in detections:
            self._draw_detection(out, det)

        self._draw_hud(out, detections, latency_ms)
        return out

    def _draw_detection(self, frame, det: Detection) -> None:
        direction = det.display_direction
        color = BOX_COLORS.get(direction, DEFAULT_COLOR) if direction else DEFAULT_COLOR
        x1, y1, x2, y2 = (int(v) for v in det.box.as_tuple)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        name = det.name if len(det.name) <= 24 else det.name[:22] + ".."
        parts = [name, f"{det.confidence:.2f}"]
        if det.distance_m is not None:
            parts.append(f"~{det.distance_m:g}m")
        if direction is not None:
            parts.append(direction.short)
        if det.track_id is not None:
            parts.append(f"#{det.track_id}")
        label = "  ".join(parts)

        scale = 0.5
        (tw, th), baseline = cv2.getTextSize(label, FONT, scale, 1)
        ty = y1 - 6 if y1 - th - 10 > 0 else y2 + th + 6
        self._panel(frame, (x1 - 2, ty - th - 4), (x1 + tw + 8, ty + baseline + 2), 0.65)
        cv2.putText(frame, label, (x1 + 2, ty), FONT, scale, color, 1, cv2.LINE_AA)

    def _draw_hud(self, frame, detections: list[Detection], latency_ms: float) -> None:
        width = frame.shape[1]
        voice_state = "ON" if self.voice.enabled else "OFF"
        if self.detector.tracking_error:
            tracking = "IoU (fallback)"
        elif self.detector.use_tracking:
            tracking = "ByteTrack"
        else:
            tracking = "IoU"

        lines = [
            f"FPS {self.fps:5.1f}   infer {latency_ms:5.1f} ms   objects {len(detections)}",
            f"conf {self.detector.conf:.2f}   tracking {tracking}   voice {voice_state}"
            f" ({self.voice.backend_name})",
            "q quit   v voice   +/- conf   s shot   m mute   t track",
        ]
        self._panel(frame, (6, 6), (min(width - 6, 620), 6 + 22 * len(lines) + 10), 0.6)
        for i, line in enumerate(lines):
            color = WHITE if i < 2 else DIM
            cv2.putText(frame, line, (14, 28 + i * 22), FONT, 0.52, color, 1, cv2.LINE_AA)

        # last announcement, so a hearing-impaired tester can follow the log
        spoken = self.voice.last_message
        if spoken and time.perf_counter() - self._last_spoken_at < 5.0:
            self._panel(frame, (6, frame.shape[0] - 78), (min(width - 6, 700), frame.shape[0] - 40), 0.6)
            cv2.putText(frame, f'> "{spoken}"', (14, frame.shape[0] - 52), FONT, 0.55,
                        (0, 255, 180), 1, cv2.LINE_AA)

        if self.camera.reconnects:
            cv2.putText(frame, f"camera reconnects: {self.camera.reconnects}",
                        (14, frame.shape[0] - 20), FONT, 0.45, (0, 0, 255), 1, cv2.LINE_AA)

    # -- helpers ------------------------------------------------------------
    def _update_fps(self) -> None:
        now = time.perf_counter()
        last = getattr(self, "_fps_last", None)
        self._fps_last = now
        if last is None:
            return
        dt = now - last
        if dt <= 0:
            return
        instant = 1.0 / dt
        self._fps_ema = instant if self._fps_ema is None else 0.9 * self._fps_ema + 0.1 * instant
        self.fps = self._fps_ema

    def handle_key(self, key: int, annotated) -> None:
        cfg = self.cfg
        if key == ord("v"):
            enabled = self.voice.toggle()
            print(f"[app] voice {'on' if enabled else 'off'}", flush=True)
        elif key in (ord("+"), ord("=")):
            print(f"[app] conf = {self.detector.set_conf(self.detector.conf + 0.05):.2f}", flush=True)
        elif key == ord("-"):
            print(f"[app] conf = {self.detector.set_conf(self.detector.conf - 0.05):.2f}", flush=True)
        elif key == ord("s"):
            self.save_screenshot(annotated, reason="manual")
        elif key == ord("m"):
            self.voice.stop_speaking()
            print("[app] stopped speaking", flush=True)
        elif key == ord("t"):
            self.detector.use_tracking = not self.detector.use_tracking
            self.stats["tracking"] = "bytetrack" if self.detector.use_tracking else "iou"
            print(f"[app] tracking {'on' if self.detector.use_tracking else 'off'}", flush=True)

    def save_screenshot(self, annotated, reason: str = "manual") -> Path | None:
        if not self.cfg.snapshot_dir:
            return None
        self.cfg.snapshot_dir.mkdir(parents=True, exist_ok=True)
        path = self.cfg.snapshot_dir / f"{reason}_{datetime.now():%Y%m%d_%H%M%S_%f}.jpg"
        cv2.imwrite(str(path), annotated)
        print(f"[shot] {path}", flush=True)
        return path

    def shutdown(self, started: float) -> None:
        elapsed = time.perf_counter() - started
        self.camera.stop()
        self.voice.stop()
        if self._writer is not None:
            self._writer.release()
            self._writer = None
        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = None
        if not self.cfg.headless:
            cv2.destroyAllWindows()

        self.stats["camera_reconnects"] = self.camera.reconnects
        self.stats["seconds"] = round(elapsed, 2)
        self.stats["avg_fps"] = round(self.stats["frames"] / elapsed, 2) if elapsed > 0 else 0.0
        self.stats["announcements_spoken"] = self.voice.spoken_count
        self.stats["announcements_dropped"] = self.voice.dropped_count
        self.stats["voice_backend"] = self.voice.backend_name
        self.stats["frames_read"] = self.camera.frames_read

        print(
            f"[app] {self.stats['frames']} frames in {elapsed:.1f}s "
            f"({self.stats['avg_fps']:.1f} FPS), "
            f"{self.stats['detections']} detections, "
            f"{self.stats['announcements']} announcements queued, "
            f"{self.voice.spoken_count} spoken",
            flush=True,
        )
        if self.cfg.stats_json:
            self.cfg.stats_json.parent.mkdir(parents=True, exist_ok=True)
            self.cfg.stats_json.write_text(
                json.dumps(self.stats, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            print(f"[app] stats -> {self.cfg.stats_json}", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Assistive navigation prototype (research only)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--names", type=Path, default=DEFAULT_NAMES)
    parser.add_argument(
        "--camera",
        default="0",
        help="camera index (0, 1, ...) or alias (laptop, droidcam, auto)",
    )
    parser.add_argument(
        "--camera-config",
        type=Path,
        default=DEFAULT_CAMERA_CONFIG,
        help="JSON file mapping aliases to indices",
    )
    parser.add_argument(
        "--camera-backend",
        default="auto",
        choices=["auto", "dshow", "msmf", "any", "v4l2", "avfoundation"],
        help="capture backend; 'auto' tries the platform order",
    )
    parser.add_argument(
        "--camera-scan-limit",
        type=int,
        default=6,
        help="indices 0..N-1 to probe when resolving an alias",
    )
    parser.add_argument("--source", default=None, help="video file instead of a camera")
    parser.add_argument("--conf", type=float, default=0.35, help="confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="inference size")
    parser.add_argument("--device", default=None, help="'0' for GPU, 'cpu' for CPU")
    parser.add_argument("--camera-width", type=int, default=1280)
    parser.add_argument("--camera-height", type=int, default=720)
    parser.add_argument("--no-track", action="store_true", help="disable ByteTrack")
    parser.add_argument("--max-missing", type=int, default=15, help="frames before a track is dropped")
    parser.add_argument("--stable-frames", type=int, default=3, help="frames before a zone change is accepted")
    parser.add_argument("--left-ratio", type=float, default=1 / 3, help="left zone edge")
    parser.add_argument("--right-ratio", type=float, default=2 / 3, help="right zone edge")
    parser.add_argument(
        "--object-heights",
        type=Path,
        default=DEFAULT_HEIGHTS_PATH,
        help="JSON of typical object heights (metres) for approximate distance",
    )
    parser.add_argument(
        "--vertical-fov",
        type=float,
        default=60.0,
        help="camera vertical field of view in degrees (default: 60)",
    )
    parser.add_argument(
        "--no-distance",
        action="store_true",
        help="do not estimate or speak distance",
    )

    voice = parser.add_argument_group("voice")
    voice.add_argument("--no-voice", action="store_true", help="start with speech disabled")
    voice.add_argument("--voice-backend", default="auto", choices=["auto", "pyttsx3", "sapi", "console"])
    voice.add_argument("--voice", default=None, help="substring of the voice name to use")
    voice.add_argument("--rate", type=int, default=175, help="speech rate (words per minute)")
    voice.add_argument("--volume", type=float, default=1.0, help="0.0 - 1.0")
    voice.add_argument("--cooldown", type=float, default=6.0, help="seconds before a track repeats")
    voice.add_argument("--min-gap", type=float, default=1.5, help="seconds between utterances")

    run = parser.add_argument_group("run")
    run.add_argument("--headless", action="store_true", help="no window (for tests)")
    run.add_argument("--max-frames", type=int, default=0, help="stop after N frames")
    run.add_argument("--camera-timeout", type=float, default=15.0)
    run.add_argument("--loop", action="store_true", help="loop a video file instead of stopping at the end")
    run.add_argument("--snapshot-dir", type=Path, default=PROJECT_ROOT / "checks" / "screenshots")
    run.add_argument("--snapshot-every", type=int, default=0, help="save every Nth frame")
    run.add_argument("--record", action="store_true", help="write an annotated MP4")
    run.add_argument("--log-csv", type=Path, default=None)
    run.add_argument("--stats-json", type=Path, default=None)
    run.add_argument("--list-classes", action="store_true", help="print the 35 names and exit")
    run.add_argument("--list-cameras", action="store_true", help="probe cameras, print indices and exit")
    run.add_argument("--list-voices", action="store_true", help="print available TTS voices and exit")
    return parser


def config_from_args(args: argparse.Namespace) -> AppConfig:
    return AppConfig(
        weights=args.weights,
        names=args.names,
        camera=args.camera,
        camera_config=args.camera_config,
        camera_backend=args.camera_backend,
        camera_scan_limit=args.camera_scan_limit,
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        camera_width=args.camera_width,
        camera_height=args.camera_height,
        use_tracking=not args.no_track,
        max_missing=args.max_missing,
        stable_frames=args.stable_frames,
        left_ratio=args.left_ratio,
        right_ratio=args.right_ratio,
        distance_enabled=not args.no_distance,
        object_heights=args.object_heights,
        vertical_fov=args.vertical_fov,
        voice_enabled=not args.no_voice,
        voice_backend=args.voice_backend,
        voice_hint=args.voice,
        rate=args.rate,
        volume=args.volume,
        cooldown=args.cooldown,
        min_gap=args.min_gap,
        headless=args.headless,
        max_frames=args.max_frames,
        camera_timeout=args.camera_timeout,
        loop_video=args.loop,
        snapshot_dir=args.snapshot_dir,
        snapshot_every=args.snapshot_every,
        record=args.record,
        stats_json=args.stats_json,
        log_csv=args.log_csv,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.list_classes:
        try:
            names = load_class_names(args.names)
        except Exception as exc:  # noqa: BLE001
            print(exc, file=sys.stderr)
            return 1
        for class_id in sorted(names):
            print(f"{class_id:2d}  {names[class_id]}")
        print(f"\n{len(names)} names from {args.names}")
        return 0

    if args.list_cameras:
        try:
            cameras = probe_cameras(
                width=args.camera_width,
                height=args.camera_height,
                max_index=args.camera_scan_limit,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[camera] {exc}", file=sys.stderr)
            return 1
        if not cameras:
            print(
                "No cameras detected.\n"
                "  - connect a webcam, or start DroidCam and wait for the phone preview\n"
                "  - close apps that may be holding the camera (Teams, Zoom, Camera)\n"
                "  - check Windows Settings > Privacy & security > Camera",
                file=sys.stderr,
            )
            return 1
        for info in cameras:
            print(info.describe())
        if not any(info.name for info in cameras):
            print(
                "\n(device names unavailable; install optional 'pygrabber' "
                "for name-based aliases, or use the indices above)"
            )
        print(f"\n{len(cameras)} camera(s) found.")
        print(
            "Use --camera <index>, or pin an alias in "
            f"{args.camera_config}:\n"
            '  {"aliases": {"droidcam": {"index": 2, "backend": "dshow"}}}'
        )
        return 0

    if args.list_voices:
        try:
            import pyttsx3

            engine = pyttsx3.init()
            for voice in engine.getProperty("voices"):
                print(f"{voice.id}\n    {voice.name}")
            engine.stop()
        except Exception as exc:  # noqa: BLE001
            print(f"pyttsx3 unavailable: {exc}", file=sys.stderr)
            return 1
        return 0

    cfg = config_from_args(args)
    try:
        app = AssistiveNavApp(cfg)
    except (CameraError, FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"[app] {exc}", file=sys.stderr)
        return 1
    return app.run()


if __name__ == "__main__":
    raise SystemExit(main())
