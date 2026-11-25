#!/usr/bin/env python3
"""
live_guide.py
=============

Demo assistive-navigation app: YOLO11 detection -> navigation brain -> voice.

The distance and motion are simulated by default so the feature can be shown
without extra hardware. For a real system, replace them with metric depth and
a tracker:

* distance: depth camera, LiDAR, or calibrated monocular depth
* motion: ByteTrack/DeepSORT centroid history across video frames

Examples
--------
Webcam with the COCO nano model (downloads ``yolo11n.pt`` on first run)::

    python assistive/live_guide.py

DroidCam virtual camera (uses Assistive-Navigation/config/cameras.json)::

    python assistive/live_guide.py --weights Assistive-Navigation/models/best.pt \
        --names class_names.json --source droidcam

Audio-free DroidCam check (prints [VOICE] instead of speaking)::

    python assistive/live_guide.py --weights Assistive-Navigation/models/best.pt \
        --names class_names.json --source droidcam \
        --voice-backend console --no-show --max-frames 60

Phone camera over Wi-Fi using an IP-camera app::

    python assistive/live_guide.py \
        --source http://192.168.1.15:8080/video \
        --motion-mode left \
        --position-mode ahead

Guaranteed demo sentence from one photo::

    python assistive/live_guide.py \
        --source path/to/person.jpg \
        --motion-mode left \
        --position-mode ahead

Use your trained 35-class checkpoint and recovered display names::

    python assistive/live_guide.py \
        --weights path/to/best.pt \
        --source 0 \
        --names class_names.json

Notes
-----
* All paths are relative to the project root unless you pass absolute ones.
* ``--no-voice`` prints the guidance without speaking (useful for testing).
"""

from __future__ import annotations

import os

# Some Windows Python installs ship more than one OpenMP runtime (NumPy,
# PyTorch, OpenCV). Allowing the duplicate keeps this demo from crashing at
# shutdown. This is a demo-only workaround, not a production fix.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import sys
from pathlib import Path

import cv2
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from assistive.navigation_brain import Detection, NavigationBrain  # noqa: E402
from assistive.voice import Speaker  # noqa: E402

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="YOLO11 demo with spoken navigation guidance.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--weights",
        default=None,
        help="Checkpoint path. Omit to use the newest runs/**/best.pt, or yolo11n.pt.",
    )
    parser.add_argument(
        "--source",
        default="0",
        help="Image, folder, video, URL, or webcam index such as 0 / webcam.",
    )
    parser.add_argument("--names", default=None, help="Optional class ID -> name JSON.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.45)
    parser.add_argument("--device", default="", help="0, cpu, mps; blank = auto.")
    parser.add_argument(
        "--distance",
        type=float,
        default=20.0,
        help="Simulated distance in metres used by the demo brain.",
    )
    parser.add_argument("--no-distance", action="store_true", help="Omit distance text.")
    parser.add_argument(
        "--motion-mode",
        choices=("auto", "left", "right", "none"),
        default="auto",
        help="'auto' needs video + tracking; left/right are demo overrides.",
    )
    parser.add_argument(
        "--position-mode",
        choices=("auto", "ahead", "left", "right", "none"),
        default="auto",
        help="'auto' uses box centre; the others are demo overrides.",
    )
    parser.add_argument("--cooldown", type=float, default=3.0, help="Seconds between repeats.")
    parser.add_argument("--rate", type=int, default=165, help="Speech rate (words/minute).")
    parser.add_argument(
        "--voice-backend",
        choices=("pyttsx3", "console"),
        default="pyttsx3",
        help="Speech engine. 'console' prints [VOICE] lines for audio-free testing.",
    )
    parser.add_argument(
        "--voice",
        default=None,
        help="Substring of a Windows voice name, e.g. Zira or David.",
    )
    parser.add_argument("--no-voice", action="store_true", help="Print only; do not speak.")
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open the preview window.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Stop after N frames (0 = unlimited). Useful for DroidCam checks.",
    )
    return parser


def resolve_path(raw: str) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    return path


def discover_weights() -> str:
    runs = PROJECT_ROOT / "runs"
    if runs.is_dir():
        candidates = sorted(
            runs.rglob("weights/best.pt"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if candidates:
            return str(candidates[0])
    return "yolo11n.pt"


def load_names(path: Path) -> dict[int, str]:
    raw = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(raw, dict) and "names" in raw:
        raw = raw["names"]
    return {int(k): str(v) for k, v in raw.items() if str(k).isdigit()}


def resolve_names(model: YOLO, explicit: str | None) -> dict[int, str]:
    if explicit:
        return load_names(resolve_path(explicit))

    custom = PROJECT_ROOT / "class_names.json"
    if custom.exists() and len(model.names) <= 35:
        return load_names(custom)

    return {int(k): str(v) for k, v in model.names.items()}


def camera_alias_index(alias: str) -> int | None:
    """Read a camera index from Assistive-Navigation/config/cameras.json."""

    config = PROJECT_ROOT / "Assistive-Navigation" / "config" / "cameras.json"
    if not config.exists():
        return None
    try:
        data = json.loads(config.read_text(encoding="utf-8-sig"))
        entry = data.get("aliases", {}).get(alias, {})
        index = entry.get("index")
        return int(index) if index is not None else None
    except (ValueError, TypeError, OSError):
        return None


def parse_source(raw: str) -> tuple[object, bool]:
    """Return ``(source, is_webcam)``."""

    text = raw.strip()
    if text.lower() in {"webcam", "camera"}:
        return 0, True

    lowered = text.lower()
    if lowered in {"droidcam", "laptop"}:
        index = camera_alias_index(lowered)
        if index is None:
            raise SystemExit(
                f"Camera alias {lowered!r} was not found. "
                "Start DroidCam and update Assistive-Navigation/config/cameras.json."
            )
        return index, True

    if text.isdigit():
        return int(text), True

    path = resolve_path(text)
    if path.exists():
        return str(path), False
    return text, False


def iter_frames(source: object, is_webcam: bool):
    """Yield frames from a webcam, image, video, or folder of images."""

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video/stream source: {source!r}")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield frame
            if is_webcam:
                continue
    finally:
        cap.release()


def iter_image_folder(folder: Path):
    files = sorted(
        p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not files:
        raise SystemExit(f"No images found in {folder}")
    for file in files:
        frame = cv2.imread(str(file))
        if frame is not None:
            yield frame


def iter_single_image(path: Path):
    frame = cv2.imread(str(path))
    if frame is None:
        raise SystemExit(f"Could not read image: {path}")
    yield frame


def draw_message(frame, message: str | None):
    if not message:
        return frame

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.7
    thickness = 2
    (tw, th), baseline = cv2.getTextSize(message, font, scale, thickness)
    x, y = 16, 34
    cv2.rectangle(
        frame,
        (x - 8, y - th - 10),
        (x + tw + 8, y + baseline + 6),
        (0, 0, 0),
        -1,
    )
    cv2.putText(frame, message, (x, y), font, scale, (255, 255, 255), thickness)
    return frame


def main() -> None:
    args = build_parser().parse_args()

    weights = args.weights or discover_weights()
    model = YOLO(weights)
    names = resolve_names(model, args.names)
    speaker = Speaker(
        rate=args.rate,
        enabled=not args.no_voice,
        backend=args.voice_backend,
        voice_hint=args.voice,
    )

    source, is_webcam = parse_source(args.source)
    source_path = Path(source) if isinstance(source, str) else None
    is_folder = bool(source_path and source_path.is_dir())

    frame_width = 640
    if is_webcam:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise SystemExit(
                f"Could not open webcam {source!r}. "
                "Close other camera apps and try --source 1."
            )
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        cap.release()
        frames = iter_frames(source, is_webcam=True)
    elif is_folder:
        frames = iter_image_folder(source_path)
    elif source_path and source_path.suffix.lower() in IMAGE_EXTENSIONS:
        frames = iter_single_image(source_path)
    else:
        frames = iter_frames(source, is_webcam=False)

    brain = NavigationBrain(
        frame_width,
        distance_m=None if args.no_distance else args.distance,
        motion_mode=args.motion_mode,
        position_mode=args.position_mode,
        min_conf=args.conf,
        cooldown_s=args.cooldown,
    )

    voice_state = "off" if args.no_voice else args.voice_backend
    print(f"weights : {weights}")
    print(f"source  : {args.source}")
    print(f"voice   : {voice_state}")
    print("Press q in the preview window to quit.\n")

    device = args.device or None
    seen_frames = False

    for frame_index, frame in enumerate(frames, start=1):
        seen_frames = True
        results = model.track(
            frame,
            imgsz=args.imgsz,
            conf=args.conf,
            persist=True,
            tracker="bytetrack.yaml",
            device=device,
            verbose=False,
        )[0]

        detections: list[Detection] = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
            cls_id = int(box.cls[0])
            track_id = int(box.id[0]) if box.id is not None else None
            detections.append(
                Detection(
                    name=names.get(cls_id, f"class_{cls_id:02d}"),
                    conf=float(box.conf[0]),
                    box=(x1, y1, x2, y2),
                    depth_m=args.distance if not args.no_distance else None,
                    track_id=track_id,
                )
            )

        message = brain.update(detections)
        if message:
            print(f"GUIDANCE: {message}")
            speaker.say(message)

        annotated = draw_message(results.plot(), message)
        if not args.no_show:
            cv2.imshow("Assistive Navigator", annotated)
            if cv2.waitKey(1 if is_webcam else 0) & 0xFF == ord("q"):
                break

        if args.max_frames and frame_index >= args.max_frames:
            break

    cv2.destroyAllWindows()

    if is_webcam and not seen_frames:
        raise SystemExit("Webcam opened but produced no frames. Try --source 1.")


if __name__ == "__main__":
    main()
