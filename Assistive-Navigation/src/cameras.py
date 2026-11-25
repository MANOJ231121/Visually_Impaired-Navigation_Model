"""Camera discovery, alias resolution and low-latency capture opening.

Every supported source ends up as an OpenCV camera index. The detection,
tracking, direction and voice pipeline is deliberately unaware of *which*
camera produced a frame: ``detection.CameraStream`` wraps this module and hands
frames to the existing loop unchanged.

Selection rules
---------------
``--camera 0`` / ``--camera 2``
    An explicit OpenCV index. Nothing is probed and no device is opened here;
    the normal open path reports a useful error if the index is invalid.
``--camera laptop``
    A configured alias from ``config/cameras.json``, or a device-name match
    (for example "Integrated Camera"). If Windows device names are unavailable
    it falls back to index 0, preserving the historical default.
``--camera droidcam``
    A configured alias, or a device-name match (for example "DroidCam Source
    3"). The index is **never guessed**: if it cannot be identified, the error
    explains how to list cameras and pin the alias.
``--camera auto``
    The first camera that successfully returns a frame.

Alias configuration
-------------------
Windows device names are not exposed by OpenCV. If the optional ``pygrabber``
package is installed, this module uses it to read DirectShow names. When it is
not available, pin aliases explicitly::

    {
      "aliases": {
        "droidcam": {"index": 2, "backend": "dshow", "name": "DroidCam"}
      }
    }

Keys beginning with ``_`` are treated as comments, so the shipped example file
can stay in place without affecting behaviour.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CAMERA_CONFIG = PROJECT_ROOT / "config" / "cameras.json"

#: Human-readable backend names used in logs and in ``--camera-backend``.
BACKEND_LABELS = {
    "dshow": "DirectShow",
    "msmf": "Media Foundation",
    "any": "default",
    "v4l2": "V4L2",
    "avfoundation": "AVFoundation",
}


class CameraError(RuntimeError):
    """Raised when a camera cannot be selected or opened."""


@dataclass(frozen=True)
class CameraInfo:
    """One usable camera (or an explicitly requested index)."""

    index: int
    backend: str | None = None
    width: int = 0
    height: int = 0
    name: str = ""

    @property
    def label(self) -> str:
        """Short description used in logs, stats and error messages."""
        name = self.name or "camera"
        backend = f", {self.backend}" if self.backend else ""
        return f"{name} (index {self.index}{backend})"

    def describe(self) -> str:
        """One line for ``--list-cameras``."""
        name = self.name or "unnamed camera"
        size = f"  {self.width}x{self.height}" if self.width and self.height else ""
        backend = f"  [{self.backend}]" if self.backend else ""
        return f"{self.index}: {name}{size}{backend}"


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------


def _default_backend_order() -> tuple[str, ...]:
    if sys.platform.startswith("win"):
        return ("dshow", "msmf", "any")
    if sys.platform == "darwin":
        return ("avfoundation", "any")
    return ("v4l2", "any")


DEFAULT_BACKEND_ORDER = _default_backend_order()


def _backend_code(name: str) -> int:
    key = str(name).strip().lower()
    mapping = {
        "any": getattr(cv2, "CAP_ANY", 0),
        "dshow": getattr(cv2, "CAP_DSHOW", getattr(cv2, "CAP_ANY", 0)),
        "msmf": getattr(cv2, "CAP_MSMF", getattr(cv2, "CAP_ANY", 0)),
        "v4l2": getattr(cv2, "CAP_V4L2", getattr(cv2, "CAP_ANY", 0)),
        "avfoundation": getattr(cv2, "CAP_AVFOUNDATION", getattr(cv2, "CAP_ANY", 0)),
    }
    if key not in mapping:
        choices = ", ".join(sorted(BACKEND_LABELS))
        raise CameraError(f"Unknown camera backend '{name}'. Use one of: {choices}.")
    return mapping[key]


def normalize_backend(name: str | None) -> str | None:
    """Return a canonical backend name, or ``None`` for automatic choice."""
    if name is None:
        return None
    key = str(name).strip().lower()
    if key in ("", "auto"):
        return None
    if key not in BACKEND_LABELS:
        choices = ", ".join(sorted(BACKEND_LABELS))
        raise CameraError(f"Unknown camera backend '{name}'. Use one of: {choices}.")
    return key


def _default_capture(index: int, backend: str):
    return cv2.VideoCapture(int(index), _backend_code(backend))


# ---------------------------------------------------------------------------
# Spec parsing and config
# ---------------------------------------------------------------------------


def parse_camera_spec(spec: str | int | None) -> int | str:
    """Turn a CLI value into an integer index or a lower-case alias."""
    if isinstance(spec, bool):  # bool is an int subclass; reject it explicitly
        raise CameraError("Camera must be an index or an alias, not a boolean.")
    if isinstance(spec, int):
        if spec < 0:
            raise CameraError(f"Camera index must be >= 0, got {spec}.")
        return spec
    text = str(spec).strip() if spec is not None else ""
    if not text:
        raise CameraError("Empty camera value. Use an index (0, 1, ...) or an alias.")
    try:
        index = int(text)
    except ValueError:
        return text.lower()
    if index < 0:
        raise CameraError(f"Camera index must be >= 0, got {index}.")
    return index


def load_camera_config(path: str | Path | None = None) -> dict[str, object]:
    """Read alias definitions, ignoring keys that start with ``_``.

    Both layouts are accepted::

        {"aliases": {"droidcam": {"index": 2}}}
        {"droidcam": {"index": 2}}
    """
    config_path = Path(path) if path is not None else DEFAULT_CAMERA_CONFIG
    if not config_path.is_file():
        return {}
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CameraError(f"Could not read camera config {config_path}: {exc}") from exc
    if not isinstance(data, dict):
        raise CameraError(f"Camera config {config_path} must contain a JSON object.")

    aliases = data.get("aliases", data)
    if not isinstance(aliases, dict):
        raise CameraError(f"'aliases' in {config_path} must be a JSON object.")

    result: dict[str, object] = {}
    for key, value in aliases.items():
        if str(key).startswith("_"):
            continue
        result[str(key).strip().lower()] = value
    return result


def _entry_to_info(
    alias: str,
    entry: object,
    default_backend: str | None,
) -> CameraInfo:
    """Convert a config value (int, numeric string or object) to ``CameraInfo``."""
    name = alias
    entry_backend: str | None = None
    if isinstance(entry, bool):
        raise CameraError(f"Camera alias '{alias}' must be an index or an object.")
    if isinstance(entry, int):
        index = entry
    elif isinstance(entry, str):
        try:
            index = int(entry.strip())
        except ValueError as exc:
            raise CameraError(
                f"Camera alias '{alias}' must contain an integer index, got '{entry}'."
            ) from exc
    elif isinstance(entry, dict):
        if "index" not in entry:
            raise CameraError(f"Camera alias '{alias}' is missing the 'index' field.")
        try:
            index = int(entry["index"])
        except (TypeError, ValueError) as exc:
            raise CameraError(
                f"Camera alias '{alias}' has a non-integer 'index': {entry['index']!r}"
            ) from exc
        entry_backend = normalize_backend(entry.get("backend"))
        if entry.get("name"):
            name = str(entry["name"])
    else:
        raise CameraError(
            f"Camera alias '{alias}' must be an integer or an object, got {type(entry).__name__}."
        )

    if index < 0:
        raise CameraError(f"Camera alias '{alias}' has a negative index ({index}).")

    backend = default_backend if default_backend is not None else entry_backend
    return CameraInfo(index=index, backend=backend, name=name)


# ---------------------------------------------------------------------------
# Device names and probing
# ---------------------------------------------------------------------------


def directshow_names() -> list[str]:
    """Return DirectShow device names, or ``[]`` when pygrabber is absent.

    ``pygrabber`` is an optional convenience, not a required dependency. It is
    only imported when an alias actually needs name matching.
    """
    if not sys.platform.startswith("win"):
        return []
    try:
        from pygrabber.dshow_graph import FilterGraph
    except Exception:  # noqa: BLE001 - optional dependency and COM can fail
        return []
    try:
        return [str(name) for name in FilterGraph().get_input_devices()]
    except Exception:  # noqa: BLE001 - a broken DirectShow graph must not crash
        return []


def probe_cameras(
    *,
    indices: Iterable[int] | None = None,
    backends: Iterable[str] | None = None,
    width: int = 640,
    height: int = 480,
    max_index: int = 6,
    capture_factory: Callable[[int, str], object] | None = None,
    name_provider: Callable[[], Iterable[str]] | None = None,
) -> list[CameraInfo]:
    """Open each candidate index and keep the ones that return a frame.

    Every capture is released before the next attempt. Failures are expected
    while scanning (unused indices, cameras held by another app) and are
    silently skipped; the caller gets the list of usable cameras.
    """
    candidates = list(indices) if indices is not None else list(range(max(0, int(max_index))))
    backend_order = tuple(backends) if backends is not None else DEFAULT_BACKEND_ORDER
    factory = capture_factory or _default_capture
    name_fn = name_provider or directshow_names
    try:
        names = [str(name) for name in name_fn()]
    except Exception:  # noqa: BLE001 - name discovery is best-effort
        names = []

    found: list[CameraInfo] = []
    seen: set[int] = set()
    for index in candidates:
        index = int(index)
        if index < 0 or index in seen:
            continue
        for backend in backend_order:
            key = str(backend).strip().lower()
            if key == "auto":
                key = "any"
            cap = None
            try:
                cap = factory(index, key)
                if not cap.isOpened():
                    continue
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
                ok, frame = cap.read()
                if not ok or frame is None:
                    continue
                frame_height, frame_width = frame.shape[:2]
                name = names[index] if index < len(names) else ""
                found.append(
                    CameraInfo(
                        index=index,
                        backend=key,
                        width=int(frame_width),
                        height=int(frame_height),
                        name=name,
                    )
                )
                seen.add(index)
                break
            except Exception:  # noqa: BLE001 - one bad index must not stop the scan
                continue
            finally:
                if cap is not None:
                    try:
                        cap.release()
                    except Exception:  # noqa: BLE001
                        pass
    return found


# ---------------------------------------------------------------------------
# Alias resolution
# ---------------------------------------------------------------------------


#: Name fragments used when pygrabber is available but no alias is configured.
ALIAS_PATTERNS = {
    "laptop": (
        "integrated",
        "built-in",
        "builtin",
        "internal",
        "laptop",
        "webcam",
        "facetime",
        "hd camera",
    ),
    "webcam": (
        "integrated",
        "built-in",
        "builtin",
        "internal",
        "laptop",
        "webcam",
        "facetime",
        "hd camera",
    ),
    "droidcam": ("droidcam", "droid cam", "droid", "android", "phone", "mobile"),
    "phone": (
        "droidcam",
        "droid cam",
        "droid",
        "android",
        "phone",
        "mobile",
    ),
}


def _no_cameras_message() -> str:
    return (
        "No cameras were detected.\n"
        "  - connect a webcam, or start DroidCam and wait for the phone preview\n"
        "  - run: python src/main.py --list-cameras\n"
        "  - close apps that may be holding the camera (Teams, Zoom, Camera)\n"
        "  - or use a video file: --source path\\to\\video.mp4"
    )


def _alias_not_found_message(
    alias: str,
    infos: list[CameraInfo],
    config_path: str | Path,
) -> str:
    detected = "\n".join(f"  - {info.describe()}" for info in infos) or "  (none)"
    lines = [
        f"Camera alias '{alias}' was not detected.",
        "Detected cameras:",
        detected,
        "Run `python src/main.py --list-cameras` and either:",
        "  - pass the index directly: --camera 2",
        f"  - or pin the alias in {config_path}:",
        '      {"aliases": {"' + alias + '": {"index": 2, "backend": "dshow"}}}',
    ]
    if alias in ("droidcam", "phone"):
        lines += [
            "DroidCam checklist:",
            "  - open the DroidCam Client and wait for the phone preview",
            "  - keep the phone and laptop on the same Wi-Fi network",
            "  - confirm the client shows the camera as connected",
        ]
    return "\n".join(lines)


def resolve_camera(
    spec: str | int | None = None,
    *,
    config_path: str | Path | None = DEFAULT_CAMERA_CONFIG,
    backend: str | None = None,
    width: int = 1280,
    height: int = 720,
    max_index: int = 6,
    probe_fn: Callable[..., list[CameraInfo]] | None = None,
    name_provider: Callable[[], Iterable[str]] | None = None,
) -> CameraInfo:
    """Resolve a CLI camera value to a concrete index and backend.

    Explicit indices are returned immediately. Aliases use, in order: the JSON
    config, a DirectShow device-name match, then a live probe. ``droidcam`` is
    never mapped to a guessed index.
    """
    default_backend = normalize_backend(backend)
    parsed = parse_camera_spec("0" if spec is None else spec)

    if isinstance(parsed, int):
        return CameraInfo(index=parsed, backend=default_backend, width=width, height=height)

    alias = parsed
    aliases = load_camera_config(config_path)
    if alias in aliases:
        return _entry_to_info(alias, aliases[alias], default_backend)

    patterns = ALIAS_PATTERNS.get(alias)
    if patterns is None and alias not in ("auto", "first"):
        known = sorted(set(ALIAS_PATTERNS) | set(aliases) | {"auto"})
        raise CameraError(
            f"Unknown camera '{alias}'. Use an index, one of {known}, "
            f"or add an alias to {config_path}."
        )

    # Name matching first: it avoids opening the DroidCam virtual camera twice
    # (once to identify it, once for the stream). DirectShow names are only
    # available through the optional pygrabber package.
    name_fn = name_provider or directshow_names
    try:
        names = [str(name) for name in name_fn()]
    except Exception:  # noqa: BLE001 - name discovery is best-effort
        names = []

    if patterns is not None and names:
        for index, name in enumerate(names):
            lowered = name.lower()
            if any(pattern in lowered for pattern in patterns):
                # Do not force DirectShow here: a DirectShow index can fail to
                # open while the same device is available through Media
                # Foundation (common with virtual cameras such as DroidCam).
                # `open_camera` will try the platform order unless the user
                # explicitly pinned a backend.
                return CameraInfo(
                    index=index,
                    backend=default_backend,
                    width=width,
                    height=height,
                    name=name,
                )

    probe = probe_fn or probe_cameras
    infos = probe(
        width=width,
        height=height,
        max_index=max_index,
        name_provider=lambda: names,
    )

    if alias in ("auto", "first"):
        if not infos:
            raise CameraError(_no_cameras_message())
        first = infos[0]
        return CameraInfo(
            index=first.index,
            backend=default_backend or first.backend,
            width=first.width or width,
            height=first.height or height,
            name=first.name,
        )

    assert patterns is not None
    for info in infos:
        lowered = info.name.lower()
        if any(pattern in lowered for pattern in patterns):
            return CameraInfo(
                index=info.index,
                backend=default_backend or info.backend,
                width=info.width or width,
                height=info.height or height,
                name=info.name,
            )

    # Historical behaviour: index 0 is the laptop webcam. Only use it when the
    # system exposes no device names at all, so we cannot accidentally select a
    # named DroidCam that happens to be index 0.
    if alias in ("laptop", "webcam") and not names:
        available = {info.index for info in infos}
        if 0 in available:
            print(
                "[camera] device names unavailable; assuming index 0 for 'laptop' "
                "(use --camera <index> to override)",
                flush=True,
            )
            return CameraInfo(
                index=0,
                backend=default_backend,
                width=width,
                height=height,
                name="laptop (assumed)",
            )

    if not infos and not names:
        raise CameraError(_no_cameras_message())
    raise CameraError(_alias_not_found_message(alias, infos, config_path))


# ---------------------------------------------------------------------------
# Opening
# ---------------------------------------------------------------------------


def _camera_error_message(
    index: int,
    label: str | None,
    backend: str | None,
    errors: list[str],
) -> str:
    name = label or f"index {index}"
    backend_text = f", backend {backend}" if backend else ""
    lines = [
        f"Could not open camera {name} (index {index}{backend_text}).",
        "Tried: " + ("; ".join(errors) if errors else "no attempts"),
        "",
        "  - close other apps that use the camera (Teams, Zoom, Camera)",
        "  - check Windows Settings > Privacy & security > Camera",
        "  - run: python src/main.py --list-cameras",
        "  - pass an explicit index: --camera 0, --camera 1, ...",
    ]
    if "droid" in (label or "").lower() or "phone" in (label or "").lower():
        lines += [
            "  - DroidCam: open the DroidCam Client and wait for the phone preview",
            "  - keep the phone and laptop on the same Wi-Fi network",
            "  - confirm the client shows the camera as connected",
            "  - close other apps that may be holding the virtual camera",
        ]
    else:
        lines.append("  - or use a video file: --source path\\to\\video.mp4")
    return "\n".join(lines)


def open_camera(
    index: int = 0,
    width: int = 1280,
    height: int = 720,
    fps: int = 30,
    backend: str | None = None,
    label: str | None = None,
    capture_factory: Callable[[int, str], object] | None = None,
    first_frame_timeout: float = 1.5,
) -> cv2.VideoCapture:
    """Open a camera and return only after it has produced a frame.

    When ``backend`` is given, only that backend is tried. Otherwise the
    platform order (DirectShow, Media Foundation, default on Windows) is used.
    Buffering is reduced to one frame where the driver supports it, and a
    short first-frame wait gives virtual cameras such as DroidCam time to start.
    """
    try:
        index = int(index)
    except (TypeError, ValueError) as exc:
        raise CameraError(f"Camera index must be an integer, got {index!r}.") from exc
    if index < 0:
        raise CameraError(f"Camera index must be >= 0, got {index}.")

    factory = capture_factory or _default_capture
    preferred = normalize_backend(backend)
    if preferred is not None:
        attempts = [(preferred, BACKEND_LABELS.get(preferred, preferred))]
    else:
        attempts = [(key, BACKEND_LABELS.get(key, key)) for key in DEFAULT_BACKEND_ORDER]

    errors: list[str] = []
    for key, backend_label in attempts:
        cap = None
        try:
            cap = factory(index, key)
            if not cap.isOpened():
                errors.append(f"{backend_label}: not opened")
                try:
                    cap.release()
                except Exception:  # noqa: BLE001
                    pass
                cap = None
                continue

            # One-frame buffer keeps live video close to real time. Drivers are
            # allowed to ignore this property; failure is not an error.
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
            cap.set(cv2.CAP_PROP_FPS, int(fps))

            deadline = time.monotonic() + max(0.0, float(first_frame_timeout))
            ok = False
            frame = None
            while True:
                ok, frame = cap.read()
                if ok and frame is not None:
                    break
                if time.monotonic() >= deadline:
                    break
                time.sleep(0.1)

            if ok and frame is not None:
                actual_height, actual_width = frame.shape[:2]
                display = label or f"camera (index {index})"
                print(
                    f"[camera] {display} opened via {backend_label} "
                    f"({actual_width}x{actual_height})",
                    flush=True,
                )
                return cap

            errors.append(f"{backend_label}: opened but no frame")
        except Exception as exc:  # noqa: BLE001 - try the next backend
            errors.append(f"{backend_label}: {exc}")
        if cap is not None:
            try:
                cap.release()
            except Exception:  # noqa: BLE001
                pass

    raise CameraError(_camera_error_message(index, label, preferred, errors))
