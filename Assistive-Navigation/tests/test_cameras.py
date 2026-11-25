"""Tests for camera discovery, alias resolution and capture opening.

No real camera is opened: every test injects a fake capture factory or a fake
probe. The tests therefore run on CI machines without a webcam.
"""

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

import detection  # noqa: E402
from cameras import (  # noqa: E402
    CameraError,
    CameraInfo,
    load_camera_config,
    normalize_backend,
    open_camera,
    parse_camera_spec,
    probe_cameras,
    resolve_camera,
)


FRAME = np.full((48, 64, 3), 127, np.uint8)


class FakeCapture:
    """Minimal stand-in for ``cv2.VideoCapture``."""

    def __init__(self, frame=None, opened: bool = True) -> None:
        self.frame = frame
        self.opened = opened
        self.released = False
        self.sets: list[tuple[int, float]] = []

    def isOpened(self) -> bool:  # noqa: N802 - OpenCV naming
        return self.opened

    def read(self):
        if self.frame is None:
            return False, None
        return True, self.frame

    def set(self, prop: int, value) -> bool:
        self.sets.append((prop, value))
        return True

    def release(self) -> None:
        self.released = True


class ParseTests(unittest.TestCase):
    def test_indices(self) -> None:
        self.assertEqual(parse_camera_spec(0), 0)
        self.assertEqual(parse_camera_spec("2"), 2)
        self.assertEqual(parse_camera_spec(" 03 "), 3)

    def test_aliases_are_lower_case(self) -> None:
        self.assertEqual(parse_camera_spec(" DroidCam "), "droidcam")
        self.assertEqual(parse_camera_spec("laptop"), "laptop")

    def test_invalid(self) -> None:
        with self.assertRaises(CameraError):
            parse_camera_spec(None)
        with self.assertRaises(CameraError):
            parse_camera_spec("")
        with self.assertRaises(CameraError):
            parse_camera_spec(-1)
        with self.assertRaises(CameraError):
            parse_camera_spec(True)

    def test_backend_normalisation(self) -> None:
        self.assertIsNone(normalize_backend(None))
        self.assertIsNone(normalize_backend("auto"))
        self.assertEqual(normalize_backend("DSHOW"), "dshow")
        with self.assertRaises(CameraError):
            normalize_backend("telepathy")


class ConfigTests(unittest.TestCase):
    def test_missing_file_is_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(load_camera_config(Path(tmp) / "nope.json"), {})

    def test_aliases_object(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cameras.json"
            path.write_text(
                '{"_comment": "ignored", "aliases": {"DroidCam": {"index": 2}}}',
                encoding="utf-8",
            )
            config = load_camera_config(path)
        self.assertIn("droidcam", config)
        self.assertNotIn("_comment", config)

    def test_top_level_aliases(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cameras.json"
            path.write_text('{"laptop": 0, "_note": "ignored"}', encoding="utf-8")
            config = load_camera_config(path)
        self.assertEqual(config, {"laptop": 0})

    def test_invalid_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cameras.json"
            path.write_text("{not json", encoding="utf-8")
            with self.assertRaises(CameraError):
                load_camera_config(path)

    def test_missing_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cameras.json"
            path.write_text('{"aliases": {"phone": {"backend": "dshow"}}}', encoding="utf-8")
            with self.assertRaises(CameraError):
                resolve_camera("phone", config_path=path, probe_fn=lambda **_: [])


class ResolveTests(unittest.TestCase):
    def test_explicit_index_does_not_probe(self) -> None:
        def boom(**_kwargs):
            raise AssertionError("explicit indices must not be probed")

        info = resolve_camera("3", probe_fn=boom)
        self.assertEqual(info.index, 3)
        self.assertIsNone(info.backend)

    def test_explicit_index_keeps_backend_override(self) -> None:
        info = resolve_camera(4, backend="dshow", probe_fn=lambda **_: [])
        self.assertEqual(info.index, 4)
        self.assertEqual(info.backend, "dshow")

    def test_config_alias_wins(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cameras.json"
            path.write_text(
                '{"aliases": {"droidcam": '
                '{"index": 2, "backend": "dshow", "name": "DroidCam"}}}',
                encoding="utf-8",
            )
            info = resolve_camera(
                "droidcam",
                config_path=path,
                probe_fn=lambda **_: (_ for _ in ()).throw(
                    AssertionError("configured aliases must not be probed")
                ),
            )
        self.assertEqual(info.index, 2)
        self.assertEqual(info.backend, "dshow")
        self.assertEqual(info.name, "DroidCam")

    def test_cli_backend_overrides_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cameras.json"
            path.write_text(
                '{"aliases": {"droidcam": {"index": 2, "backend": "dshow"}}}',
                encoding="utf-8",
            )
            info = resolve_camera("droidcam", config_path=path, backend="msmf")
        self.assertEqual(info.backend, "msmf")

    def test_alias_matches_device_name_without_probe(self) -> None:
        def boom(**_kwargs):
            raise AssertionError("a name match should avoid the probe")

        info = resolve_camera(
            "droidcam",
            config_path=Path("does-not-exist.json"),
            name_provider=lambda: ["Integrated Camera", "DroidCam Source 3"],
            probe_fn=boom,
        )
        self.assertEqual(info.index, 1)
        self.assertEqual(info.name, "DroidCam Source 3")

    def test_laptop_alias_matches_integrated_camera(self) -> None:
        info = resolve_camera(
            "laptop",
            config_path=Path("does-not-exist.json"),
            name_provider=lambda: ["Integrated Camera", "DroidCam Source 3"],
            probe_fn=lambda **_: [],
        )
        self.assertEqual(info.index, 0)

    def test_droidcam_alias_matches_droidcam_device(self) -> None:
        info = resolve_camera(
            "droidcam",
            config_path=Path("does-not-exist.json"),
            name_provider=lambda: ["Integrated Camera", "DroidCam Source 3"],
            probe_fn=lambda **_: [],
        )
        self.assertEqual(info.index, 1)
        self.assertEqual(info.name, "DroidCam Source 3")

    def test_phone_alias_accepts_droidcam(self) -> None:
        info = resolve_camera(
            "phone",
            config_path=Path("does-not-exist.json"),
            name_provider=lambda: ["Integrated Camera", "DroidCam Source 3"],
            probe_fn=lambda **_: [],
        )
        self.assertEqual(info.index, 1)

    def test_auto_uses_first_probed_camera(self) -> None:
        probed = [
            CameraInfo(2, "dshow", 640, 480, "DroidCam Source 3"),
            CameraInfo(0, "msmf", 1280, 720, "Integrated Camera"),
        ]
        info = resolve_camera(
            "auto",
            config_path=Path("does-not-exist.json"),
            name_provider=lambda: [],
            probe_fn=lambda **_: probed,
        )
        self.assertEqual(info.index, 2)
        self.assertEqual(info.backend, "dshow")

    def test_auto_without_cameras_has_hint(self) -> None:
        with self.assertRaises(CameraError) as ctx:
            resolve_camera(
                "auto",
                config_path=Path("does-not-exist.json"),
                name_provider=lambda: [],
                probe_fn=lambda **_: [],
            )
        self.assertIn("No cameras were detected", str(ctx.exception))
        self.assertIn("--list-cameras", str(ctx.exception))

    def test_laptop_falls_back_to_zero_only_without_names(self) -> None:
        info = resolve_camera(
            "laptop",
            config_path=Path("does-not-exist.json"),
            name_provider=lambda: [],
            probe_fn=lambda **_: [CameraInfo(0, "dshow", 640, 480, "")],
        )
        self.assertEqual(info.index, 0)
        self.assertIn("assumed", info.name)

    def test_droidcam_never_guesses_an_index(self) -> None:
        with self.assertRaises(CameraError) as ctx:
            resolve_camera(
                "droidcam",
                config_path=Path("does-not-exist.json"),
                name_provider=lambda: [],
                probe_fn=lambda **_: [CameraInfo(0, "dshow", 640, 480, "")],
            )
        message = str(ctx.exception)
        self.assertIn("was not detected", message)
        self.assertIn("DroidCam checklist", message)
        self.assertIn("--camera 2", message)

    def test_unknown_alias(self) -> None:
        with self.assertRaises(CameraError) as ctx:
            resolve_camera(
                "toaster",
                config_path=Path("does-not-exist.json"),
                probe_fn=lambda **_: [],
            )
        self.assertIn("Unknown camera", str(ctx.exception))


class ProbeTests(unittest.TestCase):
    def test_probe_dedupes_and_releases(self) -> None:
        created: list[tuple[int, str, FakeCapture]] = []

        def factory(index: int, backend: str):
            frame = None
            opened = True
            if index == 0 and backend == "msmf":
                frame = FRAME
            elif index == 2 and backend == "dshow":
                frame = FRAME
            elif index == 0 and backend == "dshow":
                frame = None  # opened but no frame
            else:
                opened = False
            cap = FakeCapture(frame=frame, opened=opened)
            created.append((index, backend, cap))
            return cap

        found = probe_cameras(
            indices=[0, 1, 2],
            backends=["dshow", "msmf", "any"],
            capture_factory=factory,
            name_provider=lambda: ["Integrated Camera", "Other", "DroidCam Source 3"],
        )

        self.assertEqual([info.index for info in found], [0, 2])
        self.assertEqual([info.backend for info in found], ["msmf", "dshow"])
        self.assertEqual(found[0].name, "Integrated Camera")
        self.assertEqual(found[1].name, "DroidCam Source 3")
        self.assertEqual(found[0].width, 64)
        self.assertTrue(all(cap.released for _, _, cap in created))

    def test_probe_skips_a_factory_that_raises(self) -> None:
        def factory(index: int, backend: str):
            if index == 0:
                raise RuntimeError("driver exploded")
            return FakeCapture(frame=FRAME)

        found = probe_cameras(
            indices=[0, 1],
            backends=["dshow"],
            capture_factory=factory,
            name_provider=lambda: [],
        )
        self.assertEqual([info.index for info in found], [1])


class OpenCameraTests(unittest.TestCase):
    def test_preferred_backend_is_the_only_attempt(self) -> None:
        calls: list[tuple[int, str]] = []

        def factory(index: int, backend: str):
            calls.append((index, backend))
            return FakeCapture(frame=FRAME)

        cap = open_camera(
            2,
            1280,
            720,
            30,
            backend="dshow",
            label="DroidCam",
            capture_factory=factory,
            first_frame_timeout=0,
        )
        self.assertIsInstance(cap, FakeCapture)
        self.assertEqual(calls, [(2, "dshow")])
        self.assertFalse(cap.released)
        properties = [prop for prop, _ in cap.sets]
        self.assertIn(cv2.CAP_PROP_BUFFERSIZE, properties)
        self.assertIn(cv2.CAP_PROP_FRAME_WIDTH, properties)
        self.assertIn(cv2.CAP_PROP_FRAME_HEIGHT, properties)

    def test_auto_falls_through_backends(self) -> None:
        calls: list[str] = []

        def factory(index: int, backend: str):
            calls.append(backend)
            if backend == "dshow":
                return FakeCapture(opened=False)
            return FakeCapture(frame=FRAME)

        cap = open_camera(
            0,
            backend="auto",
            capture_factory=factory,
            first_frame_timeout=0,
        )
        self.assertIsInstance(cap, FakeCapture)
        self.assertEqual(calls, ["dshow", "msmf"])

    def test_failure_message_mentions_droidcam_and_listing(self) -> None:
        created: list[FakeCapture] = []

        def factory(_index: int, _backend: str):
            cap = FakeCapture(opened=False)
            created.append(cap)
            return cap

        with self.assertRaises(CameraError) as ctx:
            open_camera(
                3,
                backend="auto",
                label="DroidCam",
                capture_factory=factory,
                first_frame_timeout=0,
            )
        message = str(ctx.exception)
        self.assertIn("Could not open camera DroidCam", message)
        self.assertIn("--list-cameras", message)
        self.assertIn("same Wi-Fi", message)
        self.assertTrue(created)
        self.assertTrue(all(cap.released for cap in created))


class CameraStreamWiringTests(unittest.TestCase):
    def test_backend_and_label_are_forwarded(self) -> None:
        calls: list[tuple[tuple, dict]] = []
        original = detection.open_camera

        def fake_open(*args, **kwargs):
            calls.append((args, kwargs))
            return FakeCapture(frame=FRAME)

        detection.open_camera = fake_open
        try:
            stream = detection.CameraStream(
                source=2,
                width=640,
                height=480,
                backend="dshow",
                label="DroidCam Source 3",
            )
            cap = stream._open()
        finally:
            detection.open_camera = original

        self.assertIsInstance(cap, FakeCapture)
        args, kwargs = calls[0]
        self.assertEqual(args, (2, 640, 480, 30))
        self.assertEqual(kwargs["backend"], "dshow")
        self.assertEqual(kwargs["label"], "DroidCam Source 3")
        self.assertEqual(stream.label, "DroidCam Source 3")

    def test_default_label_for_index(self) -> None:
        stream = detection.CameraStream(source=1)
        self.assertEqual(stream.label, "camera 1")


if __name__ == "__main__":
    unittest.main()
