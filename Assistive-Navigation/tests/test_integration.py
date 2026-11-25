"""End-to-end test on the synthetic scenario clip.

Skipped automatically when the trained weights or the clip are missing.
"""

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from main import main as run_app  # noqa: E402

WEIGHTS = ROOT / "models" / "best.pt"
SCENARIO = ROOT / "checks" / "scenario.mp4"


@unittest.skipUnless(WEIGHTS.is_file(), "trained weights not present")
@unittest.skipUnless(SCENARIO.is_file(), "scenario clip not present")
class PipelineTests(unittest.TestCase):
    def test_full_pipeline_on_scenario(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            stats_path = tmp_path / "stats.json"
            csv_path = tmp_path / "dets.csv"

            code = run_app(
                [
                    "--source", str(SCENARIO),
                    "--headless",
                    "--voice-backend", "console",
                    "--cooldown", "0",
                    "--min-gap", "0",
                    "--max-frames", "150",
                    "--stats-json", str(stats_path),
                    "--log-csv", str(csv_path),
                    "--snapshot-dir", str(tmp_path / "shots"),
                ]
            )
            self.assertEqual(code, 0)
            self.assertTrue(stats_path.is_file())
            self.assertTrue(csv_path.is_file())

            stats = json.loads(stats_path.read_text(encoding="utf-8"))
            self.assertEqual(stats["frames"], 150)
            self.assertGreaterEqual(stats["appeared"], 4)
            self.assertGreaterEqual(stats["disappeared"], 1)
            self.assertGreaterEqual(stats["direction_changes"], 1)
            self.assertGreaterEqual(stats["unique_tracks"], 3)
            # Sequential mode must not race ahead of the consumer.
            self.assertLessEqual(stats["frames_read"], 160)

            with csv_path.open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))

        self.assertTrue(
            {"chair", "book", "mobile phone"} <= {r["name"] for r in rows},
            "the scenario objects should all be detected",
        )

        chair_directions = {r["direction"] for r in rows if r["name"] == "chair"}
        self.assertEqual(chair_directions, {"left", "center", "right"})

        phone_frames = sorted(
            {int(r["frame"]) for r in rows if r["name"] == "mobile phone"}
        )
        self.assertGreater(len(phone_frames), 10)
        gaps = [b - a for a, b in zip(phone_frames, phone_frames[1:])]
        self.assertGreater(max(gaps), 15, "the phone should disappear then return")


if __name__ == "__main__":
    unittest.main()
