import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "report" / "PROJECT_REPORT.md"
NAMES = json.loads((ROOT / "class_names.json").read_text(encoding="utf-8"))
names = {int(k): v for k, v in NAMES.items() if k.isdigit() and v}

text = REPORT.read_text(encoding="utf-8")
out_lines = []
changed = 0
for line in text.split("\n"):
    m = re.match(r"^\| (\d+) \| class_\d\d \|", line)
    if m:
        cid = int(m.group(1))
        line = line.replace(f"| class_{cid:02d} |", f"| {names[cid]} |")
        changed += 1
    out_lines.append(line)

text = "\n".join(out_lines)
assert changed == 35, changed

old_intro = (
    "All 35 classes on the held-out test split (795 images, 1,220 boxes), ordered by ID. Precision, recall\n"
    "and F1 use the per-class F1-maximising confidence threshold; AP@50 and AP@50-95 integrate the curve."
)
new_intro = (
    "All 35 classes on the held-out test split (795 images, 1,220 boxes), ordered by ID. Precision, recall\n"
    "and F1 use the per-class F1-maximising confidence threshold; AP@50 and AP@50-95 integrate the curve.\n"
    "Names are the display labels recovered in §3.3; the ID remains the authoritative key."
)
assert old_intro in text
text = text.replace(old_intro, new_intro)

REPORT.write_text(text, encoding="utf-8", newline="\n")
print(f"rewrote {changed} rows; file is {len(text.splitlines())} lines")
