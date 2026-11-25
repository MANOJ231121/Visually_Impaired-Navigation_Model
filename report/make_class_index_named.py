"""Build the named class-index figure for the report.

One real example per class ID, cropped from its own label box in the read-only
source dataset, laid out as a 7 x 5 grid and labelled "ID name".  Nothing in the
source dataset is modified; the output goes to report/figures/.
"""

from __future__ import annotations

import csv
import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "final batches"
INVENTORY = ROOT / "dataset_class_analysis" / "visual_audit" / "class_visual_inventory.csv"
NAMES = ROOT / "model_training" / "class_names.json"
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)

COLS, ROWS = 7, 5


def load_names() -> dict[int, str]:
    data = json.loads(NAMES.read_text(encoding="utf-8"))
    return {int(k): v for k, v in data.items() if k.isdigit() and v}


def pick_rows() -> dict[int, dict]:
    """Prefer a medium-sized example, which is the most readable crop."""
    rows = list(csv.DictReader(INVENTORY.open(encoding="utf-8-sig")))
    by_class: dict[int, list[dict]] = {}
    for r in rows:
        by_class.setdefault(int(r["class_id"]), []).append(r)
    chosen = {}
    for cid, group in by_class.items():
        medium = [r for r in group if r["size_tier"] == "medium"]
        chosen[cid] = (medium or group)[0]
    return chosen


def crop_example(row: dict) -> Image.Image:
    """Crop the largest box of this class from the source image."""
    img_path = SOURCE / row["image_path"]
    lbl_path = SOURCE / row["label_path"]
    cid = int(row["class_id"])
    best = None
    for line in lbl_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5 or int(float(parts[0])) != cid:
            continue
        box = [float(v) for v in parts[1:]]
        if best is None or box[2] * box[3] > best[2] * best[3]:
            best = box
    with Image.open(img_path) as im:
        im = im.convert("RGB")
        W, H = im.size
        if best is None:
            return im
        cx, cy, w, h = best
        pad = 0.14
        x1 = int(max(0, (cx - w / 2 - w * pad) * W))
        y1 = int(max(0, (cy - h / 2 - h * pad) * H))
        x2 = int(min(W, (cx + w / 2 + w * pad) * W))
        y2 = int(min(H, (cy + h / 2 + h * pad) * H))
        return im.crop((x1, y1, max(x1 + 8, x2), max(y1 + 8, y2)))


def main() -> None:
    names = load_names()
    rows = pick_rows()

    fig_w, fig_h = 9.0, 6.9
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=240, facecolor="white")
    cell_w, cell_h = fig_w / COLS, fig_h / ROWS
    label_h = 0.40

    for i in range(35):
        col, row = i % COLS, i // COLS
        x0 = col * cell_w
        y0 = fig_h - (row + 1) * cell_h
        ax = fig.add_axes(
            [
                (x0 + 0.035) / fig_w,
                (y0 + label_h + 0.03) / fig_h,
                (cell_w - 0.07) / fig_w,
                (cell_h - label_h - 0.09) / fig_h,
            ]
        )
        try:
            im = crop_example(rows[i])
            ax.imshow(im)
        except Exception as exc:  # noqa: BLE001
            print(f"class {i}: {exc}")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#c9c9c9")
            spine.set_linewidth(0.6)

        name = names.get(i, "")
        wrapped = textwrap.wrap(name, width=16) or [""]
        label = f"{i}  {wrapped[0]}" + "".join(f"\n{w}" for w in wrapped[1:])
        fig.text(
            (x0 + cell_w / 2) / fig_w,
            (y0 + label_h - 0.02) / fig_h,
            label,
            ha="center",
            va="top",
            fontsize=10.4,
            color="#1c1c1c",
            linespacing=1.05,
        )

    png = OUT / "class_index_named.png"
    fig.savefig(png, facecolor="white")
    plt.close(fig)
    print(f"wrote {png} {Image.open(png).size}")


if __name__ == "__main__":
    main()
