"""Build the three report figures from existing audit artefacts.

Reads only; every source image and CSV stays untouched. Outputs go to
report/figures/ and are referenced from PROJECT_REPORT.md.
"""

from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image, ImageChops
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)

CSV = ROOT / "dataset_class_analysis" / "class_distribution.csv"
CLASS_INDEX = Path(__file__).resolve().parent / "figures" / "class_index_named.png"
CONFLICT = (
    ROOT
    / "dataset_verification"
    / "visualizations"
    / "class_conflicts"
    / "C01_7911_376__itmbghqk.png"
)


def trim_white(path: Path, pad: int = 8) -> Image.Image:
    """Crop a near-white border so the picture does not waste page space."""
    im = Image.open(path).convert("RGB")
    bg = Image.new("RGB", im.size, (255, 255, 255))
    diff = ImageChops.difference(im, bg).convert("L")
    box = diff.point(lambda v: 255 if v > 12 else 0).getbbox()
    if box is None:
        return im
    left, top, right, bottom = box
    left = max(0, left - pad)
    top = max(0, top - pad)
    right = min(im.width, right + pad)
    bottom = min(im.height, bottom + pad)
    return im.crop((left, top, right, bottom))


def distribution() -> None:
    rows = list(csv.DictReader(CSV.open(encoding="utf-8-sig")))
    rows.sort(key=lambda r: int(r["class_id"]))
    ids = [int(r["class_id"]) for r in rows]
    train = np.array([int(r["train_boxes"]) for r in rows])
    val = np.array([int(r["val_boxes"]) for r in rows])
    test = np.array([int(r["test_boxes"]) for r in rows])
    total = train + val + test

    fig, ax = plt.subplots(figsize=(9.0, 3.5), dpi=220)
    ax.bar(ids, train, width=0.72, color="#2f5d8a", label="train", linewidth=0)
    ax.bar(ids, val, width=0.72, bottom=train, color="#5aa9c9", label="validation", linewidth=0)
    ax.bar(ids, test, width=0.72, bottom=train + val, color="#e2a33c", label="test", linewidth=0)

    mean = total.mean()
    ax.axhline(mean, color="#8a2d2d", linestyle=(0, (5, 3)), linewidth=1.1)
    ax.text(34.6, mean + 14, f"mean {mean:.0f}", ha="right", va="bottom",
            fontsize=7.5, color="#8a2d2d")

    ax.annotate(f"class {int(total.argmax())}: {total.max()} boxes",
                xy=(int(total.argmax()), total.max()), xytext=(0.6, 985),
                ha="left", va="bottom", fontsize=7.5, color="#1c1c1c",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none"),
                arrowprops=dict(arrowstyle="-", linewidth=0.8, color="#555555"))

    ax.set_xlim(-0.8, 34.8)
    ax.set_ylim(0, 1120)
    ax.set_xticks(ids)
    ax.set_xticklabels([str(i) for i in ids], fontsize=7)
    ax.set_xlabel("class ID", fontsize=8.5)
    ax.set_ylabel("bounding boxes", fontsize=8.5)
    ax.tick_params(axis="y", labelsize=7.5)
    ax.grid(axis="y", color="#dddddd", linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="upper right", ncol=3, fontsize=7.5, frameon=False,
              handlelength=1.2, columnspacing=1.4)

    fig.tight_layout(pad=0.3)
    path = OUT / "fig_class_distribution.png"
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    print(f"{path.name}: {Image.open(path).size}")


def overview() -> None:
    im = trim_white(CLASS_INDEX)
    scale = 1800 / im.width
    im = im.resize((1800, round(im.height * scale)), Image.LANCZOS)
    path = OUT / "fig_class_index.jpg"
    im.save(path, quality=88, subsampling=0, optimize=True)
    print(f"{path.name}: {im.size} {path.stat().st_size // 1024} KB")


def conflict() -> None:
    im = trim_white(CONFLICT, pad=4)
    scale = 1800 / im.width
    im = im.resize((1800, round(im.height * scale)), Image.LANCZOS)
    path = OUT / "fig_class_conflict.jpg"
    im.save(path, quality=92, subsampling=0, optimize=True)
    print(f"{path.name}: {im.size} {path.stat().st_size // 1024} KB")


if __name__ == "__main__":
    distribution()
    overview()
    conflict()
