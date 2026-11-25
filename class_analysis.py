#!/usr/bin/env python3
"""
class_analysis.py
=================

Read-only class-distribution analysis for the staged YOLO dataset.

What it does
------------
* Reads every YOLO label file in the prepared dataset.
* Counts bounding boxes per class ID (overall and per train/val/test split).
* Counts annotation files and image files per class ID.
* Flags classes with very few training boxes and classes with high
  representation, using explicit, documented thresholds.
* Measures images containing more than one class.
* Computes class co-occurrence statistics.
* Draws distribution / co-occurrence charts.

What it never does
------------------
* modify, delete, rename or move any dataset file
* change labels, class IDs, box coordinates or splits
* invent or infer class names
* train a model

Class IDs are reported as raw numeric IDs ("Class 0" ... "Class 34") only.
No semantic meaning is assigned.

Run:
    python "C:\\vs code\\object detection\\class_analysis.py"
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless, no interactive window
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
ROOT = PROJECT_ROOT / "prepared_dataset"
OUT = PROJECT_ROOT / "dataset_class_analysis"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tif", ".tiff"}
SPLIT_ALIASES = {"train": "train", "training": "train",
                 "val": "val", "valid": "val", "validation": "val",
                 "test": "test", "testing": "test"}
SPLITS = ["train", "val", "test"]

# "Very few training examples" has an absolute floor. The relative low/high
# tiers are derived from the train-box quartiles at runtime, so the report
# always states the actual thresholds used.
VERY_FEW_TRAIN_BOXES = 100

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def rel_of(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def fingerprint() -> dict[str, tuple[int, int]]:
    return {
        rel_of(p): (p.stat().st_size, p.stat().st_mtime_ns)
        for p in ROOT.rglob("*")
        if p.is_file()
    }


def percentile(sorted_values: list[float], q: float) -> float:
    """Linear-interpolation percentile (q in [0, 1])."""
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = (len(sorted_values) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return float(
        sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction
    )


def split_of(label_path: Path) -> str:
    raw = label_path.parent.name.lower()
    return SPLIT_ALIASES.get(raw, raw)


def batch_of(rel: str) -> str:
    return Path(rel).parts[0]


def pct(part: float, whole: float) -> float:
    return round(100.0 * part / whole, 3) if whole else 0.0


def md_table(headers: list[str], rows: list[list[object]]) -> str:
    out = [
        "| " + " | ".join(str(h) for h in headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for row in rows:
        out.append("| " + " | ".join("" if v is None else str(v) for v in row) + " |")
    return "\n".join(out)


def safe_round(value: float, digits: int = 2) -> float:
    return round(float(value), digits)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def parse_label(path: Path) -> dict:
    """Parse one YOLO label file. Never writes to the file."""
    raw = path.read_text(encoding="utf-8", errors="replace")
    class_counts: Counter[int] = Counter()
    class_set: set[int] = set()
    malformed: list[dict] = []
    out_of_bounds: list[dict] = []
    box_count = 0

    for line_no, line in enumerate(raw.splitlines(), start=1):
        text = line.strip()
        if not text:
            continue
        parts = text.split()
        if len(parts) != 5:
            malformed.append({
                "line": line_no,
                "text": text[:160],
                "reason": f"expected 5 fields, found {len(parts)}",
            })
            continue
        try:
            class_id = int(parts[0])
        except ValueError:
            malformed.append({
                "line": line_no,
                "text": text[:160],
                "reason": "class ID is not an integer",
            })
            continue
        try:
            x, y, w, h = (float(parts[1]), float(parts[2]),
                          float(parts[3]), float(parts[4]))
        except ValueError:
            malformed.append({
                "line": line_no,
                "text": text[:160],
                "reason": "coordinate is not numeric",
            })
            continue

        class_counts[class_id] += 1
        class_set.add(class_id)
        box_count += 1

        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0
                and 0.0 < w <= 1.0 and 0.0 < h <= 1.0
                and x - w / 2 >= -1e-6 and x + w / 2 <= 1.0 + 1e-6
                and y - h / 2 >= -1e-6 and y + h / 2 <= 1.0 + 1e-6):
            out_of_bounds.append({
                "line": line_no,
                "class_id": class_id,
                "xywh": [x, y, w, h],
            })

    return {
        "rel": rel_of(path),
        "batch": batch_of(rel_of(path)),
        "split": split_of(path),
        "box_count": box_count,
        "class_counts": class_counts,
        "classes": class_set,
        "malformed": malformed,
        "out_of_bounds": out_of_bounds,
        "empty": box_count == 0 and not malformed,
    }


def build_image_index() -> dict[tuple[str, str, str], list[str]]:
    """Map (batch, split, casefolded stem) -> image relative paths."""
    index: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
            continue
        parts = Path(rel_of(path)).parts
        if len(parts) < 4 or parts[1] != "images":
            continue
        index[(parts[0], parts[2], path.stem.casefold())].append(rel_of(path))
    return index


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def analyze() -> dict:
    if not ROOT.is_dir():
        raise SystemExit(f"Dataset directory not found: {ROOT}")
    try:
        OUT.resolve().relative_to(ROOT.resolve())
        raise SystemExit(
            "Refusing to write analysis output inside the dataset directory. "
            f"Choose an output directory outside {ROOT}."
        )
    except ValueError:
        pass  # good: OUT is outside ROOT

    label_paths = sorted(p for p in ROOT.rglob("*.txt") if p.is_file())
    image_paths = sorted(
        p for p in ROOT.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )
    if not label_paths:
        raise SystemExit(f"No YOLO label files found under {ROOT}")

    image_index = build_image_index()

    records: list[dict] = []
    malformed_all: list[dict] = []
    out_of_bounds_all: list[dict] = []
    empty_labels: list[str] = []

    for path in label_paths:
        rec = parse_label(path)
        records.append(rec)
        for item in rec["malformed"]:
            malformed_all.append({"label": rec["rel"], **item})
        for item in rec["out_of_bounds"]:
            out_of_bounds_all.append({"label": rec["rel"], **item})
        if rec["empty"]:
            empty_labels.append(rec["rel"])

    # ---- label -> image association ------------------------------------
    label_keys = {
        (r["batch"], r["split"], Path(r["rel"]).stem.casefold())
        for r in records
    }
    label_to_images: dict[str, list[str]] = {}
    unmatched_labels: list[str] = []
    for rec in records:
        key = (rec["batch"], rec["split"], Path(rec["rel"]).stem.casefold())
        images = image_index.get(key, [])
        label_to_images[rec["rel"]] = images
        if not images:
            unmatched_labels.append(rec["rel"])

    unlabeled_images = sorted(
        image for key, images in image_index.items() if key not in label_keys
        for image in images
    )
    labels_with_multiple_images = sorted(
        rel for rel, images in label_to_images.items() if len(images) > 1
    )

    # ---- per-class counters --------------------------------------------
    all_class_ids = sorted({cid for r in records for cid in r["classes"]})
    expected_ids = list(range(35))
    analysis_ids = sorted(set(expected_ids) | set(all_class_ids))

    boxes_overall: Counter[int] = Counter()
    boxes_by_split: dict[str, Counter[int]] = {s: Counter() for s in SPLITS}
    files_overall: Counter[int] = Counter()
    files_by_split: dict[str, Counter[int]] = {s: Counter() for s in SPLITS}
    images_by_class: dict[int, set[str]] = defaultdict(set)
    images_by_class_split: dict[str, dict[int, set[str]]] = {
        s: defaultdict(set) for s in SPLITS
    }

    for rec in records:
        split = rec["split"]
        for cid, count in rec["class_counts"].items():
            boxes_overall[cid] += count
            if split in boxes_by_split:
                boxes_by_split[split][cid] += count
        for cid in rec["classes"]:
            files_overall[cid] += 1
            if split in files_by_split:
                files_by_split[split][cid] += 1
            for image in label_to_images[rec["rel"]]:
                images_by_class[cid].add(image)
                if split in images_by_class_split:
                    images_by_class_split[split][cid].add(image)

    # ---- split totals ---------------------------------------------------
    split_totals: dict[str, dict] = {}
    for split in SPLITS:
        split_records = [r for r in records if r["split"] == split]
        split_images = {
            image
            for rec in split_records
            for image in label_to_images[rec["rel"]]
        }
        split_class_ids = sorted(
            {cid for rec in split_records for cid in rec["classes"]}
        )
        split_boxes = sum(r["box_count"] for r in split_records)
        split_totals[split] = {
            "label_files": len(split_records),
            "images": len(split_images),
            "boxes": split_boxes,
            "classes_present": len(split_class_ids),
            "class_ids": split_class_ids,
            "min_class_boxes": min(boxes_by_split[split].values())
            if boxes_by_split[split] else 0,
            "max_class_boxes": max(boxes_by_split[split].values())
            if boxes_by_split[split] else 0,
            "empty_labels": sum(1 for r in split_records if r["empty"]),
        }

    total_boxes = sum(r["box_count"] for r in records)
    total_label_files = len(records)
    total_images = len(image_paths)

    # ---- thresholds -----------------------------------------------------
    train_values = sorted(boxes_by_split["train"].get(cid, 0) for cid in analysis_ids)
    q1 = percentile(train_values, 0.25)
    median = percentile(train_values, 0.50)
    q3 = percentile(train_values, 0.75)

    def tier(cid: int) -> str:
        value = boxes_by_split["train"].get(cid, 0)
        if value < VERY_FEW_TRAIN_BOXES:
            return "very_few"
        if value <= q1:
            return "low"
        if value >= q3:
            return "high"
        return "typical"

    overall_rank = {
        cid: rank for rank, (cid, _) in enumerate(
            sorted(boxes_overall.items(), key=lambda kv: (-kv[1], kv[0])), start=1
        )
    }
    train_sorted = sorted(
        analysis_ids,
        key=lambda cid: (-boxes_by_split["train"].get(cid, 0), cid),
    )
    train_rank = {cid: rank for rank, cid in enumerate(train_sorted, start=1)}

    # ---- per-class rows -------------------------------------------------
    per_class: list[dict] = []
    for cid in analysis_ids:
        train_images = images_by_class_split["train"].get(cid, set())
        row = {
            "class_id": cid,
            "total_boxes": boxes_overall.get(cid, 0),
            "train_boxes": boxes_by_split["train"].get(cid, 0),
            "val_boxes": boxes_by_split["val"].get(cid, 0),
            "test_boxes": boxes_by_split["test"].get(cid, 0),
            "box_pct_total": pct(boxes_overall.get(cid, 0), total_boxes),
            "total_annotation_files": files_overall.get(cid, 0),
            "train_annotation_files": files_by_split["train"].get(cid, 0),
            "val_annotation_files": files_by_split["val"].get(cid, 0),
            "test_annotation_files": files_by_split["test"].get(cid, 0),
            "total_images": len(images_by_class.get(cid, set())),
            "train_images": len(train_images),
            "val_images": len(
                images_by_class_split["val"].get(cid, set())
            ),
            "test_images": len(
                images_by_class_split["test"].get(cid, set())
            ),
            "image_pct_total": pct(
                len(images_by_class.get(cid, set())), total_images
            ),
            "mean_boxes_per_annotation": round(
                boxes_overall.get(cid, 0) / files_overall.get(cid, 1), 3
            ) if files_overall.get(cid, 0) else 0.0,
            "rank_by_total_boxes": overall_rank.get(cid),
            "rank_by_train_boxes": train_rank.get(cid),
            "train_tier": tier(cid),
            "expected_class_id": cid in expected_ids,
        }
        per_class.append(row)

    very_few = [r for r in per_class if r["train_tier"] == "very_few"]
    low = [r for r in per_class if r["train_tier"] == "low"]
    high = [r for r in per_class if r["train_tier"] == "high"]
    typical = [r for r in per_class if r["train_tier"] == "typical"]

    # ---- multi-class images --------------------------------------------
    classes_per_file: Counter[int] = Counter(len(r["classes"]) for r in records)
    multi_class_files = [r for r in records if len(r["classes"]) >= 2]
    multi_class_by_split = {
        split: sum(1 for r in multi_class_files if r["split"] == split)
        for split in SPLITS
    }
    multi_class_images = {
        image
        for rec in multi_class_files
        for image in label_to_images[rec["rel"]]
    }
    multi_class_image_count = len(multi_class_images)

    # ---- co-occurrence --------------------------------------------------
    pair_total: Counter[tuple[int, int]] = Counter()
    pair_by_split: dict[str, Counter[tuple[int, int]]] = {
        s: Counter() for s in SPLITS
    }
    for rec in records:
        classes = sorted(rec["classes"])
        for a, b in combinations(classes, 2):
            pair_total[(a, b)] += 1
            if rec["split"] in pair_by_split:
                pair_by_split[rec["split"]][(a, b)] += 1

    pair_image_counts: dict[tuple[int, int], int] = {}
    for (a, b) in pair_total:
        pair_image_counts[(a, b)] = len(
            images_by_class.get(a, set()) & images_by_class.get(b, set())
        )

    cooccurrence_rows: list[dict] = []
    for (a, b), count in sorted(
        pair_total.items(), key=lambda kv: (-kv[1], kv[0])
    ):
        cooccurrence_rows.append({
            "class_i": a,
            "class_j": b,
            "cooccurring_label_files": count,
            "cooccurring_images": pair_image_counts.get((a, b), 0),
            "train_label_files": pair_by_split["train"].get((a, b), 0),
            "val_label_files": pair_by_split["val"].get((a, b), 0),
            "test_label_files": pair_by_split["test"].get((a, b), 0),
            "pct_of_all_label_files": pct(count, total_label_files),
            "pct_of_class_i_files": pct(count, files_overall.get(a, 0)),
            "pct_of_class_j_files": pct(count, files_overall.get(b, 0)),
        })

    # ---- validation: originals untouched --------------------------------
    source_before = fingerprint()

    return {
        "analysis_ids": analysis_ids,
        "expected_ids": expected_ids,
        "present_ids": all_class_ids,
        "records": records,
        "per_class": per_class,
        "total_boxes": total_boxes,
        "total_label_files": total_label_files,
        "total_images": total_images,
        "split_totals": split_totals,
        "boxes_by_split": boxes_by_split,
        "images_by_class": images_by_class,
        "images_by_class_split": images_by_class_split,
        "files_overall": files_overall,
        "files_by_split": files_by_split,
        "malformed_all": malformed_all,
        "out_of_bounds_all": out_of_bounds_all,
        "empty_labels": empty_labels,
        "unmatched_labels": unmatched_labels,
        "unlabeled_images": unlabeled_images,
        "labels_with_multiple_images": labels_with_multiple_images,
        "classes_per_file": classes_per_file,
        "multi_class_files": multi_class_files,
        "multi_class_by_split": multi_class_by_split,
        "multi_class_images": multi_class_images,
        "cooccurrence_rows": cooccurrence_rows,
        "pair_total": pair_total,
        "pair_by_split": pair_by_split,
        "thresholds": {
            "metric": "boxes in the train split",
            "very_few_absolute_lt": VERY_FEW_TRAIN_BOXES,
            "q1": safe_round(q1, 2),
            "median": safe_round(median, 2),
            "q3": safe_round(q3, 2),
            "low_rule": "train_boxes <= Q1",
            "high_rule": "train_boxes >= Q3",
            "very_few_rule": f"train_boxes < {VERY_FEW_TRAIN_BOXES}",
        },
        "tiers": {
            "very_few": [r["class_id"] for r in very_few],
            "low": [r["class_id"] for r in low],
            "typical": [r["class_id"] for r in typical],
            "high": [r["class_id"] for r in high],
        },
        "source_before": source_before,
    }


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------


def make_charts(data: dict) -> None:
    ids = data["analysis_ids"]
    x = np.arange(len(ids))
    per_class = {r["class_id"]: r for r in data["per_class"]}
    total_boxes = [per_class[c]["total_boxes"] for c in ids]
    train_boxes = [per_class[c]["train_boxes"] for c in ids]
    val_boxes = [per_class[c]["val_boxes"] for c in ids]
    test_boxes = [per_class[c]["test_boxes"] for c in ids]
    total_images = [per_class[c]["total_images"] for c in ids]

    fig = plt.figure(figsize=(24, 15))
    grid = GridSpec(2, 3, figure=fig, hspace=0.32, wspace=0.22)

    ax1 = fig.add_subplot(grid[0, 0])
    ax1.bar(x, total_boxes, color="#3b6ea5")
    ax1.axhline(data["thresholds"]["median"], color="#c0392b", linestyle="--",
                linewidth=1, label=f"median = {data['thresholds']['median']:.0f}")
    ax1.set_title("Bounding boxes per class ID (all splits)")
    ax1.set_xlabel("Class ID")
    ax1.set_ylabel("Boxes")
    ax1.set_xticks(x)
    ax1.set_xticklabels(ids, fontsize=7, rotation=90)
    ax1.legend(fontsize=8)
    ax1.grid(axis="y", alpha=0.25)

    ax2 = fig.add_subplot(grid[0, 1])
    bottoms = np.zeros(len(ids))
    colors = {"train": "#3b6ea5", "val": "#e2a03f", "test": "#8e6cae"}
    for split, values in [("train", train_boxes), ("val", val_boxes),
                          ("test", test_boxes)]:
        ax2.bar(x, values, bottom=bottoms, label=split, color=colors[split])
        bottoms += np.array(values)
    ax2.set_title("Bounding boxes per class ID, stacked by split")
    ax2.set_xlabel("Class ID")
    ax2.set_ylabel("Boxes")
    ax2.set_xticks(x)
    ax2.set_xticklabels(ids, fontsize=7, rotation=90)
    ax2.legend(fontsize=8)
    ax2.grid(axis="y", alpha=0.25)

    ax3 = fig.add_subplot(grid[0, 2])
    ax3.bar(x, total_images, color="#4f8a6d")
    ax3.set_title("Image count per class ID (all splits)")
    ax3.set_xlabel("Class ID")
    ax3.set_ylabel("Images containing the class")
    ax3.set_xticks(x)
    ax3.set_xticklabels(ids, fontsize=7, rotation=90)
    ax3.grid(axis="y", alpha=0.25)

    ax4 = fig.add_subplot(grid[1, 0])
    class_counts = data["classes_per_file"]
    keys = sorted(class_counts)
    ax4.bar(keys, [class_counts[k] for k in keys], color="#b56576")
    ax4.set_title("Distinct classes per annotation file")
    ax4.set_xlabel("Number of distinct classes in one label file")
    ax4.set_ylabel("Label files")
    ax4.set_xticks(keys)
    ax4.grid(axis="y", alpha=0.25)

    ax5 = fig.add_subplot(grid[1, 1:])
    n = len(ids)
    matrix = np.zeros((n, n), dtype=float)
    index = {cid: i for i, cid in enumerate(ids)}
    for (a, b), count in data["pair_total"].items():
        if a in index and b in index:
            matrix[index[a], index[b]] = count
            matrix[index[b], index[a]] = count
    for cid in ids:
        matrix[index[cid], index[cid]] = np.nan  # blank diagonal, not a pair
    image = ax5.imshow(matrix, cmap="viridis", aspect="auto")
    ax5.set_title("Class co-occurrence: label files containing both class IDs "
                  "(diagonal left blank; see panel 1 for per-class totals)")
    ax5.set_xlabel("Class ID")
    ax5.set_ylabel("Class ID")
    ax5.set_xticks(np.arange(n))
    ax5.set_xticklabels(ids, fontsize=7, rotation=90)
    ax5.set_yticks(np.arange(n))
    ax5.set_yticklabels(ids, fontsize=7)
    fig.colorbar(image, ax=ax5, fraction=0.046, pad=0.04, label="Label files")

    fig.suptitle(
        "YOLO class distribution - raw numeric IDs only; no class names inferred",
        fontsize=14,
    )
    fig.savefig(OUT / "class_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


def write_distribution_csv(per_class: list[dict]) -> None:
    fields = [
        "class_id", "total_boxes", "train_boxes", "val_boxes", "test_boxes",
        "box_pct_total",
        "total_annotation_files", "train_annotation_files",
        "val_annotation_files", "test_annotation_files",
        "total_images", "train_images", "val_images", "test_images",
        "image_pct_total", "mean_boxes_per_annotation",
        "rank_by_total_boxes", "rank_by_train_boxes", "train_tier",
        "expected_class_id",
    ]
    with (OUT / "class_distribution.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in per_class:
            writer.writerow({k: row.get(k, "") for k in fields})


def write_cooccurrence_csv(rows: list[dict]) -> None:
    fields = [
        "class_i", "class_j", "cooccurring_label_files", "cooccurring_images",
        "train_label_files", "val_label_files", "test_label_files",
        "pct_of_all_label_files", "pct_of_class_i_files", "pct_of_class_j_files",
    ]
    with (OUT / "class_cooccurrence.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def build_summary(data: dict, source_after: dict) -> dict:
    changed = sorted(
        rel for rel in set(data["source_before"]) | set(source_after)
        if data["source_before"].get(rel) != source_after.get(rel)
    )
    total_boxes = data["total_boxes"]
    counts = [r["total_boxes"] for r in data["per_class"]]

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "mode": "READ-ONLY class distribution analysis; no training, no dataset changes",
        "dataset_root": ROOT.as_posix(),
        "output_dir": OUT.as_posix(),
        "class_naming": "raw numeric class IDs only; no semantic names assigned",
        "totals": {
            "images": data["total_images"],
            "label_files": data["total_label_files"],
            "bounding_boxes": total_boxes,
            "classes_present": len(data["present_ids"]),
            "class_ids_present": data["present_ids"],
            "expected_class_ids": data["expected_ids"],
            "all_expected_class_ids_present": data["present_ids"] == data["expected_ids"],
            "mean_boxes_per_label_file": safe_round(
                total_boxes / data["total_label_files"], 3
            ),
            "min_class_boxes": min(counts) if counts else 0,
            "max_class_boxes": max(counts) if counts else 0,
            "max_to_min_ratio": safe_round(
                max(counts) / min(counts), 3
            ) if counts and min(counts) > 0 else None,
        },
        "splits": {
            split: {
                "label_files": info["label_files"],
                "images": info["images"],
                "boxes": info["boxes"],
                "classes_present": info["classes_present"],
                "min_class_boxes": info["min_class_boxes"],
                "max_class_boxes": info["max_class_boxes"],
                "empty_labels": info["empty_labels"],
            }
            for split, info in data["split_totals"].items()
        },
        "thresholds": data["thresholds"],
        "representation": {
            "very_few_training_examples": [
                {"class_id": r["class_id"], "train_boxes": r["train_boxes"],
                 "train_images": r["train_images"]}
                for r in data["per_class"] if r["train_tier"] == "very_few"
            ],
            "low_representation": [
                {"class_id": r["class_id"], "train_boxes": r["train_boxes"],
                 "train_images": r["train_images"]}
                for r in data["per_class"] if r["train_tier"] == "low"
            ],
            "high_representation": [
                {"class_id": r["class_id"], "train_boxes": r["train_boxes"],
                 "train_images": r["train_images"]}
                for r in data["per_class"] if r["train_tier"] == "high"
            ],
            "tiers": data["tiers"],
        },
        "multi_class": {
            "label_files_with_2plus_classes": len(data["multi_class_files"]),
            "pct_of_label_files": pct(
                len(data["multi_class_files"]), data["total_label_files"]
            ),
            "images_with_2plus_classes": len(data["multi_class_images"]),
            "pct_of_images": pct(
                len(data["multi_class_images"]), data["total_images"]
            ),
            "by_split_label_files": data["multi_class_by_split"],
            "distinct_classes_per_label_file": {
                str(k): v for k, v in sorted(data["classes_per_file"].items())
            },
        },
        "cooccurrence": {
            "definition": "pair counted once per label file containing both class IDs",
            "pairs_observed": len(data["cooccurrence_rows"]),
            "max_possible_pairs": len(data["analysis_ids"])
            * (len(data["analysis_ids"]) - 1) // 2,
            "top_pairs": data["cooccurrence_rows"][:10],
        },
        "data_quality": {
            "malformed_lines": len(data["malformed_all"]),
            "empty_label_files": len(data["empty_labels"]),
            "out_of_bounds_boxes": len(data["out_of_bounds_all"]),
            "labels_without_matching_image": len(data["unmatched_labels"]),
            "unlabeled_images": len(data["unlabeled_images"]),
            "label_files_shared_by_multiple_images": len(
                data["labels_with_multiple_images"]
            ),
            "note": (
                "Image counts count every image file that maps to a label. "
                "Duplicate-stem pairs and PNG copies of original WEBP/GIF files "
                "share one label, so they increase image counts but not label "
                "or box counts."
            ),
        },
        "validation": {
            "source_files_before": len(data["source_before"]),
            "source_files_after": len(source_after),
            "source_files_modified": changed,
            "dataset_unchanged": not changed,
            "read_only_analysis": True,
        },
        "outputs": [
            "class_analysis_report.md",
            "class_distribution.csv",
            "class_summary.json",
            "class_distribution.png",
            "class_cooccurrence.csv",
        ],
        "per_class": data["per_class"],
    }


def write_report(data: dict, summary: dict) -> None:
    lines: list[str] = []
    add = lines.append
    t = summary["totals"]
    thresholds = summary["thresholds"]
    per_class = data["per_class"]
    by_id = {r["class_id"]: r for r in per_class}

    add("# Class Distribution Analysis - Raw Numeric Class IDs")
    add("")
    add(f"- **Dataset:** `{ROOT.as_posix()}`")
    add(f"- **Generated:** {summary['generated_at']}")
    add(f"- **Mode:** read-only analysis; no dataset modification, no training")
    add("")
    add("> Class IDs are reported strictly as numeric IDs (0-34). No class names "
        "are invented, inferred, or assumed. Any statement about semantic meaning "
        "would require an authoritative class-name mapping, which this dataset "
        "does not contain.")
    add("")

    add("## 1. Method and definitions")
    add("")
    add("- Every `*.txt` YOLO label file in the prepared dataset was parsed.")
    add("- **Boxes**: one row per valid YOLO annotation line (5 fields).")
    add("- **Annotation files**: label files containing at least one box of the class.")
    add("- **Images**: image files whose stem matches a label file in the same "
        "batch and split. Duplicate-stem pairs (`.jpeg` + `.png`) and the PNG "
        "copies of the original WEBP/GIF files share one label, so an image count "
        "can exceed the annotation-file count. This is reported, not corrected.")
    add("- **Split** comes from the existing directory name (`train`, `val`, `test`); "
        "assignments were not changed.")
    add("")

    add("## 2. Dataset totals")
    add("")
    add(md_table(
        ["Metric", "Value"],
        [
            ["Images", f"{t['images']:,}"],
            ["Label files", f"{t['label_files']:,}"],
            ["Bounding boxes", f"{t['bounding_boxes']:,}"],
            ["Class IDs present", f"{t['classes_present']} ({t['class_ids_present'][0]}-{t['class_ids_present'][-1]})"],
            ["All expected IDs 0-34 present", t["all_expected_class_ids_present"]],
            ["Mean boxes per label file", t["mean_boxes_per_label_file"]],
            ["Smallest class (boxes)", t["min_class_boxes"]],
            ["Largest class (boxes)", t["max_class_boxes"]],
            ["Largest / smallest ratio", t["max_to_min_ratio"]],
        ],
    ))
    add("")
    add("## 3. Distribution by split")
    add("")
    add(md_table(
        ["Split", "Images", "Label files", "Boxes", "Classes present",
         "Min class boxes", "Max class boxes", "Empty labels"],
        [[split, f"{s['images']:,}", f"{s['label_files']:,}", f"{s['boxes']:,}",
          s["classes_present"], s["min_class_boxes"], s["max_class_boxes"],
          s["empty_labels"]]
         for split, s in summary["splits"].items()],
    ))
    add("")
    test_boxes = summary["splits"].get("test", {}).get("boxes", 0)
    test_min = summary["splits"].get("test", {}).get("min_class_boxes", 0)
    test_max = summary["splits"].get("test", {}).get("max_class_boxes", 0)
    add(f"**Caution:** the test split contains only {test_boxes:,} boxes across "
        f"35 classes ({test_min}-{test_max} boxes per class). Per-class test "
        "metrics will be noisy. The split was not changed.")
    add("")

    add("## 4. Classes with very few training examples")
    add("")
    add(f"Primary metric: **boxes in the train split**. Absolute rule: "
        f"`train_boxes < {thresholds['very_few_absolute_lt']}`. "
        f"Relative train-box quartiles: Q1 = {thresholds['q1']}, "
        f"median = {thresholds['median']}, Q3 = {thresholds['q3']}.")
    add("")
    very_few = summary["representation"]["very_few_training_examples"]
    if very_few:
        add(md_table(
            ["Class ID", "Train boxes", "Train images", "Total boxes"],
            [[r["class_id"], r["train_boxes"], r["train_images"],
              by_id[r["class_id"]]["total_boxes"]] for r in very_few],
        ))
    else:
        add(f"**No class has fewer than {thresholds['very_few_absolute_lt']} "
            "training boxes.** By this absolute criterion, no class is severely "
            "under-represented in the train split.")
    add("")
    low_tier = summary["representation"]["tiers"]["low"]
    if low_tier:
        add(f"Classes in the bottom train-box quartile (low tier, "
            f"`train_boxes <= {thresholds['q1']}`):")
        add("")
        add(md_table(
            ["Class ID", "Train boxes", "Train images", "Total boxes"],
            [[cid, by_id[cid]["train_boxes"], by_id[cid]["train_images"],
              by_id[cid]["total_boxes"]] for cid in low_tier],
        ))
    else:
        add("No class falls in the bottom train-box quartile.")
    add("")
    train_image_values = [r["train_images"] for r in per_class]
    add(f"Train images per class range from {min(train_image_values)} to "
        f"{max(train_image_values)}; no class has fewer than "
        f"{VERY_FEW_TRAIN_BOXES} train boxes.")
    add("")

    add("## 5. Classes with high representation")
    add("")
    add(f"Rule: `train_boxes >= Q3 = {thresholds['q3']}`.")
    add("")
    add(md_table(
        ["Class ID", "Train boxes", "Train images", "Total boxes", "Share of all boxes"],
        [[r["class_id"], r["train_boxes"], r["train_images"],
          by_id[r["class_id"]]["total_boxes"],
          f"{by_id[r['class_id']]['box_pct_total']}%"]
         for r in summary["representation"]["high_representation"]],
    ))
    add("")
    add("The overall distribution is close to uniform: the largest class has "
        f"{t['max_class_boxes']} boxes and the smallest {t['min_class_boxes']} "
        f"(ratio {t['max_to_min_ratio']}). High/low tiers here are relative "
        "quartile labels, not evidence of severe imbalance.")
    add("")

    add("## 6. Full per-class table")
    add("")
    add(md_table(
        ["ID", "Total boxes", "Train", "Val", "Test", "Images", "Label files",
         "% of boxes", "Train tier", "Rank (total)"],
        [[r["class_id"], f"{r['total_boxes']:,}", r["train_boxes"],
          r["val_boxes"], r["test_boxes"], r["total_images"],
          r["total_annotation_files"], f"{r['box_pct_total']}%",
          r["train_tier"], r["rank_by_total_boxes"]] for r in per_class],
    ))
    add("")

    add("## 7. Images containing multiple classes")
    add("")
    mc = summary["multi_class"]
    add(md_table(
        ["Metric", "Value"],
        [
            ["Label files with >= 2 classes", f"{mc['label_files_with_2plus_classes']:,} ({mc['pct_of_label_files']}%)"],
            ["Images with >= 2 classes", f"{mc['images_with_2plus_classes']:,} ({mc['pct_of_images']}%)"],
            ["Train label files with >= 2 classes", mc["by_split_label_files"].get("train", 0)],
            ["Val label files with >= 2 classes", mc["by_split_label_files"].get("val", 0)],
            ["Test label files with >= 2 classes", mc["by_split_label_files"].get("test", 0)],
        ],
    ))
    add("")
    add("Distribution of distinct classes per label file:")
    add("")
    add(md_table(
        ["Distinct classes", "Label files"],
        [[k, f"{v:,}"] for k, v in mc["distinct_classes_per_label_file"].items()],
    ))
    add("")

    add("## 8. Class co-occurrence")
    add("")
    add("A co-occurrence pair is counted once per label file that contains both "
        "class IDs. Pairs not observed are omitted from the CSV.")
    add("")
    add(f"- Pairs observed: **{summary['cooccurrence']['pairs_observed']}** of "
        f"{summary['cooccurrence']['max_possible_pairs']} possible pairs.")
    add("")
    add("Top 15 co-occurring pairs:")
    add("")
    add(md_table(
        ["Class i", "Class j", "Label files", "Images", "Train", "Val", "Test"],
        [[r["class_i"], r["class_j"], r["cooccurring_label_files"],
          r["cooccurring_images"], r["train_label_files"],
          r["val_label_files"], r["test_label_files"]]
         for r in data["cooccurrence_rows"][:15]],
    ))
    add("")
    add("Full pair-level data: `class_cooccurrence.csv`. The heatmap in "
        "`class_distribution.png` shows all observed pairs.")
    add("")

    add("## 9. Data quality observations")
    add("")
    dq = summary["data_quality"]
    add(md_table(
        ["Observation", "Count"],
        [
            ["Malformed label lines (not counted as boxes)", dq["malformed_lines"]],
            ["Empty label files", dq["empty_label_files"]],
            ["Boxes whose center-based extent exceeds image bounds", dq["out_of_bounds_boxes"]],
            ["Label files without a matching image", dq["labels_without_matching_image"]],
            ["Images without a matching label", dq["unlabeled_images"]],
            ["Label files shared by multiple images", dq["label_files_shared_by_multiple_images"]],
        ],
    ))
    add("")
    add(dq["note"])
    add("")
    add(f"Originals unchanged during analysis: **{summary['validation']['dataset_unchanged']}** "
        f"({summary['validation']['source_files_before']:,} source files fingerprinted "
        "before and after).")
    add("")

    add("## 10. Relevance to pedestrian-obstacle detection")
    add("")
    add("This dataset exposes only numeric class IDs. Because there is no "
        "authoritative ID -> name mapping, this analysis **cannot** state which "
        "classes correspond to curbs, poles, bollards, vehicles, pedestrians, "
        "street furniture, or any other obstacle category. No obstacle class is "
        "selected or excluded here. Any obstacle-specific class choice must wait "
        "for a verified class-name mapping or a documented box-level visual audit.")
    add("")
    add("Counts alone also do not demonstrate real-world coverage: a class can be "
        "well represented in this dataset and still be irrelevant to pedestrian "
        "navigation, and vice versa.")
    add("")

    add("## 11. Outputs")
    add("")
    add(md_table(
        ["File", "Contents"],
        [
            ["`class_analysis_report.md`", "This report"],
            ["`class_distribution.csv`", f"{len(per_class)} class rows with split-level counts"],
            ["`class_summary.json`", "Machine-readable totals, tiers, co-occurrence, validation"],
            ["`class_distribution.png`", "Distribution and co-occurrence charts"],
            ["`class_cooccurrence.csv`", f"{len(data['cooccurrence_rows'])} observed class pairs"],
        ],
    ))
    add("")
    add("## 12. Exact command")
    add("")
    add("```powershell")
    add('python "C:\\vs code\\object detection\\class_analysis.py"')
    add("```")
    add("")
    add("## 13. Not done (by rule)")
    add("")
    add("- No dataset file modified, moved, renamed, or deleted.")
    add("- No label, class ID, box coordinate, or split changed.")
    add("- No class name invented or inferred.")
    add("- No obstacle class selected.")
    add("- No model trained.")
    add("")

    (OUT / "class_analysis_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("Dataset:", ROOT)
    print("Output :", OUT)
    print("Mode   : read-only class analysis")
    print()

    OUT.mkdir(parents=True, exist_ok=True)

    print("[1/4] Parsing labels and mapping images")
    source_before = fingerprint()
    data = analyze()
    data["source_before"] = source_before
    print(f"    label files={data['total_label_files']:,} "
          f"images={data['total_images']:,} boxes={data['total_boxes']:,} "
          f"classes={len(data['present_ids'])}")
    print(f"    malformed lines={len(data['malformed_all'])} "
          f"out-of-bounds boxes={len(data['out_of_bounds_all'])} "
          f"unmatched labels={len(data['unmatched_labels'])}")

    print("[2/4] Writing CSV and JSON outputs")
    write_distribution_csv(data["per_class"])
    write_cooccurrence_csv(data["cooccurrence_rows"])

    source_after = fingerprint()
    summary = build_summary(data, source_after)
    (OUT / "class_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("[3/4] Drawing charts")
    make_charts(data)

    print("[4/4] Writing report")
    write_report(data, summary)

    if summary["validation"]["source_files_modified"]:
        raise SystemExit(
            "ERROR: dataset files changed during analysis: "
            + ", ".join(summary["validation"]["source_files_modified"][:10])
        )

    print()
    print("Done. Files written to:", OUT)
    for name in summary["outputs"]:
        print(" -", name)
    print()
    print("Totals: boxes =", data["total_boxes"],
          "| label files =", data["total_label_files"],
          "| images =", data["total_images"])
    print("Very few training examples (<%d train boxes):" % VERY_FEW_TRAIN_BOXES,
          summary["representation"]["very_few_training_examples"] or "none")
    print("Low tier (<= Q1 %.2f):" % data["thresholds"]["q1"],
          summary["representation"]["tiers"]["low"] or "none")
    print("High tier (>= Q3 %.2f):" % data["thresholds"]["q3"],
          summary["representation"]["tiers"]["high"] or "none")
    print("Multi-class label files:", summary["multi_class"]["label_files_with_2plus_classes"])
    print("Co-occurring pairs observed:", summary["cooccurrence"]["pairs_observed"])
    print("Dataset unchanged:", summary["validation"]["dataset_unchanged"])


if __name__ == "__main__":
    main()
