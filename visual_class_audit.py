#!/usr/bin/env python3
"""
visual_class_audit.py
=====================

Read-only visual audit of a YOLO dataset with numeric class IDs.

Purpose
-------
Generate contact sheets so a human can look at examples of every class ID and
decide what each numeric ID represents. The script never assigns class names
and never treats any class as an obstacle.

What it does
------------
* Parses every YOLO label file in the prepared dataset.
* Selects up to 20 examples per class ID (0-34), stratified by:
    - batch (all represented batches are used),
    - bounding-box size (small / medium / large tertiles of that class).
* Avoids selecting multiple copies of the same source image:
    - exact duplicates are detected by MD5,
    - duplicate-stem variants share one label and only one image is used,
    - images already used for another class are avoided when alternatives exist.
* Draws only the target class's boxes, with the numeric class ID on each box.
* Writes one contact sheet per class, a combined visual index, a CSV
  inventory, an HTML/Markdown index, and a report.

What it never does
------------------
* modify, move, rename or delete any dataset file
* change annotations, class IDs, coordinates or splits
* infer, assign or suggest class names
* label an object as an obstacle
* train a model

Run:
    python "C:\\vs code\\object detection\\visual_class_audit.py"
"""

from __future__ import annotations

import csv
import hashlib
import html
import json
import math
import random
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from itertools import combinations
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
ROOT = PROJECT_ROOT / "prepared_dataset"
OUT = PROJECT_ROOT / "dataset_class_analysis" / "visual_audit"
CLASS_ANALYSIS_DIR = PROJECT_ROOT / "dataset_class_analysis"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tif", ".tiff"}
SPLIT_ALIASES = {"train": "train", "training": "train",
                 "val": "val", "valid": "val", "validation": "val",
                 "test": "test", "testing": "test"}
BATCHES = ["Batch 1", "Batch 2", "Batch 3", "Batch 4"]
TIERS = ["small", "medium", "large"]
TIER_TARGETS = [7, 7, 6]  # sums to 20
SAMPLES_PER_CLASS = 20
CLASS_IDS = list(range(35))
SEED = 20261109

# Rendering
TILE_W = 420
TILE_IMG_H = 340
TILE_HEADER_H = 94
SHEET_COLS = 4
SHEET_MARGIN = 24
SHEET_GAP = 14
TITLE_H = 118
FOOTER_H = 38

# Image choice preference when a label maps to multiple image files
EXT_PRIORITY = {".png": 0, ".jpg": 1, ".jpeg": 2, ".webp": 3,
                ".gif": 4, ".bmp": 5, ".tif": 6, ".tiff": 7}

# Inconsistency heuristics
TINY_AREA = 1e-4
EXTREME_ASPECT_HIGH = 10.0
EXTREME_ASPECT_LOW = 0.1
SAME_CLASS_IOU = 0.90
CROSS_CLASS_IOU = 0.80
MAX_EXAMPLES_PER_ISSUE = 8


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def rel_of(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def split_of(label_path: Path) -> str:
    raw = label_path.parent.name.lower()
    return SPLIT_ALIASES.get(raw, raw)


def batch_of(rel: str) -> str:
    return Path(rel).parts[0]


def md5_file(path: Path) -> str:
    h = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fingerprint() -> dict[str, tuple[int, int]]:
    return {
        rel_of(p): (p.stat().st_size, p.stat().st_mtime_ns)
        for p in ROOT.rglob("*")
        if p.is_file()
    }


def iou(a: dict, b: dict) -> float:
    ax1, ay1 = a["cx"] - a["w"] / 2, a["cy"] - a["h"] / 2
    ax2, ay2 = a["cx"] + a["w"] / 2, a["cy"] + a["h"] / 2
    bx1, by1 = b["cx"] - b["w"] / 2, b["cy"] - b["h"] / 2
    bx2, by2 = b["cx"] + b["w"] / 2, b["cy"] + b["h"] / 2
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = a["w"] * a["h"] + b["w"] * b["h"] - inter
    return inter / union if union > 0 else 0.0


def dhash_image(image: Image.Image) -> int:
    gray = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    pixels = list(gray.getdata())
    bits = 0
    for row in range(8):
        for col in range(8):
            left = pixels[row * 9 + col]
            right = pixels[row * 9 + col + 1]
            bits = (bits << 1) | (1 if left > right else 0)
    return bits


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def nonblack_fraction(image: Image.Image, threshold: int = 20) -> float:
    """Fraction of pixels whose luminance is above a small black threshold."""
    histogram = image.convert("L").histogram()
    total = sum(histogram)
    if total == 0:
        return 0.0
    return 1.0 - sum(histogram[: threshold + 1]) / total


def image_signature_format(path: Path) -> str:
    """Detect the actual image encoding from the file header."""
    with path.open("rb") as fh:
        head = fh.read(16)
    if head.startswith(b"\xff\xd8\xff"):
        return "jpg"
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if head.startswith((b"GIF87a", b"GIF89a")):
        return "gif"
    if head.startswith(b"RIFF") and head[8:12] == b"WEBP":
        return "webp"
    if head.startswith(b"BM"):
        return "bmp"
    if head.startswith((b"II*\x00", b"MM\x00*")):
        return "tiff"
    return "unknown"


def normalized_extension(path: Path) -> str:
    ext = path.suffix.lower().lstrip(".")
    return {"jpeg": "jpg", "tif": "tiff"}.get(ext, ext)


def md_table(headers: list[str], rows: list[list[object]]) -> str:
    out = [
        "| " + " | ".join(str(h) for h in headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for row in rows:
        out.append("| " + " | ".join("" if v is None else str(v) for v in row) + " |")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Fonts and text helpers
# ---------------------------------------------------------------------------


def find_font(bold: bool) -> Path | None:
    names = (["arialbd.ttf", "seguisb.ttf", "calibrib.ttf"]
             if bold else ["arial.ttf", "segoeui.ttf", "calibri.ttf"])
    candidates = [Path(r"C:\Windows\Fonts") / n for n in names]
    try:
        import matplotlib
        base = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
        candidates.append(base / ("DejaVuSans-Bold.ttf" if bold
                                  else "DejaVuSans.ttf"))
    except Exception:
        pass
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


FONT_BOLD_PATH = find_font(True)
FONT_REGULAR_PATH = find_font(False)


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    path = FONT_BOLD_PATH if bold else FONT_REGULAR_PATH
    if path:
        try:
            return ImageFont.truetype(str(path), size)
        except Exception:
            pass
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def fit_text(draw: ImageDraw.ImageDraw, text: str, fnt,
             max_width: int) -> str:
    if draw.textlength(text, font=fnt) <= max_width:
        return text
    suffix = "..."
    low, high = 0, len(text)
    while low < high:
        mid = (low + high + 1) // 2
        if draw.textlength(text[:mid] + suffix, font=fnt) <= max_width:
            low = mid
        else:
            high = mid - 1
    return text[:low] + suffix


def draw_text_box(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str,
                  fnt, fill: str, bg: str) -> None:
    x, y = xy
    bbox = draw.textbbox((x, y), text, font=fnt)
    pad = 3
    draw.rectangle(
        (bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad),
        fill=bg,
    )
    draw.text((x, y), text, font=fnt, fill=fill)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def parse_label(path: Path) -> dict:
    rel = rel_of(path)
    raw = path.read_text(encoding="utf-8", errors="replace")
    boxes: list[dict] = []
    malformed: list[dict] = []
    invalid_geometry = 0
    out_of_bounds = 0

    for line_no, line in enumerate(raw.splitlines(), start=1):
        text = line.strip()
        if not text:
            continue
        parts = text.split()
        if len(parts) != 5:
            malformed.append({"line": line_no, "reason": "expected 5 fields",
                              "text": text[:120]})
            continue
        try:
            class_id = int(parts[0])
            cx, cy, w, h = (float(parts[1]), float(parts[2]),
                            float(parts[3]), float(parts[4]))
        except ValueError:
            malformed.append({"line": line_no, "reason": "non-numeric value",
                              "text": text[:120]})
            continue

        if not (0.0 < w <= 1.0 and 0.0 < h <= 1.0
                and 0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0):
            invalid_geometry += 1
        elif (cx - w / 2 < -1e-6 or cx + w / 2 > 1.0 + 1e-6
              or cy - h / 2 < -1e-6 or cy + h / 2 > 1.0 + 1e-6):
            out_of_bounds += 1

        boxes.append({
            "class_id": class_id,
            "cx": cx, "cy": cy, "w": w, "h": h,
            "area": w * h,
            "aspect": max(w / h, h / w) if w > 0 and h > 0 else math.inf,
            "line": line_no,
        })

    classes = {b["class_id"] for b in boxes}
    return {
        "rel": rel,
        "batch": batch_of(rel),
        "split": split_of(path),
        "boxes": boxes,
        "classes": classes,
        "malformed": malformed,
        "invalid_geometry": invalid_geometry,
        "out_of_bounds": out_of_bounds,
    }


def build_image_index() -> dict[tuple[str, str, str], list[str]]:
    index: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
            continue
        parts = Path(rel_of(path)).parts
        if len(parts) < 4 or parts[1] != "images":
            continue
        index[(parts[0], parts[2], path.stem.casefold())].append(rel_of(path))
    return index


def choose_image(images: list[str]) -> str:
    return sorted(images, key=lambda r: (EXT_PRIORITY.get(Path(r).suffix.lower(), 99), r))[0]


# ---------------------------------------------------------------------------
# Annotation inconsistency scan
# ---------------------------------------------------------------------------


def scan_inconsistencies(records: list[dict]) -> dict:
    issues = {
        "malformed_lines": 0,
        "invalid_geometry_boxes": 0,
        "out_of_bounds_boxes": 0,
        "tiny_boxes": 0,
        "extreme_aspect_boxes": 0,
        "same_class_duplicate_pairs": 0,
        "cross_class_overlap_pairs": 0,
    }
    examples: dict[str, list[dict]] = defaultdict(list)

    for rec in records:
        issues["malformed_lines"] += len(rec["malformed"])
        issues["invalid_geometry_boxes"] += rec["invalid_geometry"]
        issues["out_of_bounds_boxes"] += rec["out_of_bounds"]

        for box in rec["boxes"]:
            if box["area"] < TINY_AREA:
                issues["tiny_boxes"] += 1
                if len(examples["tiny_boxes"]) < MAX_EXAMPLES_PER_ISSUE:
                    examples["tiny_boxes"].append({
                        "label": rec["rel"], "line": box["line"],
                        "class_id": box["class_id"],
                        "area": round(box["area"], 8),
                    })
            if box["aspect"] > EXTREME_ASPECT_HIGH or box["aspect"] < EXTREME_ASPECT_LOW:
                issues["extreme_aspect_boxes"] += 1
                if len(examples["extreme_aspect_boxes"]) < MAX_EXAMPLES_PER_ISSUE:
                    examples["extreme_aspect_boxes"].append({
                        "label": rec["rel"], "line": box["line"],
                        "class_id": box["class_id"],
                        "aspect": round(box["aspect"], 2),
                    })

        for a, b in combinations(rec["boxes"], 2):
            if a["class_id"] == b["class_id"]:
                value = iou(a, b)
                if value >= SAME_CLASS_IOU:
                    issues["same_class_duplicate_pairs"] += 1
                    if len(examples["same_class_duplicate_pairs"]) < MAX_EXAMPLES_PER_ISSUE:
                        examples["same_class_duplicate_pairs"].append({
                            "label": rec["rel"],
                            "lines": [a["line"], b["line"]],
                            "class_id": a["class_id"],
                            "iou": round(value, 4),
                        })
            else:
                value = iou(a, b)
                if value >= CROSS_CLASS_IOU:
                    issues["cross_class_overlap_pairs"] += 1
                    if len(examples["cross_class_overlap_pairs"]) < MAX_EXAMPLES_PER_ISSUE:
                        examples["cross_class_overlap_pairs"].append({
                            "label": rec["rel"],
                            "lines": [a["line"], b["line"]],
                            "class_ids": [a["class_id"], b["class_id"]],
                            "iou": round(value, 4),
                        })
    return {"counts": issues, "examples": dict(examples)}


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


def build_candidates(records: list[dict],
                     label_to_images: dict[str, list[str]],
                     image_md5: dict[str, str]) -> dict[int, list[dict]]:
    by_class: dict[int, list[dict]] = defaultdict(list)

    # Per-class area tertiles from every box of that class.
    areas_by_class: dict[int, list[float]] = defaultdict(list)
    for rec in records:
        for box in rec["boxes"]:
            areas_by_class[box["class_id"]].append(box["area"])
    thresholds: dict[int, tuple[float, float]] = {}
    for cid, areas in areas_by_class.items():
        ordered = sorted(areas)
        q1 = ordered[int(0.3333 * (len(ordered) - 1))]
        q2 = ordered[int(0.6667 * (len(ordered) - 1))]
        thresholds[cid] = (q1, q2)

    for rec in records:
        images = label_to_images.get(rec["rel"], [])
        if not images:
            continue
        image_rel = choose_image(images)
        for cid in sorted(rec["classes"]):
            class_boxes = [b for b in rec["boxes"] if b["class_id"] == cid]
            if not class_boxes:
                continue
            max_area = max(b["area"] for b in class_boxes)
            q1, q2 = thresholds.get(cid, (0.0, 0.0))
            tier = ("small" if max_area <= q1
                    else "medium" if max_area <= q2 else "large")
            by_class[cid].append({
                "class_id": cid,
                "label_rel": rec["rel"],
                "image_rel": image_rel,
                "available_images": len(images),
                "batch": rec["batch"],
                "split": rec["split"],
                "boxes": class_boxes,
                "all_box_count": len(rec["boxes"]),
                "max_area": max_area,
                "size_tier": tier,
                "tier_q1": q1,
                "tier_q2": q2,
                "image_md5": image_md5.get(image_rel, ""),
                "stem_key": (rec["batch"], rec["split"],
                             Path(rec["rel"]).stem.casefold()),
            })
    return by_class


def select_for_class(cid: int, candidates: list[dict],
                     global_used: set[str]) -> list[dict]:
    rng = random.Random(SEED + cid)
    ordered = candidates[:]
    rng.shuffle(ordered)
    # Prefer images not already used by another class.
    ordered.sort(key=lambda c: c["image_md5"] in global_used)

    batch_order = BATCHES[:]
    offset = cid % len(batch_order)
    batch_order = batch_order[offset:] + batch_order[:offset]

    selected: list[dict] = []
    used_hashes: set[str] = set()
    used_stems: set[tuple] = set()

    def usable(c: dict) -> bool:
        return (c["image_md5"] not in used_hashes
                and c["stem_key"] not in used_stems)

    def take(c: dict) -> None:
        selected.append(c)
        used_hashes.add(c["image_md5"])
        used_stems.add(c["stem_key"])

    # Stratified pass: size tier, round-robin over batches.
    for tier, target in zip(TIERS, TIER_TARGETS):
        pool = [c for c in ordered if c["size_tier"] == tier and usable(c)]
        got = 0
        while got < target:
            progressed = False
            for batch in batch_order:
                if got >= target:
                    break
                pick = next((c for c in pool
                             if c["batch"] == batch and usable(c)), None)
                if pick:
                    take(pick)
                    got += 1
                    progressed = True
            if not progressed:
                break

    # Fill any remaining slots across all tiers.
    while len(selected) < SAMPLES_PER_CLASS:
        progressed = False
        for batch in batch_order:
            if len(selected) >= SAMPLES_PER_CLASS:
                break
            pick = next((c for c in ordered
                         if c["batch"] == batch and usable(c)), None)
            if pick:
                take(pick)
                progressed = True
        if not progressed:
            break

    # Guarantee batch coverage if alternatives exist.
    present = {c["batch"] for c in selected}
    for batch in BATCHES:
        if batch in present:
            continue
        pool = [c for c in ordered if c["batch"] == batch and usable(c)]
        if not pool:
            continue
        if len(selected) < SAMPLES_PER_CLASS:
            take(pool[0])
            present.add(batch)
            continue
        # Replace a sample from the most represented batch, keeping tier coverage.
        counts = Counter(c["batch"] for c in selected)
        donor_batch = counts.most_common(1)[0][0]
        for index, sample in enumerate(selected):
            if sample["batch"] != donor_batch:
                continue
            tier_count = sum(1 for s in selected
                             if s["size_tier"] == sample["size_tier"])
            if tier_count > 1:
                selected.pop(index)
                used_hashes.discard(sample["image_md5"])
                used_stems.discard(sample["stem_key"])
                take(pool[0])
                present.add(batch)
                break

    # Order samples: size tier then batch then path, for easy visual comparison.
    tier_rank = {tier: i for i, tier in enumerate(TIERS)}
    selected.sort(key=lambda c: (tier_rank[c["size_tier"]],
                                 BATCHES.index(c["batch"])
                                 if c["batch"] in BATCHES else 99,
                                 c["image_rel"]))
    for index, sample in enumerate(selected, start=1):
        sample["sample_index"] = index
    return selected


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------


def load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        if getattr(image, "n_frames", 1) > 1:
            image.seek(0)
        return image.convert("RGB")


def draw_boxes_on_image(image: Image.Image, boxes: list[dict]) -> Image.Image:
    draw = ImageDraw.Draw(image)
    width, height = image.size
    line_width = max(3, min(width, height) // 140)
    fnt = font(max(14, min(width, height) // 22), bold=True)
    for box in boxes:
        x1 = (box["cx"] - box["w"] / 2) * width
        y1 = (box["cy"] - box["h"] / 2) * height
        x2 = (box["cx"] + box["w"] / 2) * width
        y2 = (box["cy"] + box["h"] / 2) * height
        x1, x2 = sorted((max(0, min(width - 1, x1)),
                         max(0, min(width - 1, x2))))
        y1, y2 = sorted((max(0, min(height - 1, y1)),
                         max(0, min(height - 1, y2))))
        # White outer stroke keeps the red box visible on light photos.
        draw.rectangle((x1, y1, x2, y2), outline="white", width=line_width + 2)
        draw.rectangle((x1, y1, x2, y2), outline="#ff2d55", width=line_width)
        draw_text_box(draw, (int(x1) + 2, int(y1) + 2), str(box["class_id"]),
                      fnt, "white", "#c4002b")
    return image


def paste_letterbox(sheet: Image.Image, image: Image.Image,
                    box: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    left, top, right, bottom = box
    area_w, area_h = right - left, bottom - top
    scale = min(area_w / image.width, area_h / image.height)
    new_w = max(1, int(image.width * scale))
    new_h = max(1, int(image.height * scale))
    resized = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    x = left + (area_w - new_w) // 2
    y = top + (area_h - new_h) // 2
    sheet.paste(resized, (x, y))
    return x, y, new_w, new_h


def draw_tile(sheet: Image.Image, sample: dict, tile_box: tuple[int, int],
              image_area: tuple[int, int], draw: ImageDraw.ImageDraw) -> None:
    x, y = tile_box
    header_h, image_h = image_area
    cid = sample["class_id"]

    # Background
    draw.rectangle((x, y, x + TILE_W - 1, y + header_h + image_h - 1),
                   fill="#14161a", outline="#3a3f46")

    # Header text
    id_font = font(30, bold=True)
    info_font = font(15)
    small_font = font(13)
    draw.text((x + 10, y + 6), f"CLASS {cid:02d}", font=id_font, fill="#ffcc00")
    sample_text = f"sample {sample['sample_index']}/{sample['samples_in_class']}"
    draw.text((x + TILE_W - 12 - draw.textlength(sample_text, font=info_font),
               y + 10), sample_text, font=info_font, fill="#cfd3d8")
    filename = fit_text(draw, Path(sample["image_rel"]).name, small_font,
                        TILE_W - 20)
    draw.text((x + 10, y + 46), filename, font=small_font, fill="#ffffff")
    meta = (f"{sample['batch']} | {sample['split']} | "
            f"{sample['size_tier']} box | area {sample['max_area']:.2%}")
    draw.text((x + 10, y + 66), fit_text(draw, meta, small_font, TILE_W - 20),
              font=small_font, fill="#9aa2ac")

    # Image area
    image_left = x + 8
    image_top = y + header_h
    image_right = x + TILE_W - 8
    image_bottom = image_top + image_h - 8
    draw.rectangle((image_left, image_top, image_right, image_bottom),
                   fill="#000000")

    path = ROOT / Path(sample["image_rel"])
    try:
        image = load_rgb(path)
    except Exception as exc:  # noqa: BLE001
        draw.text((image_left + 10, image_top + 10),
                  f"IMAGE LOAD ERROR: {exc}", font=small_font, fill="#ff5555")
        return

    # Record source-image properties used by the quality notes in the report.
    sample["image_size"] = image.size
    sample["nonblack_fraction"] = nonblack_fraction(image)

    # Compute a crop hash of the largest target box from the clean image.
    largest = max(sample["boxes"], key=lambda b: b["area"])
    margin = 0.12
    cx1 = int(max(0, (largest["cx"] - largest["w"] / 2 - margin * largest["w"])
                 * image.width))
    cy1 = int(max(0, (largest["cy"] - largest["h"] / 2 - margin * largest["h"])
                 * image.height))
    cx2 = int(min(image.width, (largest["cx"] + largest["w"] / 2
                                + margin * largest["w"]) * image.width))
    cy2 = int(min(image.height, (largest["cy"] + largest["h"] / 2
                                 + margin * largest["h"]) * image.height))
    if cx2 - cx1 >= 8 and cy2 - cy1 >= 8:
        sample["crop_dhash"] = dhash_image(image.crop((cx1, cy1, cx2, cy2)))
    else:
        sample["crop_dhash"] = None

    image = draw_boxes_on_image(image, sample["boxes"])
    px, py, pw, ph = paste_letterbox(
        sheet, image, (image_left, image_top, image_right, image_bottom)
    )

    # Re-draw boxes in sheet coordinates so they sit exactly on the pasted image.
    box_draw = ImageDraw.Draw(sheet)
    line_width = max(3, min(pw, ph) // 120)
    label_font = font(max(14, min(pw, ph) // 24), bold=True)
    for box in sample["boxes"]:
        bx1 = px + (box["cx"] - box["w"] / 2) * pw
        by1 = py + (box["cy"] - box["h"] / 2) * ph
        bx2 = px + (box["cx"] + box["w"] / 2) * pw
        by2 = py + (box["cy"] + box["h"] / 2) * ph
        bx1, bx2 = sorted((max(px, bx1), min(px + pw - 1, bx2)))
        by1, by2 = sorted((max(py, by1), min(py + ph - 1, by2)))
        box_draw.rectangle((bx1, by1, bx2, by2), outline="white",
                           width=line_width + 2)
        box_draw.rectangle((bx1, by1, bx2, by2), outline="#ff2d55",
                           width=line_width)
        draw_text_box(box_draw, (int(bx1) + 2, int(by1) + 2), str(cid),
                      label_font, "white", "#c4002b")


def build_contact_sheet(cid: int, samples: list[dict], generated_at: str) -> Path:
    rows = max(1, math.ceil(len(samples) / SHEET_COLS))
    width = SHEET_MARGIN * 2 + SHEET_COLS * TILE_W + (SHEET_COLS - 1) * SHEET_GAP
    height = (TITLE_H + SHEET_MARGIN + rows * (TILE_HEADER_H + TILE_IMG_H)
              + (rows - 1) * SHEET_GAP + FOOTER_H)
    sheet = Image.new("RGB", (width, height), "#0b0d10")
    draw = ImageDraw.Draw(sheet)

    title_font = font(38, bold=True)
    sub_font = font(18)
    draw.text((SHEET_MARGIN, 18), f"CLASS {cid:02d} - VISUAL AUDIT",
              font=title_font, fill="#ffcc00")
    subtitle = (f"{len(samples)} examples | numeric class ID only | "
                "no class name inferred | red boxes = target class annotations")
    draw.text((SHEET_MARGIN, 68), subtitle, font=sub_font, fill="#c7ccd3")
    draw.line((SHEET_MARGIN, TITLE_H - 8, width - SHEET_MARGIN, TITLE_H - 8),
              fill="#3a3f46", width=2)

    for index, sample in enumerate(samples):
        row = index // SHEET_COLS
        col = index % SHEET_COLS
        x = SHEET_MARGIN + col * (TILE_W + SHEET_GAP)
        y = TITLE_H + row * (TILE_HEADER_H + TILE_IMG_H + SHEET_GAP)
        sample["samples_in_class"] = len(samples)
        draw_tile(sheet, sample, (x, y), (TILE_HEADER_H, TILE_IMG_H), draw)

    footer = (f"Dataset: {ROOT.as_posix()} | generated {generated_at} | "
              "read-only audit | no class names assigned")
    draw.text((SHEET_MARGIN, height - FOOTER_H + 8),
              fit_text(draw, footer, font(13), width - 2 * SHEET_MARGIN),
              font=font(13), fill="#8b929b")

    path = OUT / f"class_{cid:02d}_contact_sheet.png"
    sheet.save(path, format="PNG")
    return path


def build_combined_index(index_samples: dict[int, dict], generated_at: str) -> Path:
    cols = 7
    rows = 5
    tile_w = 300
    image_h = 230
    header_h = 42
    gap = 10
    margin = 20
    title_h = 84
    footer_h = 30
    width = margin * 2 + cols * tile_w + (cols - 1) * gap
    height = title_h + margin + rows * (header_h + image_h) + (rows - 1) * gap + footer_h
    sheet = Image.new("RGB", (width, height), "#0b0d10")
    draw = ImageDraw.Draw(sheet)

    draw.text((margin, 14), "COMBINED CLASS INDEX - ONE EXAMPLE PER CLASS ID",
              font=font(30, bold=True), fill="#ffcc00")
    draw.text((margin, 52),
              "Numeric IDs only. No class names inferred. Use the per-class "
              "contact sheets for 20 examples each.",
              font=font(15), fill="#c7ccd3")

    for cid in CLASS_IDS:
        index = CLASS_IDS.index(cid)
        row = index // cols
        col = index % cols
        x = margin + col * (tile_w + gap)
        y = title_h + row * (header_h + image_h + gap)
        draw.rectangle((x, y, x + tile_w - 1, y + header_h + image_h - 1),
                       fill="#14161a", outline="#3a3f46")
        draw.text((x + 8, y + 6), f"CLASS {cid:02d}", font=font(21, bold=True),
                  fill="#ffcc00")
        sample = index_samples.get(cid)
        if not sample:
            draw.text((x + 8, y + header_h + 10), "no sample",
                      font=font(14), fill="#ff5555")
            continue
        sample_text = f"{sample['size_tier']}"
        draw.text((x + tile_w - 8 - draw.textlength(sample_text, font=font(12)),
                   y + 12), sample_text, font=font(12), fill="#9aa2ac")
        area = (x + 6, y + header_h, x + tile_w - 6, y + header_h + image_h - 6)
        draw.rectangle(area, fill="#000000")
        try:
            image = load_rgb(ROOT / Path(sample["image_rel"]))
        except Exception:
            continue
        image = draw_boxes_on_image(image, sample["boxes"])
        px, py, pw, ph = paste_letterbox(sheet, image, area)
        box_draw = ImageDraw.Draw(sheet)
        lw = max(2, min(pw, ph) // 130)
        for box in sample["boxes"]:
            bx1 = px + (box["cx"] - box["w"] / 2) * pw
            by1 = py + (box["cy"] - box["h"] / 2) * ph
            bx2 = px + (box["cx"] + box["w"] / 2) * pw
            by2 = py + (box["cy"] + box["h"] / 2) * ph
            box_draw.rectangle((bx1, by1, bx2, by2), outline="white", width=lw + 1)
            box_draw.rectangle((bx1, by1, bx2, by2), outline="#ff2d55", width=lw)
        name = fit_text(draw, Path(sample["image_rel"]).name, font(11), tile_w - 12)
        draw.text((x + 6, y + header_h + image_h - 20), name,
                  font=font(11), fill="#e8eaed")

    footer = (f"Generated {generated_at} | read-only audit | "
              "human review required to assign any meaning to class IDs")
    draw.text((margin, height - footer_h + 6),
              fit_text(draw, footer, font(12), width - 2 * margin),
              font=font(12), fill="#8b929b")

    path = OUT / "class_index.png"
    sheet.save(path, format="PNG")
    return path


# ---------------------------------------------------------------------------
# Similarity and report data
# ---------------------------------------------------------------------------


def class_visual_similarity(samples_by_class: dict[int, list[dict]]) -> list[dict]:
    hashes: dict[int, list[int]] = {}
    for cid, samples in samples_by_class.items():
        values = [s["crop_dhash"] for s in samples if s.get("crop_dhash") is not None]
        if values:
            hashes[cid] = values

    rows: list[dict] = []
    for a, b in combinations(sorted(hashes), 2):
        ha, hb = hashes[a], hashes[b]
        matrix = [[hamming(x, y) for y in hb] for x in ha]
        mean_min_ab = sum(min(row) for row in matrix) / len(ha)
        mean_min_ba = sum(min(col) for col in zip(*matrix)) / len(hb)
        mean_min = (mean_min_ab + mean_min_ba) / 2
        close_pairs = sum(1 for row in matrix for value in row if value <= 10)
        rows.append({
            "class_i": a,
            "class_j": b,
            "mean_min_dhash_distance": round(mean_min, 2),
            "pct_crop_pairs_hamming_le_10": round(
                100.0 * close_pairs / (len(ha) * len(hb)), 2
            ),
            "crops_compared": len(ha) * len(hb),
        })
    rows.sort(key=lambda r: (r["mean_min_dhash_distance"],
                             -r["pct_crop_pairs_hamming_le_10"],
                             r["class_i"], r["class_j"]))
    return rows


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------


def write_inventory(samples_by_class: dict[int, list[dict]]) -> None:
    fields = [
        "class_id", "sample_index", "contact_sheet", "image_path", "label_path",
        "batch", "split", "image_width", "image_height",
        "class_boxes_in_image", "all_boxes_in_image",
        "max_box_area_norm", "max_box_area_px", "size_tier",
        "class_area_tertile_q1", "class_area_tertile_q2",
        "available_images_for_label", "image_md5", "crop_dhash_hex",
        "nonblack_fraction",
    ]
    with (OUT / "class_visual_inventory.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for cid in CLASS_IDS:
            for sample in samples_by_class.get(cid, []):
                try:
                    with Image.open(ROOT / Path(sample["image_rel"])) as image:
                        width, height = image.size
                except Exception:
                    width = height = ""
                writer.writerow({
                    "class_id": cid,
                    "sample_index": sample["sample_index"],
                    "contact_sheet": f"class_{cid:02d}_contact_sheet.png",
                    "image_path": sample["image_rel"],
                    "label_path": sample["label_rel"],
                    "batch": sample["batch"],
                    "split": sample["split"],
                    "image_width": width,
                    "image_height": height,
                    "class_boxes_in_image": len(sample["boxes"]),
                    "all_boxes_in_image": sample["all_box_count"],
                    "max_box_area_norm": round(sample["max_area"], 6),
                    "max_box_area_px": (
                        round(sample["max_area"] * width * height)
                        if width and height else ""
                    ),
                    "size_tier": sample["size_tier"],
                    "class_area_tertile_q1": round(sample["tier_q1"], 6),
                    "class_area_tertile_q2": round(sample["tier_q2"], 6),
                    "available_images_for_label": sample["available_images"],
                    "image_md5": sample["image_md5"],
                    "crop_dhash_hex": (
                        f"{sample['crop_dhash']:016x}"
                        if sample.get("crop_dhash") is not None else ""
                    ),
                    "nonblack_fraction": (
                        round(sample["nonblack_fraction"], 4)
                        if sample.get("nonblack_fraction") is not None else ""
                    ),
                })


def write_indexes(samples_by_class: dict[int, list[dict]],
                  summary_rows: list[dict], generated_at: str) -> None:
    # Markdown index
    lines = [
        "# Visual Class Audit - Combined Class Index",
        "",
        "Numeric class IDs only. No class names are inferred or assigned.",
        "",
        f"Generated: {generated_at}",
        "",
        md_table(
            ["Class ID", "Examples", "Unique candidate images", "Batches",
             "Small / Medium / Large", "Contact sheet"],
            [[r["class_id"], r["examples_reviewed"],
              r["unique_candidate_images"], r["batches"],
              f"{r['small']} / {r['medium']} / {r['large']}",
              f"[open]({r['contact_sheet']})"] for r in summary_rows],
        ),
        "",
        "Also see: [combined image index](class_index.png), "
        "[inventory](class_visual_inventory.csv), "
        "[report](visual_audit_report.md).",
        "",
    ]
    (OUT / "class_index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # HTML index
    parts = [
        "<!doctype html>",
        '<html lang="en"><head><meta charset="utf-8">',
        "<title>Visual Class Audit - Class Index</title>",
        "<style>",
        "body{font-family:system-ui,Segoe UI,Arial,sans-serif;margin:24px;"
        "background:#0b0d10;color:#e8eaed}",
        "h1{color:#ffcc00} table{border-collapse:collapse;width:100%;"
        "max-width:1100px} th,td{padding:7px 10px;border-bottom:1px solid #3a3f46;"
        "text-align:left} th{color:#ffcc00} a{color:#7fc4ff} "
        ".note{color:#9aa2ac;max-width:900px}",
        "</style></head><body>",
        "<h1>Visual Class Audit - Combined Class Index</h1>",
        '<p class="note">Numeric class IDs only. No class names are inferred or '
        "assigned. Human review is required to decide what each ID represents. "
        "Appearance alone does not establish that an object is a pedestrian "
        "obstacle.</p>",
        f'<p class="note">Generated: {html.escape(generated_at)} | '
        f'<a href="class_index.png">combined image index</a> | '
        f'<a href="class_visual_inventory.csv">inventory CSV</a> | '
        f'<a href="visual_audit_report.md">report</a></p>',
        "<table><tr><th>Class ID</th><th>Examples</th>"
        "<th>Unique candidate images</th><th>Batches</th>"
        "<th>Small / Medium / Large</th><th>Contact sheet</th></tr>",
    ]
    for row in summary_rows:
        parts.append(
            f"<tr><td>Class {row['class_id']:02d}</td>"
            f"<td>{row['examples_reviewed']}</td>"
            f"<td>{row['unique_candidate_images']}</td>"
            f"<td>{html.escape(row['batches'])}</td>"
            f"<td>{row['small']} / {row['medium']} / {row['large']}</td>"
            f'<td><a href="{html.escape(row["contact_sheet"])}">open PNG</a></td>'
            "</tr>"
        )
    parts += ["</table></body></html>"]
    (OUT / "class_index.html").write_text("\n".join(parts) + "\n", encoding="utf-8")


def write_report(samples_by_class: dict[int, list[dict]],
                 summary_rows: list[dict],
                 similarity_rows: list[dict],
                 inconsistencies: dict,
                 stats: dict,
                 generated_at: str) -> None:
    lines: list[str] = []
    add = lines.append
    add("# Visual Class Audit - Numeric Class IDs")
    add("")
    add(f"- **Dataset:** `{ROOT.as_posix()}`")
    add(f"- **Output:** `{OUT.as_posix()}`")
    add(f"- **Generated:** {generated_at}")
    add("- **Mode:** read-only; no dataset, annotation, class ID or split changed")
    add("")
    add("> This audit shows examples only. It does **not** assign class names "
        "and does **not** decide which classes are pedestrian obstacles. A human "
        "reviewer must interpret each numeric ID from the images.")
    add("")

    add("## 1. Method")
    add("")
    add(f"- Parsed **{stats['label_files']:,}** YOLO label files and "
        f"**{stats['boxes']:,}** bounding boxes across "
        f"{stats['classes']} class IDs (0-34).")
    add(f"- Selected up to **{SAMPLES_PER_CLASS}** examples per class, stratified "
        "by bounding-box size and spread across batches.")
    add("- **Size tiers** are per-class tertiles of the target box area "
        "(`w*h`, normalized): small <= Q1, medium <= Q2, large > Q2. Targets "
        "are 7 small / 7 medium / 6 large, refilled from other tiers when a "
        "tier is exhausted.")
    add("- **Batch coverage:** selection round-robins over Batch 1-4; a final "
        "pass adds or substitutes a sample if a batch is missing and an "
        "alternative exists.")
    add("- **Duplicate avoidance:** exact image duplicates are deduplicated by "
        "MD5; duplicate-stem images share one label, so only one image is used; "
        "images already used by another class are avoided when alternatives "
        "exist.")
    add("- **Drawing:** only boxes of the target class are drawn. Each box is "
        "labelled with the numeric ID. Header shows filename, batch, split, "
        "size tier and box area.")
    add(f"- Selection seed: `{SEED}` (deterministic for this dataset).")
    add("")

    add("## 2. Examples reviewed per class")
    add("")
    add(md_table(
        ["Class ID", "Examples", "Unique candidates", "Batches",
         "Small / Medium / Large", "Contact sheet"],
        [[r["class_id"], r["examples_reviewed"], r["unique_candidate_images"],
          r["batches"], f"{r['small']} / {r['medium']} / {r['large']}",
          f"[PNG]({r['contact_sheet']})"] for r in summary_rows],
    ))
    add("")
    add(f"Total examples selected: **{stats['examples_total']}**. "
        f"Unique source images used: **{stats['unique_images_used']}** "
        f"(some images legitimately appear in more than one class sheet: "
        f"{stats['reused_images']} repeated selections).")
    add("")

    add("## 3. Missing or insufficient examples")
    add("")
    insufficient = [r for r in summary_rows
                    if r["examples_reviewed"] < SAMPLES_PER_CLASS]
    if insufficient:
        add(md_table(
            ["Class ID", "Examples", "Reason"],
            [[r["class_id"], r["examples_reviewed"],
              r["shortfall_reason"]] for r in insufficient],
        ))
    else:
        add(f"**None.** Every class ID 0-34 has {SAMPLES_PER_CLASS} selected "
            "examples, with all available batches represented.")
    add("")
    add(f"Classes with no candidates at all: "
        f"{stats['classes_without_candidates'] or 'none'}.")
    add("")

    add("### Low-content selected images")
    add("")
    low_content = [
        sample
        for samples in samples_by_class.values()
        for sample in samples
        if sample.get("nonblack_fraction", 1.0) < 0.10
    ]
    if low_content:
        add("These selected source images are mostly black or letterboxed. "
            "They are valid source images, not rendering failures: the contact "
            "sheet reproduces them faithfully. They can look almost empty at "
            "thumbnail size, so open the source path to inspect them.")
        add("")
        add(md_table(
            ["Class ID", "Sample", "Image", "Size", "Non-black pixels",
             "Batch / split"],
            [[s["class_id"], s["sample_index"], f"`{s['image_rel']}`",
              f"{s['image_size'][0]}x{s['image_size'][1]}",
              f"{s['nonblack_fraction']:.2%}",
              f"{s['batch']} / {s['split']}"] for s in low_content],
        ))
    else:
        add("No selected source image has less than 10% non-black pixels.")
    add("")

    add("## 4. Classes that appear visually similar")
    add("")
    add("The table below compares the **annotated object crops** using a "
        "difference hash (dHash) of the largest target box per selected image. "
        "A low mean minimum Hamming distance means the crops look visually "
        "similar at a coarse level.")
    add("")
    add("**Important:** this is image-level visual similarity only. It does not "
        "prove that two class IDs mean the same thing, and it must not be used "
        "to merge or rename classes. It is a triage aid for human review.")
    add("")
    add(md_table(
        ["Class i", "Class j", "Mean min dHash distance", "% crop pairs <= 10", "Crops compared"],
        [[r["class_i"], r["class_j"], r["mean_min_dhash_distance"],
          r["pct_crop_pairs_hamming_le_10"], r["crops_compared"]]
         for r in similarity_rows[:12]],
    ))
    add("")
    add("Pairs with the lowest distance should be reviewed side by side in the "
        "two contact sheets before drawing any conclusion.")
    add("")

    add("## 5. Annotation inconsistencies noticed")
    add("")
    counts = inconsistencies["counts"]
    add(md_table(
        ["Heuristic", "Count", "Threshold"],
        [
            ["Malformed label lines", counts["malformed_lines"], "not 5 fields / non-numeric"],
            ["Invalid geometry (w/h outside (0,1] or center outside [0,1])",
             counts["invalid_geometry_boxes"], "YOLO cx,cy,w,h"],
            ["Boxes whose center-based extent exceeds image bounds",
             counts["out_of_bounds_boxes"], "cx+/-w/2, cy+/-h/2"],
            ["Tiny boxes", counts["tiny_boxes"], f"area < {TINY_AREA}"],
            ["Extreme aspect-ratio boxes", counts["extreme_aspect_boxes"],
             f"ratio > {EXTREME_ASPECT_HIGH} or < {EXTREME_ASPECT_LOW}"],
            ["Same-class near-duplicate box pairs", counts["same_class_duplicate_pairs"],
             f"IoU >= {SAME_CLASS_IOU}"],
            ["Cross-class overlapping box pairs", counts["cross_class_overlap_pairs"],
             f"IoU >= {CROSS_CLASS_IOU}"],
        ],
    ))
    add("")
    examples = inconsistencies["examples"]
    if any(examples.values()):
        add("Examples (first few per heuristic):")
        add("")
        for key, values in examples.items():
            if not values:
                continue
            add(f"**{key}**")
            add("")
            for item in values:
                add(f"- `{item['label']}` - {json.dumps({k: v for k, v in item.items() if k != 'label'})}")
            add("")
    else:
        add("No malformed lines, invalid geometry, out-of-bounds boxes, tiny "
            "boxes, extreme aspect ratios, near-duplicate same-class boxes or "
            "cross-class high-IoU overlaps were detected by these heuristics.")
        add("")
    add("These checks are structural only. They cannot detect a semantically "
        "wrong label, a missed object, or two IDs used for the same concept. "
        "Those require human inspection of the contact sheets.")
    add("")
    add(f"Also relevant: **{stats['shared_label_files']}** label files are "
        "shared by multiple image files (duplicate stems / PNG copies). The "
        "audit selects only one image per label to avoid duplicate examples.")
    add("")

    add("### File extension vs actual encoding")
    add("")
    format_rows = stats["format_mismatches"]
    if format_rows:
        add("Some image files are named with an extension that does not match "
            "their actual encoding. This audit reads images by content, so it "
            "is unaffected, but tools that trust file extensions may misread "
            "these files. The files were not renamed.")
        add("")
        add(md_table(
            ["Extension", "Actual encoding", "Count", "Examples"],
            [[f".{r['ext']}", r["actual"], r["count"],
              ", ".join(f"`{x}`" for x in r["examples"])]
             for r in format_rows],
        ))
    else:
        add("All image file extensions match their actual encoding.")
    add("")

    add("## 6. Read-only verification")
    add("")
    add(f"- Dataset files fingerprinted before and after: "
        f"**{stats['source_files']:,}**")
    add(f"- Files modified: **{stats['modified_files']}**")
    add(f"- Dataset unchanged: **{stats['dataset_unchanged']}**")
    add("")

    add("## 7. Obstacle-detection note")
    add("")
    add("No class is treated as a pedestrian obstacle in this audit. Whether a "
        "class is relevant to assistive navigation depends on what the human "
        "reviewer identifies in these images, not on its frequency or "
        "appearance alone. A class that looks like an obstacle can be "
        "irrelevant in context, and a visually ordinary class can matter.")
    add("")

    add("## 8. Outputs")
    add("")
    add(md_table(
        ["File", "Contents"],
        [
            ["`class_00_contact_sheet.png` ... `class_34_contact_sheet.png`",
             "One 20-example contact sheet per class"],
            ["`class_index.png`", "Combined visual index, one example per class"],
            ["`class_index.html` / `class_index.md`", "Clickable class index"],
            ["`class_visual_inventory.csv`", f"{stats['examples_total']} selected examples"],
            ["`visual_audit_report.md`", "This report"],
        ],
    ))
    add("")
    add("## 9. Exact command")
    add("")
    add("```powershell")
    add('python "C:\\vs code\\object detection\\visual_class_audit.py"')
    add("```")
    add("")
    add("## 10. Not done (by rule)")
    add("")
    add("- No dataset image or label modified, moved, renamed or deleted.")
    add("- No class name inferred or assigned.")
    add("- No class treated as an obstacle.")
    add("- No class ID, annotation, coordinate or split changed.")
    add("- No model trained.")
    add("")

    (OUT / "visual_audit_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("Dataset:", ROOT)
    print("Output :", OUT)
    print("Mode   : read-only visual audit")
    print()

    if not ROOT.is_dir():
        raise SystemExit(f"Dataset directory not found: {ROOT}")
    try:
        OUT.resolve().relative_to(ROOT.resolve())
        raise SystemExit(
            "Refusing to write visual audit output inside the dataset directory."
        )
    except ValueError:
        pass

    OUT.mkdir(parents=True, exist_ok=True)
    # Remove only this script's previous contact sheets so stale files cannot
    # be mistaken for current output. Dataset files are untouched.
    for old in OUT.glob("class_??_contact_sheet.png"):
        old.unlink()

    source_before = fingerprint()

    print("[1/6] Parsing annotations")
    label_paths = sorted(p for p in ROOT.rglob("*.txt") if p.is_file())
    image_paths = sorted(
        p for p in ROOT.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )
    if not label_paths:
        raise SystemExit(f"No label files found under {ROOT}")
    records = [parse_label(p) for p in label_paths]
    total_boxes = sum(len(r["boxes"]) for r in records)
    present_ids = sorted({b["class_id"] for r in records for b in r["boxes"]})
    print(f"    label files={len(records):,} images={len(image_paths):,} "
          f"boxes={total_boxes:,} classes={len(present_ids)}")

    format_counts: dict[tuple[str, str], int] = defaultdict(int)
    format_examples: dict[tuple[str, str], list[str]] = defaultdict(list)
    for path in image_paths:
        ext = normalized_extension(path)
        actual = image_signature_format(path)
        if actual != ext:
            key = (ext, actual)
            format_counts[key] += 1
            if len(format_examples[key]) < 3:
                format_examples[key].append(rel_of(path))

    print("[2/6] Mapping labels to images and hashing images")
    image_index = build_image_index()
    label_to_images = {}
    for rec in records:
        key = (rec["batch"], rec["split"], Path(rec["rel"]).stem.casefold())
        label_to_images[rec["rel"]] = image_index.get(key, [])

    image_md5: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {
            pool.submit(md5_file, ROOT / Path(rel_of(path))): rel_of(path)
            for path in image_paths
        }
        done = 0
        for future in as_completed(futures):
            rel = futures[future]
            try:
                image_md5[rel] = future.result()
            except Exception as exc:  # noqa: BLE001
                print(f"    hash error {rel}: {exc}")
            done += 1
            if done % 4000 == 0:
                print(f"    hashed {done:,}/{len(image_paths):,}")

    print("[3/6] Selecting stratified examples")
    candidates_by_class = build_candidates(records, label_to_images, image_md5)
    samples_by_class: dict[int, list[dict]] = {}
    global_used: set[str] = set()
    classes_without_candidates: list[int] = []
    for cid in CLASS_IDS:
        candidates = candidates_by_class.get(cid, [])
        if not candidates:
            classes_without_candidates.append(cid)
            samples_by_class[cid] = []
            continue
        selected = select_for_class(cid, candidates, global_used)
        samples_by_class[cid] = selected
        for sample in selected:
            global_used.add(sample["image_md5"])
        print(f"    class {cid:02d}: {len(selected):2d} examples "
              f"({len(candidates):,} candidates)")

    print("[4/6] Drawing contact sheets and computing crop hashes")
    generated_at = datetime.now().isoformat(timespec="seconds")
    sheet_paths: dict[int, Path] = {}
    for cid in CLASS_IDS:
        sheet_paths[cid] = build_contact_sheet(
            cid, samples_by_class[cid], generated_at
        )
        print(f"    wrote {sheet_paths[cid].name}")

    print("[5/6] Building combined index and similarity table")
    index_samples = {}
    for cid in CLASS_IDS:
        samples = samples_by_class.get(cid, [])
        if samples:
            index_samples[cid] = max(samples, key=lambda s: s["max_area"])
    build_combined_index(index_samples, generated_at)
    similarity_rows = class_visual_similarity(samples_by_class)

    inconsistencies = scan_inconsistencies(records)

    summary_rows = []
    for cid in CLASS_IDS:
        samples = samples_by_class.get(cid, [])
        tier_counts = Counter(s["size_tier"] for s in samples)
        batches = sorted({s["batch"] for s in samples},
                         key=lambda b: BATCHES.index(b) if b in BATCHES else 99)
        unique_candidates = len({
            (c["image_md5"], c["stem_key"])
            for c in candidates_by_class.get(cid, [])
        })
        summary_rows.append({
            "class_id": cid,
            "examples_reviewed": len(samples),
            "unique_candidate_images": unique_candidates,
            "batches": ", ".join(b.replace("Batch ", "B") for b in batches) or "none",
            "small": tier_counts.get("small", 0),
            "medium": tier_counts.get("medium", 0),
            "large": tier_counts.get("large", 0),
            "contact_sheet": f"class_{cid:02d}_contact_sheet.png",
            "shortfall_reason": (
                "no matching annotations" if not samples
                else "not enough unique images after deduplication"
            ),
        })

    source_after = fingerprint()
    changed = sorted(
        rel for rel in set(source_before) | set(source_after)
        if source_before.get(rel) != source_after.get(rel)
    )

    selected_hashes = {
        s["image_md5"] for samples in samples_by_class.values() for s in samples
    }
    examples_total = sum(len(v) for v in samples_by_class.values())
    stats = {
        "label_files": len(records),
        "images": len(image_paths),
        "boxes": total_boxes,
        "classes": len(present_ids),
        "examples_total": examples_total,
        "unique_images_used": len(selected_hashes),
        "reused_images": examples_total - len(selected_hashes),
        "classes_without_candidates": classes_without_candidates,
        "shared_label_files": sum(
            1 for images in label_to_images.values() if len(images) > 1
        ),
        "format_mismatches": [
            {
                "ext": ext,
                "actual": actual,
                "count": count,
                "examples": format_examples[(ext, actual)],
            }
            for (ext, actual), count in sorted(
                format_counts.items(), key=lambda item: -item[1]
            )
        ],
        "source_files": len(source_before),
        "modified_files": len(changed),
        "dataset_unchanged": not changed,
        "changed_files": changed[:20],
    }

    print("[6/6] Writing inventory, indexes and report")
    write_inventory(samples_by_class)
    write_indexes(samples_by_class, summary_rows, generated_at)
    write_report(samples_by_class, summary_rows, similarity_rows,
                 inconsistencies, stats, generated_at)

    if changed:
        raise SystemExit(
            "ERROR: dataset files changed during the audit: "
            + ", ".join(changed[:10])
        )

    print()
    print("Done. Visual audit written to:", OUT)
    print("Contact sheets:", len(list(OUT.glob("class_??_contact_sheet.png"))))
    print("Examples selected:", examples_total)
    print("Unique source images used:", stats["unique_images_used"])
    print("Classes with fewer than 20 examples:",
          [r["class_id"] for r in summary_rows
           if r["examples_reviewed"] < SAMPLES_PER_CLASS] or "none")
    print("Dataset unchanged:", stats["dataset_unchanged"])
    print("Report:", OUT / "visual_audit_report.md")


if __name__ == "__main__":
    main()
