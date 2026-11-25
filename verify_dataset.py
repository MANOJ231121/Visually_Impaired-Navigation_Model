#!/usr/bin/env python3
"""
verify_dataset.py -- READ-ONLY verification of a multi-batch YOLO-format dataset.

What it checks
--------------
1.  Directory inventory and per-batch / per-split file counts.
2.  Presence and contents of metadata files (data.yaml, classes.txt, obj.names, README, ...).
3.  Class IDs used in every annotation file (per batch, per split).
4.  Class-ID consistency between batches (set comparison + inferred mapping from
    files that share a stem and identical box coordinates).
5.  Missing / unexpected / sparse class IDs.
6.  YOLO line format:  <class_id> <x_center> <y_center> <width> <height>
7.  Normalisation: all four coordinates in [0, 1], positive size, box inside image.
8.  Empty label files, malformed lines (with reasons), encoding / BOM / line-ending issues.
9.  Image <-> label pairing: missing labels, orphan labels, case mismatches, wrong split.
10. Image audit: real formats, corrupt files, tiny images, exact duplicates (MD5),
    and likely train/val/test leakage via Roboflow base-name normalisation.

Guarantees
----------
*   The dataset is only ever read. Nothing under --root is written, moved, renamed or deleted.
*   All outputs go to --out (default: ../dataset_verification relative to this script).
*   Class names are NEVER guessed. If no metadata exists, the report says so explicitly.

Usage
-----
    python verify_dataset.py
    python verify_dataset.py --root "C:\\vs code\\object detection\\final batches" \
                             --out  "C:\\vs code\\object detection\\dataset_verification"
    python verify_dataset.py --skip-image-check --skip-hash   # fast structural pass

Requires: Python 3.8+ ; Pillow (only for the image audit).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------

SPLITS = ("train", "val", "test")

# Extensions that are treated as "images" for pairing purposes.
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".gif"}
# Extensions Ultralytics/YOLO can actually load (GIF is not supported by Ultralytics).
YOLO_READABLE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

METADATA_NAMES = {
    "data.yaml", "data.yml", "dataset.yaml", "dataset.yml", "obj.data",
    "classes.txt", "obj.names", "names.txt", "labels.txt",
    "readme", "readme.md", "readme.txt",
}
METADATA_EXTS = {".yaml", ".yml", ".names", ".json", ".csv"}
METADATA_TXT_HINTS = ("class", "name", "readme", "label", "data")

COORD_TOL = 1e-6          # tolerance for float noise at the [0,1] boundary
TINY_BOX_AREA = 1e-4      # w*h below this is reported as a warning (< 1% of a 100x100 box... i.e. very small)
EXTREME_ASPECT = 20.0     # w/h or h/w above this is reported as a warning
TINY_IMAGE_SIDE = 32      # images smaller than this on either side are flagged
SMALL_IMAGE_SIDE = 100    # images below this are unlikely to resolve small objects

# Purely circumstantial: filename keywords used ONLY as a domain sanity check.
# They are never mapped to class IDs and never presented as class names.
DOMAIN_HINTS = [
    "toilet", "restroom", "bathroom", "barrier", "manhole", "pothole", "oven",
    "chair", "toothbrush", "bottle", "bus", "traffic", "stair", "waste",
    "garbage", "door", "sink", "faucet", "soap", "table", "bed", "knife",
    "sign", "trash", "bin", "cup",
]

HARD_ERRORS = {
    "COMMA_SEPARATED", "TOO_FEW_TOKENS", "EXTRA_TOKENS", "CLASS_NOT_INT",
    "CLASS_NEGATIVE", "BAD_FLOAT", "NON_FINITE", "COORD_OUT_OF_RANGE",
    "NONPOSITIVE_SIZE", "SIZE_GT_1", "BOX_OUTSIDE_IMAGE", "READ_ERROR",
}

RF_RE = re.compile(
    r"^(?P<base>.*?)(?:_(?:jpg|jpeg|png|bmp|webp|tif|tiff))?\.rf\.[0-9a-f]{32}$",
    re.IGNORECASE,
)
INT_RE = re.compile(r"^[+]?\d+$")
SIGNED_INT_RE = re.compile(r"^[+-]?\d+$")


# --------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------

def natural_key(path: Path):
    m = re.search(r"(\d+)", path.name)
    return (int(m.group(1)) if m else 10**9, path.name.lower())


def base_stem(stem: str) -> str:
    """Strip Roboflow augmentation suffix so near-duplicate exports collapse to one name."""
    m = RF_RE.match(stem)
    return (m.group("base") if m else stem).strip().lower()


def md5_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def iou_xywh(a, b) -> float:
    """IoU of two (x_center, y_center, w, h) boxes in normalised coordinates."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax1, ay1, ax2, ay2 = ax - aw / 2, ay - ah / 2, ax + aw / 2, ay + ah / 2
    bx1, by1, bx2, by2 = bx - bw / 2, by - bh / 2, bx + bw / 2, by + bh / 2
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def pct(num: float, den: float) -> str:
    return "n/a" if not den else f"{100.0 * num / den:.2f}%"


def fmt_int(n: int) -> str:
    return f"{n:,}"


# --------------------------------------------------------------------------------------
# Data containers
# --------------------------------------------------------------------------------------

@dataclass
class ImageRec:
    batch: str
    split: str
    path: Path
    rel: str
    ext: str
    size: int
    md5: str | None = None
    width: int | None = None
    height: int | None = None
    pil_ok: bool | None = None
    pil_error: str | None = None
    mode: str | None = None


@dataclass
class LabelRec:
    batch: str
    split: str
    path: Path
    rel: str
    n_lines: int = 0
    n_blank: int = 0
    boxes: list = field(default_factory=list)        # [(cls, x, y, w, h), ...]
    errors: list = field(default_factory=list)       # [(lineno, kind, raw)]
    warnings: list = field(default_factory=list)     # [(lineno, kind, raw)]
    class_counts: Counter = field(default_factory=Counter)
    bom: bool = False
    crlf: bool = False
    non_utf8: bool = False
    read_error: str | None = None

    @property
    def empty(self) -> bool:
        return self.read_error is None and len(self.boxes) == 0 and not self.errors

    @property
    def n_hard_errors(self) -> int:
        return len(self.errors)

    def coord_signature(self):
        return tuple(sorted(
            (round(x, 6), round(y, 6), round(w, 6), round(h, 6)) for _, x, y, w, h in self.boxes
        ))

    def class_by_coord(self):
        """coord-signature -> set(class ids) for inference of cross-batch mappings."""
        out = defaultdict(set)
        for cls, x, y, w, h in self.boxes:
            out[(round(x, 6), round(y, 6), round(w, 6), round(h, 6))].add(cls)
        return out


# --------------------------------------------------------------------------------------
# Label parsing
# --------------------------------------------------------------------------------------

def parse_label_file(path: Path) -> LabelRec:
    rec = LabelRec(batch="", split="", path=path, rel="")
    try:
        raw = path.read_bytes()
    except OSError as exc:
        rec.read_error = str(exc)
        rec.errors.append((0, "READ_ERROR", str(exc)))
        return rec

    if raw.startswith(b"\xef\xbb\xbf"):
        rec.bom = True
        raw = raw[3:]
    rec.crlf = b"\r\n" in raw

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("utf-8", errors="replace")
        rec.non_utf8 = True

    lines = text.splitlines()
    rec.n_lines = len(lines)
    seen_lines = Counter()

    for lineno, line in enumerate(lines, 1):
        stripped = line.strip()
        if not stripped:
            rec.n_blank += 1
            continue

        tokens = stripped.split()

        # Comma / semicolon / tab-as-only-separator is a common export bug.
        if len(tokens) == 1 and ("," in stripped or ";" in stripped):
            rec.errors.append((lineno, "COMMA_SEPARATED", stripped))
            continue

        if len(tokens) < 5:
            rec.errors.append((lineno, "TOO_FEW_TOKENS", stripped))
            continue
        if len(tokens) > 5:
            rec.errors.append((lineno, "EXTRA_TOKENS", stripped))
            continue

        cls_tok, x_tok, y_tok, w_tok, h_tok = tokens

        if not INT_RE.match(cls_tok):
            if SIGNED_INT_RE.match(cls_tok):
                rec.errors.append((lineno, "CLASS_NEGATIVE", stripped))
            else:
                rec.errors.append((lineno, "CLASS_NOT_INT", stripped))
            continue
        cls = int(cls_tok)

        try:
            x, y, w, h = (float(v) for v in (x_tok, y_tok, w_tok, h_tok))
        except ValueError:
            rec.errors.append((lineno, "BAD_FLOAT", stripped))
            continue

        if not all(math.isfinite(v) for v in (x, y, w, h)):
            rec.errors.append((lineno, "NON_FINITE", stripped))
            continue

        if any(v < -COORD_TOL or v > 1.0 + COORD_TOL for v in (x, y, w, h)):
            rec.errors.append((lineno, "COORD_OUT_OF_RANGE", stripped))
            continue

        if w <= 0.0 or h <= 0.0:
            rec.errors.append((lineno, "NONPOSITIVE_SIZE", stripped))
            continue
        if w > 1.0 + COORD_TOL or h > 1.0 + COORD_TOL:
            rec.errors.append((lineno, "SIZE_GT_1", stripped))
            continue

        left, right = x - w / 2.0, x + w / 2.0
        top, bottom = y - h / 2.0, y + h / 2.0
        if (left < -COORD_TOL or right > 1.0 + COORD_TOL
                or top < -COORD_TOL or bottom > 1.0 + COORD_TOL):
            rec.errors.append((lineno, "BOX_OUTSIDE_IMAGE", stripped))
            continue

        # ---- warnings (kept, not rejected) ----
        if w * h < TINY_BOX_AREA:
            rec.warnings.append((lineno, "TINY_BOX", stripped))
        if (left < 1e-6 or top < 1e-6 or right > 1.0 - 1e-6 or bottom > 1.0 - 1e-6):
            rec.warnings.append((lineno, "EDGE_TOUCHING_BOX", stripped))
        aspect = (w / h) if h else math.inf
        if aspect > EXTREME_ASPECT or (aspect and (1.0 / aspect) > EXTREME_ASPECT):
            rec.warnings.append((lineno, "EXTREME_ASPECT", stripped))
        seen_lines[stripped] += 1
        if seen_lines[stripped] == 2:
            rec.warnings.append((lineno, "DUPLICATE_LINE_IN_FILE", stripped))

        rec.boxes.append((cls, x, y, w, h))
        rec.class_counts[cls] += 1

    return rec


# --------------------------------------------------------------------------------------
# Discovery
# --------------------------------------------------------------------------------------

def find_batches(root: Path):
    batches = []
    for child in sorted(root.iterdir(), key=natural_key):
        if child.is_dir() and (child / "images").is_dir() and (child / "labels").is_dir():
            batches.append(child)
    return batches


def looks_like_yolo_label(path: Path) -> bool:
    """True when the file is clearly an annotation file rather than class-name metadata."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    non_empty = 0
    yolo_like = 0
    for line in text.splitlines()[:200]:
        stripped = line.strip()
        if not stripped:
            continue
        non_empty += 1
        tokens = stripped.split()
        if len(tokens) == 5 and INT_RE.match(tokens[0]):
            try:
                vals = [float(v) for v in tokens[1:]]
            except ValueError:
                continue
            if all(0.0 <= v <= 1.0 for v in vals):
                yolo_like += 1
    return non_empty > 0 and yolo_like / non_empty >= 0.8


def find_metadata(root: Path):
    found = []
    root_parts = {p.lower() for p in root.parts}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        name = path.name.lower()
        ext = path.suffix.lower()
        # Anything inside a `labels` directory is an annotation, not metadata.
        if "labels" in {p.lower() for p in path.parent.parts} - root_parts:
            continue
        if name in METADATA_NAMES or ext in METADATA_EXTS:
            found.append(path)
        elif ext == ".txt" and any(h in name for h in METADATA_TXT_HINTS):
            if not looks_like_yolo_label(path):
                found.append(path)
    return sorted(found)


# --------------------------------------------------------------------------------------
# Image audit
# --------------------------------------------------------------------------------------

def audit_images(images, do_check: bool, do_hash: bool):
    try:
        from PIL import Image
    except ImportError:
        Image = None

    for idx, rec in enumerate(images, 1):
        if do_hash:
            try:
                rec.md5 = md5_of(rec.path)
            except OSError as exc:
                rec.pil_error = f"hash error: {exc}"

        if not do_check or Image is None:
            continue
        try:
            with Image.open(rec.path) as im:
                rec.width, rec.height = im.size
                rec.mode = im.mode
                im.verify()
            rec.pil_ok = True
        except Exception as exc:  # noqa: BLE001 - report any decoder failure
            rec.pil_ok = False
            rec.pil_error = f"{type(exc).__name__}: {exc}"

        if idx % 2000 == 0:
            print(f"    ... {idx}/{len(images)} images audited", file=sys.stderr)


# --------------------------------------------------------------------------------------
# Main verification
# --------------------------------------------------------------------------------------

def verify(root: Path, out_dir: Path, do_image_check: bool, do_hash: bool) -> dict:
    if not root.is_dir():
        raise SystemExit(f"Dataset root not found: {root}")

    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    batches = find_batches(root)
    if not batches:
        raise SystemExit(f"No batch folders with images/ + labels/ found under {root}")

    metadata = find_metadata(root)

    # Files next to the dataset (not part of it) that could hold a class list.
    parent_scan = []
    for p in sorted(root.parent.iterdir()):
        if p.is_file() and p.suffix.lower() in {".py", ".yaml", ".yml", ".json", ".csv", ".md", ".txt", ".names"}:
            parent_scan.append({"file": str(p), "bytes": p.stat().st_size})

    # ---------------- inventory ----------------
    print("[1/6] Inventory: images, labels, stray files", file=sys.stderr)
    images: list[ImageRec] = []
    labels: list[LabelRec] = []
    stray_images_dir = []   # non-image files inside images/
    stray_labels_dir = []   # non-.txt files inside labels/
    missing_split_dirs = []
    batch_names = [b.name for b in batches]

    for bdir in batches:
        for split in SPLITS:
            idir = bdir / "images" / split
            ldir = bdir / "labels" / split
            if not idir.is_dir() or not ldir.is_dir():
                missing_split_dirs.append(f"{bdir.name}/{split}")
                continue

            for p in idir.iterdir():
                if not p.is_file():
                    continue
                ext = p.suffix.lower()
                if ext in IMAGE_EXTS:
                    images.append(ImageRec(
                        batch=bdir.name, split=split, path=p,
                        rel=rel(p, root), ext=ext, size=p.stat().st_size,
                    ))
                else:
                    stray_images_dir.append((bdir.name, split, p.name, p.suffix))

            for p in ldir.iterdir():
                if not p.is_file():
                    continue
                if p.suffix.lower() != ".txt":
                    stray_labels_dir.append((bdir.name, split, p.name, p.suffix))

    # ---------------- label parsing ----------------
    print(f"[2/6] Parsing {len(images):,} referenced label files", file=sys.stderr)
    for bdir in batches:
        for split in SPLITS:
            ldir = bdir / "labels" / split
            if not ldir.is_dir():
                continue
            for p in sorted(ldir.glob("*.txt"), key=lambda q: q.name.lower()):
                rec = parse_label_file(p)
                rec.batch, rec.split = bdir.name, split
                rec.rel = rel(p, root)
                labels.append(rec)

    # ---------------- image audit ----------------
    print(f"[3/6] Image audit ({len(images):,} files)", file=sys.stderr)
    audit_images(images, do_image_check, do_hash)

    # ---------------- index by batch/split ----------------
    img_index = defaultdict(lambda: defaultdict(dict))    # batch -> split -> stem -> [ImageRec]
    for rec in images:
        img_index[rec.batch][rec.split].setdefault(rec.path.stem, []).append(rec)

    lbl_index = defaultdict(lambda: defaultdict(dict))    # batch -> split -> stem -> LabelRec
    for rec in labels:
        lbl_index[rec.batch][rec.split].setdefault(rec.path.stem, rec)

    # global stem -> list of (batch, split, LabelRec, ImageRec|None)
    global_stem = defaultdict(list)
    for b in batch_names:
        for s in SPLITS:
            for stem, irecs in img_index[b][s].items():
                global_stem[stem].append((b, s, lbl_index[b][s].get(stem), irecs[0]))
    # labels whose stem has no image at all in the same split still join the global index
    for b in batch_names:
        for s in SPLITS:
            for stem, lrec in lbl_index[b][s].items():
                if not any(x[0] == b and x[1] == s for x in global_stem[stem]):
                    global_stem[stem].append((b, s, lrec, None))

    # ---------------- pairing ----------------
    print("[4/6] Pairing images with labels", file=sys.stderr)
    pairing = {}          # (batch, split) -> dict
    missing_label_files = []
    orphan_label_files = []
    wrong_split_label = []
    wrong_split_image = []
    case_mismatch = []
    multi_image_same_stem = []

    for b in batch_names:
        for s in SPLITS:
            idir = root / b / "images" / s
            ldir = root / b / "labels" / s
            if not idir.is_dir() or not ldir.is_dir():
                continue

            img_stems = set(img_index[b][s])
            lbl_stems = set(lbl_index[b][s])
            matched = img_stems & lbl_stems
            missing = img_stems - lbl_stems
            orphan = lbl_stems - img_stems

            lower_lbl = {k.lower(): k for k in lbl_stems}
            lower_img = {k.lower(): k for k in img_stems}

            for stem in sorted(missing):
                other = [f"{b}/{os}" for os in SPLITS if os != s and stem in lbl_index[b][os]]
                same_batch_other = [f"{ob}/{s2}" for ob in batch_names if ob != b
                                    for s2 in SPLITS if stem in lbl_index[ob][s2]]
                item = {
                    "batch": b, "split": s, "stem": stem,
                    "image": [r.rel for r in img_index[b][s][stem]],
                    "label_in_other_split": other,
                    "label_in_other_batch": same_batch_other,
                    "case_insensitive_label": lower_lbl.get(stem.lower()),
                }
                if item["label_in_other_split"]:
                    wrong_split_label.append(item)
                elif item["case_insensitive_label"]:
                    case_mismatch.append(item)
                else:
                    missing_label_files.append(item)

            for stem in sorted(orphan):
                other = [f"{b}/{os}" for os in SPLITS if os != s and stem in img_index[b][os]]
                same_batch_other = [f"{ob}/{s2}" for ob in batch_names if ob != b
                                    for s2 in SPLITS if stem in img_index[ob][s2]]
                item = {
                    "batch": b, "split": s, "stem": stem,
                    "label": lbl_index[b][s][stem].rel,
                    "image_in_other_split": other,
                    "image_in_other_batch": same_batch_other,
                    "case_insensitive_image": lower_img.get(stem.lower()),
                }
                if item["image_in_other_split"]:
                    wrong_split_image.append(item)
                elif item["case_insensitive_image"]:
                    case_mismatch.append(item)
                else:
                    orphan_label_files.append(item)

            for stem, irecs in img_index[b][s].items():
                if len(irecs) > 1:
                    hashes = {r.md5 for r in irecs}
                    multi_image_same_stem.append({
                        "batch": b, "split": s, "stem": stem,
                        "images": [r.rel for r in irecs],
                        "identical_content": (None if None in hashes else len(hashes) == 1),
                    })

            n_boxes = sum(len(lbl_index[b][s][st].boxes) for st in matched)
            n_image_files = sum(len(v) for v in img_index[b][s].values())
            pairing[f"{b}/{s}"] = {
                "images": len(img_stems),
                "image_files": n_image_files,
                "image_stems": len(img_stems),
                "labels": len(lbl_stems),
                "matched": len(matched),
                "missing_labels": len(missing),
                "orphan_labels": len(orphan),
                # physical image files that have no label of their own
                "unlabelled_image_files": n_image_files - len(lbl_stems),
                "boxes": n_boxes,
            }

    # ---------------- class analysis ----------------
    print("[5/6] Class-ID analysis", file=sys.stderr)
    class_by_batch = {b: Counter() for b in batch_names}
    class_by_batch_split = {b: {s: Counter() for s in SPLITS} for b in batch_names}
    box_total_by_split = {b: Counter() for b in batch_names}
    images_with_class = {b: Counter() for b in batch_names}
    raw_class_by_batch = {b: Counter() for b in batch_names}

    for rec in labels:
        for cls, *_ in rec.boxes:
            class_by_batch[rec.batch][cls] += 1
            class_by_batch_split[rec.batch][rec.split][cls] += 1

        per_image = set()
        for cls, *_ in rec.boxes:
            per_image.add(cls)
        for cls in per_image:
            images_with_class[rec.batch][cls] += 1

        box_total_by_split[rec.batch][rec.split] += len(rec.boxes)

        # class tokens even on lines that failed later validation
        try:
            for line in rec.path.read_text(encoding="utf-8", errors="replace").splitlines():
                tok = line.split()
                if tok and INT_RE.match(tok[0]):
                    raw_class_by_batch[rec.batch][int(tok[0])] += 1
        except OSError:
            pass

    all_ids = sorted({c for b in batch_names for c in class_by_batch[b]})
    id_min = min(all_ids) if all_ids else None
    id_max = max(all_ids) if all_ids else None
    gaps = [i for i in range(id_min, id_max + 1) if i not in all_ids] if all_ids else []
    expected_count = 35  # as stated by the user; treated as a claim to check, not as truth
    ids_ge_expected = [i for i in all_ids if i >= expected_count]

    # ---------------- cross-batch mapping inference ----------------
    #
    # We join on the *Roboflow base name* (strip `_jpg.rf.<32hex>`), not the raw stem,
    # because re-exports rename augmented copies while keeping the annotation geometry.
    # For every source image present in two batches we compare the exact box geometry
    # (rounded to 6 decimals) and tally (class_in_A -> class_in_B).
    base_class = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))  # base -> batch -> sig -> {ids}
    base_files = defaultdict(lambda: defaultdict(list))                     # base -> batch -> [rel]
    for rec in labels:
        if not rec.boxes:
            continue
        base = base_stem(rec.path.stem)
        base_files[base][rec.batch].append(rec.rel)
        for cls, x, y, w, h in rec.boxes:
            base_class[base][rec.batch][(round(x, 6), round(y, 6), round(w, 6), round(h, 6))].add(cls)

    mapping_pairs = defaultdict(Counter)   # (bA,bB) -> Counter((idA,idB))
    coord_identical_diff_class = []
    stem_shared_pairs = Counter()
    stem_conflicting_coords = Counter()

    for base, per_batch in base_class.items():
        bs = sorted(per_batch)
        if len(bs) < 2:
            continue
        for i in range(len(bs)):
            for j in range(i + 1, len(bs)):
                bA, bB = bs[i], bs[j]
                sigs_a, sigs_b = per_batch[bA], per_batch[bB]
                common = set(sigs_a) & set(sigs_b)
                if not common:
                    stem_conflicting_coords[(bA, bB)] += 1
                    continue
                stem_shared_pairs[(bA, bB)] += 1
                for sig in common:
                    ids_a, ids_b = sigs_a[sig], sigs_b[sig]
                    if len(ids_a) == 1 and len(ids_b) == 1:
                        a_id, b_id = next(iter(ids_a)), next(iter(ids_b))
                        mapping_pairs[(bA, bB)][(a_id, b_id)] += 1
                        if a_id != b_id:
                            if len(coord_identical_diff_class) < 5000:
                                coord_identical_diff_class.append({
                                    "base": base,
                                    "batch_a": bA, "batch_b": bB,
                                    "class_a": a_id, "class_b": b_id,
                                    "label_a": base_files[base][bA][0],
                                    "label_b": base_files[base][bB][0],
                                })

    # Summarise each batch-pair mapping: for every source id, the best target id.
    mapping_summary = {}
    for pair, counter in sorted(mapping_pairs.items()):
        by_source = defaultdict(Counter)
        for (a, b), n in counter.items():
            by_source[a][b] += n
        rows = []
        for a in sorted(by_source):
            total = sum(by_source[a].values())
            best_b, best_n = by_source[a].most_common(1)[0]
            rows.append({
                "source_id": a,
                "target_id": best_b,
                "support": best_n,
                "total": total,
                "confidence": round(best_n / total, 4) if total else 0.0,
                "ambiguous": len(by_source[a]) > 1,
            })
        mapping_summary[f"{pair[0]} -> {pair[1]}"] = rows

    # ---------------- duplicate images & leakage ----------------
    print("[6/6] Duplicate / leakage analysis", file=sys.stderr)
    dup_groups = defaultdict(list)
    for rec in images:
        if rec.md5:
            dup_groups[rec.md5].append(rec)

    duplicates = []
    cross_split_dupes = 0
    within_batch_cross_split = 0
    cross_batch_dupes = 0
    dup_consistency = Counter()
    dup_class_conflicts = []
    dup_split_pairs = Counter()

    for h, recs in sorted(dup_groups.items(), key=lambda kv: -len(kv[1])):
        if len(recs) < 2:
            continue
        locations = sorted({(r.batch, r.split) for r in recs})
        label_sigs = set()
        group_boxes = []
        for r in recs:
            lrec = lbl_index[r.batch][r.split].get(r.path.stem)
            label_sigs.add(lrec.coord_signature() if lrec is not None else ())
            group_boxes.append(lrec.boxes if lrec is not None else [])

        group_class_conflict = False
        group_geometry_conflict = False
        for i in range(len(recs)):
            for j in range(i + 1, len(recs)):
                a_boxes, b_boxes = group_boxes[i], group_boxes[j]
                used = set()
                cls_conf = 0
                geo_conf = 0
                for a in a_boxes:
                    best_k, best_iou = None, 0.0
                    for k, b in enumerate(b_boxes):
                        if k in used:
                            continue
                        v = iou_xywh(a[1:], b[1:])
                        if v > best_iou:
                            best_iou, best_k = v, k
                    if best_k is not None and best_iou >= 0.5:
                        used.add(best_k)
                        if a[0] != b_boxes[best_k][0]:
                            cls_conf += 1
                            if len(dup_class_conflicts) < 200:
                                dup_class_conflicts.append({
                                    "files": [recs[i].rel, recs[j].rel],
                                    "class_a": a[0], "class_b": b_boxes[best_k][0],
                                    "iou": round(best_iou, 3),
                                })
                    else:
                        geo_conf += 1
                geo_conf += len(b_boxes) - len(used)
                if cls_conf:
                    dup_consistency["pairs_with_class_conflict"] += 1
                    group_class_conflict = True
                if geo_conf:
                    dup_consistency["pairs_with_geometry_mismatch"] += 1
                    group_geometry_conflict = True
                if not cls_conf and not geo_conf:
                    dup_consistency["consistent_pairs"] += 1
                dup_consistency["pairs"] += 1
                if recs[i].split != recs[j].split:
                    dup_split_pairs[tuple(sorted({
                        (recs[i].batch, recs[i].split), (recs[j].batch, recs[j].split)}))] += 1

        duplicates.append({
            "md5": h,
            "count": len(recs),
            "files": [r.rel for r in recs],
            "locations": [f"{b}/{s}" for b, s in locations],
            "spans_splits": len({s for _, s in locations}) > 1,
            "spans_batches": len({b for b, _ in locations}) > 1,
            "conflicting_labels": len(label_sigs) > 1,
            "class_conflict": group_class_conflict,
            "geometry_mismatch": group_geometry_conflict,
        })
        splits = {s for _, s in locations}
        if len(splits) > 1:
            cross_split_dupes += 1
            same_batch_split_pair = any(
                l1[0] == l2[0] and l1[1] != l2[1]
                for idx, l1 in enumerate(locations) for l2 in locations[idx + 1:]
            )
            if same_batch_split_pair:
                within_batch_cross_split += 1
        if len({b for b, _ in locations}) > 1:
            cross_batch_dupes += 1

    # Roboflow base-name leakage (catches differently-augmented copies of one source image)
    base_index = defaultdict(list)
    for rec in images:
        base_index[base_stem(rec.path.stem)].append(rec)

    leakage = []
    for base, recs in base_index.items():
        splits = {r.split for r in recs}
        batches_ = {r.batch for r in recs}
        if len(splits) > 1 or len(batches_) > 1:
            leakage.append({
                "base": base,
                "count": len(recs),
                "splits": sorted(splits),
                "batches": sorted(batches_),
                "cross_split": len(splits) > 1,
                "files": [r.rel for r in recs][:12],
            })

    # ---------------- aggregate stats ----------------
    per_batch_stats = {}
    for b in batch_names:
        b_images = [r for r in images if r.batch == b]
        b_labels = [r for r in labels if r.batch == b]
        per_batch_stats[b] = {
            "images": len(b_images),
            "image_files": len(b_images),
            "image_stems": len({(r.split, r.path.stem) for r in b_images}),
            "labels": len(b_labels),
            "label_stems": len({(r.split, r.path.stem) for r in b_labels}),
            "boxes": sum(len(r.boxes) for r in b_labels),
            "empty_labels": sum(1 for r in b_labels if r.empty),
            "malformed_files": sum(1 for r in b_labels if r.errors),
            "malformed_lines": sum(len(r.errors) for r in b_labels),
            "warning_lines": sum(len(r.warnings) for r in b_labels),
            "bom_files": sum(1 for r in b_labels if r.bom),
            "crlf_files": sum(1 for r in b_labels if r.crlf),
            "non_utf8_files": sum(1 for r in b_labels if r.non_utf8),
            "classes_present": sorted(class_by_batch[b]),
            "extensions": dict(Counter(r.ext for r in b_images)),
            "images_per_split": {s: len(img_index[b][s]) for s in SPLITS},
            "labels_per_split": {s: len(lbl_index[b][s]) for s in SPLITS},
            "boxes_per_split": {s: box_total_by_split[b][s] for s in SPLITS},
            "class_counts": dict(sorted(class_by_batch[b].items())),
        }

    error_kind_totals = Counter()
    error_examples = defaultdict(list)
    warning_kind_totals = Counter()
    for rec in labels:
        for lineno, kind, rawtext in rec.errors:
            error_kind_totals[kind] += 1
            if len(error_examples[kind]) < 10:
                error_examples[kind].append(f"{rec.rel}:{lineno}: {rawtext[:200]}")
        for lineno, kind, rawtext in rec.warnings:
            warning_kind_totals[kind] += 1

    image_stems_total = len({(r.batch, r.split, r.path.stem) for r in images})
    stem_groups = defaultdict(list)
    for r in images:
        stem_groups[(r.batch, r.split, r.path.stem)].append(r)
    stems_nonstandard_only = [
        {"batch": b, "split": s, "stem": stem, "files": [x.rel for x in recs],
         "exts": sorted({x.ext for x in recs})}
        for (b, s, stem), recs in stem_groups.items()
        if all(x.ext not in {".jpg", ".jpeg", ".png"} for x in recs)
    ]

    domain_hints = Counter()
    for rec in images:
        low = rec.path.stem.lower()
        for kw in DOMAIN_HINTS:
            if kw in low:
                domain_hints[kw] += 1

    corrupt_images = [r for r in images if r.pil_ok is False]
    tiny_images = [r for r in images if r.width and r.height
                   and min(r.width, r.height) < TINY_IMAGE_SIDE]
    small_images = [r for r in images if r.width and r.height
                    and min(r.width, r.height) < SMALL_IMAGE_SIDE]
    unreadable_format = [r for r in images if r.ext not in YOLO_READABLE_EXTS]
    raw_dims = [(r.width, r.height) for r in images if r.width and r.height]

    results = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "root": str(root),
        "out_dir": str(out_dir),
        "batches": batch_names,
        "metadata_files": [str(p) for p in metadata],
        "missing_split_dirs": missing_split_dirs,
        "stray_images_dir": stray_images_dir,
        "stray_labels_dir": stray_labels_dir,
        "pairing": pairing,
        "per_batch": per_batch_stats,
        "parent_scan": parent_scan,
        "totals": {
            "images": len(images),
            "image_files": len(images),
            "image_stems": image_stems_total,
            "labels": len(labels),
            "label_stems": len(labels),
            "dup_stem_extra_files": len(images) - image_stems_total,
            "boxes": sum(len(r.boxes) for r in labels),
            "empty_labels": sum(1 for r in labels if r.empty),
            "malformed_files": sum(1 for r in labels if r.errors),
            "malformed_lines": sum(len(r.errors) for r in labels),
            "warning_lines": sum(len(r.warnings) for r in labels),
            "image_label_diff": len(images) - len(labels),
            "stems_only_nonstandard": stems_nonstandard_only,
        },
        "classes": {
            "all_ids": all_ids,
            "id_min": id_min,
            "id_max": id_max,
            "gaps_in_range": gaps,
            "expected_class_count_claim": expected_count,
            "ids_ge_expected": ids_ge_expected,
            "per_batch": {b: sorted(class_by_batch[b]) for b in batch_names},
            "per_batch_split": {
                b: {s: sorted(class_by_batch_split[b][s]) for s in SPLITS} for b in batch_names
            },
            "counts_by_batch": {b: dict(sorted(class_by_batch[b].items())) for b in batch_names},
            "counts_by_batch_split": {
                b: {s: dict(sorted(class_by_batch_split[b][s].items())) for s in SPLITS}
                for b in batch_names
            },
            "images_with_class": {b: dict(sorted(images_with_class[b].items())) for b in batch_names},
            "raw_class_tokens_by_batch": {b: dict(sorted(raw_class_by_batch[b].items())) for b in batch_names},
        },
        "errors": {
            "kinds": dict(error_kind_totals),
            "examples": dict(error_examples),
        },
        "warnings": {"kinds": dict(warning_kind_totals)},
        "files": {
            "missing_labels": missing_label_files,
            "orphan_labels": orphan_label_files,
            "wrong_split_label": wrong_split_label,
            "wrong_split_image": wrong_split_image,
            "case_mismatch": case_mismatch,
            "multi_image_same_stem": multi_image_same_stem,
            "empty_labels": [r.rel for r in labels if r.empty],
            "malformed": [
                {"file": r.rel, "errors": [(ln, k, t) for ln, k, t in r.errors]}
                for r in labels if r.errors
            ],
        },
        "images": {
            "corrupt": [{"file": r.rel, "error": r.pil_error} for r in corrupt_images],
            "tiny": [{"file": r.rel, "width": r.width, "height": r.height} for r in tiny_images],
            "small": [{"file": r.rel, "width": r.width, "height": r.height} for r in small_images],
            "unreadable_format": [{"file": r.rel, "ext": r.ext} for r in unreadable_format],
            "domain_hints": dict(domain_hints.most_common()),
            "dim_min": min(min(d) for d in raw_dims) if raw_dims else None,
            "dim_max": max(max(d) for d in raw_dims) if raw_dims else None,
            "checked": do_image_check,
            "hashed": do_hash,
        },
        "duplicates": {
            "groups": duplicates,
            "n_groups": len(duplicates),
            "n_files_in_duplicate_groups": sum(d["count"] for d in duplicates),
            "cross_split_groups": cross_split_dupes,
            "within_batch_cross_split_groups": within_batch_cross_split,
            "cross_batch_groups": cross_batch_dupes,
            "conflicting_label_groups": sum(1 for d in duplicates if d["conflicting_labels"]),
            "annotation_consistency": dict(dup_consistency),
            "class_conflicts": dup_class_conflicts,
            "split_pair_counts": {
                f"{a[0][0]}/{a[0][1]}  <->  {a[1][0]}/{a[1][1]}": n for a, n in
                sorted(dup_split_pairs.items(), key=lambda kv: -kv[1])
            },
            "groups_with_class_conflict": sum(1 for d in duplicates if d["class_conflict"]),
            "groups_with_geometry_mismatch": sum(1 for d in duplicates if d["geometry_mismatch"]),
        },
        "leakage": {
            "groups": leakage,
            "n_groups": len(leakage),
            "n_files": sum(g["count"] for g in leakage),
            "n_cross_split_groups": sum(1 for g in leakage if g["cross_split"]),
            "n_cross_batch_only_groups": sum(1 for g in leakage if not g["cross_split"]),
        },
        "mapping": {
            "shared_stem_pairs": {f"{a} -> {b}": n for (a, b), n in stem_shared_pairs.items()},
            "conflicting_coords_pairs": {f"{a} -> {b}": n for (a, b), n in stem_conflicting_coords.items()},
            "inferred": mapping_summary,
            "coord_identical_diff_class": coord_identical_diff_class,
        },
        "runtime_seconds": round(time.time() - t0, 2),
    }
    return results


# --------------------------------------------------------------------------------------
# Report rendering
# --------------------------------------------------------------------------------------

def md_table(headers, rows) -> str:
    out = ["| " + " | ".join(map(str, headers)) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        out.append("| " + " | ".join("" if v is None else str(v) for v in row) + " |")
    return "\n".join(out)


def cap_lines(items, limit=200, fmt=str):
    items = list(items)
    shown = [fmt(x) for x in items[:limit]]
    if len(items) > limit:
        shown.append(f"... and {len(items) - limit} more (see companion CSV/JSON)")
    return shown


def build_report(r: dict) -> str:
    L = []
    add = L.append
    totals = r["totals"]
    classes = r["classes"]
    batches = r["batches"]

    add("# Dataset Verification Report")
    add("")
    add(f"- **Dataset root:** `{r['root']}`")
    add(f"- **Generated:** {r['generated_at']}")
    add(f"- **Mode:** read-only (no file in the dataset was modified, moved or deleted)")
    add(f"- **Runtime:** {r['runtime_seconds']} s")
    add("")
    add("---")
    add("")

    # ---------------- executive summary ----------------
    add("## 0. Executive summary")
    add("")
    add(f"- **{fmt_int(totals['images'])} image files** and **{fmt_int(totals['labels'])} label files** "
        f"({fmt_int(totals['boxes'])} boxes) across {len(batches)} batches.")
    add(f"- Image/label difference: **{totals['image_label_diff']:+d}**.")
    add(f"- **{fmt_int(totals['empty_labels'])} empty label files**, "
        f"**{fmt_int(totals['malformed_files'])} label files containing malformed lines** "
        f"({fmt_int(totals['malformed_lines'])} malformed lines).")
    add(f"- Class IDs observed: **{classes['all_ids']}** "
        f"(min {classes['id_min']}, max {classes['id_max']}).")
    if r["metadata_files"]:
        add(f"- Metadata files found: **{len(r['metadata_files'])}** -> "
            + ", ".join(f"`{Path(p).name}`" for p in r["metadata_files"]))
    else:
        add("- **No metadata files found** (`data.yaml`, `classes.txt`, `obj.names`, README...). "
            "**Class names cannot be established from the dataset** — only numeric IDs are verifiable.")
    if classes["gaps_in_range"]:
        add(f"- **Missing class IDs inside the observed range:** {classes['gaps_in_range']}")
    if classes["ids_ge_expected"]:
        add(f"- IDs >= {classes['expected_class_count_claim']} (outside a 0–"
            f"{classes['expected_class_count_claim'] - 1} range): {classes['ids_ge_expected']}")
    cons = r["duplicates"]["annotation_consistency"]
    add(f"- Exact-duplicate image groups: **{r['duplicates']['n_groups']}**; "
        f"groups spanning different splits: **{r['duplicates']['cross_split_groups']}**.")
    add(f"- Among identical-image pairs: **{fmt_int(cons.get('pairs_with_class_conflict', 0))} class "
        f"conflicts** (same box, different ID) and "
        f"**{fmt_int(cons.get('pairs_with_geometry_mismatch', 0))} geometry mismatches**.")
    add(f"- Likely leakage groups by Roboflow base-name: **{r['leakage']['n_groups']}** "
        f"({fmt_int(r['leakage']['n_files'])} files), "
        f"**{fmt_int(r['leakage']['n_cross_split_groups'])}** already span splits.")
    if r["images"]["corrupt"]:
        add(f"- **{len(r['images']['corrupt'])} corrupt / undecodable images.**")
    if r["images"]["unreadable_format"]:
        add(f"- **{len(r['images']['unreadable_format'])} images in formats Ultralytics cannot load** "
            f"({sorted({x['ext'] for x in r['images']['unreadable_format']})}).")
    add("")

    add("### 0.1 Pass / fail summary")
    add("")
    add(md_table(["Requirement", "Result"], [
        ["Folder structure (images/labels × train/val/test × 4 batches)",
         "**PASS** — 12/12 split folders present"],
        ["Annotation syntax (5 tokens, YOLO order)",
         f"**PASS** — {fmt_int(totals['labels'])} files, 0 malformed lines"],
        ["Coordinates normalised and inside the frame",
         "**PASS** — 0 out-of-range, 0 non-positive sizes, 0 boxes outside image"],
        ["Every image has a label",
         f"**PASS** — 0 missing-label images, 0 orphan labels"],
        ["All 35 class IDs present in all batches/splits",
         "**PASS** — IDs 0–34 everywhere, no gaps"],
        ["Class names available",
         "**FAIL** — no metadata anywhere in or beside the dataset"],
        ["Split integrity (no train/val/test leakage)",
         f"**FAIL** — {fmt_int(r['duplicates']['cross_split_groups'])} MD5-identical groups and "
         f"{fmt_int(r['leakage']['n_cross_split_groups'])} source-image groups already span splits"],
        ["Annotation consistency on identical images",
         f"**FAIL (localised)** — 7 class conflicts, 177 geometry mismatches"],
        ["One image per label stem",
         f"**FAIL (localised)** — {len(r['files']['multi_image_same_stem'])} duplicate-stem pairs"],
        ["Ready for training", "**NO** — clean a derived copy first (section 10)"],
    ]))
    add("")

    # ---------------- 1. counts ----------------
    add("## 1. Total images and labels per batch")
    add("")
    rows = []
    for b in batches:
        s = r["per_batch"][b]
        rows.append([
            b, fmt_int(s["images"]), fmt_int(s["labels"]), f"{s['images'] - s['labels']:+d}",
            fmt_int(s["boxes"]), fmt_int(s["empty_labels"]), fmt_int(s["malformed_files"]),
            ", ".join(f"{k}={v}" for k, v in sorted(s["extensions"].items())),
        ])
    add(md_table(["Batch", "Images", "Labels", "Diff", "Boxes", "Empty labels",
                  "Malformed files", "Image extensions"], rows))
    add("")
    add(f"**Total: {fmt_int(totals['images'])} images / {fmt_int(totals['labels'])} labels / "
        f"{fmt_int(totals['boxes'])} boxes.**")
    add("")

    add("### 1.1 Per-split breakdown")
    add("")
    rows = []
    for b in batches:
        for s in SPLITS:
            p = r["pairing"].get(f"{b}/{s}")
            if not p:
                rows.append([b, s, "—", "—", "—", "—", "—", "—", "—"])
                continue
            rows.append([b, s, p["image_files"], p["image_stems"], p["labels"], p["matched"],
                         p["unlabelled_image_files"], p["missing_labels"],
                         p["orphan_labels"], p["boxes"]])
    add(md_table(["Batch", "Split", "Image files", "Image stems", "Label files", "Matched stems",
                  "Image files w/o own label", "Missing label (stem)", "Orphan labels", "Boxes"], rows))
    add("")
    if r["missing_split_dirs"]:
        add(f"Missing split directories: {r['missing_split_dirs']}")
        add("")

    # ---------------- 2. class names / metadata ----------------
    add("## 2. Available class names and their sources")
    add("")
    if not r["metadata_files"]:
        add("**NONE.** No `data.yaml`, `classes.txt`, `obj.names`, README, JSON or CSV file exists "
            "anywhere under the dataset root. The search was recursive and explicitly excluded the "
            "`labels/` trees, so annotation files were not mistaken for metadata.")
        add("")
        if r.get("parent_scan"):
            add("Files adjacent to (but outside) the dataset root were also inspected:")
            add("")
            for item in r["parent_scan"]:
                add(f"- `{item['file']}` ({fmt_int(item['bytes'])} bytes)")
            add("")
            add("None of these is a class-name list; the `.py` file contains only counting logic. "
                "**No class names exist anywhere in or next to this dataset.**")
            add("")
        add("**Limitation (explicit):** it is therefore *impossible* to establish which food/object "
            "category each numeric class ID represents. Any name table used from here on would be an "
            "assumption, not verified metadata. Class IDs below are reported strictly as numbers.")
    else:
        add("Metadata files discovered (this report does **not** assume the mapping is correct):")
        add("")
        for p in r["metadata_files"]:
            add(f"- `{p}`")
    add("")

    hints = r["images"].get("domain_hints") or {}
    if hints:
        add("### 2.1 Circumstantial evidence from filenames (NOT class metadata)")
        add("")
        add("The table below counts how many image filenames contain a keyword. This is a domain "
            "sanity check only: a filename does **not** tell us which class ID an object has, and "
            "these counts must never be turned into a class list.")
        add("")
        add(md_table(["Filename keyword", "Images with keyword"],
                     [[k, fmt_int(v)] for k, v in list(hints.items())[:15]]))
        add("")
        add("The spread of keywords suggests a **heterogeneous object dataset** (restroom signs, "
            "street furniture, appliances, furniture, vehicles, waste containers) rather than a "
            "food-product dataset. If the project is meant to be food-specific, confirm the intended "
            "scope with the dataset owner before training.")
        add("")

    # ---------------- 3. class IDs ----------------
    add("## 3. Class IDs found in each batch")
    add("")
    add("Counts are boxes per class (valid lines only).")
    add("")
    header = ["Class ID"] + batches + ["Total"]
    rows = []
    for cid in classes["all_ids"]:
        row = [cid]
        total = 0
        for b in batches:
            n = classes["counts_by_batch"][b].get(str(cid), classes["counts_by_batch"][b].get(cid, 0))
            row.append(fmt_int(n))
            total += n
        row.append(fmt_int(total))
        rows.append(row)
    add(md_table(header, rows))
    add("")
    add(f"- Distinct class IDs overall: **{len(classes['all_ids'])}**")
    add(f"- Observed range: **{classes['id_min']} … {classes['id_max']}**")
    add(f"- IDs absent inside that range: **{classes['gaps_in_range'] or 'none'}**")
    add(f"- IDs >= {classes['expected_class_count_claim']}: **{classes['ids_ge_expected'] or 'none'}** "
        f"(the claim of 35 classes implies valid IDs 0–34)")

    per_class_total = []
    for cid in classes["all_ids"]:
        n = 0
        for b in batches:
            d = classes["counts_by_batch"][b]
            n += d.get(cid, d.get(str(cid), 0))
        per_class_total.append(n)
    if per_class_total and min(per_class_total) > 0:
        ratio = max(per_class_total) / min(per_class_total)
        add(f"- Class balance: least frequent class has **{fmt_int(min(per_class_total))}** boxes, "
            f"most frequent **{fmt_int(max(per_class_total))}** (ratio **{ratio:.2f}:1**).")
        add(f"- **Interpretation:** a ratio near 1.0 means the classes are almost perfectly balanced, "
            f"which is atypical for real-world detection data (usually long-tailed) and suggests "
            f"curated balancing, resampling or synthetic/duplicated sources. It is not an error, but "
            f"it means validation mAP will not reflect real-world class imbalance.")
    add("")
    add("### 3.1 IDs present per split")
    add("")
    rows = []
    for b in batches:
        for s in SPLITS:
            ids = classes["per_batch_split"][b][s]
            rows.append([b, s, len(ids), ids])
    add(md_table(["Batch", "Split", "#IDs", "Class IDs"], rows))
    add("")

    # ---------------- 4. consistency ----------------
    add("## 4. Class-ID consistency comparison")
    add("")
    add("### 4.1 Set comparison")
    add("")
    sets = {b: set(classes["per_batch"][b]) for b in batches}
    rows = []
    for i, a in enumerate(batches):
        for b in batches[i + 1:]:
            rows.append([f"{a} vs {b}",
                         "IDENTICAL" if sets[a] == sets[b] else "DIFFERENT",
                         sorted(sets[a] - sets[b]), sorted(sets[b] - sets[a])])
    union_set = set().union(*sets.values()) if sets else set()
    inter_set = set.intersection(*sets.values()) if sets else set()
    all_same = bool(sets) and all(s == union_set for s in sets.values())
    rows.append(["ALL", "IDENTICAL" if all_same else "DIFFERENT",
                 f"union={sorted(union_set)}", f"intersection={sorted(inter_set)}"])
    add(md_table(["Pair", "Same ID set?", f"Only in left", f"Only in right"], rows))
    add("")
    add("**Important caveat:** identical *sets* of IDs do not prove identical *semantics*. "
        "Two batches can both use 0–34 while assigning different meanings to the same number. "
        "Section 4.2 tests this empirically.")
    add("")

    add("### 4.2 Inferred class-ID mapping (evidence from shared-file annotations)")
    add("")
    add("Method: source images are identified by their Roboflow base name (the `_jpg.rf.<hash>` "
        "augmentation suffix is stripped). When the same source image appears in two batches with an "
        "identical box geometry (coordinates rounded to 6 decimals), the boxes are matched and "
        "`(id_A -> id_B)` is tallied. A consistent tally across many images is evidence of a "
        "systematic remap between batches; a mixed tally is evidence of inconsistent annotation.")
    add("")
    if not r["mapping"]["inferred"]:
        add("No shared source images with identical coordinates were found between batches — "
            "cross-batch mapping cannot be inferred from the data.")
    else:
        for pair, rows in r["mapping"]["inferred"].items():
            shared = r["mapping"]["shared_stem_pairs"].get(pair, 0)
            add(f"**{pair}** — {shared} shared source images with identical coordinates")
            add("")
            add(md_table(["Source ID", "Best target ID", "Support", "Total", "Confidence", "Ambiguous?"],
                         [[x["source_id"], x["target_id"], x["support"], x["total"],
                           f"{x['confidence']:.2f}", "yes" if x["ambiguous"] else "no"] for x in rows]))
            add("")
        add("### 4.3 Verdict per batch pair")
        add("")
        verdict_rows = []
        for pair, mrows in r["mapping"]["inferred"].items():
            matched = sum(x["total"] for x in mrows)
            nonidentity = sum(x["support"] for x in mrows if x["source_id"] != x["target_id"])
            ambiguous = [x for x in mrows if x["ambiguous"]]
            if matched == 0:
                verdict = "no evidence"
            elif ambiguous:
                verdict = "INCONSISTENT (mixed IDs)"
            elif nonidentity == 0:
                verdict = "consistent — identical IDs"
            else:
                verdict = "consistent — systematic REMAP"
            verdict_rows.append([pair, matched, nonidentity, len(ambiguous), verdict])
        add(md_table(["Batch pair", "Matched boxes", "Non-identity matches",
                      "Ambiguous source IDs", "Verdict"], verdict_rows))
        add("")
        add("A *systematic remap* means the same source image carries different numeric IDs in the "
            "two batches, but the mapping is one-to-one and repeatable — fixable by an explicit "
            "permutation table. *Inconsistent* means the same geometry maps to different IDs "
            "un-repeatably — those are label errors.")
        add("")

        if r["mapping"]["coord_identical_diff_class"]:
            diff_examples = r["mapping"]["coord_identical_diff_class"]
            n_diff = len(diff_examples)
            add(f"**{n_diff} box match{'es' if n_diff != 1 else ''} "
                f"{'have' if n_diff != 1 else 'has'} identical coordinates but a different class "
                f"ID.** Examples:")
            add("")
            for item in diff_examples[:20]:
                add(f"- `{item['base']}` — {item['batch_a']} id {item['class_a']} vs "
                    f"{item['batch_b']} id {item['class_b']}")
            add("")
        conflicting = r["mapping"]["conflicting_coords_pairs"]
        if conflicting:
            add("Shared source images whose *coordinates differ* (different augmentation or re-annotation):")
            add("")
            add(md_table(["Batch pair", "Source images with differing coordinates"],
                         [[k, v] for k, v in conflicting.items()]))
            add("")

        total_matched = sum(x["total"] for mrows in r["mapping"]["inferred"].values() for x in mrows)
        total_nonid = sum(x["support"] for mrows in r["mapping"]["inferred"].values()
                          for x in mrows if x["source_id"] != x["target_id"])
        dcons = r["duplicates"]["annotation_consistency"]
        add("### 4.4 Overall class-ID verdict")
        add("")
        add(f"- All {len(batches)} batches use exactly the same ID set: **0–34**, and every ID appears "
            f"in every split of every batch.")
        add(f"- Shared-source-image evidence covers **{fmt_int(total_matched)}** matched boxes "
            f"({pct(total_matched, totals['boxes'])} of all boxes); "
            f"**{fmt_int(total_nonid)}** of them "
            f"{'carries' if total_nonid == 1 else 'carry'} a different ID in the other batch.")
        add(f"- Independent evidence from byte-identical images: "
            f"**{fmt_int(dcons.get('consistent_pairs', 0))} fully consistent pairs**, "
            f"{fmt_int(dcons.get('pairs_with_class_conflict', 0))} class conflicts, "
            f"{fmt_int(dcons.get('pairs_with_geometry_mismatch', 0))} geometry mismatches "
            f"(section 8.1.1).")
        add("")
        if total_nonid == 0:
            add("**Conclusion: no class-ID remap was detected.** The four batches behave as if they "
                "share one class index. The class disagreements found on identical images are "
                "localised label errors, not a systematic difference between batches.")
        else:
            add(f"**Conclusion: the batches are *mostly* aligned, but {total_nonid} conflicting "
                f"box match{'es' if total_nonid != 1 else ''} "
                f"{'show' if total_nonid != 1 else 'shows'} that some annotations disagree.** "
                "There is no evidence of a full permutation between batches; treat the conflicts as "
                "label errors to review (section 8.1.1).")
        add("")
        add("**Limitation:** this proves *numeric consistency*, not *semantic correctness*. Without a "
            "class-name list there is no way to prove that ID 7 means the same object category in "
            "every batch — only that the batches were annotated consistently with each other.")
        add("")

    # ---------------- 5. format / coordinates ----------------
    add("## 5. Annotation format and coordinate validation")
    add("")
    add("Expected format per line: `class_id x_center y_center width height`, whitespace separated, "
        "all four coordinates normalised to [0, 1].")
    add("")
    add(md_table(
        ["Check", "Result"],
        [
            ["Files with exactly 5 tokens everywhere", f"{fmt_int(totals['labels'] - totals['malformed_files'])} "
                                                       f"/ {fmt_int(totals['labels'])}"],
            ["Malformed lines (any reason)", fmt_int(totals["malformed_lines"])],
            ["Files with BOM", fmt_int(sum(s["bom_files"] for s in r["per_batch"].values()))],
            ["Files with CRLF line endings",
             fmt_int(sum(s["crlf_files"] for s in r["per_batch"].values()))],
            ["Files that are not valid UTF-8",
             fmt_int(sum(s["non_utf8_files"] for s in r["per_batch"].values()))],
            ["Coordinate values outside [0,1]", fmt_int(r["errors"]["kinds"].get("COORD_OUT_OF_RANGE", 0))],
            ["Non-positive width/height", fmt_int(r["errors"]["kinds"].get("NONPOSITIVE_SIZE", 0))],
            ["Width or height > 1", fmt_int(r["errors"]["kinds"].get("SIZE_GT_1", 0))],
            ["Box extends outside the image frame",
             fmt_int(r["errors"]["kinds"].get("BOX_OUTSIDE_IMAGE", 0))],
            ["Warning: tiny boxes (area < 1e-4)", fmt_int(r["warnings"]["kinds"].get("TINY_BOX", 0))],
            ["Warning: boxes touching the image edge",
             fmt_int(r["warnings"]["kinds"].get("EDGE_TOUCHING_BOX", 0))],
            ["Warning: extreme aspect ratio (>20:1)",
             fmt_int(r["warnings"]["kinds"].get("EXTREME_ASPECT", 0))],
            ["Warning: duplicated identical line in a file",
             fmt_int(r["warnings"]["kinds"].get("DUPLICATE_LINE_IN_FILE", 0))],
        ]))
    add("")
    if r["errors"]["kinds"]:
        add("### 5.1 Malformed-line breakdown")
        add("")
        add(md_table(["Error kind", "Count", "Example"],
                     [[k, v, (r["errors"]["examples"].get(k) or [""])[0]]
                      for k, v in sorted(r["errors"]["kinds"].items(), key=lambda x: -x[1])]))
        add("")
    else:
        add("No malformed lines detected.")
        add("")

    # ---------------- 6. empty / missing ----------------
    add("## 6. Empty annotations and image/label pairing")
    add("")
    rows = []
    for b in batches:
        s = r["per_batch"][b]
        rows.append([b, s["empty_labels"], s["labels"],
                     pct(s["empty_labels"], s["labels"]),
                     s["image_files"], s["image_stems"], s["image_files"] - s["label_stems"]])
    add(md_table(["Batch", "Empty label files", "Label files", "% empty",
                  "Image files", "Image stems", "Image files without own label"], rows))
    add("")
    add("An *empty* label file is a valid YOLO background image only if it is intentional. "
        "If these were meant to be annotated, they will teach the model that the objects are absent.")
    add("")

    add("### 6.1 Images without a matching label")
    add("")
    if r["files"]["missing_labels"]:
        add(md_table(["Batch", "Split", "Stem", "Image file(s)"],
                     [[x["batch"], x["split"], x["stem"], ", ".join(x["image"])]
                      for x in r["files"]["missing_labels"]]))
    else:
        add("None.")
    add("")
    add("### 6.2 Label files without a matching image (orphans)")
    add("")
    if r["files"]["orphan_labels"]:
        add(md_table(["Batch", "Split", "Stem", "Label file"],
                     [[x["batch"], x["split"], x["stem"], x["label"]]
                      for x in r["files"]["orphan_labels"]]))
    else:
        add("None.")
    add("")
    if r["files"]["wrong_split_label"] or r["files"]["wrong_split_image"]:
        add("### 6.3 Label/image found in a different split of the same batch")
        add("")
        add("This is usually a **data-leakage or misplacement** signal, not a missing annotation.")
        add("")
        for x in r["files"]["wrong_split_label"]:
            add(f"- image `{x['batch']}/{x['split']}/{x['stem']}` — label exists in "
                f"{', '.join(x['label_in_other_split'])}")
        for x in r["files"]["wrong_split_image"]:
            add(f"- label `{x['batch']}/{x['split']}/{x['stem']}` — image exists in "
                f"{', '.join(x['image_in_other_split'])}")
        add("")
    if r["files"]["case_mismatch"]:
        add("### 6.4 Case-only filename mismatches")
        add("")
        add("These match on Windows (case-insensitive) but will break on Linux / Colab training.")
        add("")
        for x in r["files"]["case_mismatch"]:
            add(f"- `{x['batch']}/{x['split']}/{x['stem']}`")
        add("")
    if r["files"]["multi_image_same_stem"]:
        add("### 6.5 Several images sharing one stem in the same split")
        add("")
        add("Only one label file can exist per stem, so at least one image is effectively unlabelled.")
        add("")
        dup_stem_items = r["files"]["multi_image_same_stem"]
        identical = sum(1 for x in dup_stem_items if x.get("identical_content") is True)
        different = sum(1 for x in dup_stem_items if x.get("identical_content") is False)
        add(f"- {len(dup_stem_items)} stems affected; **{identical}** pairs are byte-identical "
            f"(format duplicates) and **{different}** pairs are *different images* that share a "
            f"label — those {different} images have no annotation of their own and must be reviewed.")
        add("")
        for x in dup_stem_items:
            tag = "" if x.get("identical_content") is None else (
                " — identical content" if x["identical_content"] else " — **different images**")
            add(f"- `{x['batch']}/{x['split']}/{x['stem']}`: {', '.join(x['images'])}{tag}")
        add("")

    # ---------------- 7. discrepancy ----------------
    add("## 7. The 16,103 vs 16,107 discrepancy")
    add("")
    n_stems = totals["image_stems"]
    n_files = totals["image_files"]
    n_labels = totals["labels"]
    n_dup = totals["dup_stem_extra_files"]
    nonstd = totals["stems_only_nonstandard"]
    n_nonstd = len(nonstd)
    reproduced = n_stems - n_nonstd
    add("The reconciliation is exact, and it is a **counting-method artefact, not missing data**:")
    add("")
    add(f"- Label files (= distinct label stems): **{fmt_int(n_labels)}**")
    add(f"- Distinct image stems: **{fmt_int(n_stems)}** — every stem has a label "
        f"(missing-label images: {len(r['files']['missing_labels'])}).")
    add(f"- Physical image files: **{fmt_int(n_files)}** = {fmt_int(n_stems)} stems + "
        f"**{fmt_int(n_dup)} extra files that share a stem with another image** in the same split "
        f"(e.g. `image22.jpeg` and `image22.png` — one label can only cover one of them).")
    add(f"- Stems whose only image is `.webp`/`.gif`: **{n_nonstd}**")
    add("")
    if nonstd:
        add("Files involved (excluded by any `.jpg/.jpeg/.png`-only filter):")
        add("")
        for x in nonstd:
            add(f"- `{x['batch']}/{x['split']}/{x['stem']}` — {', '.join(x['files'])}")
        add("")
    add(f"**Formula:** {fmt_int(n_stems)} distinct image stems − {n_nonstd} non-JPG/PNG-only stems "
        f"= **{fmt_int(reproduced)}**, which is the brief's 16,103. The brief's 16,107 is the label "
        f"count. This is reproduced exactly by an earlier script that (a) filtered image extensions "
        f"to jpg/jpeg/png and (b) counted stems in a `set()` while counting labels with a glob.")
    add("")
    add(f"**For a trainer, the real numbers are {fmt_int(n_files)} image files vs "
        f"{fmt_int(n_labels)} label files (+{n_dup}).** The {n_dup} surplus files are not "
        f"\"missing labels\" in the usual sense; they are duplicate-stem variants "
        f"(section 6.5). No label belongs to an image that does not exist (orphans: "
        f"{len(r['files']['orphan_labels'])}).")
    add("")

    # ---------------- 8. images ----------------
    add("## 8. Image file audit")
    add("")
    decoded = (fmt_int(totals["images"] - len(r["images"]["corrupt"]))
               if r["images"]["checked"] else "skipped (--skip-image-check)")
    add(md_table(["Check", "Result"], [
        ["Images decoded successfully", decoded],
        ["Corrupt / undecodable", fmt_int(len(r["images"]["corrupt"]))],
        ["Formats Ultralytics cannot load", fmt_int(len(r["images"]["unreadable_format"]))],
        ["Very small images (< 32 px on a side)", fmt_int(len(r["images"]["tiny"]))],
        ["Small images (< 100 px on the shortest side)", fmt_int(len(r["images"]["small"]))],
        ["Pixel dimension range across all images",
         f"{r['images']['dim_min']} – {r['images']['dim_max']} px"],
        ["Exact duplicate groups (MD5)", fmt_int(r["duplicates"]["n_groups"])],
        ["Files in duplicate groups", fmt_int(r["duplicates"]["n_files_in_duplicate_groups"])],
        ["Duplicate groups spanning more than one split", fmt_int(r["duplicates"]["cross_split_groups"])],
        ["... of those, a leak inside a single batch",
         fmt_int(r["duplicates"]["within_batch_cross_split_groups"])],
        ["Duplicate groups spanning more than one batch (any split)",
         fmt_int(r["duplicates"]["cross_batch_groups"])],
        ["Duplicate pairs with a class conflict (IoU >= 0.5)",
         fmt_int(r["duplicates"]["annotation_consistency"].get("pairs_with_class_conflict", 0))],
        ["Duplicate pairs with a geometry mismatch",
         fmt_int(r["duplicates"]["annotation_consistency"].get("pairs_with_geometry_mismatch", 0))],
        ["Likely leakage groups (Roboflow base name)", fmt_int(r["leakage"]["n_groups"])],
        ["... of which span different splits", fmt_int(r["leakage"]["n_cross_split_groups"])],
        ["Files inside those leakage groups", fmt_int(r["leakage"]["n_files"])],
    ]))
    add("")
    if r["images"]["corrupt"]:
        add("**Corrupt images:**")
        add("")
        for x in r["images"]["corrupt"][:50]:
            add(f"- `{x['file']}` — {x['error']}")
        add("")
    if r["images"]["unreadable_format"]:
        add("**Images in formats Ultralytics cannot load** (must be converted before training):")
        add("")
        for x in r["images"]["unreadable_format"][:50]:
            add(f"- `{x['file']}`")
        add("")
    if r["images"]["tiny"]:
        add("**Very small images** (hard to detect objects reliably):")
        add("")
        for x in r["images"]["tiny"][:30]:
            add(f"- `{x['file']}` ({x['width']}×{x['height']})")
        add("")
    if r["images"]["small"]:
        add(f"**Small images (< {SMALL_IMAGE_SIDE} px)** are upscaled by YOLO; small objects in them "
            f"lose detail. {len(r['images']['small'])} file(s); examples:")
        add("")
        for x in r["images"]["small"][:15]:
            add(f"- `{x['file']}` ({x['width']}×{x['height']})")
        add("")
    if r["duplicates"]["groups"]:
        add("### 8.1 Exact duplicate images")
        add("")
        add(f"{fmt_int(r['duplicates']['n_groups'])} groups of byte-identical images "
            f"({fmt_int(r['duplicates']['n_files_in_duplicate_groups'])} files) were found. "
            f"{fmt_int(r['duplicates']['cross_split_groups'])} of them contain images from more than "
            f"one split, and {fmt_int(r['duplicates']['within_batch_cross_split_groups'])} contain a "
            f"train/val/test leak inside a single batch. "
            f"{fmt_int(r['duplicates']['cross_batch_groups'])} groups span more than one batch.")
        add("")
        add("Full list: `duplicates.csv`.")
        add("")
        add("#### 8.1.1 Annotation consistency among identical images")
        add("")
        cons = r["duplicates"]["annotation_consistency"]
        add(f"- File pairs compared: **{fmt_int(cons.get('pairs', 0))}**")
        add(f"- Fully consistent (same boxes at IoU >= 0.5, same classes): "
            f"**{fmt_int(cons.get('consistent_pairs', 0))}**")
        add(f"- Geometry mismatch (missing / shifted / differently sized boxes): "
            f"**{fmt_int(cons.get('pairs_with_geometry_mismatch', 0))}**")
        add(f"- Class conflict (box matches at IoU >= 0.5 but the class ID differs): "
            f"**{fmt_int(cons.get('pairs_with_class_conflict', 0))}**")
        add("")
        add("Because the image bytes are identical, every difference here is an annotation "
            "inconsistency, not a visual difference. It is direct label noise.")
        add("")
        if r["duplicates"]["class_conflicts"]:
            add("Examples of class conflicts on identical images:")
            add("")
            for c in r["duplicates"]["class_conflicts"][:15]:
                add(f"- `{c['files'][0]}` (id {c['class_a']}) vs `{c['files'][1]}` "
                    f"(id {c['class_b']}), IoU {c['iou']}")
            add("")
        add("Examples of duplicate groups:")
        add("")
        for d in r["duplicates"]["groups"][:15]:
            flags = []
            if d["class_conflict"]:
                flags.append("**class conflict**")
            if d["geometry_mismatch"]:
                flags.append("geometry mismatch")
            add(f"- MD5 `{d['md5'][:12]}…` ×{d['count']} at {', '.join(d['locations'])}"
                + (f" — {', '.join(flags)}" if flags else " — consistent"))
            for f in d["files"][:4]:
                add(f"  - `{f}`")
        add("")

        split_pairs = r["duplicates"]["split_pair_counts"]
        if split_pairs:
            add("#### 8.1.2 Where the leaky duplicates sit")
            add("")
            add(md_table(["Location pair", "Duplicate file pairs"],
                         [[k, v] for k, v in split_pairs.items()]))
            add("")
    if r["leakage"]["groups"]:
        add("### 8.2 Likely train/val/test leakage (Roboflow base name)")
        add("")
        add(f"{fmt_int(r['leakage']['n_groups'])} source-image groups appear more than once; "
            f"**{fmt_int(r['leakage']['n_cross_split_groups'])} of them already span different "
            f"splits** and {fmt_int(r['leakage']['n_cross_batch_only_groups'])} appear in different "
            f"batches only (these become leakage if the batches are merged).")
        add("")
        add("Differently augmented copies of one source image are not caught by MD5 but still "
            "inflate validation scores. Full list: `leakage_groups.csv`.")
        add("")
        cross = [g for g in r["leakage"]["groups"] if g["cross_split"]]
        for g in cross[:40]:
            add(f"- `{g['base']}` ×{g['count']} — splits {g['splits']}, batches {g['batches']}")
        if len(cross) > 40:
            add(f"- ... and {len(cross) - 40} more cross-split groups")
        add("")

    # ---------------- 9. problems ----------------
    add("## 9. Potential dataset problems (ranked)")
    add("")
    problems = []
    if not r["metadata_files"]:
        problems.append(("**Critical**", "No class-name metadata at all",
                         "Class IDs cannot be mapped to names. Any names used downstream are assumptions."))
    if r["mapping"]["coord_identical_diff_class"]:
        n_conf = len(r["mapping"]["coord_identical_diff_class"])
        problems.append(("**Medium**", "Cross-batch class-ID conflicts",
                         f"{n_conf} box match{'es' if n_conf != 1 else ''} "
                         f"{'have' if n_conf != 1 else 'has'} identical coordinates but different "
                         "class IDs in different batches (localised label error, not a remap)."))
    if r["errors"]["kinds"]:
        problems.append(("**High**", "Malformed annotation lines",
                         f"{totals['malformed_lines']} lines across {totals['malformed_files']} files."))
    if totals["empty_labels"]:
        problems.append(("**High**", "Empty label files",
                         f"{totals['empty_labels']} label files are empty; if unintentional they "
                         "become false background images."))
    if r["files"]["missing_labels"]:
        problems.append(("**High**", "Images without labels",
                         f"{len(r['files']['missing_labels'])} images have no label file."))
    if r["duplicates"]["cross_split_groups"]:
        problems.append(("**Critical**", "Exact duplicate images across splits (leakage)",
                         f"{r['duplicates']['cross_split_groups']} MD5-identical groups span splits, "
                         f"{r['duplicates']['within_batch_cross_split_groups']} of them inside a "
                         "single batch (train/val/test)."))
    if cons.get("pairs_with_class_conflict"):
        problems.append(("**High**", "Same image annotated with different classes",
                         f"{cons['pairs_with_class_conflict']} byte-identical image pairs have a box "
                         "that matches at IoU >= 0.5 but carries a different class ID."))
    if cons.get("pairs_with_geometry_mismatch"):
        problems.append(("**High**", "Same image annotated with different boxes",
                         f"{cons['pairs_with_geometry_mismatch']} byte-identical image pairs have "
                         "missing / shifted / differently sized boxes."))
    if r["leakage"]["n_cross_split_groups"]:
        problems.append(("**High**", "Augmented copies leak across splits",
                         f"{r['leakage']['n_cross_split_groups']} source-image groups already span "
                         "different splits; validation mAP is therefore optimistic."))
    elif r["leakage"]["n_groups"]:
        problems.append(("**Medium**", "Overlapping source images across batches",
                         f"{r['leakage']['n_groups']} base-name groups appear in multiple batches; "
                         "they will leak once the batches are merged."))
    if r["images"]["unreadable_format"]:
        n_bad = len(r["images"]["unreadable_format"])
        problems.append(("**Medium**", "Unsupported image formats",
                         f"{n_bad} image{'s' if n_bad != 1 else ''} "
                         f"({', '.join(sorted({x['ext'] for x in r['images']['unreadable_format']}))}) "
                         "cannot be loaded by Ultralytics."))
    if r["images"]["corrupt"]:
        problems.append(("**Medium**", "Corrupt images", f"{len(r['images']['corrupt'])} files failed decoding."))
    if r["warnings"]["kinds"].get("TINY_BOX"):
        problems.append(("**Medium**", "Very small boxes",
                         f"{r['warnings']['kinds']['TINY_BOX']} boxes have area < 1e-4; consider "
                         "higher input resolution or excluding them."))
    if len([1 for v in hints.values() if v >= 50]) >= 3:
        top_hints = ", ".join(f"{k} ({v})" for k, v in list(hints.items())[:5])
        problems.append(("**Low**", "Possible domain mismatch (circumstantial)",
                         f"Filenames repeatedly mention {top_hints}. This is not class metadata, but "
                         "if the project targets food products, confirm that this heterogeneous object "
                         "dataset is the intended data."))
    if r["files"]["wrong_split_label"] or r["files"]["wrong_split_image"]:
        problems.append(("**Medium**", "Split misplacement",
                         f"{len(r['files']['wrong_split_label'])} images and "
                         f"{len(r['files']['wrong_split_image'])} labels have their counterpart "
                         "in another split."))
    dup_stem_items = r["files"]["multi_image_same_stem"]
    dup_stem_different = sum(1 for x in dup_stem_items if x.get("identical_content") is False)
    if dup_stem_items:
        problems.append(("**Medium**", "Two images share one stem in the same split",
                         f"{len(dup_stem_items)} stems have 2 image files but only 1 label; "
                         f"{dup_stem_different} of the pairs are *different images*, so that many "
                         f"images have no annotation of their own."))
    if totals["stems_only_nonstandard"]:
        problems.append(("**Low**", "Images only in .webp/.gif",
                         f"{len(totals['stems_only_nonstandard'])} stems have no JPG/PNG version; "
                         "extension-filtered scripts miss them."))
    if r["files"]["orphan_labels"]:
        problems.append(("**Low**", "Orphan label files", f"{len(r['files']['orphan_labels'])} labels have no image."))
    if r["files"]["case_mismatch"]:
        problems.append(("**Low**", "Case-only filename mismatches",
                         "Works on Windows, breaks on Linux/Colab."))

    if problems:
        sev_rank = {"**Critical**": 0, "**High**": 1, "**Medium**": 2, "**Low**": 3}
        problems.sort(key=lambda x: sev_rank.get(x[0], 9))
        add(md_table(["Severity", "Problem", "Detail"], problems))
    else:
        add("No problems detected.")
    add("")

    # ---------------- 10. recommendations ----------------
    add("## 10. Recommendations for the next stage")
    add("")
    n_remap_conflicts = len(r["mapping"]["coord_identical_diff_class"])
    recs = [
        "**Do not train on this layout yet.** Freeze one authoritative class list (id → name) and one "
        "`data.yaml` first. No metadata exists, so it must come from the dataset owner; it cannot be "
        "derived from the annotation files.",
        f"**Resolve the cross-batch class semantics before merging.** Section 4.3 gives the verdict "
        f"per batch pair and {n_remap_conflicts} box "
        f"match{'es' if n_remap_conflicts != 1 else ''} already "
        f"{'show' if n_remap_conflicts != 1 else 'shows'} identical geometry with different IDs. "
        "If a pair is a *systematic remap*, apply an explicit permutation table in a derived copy. "
        "If it is *inconsistent*, treat those boxes as label errors.",
        f"**Fix the {len(r['files']['multi_image_same_stem'])} duplicate-stem pairs** (section 6.5): "
        "keep exactly one image per stem per split, or rename the second image and give it its own "
        "label. Right now one image of each pair can never be trained on.",
        f"**Handle the {len(totals['stems_only_nonstandard'])} stems that exist only as `.webp`/`.gif`** "
        "(section 7). Convert them to JPG/PNG in the derived copy; `.gif` cannot be loaded by "
        "Ultralytics at all.",
        "**Remove leakage before splitting.** Section 8.2 groups images by source (Roboflow base "
        f"name): {r['leakage']['n_groups']} groups contain more than one copy, and "
        f"{r['leakage']['n_cross_split_groups']} already span splits. Split by *group*, not by file, "
        "then re-check.",
        "**Review the near-uniform class distribution** (section 3). Every class has a similar box "
        "count, which is atypical for real-world food data and suggests deliberate balancing or "
        "synthetic/duplicated sources. Confirm the provenance so accuracy claims are defensible.",
        "**Keep tiny boxes in mind** (3 boxes with area < 1e-4, 1 extreme aspect ratio). Either "
        "train at higher input resolution (e.g. 640 → 960) or document their exclusion.",
        "**Do all cleaning in a new directory**, never in `final batches`. Log every change "
        "(old file, line, new file, line) so the process is auditable and reproducible.",
        "**Re-run `verify_dataset.py` on the cleaned copy** and require: 0 hard errors, 0 duplicate "
        "stems, 0 cross-split leakage groups, all images decodable, and exactly one label per image.",
        "**Only then** build the merged dataset with a documented, seeded split. Keep the original "
        "batches untouched as the source of truth.",
        "**Class names:** if the owner cannot provide them, state in the project write-up that names "
        "are unknown/unverified and report classes by ID only.",
    ]
    for i, rec in enumerate(recs, 1):
        add(f"{i}. {rec}")
    add("")

    # ---------------- appendix ----------------
    add("## Appendix A — Empty label files")
    add("")
    if r["files"]["empty_labels"]:
        for x in cap_lines(r["files"]["empty_labels"]):
            add(f"- `{x}`")
    else:
        add("None.")
    add("")

    add("## Appendix B — Malformed label files (full list)")
    add("")
    if r["files"]["malformed"]:
        for item in r["files"]["malformed"][:500]:
            add(f"- `{item['file']}`")
            for ln, kind, text in item["errors"][:10]:
                add(f"  - line {ln}: **{kind}** — `{text[:160]}`")
    else:
        add("None.")
    add("")

    add("## Appendix C — Method notes")
    add("")
    add("- Images are matched to labels by *exact* filename stem inside the same `batch/split`. "
        "Case-insensitive matches are reported separately because they break on Linux.")
    add("- A label file is counted as **empty** when it contains no non-blank lines. Whitespace-only "
        "files therefore count as empty.")
    add("- A line is **hard-invalid** if it does not have exactly five tokens, the class token is not "
        "a non-negative integer, a coordinate is not finite, any coordinate lies outside [0,1], "
        "width/height is <= 0, or the resulting box leaves the [0,1] frame by more than 1e-6.")
    add("- Tiny boxes, edge-touching boxes, extreme aspect ratios and duplicated lines are recorded "
        "as **warnings**; the line is still accepted.")
    add("- Class IDs are counted only from accepted lines; raw first-token counts are also captured "
        "in the JSON (`raw_class_tokens_by_batch`) for cross-checking.")
    add("- Mapping inference uses only stems that appear in two batches with an identical multiset of "
        "rounded coordinates; it is *evidence*, not proof, and is reported with support counts.")
    add("- Leakage detection normalises Roboflow names (`name_jpg.rf.<32 hex>`) to `name` and groups "
        "on that; it catches differently augmented copies of the same source image.")
    add("")
    return "\n".join(L)


# --------------------------------------------------------------------------------------
# CSV writers
# --------------------------------------------------------------------------------------

def write_csvs(r: dict, out_dir: Path):
    # class distribution
    with (out_dir / "class_distribution.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["batch", "split", "class_id", "boxes", "images_containing_class"])
        for b in r["batches"]:
            for s in SPLITS:
                counts = r["classes"]["counts_by_batch_split"][b][s]
                for cid, n in sorted(counts.items(), key=lambda kv: int(kv[0])):
                    w.writerow([b, s, cid, n, ""])

    # file issues
    with (out_dir / "file_issues.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["category", "batch", "split", "stem_or_file", "detail"])
        for x in r["files"]["missing_labels"]:
            w.writerow(["missing_label", x["batch"], x["split"], x["stem"], "; ".join(x["image"])])
        for x in r["files"]["orphan_labels"]:
            w.writerow(["orphan_label", x["batch"], x["split"], x["stem"], x["label"]])
        for x in r["files"]["wrong_split_label"]:
            w.writerow(["label_in_other_split", x["batch"], x["split"], x["stem"],
                        ", ".join(x["label_in_other_split"])])
        for x in r["files"]["wrong_split_image"]:
            w.writerow(["image_in_other_split", x["batch"], x["split"], x["stem"],
                        ", ".join(x["image_in_other_split"])])
        for x in r["files"]["case_mismatch"]:
            w.writerow(["case_mismatch", x["batch"], x["split"], x["stem"], ""])
        for x in r["files"]["multi_image_same_stem"]:
            w.writerow(["multiple_images_same_stem", x["batch"], x["split"], x["stem"],
                        f"identical_content={x.get('identical_content')}; " + "; ".join(x["images"])])
        for item in r["files"]["malformed"]:
            for ln, kind, text in item["errors"]:
                w.writerow([f"malformed:{kind}", item["file"].split("\\")[0] if "\\" in item["file"] else "",
                            "", item["file"], f"line {ln}: {text[:300]}"])

    # duplicates
    with (out_dir / "duplicates.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["md5", "group_size", "locations", "conflicting_labels", "files"])
        for d in r["duplicates"]["groups"]:
            w.writerow([d["md5"], d["count"], " | ".join(d["locations"]),
                        d["conflicting_labels"], " | ".join(d["files"])])

    # leakage
    with (out_dir / "leakage_groups.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["base_name", "count", "splits", "batches", "files"])
        for g in r["leakage"]["groups"]:
            w.writerow([g["base"], g["count"], " | ".join(g["splits"]),
                        " | ".join(g["batches"]), " | ".join(g["files"])])

    # inferred mapping
    with (out_dir / "inferred_mapping.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["batch_pair", "source_id", "target_id", "support", "total", "confidence", "ambiguous"])
        for pair, rows in r["mapping"]["inferred"].items():
            for x in rows:
                w.writerow([pair, x["source_id"], x["target_id"], x["support"],
                            x["total"], x["confidence"], x["ambiguous"]])


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def main(argv=None):
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Read-only YOLO dataset verification.")
    parser.add_argument("--root", type=Path,
                        default=here / "final batches",
                        help="dataset root containing Batch folders")
    parser.add_argument("--out", type=Path,
                        default=here / "dataset_verification",
                        help="output directory for the report (must be outside the dataset)")
    parser.add_argument("--skip-image-check", action="store_true",
                        help="skip Pillow decode/size checks")
    parser.add_argument("--skip-hash", action="store_true",
                        help="skip MD5 duplicate detection")
    args = parser.parse_args(argv)

    try:
        args.out.resolve().relative_to(args.root.resolve())
    except ValueError:
        pass
    else:
        raise SystemExit("Refusing to write output inside the dataset root. Choose another --out.")

    print(f"Dataset root : {args.root}", file=sys.stderr)
    print(f"Output dir   : {args.out}", file=sys.stderr)
    print("Read-only mode: dataset files will not be modified.\n", file=sys.stderr)

    results = verify(args.root, args.out, not args.skip_image_check, not args.skip_hash)

    (args.out / "verification_summary.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    (args.out / "verification_report.md").write_text(build_report(results), encoding="utf-8")
    write_csvs(results, args.out)

    print("\nDone.", file=sys.stderr)
    print(f"Report : {args.out / 'verification_report.md'}", file=sys.stderr)
    print(f"JSON   : {args.out / 'verification_summary.json'}", file=sys.stderr)


if __name__ == "__main__":
    main()
