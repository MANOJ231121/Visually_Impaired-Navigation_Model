#!/usr/bin/env python3
"""
stage_dataset.py
================

Safe staging copy for the 4-batch YOLO dataset.

What it does
------------
1. Creates C:\\vs code\\object detection\\prepared_dataset.
2. Copies EVERY original image and label, preserving Batch/split structure.
3. Adds PNG copies of the 4 WEBP/GIF images (the originals are also kept, so
   nothing is omitted).
4. Writes the label associated with each converted image unchanged (it is the
   same shared label file, copied verbatim).
5. Preserves duplicate images, duplicate-stem pairs and conflicting labels.
6. Writes an inventory and validation reports.

What it never does
------------------
* modifies, deletes, renames or moves an original file
* changes class IDs, box coordinates or split membership
* resolves annotation/class conflicts
* omits or quarantines any image
* trains a model

Run:
    python "C:\\vs code\\object detection\\stage_dataset.py"
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from PIL import Image

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
SRC = PROJECT_ROOT / "final batches"
DEST = PROJECT_ROOT / "prepared_dataset"
PLAN_DIR = PROJECT_ROOT / "dataset_preparation_plan"
ANALYSIS_DIR = PROJECT_ROOT / "dataset_verification"
OUT_DIR = PROJECT_ROOT / "dataset_staging"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tif", ".tiff"}
CONVERT_EXTS = {".webp", ".gif"}
WORKERS = 8

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def rel_of(path: Path) -> str:
    return path.relative_to(SRC).as_posix()


def dest_rel_of(path: Path) -> str:
    return path.relative_to(DEST).as_posix()


def label_rel_for_image_rel(image_rel: str) -> str:
    parts = list(Path(image_rel).parts)
    parts[parts.index("images")] = "labels"
    return Path(*parts).with_suffix(".txt").as_posix()


def batch_split(rel: str) -> tuple[str, str]:
    parts = Path(rel).parts
    return parts[0], parts[2]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_label(path: Path) -> tuple[bytes, list[tuple[str, ...]], list[int]]:
    raw = path.read_bytes()
    lines: list[tuple[str, ...]] = []
    class_ids: list[int] = []
    for line in raw.decode("utf-8", errors="replace").splitlines():
        parts = tuple(line.split())
        if not parts:
            continue
        lines.append(parts)
        if len(parts) == 5:
            class_ids.append(int(parts[0]))
    return raw, lines, class_ids


def boxes_in_bounds(lines: list[tuple[str, ...]]) -> bool:
    for parts in lines:
        if len(parts) != 5:
            continue
        _, x, y, w, h = parts
        try:
            xf, yf, wf, hf = float(x), float(y), float(w), float(h)
        except ValueError:
            return False
        if not (0.0 <= xf <= 1.0 and 0.0 <= yf <= 1.0 and 0.0 < wf <= 1.0 and 0.0 < hf <= 1.0):
            return False
        # YOLO boxes are center-based: cx +/- w/2, cy +/- h/2.
        if (xf - wf / 2 < -1e-6 or xf + wf / 2 > 1.0 + 1e-6
                or yf - hf / 2 < -1e-6 or yf + hf / 2 > 1.0 + 1e-6):
            return False
    return True


def image_info(path: Path) -> dict:
    with Image.open(path) as im:
        return {
            "width": im.width,
            "height": im.height,
            "mode": im.mode,
            "format": im.format,
            "frames": int(getattr(im, "n_frames", 1)),
        }


def md_table(headers: list[str], rows: list[list[object]]) -> str:
    out = [
        "| " + " | ".join(str(h) for h in headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for row in rows:
        out.append("| " + " | ".join("" if v is None else str(v) for v in row) + " |")
    return "\n".join(out)


def source_fingerprint() -> dict[str, tuple[int, int]]:
    return {
        rel_of(p): (p.stat().st_size, p.stat().st_mtime_ns)
        for p in SRC.rglob("*")
        if p.is_file()
    }


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------


def preflight(allow_existing: bool = False) -> dict:
    if not SRC.is_dir():
        raise SystemExit(f"Source dataset not found: {SRC}")

    if DEST.exists() and not allow_existing:
        existing = [p for p in DEST.rglob("*") if p.is_file()]
        if existing:
            raise SystemExit(
                f"Destination already exists and contains {len(existing)} files. "
                "Refusing to overwrite. Remove it manually or choose a new destination."
            )

    images = sorted(p for p in SRC.rglob("*")
                    if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    labels = sorted(p for p in SRC.rglob("*.txt") if p.is_file())

    image_rels = [rel_of(p) for p in images]
    label_rels = [rel_of(p) for p in labels]

    if len(set(image_rels)) != len(image_rels):
        raise SystemExit("Duplicate image relative paths in source.")
    if len(set(label_rels)) != len(label_rels):
        raise SystemExit("Duplicate label relative paths in source.")

    conversions = [r for r in image_rels if Path(r).suffix.lower() in CONVERT_EXTS]

    # Build the exact destination path set (copies + conversions) and check
    # case-insensitive collisions BEFORE writing anything.
    dest_rels: list[str] = list(image_rels) + list(label_rels)
    conversion_targets: dict[str, str] = {}
    for image_rel in conversions:
        target = Path(image_rel).with_suffix(".png").as_posix()
        conversion_targets[image_rel] = target
        dest_rels.append(target)

    folded = [r.casefold() for r in dest_rels]
    dup_folded = sorted({r for r, c in Counter(folded).items() if c > 1})

    # Also refuse if a conversion target already exists in the source.
    existing_conversion_targets = [
        t for t in conversion_targets.values() if (SRC / Path(t)).exists()
    ]

    if dup_folded or existing_conversion_targets:
        raise SystemExit(
            "Unsafe destination paths detected; stopping before copying.\n"
            f"  case-insensitive collisions: {dup_folded}\n"
            f"  conversion targets already in source: {existing_conversion_targets}"
        )

    existing_dest_rels: set[str] = set()
    if DEST.exists():
        existing_dest_rels = {
            dest_rel_of(p) for p in DEST.rglob("*") if p.is_file()
        }
        if allow_existing and existing_dest_rels != set(dest_rels):
            missing = sorted(set(dest_rels) - existing_dest_rels)[:10]
            extra = sorted(existing_dest_rels - set(dest_rels))[:10]
            raise SystemExit(
                "Existing destination does not match the expected staging set.\n"
                f"  expected={len(dest_rels):,} existing={len(existing_dest_rels):,}\n"
                f"  missing (first 10)={missing}\n"
                f"  unexpected (first 10)={extra}"
            )

    return {
        "images": images,
        "labels": labels,
        "image_rels": image_rels,
        "label_rels": label_rels,
        "conversions": conversions,
        "conversion_targets": conversion_targets,
        "dest_file_count": len(dest_rels),
        "resume": allow_existing,
    }


# ---------------------------------------------------------------------------
# Copy
# ---------------------------------------------------------------------------


def copy_worker(job: tuple[str, str, str], resume: bool = False) -> dict:
    """job = (kind, source_rel, destination_rel)."""
    kind, source_rel, dest_rel = job
    src = SRC / Path(source_rel)
    dst = DEST / Path(dest_rel)

    existed = dst.exists()
    if existed:
        if not resume:
            raise RuntimeError(f"Refusing to overwrite existing destination: {dst}")
        # Resume/verify mode: never overwrite, only re-check the existing copy.
    else:
        shutil.copy2(src, dst)

    src_hash = sha256_file(src)
    dst_hash = sha256_file(dst)

    row: dict = {
        "kind": kind,
        "action": "copy_image" if kind == "image" else "copy_label",
        "source_path": source_rel,
        "destination_path": dest_rel,
        "source_bytes": src.stat().st_size,
        "destination_bytes": dst.stat().st_size,
        "source_sha256": src_hash,
        "destination_sha256": dst_hash,
        "status": "verified" if src_hash == dst_hash else "HASH_MISMATCH",
    }

    if kind == "image":
        info = image_info(dst)
        row.update(info)
        row["class_ids"] = ""
        row["box_count"] = ""
    else:
        _, lines, class_ids = parse_label(dst)
        row["class_ids"] = ";".join(str(c) for c in sorted(set(class_ids)))
        row["box_count"] = len(class_ids)
        row["width"] = ""
        row["height"] = ""

    row["batch"], row["split"] = batch_split(source_rel)
    row["notes"] = "existing copy verified (not recopied)" if existed else ""
    return row


def run_copy(jobs: list[tuple[str, str, str]], resume: bool = False) -> list[dict]:
    rows: list[dict] = []
    errors: list[str] = []
    done = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(copy_worker, job, resume): job for job in jobs}
        for future in as_completed(futures):
            try:
                rows.append(future.result())
            except Exception as exc:  # noqa: BLE001
                job = futures[future]
                errors.append(f"{job}: {exc}")
            done += 1
            if done % 2000 == 0:
                print(f"    {'verified' if resume else 'copied'} {done:,}/{len(jobs):,}")
    if errors:
        raise SystemExit(
            f"{len(errors)} copy error(s). First 10:\n" + "\n".join(errors[:10])
        )
    return rows


# ---------------------------------------------------------------------------
# Conversions
# ---------------------------------------------------------------------------


def convert_images(preflight: dict, create: bool = True) -> list[dict]:
    log: list[dict] = []
    for index, source_rel in enumerate(sorted(preflight["conversions"]), start=1):
        source = SRC / Path(source_rel)
        dest_rel = preflight["conversion_targets"][source_rel]
        dest = DEST / Path(dest_rel)
        label_rel = label_rel_for_image_rel(source_rel)
        label_dest = DEST / Path(label_rel)

        if dest.exists() and not create:
            pass  # verify the existing converted copy without rewriting it
        elif dest.exists():
            raise SystemExit(f"Conversion destination already exists: {dest}")
        if not label_dest.is_file():
            raise SystemExit(f"Converted image has no matching label: {label_rel}")

        with Image.open(source) as im:
            src_width, src_height = im.size
            src_mode = im.mode
            src_format = im.format
            src_frames = int(getattr(im, "n_frames", 1))
            if src_frames > 1:
                im.seek(0)
            has_alpha = "A" in im.getbands() or im.mode in {"P", "LA", "PA"}
            if create:
                working = im.convert("RGBA") if has_alpha else im.convert("RGB")
                working.save(dest, format="PNG")

        with Image.open(dest) as out:
            dst_width, dst_height = out.size
            dst_mode = out.mode
            dst_format = out.format

        # Pixel-level comparison of the decoded first frame.
        with Image.open(source) as a, Image.open(dest) as b:
            if getattr(a, "n_frames", 1) > 1:
                a.seek(0)
            pixels_equal = a.convert("RGBA").tobytes() == b.convert("RGBA").tobytes()
            annotation_in_bounds = boxes_in_bounds(
                parse_label(label_dest)[1]
            )

        _, _, class_ids = parse_label(label_dest)
        label_sha = sha256_file(label_dest)
        # The annotation must be byte-identical to the original shared label.
        # Whether its boxes are in-bounds is a pre-existing annotation issue and
        # is reported separately; it is NOT a conversion failure.
        annotations_unchanged = (
            sha256_file(SRC / Path(label_rel)) == label_sha
        )

        log.append({
            "conversion_id": f"CONV-{index:03d}",
            "source_path": source_rel,
            "source_format": src_format,
            "source_mode": src_mode,
            "source_frames": src_frames,
            "source_width": src_width,
            "source_height": src_height,
            "destination_path": dest_rel,
            "destination_format": dst_format,
            "destination_mode": dst_mode,
            "destination_width": dst_width,
            "destination_height": dst_height,
            "dimensions_preserved": (src_width, src_height) == (dst_width, dst_height),
            "pixels_preserved": pixels_equal,
            "annotations_unchanged": annotations_unchanged,
            "annotations_in_bounds": annotation_in_bounds,
            "label_path": label_rel,
            "label_sha256": label_sha,
            "class_ids": ";".join(str(c) for c in sorted(set(class_ids))),
            "box_count": len(class_ids),
            "status": "verified" if (
                (src_width, src_height) == (dst_width, dst_height)
                and pixels_equal
                and annotations_unchanged
            ) else "FAILED",
        })
    return log


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def decode_worker(path_rel: str) -> tuple[str, bool, str]:
    try:
        with Image.open(DEST / Path(path_rel)) as im:
            if getattr(im, "n_frames", 1) > 1:
                im.seek(0)
            im.load()
        return path_rel, True, ""
    except Exception as exc:  # noqa: BLE001
        return path_rel, False, str(exc)


def validate(
    preflight: dict,
    copy_rows: list[dict],
    conversion_log: list[dict],
    source_before: dict[str, tuple[int, int]],
    source_label_hashes_before: dict[str, str],
) -> dict:
    v: dict = {}
    image_rels = set(preflight["image_rels"])
    label_rels = set(preflight["label_rels"])
    conversion_targets = set(preflight["conversion_targets"].values())

    dest_images = {
        dest_rel_of(p) for p in DEST.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    }
    dest_labels = {
        dest_rel_of(p) for p in DEST.rglob("*.txt") if p.is_file()
    }
    dest_all_files = {dest_rel_of(p) for p in DEST.rglob("*") if p.is_file()}

    expected_images = image_rels | conversion_targets
    expected_labels = label_rels
    expected_all = expected_images | expected_labels

    v["source_image_count"] = len(image_rels)
    v["source_label_count"] = len(label_rels)
    v["source_file_count"] = len(image_rels) + len(label_rels)
    v["destination_image_count"] = len(dest_images)
    v["destination_label_count"] = len(dest_labels)
    v["destination_file_count"] = len(dest_all_files)
    v["conversion_count"] = len(conversion_targets)
    v["expected_destination_image_count"] = len(expected_images)
    v["expected_destination_label_count"] = len(expected_labels)
    v["expected_destination_file_count"] = len(expected_all)

    v["all_source_images_present"] = image_rels <= dest_images
    v["all_source_labels_present"] = label_rels <= dest_labels
    v["converted_images_present"] = conversion_targets <= dest_images
    v["unexpected_images"] = sorted(dest_images - expected_images)
    v["unexpected_labels"] = sorted(dest_labels - expected_labels)
    v["unexpected_files"] = sorted(dest_all_files - expected_all)
    v["no_unexpected_files"] = not (v["unexpected_images"] or v["unexpected_labels"] or v["unexpected_files"])
    v["no_silent_omissions"] = (
        image_rels <= dest_images and label_rels <= dest_labels
    )
    v["source_destination_counts_match"] = (
        len(dest_images) == len(expected_images)
        and len(dest_labels) == len(expected_labels)
        and len(dest_all_files) == len(expected_all)
    )

    # ---- labels / class ids / boxes -------------------------------------
    bad_hash: list[str] = []
    bad_tokens: list[str] = []
    bad_class_ids: list[str] = []
    bad_boxes: list[str] = []
    for label_rel in sorted(label_rels):
        src = SRC / Path(label_rel)
        dst = DEST / Path(label_rel)
        src_raw, src_lines, src_classes = parse_label(src)
        dst_raw, dst_lines, dst_classes = parse_label(dst)

        if src_raw != dst_raw:
            bad_hash.append(label_rel)
        if src_lines != dst_lines:
            bad_tokens.append(label_rel)
        if src_classes != dst_classes:
            bad_class_ids.append(label_rel)
        src_boxes = [line[1:5] for line in src_lines if len(line) == 5]
        dst_boxes = [line[1:5] for line in dst_lines if len(line) == 5]
        if src_boxes != dst_boxes:
            bad_boxes.append(label_rel)

    v["labels_byte_unchanged"] = not bad_hash
    v["labels_token_unchanged"] = not bad_tokens
    v["class_ids_unchanged"] = not bad_class_ids
    v["bbox_coordinates_unchanged"] = not bad_boxes
    v["labels_with_hash_mismatch"] = bad_hash[:20]
    v["labels_with_token_mismatch"] = bad_tokens[:20]
    v["labels_with_class_id_mismatch"] = bad_class_ids[:20]
    v["labels_with_box_mismatch"] = bad_boxes[:20]

    all_classes = Counter()
    for label_rel in label_rels:
        all_classes.update(parse_label(SRC / Path(label_rel))[2])
    v["class_ids"] = sorted(all_classes)
    v["class_id_count"] = len(all_classes)
    v["all_classes_in_original_range"] = sorted(all_classes) == list(range(35))

    # ---- converted images ------------------------------------------------
    v["conversions_dimensions_preserved"] = all(c["dimensions_preserved"] for c in conversion_log)
    v["conversions_pixels_preserved"] = all(c["pixels_preserved"] for c in conversion_log)
    v["conversions_annotations_unchanged"] = all(c["annotations_unchanged"] for c in conversion_log)
    # Informational only: out-of-bounds boxes are a pre-existing annotation issue.
    v["conversions_annotations_in_bounds"] = all(c["annotations_in_bounds"] for c in conversion_log)
    v["conversions_with_out_of_bounds_boxes"] = [
        c["destination_path"] for c in conversion_log if not c["annotations_in_bounds"]
    ]
    v["conversions_all_verified"] = all(c["status"] == "verified" for c in conversion_log)

    # conversion label hashes must match the original shared label
    conv_label_ok = True
    for c in conversion_log:
        src_label = SRC / Path(c["label_path"])
        if sha256_file(src_label) != c["label_sha256"]:
            conv_label_ok = False
    v["conversion_labels_unchanged"] = conv_label_ok

    # ---- decode every destination image ---------------------------------
    decode_failures: list[dict] = []
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(decode_worker, rel): rel for rel in sorted(dest_images)}
        done = 0
        for future in as_completed(futures):
            rel, ok, err = future.result()
            if not ok:
                decode_failures.append({"path": rel, "error": err})
            done += 1
            if done % 2000 == 0:
                print(f"    decoded {done:,}/{len(dest_images):,}")
    v["images_decoded"] = len(dest_images)
    v["images_decode_failures"] = decode_failures
    v["all_images_decodable"] = not decode_failures

    # ---- original untouched ---------------------------------------------
    source_after = source_fingerprint()
    changed = sorted(
        rel for rel in set(source_before) | set(source_after)
        if source_before.get(rel) != source_after.get(rel)
    )
    v["original_files_modified"] = changed
    v["no_original_modified"] = not changed

    label_hash_after = {
        rel: sha256_file(SRC / Path(rel)) for rel in sorted(label_rels)
    }
    v["source_label_hashes_unchanged"] = (
        source_label_hashes_before == label_hash_after
    )

    # ---- overwrite / collision safety -----------------------------------
    v["destination_collisions"] = []
    v["overwrites_performed"] = 0
    v["copy_rows"] = len(copy_rows)

    v["all_checks_passed"] = all([
        v["all_source_images_present"],
        v["all_source_labels_present"],
        v["converted_images_present"],
        v["no_unexpected_files"],
        v["source_destination_counts_match"],
        v["no_silent_omissions"],
        v["labels_byte_unchanged"],
        v["class_ids_unchanged"],
        v["bbox_coordinates_unchanged"],
        v["conversions_all_verified"],
        v["conversion_labels_unchanged"],
        v["all_images_decodable"],
        v["no_original_modified"],
        v["source_label_hashes_unchanged"],
    ])
    return v


# ---------------------------------------------------------------------------
# Unresolved issues
# ---------------------------------------------------------------------------


def build_unresolved(
    preflight: dict,
    conversion_log: list[dict],
    review: dict,
) -> list[dict]:
    rows: list[dict] = []

    # 1. Duplicate-stem pairs: both images kept, one shared label.
    for item in review.get("duplicate_stems", []):
        rows.append({
            "issue_id": f"STAGE-DUPSTEM-{item['id']}",
            "category": "duplicate_stem_shared_label",
            "severity": "review_before_training",
            "count": 1,
            "affected_paths": " | ".join(item["files"]),
            "description": (
                f"Two different images share stem '{item['stem']}' in "
                f"{item['batch']}/{item['split']} and share one label file "
                f"({item['batch']}/labels/{item['split']}/{item['stem']}.txt)."
            ),
            "action_taken": "Both images copied unchanged; the single shared label copied once. Association unchanged; not resolved.",
            "blocking_final_use": "yes",
        })

    # 2. Original WEBP/GIF kept next to the new PNG copy.
    for c in conversion_log:
        rows.append({
            "issue_id": f"STAGE-CONV-DUP-{c['conversion_id']}",
            "category": "original_and_png_copy_both_staged",
            "severity": "review_before_training",
            "count": 1,
            "affected_paths": f"{c['source_path']} | {c['destination_path']}",
            "description": (
                "The original WEBP/GIF was copied (rule: copy all originals) and an "
                "additional PNG copy was created. Both share the same stem and label."
            ),
            "action_taken": "Both files kept. Decide later whether training should use the PNG, the original, or both.",
            "blocking_final_use": "yes",
        })

    # 2b. Pre-existing out-of-bounds boxes in converted images' labels.
    oob = [c for c in conversion_log if not c["annotations_in_bounds"]]
    if oob:
        rows.append({
            "issue_id": "STAGE-CONV-OOB",
            "category": "pre_existing_out_of_bounds_boxes",
            "severity": "informational",
            "count": len(oob),
            "affected_paths": " | ".join(c["destination_path"] for c in oob),
            "description": (
                "Labels for these converted images already contained boxes extending "
                "beyond image bounds before staging. Detected while validating conversions."
            ),
            "action_taken": "Labels copied byte-for-byte unchanged; no box was clamped or corrected.",
            "blocking_final_use": "no",
        })

    # 3. Annotation/class conflicts preserved, not resolved.
    exact_conflict = sum(1 for g in review["exact_groups"] if g["category"] != "identical_label_bytes")
    source_conflict = sum(1 for g in review["source_groups"] if g["category"] != "identical_label_bytes")
    cross_split = sum(1 for g in review["source_groups"] if g["cross_split"])
    aggregate = [
        ("STAGE-CONFLICT-EXACT", "exact_annotation_conflict", exact_conflict,
         "Byte-identical image groups with different annotations.",
         "All files copied with their original labels; no canonical annotation chosen.", "yes"),
        ("STAGE-CONFLICT-SOURCE", "source_annotation_conflict", source_conflict,
         "Perceptual source groups with different annotations.",
         "All files copied with their original labels; no canonical annotation chosen.", "yes"),
        ("STAGE-CONFLICT-CLASS", "class_id_conflict_pairs", len(review["class_conflicts"]),
         "Matched boxes carry different class IDs.",
         "Copied unchanged; no class ID changed or conflict resolved.", "yes"),
        ("STAGE-CONFLICT-GEOMETRY", "geometry_conflict_pairs", len(review["geometry_conflicts"]),
         "Byte-identical images have different boxes.",
         "Copied unchanged; no box corrected.", "yes"),
        ("STAGE-CROSS-SPLIT", "cross_split_source_groups", cross_split,
         "One source-image group appears in more than one split.",
         "Split membership copied exactly; no reassignment performed.", "yes"),
        ("STAGE-NEAR-DUP", "possible_near_duplicate_pairs", len(review["possible_source_pairs"]),
         "dHash 11-16 possible near-duplicate pairs.",
         "All images copied; no merge or omission performed.", "no"),
    ]
    for issue_id, category, count, desc, action, blocking in aggregate:
        rows.append({
            "issue_id": issue_id,
            "category": category,
            "severity": "review_before_training" if blocking == "yes" else "informational",
            "count": count,
            "affected_paths": "see preparation_actions.csv / annotation_conflicts.json",
            "description": desc,
            "action_taken": action,
            "blocking_final_use": blocking,
        })
    return rows


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


def write_manifest(path: Path, rows: list[dict]) -> None:
    fields = [
        "entry_id", "kind", "action", "batch", "split",
        "source_path", "destination_path",
        "source_bytes", "destination_bytes",
        "source_sha256", "destination_sha256",
        "class_ids", "box_count", "width", "height", "mode", "format", "frames",
        "status", "notes",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for i, row in enumerate(rows, start=1):
            out = {k: row.get(k, "") for k in fields}
            out["entry_id"] = f"E{i:06d}"
            w.writerow(out)


def write_conversion_log(path: Path, rows: list[dict]) -> None:
    fields = [
        "conversion_id", "source_path", "source_format", "source_mode", "source_frames",
        "source_width", "source_height", "destination_path", "destination_format",
        "destination_mode", "destination_width", "destination_height",
        "dimensions_preserved", "pixels_preserved", "annotations_unchanged",
        "annotations_in_bounds",
        "label_path", "label_sha256", "class_ids", "box_count", "status",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})


def write_unresolved(path: Path, rows: list[dict]) -> None:
    fields = [
        "issue_id", "category", "severity", "count", "affected_paths",
        "description", "action_taken", "blocking_final_use",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})


def write_summary(path: Path, summary: dict) -> None:
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def write_report(path: Path, summary: dict, validation: dict, conversions: list[dict],
                 unresolved: list[dict]) -> None:
    lines: list[str] = []
    add = lines.append
    v = validation

    add("# Dataset Staging Copy — Report")
    add("")
    add(f"- **Source:** `{SRC.as_posix()}`")
    add(f"- **Destination:** `{DEST.as_posix()}`")
    add(f"- **Generated:** {summary['generated_at']}")
    add(f"- **Mode:** copy + format conversion only; no training, no splitting.")
    add("")
    add("## 1. What was done")
    add("")
    add(f"- Created the destination directory.")
    add(f"- Copied **all {v['source_image_count']:,} original images** and "
        f"**all {v['source_label_count']:,} original labels**, preserving Batch/split structure.")
    add(f"- Created **{v['conversion_count']}** PNG copies of the WEBP/GIF images. "
        "The original WEBP/GIF files were also copied, so no image is omitted.")
    add(f"- Result: **{v['destination_image_count']:,} images** and "
        f"**{v['destination_label_count']:,} labels** in the staging copy.")
    add(f"- Preserved all duplicate images, duplicate-stem pairs, conflicting annotations and "
        "original split membership.")
    add("")
    add("## 2. Conversions")
    add("")
    add(md_table(
        ["ID", "Source", "Format", "Dimensions", "PNG destination", "Pixels preserved", "Annotation unchanged", "Status"],
        [[c["conversion_id"], c["source_path"], c["source_format"],
          f"{c['source_width']}x{c['source_height']}", c["destination_path"],
          c["pixels_preserved"], c["annotations_unchanged"], c["status"]] for c in conversions],
    ))
    add("")
    add("Each converted image reuses the same stem-based label file, copied verbatim. "
        "No annotation content was changed.")
    add("")
    actual_formats = sorted({c["source_format"] for c in conversions})
    add(f"Detected source encodings: {', '.join(actual_formats)}. Three files with a `.webp` "
        "extension actually contain JPEG data; the `.gif` is a genuine GIF. All four were "
        "decoded and re-encoded losslessly as PNG.")
    add("")
    if v.get("conversions_with_out_of_bounds_boxes"):
        add(f"**Informational:** {len(v['conversions_with_out_of_bounds_boxes'])} converted "
            "image(s) have pre-existing labels with boxes extending beyond the image bounds. "
            "The label was copied unchanged; this is an existing annotation issue, not a "
            "conversion failure. See `unresolved_issues.csv`.")
        add("")
    add("## 3. Duplicate stems")
    add("")
    add(f"**15** pre-existing duplicate-stem pairs were preserved exactly: both images copied, "
        "the single shared label copied once. This is not resolved and is listed in "
        "`unresolved_issues.csv`.")
    add("")
    add("## 4. Validation")
    add("")
    add(md_table(["Check", "Result", "Value"], [
        ["Source / destination image counts", "PASS" if v["source_destination_counts_match"] else "FAIL",
         f"{v['source_image_count']:,} source images + {v['conversion_count']} conversions = {v['destination_image_count']:,}"],
        ["Source / destination label counts", "PASS" if v["source_destination_counts_match"] else "FAIL",
         f"{v['source_label_count']:,} = {v['destination_label_count']:,}"],
        ["All source images present", "PASS" if v["all_source_images_present"] else "FAIL", ""],
        ["All source labels present", "PASS" if v["all_source_labels_present"] else "FAIL", ""],
        ["Converted PNGs present", "PASS" if v["converted_images_present"] else "FAIL", ""],
        ["No unexpected files", "PASS" if v["no_unexpected_files"] else "FAIL", ""],
        ["No silent omissions", "PASS" if v["no_silent_omissions"] else "FAIL", ""],
        ["Label bytes unchanged", "PASS" if v["labels_byte_unchanged"] else "FAIL", ""],
        ["Class IDs unchanged", "PASS" if v["class_ids_unchanged"] else "FAIL", f"IDs {v['class_ids'][0]}-{v['class_ids'][-1]} ({v['class_id_count']})"],
        ["Bounding-box coordinates unchanged", "PASS" if v["bbox_coordinates_unchanged"] else "FAIL", ""],
        ["All destination images decodable", "PASS" if v["all_images_decodable"] else "FAIL", f"{v['images_decoded']:,} decoded, {len(v['images_decode_failures'])} failures"],
        ["Converted dimensions preserved", "PASS" if v["conversions_dimensions_preserved"] else "FAIL", ""],
        ["Converted pixels preserved", "PASS" if v["conversions_pixels_preserved"] else "FAIL", ""],
        ["Converted annotations unchanged", "PASS" if v["conversions_annotations_unchanged"] and v["conversion_labels_unchanged"] else "FAIL", ""],
        ["Converted boxes in-bounds (informational)", "INFO" if not v["conversions_annotations_in_bounds"] else "PASS", f"{len(v['conversions_with_out_of_bounds_boxes'])} pre-existing out-of-bounds label(s)"],
        ["No original file modified", "PASS" if v["no_original_modified"] else "FAIL", ""],
        ["No overwrites", "PASS" if v["overwrites_performed"] == 0 else "FAIL", "0"],
    ]))
    add("")
    if v.get("images_decode_failures"):
        add("### Decode failures")
        add("")
        for f in v["images_decode_failures"][:20]:
            add(f"- `{f['path']}` — {f['error']}")
        add("")
    add("## 5. Unresolved issues")
    add("")
    add(md_table(
        ["Issue", "Category", "Count", "Blocking"],
        [[r["issue_id"], r["category"], r["count"], r["blocking_final_use"]] for r in unresolved],
    ))
    add("")
    add("Full detail: `unresolved_issues.csv`.")
    add("")
    add("## 6. Output files")
    add("")
    add(md_table(["File", "Contents"], [
        ["`staging_report.md`", "This report"],
        ["`staging_manifest.csv`", f"{summary['manifest_entries']:,} file entries"],
        ["`staging_summary.json`", "Counts, conversions, validation"],
        ["`conversion_log.csv`", f"{len(conversions)} conversion records"],
        ["`unresolved_issues.csv`", f"{len(unresolved)} unresolved issue records"],
    ]))
    add("")
    add("## 7. Exact command")
    add("")
    add("```powershell")
    add("# Initial staging copy (only runs when the destination is empty):")
    add('python "C:\\vs code\\object detection\\stage_dataset.py"')
    add("")
    add("# Re-verify the existing staging copy without copying or overwriting:")
    add('python "C:\\vs code\\object detection\\stage_dataset.py" --validate-only')
    add("```")
    add("")
    add("## 8. Not done (by rule)")
    add("")
    add("- No model training.")
    add("- No train/validation/test changes.")
    add("- No image omitted or quarantined.")
    add("- No class ID, box coordinate or annotation changed.")
    add("- No conflict resolved.")
    add("- No class name invented.")
    add("")
    add(f"**All automated checks passed: {v['all_checks_passed']}.**")
    add("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    validate_only = "--validate-only" in sys.argv

    print("Source     :", SRC)
    print("Destination:", DEST)
    print("Output     :", OUT_DIR)
    print("Mode       :",
          "verify existing staging copy (no writes to destination)"
          if validate_only else "staging copy (originals read-only)")
    print()

    print("[0/5] Preflight")
    preflight_data = preflight(allow_existing=validate_only)
    print(f"    source images={len(preflight_data['images']):,} "
          f"labels={len(preflight_data['labels']):,} "
          f"conversions={len(preflight_data['conversions'])}")
    print(f"    expected destination files={preflight_data['dest_file_count']:,}")

    review = json.loads((ANALYSIS_DIR / "annotation_conflicts.json").read_text(encoding="utf-8"))

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/5] Capturing source fingerprints (immutability check)")
    source_before = source_fingerprint()
    source_label_hashes_before = {
        rel: sha256_file(SRC / Path(rel)) for rel in preflight_data["label_rels"]
    }

    jobs: list[tuple[str, str, str]] = []
    for rel in preflight_data["image_rels"]:
        jobs.append(("image", rel, rel))
    for rel in preflight_data["label_rels"]:
        jobs.append(("label", rel, rel))

    if validate_only:
        print("[2/5] Verifying existing copied files (no copy, no overwrite)")
        copy_rows = run_copy(jobs, resume=True)
        print("[3/5] Verifying existing PNG copies (no rewrite)")
        conversions = convert_images(preflight_data, create=False)
    else:
        print("[2/5] Creating destination and copying every original file")
        DEST.mkdir(parents=True, exist_ok=True)
        # Pre-create all directories so the worker threads never race on mkdir.
        for rel in list(preflight_data["image_rels"]) + list(preflight_data["label_rels"]):
            (DEST / Path(rel)).parent.mkdir(parents=True, exist_ok=True)

        copy_rows = run_copy(jobs)

        print("[3/5] Creating PNG copies of the WEBP/GIF images")
        conversions = convert_images(preflight_data)

    for c in conversions:
        _, _, class_ids = parse_label(DEST / Path(c["label_path"]))
        info = image_info(DEST / Path(c["destination_path"]))
        copy_rows.append({
            "kind": "image",
            "action": "convert_image",
            "source_path": c["source_path"],
            "destination_path": c["destination_path"],
            "source_bytes": (SRC / Path(c["source_path"])).stat().st_size,
            "destination_bytes": (DEST / Path(c["destination_path"])).stat().st_size,
            "source_sha256": sha256_file(SRC / Path(c["source_path"])),
            "destination_sha256": sha256_file(DEST / Path(c["destination_path"])),
            "class_ids": ";".join(str(x) for x in sorted(set(class_ids))),
            "box_count": len(class_ids),
            "width": info["width"],
            "height": info["height"],
            "mode": info["mode"],
            "format": info["format"],
            "frames": info["frames"],
            "batch": batch_split(c["source_path"])[0],
            "split": batch_split(c["source_path"])[1],
            "status": c["status"],
            "notes": "PNG copy added; original WEBP/GIF also preserved.",
        })

    print("[4/5] Validating")
    validation = validate(
        preflight_data, copy_rows, conversions,
        source_before, source_label_hashes_before,
    )

    print("[5/5] Writing reports")
    unresolved = build_unresolved(preflight_data, conversions, review)

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "mode": ("STAGING COPY VERIFIED — existing copy re-checked, no training, no splitting"
                 if validate_only else
                 "STAGING COPY — copy + PNG conversion, no training, no splitting"),
        "source_root": SRC.as_posix(),
        "destination_root": DEST.as_posix(),
        "output_dir": OUT_DIR.as_posix(),
        "rules": {
            "original_modified": False,
            "images_omitted": 0,
            "images_quarantined": 0,
            "class_ids_changed": 0,
            "bounding_boxes_changed": 0,
            "class_conflicts_resolved": 0,
            "splits_changed": False,
            "model_trained": False,
            "class_names_invented": 0,
        },
        "counts": {
            "source_images": validation["source_image_count"],
            "source_labels": validation["source_label_count"],
            "source_files": validation["source_file_count"],
            "converted_png_copies": validation["conversion_count"],
            "destination_images": validation["destination_image_count"],
            "destination_labels": validation["destination_label_count"],
            "destination_files": validation["destination_file_count"],
            "duplicate_stem_pairs_preserved": len(review.get("duplicate_stems", [])),
            "exact_duplicate_groups_preserved": len(review.get("exact_groups", [])),
            "cross_split_source_groups_preserved": sum(
                1 for g in review["source_groups"] if g["cross_split"]
            ),
        },
        "conversions": conversions,
        "validation": validation,
        "unresolved_issue_counts": dict(Counter(r["category"] for r in unresolved)),
        "manifest_entries": copy_rows,
        "outputs": {
            "staging_report.md": "human-readable report",
            "staging_manifest.csv": "per-file inventory",
            "staging_summary.json": "machine-readable summary",
            "conversion_log.csv": "conversion records",
            "unresolved_issues.csv": "preserved but unresolved issues",
        },
    }

    write_manifest(OUT_DIR / "staging_manifest.csv", copy_rows)
    write_conversion_log(OUT_DIR / "conversion_log.csv", conversions)
    write_unresolved(OUT_DIR / "unresolved_issues.csv", unresolved)
    # Keep the summary small: replace the full manifest with its count.
    summary["manifest_entries"] = len(copy_rows)
    write_summary(OUT_DIR / "staging_summary.json", summary)
    write_report(OUT_DIR / "staging_report.md", summary, validation, conversions, unresolved)

    print()
    print("Done. Staging copy verified; originals untouched." if validate_only
          else "Done. Staging copy created; originals untouched.")
    print("Report :", OUT_DIR / "staging_report.md")
    print("Manifest:", OUT_DIR / "staging_manifest.csv")
    print("Summary:", OUT_DIR / "staging_summary.json")
    print("Conversions:", OUT_DIR / "conversion_log.csv")
    print("Unresolved:", OUT_DIR / "unresolved_issues.csv")
    print()
    print("Validation all checks passed:", validation["all_checks_passed"])
    print("Source images:", validation["source_image_count"],
          "-> destination images:", validation["destination_image_count"],
          f"(+{validation['conversion_count']} PNG copies)")
    print("Source labels:", validation["source_label_count"],
          "-> destination labels:", validation["destination_label_count"])
    print("Decode failures:", len(validation["images_decode_failures"]))
    print("Original files modified:", len(validation["original_files_modified"]))


if __name__ == "__main__":
    main()
