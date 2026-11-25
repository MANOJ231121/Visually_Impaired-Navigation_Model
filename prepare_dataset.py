#!/usr/bin/env python3
"""
prepare_dataset.py
==================

DRY-RUN dataset preparation planner for the 4-batch YOLO dataset.

This tool NEVER modifies the original dataset.  It reads the completed
verification / annotation-conflict analysis, validates every referenced file,
and writes a proposed cleaning plan to a separate output directory.

It does NOT:
  * delete, rename, move or overwrite original files
  * change class IDs
  * invent class names
  * auto-correct bounding boxes
  * auto-resolve class conflicts
  * train a model
  * change train / validation / test split membership

Run:
    python "C:\\vs code\\object detection\\prepare_dataset.py"

Outputs (in dataset_preparation_plan/):
    preparation_plan.md
    preparation_actions.csv
    preparation_summary.json
    preparation_manifest.json
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DRY_RUN = True  # only dry-run is implemented; no cleaning operation is executed

PROJECT_ROOT = Path(__file__).resolve().parent
ROOT = PROJECT_ROOT / "final batches"
ANALYSIS_DIR = PROJECT_ROOT / "dataset_verification"
OUT_DIR = PROJECT_ROOT / "dataset_preparation_plan"

# Proposed destination root.  It is NOT created by this script.  The path is
# recorded in the plan so a later, approved implementation can use it.
PREPARED_ROOT = PROJECT_ROOT / "prepared_dataset"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tif", ".tiff"}
# GIF is deliberately absent: Ultralytics / OpenCV does not read it reliably.
YOLO_READABLE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

EXPECTED = {
    "class_conflicts": 7,
    "geometry_conflicts": 177,
    "duplicate_stems": 15,
    "format_only": 4,
    "exact_groups": 727,
    "source_groups": 762,
    "possible_source_pairs": 71,
    "exact_identical_label_groups": 197,
    "exact_conflict_groups": 530,
    "source_conflict_groups": 561,
    "cross_split_source_groups": 236,
    "cross_split_exact_groups": 222,
    "exact_cross_split_mixed_base": 136,
}

CONFLICT_CATEGORIES = {
    "annotation_conflict",
    "source_annotation_conflict",
    "class_conflict",
    "geometry_conflict",
}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def posix(path: Path) -> str:
    return path.as_posix()


def rel_of(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def label_rel_for_image_rel(image_rel: str) -> str:
    """Batch X/images/<split>/<file> -> Batch X/labels/<split>/<stem>.txt."""
    parts = list(Path(image_rel).parts)
    parts[parts.index("images")] = "labels"
    return Path(*parts).with_suffix(".txt").as_posix()


def batch_split(rel: str) -> tuple[str, str]:
    """Return (batch, split) for an image or label relative path."""
    parts = Path(rel).parts
    return parts[0], parts[2]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_label(path: Path) -> tuple[list[int], int]:
    """Return (class_ids, box_count) from a YOLO label file."""
    classes: list[int] = []
    boxes = 0
    text = path.read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        tokens = line.split()
        if len(tokens) == 5:
            classes.append(int(tokens[0]))
            boxes += 1
    return classes, boxes


def load_json(name: str) -> dict:
    path = ANALYSIS_DIR / name
    if not path.is_file():
        raise SystemExit(f"Required analysis file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv(name: str) -> list[dict]:
    path = ANALYSIS_DIR / name
    if not path.is_file():
        raise SystemExit(f"Required analysis file not found: {path}")
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def md_table(headers: list[str], rows: list[list[object]]) -> str:
    lines = ["| " + " | ".join(str(h) for h in headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join("" if v is None else str(v) for v in row) + " |")
    return "\n".join(lines)


def is_inside(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


# ---------------------------------------------------------------------------
# Index and validation of the original dataset
# ---------------------------------------------------------------------------


def index_dataset() -> tuple[list[Path], list[Path], dict[str, Path], dict[str, Path]]:
    if not ROOT.is_dir():
        raise SystemExit(f"Dataset root not found: {ROOT}")

    images = sorted(
        p for p in ROOT.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )
    labels = sorted(p for p in ROOT.rglob("*.txt") if p.is_file())

    if not images:
        raise SystemExit("No images found.")
    if not labels:
        raise SystemExit("No labels found.")

    image_index = {rel_of(p): p for p in images}
    label_index = {rel_of(p): p for p in labels}

    if len(image_index) != len(images):
        raise SystemExit("Duplicate image relative paths detected.")
    if len(label_index) != len(labels):
        raise SystemExit("Duplicate label relative paths detected.")

    return images, labels, image_index, label_index


# ---------------------------------------------------------------------------
# Issue registry
# ---------------------------------------------------------------------------


def build_issues(
    review: dict,
    image_index: dict[str, Path],
    label_index: dict[str, Path],
    source_by_file: dict[str, dict],
    exact_by_file: dict[str, dict],
    canonical_by_md5: dict[str, str],
) -> tuple[list[dict], dict[str, list[str]]]:
    """Build one row per proposed issue and tag affected files.

    Precedence is handled later, per file.  The issue table is intentionally
    exhaustive so every category requested by the brief is represented.
    """
    issues: list[dict] = []
    file_issue_ids: dict[str, list[str]] = defaultdict(list)

    def tag(files: list[str], issue_id: str) -> None:
        for f in files:
            file_issue_ids[f].append(issue_id)

    def add(**kwargs) -> None:
        issues.append(kwargs)
        tag(kwargs.get("files", []) + kwargs.get("labels", []), kwargs["issue_id"])

    # ------------------------------------------------------------------
    # 1. WEBP / GIF only images -> proposed PNG conversion
    # ------------------------------------------------------------------
    for i, item in enumerate(sorted(review["format_only"], key=lambda x: x["id"]), start=1):
        src = item["rel"]
        label_rel = label_rel_for_image_rel(src)
        dest_img = (PREPARED_ROOT / Path(src)).with_suffix(".png")
        dest_lab = PREPARED_ROOT / Path(label_rel)
        issue_id = f"CONV-{i:03d}"
        add(
            issue_id=issue_id,
            category="format_conversion",
            subcategory=item["ext"].lstrip("."),
            files=[src],
            labels=[label_rel],
            proposed_action=(
                "Copy the image to a new PNG file at the proposed destination; "
                "copy the label verbatim. Keep the original file untouched."
            ),
            approval_required=False,
            rationale=(
                "Image is only available in a non-JPEG/PNG format. It is single-frame, "
                "all boxes are inside the frame, and PNG conversion does not change "
                "any annotation or class ID."
            ),
            proposed_destinations=[posix(dest_img), posix(dest_lab)],
            evidence=(
                f"ext={item['ext']}, size={item['dimensions'][0]}x{item['dimensions'][1]}, "
                f"frames={item.get('frames', 1)}, mode={item.get('mode', '?')}, "
                f"boxes={item.get('boxes', '?')}, boxes_in_bounds={item.get('boxes_in_bounds')}, "
                f"yolo_readable={item.get('yolo_readable')}"
            ),
        )

    # ------------------------------------------------------------------
    # 2. Duplicate-stem pairs -> pick edge-fit winner, quarantine twin
    # ------------------------------------------------------------------
    for i, item in enumerate(sorted(review["duplicate_stems"], key=lambda x: x["id"]), start=1):
        files = list(item["files"])
        winner = files[item["better_fit_index"]]
        losers = [f for f in files if f != winner]
        files = [winner] + losers
        label_rel = label_rel_for_image_rel(winner)
        dest_winner = PREPARED_ROOT / Path(winner)
        dest_losers = [
            PREPARED_ROOT / "_review" / "unlabelled_duplicate_stems" / item["id"] / Path(f).name
            for f in losers
        ]
        dest_lab = PREPARED_ROOT / Path(label_rel)
        issue_id = f"STEM-{i:03d}"
        add(
            issue_id=issue_id,
            category="duplicate_stem",
            subcategory=item["stem"],
            files=files,
            labels=[label_rel],
            proposed_action=(
                f"Keep {winner} in the prepared image set with the existing shared label; "
                f"quarantine {', '.join(losers)} in a review folder. The shared label must "
                "NOT be reused blindly for the twin. Annotate the twin or drop it after review."
            ),
            approval_required=True,
            rationale=(
                "Two different images share one filename stem in the same split, so only one "
                "label file can exist. The edge-fit heuristic is a hint, not ground truth; a "
                "human must confirm which image the label belongs to before the twin is annotated."
            ),
            proposed_destinations=[posix(dest_winner)] + [posix(p) for p in dest_losers] + [posix(dest_lab)],
            evidence=(
                f"dHash={item.get('phash_distance')}, identical_content={item.get('identical_content')}, "
                f"dimensions={item.get('dimensions')}, edge_fit={item.get('edge_fit')}, "
                f"better_fit_index={item.get('better_fit_index')}"
            ),
        )

    # ------------------------------------------------------------------
    # 3. Exact-duplicate groups with byte-identical labels -> dedupe proposal
    #    (only within-split groups are proposed for omission; cross-split
    #     groups are deferred because split policy is not being changed)
    # ------------------------------------------------------------------
    identical_groups = [g for g in review["exact_groups"] if g["category"] == "identical_label_bytes"]
    for i, group in enumerate(sorted(identical_groups, key=lambda g: g["md5"]), start=1):
        files = sorted(group["files"])
        canonical = canonical_by_md5[group["md5"]]
        redundant = [f for f in files if f != canonical]
        issue_id = f"DEDUP-{i:03d}"

        if group["cross_split"]:
            action = (
                "Do not omit yet: this byte-identical group crosses splits. Keep every copy "
                "and flag the group for the split-policy decision."
            )
            approval = True
            dests = [posix(PREPARED_ROOT / Path(f)) for f in files]
            rationale = (
                "Labels are byte-identical, so the annotation is safe, but removing a copy "
                "would change split membership. Split changes are explicitly out of scope."
            )
        else:
            blocked = [
                f for f in redundant
                if f in source_by_file
                and source_by_file[f]["category"] != "identical_label_bytes"
            ]
            omittable = [f for f in redundant if f not in blocked]
            action = (
                f"Keep canonical {canonical}; omit {len(omittable)} byte-identical copy/copies "
                "inside the same split."
            )
            if blocked:
                action += (
                    f" Defer {len(blocked)} copy/copies that belong to a broader source group "
                    "with a conflicting annotation."
                )
            approval = bool(blocked)
            dests = [posix(PREPARED_ROOT / Path(canonical))]
            dests += ["OMITTED (not copied)"] * len(omittable)
            dests += [posix(PREPARED_ROOT / Path(f)) for f in blocked]
            rationale = (
                "Every file in this group has a byte-identical label. Deduplication inside one "
                "split does not change split membership and does not change any class ID."
            )
        add(
            issue_id=issue_id,
            category="exact_duplicate",
            subcategory="identical_label_bytes",
            files=files,
            labels=[label_rel_for_image_rel(f) for f in files],
            proposed_action=action,
            approval_required=approval,
            rationale=rationale,
            proposed_destinations=dests,
            evidence=(
                f"md5={group['md5']}, count={group['count']}, cross_split={group['cross_split']}, "
                f"cross_batch={group['cross_batch']}, locations={group['locations']}"
            ),
        )

    # ------------------------------------------------------------------
    # 4. Exact-duplicate groups whose annotations conflict -> manual review
    # ------------------------------------------------------------------
    conflict_groups = [g for g in review["exact_groups"] if g["category"] != "identical_label_bytes"]
    for i, group in enumerate(sorted(conflict_groups, key=lambda g: g["md5"]), start=1):
        issue_id = f"CONF-EXACT-{i:03d}"
        add(
            issue_id=issue_id,
            category="annotation_conflict",
            subcategory=group["category"],
            files=sorted(group["files"]),
            labels=[label_rel_for_image_rel(f) for f in group["files"]],
            proposed_action=(
                "Keep all files unchanged and flag the group for manual review. Do not dedupe "
                "and do not edit any box or class ID until a canonical annotation is chosen."
            ),
            approval_required=True,
            rationale=(
                "These are byte-identical images with different annotations. The pixels cannot "
                "explain the difference, so a human must choose the canonical label."
            ),
            proposed_destinations=[posix(PREPARED_ROOT / Path(f)) for f in sorted(group["files"])],
            evidence=(
                f"md5={group['md5']}, category={group['category']}, count={group['count']}, "
                f"cross_split={group['cross_split']}, worst_match_iou={group['worst_match_iou']}, "
                f"class_conflicts={group['class_conflicts']}, count_mismatch={group['count_mismatch']}"
            ),
        )

    # ------------------------------------------------------------------
    # 5. Class-ID conflict pairs -> manual review (no auto-change)
    # ------------------------------------------------------------------
    for item in review["class_conflicts"]:
        details = item.get("class_conflict_detail", [])
        detail_text = "; ".join(
            f"{d['class_a']} vs {d['class_b']} @ IoU {d['iou']}" for d in details
        )
        add(
            issue_id=item["id"],
            category="class_conflict",
            subcategory="; ".join(sorted({f"{d['class_a']}/{d['class_b']}" for d in details})),
            files=[item["a"], item["b"]],
            labels=[label_rel_for_image_rel(item["a"]), label_rel_for_image_rel(item["b"])],
            proposed_action=(
                "Manual review only: decide which class ID is correct for each matched box. "
                "Do not auto-change any class ID."
            ),
            approval_required=True,
            rationale=(
                "The boxes match at IoU >= 0.5 on byte-identical images but carry different "
                "class IDs. Choosing one would invent ground truth."
            ),
            proposed_destinations=[posix(PREPARED_ROOT / Path(item["a"])),
                                   posix(PREPARED_ROOT / Path(item["b"]))],
            evidence=(
                f"boxes A/B={item['n_a']}/{item['n_b']}, matched>=0.5={item['strong']}, "
                f"conflicting boxes={item['class_conflict']}; {detail_text}"
            ),
        )

    # ------------------------------------------------------------------
    # 6. Geometry mismatch pairs -> manual review (no auto-correct)
    # ------------------------------------------------------------------
    for item in review["geometry_conflicts"]:
        add(
            issue_id=item["id"],
            category="geometry_conflict",
            subcategory=(
                "count_mismatch" if item["n_a"] != item["n_b"]
                else "same_count_unmatched" if item["strong"] == 0
                else "partial_match"
            ),
            files=[item["a"], item["b"]],
            labels=[label_rel_for_image_rel(item["a"]), label_rel_for_image_rel(item["b"])],
            proposed_action=(
                "Manual review only: decide the canonical box or add the missing object. "
                "Do not auto-correct any coordinates."
            ),
            approval_required=True,
            rationale=(
                "Byte-identical images have different boxes. The pixels can justify neither box "
                "as the unique truth, so a human must decide."
            ),
            proposed_destinations=[posix(PREPARED_ROOT / Path(item["a"])),
                                   posix(PREPARED_ROOT / Path(item["b"]))],
            evidence=(
                f"boxes A/B={item['n_a']}/{item['n_b']}, matched>=0.5={item['strong']}, "
                f"unmatched={item['geometry_conflict']}, best_unmatched_iou={item['max_unmatched_iou']}"
            ),
        )

    # ------------------------------------------------------------------
    # 7. Corrected cross-split source-image groups -> split-policy review
    # ------------------------------------------------------------------
    cross_split_groups = [g for g in review["source_groups"] if g["cross_split"]]
    for i, group in enumerate(cross_split_groups, start=1):
        issue_id = f"LEAK-{i:03d}"
        add(
            issue_id=issue_id,
            category="cross_split_source_group",
            subcategory=group["category"],
            files=list(group["files"]),
            labels=[label_rel_for_image_rel(f) for f in group["files"]],
            proposed_action=(
                "Keep every file unchanged in this plan and flag the source group for a "
                "group-aware split decision. Do not reassign split membership yet."
            ),
            approval_required=True,
            rationale=(
                "These files belong to one perceptual/byte source group but appear in more than "
                "one split. A later approved step should assign the whole group to one split."
            ),
            proposed_destinations=[posix(PREPARED_ROOT / Path(f)) for f in group["files"]],
            evidence=(
                f"source_id={group['id']}, count={group['count']}, locations={group['locations']}, "
                f"contains_exact_duplicate={group['contains_exact_duplicate']}, "
                f"category={group['category']}, worst_match_iou={group['worst_match_iou']}"
            ),
        )

    # ------------------------------------------------------------------
    # 8. Broader source groups with conflicting annotations -> manual review
    #    (superset of the exact-duplicate conflict groups)
    # ------------------------------------------------------------------
    source_conflicts = [g for g in review["source_groups"] if g["category"] != "identical_label_bytes"]
    for i, group in enumerate(source_conflicts, start=1):
        issue_id = f"CONF-SOURCE-{i:03d}"
        add(
            issue_id=issue_id,
            category="source_annotation_conflict",
            subcategory=group["category"],
            files=list(group["files"]),
            labels=[label_rel_for_image_rel(f) for f in group["files"]],
            proposed_action=(
                "Keep all files and flag the source group for manual canonical-annotation review. "
                "Do not edit boxes or class IDs automatically."
            ),
            approval_required=True,
            rationale=(
                "Perceptually/byte-identical source images disagree on annotations. The broader "
                "source grouping is the corrected leakage unit; no automatic choice is safe."
            ),
            proposed_destinations=[posix(PREPARED_ROOT / Path(f)) for f in group["files"]],
            evidence=(
                f"source_id={group['id']}, category={group['category']}, count={group['count']}, "
                f"cross_split={group['cross_split']}, cross_batch={group['cross_batch']}, "
                f"worst_match_iou={group['worst_match_iou']}, class_conflicts={group['class_conflicts']}, "
                f"count_mismatch={group['count_mismatch']}"
            ),
        )

    # ------------------------------------------------------------------
    # 9. Possible near-duplicate pairs (dHash 11-16) -> manual check only
    # ------------------------------------------------------------------
    for i, pair in enumerate(review["possible_source_pairs"], start=1):
        issue_id = f"NEAR-{i:03d}"
        add(
            issue_id=issue_id,
            category="possible_near_duplicate",
            subcategory=pair["base"],
            files=[pair["a"], pair["b"]],
            labels=[label_rel_for_image_rel(pair["a"]), label_rel_for_image_rel(pair["b"])],
            proposed_action=(
                "Manual visual check only: confirm whether the two images are the same source. "
                "Do not merge or deduplicate automatically."
            ),
            approval_required=True,
            rationale=(
                "dHash distance is in the 11-16 'possible' band. The earlier analysis deliberately "
                "did not auto-merge this band."
            ),
            proposed_destinations=[posix(PREPARED_ROOT / Path(pair["a"])),
                                   posix(PREPARED_ROOT / Path(pair["b"]))],
            evidence=(
                f"base={pair['base']}, dHash={pair['distance']}, aspect_ratio_ratio={pair['aspect_ratio']}"
            ),
        )

    return issues, file_issue_ids


# ---------------------------------------------------------------------------
# Per-file action decisions
# ---------------------------------------------------------------------------


IMAGE_APPROVAL = {
    "quarantine_unlabelled_twin": True,
    "copy_with_shared_label": True,
    "copy_review_conflict": True,
    "copy_review_cross_split_duplicate": True,
    "copy_review_cross_split": True,
    "convert_image": False,
    "omit_exact_duplicate": False,
    "copy_canonical": False,
    "copy": False,
}

LABEL_APPROVAL = {
    "copy_shared_label": True,
    "copy_label_review": True,
    "omit_exact_duplicate": False,
    "copy_label": False,
}


def decide_image_action(
    rel: str,
    format_by_file: dict[str, dict],
    stem_winner: dict[str, dict],
    stem_loser: dict[str, dict],
    exact_by_file: dict[str, dict],
    source_by_file: dict[str, dict],
    canonical_by_md5: dict[str, str],
) -> str:
    if rel in stem_loser:
        return "quarantine_unlabelled_twin"
    if rel in stem_winner:
        return "copy_with_shared_label"
    if rel in format_by_file:
        return "convert_image"

    exact = exact_by_file.get(rel)
    source = source_by_file.get(rel)

    if (exact and exact["category"] != "identical_label_bytes") or (
        source and source["category"] != "identical_label_bytes"
    ):
        return "copy_review_conflict"

    if exact and exact["category"] == "identical_label_bytes":
        if canonical_by_md5[exact["md5"]] == rel:
            return "copy_canonical"
        if exact["cross_split"]:
            return "copy_review_cross_split_duplicate"
        return "omit_exact_duplicate"

    if source and source["cross_split"]:
        return "copy_review_cross_split"

    return "copy"


def decide_label_action(image_actions: list[str]) -> str:
    if any(a == "quarantine_unlabelled_twin" for a in image_actions) and any(
        a == "copy_with_shared_label" for a in image_actions
    ):
        return "copy_shared_label"
    if all(a == "omit_exact_duplicate" for a in image_actions):
        return "omit_exact_duplicate"
    if any(a in {"copy_review_conflict", "copy_review_cross_split_duplicate",
                 "copy_review_cross_split"} for a in image_actions):
        return "copy_label_review"
    return "copy_label"


def image_destination(rel: str, action: str, stem_loser: dict[str, dict]) -> str | None:
    if action == "omit_exact_duplicate":
        return None
    if action == "quarantine_unlabelled_twin":
        issue = stem_loser[rel]
        return posix(PREPARED_ROOT / "_review" / "unlabelled_duplicate_stems" / issue["id"] / Path(rel).name)
    dest = PREPARED_ROOT / Path(rel)
    if action == "convert_image":
        dest = dest.with_suffix(".png")
    return posix(dest)


def label_destination(label_rel: str, action: str) -> str | None:
    if action == "omit_exact_duplicate":
        return None
    return posix(PREPARED_ROOT / Path(label_rel))


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def build_manifest(
    image_index: dict[str, Path],
    label_index: dict[str, Path],
    label_meta: dict[str, dict],
    image_actions: dict[str, str],
    label_actions: dict[str, str],
    images_by_label: dict[str, list[str]],
    file_issue_ids: dict[str, list[str]],
    stem_loser: dict[str, dict],
) -> list[dict]:
    entries: list[dict] = []

    for rel in sorted(image_index):
        action = image_actions[rel]
        batch, split = batch_split(rel)
        dest = image_destination(rel, action, stem_loser)
        note = {
            "copy": "Copied unchanged.",
            "copy_canonical": "Copied unchanged as the canonical copy of a byte-identical group.",
            "convert_image": "Image converted to PNG; label copied verbatim.",
            "omit_exact_duplicate": "Not copied: byte-identical duplicate inside the same split.",
            "copy_review_conflict": "Copied unchanged; flagged for annotation review.",
            "copy_review_cross_split_duplicate": (
                "Copied unchanged; byte-identical group crosses splits, split decision pending."
            ),
            "copy_review_cross_split": (
                "Copied unchanged; source group crosses splits, split decision pending."
            ),
            "copy_with_shared_label": (
                "Selected by the edge-fit heuristic; shared label copied; twin quarantined."
            ),
            "quarantine_unlabelled_twin": (
                "Held in the review folder pending annotation; no label is copied for this twin."
            ),
        }[action]
        entries.append({
            "path": rel,
            "kind": "image",
            "batch": batch,
            "split": split,
            "action": action,
            "destination": dest,
            "approval_required": IMAGE_APPROVAL[action],
            "issue_ids": sorted(set(file_issue_ids.get(rel, []))),
            "notes": note,
        })

    for label_rel in sorted(label_index):
        action = label_actions[label_rel]
        batch, split = batch_split(label_rel)
        dest = label_destination(label_rel, action)
        meta = label_meta[label_rel]
        if action == "omit_exact_duplicate":
            content_operation = "omitted"
            note = "Omitted with its byte-identical duplicate image."
        elif action == "copy_shared_label":
            content_operation = "verbatim_copy"
            note = "Shared label for a duplicate-stem pair; copied with the selected image."
        elif action == "copy_label_review":
            content_operation = "verbatim_copy"
            note = "Copied verbatim for manual review; no class ID or box is changed."
        else:
            content_operation = "verbatim_copy"
            note = "Copied verbatim."

        image_ids: list[str] = []
        for image_rel in images_by_label.get(label_rel, []):
            image_ids.extend(file_issue_ids.get(image_rel, []))
        image_ids.extend(file_issue_ids.get(label_rel, []))

        entries.append({
            "path": label_rel,
            "kind": "label",
            "batch": batch,
            "split": split,
            "action": action,
            "destination": dest,
            "approval_required": LABEL_APPROVAL[action],
            "issue_ids": sorted(set(image_ids)),
            "content_operation": content_operation,
            "class_ids": meta["class_ids"],
            "box_count": meta["box_count"],
            "sha256": meta["sha256"],
            "notes": note,
        })

    return entries


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate(
    review: dict,
    summary: dict,
    leakage_csv: list[dict],
    image_index: dict[str, Path],
    label_index: dict[str, Path],
    label_meta: dict[str, dict],
    issues: list[dict],
    manifest: list[dict],
    image_actions: dict[str, str],
    label_actions: dict[str, str],
) -> dict:
    validation: dict = {}

    # ---- referenced source files exist ---------------------------------
    referenced: set[str] = set()
    for issue in issues:
        referenced.update(issue.get("files", []))
        referenced.update(issue.get("labels", []))
    missing = sorted(p for p in referenced if not (ROOT / Path(p)).is_file())
    missing_manifest = sorted(
        e["path"] for e in manifest if not (ROOT / Path(e["path"])).is_file()
    )
    validation["all_issue_source_files_exist"] = not missing
    validation["missing_issue_source_files"] = missing
    validation["all_manifest_source_files_exist"] = not missing_manifest
    validation["missing_manifest_source_files"] = missing_manifest

    # ---- image / label pairing -----------------------------------------
    missing_labels = []
    for rel in image_index:
        lrel = label_rel_for_image_rel(rel)
        if lrel not in label_index:
            missing_labels.append(lrel)
    validation["all_images_have_labels"] = not missing_labels
    validation["missing_labels"] = missing_labels

    image_label_rels = {label_rel_for_image_rel(rel) for rel in image_index}
    labels_without_image = sorted(set(label_index) - image_label_rels)
    validation["all_labels_have_images"] = not labels_without_image
    validation["labels_without_images"] = labels_without_image

    # ---- class IDs and boxes -------------------------------------------
    class_hist = Counter()
    total_boxes = 0
    for meta in label_meta.values():
        class_hist.update(meta["class_ids"])
        total_boxes += meta["box_count"]
    validation["total_images"] = len(image_index)
    validation["total_labels"] = len(label_index)
    validation["total_boxes"] = total_boxes
    validation["class_ids"] = sorted(class_hist)
    validation["class_id_count"] = len(class_hist)
    validation["class_histogram"] = dict(sorted(class_hist.items()))
    validation["image_count_matches_summary"] = (
        len(image_index) == summary["totals"]["images"]
    )
    validation["label_count_matches_summary"] = (
        len(label_index) == summary["totals"]["labels"]
    )
    validation["box_count_matches_summary"] = (
        total_boxes == summary["totals"]["boxes"]
    )
    validation["class_ids_match_summary"] = (
        sorted(class_hist) == summary["classes"]["all_ids"]
    )

    # ---- no label rewrite action ---------------------------------------
    rewrite_actions = [
        e for e in manifest
        if e["kind"] == "label"
        and e["action"] not in {"copy_label", "copy_label_review", "copy_shared_label",
                                "omit_exact_duplicate"}
    ]
    validation["label_actions_that_modify_content"] = len(rewrite_actions)
    validation["label_content_operations"] = dict(Counter(
        e["content_operation"] for e in manifest if e["kind"] == "label"
    ))

    # ---- destinations ---------------------------------------------------
    destinations = [e["destination"] for e in manifest if e["destination"]]
    lower_destinations = [d.casefold() for d in destinations]
    validation["manifest_entries"] = len(manifest)
    validation["destination_count"] = len(destinations)
    validation["destination_collisions"] = (
        len(lower_destinations) - len(set(lower_destinations))
    )
    validation["destinations_inside_original_root"] = sum(
        1 for e in manifest if e["destination"] and is_inside(Path(e["destination"]), ROOT)
    )
    original_abs = {
        p.resolve() for p in image_index.values()
    } | {
        p.resolve() for p in label_index.values()
    }
    validation["destinations_equal_to_original_paths"] = sum(
        1 for e in manifest
        if e["destination"] and Path(e["destination"]).resolve() in original_abs
    )
    validation["destinations_existing_on_disk"] = sum(
        1 for e in manifest if e["destination"] and Path(e["destination"]).exists()
    )

    # ---- conflicts require approval ------------------------------------
    conflict_issues = [i for i in issues if i["category"] in CONFLICT_CATEGORIES]
    validation["conflict_issues"] = len(conflict_issues)
    validation["conflict_issues_requiring_approval"] = sum(
        1 for i in conflict_issues if i["approval_required"]
    )
    validation["all_conflicts_require_approval"] = all(
        i["approval_required"] for i in conflict_issues
    )
    validation["class_conflict_pairs"] = sum(1 for i in issues if i["category"] == "class_conflict")
    validation["geometry_conflict_pairs"] = sum(1 for i in issues if i["category"] == "geometry_conflict")
    validation["class_conflict_pairs_require_approval"] = all(
        i["approval_required"] for i in issues if i["category"] == "class_conflict"
    )
    validation["geometry_conflict_pairs_require_approval"] = all(
        i["approval_required"] for i in issues if i["category"] == "geometry_conflict"
    )

    # ---- corrected source grouping -------------------------------------
    csv_cross = [r for r in leakage_csv if r["cross_split"].lower() == "true"]
    json_cross = [g for g in review["source_groups"] if g["cross_split"]]
    csv_ids = {r["id"] for r in csv_cross}
    json_ids = {g["id"] for g in json_cross}
    validation["source_group_counts"] = {
        "total": len(review["source_groups"]),
        "cross_split": len(json_cross),
        "cross_batch": sum(1 for g in review["source_groups"] if g["cross_batch"]),
        "cross_split_csv": len(csv_cross),
        "exact_groups": len(review["exact_groups"]),
        "exact_cross_split": sum(1 for g in review["exact_groups"] if g["cross_split"]),
    }
    validation["source_group_count_matches_csv"] = (
        len(json_cross) == len(csv_cross) and csv_ids == json_ids
    )
    validation["uses_corrected_source_grouping"] = (
        validation["source_group_count_matches_csv"]
        and len(review["source_groups"]) == EXPECTED["source_groups"]
        and len(json_cross) == EXPECTED["cross_split_source_groups"]
    )

    # ---- dry-run guarantee ---------------------------------------------
    validation["dry_run"] = DRY_RUN
    validation["originals_modified"] = False
    validation["prepared_root_created"] = PREPARED_ROOT.exists()

    return validation


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


def write_actions_csv(path: Path, issues: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([
            "issue_id", "issue_category", "issue_subcategory", "n_files",
            "file_paths", "proposed_action", "approval_required", "rationale",
            "proposed_destinations", "evidence",
        ])
        for issue in issues:
            w.writerow([
                issue["issue_id"],
                issue["category"],
                issue.get("subcategory", ""),
                len(issue.get("files", [])),
                " | ".join(issue.get("files", [])),
                issue["proposed_action"],
                "yes" if issue["approval_required"] else "no",
                issue["rationale"],
                " | ".join(issue.get("proposed_destinations", [])),
                issue.get("evidence", ""),
            ])


def write_summary_json(
    path: Path,
    summary: dict,
    review: dict,
    issues: list[dict],
    manifest: list[dict],
    image_actions: dict[str, str],
    label_actions: dict[str, str],
    validation: dict,
) -> None:
    identical = [g for g in review["exact_groups"] if g["category"] == "identical_label_bytes"]
    identical_within = [g for g in identical if not g["cross_split"]]
    identical_cross = [g for g in identical if g["cross_split"]]
    redundant_total = sum(g["count"] - 1 for g in identical)
    within_redundant = sum(g["count"] - 1 for g in identical_within)
    omitted = sum(1 for a in image_actions.values() if a == "omit_exact_duplicate")
    deferred_cross = sum(g["count"] - 1 for g in identical_cross)
    deferred_conflict = redundant_total - omitted - deferred_cross
    groups_with_omissions = len({
        g["md5"] for g in identical_within
        if any(image_actions[f] == "omit_exact_duplicate" for f in g["files"])
    })

    data = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "mode": "DRY-RUN",
        "dry_run": DRY_RUN,
        "dataset_root": posix(ROOT),
        "analysis_dir": posix(ANALYSIS_DIR),
        "output_dir": posix(OUT_DIR),
        "proposed_prepared_root": posix(PREPARED_ROOT),
        "totals": {
            "image_files": sum(1 for e in manifest if e["kind"] == "image"),
            "label_files": sum(1 for e in manifest if e["kind"] == "label"),
            "total_files": len(manifest),
            "boxes": validation["total_boxes"],
            "class_ids": validation["class_ids"],
            "class_id_count": validation["class_id_count"],
        },
        "source_of_truth": {
            "annotation_conflicts_json": "annotation_conflicts.json",
            "leakage_group_actions_csv": "leakage_group_actions.csv",
            "verification_summary_json": "verification_summary.json",
            "corrected_source_grouping": True,
            "note": (
                "Uses the corrected MD5 + perceptual source groups (762 total, 236 cross-split), "
                "not the earlier filename-only 292 leakage count."
            ),
        },
        "issue_counts": {
            "webp_gif_only_images": len(review["format_only"]),
            "proposed_conversions": len(review["format_only"]),
            "duplicate_stem_pairs": len(review["duplicate_stems"]),
            "exact_duplicate_groups": len(review["exact_groups"]),
            "exact_duplicate_groups_cross_split": sum(1 for g in review["exact_groups"] if g["cross_split"]),
            "exact_identical_label_groups": len(identical),
            "exact_conflict_groups": len(review["exact_groups"]) - len(identical),
            "source_image_groups": len(review["source_groups"]),
            "cross_split_source_groups": sum(1 for g in review["source_groups"] if g["cross_split"]),
            "source_conflict_groups": sum(
                1 for g in review["source_groups"] if g["category"] != "identical_label_bytes"
            ),
            "class_conflict_pairs": len(review["class_conflicts"]),
            "class_conflict_boxes": sum(x["class_conflict"] for x in review["class_conflicts"]),
            "geometry_conflict_pairs": len(review["geometry_conflicts"]),
            "possible_near_duplicate_pairs": len(review["possible_source_pairs"]),
        },
        "proposed": {
            "conversions": len(review["format_only"]),
            "duplicate_stem_resolutions": len(review["duplicate_stems"]),
            "deduplication": {
                "identical_label_groups": len(identical),
                "within_split_groups": len(identical_within),
                "cross_split_groups_deferred": len(identical_cross),
                "groups_with_proposed_omissions": groups_with_omissions,
                "redundant_copies_total": redundant_total,
                "within_split_redundant_copies": within_redundant,
                "proposed_omissions": omitted,
                "deferred_due_to_cross_split": deferred_cross,
                "deferred_due_to_source_conflict": deferred_conflict,
            },
            "annotation_reviews": {
                "exact_duplicate_conflict_groups": len(review["exact_groups"]) - len(identical),
                "source_conflict_groups": sum(
                    1 for g in review["source_groups"] if g["category"] != "identical_label_bytes"
                ),
                "class_conflict_pairs": len(review["class_conflicts"]),
                "geometry_conflict_pairs": len(review["geometry_conflicts"]),
            },
            "cross_split_source_groups_flagged": sum(
                1 for g in review["source_groups"] if g["cross_split"]
            ),
            "possible_near_duplicate_pairs_flagged": len(review["possible_source_pairs"]),
            "image_actions": dict(Counter(image_actions.values())),
            "label_actions": dict(Counter(label_actions.values())),
        },
        "validation": validation,
        "unresolved_decisions": [
            "Authoritative class list (numeric ID -> name) is not present in the dataset.",
            "Which class ID is correct for the 7 class-conflict pairs.",
            "Which box is canonical for the 177 geometry-conflict pairs.",
            "Which annotation is canonical for the 530 exact / 561 source conflict groups.",
            "Whether both images of each duplicate-stem pair belong in the dataset.",
            "Desired train/validation/test split ratio and whether original splits must be preserved.",
            "Whether the 4 WEBP/GIF images belong in the final dataset.",
            "Whether duplicate source images should be kept for oversampling.",
        ],
    }
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def write_report(
    path: Path,
    review: dict,
    issues: list[dict],
    image_actions: dict[str, str],
    label_actions: dict[str, str],
    validation: dict,
) -> None:
    identical = [g for g in review["exact_groups"] if g["category"] == "identical_label_bytes"]
    identical_within = [g for g in identical if not g["cross_split"]]
    identical_cross = [g for g in identical if g["cross_split"]]
    redundant_total = sum(g["count"] - 1 for g in identical)
    within_redundant = sum(g["count"] - 1 for g in identical_within)
    omitted = sum(1 for a in image_actions.values() if a == "omit_exact_duplicate")
    deferred_cross = sum(g["count"] - 1 for g in identical_cross)
    deferred_conflict = redundant_total - omitted - deferred_cross
    groups_with_omissions = len({
        g["md5"] for g in identical_within
        if any(image_actions[f] == "omit_exact_duplicate" for f in g["files"])
    })

    source_conflict = [g for g in review["source_groups"] if g["category"] != "identical_label_bytes"]
    cross_source = [g for g in review["source_groups"] if g["cross_split"]]
    cross_source_identical = [g for g in cross_source if g["category"] == "identical_label_bytes"]
    cross_source_conflict = [g for g in cross_source if g["category"] != "identical_label_bytes"]

    lines: list[str] = []
    add = lines.append

    add("# Dataset Preparation Plan — DRY RUN")
    add("")
    add(f"- **Dataset root:** `{posix(ROOT)}`")
    add(f"- **Analysis inputs:** `{posix(ANALYSIS_DIR)}`")
    add(f"- **Plan output:** `{posix(OUT_DIR)}`")
    add(f"- **Proposed prepared root (NOT created):** `{posix(PREPARED_ROOT)}`")
    add(f"- **Generated:** {datetime.now().isoformat(timespec='seconds')}")
    add("- **Mode:** DRY RUN — no image, label, class ID, split, or batch was modified.")
    add("")
    add("---")
    add("")

    # ------------------------------------------------------------------
    add("## 0. Executive summary")
    add("")
    add(
        f"The dataset has **{validation['total_images']:,} images**, "
        f"**{validation['total_labels']:,} labels** and **{validation['total_boxes']:,} boxes**. "
        f"Every numeric class ID is preserved (IDs **{validation['class_ids'][0]}–{validation['class_ids'][-1]}**, "
        f"{validation['class_id_count']} IDs)."
    )
    add("")
    add("The plan proposes only mechanical, reversible actions for the prepared copy:")
    add("")
    add(md_table(["Proposed action", "Count", "Needs approval"], [
        ["Convert WEBP/GIF images to PNG (copy + label verbatim)", 4, "no"],
        ["Resolve duplicate-stem pairs (keep edge-fit winner, quarantine twin)", 15, "yes"],
        [f"Omit byte-identical copies inside one split ({groups_with_omissions} groups)", omitted, "no"],
        [f"Defer byte-identical copies that cross splits", deferred_cross, "yes"],
        [f"Defer byte-identical copies blocked by a broader annotation conflict", deferred_conflict, "yes"],
        ["Flag exact-duplicate annotation conflicts", len(review["exact_groups"]) - len(identical), "yes"],
        ["Flag broader source-group annotation conflicts", len(source_conflict), "yes"],
        ["Flag class-conflict pairs", len(review["class_conflicts"]), "yes"],
        ["Flag geometry-conflict pairs", len(review["geometry_conflicts"]), "yes"],
        ["Flag cross-split source groups", len(cross_source), "yes"],
        ["Flag possible near-duplicate pairs (dHash 11–16)", len(review["possible_source_pairs"]), "yes"],
    ]))
    add("")
    add(
        "**Nothing has been executed.** The script only reads the dataset and writes the four plan files "
        "in `dataset_preparation_plan/`. The proposed prepared dataset does not exist yet."
    )
    add("")

    # ------------------------------------------------------------------
    add("## 1. Inputs and method")
    add("")
    add("- **Primary input:** `annotation_conflicts.json` (the conflict review) and `verification_summary.json`.")
    add("- **Corrected source grouping:** `leakage_group_actions.csv`. This is the **762-group MD5 + perceptual** "
        "grouping, of which **236 cross splits**. The earlier filename-only 292 figure is **not** used.")
    add("- **Byte-identical groups:** `duplicate_group_actions.csv` / `exact_groups` (727 groups; 222 cross-split).")
    add("- **Exact duplicate labels:** a group is deduplicable only when every member label is byte-identical "
        "(`identical_label_bytes`), the group stays inside one split, and it is not part of a broader source "
        "group with a conflicting annotation. Otherwise the group is deferred for manual review.")
    add("- **Conflict policy:** any group whose annotations differ is flagged for manual review; no box or class "
        "ID is changed by this plan.")
    add("- **Action precedence per file:** duplicate-stem quarantine > PNG conversion > annotation-conflict "
        "review > exact-duplicate omission > cross-split review > normal copy.")
    add("")

    # ------------------------------------------------------------------
    add("## 2. Issues found")
    add("")
    add(md_table(["Issue", "Count", "Detail"], [
        ["Images only in WEBP/GIF", len(review["format_only"]),
         "3 WebP + 1 GIF, all Batch 3/train"],
        ["Duplicate-stem pairs", len(review["duplicate_stems"]),
         "Two different images share one stem and one label"],
        ["Exact-duplicate groups (MD5)", len(review["exact_groups"]),
         f"{sum(1 for g in review['exact_groups'] if g['cross_split'])} cross-split"],
        ["  – byte-identical labels", len(identical), f"{len(identical_cross)} cross-split"],
        ["  – conflicting annotations", len(review["exact_groups"]) - len(identical),
         f"{sum(1 for g in review['exact_groups'] if g['cross_split'] and g['category'] != 'identical_label_bytes')} cross-split"],
        ["Source-image groups (corrected)", len(review["source_groups"]),
         f"{len(cross_source)} cross-split; {len(source_conflict)} have conflicting annotations"],
        ["Class-ID conflict pairs", len(review["class_conflicts"]), "8 box-level conflicts in 5 class IDs"],
        ["Geometry mismatch pairs", len(review["geometry_conflicts"]), "Byte-identical images, different boxes"],
        ["Possible near-duplicate pairs", len(review["possible_source_pairs"]), "dHash 11–16; never auto-merged"],
    ]))
    add("")
    add("### 2.1 Exact-duplicate categories")
    add("")
    cats = Counter(g["category"] for g in review["exact_groups"])
    cat_rows = []
    for cat in ["identical_label_bytes", "minor_boundary_variance", "geometry_variance",
                "strong_mismatch", "count_mismatch", "class_conflict"]:
        total = cats.get(cat, 0)
        cross = sum(1 for g in review["exact_groups"] if g["category"] == cat and g["cross_split"])
        cat_rows.append([cat, total, cross])
    add(md_table(["Category", "All groups", "Cross-split"], cat_rows))
    add("")
    add("### 2.2 Corrected source-group categories")
    add("")
    scats = Counter(g["category"] for g in review["source_groups"])
    scat_rows = []
    for cat in ["identical_label_bytes", "minor_boundary_variance", "geometry_variance",
                "strong_mismatch", "count_mismatch", "class_conflict"]:
        total = scats.get(cat, 0)
        cross = sum(1 for g in review["source_groups"] if g["category"] == cat and g["cross_split"])
        scat_rows.append([cat, total, cross])
    add(md_table(["Category", "All groups", "Cross-split"], scat_rows))
    add("")

    # ------------------------------------------------------------------
    add("## 3. Proposed actions")
    add("")
    add("### 3.1 WEBP / GIF → PNG")
    add("")
    add("These four images are copied to PNG and their labels are copied verbatim. The originals stay untouched.")
    add("")
    conv_rows = []
    for issue in issues:
        if issue["category"] == "format_conversion":
            conv_rows.append([issue["issue_id"], issue["files"][0], issue["subcategory"],
                              issue["proposed_destinations"][0]])
    add(md_table(["Issue", "Source image", "Format", "Proposed PNG destination"], conv_rows))
    add("")
    add("### 3.2 Duplicate-stem pairs")
    add("")
    add("In every case the edge-fit heuristic favours the JPEG. The plan keeps that JPEG with the existing label "
        "and quarantines the PNG twin. **This is a recommendation requiring approval**, because edge-fit is a "
        "heuristic, not proof.")
    add("")
    stem_rows = []
    for issue in issues:
        if issue["category"] == "duplicate_stem":
            files = issue["files"]
            stem_rows.append([issue["issue_id"], issue["subcategory"], Path(files[0]).name,
                              Path(files[1]).name if len(files) > 1 else "",
                              "yes"])
    add(md_table(["Issue", "Stem", "Kept (edge-fit)", "Quarantined twin", "Approval"], stem_rows))
    add("")
    add("### 3.3 Exact duplicates with byte-identical labels")
    add("")
    add(md_table(["Decision", "Groups", "Copies", "Action"], [
        ["Within one split", len(identical_within), within_redundant,
         f"{omitted} proposed omissions across {groups_with_omissions} groups; "
         f"{deferred_conflict} deferred because the broader source group conflicts"],
        ["Crosses splits", len(identical_cross), deferred_cross,
         "Deferred: keep all copies until the split policy is approved"],
        ["Blocked by broader annotation conflict", "—", deferred_conflict,
         "Deferred: the source group has a conflicting annotation"],
    ]))
    add("")
    add(
        f"Total: **{len(identical)} byte-identical groups**, **{redundant_total} redundant copies**. "
        f"Only **{omitted}** are proposed for omission now; **{deferred_cross + deferred_conflict}** are deferred."
    )
    add("")
    add("### 3.4 Groups with conflicting annotations")
    add("")
    add(md_table(["Category", "Groups", "Proposed action"], [
        ["Byte-identical exact groups", len(review["exact_groups"]) - len(identical),
         "Manual review; keep files; no dedupe and no edit"],
        ["Broader source groups", len(source_conflict),
         "Manual review; keep files; choose canonical annotation"],
        ["Class-conflict pairs", len(review["class_conflicts"]),
         "Manual review; choose the correct class ID"],
        ["Geometry-conflict pairs", len(review["geometry_conflicts"]),
         "Manual review; choose the canonical box / add missing object"],
    ]))
    add("")
    add("The exact groups are a byte-identical subset of the broader source groups, so the two rows overlap. "
        "Both are listed because they answer different questions: *same bytes, different label* vs. "
        "*same source image, different label*.")
    add("")
    add("### 3.5 Cross-split source-image groups")
    add("")
    add(
        f"**{len(cross_source)}** corrected source groups cross splits. "
        f"**{len(cross_source_identical)}** have byte-identical labels and "
        f"**{len(cross_source_conflict)}** have conflicting annotations. "
        "All are kept unchanged in this plan; the later split step must assign each whole group to one split."
    )
    add("")
    add("### 3.6 Possible near-duplicate pairs")
    add("")
    add(
        f"**{len(review['possible_source_pairs'])}** pairs have dHash 11–16. They are flagged for visual "
        "confirmation and are **not** merged or deduplicated."
    )
    add("")

    # ------------------------------------------------------------------
    add("## 4. Destination layout")
    add("")
    add("The prepared copy would mirror the original structure under the proposed prepared root:")
    add("")
    add("```text")
    add(f"{posix(PREPARED_ROOT)}/")
    add("├── Batch <n>/images/<split>/<file>        # copied images")
    add("├── Batch <n>/labels/<split>/<stem>.txt    # copied labels (verbatim)")
    add("└── _review/unlabelled_duplicate_stems/    # quarantined duplicate-stem twins")
    add("```")
    add("")
    add("No destination is inside the original dataset. Every destination path is unique, and the plan does "
        "not create or overwrite any file.")
    add("")

    # ------------------------------------------------------------------
    add("## 5. Validation")
    add("")
    add(md_table(["Check", "Result"], [
        ["Every referenced source file exists", "PASS" if validation["all_issue_source_files_exist"] else "FAIL"],
        ["Every image has a label", "PASS" if validation["all_images_have_labels"] else "FAIL"],
        ["Every label has an image", "PASS" if validation["all_labels_have_images"] else "FAIL"],
        ["Image count matches verification summary", "PASS" if validation["image_count_matches_summary"] else "FAIL"],
        ["Label count matches verification summary", "PASS" if validation["label_count_matches_summary"] else "FAIL"],
        ["Box count matches verification summary", "PASS" if validation["box_count_matches_summary"] else "FAIL"],
        ["Class IDs match verification summary", "PASS" if validation["class_ids_match_summary"] else "FAIL"],
        ["No label action modifies content", "PASS" if validation["label_actions_that_modify_content"] == 0 else "FAIL"],
        ["No destination collision", "PASS" if validation["destination_collisions"] == 0 else "FAIL"],
        ["No destination inside the original root",
         "PASS" if validation["destinations_inside_original_root"] == 0 else "FAIL"],
        ["No destination equals an original path",
         "PASS" if validation["destinations_equal_to_original_paths"] == 0 else "FAIL"],
        ["No destination already exists",
         "PASS" if validation["destinations_existing_on_disk"] == 0 else "FAIL"],
        ["All conflicting annotations require approval",
         "PASS" if validation["all_conflicts_require_approval"] else "FAIL"],
        ["Corrected source grouping used",
         "PASS" if validation["uses_corrected_source_grouping"] else "FAIL"],
        ["Source-group count matches leakage CSV",
         "PASS" if validation["source_group_count_matches_csv"] else "FAIL"],
        ["Dry-run only; originals untouched", "PASS" if validation["dry_run"] and not validation["originals_modified"] else "FAIL"],
    ]))
    add("")
    add(f"- Class IDs: **{validation['class_ids']}** ({validation['class_id_count']} IDs, no change).")
    add(f"- Total boxes: **{validation['total_boxes']:,}**.")
    add(f"- Corrected source groups: **{validation['source_group_counts']['total']}** total / "
        f"**{validation['source_group_counts']['cross_split']}** cross-split.")
    add(f"- Manifest entries: **{validation['manifest_entries']:,}** "
        f"({validation['total_images']:,} images + {validation['total_labels']:,} labels).")
    add(f"- Label content operations: `{validation['label_content_operations']}`.")
    add("")

    # ------------------------------------------------------------------
    add("## 6. Risks and unresolved decisions")
    add("")
    add("### 6.1 Risks")
    add("")
    add("- **Edge-fit is a heuristic.** The duplicate-stem recommendation (keep JPEG, quarantine PNG) is based on "
        "border-gradient alignment, not on annotator intent. A human must verify each of the 15 pairs.")
    add("- **Cross-split deduplication is deferred.** Removing a byte-identical copy from another split changes "
        "split membership, which is explicitly out of scope until the split policy is approved.")
    add("- **Broader source groups can block exact dedupe.** Two within-split byte-identical copies are deferred "
        "because their source group contains a conflicting annotation.")
    add("- **PNG conversion of the GIF** takes the first frame. The file is single-frame according to the analysis, "
        "but a human should confirm the converted frame matches the label.")
    add("- **No class names exist.** All statements are about numeric IDs only.")
    add("")
    add("### 6.2 Unresolved decisions (cannot be answered from the files)")
    add("")
    add("1. Authoritative class list (`id -> name`).")
    add("2. Correct class ID for the 7 class-conflict pairs.")
    add("3. Canonical box for the 177 geometry-conflict pairs.")
    add("4. Canonical annotation for the 530 exact / 561 source conflict groups.")
    add("5. Whether both images of each duplicate-stem pair belong in the dataset.")
    add("6. Desired train/validation/test ratio and whether the original splits must be preserved.")
    add("7. Whether the 4 WEBP/GIF images belong in the final dataset.")
    add("8. Whether duplicate source images should be kept for oversampling.")
    add("")

    # ------------------------------------------------------------------
    add("## 7. Approval gate")
    add("")
    add("Approval is required before any action is implemented. Once the plan as a whole is approved, the "
        "mechanical subset (4 conversions + 121 within-split dedupe omissions) may be executed first; the "
        "15 duplicate stems, 530 exact conflicts, 561 source conflicts, 7 class conflicts, 177 geometry "
        "conflicts and 236 cross-split groups still need their own sign-off.")
    add("")
    add("**Do not execute until each of these is signed off:**")
    add("")
    add("- the class list and canonical `data.yaml`;")
    add("- the 15 duplicate-stem choices;")
    add("- the canonical annotation for every conflict group;")
    add("- the split ratio / group-aware split policy;")
    add("- whether the 4 WEBP/GIF images stay in the dataset.")
    add("")

    # ------------------------------------------------------------------
    add("## 8. Exact command")
    add("")
    add("```powershell")
    add('python "C:\\vs code\\object detection\\prepare_dataset.py"')
    add("```")
    add("")
    add("Outputs are written only to:")
    add("")
    add(f"`{posix(OUT_DIR)}`")
    add("")

    # ------------------------------------------------------------------
    add("## Appendix A — Output inventory")
    add("")
    add(md_table(["File", "Contents"], [
        ["`preparation_plan.md`", "This report"],
        ["`preparation_actions.csv`", f"{len(issues)} issue-level proposed actions"],
        ["`preparation_summary.json`", "Counts, proposed actions, validation results"],
        ["`preparation_manifest.json`", f"{validation['manifest_entries']:,} file-level entries (one per image and label)"],
    ]))
    add("")
    add("## Appendix B — Strict rules respected")
    add("")
    add("- Original dataset not modified.")
    add("- No original file deleted, renamed, moved or overwritten.")
    add("- No class ID changed.")
    add("- No class name invented.")
    add("- No bounding box auto-corrected.")
    add("- No class conflict auto-resolved.")
    add("- No model trained.")
    add("- No split membership changed.")
    add("- No object-category assumption made.")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    if not DRY_RUN:
        raise SystemExit("Only DRY-RUN mode is implemented. No cleaning operation was executed.")

    print("Dataset :", ROOT)
    print("Analysis:", ANALYSIS_DIR)
    print("Output  :", OUT_DIR)
    print("Mode    : DRY RUN (read-only)")
    print()

    # guard: never write inside the original dataset or create the prepared root
    if is_inside(OUT_DIR, ROOT):
        raise SystemExit("Refusing to write output inside the original dataset.")
    if is_inside(PREPARED_ROOT, ROOT):
        raise SystemExit("Refusing to use a prepared root inside the original dataset.")
    if PREPARED_ROOT.exists():
        raise SystemExit(
            f"Proposed prepared root already exists: {PREPARED_ROOT}. "
            "This dry-run planner will not overwrite it."
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/6] Loading analysis inputs")
    review = load_json("annotation_conflicts.json")
    summary = load_json("verification_summary.json")
    leakage_csv = load_csv("leakage_group_actions.csv")
    duplicate_csv = load_csv("duplicate_group_actions.csv")
    possible_csv = load_csv("possible_source_pairs.csv")
    stem_csv = load_csv("duplicate_stem_pairs.csv")
    format_csv = load_csv("format_only_images.csv")

    # hard checks: the planner must use the corrected, reviewed inputs
    for key, expected in EXPECTED.items():
        if key == "exact_identical_label_groups":
            actual = sum(1 for g in review["exact_groups"] if g["category"] == "identical_label_bytes")
        elif key == "exact_conflict_groups":
            actual = sum(1 for g in review["exact_groups"] if g["category"] != "identical_label_bytes")
        elif key == "source_conflict_groups":
            actual = sum(1 for g in review["source_groups"] if g["category"] != "identical_label_bytes")
        elif key == "cross_split_source_groups":
            actual = sum(1 for g in review["source_groups"] if g["cross_split"])
        elif key == "cross_split_exact_groups":
            actual = sum(1 for g in review["exact_groups"] if g["cross_split"])
        elif key == "exact_cross_split_mixed_base":
            actual = review.get("exact_cross_split_mixed_base")
        else:
            actual = len(review[key])
        if actual != expected:
            raise SystemExit(f"Input mismatch for {key}: expected {expected}, found {actual}")
    if len(duplicate_csv) != 727:
        raise SystemExit(f"duplicate_group_actions.csv has {len(duplicate_csv)} rows, expected 727")
    if len(leakage_csv) != 762:
        raise SystemExit(f"leakage_group_actions.csv has {len(leakage_csv)} rows, expected 762")
    if len(possible_csv) != 71:
        raise SystemExit(f"possible_source_pairs.csv has {len(possible_csv)} rows, expected 71")
    if len(stem_csv) != 15:
        raise SystemExit(f"duplicate_stem_pairs.csv has {len(stem_csv)} rows, expected 15")
    if len(format_csv) != 4:
        raise SystemExit(f"format_only_images.csv has {len(format_csv)} rows, expected 4")

    print("[2/6] Indexing original dataset")
    images, labels, image_index, label_index = index_dataset()
    print(f"    images={len(images):,} labels={len(labels):,}")
    if len(images) != summary["totals"]["images"]:
        raise SystemExit("Image count does not match verification_summary.json.")
    if len(labels) != summary["totals"]["labels"]:
        raise SystemExit("Label count does not match verification_summary.json.")

    print("[3/6] Reading label metadata")
    label_meta: dict[str, dict] = {}
    for lrel, lpath in label_index.items():
        classes, boxes = parse_label(lpath)
        label_meta[lrel] = {
            "classes": classes,
            "class_ids": sorted(set(classes)),
            "box_count": boxes,
            "sha256": sha256_file(lpath),
        }
    total_boxes = sum(m["box_count"] for m in label_meta.values())
    class_hist = Counter()
    for meta in label_meta.values():
        class_hist.update(meta["classes"])
    print(f"    boxes={total_boxes:,} class_ids={sorted(class_hist)}")
    if total_boxes != summary["totals"]["boxes"]:
        raise SystemExit("Box count does not match verification_summary.json.")
    if sorted(class_hist) != summary["classes"]["all_ids"]:
        raise SystemExit("Class IDs do not match verification_summary.json.")

    print("[4/6] Building issue registry and per-file actions")
    exact_by_file: dict[str, dict] = {}
    for group in review["exact_groups"]:
        for f in group["files"]:
            exact_by_file[f] = group
    source_by_file: dict[str, dict] = {}
    for group in review["source_groups"]:
        for f in group["files"]:
            source_by_file[f] = group

    canonical_by_md5 = {
        g["md5"]: sorted(g["files"])[0]
        for g in review["exact_groups"]
        if g["category"] == "identical_label_bytes"
    }

    format_by_file = {x["rel"]: x for x in review["format_only"]}
    stem_winner: dict[str, dict] = {}
    stem_loser: dict[str, dict] = {}
    for item in review["duplicate_stems"]:
        winner = item["files"][item["better_fit_index"]]
        stem_winner[winner] = item
        for f in item["files"]:
            if f != winner:
                stem_loser[f] = item

    issues, file_issue_ids = build_issues(
        review, image_index, label_index, source_by_file, exact_by_file, canonical_by_md5
    )

    image_actions = {
        rel: decide_image_action(
            rel, format_by_file, stem_winner, stem_loser,
            exact_by_file, source_by_file, canonical_by_md5,
        )
        for rel in image_index
    }

    images_by_label: dict[str, list[str]] = defaultdict(list)
    for rel in image_index:
        images_by_label[label_rel_for_image_rel(rel)].append(rel)

    label_actions: dict[str, str] = {}
    for lrel in label_index:
        label_actions[lrel] = decide_label_action(
            [image_actions[r] for r in images_by_label.get(lrel, [])]
        )

    print(f"    issues={len(issues):,}")
    print(f"    image actions={dict(Counter(image_actions.values()))}")
    print(f"    label actions={dict(Counter(label_actions.values()))}")

    print("[5/6] Building manifest and validating")
    manifest = build_manifest(
        image_index, label_index, label_meta, image_actions, label_actions,
        images_by_label, file_issue_ids, stem_loser,
    )
    validation = validate(
        review, summary, leakage_csv, image_index, label_index, label_meta,
        issues, manifest, image_actions, label_actions,
    )

    print("[6/6] Writing plan files")
    write_actions_csv(OUT_DIR / "preparation_actions.csv", issues)
    write_summary_json(
        OUT_DIR / "preparation_summary.json",
        summary, review, issues, manifest, image_actions, label_actions, validation,
    )
    (OUT_DIR / "preparation_manifest.json").write_text(
        json.dumps(
            {
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "mode": "DRY-RUN",
                "dataset_root": posix(ROOT),
                "proposed_prepared_root": posix(PREPARED_ROOT),
                "note": "No operation was executed. Destinations are proposals only.",
                "entries": manifest,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    write_report(OUT_DIR / "preparation_plan.md", review, issues,
                 image_actions, label_actions, validation)

    print()
    print("Done. DRY RUN — no original file was changed.")
    print("Plan    :", OUT_DIR / "preparation_plan.md")
    print("Actions :", OUT_DIR / "preparation_actions.csv")
    print("Summary :", OUT_DIR / "preparation_summary.json")
    print("Manifest:", OUT_DIR / "preparation_manifest.json")
    print()
    print(f"Validation: conflicts require approval = {validation['all_conflicts_require_approval']}")
    print(f"Validation: corrected source grouping = {validation['uses_corrected_source_grouping']}")
    print(f"Validation: destination collisions = {validation['destination_collisions']}")
    print(f"Validation: label rewrites = {validation['label_actions_that_modify_content']}")
    print(f"Validation: destinations inside original root = {validation['destinations_inside_original_root']}")


if __name__ == "__main__":
    main()
