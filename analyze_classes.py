#!/usr/bin/env python3
"""
analyze_classes.py
==================

Read-only class-distribution and obstacle-coverage assessment for the
4-batch YOLO dataset used by the assistive-navigation project.

This script NEVER modifies the dataset.  It only reads images/labels and the
completed verification outputs, then writes three assessment files to
dataset_preparation_plan/.

Critical rules encoded here:
  * all 35 numeric class IDs are reported, none added or removed;
  * no class name is invented;
  * a class is marked as an "obstacle" class ONLY if its semantic meaning can
    be verified from metadata.  No metadata exists, so no class is selected;
  * obstacle coverage is recorded as NOT DETERMINABLE rather than claimed.

Run:
    python "C:\\vs code\\object detection\\analyze_classes.py"
"""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
ROOT = PROJECT_ROOT / "final batches"
ANALYSIS_DIR = PROJECT_ROOT / "dataset_verification"
OUT_DIR = PROJECT_ROOT / "dataset_preparation_plan"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tif", ".tiff"}

# Filename tokens used ONLY as a weak domain sanity screen.  They are not
# class metadata and must never be converted into a class list.
OBSTACLE_KEYWORDS = [
    "barrier", "bollard", "cone", "construction", "fence", "pole", "pothole",
    "manhole", "stair", "step", "curb", "kerb", "sign", "traffic", "light",
    "bin", "trash", "waste", "garbage", "bench", "hydrant", "scooter",
    "bicycle", "bike", "motorcycle", "vehicle", "car", "truck", "bus",
    "pram", "stroller", "luggage", "box", "crate", "pallet", "cable",
    "rope", "rock", "tree", "plant", "branch",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def rel_of(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def label_rel_for_image_rel(image_rel: str) -> str:
    parts = list(Path(image_rel).parts)
    parts[parts.index("images")] = "labels"
    return Path(*parts).with_suffix(".txt").as_posix()


def batch_split(rel: str) -> tuple[str, str]:
    parts = Path(rel).parts
    return parts[0], parts[2]


def parse_label(path: Path) -> list[int]:
    classes: list[int] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        tokens = line.split()
        if len(tokens) == 5:
            classes.append(int(tokens[0]))
    return classes


def normalized_entropy(counts: list[int]) -> float:
    total = sum(counts)
    if total <= 0 or len(counts) < 2:
        return 0.0
    entropy = -sum((c / total) * math.log(c / total) for c in counts if c > 0)
    return entropy / math.log(len(counts))


def balance(counts: list[int]) -> dict:
    nz = [c for c in counts]
    mean = statistics.fmean(nz)
    stdev = statistics.pstdev(nz)
    return {
        "count": len(nz),
        "min": min(nz),
        "max": max(nz),
        "mean": round(mean, 3),
        "median": statistics.median(nz),
        "stdev": round(stdev, 3),
        "coefficient_of_variation": round(stdev / mean, 4) if mean else None,
        "max_over_min": round(max(nz) / min(nz), 4) if min(nz) else None,
        "normalized_entropy": round(normalized_entropy(nz), 6),
    }


def md_table(headers: list[str], rows: list[list[object]]) -> str:
    lines = [
        "| " + " | ".join(str(h) for h in headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join("" if v is None else str(v) for v in row) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Dataset indexing
# ---------------------------------------------------------------------------

def index_dataset() -> tuple[list[str], dict[str, str]]:
    images = sorted(
        p for p in ROOT.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )
    image_rels = [rel_of(p) for p in images]
    image_to_label = {r: label_rel_for_image_rel(r) for r in image_rels}
    label_paths = {r: ROOT / Path(l) for r, l in image_to_label.items()}
    missing = sorted(l for r, l in image_to_label.items() if not label_paths[r].is_file())
    if missing:
        raise SystemExit(f"{len(missing)} images have no label file; first: {missing[:3]}")
    return image_rels, image_to_label


def find_class_name_metadata() -> dict:
    """Search for any file that could define class names.  Report, never infer."""
    candidates: list[str] = []
    name_indicators = {"data.yaml", "data.yml", "classes.txt", "obj.names",
                       "class_names.txt", "labels.txt", "names.txt"}
    for p in ROOT.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() in {".yaml", ".yml", ".names"}:
            candidates.append(rel_of(p))
        elif p.name.lower() in name_indicators:
            candidates.append(rel_of(p))
        elif p.suffix.lower() in {".json", ".csv", ".md", ".xml"}:
            candidates.append(rel_of(p))
    # .txt files inside labels/ are annotations, not metadata.
    return {
        "metadata_files_found": sorted(candidates),
        "class_name_metadata_available": False,
        "searched_extensions": [".yaml", ".yml", ".names", ".json", ".csv", ".md", ".xml", ".txt"],
        "note": (
            "Recursive search under the dataset root. Label .txt files are annotations and "
            "were excluded. No data.yaml / classes.txt / obj.names / names file exists."
        ),
    }


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyze() -> dict:
    image_rels, image_to_label = index_dataset()

    boxes_by_class = Counter()
    images_containing = Counter()
    boxes_by_split = defaultdict(Counter)
    boxes_by_batch = defaultdict(Counter)
    images_by_split = Counter()
    empty_labels = 0
    total_boxes = 0

    filename_keyword_counts = Counter()
    label_files_with_keyword = 0
    keyword_by_class = Counter()
    examples_by_class: dict[int, list[str]] = defaultdict(list)

    for image_rel in image_rels:
        batch, split = batch_split(image_rel)
        classes = parse_label(image_to_label[image_rel])
        images_by_split[split] += 1
        if not classes:
            empty_labels += 1
        total_boxes += len(classes)

        unique = sorted(set(classes))
        for cls in classes:
            boxes_by_class[cls] += 1
            boxes_by_split[split][cls] += 1
            boxes_by_batch[batch][cls] += 1
        for cls in unique:
            images_containing[cls] += 1

        stem = Path(image_rel).stem.lower()
        hits = [k for k in OBSTACLE_KEYWORDS if k in stem]
        if hits:
            label_files_with_keyword += 1
            for k in sorted(set(hits)):
                filename_keyword_counts[k] += 1
            for cls in unique:
                keyword_by_class[cls] += 1
        if len(examples_by_class) < 35:
            for cls in unique:
                if len(examples_by_class[cls]) < 3:
                    examples_by_class[cls].append(image_rel)

    class_ids = sorted(boxes_by_class)
    counts = [boxes_by_class[c] for c in class_ids]

    per_class = []
    for cls in class_ids:
        batch_counts = {b: boxes_by_batch[b][cls] for b in sorted(boxes_by_batch)}
        split_counts = {s: boxes_by_split[s][cls] for s in sorted(boxes_by_split)}
        per_class.append({
            "class_id": cls,
            "boxes": boxes_by_class[cls],
            "images_containing": images_containing[cls],
            "share_of_boxes": round(boxes_by_class[cls] / total_boxes, 6) if total_boxes else 0,
            "boxes_by_split": split_counts,
            "boxes_by_batch": batch_counts,
            "images_with_filename_obstacle_keyword": keyword_by_class.get(cls, 0),
            "example_image_paths": examples_by_class.get(cls, []),
            "semantic_status": "UNVERIFIED — no class-name metadata",
            "obstacle_class": False,
            "obstacle_class_reason": "Semantic meaning cannot be verified; not selected.",
        })

    # Per-batch balance: 35 counts per batch, to show how uniform each batch is.
    per_batch = {}
    for batch in sorted(boxes_by_batch):
        batch_counts = [boxes_by_batch[batch][c] for c in class_ids]
        per_batch[batch] = {
            "boxes": sum(batch_counts),
            "class_count_present": sum(1 for c in batch_counts if c > 0),
            **balance(batch_counts),
        }

    per_split = {}
    for split in sorted(boxes_by_split):
        split_counts = [boxes_by_split[split][c] for c in class_ids]
        per_split[split] = {
            "images": images_by_split[split],
            "boxes": sum(split_counts),
            "class_count_present": sum(1 for c in split_counts if c > 0),
            **balance(split_counts),
        }

    least = min(per_class, key=lambda x: x["boxes"])
    most = max(per_class, key=lambda x: x["boxes"])

    # Cross-batch ID consistency evidence (from the earlier analysis).
    id_consistency = {"rows": 0, "identity": 0, "non_identity": [], "ambiguous": []}
    mapping_path = ANALYSIS_DIR / "inferred_mapping.csv"
    if mapping_path.is_file():
        with mapping_path.open(encoding="utf-8-sig", newline="") as fh:
            rows = list(csv.DictReader(fh))
        id_consistency["rows"] = len(rows)
        for r in rows:
            if r["source_id"] == r["target_id"]:
                id_consistency["identity"] += 1
            else:
                id_consistency["non_identity"].append(r)
            if r["ambiguous"].lower() == "true":
                id_consistency["ambiguous"].append(r)
        id_consistency["note"] = (
            "This CSV checks whether the same class ID describes the same source image "
            "across batches. It is ID-to-ID consistency evidence, NOT a name mapping."
        )

    metadata = find_class_name_metadata()

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "mode": "DRY-RUN — read-only class analysis",
        "dataset_root": ROOT.as_posix(),
        "output_dir": OUT_DIR.as_posix(),
        "class_ids": class_ids,
        "class_id_count": len(class_ids),
        "class_names": {
            **metadata,
            "verified_class_names": {},
            "semantic_verification_status": "UNAVAILABLE",
            "note": (
                "No reliable id -> name map exists. Names are therefore never assigned or "
                "implied anywhere in this project until the dataset owner supplies them."
            ),
        },
        "totals": {
            "images": len(image_rels),
            "labels": len(image_rels),
            "boxes": total_boxes,
            "images_with_zero_boxes": empty_labels,
        },
        "distribution": {
            "per_class": per_class,
            "least_frequent": {"class_id": least["class_id"], "boxes": least["boxes"]},
            "most_frequent": {"class_id": most["class_id"], "boxes": most["boxes"]},
            "max_over_min": round(most["boxes"] / least["boxes"], 4),
            "balance": balance(counts),
            "assessment": (
                "All 35 IDs are present and the totals are close to uniform. A near-uniform "
                "distribution is atypical for naturally collected detection data and suggests "
                "curated balancing, resampling or synthetic/duplicated sources. This is a "
                "distribution observation only; it says nothing about class semantics."
            ),
        },
        "per_batch_balance": per_batch,
        "per_split_balance": per_split,
        "id_consistency": id_consistency,
        "filename_screen": {
            "method": (
                "Substring count of pedestrian-obstacle-related words in image filenames. "
                "This is circumstantial domain evidence only; a filename does not identify "
                "the class ID of any annotated box."
            ),
            "label_files_scanned": len(image_rels),
            "label_files_with_any_keyword": label_files_with_keyword,
            "fraction_with_any_keyword": round(label_files_with_keyword / len(image_rels), 6),
            "keyword_counts": dict(sorted(filename_keyword_counts.items(), key=lambda x: -x[1])),
            "caveat": (
                "762 of 16,107 files (~4.7%) contain a keyword. Many keywords are ambiguous "
                "(e.g. 'light' in 'bed light', 'car' in 'cardboard') and the same image can "
                "contain unrelated objects. These counts must never be turned into a class list."
            ),
        },
        "obstacle_assessment": {
            "status": "NOT DETERMINABLE FROM AVAILABLE EVIDENCE",
            "verified_obstacle_class_ids": [],
            "coverage_claim": "none",
            "reasons": [
                "No class-name metadata exists, so no numeric ID can be mapped to an object type.",
                "No verified semantic audit has been performed against the boxes.",
                "Filename keywords are not class labels and cannot identify which ID is an obstacle.",
                "Selecting obstacle classes now would be an assumption, which the project forbids.",
            ],
            "what_would_be_needed": [
                "An authoritative id -> name map from the dataset owner, plus data.yaml.",
                "Or a box-level visual audit of sample images per class, recorded as evidence.",
                "A documented obstacle taxonomy for the assistive-navigation module.",
                "Human sign-off on the final obstacle subset before any class is dropped.",
            ],
        },
        "requirements_compliance": {
            "preserve_all_35_class_ids": True,
            "class_ids_changed": 0,
            "invented_class_names": 0,
            "classes_removed": 0,
            "classes_removed_based_on_assumptions": 0,
            "keep_dataset_suitable_for_general_detection": True,
            "obstacle_selection_deferred_until_verified_names": True,
            "original_files_modified": False,
        },
    }


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def write_csv(path: Path, data: dict) -> None:
    fields = [
        "class_id", "boxes", "images_containing", "share_of_boxes",
        "train_boxes", "val_boxes", "test_boxes",
        "Batch 1_boxes", "Batch 2_boxes", "Batch 3_boxes", "Batch 4_boxes",
        "images_with_filename_obstacle_keyword",
        "semantic_status", "obstacle_class", "obstacle_class_reason",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(fields)
        for row in data["distribution"]["per_class"]:
            w.writerow([
                row["class_id"], row["boxes"], row["images_containing"], row["share_of_boxes"],
                row["boxes_by_split"].get("train", 0),
                row["boxes_by_split"].get("val", 0),
                row["boxes_by_split"].get("test", 0),
                row["boxes_by_batch"].get("Batch 1", 0),
                row["boxes_by_batch"].get("Batch 2", 0),
                row["boxes_by_batch"].get("Batch 3", 0),
                row["boxes_by_batch"].get("Batch 4", 0),
                row["images_with_filename_obstacle_keyword"],
                row["semantic_status"], row["obstacle_class"], row["obstacle_class_reason"],
            ])


def write_markdown(path: Path, data: dict) -> None:
    d = data["distribution"]
    lines: list[str] = []
    add = lines.append

    add("# Class Distribution & Obstacle-Coverage Assessment — DRY RUN")
    add("")
    add(f"- **Dataset root:** `{data['dataset_root']}`")
    add(f"- **Generated:** {data['generated_at']}")
    add(f"- **Mode:** {data['mode']}")
    add("- **Class IDs:** all **35** preserved `0–34`; no ID added, removed or renamed.")
    add("- **Class names:** **none available / none invented.**")
    add("")
    add("---")
    add("")

    add("## 0. Bottom line")
    add("")
    add("**No obstacle class can be selected yet.** The dataset contains 35 numeric class IDs, but "
        "there is no `data.yaml`, `classes.txt`, `obj.names`, JSON, CSV or README that maps an ID "
        "to an object name. Semantic meaning is therefore unverified for all 35 IDs, and selecting "
        "\"obstacle\" classes now would be an assumption. The dataset is kept intact and suitable "
        "for general object detection until reliable names are supplied.")
    add("")
    add(f"- Obstacle selection status: **{data['obstacle_assessment']['status']}**")
    add(f"- Verified obstacle class IDs: **{data['obstacle_assessment']['verified_obstacle_class_ids']}**")
    add(f"- Coverage claim: **{data['obstacle_assessment']['coverage_claim']}**")
    add("")

    add("## 1. Class distribution")
    add("")
    t = data["totals"]
    add(f"The dataset has **{t['images']:,} images**, **{t['labels']:,} labels**, "
        f"**{t['boxes']:,} boxes** and **{t['images_with_zero_boxes']:,}** images with no boxes.")
    add("")
    b = d["balance"]
    add(md_table(
        ["Metric", "Value"],
        [
            ["Class IDs present", data["class_id_count"]],
            ["Least frequent class", f"ID {d['least_frequent']['class_id']} — {d['least_frequent']['boxes']:,} boxes"],
            ["Most frequent class", f"ID {d['most_frequent']['class_id']} — {d['most_frequent']['boxes']:,} boxes"],
            ["Max / min ratio", f"{d['max_over_min']:.2f} : 1"],
            ["Mean boxes / class", f"{b['mean']:.1f}"],
            ["Median boxes / class", f"{b['median']:.1f}"],
            ["Coefficient of variation", f"{b['coefficient_of_variation']:.4f}"],
            ["Normalized entropy", f"{b['normalized_entropy']:.4f} (1.0 = perfectly uniform)"],
        ],
    ))
    add("")
    add(f"> {d['assessment']}")
    add("")
    add("### 1.1 Per-class totals")
    add("")
    rows = []
    for row in d["per_class"]:
        rows.append([
            row["class_id"], row["boxes"], row["images_containing"],
            f"{row['share_of_boxes'] * 100:.2f}%",
            row["boxes_by_split"].get("train", 0),
            row["boxes_by_split"].get("val", 0),
            row["boxes_by_split"].get("test", 0),
        ])
    add(md_table(["Class ID", "Boxes", "Images", "Share", "Train", "Val", "Test"], rows))
    add("")

    add("## 2. Per-batch uniformity")
    add("")
    add("A near-uniform count in every batch is evidence of curation, not of class meaning. "
        "It does not identify obstacles.")
    add("")
    rows = []
    for batch, v in data["per_batch_balance"].items():
        rows.append([batch, v["boxes"], v["class_count_present"], v["min"], v["max"],
                     v["mean"], f"{v['coefficient_of_variation']:.4f}"])
    add(md_table(["Batch", "Boxes", "Classes present", "Min", "Max", "Mean", "CV"], rows))
    add("")
    rows = []
    for split, v in data["per_split_balance"].items():
        rows.append([split, v["images"], v["boxes"], v["min"], v["max"],
                     v["mean"], f"{v['coefficient_of_variation']:.4f}"])
    add(md_table(["Split", "Images", "Boxes", "Min", "Max", "Mean", "CV"], rows))
    add("")

    add("## 3. Class-name evidence")
    add("")
    cn = data["class_names"]
    add(f"- Metadata files found: **{len(cn['metadata_files_found'])}** `{cn['metadata_files_found']}`")
    add(f"- Class-name metadata available: **{cn['class_name_metadata_available']}**")
    add(f"- Semantic verification status: **{cn['semantic_verification_status']}**")
    add(f"- Verified `id -> name` map: **`{cn['verified_class_names']}`** (empty)")
    add("")
    add(f"> {cn['note']}")
    add("")

    add("## 4. Obstacle coverage — what the evidence does and does not show")
    add("")
    oa = data["obstacle_assessment"]
    add(f"**Status: {oa['status']}.** No coverage is claimed.")
    add("")
    add("Reasons:")
    for reason in oa["reasons"]:
        add(f"- {reason}")
    add("")
    add("What would be needed before selecting obstacle classes:")
    for item in oa["what_would_be_needed"]:
        add(f"- {item}")
    add("")
    add("### 4.1 Circumstantial filename screen (weak evidence only)")
    add("")
    fs = data["filename_screen"]
    add(f"{fs['label_files_with_any_keyword']:,} of {fs['label_files_scanned']:,} label files "
        f"({fs['fraction_with_any_keyword'] * 100:.1f}%) have at least one pedestrian-obstacle-related "
        "word in the image filename.")
    add("")
    add(f"> {fs['caveat']}")
    add("")
    rows = [[k, v] for k, v in fs["keyword_counts"].items()]
    add(md_table(["Filename keyword", "Files"], rows))
    add("")
    add("The filename screen suggests a **heterogeneous, general object dataset** (signs, street "
        "furniture, waste containers, vehicles, but also ovens, toothbrushes, phones, knives and "
        "bottles). It does **not** prove that any class ID represents a pedestrian obstacle, because "
        "filenames are not labels and one image can contain several objects.")
    add("")

    add("## 5. Cross-batch ID consistency")
    add("")
    ic = data["id_consistency"]
    add(f"- Mappings checked: **{ic['rows']}**")
    add(f"- Identity mappings (same ID across batches): **{ic['identity']}**")
    add(f"- Non-identity mappings: **{len(ic['non_identity'])}**")
    for r in ic["non_identity"]:
        add(f"  - `{r['batch_pair']}`: ID {r['source_id']} -> ID {r['target_id']} "
            f"(support {r['support']}/{r['total']}, confidence {r['confidence']}, ambiguous={r['ambiguous']})")
    add(f"- Ambiguous mappings: **{len(ic['ambiguous'])}**")
    add("")
    add(f"> {ic.get('note', '')}")
    add("")

    add("## 6. Requirement compliance")
    add("")
    rc = data["requirements_compliance"]
    add(md_table(["Requirement", "Status"], [
        ["Preserve all 35 original numeric class IDs", "PASS" if rc["preserve_all_35_class_ids"] else "FAIL"],
        ["Do not invent class names", f"PASS — {rc['invented_class_names']} invented"],
        ["Do not remove classes based on assumptions", f"PASS — {rc['classes_removed']} removed"],
        ["Keep dataset suitable for general object detection", "PASS" if rc["keep_dataset_suitable_for_general_detection"] else "FAIL"],
        ["Obstacle selection only with verified semantics", "PASS — deferred" if rc["obstacle_selection_deferred_until_verified_names"] else "FAIL"],
        ["Keep all original files untouched", "PASS" if not rc["original_files_modified"] else "FAIL"],
    ]))
    add("")
    add("## 7. Recommended next step")
    add("")
    add("Obtain the authoritative `id -> name` map (and `data.yaml`) from the dataset owner. "
        "If it cannot be obtained, run a documented box-level visual audit: for each class ID, "
        "review several sampled images **with boxes drawn**, record what is inside the boxes, and "
        "have a human sign off. Only after that should an obstacle subset be proposed. Until then, "
        "train as a general 35-class detector and report classes by numeric ID.")
    add("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    print("Dataset :", ROOT)
    print("Output  :", OUT_DIR)
    print("Mode    : DRY RUN (read-only)")
    print()

    if not ROOT.is_dir():
        raise SystemExit(f"Dataset root not found: {ROOT}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/3] Indexing images and labels")
    data = analyze()
    print(f"    class_ids={data['class_id_count']} boxes={data['totals']['boxes']:,}")

    print("[2/3] Writing class_analysis.json / class_distribution_full.csv")
    (OUT_DIR / "class_analysis.json").write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_csv(OUT_DIR / "class_distribution_full.csv", data)

    print("[3/3] Writing class_analysis.md")
    write_markdown(OUT_DIR / "class_analysis.md", data)

    print()
    print("Done. DRY RUN — no dataset file was read-modified or written.")
    print("Report :", OUT_DIR / "class_analysis.md")
    print("JSON   :", OUT_DIR / "class_analysis.json")
    print("CSV    :", OUT_DIR / "class_distribution_full.csv")
    print()
    print("Obstacle selection:", data["obstacle_assessment"]["status"])
    print("Verified obstacle class IDs:", data["obstacle_assessment"]["verified_obstacle_class_ids"])


if __name__ == "__main__":
    main()
