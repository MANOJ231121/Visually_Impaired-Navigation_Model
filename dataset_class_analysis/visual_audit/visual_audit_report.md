# Visual Class Audit - Numeric Class IDs

- **Dataset:** `prepared_dataset`
- **Output:** `dataset_class_analysis/visual_audit`
- **Generated:** 2026-11-09T18:50:19
- **Mode:** read-only; no dataset, annotation, class ID or split changed

> This audit shows examples only. It does **not** assign class names and does **not** decide which classes are pedestrian obstacles. A human reviewer must interpret each numeric ID from the images.

## 1. Method

- Parsed **16,107** YOLO label files and **23,043** bounding boxes across 35 class IDs (0-34).
- Selected up to **20** examples per class, stratified by bounding-box size and spread across batches.
- **Size tiers** are per-class tertiles of the target box area (`w*h`, normalized): small <= Q1, medium <= Q2, large > Q2. Targets are 7 small / 7 medium / 6 large, refilled from other tiers when a tier is exhausted.
- **Batch coverage:** selection round-robins over Batch 1-4; a final pass adds or substitutes a sample if a batch is missing and an alternative exists.
- **Duplicate avoidance:** exact image duplicates are deduplicated by MD5; duplicate-stem images share one label, so only one image is used; images already used by another class are avoided when alternatives exist.
- **Drawing:** only boxes of the target class are drawn. Each box is labelled with the numeric ID. Header shows filename, batch, split, size tier and box area.
- Selection seed: `20261109` (deterministic for this dataset).

## 2. Examples reviewed per class

| Class ID | Examples | Unique candidates | Batches | Small / Medium / Large | Contact sheet |
|---|---|---|---|---|---|
| 0 | 20 | 717 | B1, B2, B3, B4 | 7 / 7 / 6 | [PNG](class_00_contact_sheet.png) |
| 1 | 20 | 583 | B1, B2, B3, B4 | 7 / 7 / 6 | [PNG](class_01_contact_sheet.png) |
| 2 | 20 | 486 | B1, B2, B3, B4 | 7 / 7 / 6 | [PNG](class_02_contact_sheet.png) |
| 3 | 20 | 519 | B1, B2, B3, B4 | 7 / 7 / 6 | [PNG](class_03_contact_sheet.png) |
| 4 | 20 | 522 | B1, B2, B3, B4 | 7 / 7 / 6 | [PNG](class_04_contact_sheet.png) |
| 5 | 20 | 464 | B1, B2, B3, B4 | 7 / 7 / 6 | [PNG](class_05_contact_sheet.png) |
| 6 | 20 | 534 | B1, B2, B3, B4 | 7 / 7 / 6 | [PNG](class_06_contact_sheet.png) |
| 7 | 20 | 459 | B1, B2, B3, B4 | 7 / 7 / 6 | [PNG](class_07_contact_sheet.png) |
| 8 | 20 | 553 | B1, B2, B3, B4 | 7 / 7 / 6 | [PNG](class_08_contact_sheet.png) |
| 9 | 20 | 590 | B1, B2, B3, B4 | 7 / 7 / 6 | [PNG](class_09_contact_sheet.png) |
| 10 | 20 | 456 | B1, B2, B3, B4 | 7 / 7 / 6 | [PNG](class_10_contact_sheet.png) |
| 11 | 20 | 556 | B1, B2, B3, B4 | 7 / 7 / 6 | [PNG](class_11_contact_sheet.png) |
| 12 | 20 | 528 | B1, B2, B3, B4 | 7 / 7 / 6 | [PNG](class_12_contact_sheet.png) |
| 13 | 20 | 481 | B1, B2, B3, B4 | 7 / 7 / 6 | [PNG](class_13_contact_sheet.png) |
| 14 | 20 | 469 | B1, B2, B3, B4 | 7 / 7 / 6 | [PNG](class_14_contact_sheet.png) |
| 15 | 20 | 485 | B1, B2, B3, B4 | 7 / 7 / 6 | [PNG](class_15_contact_sheet.png) |
| 16 | 20 | 602 | B1, B2, B3, B4 | 7 / 7 / 6 | [PNG](class_16_contact_sheet.png) |
| 17 | 20 | 588 | B1, B2, B3, B4 | 7 / 7 / 6 | [PNG](class_17_contact_sheet.png) |
| 18 | 20 | 545 | B1, B2, B3, B4 | 7 / 7 / 6 | [PNG](class_18_contact_sheet.png) |
| 19 | 20 | 647 | B1, B2, B3, B4 | 7 / 7 / 6 | [PNG](class_19_contact_sheet.png) |
| 20 | 20 | 437 | B1, B2, B3, B4 | 7 / 7 / 6 | [PNG](class_20_contact_sheet.png) |
| 21 | 20 | 540 | B1, B2, B3, B4 | 7 / 7 / 6 | [PNG](class_21_contact_sheet.png) |
| 22 | 20 | 582 | B1, B2, B3, B4 | 7 / 7 / 6 | [PNG](class_22_contact_sheet.png) |
| 23 | 20 | 568 | B1, B2, B3, B4 | 7 / 7 / 6 | [PNG](class_23_contact_sheet.png) |
| 24 | 20 | 565 | B1, B2, B3, B4 | 7 / 7 / 6 | [PNG](class_24_contact_sheet.png) |
| 25 | 20 | 503 | B1, B2, B3, B4 | 7 / 7 / 6 | [PNG](class_25_contact_sheet.png) |
| 26 | 20 | 532 | B1, B2, B3, B4 | 7 / 7 / 6 | [PNG](class_26_contact_sheet.png) |
| 27 | 20 | 559 | B1, B2, B3, B4 | 7 / 7 / 6 | [PNG](class_27_contact_sheet.png) |
| 28 | 20 | 522 | B1, B2, B3, B4 | 7 / 7 / 6 | [PNG](class_28_contact_sheet.png) |
| 29 | 20 | 559 | B1, B2, B3, B4 | 7 / 7 / 6 | [PNG](class_29_contact_sheet.png) |
| 30 | 20 | 420 | B1, B2, B3, B4 | 7 / 7 / 6 | [PNG](class_30_contact_sheet.png) |
| 31 | 20 | 513 | B1, B2, B3, B4 | 7 / 7 / 6 | [PNG](class_31_contact_sheet.png) |
| 32 | 20 | 410 | B1, B2, B3, B4 | 7 / 7 / 6 | [PNG](class_32_contact_sheet.png) |
| 33 | 20 | 564 | B1, B2, B3, B4 | 7 / 7 / 6 | [PNG](class_33_contact_sheet.png) |
| 34 | 20 | 606 | B1, B2, B3, B4 | 7 / 7 / 6 | [PNG](class_34_contact_sheet.png) |

Total examples selected: **700**. Unique source images used: **700** (some images legitimately appear in more than one class sheet: 0 repeated selections).

## 3. Missing or insufficient examples

**None.** Every class ID 0-34 has 20 selected examples, with all available batches represented.

Classes with no candidates at all: none.

### Low-content selected images

These selected source images are mostly black or letterboxed. They are valid source images, not rendering failures: the contact sheet reproduces them faithfully. They can look almost empty at thumbnail size, so open the source path to inspect them.

| Class ID | Sample | Image | Size | Non-black pixels | Batch / split |
|---|---|---|---|---|---|
| 33 | 6 | `Batch 4/images/train/791881_417.jpg` | 450x112 | 9.34% | Batch 4 / train |

## 4. Classes that appear visually similar

The table below compares the **annotated object crops** using a difference hash (dHash) of the largest target box per selected image. A low mean minimum Hamming distance means the crops look visually similar at a coarse level.

**Important:** this is image-level visual similarity only. It does not prove that two class IDs mean the same thing, and it must not be used to merge or rename classes. It is a triage aid for human review.

| Class i | Class j | Mean min dHash distance | % crop pairs <= 10 | Crops compared |
|---|---|---|---|---|
| 31 | 33 | 15.55 | 0.5 | 400 |
| 20 | 29 | 18.19 | 0.53 | 380 |
| 8 | 32 | 19.25 | 0.5 | 400 |
| 8 | 20 | 19.65 | 0.26 | 380 |
| 6 | 30 | 19.65 | 0.0 | 400 |
| 20 | 32 | 19.89 | 0.26 | 380 |
| 16 | 20 | 19.99 | 0.0 | 380 |
| 30 | 33 | 20.12 | 0.0 | 400 |
| 23 | 29 | 20.2 | 0.0 | 400 |
| 16 | 29 | 20.3 | 0.25 | 400 |
| 5 | 20 | 20.43 | 0.0 | 380 |
| 2 | 3 | 20.48 | 0.25 | 400 |

Pairs with the lowest distance should be reviewed side by side in the two contact sheets before drawing any conclusion.

## 5. Annotation inconsistencies noticed

| Heuristic | Count | Threshold |
|---|---|---|
| Malformed label lines | 0 | not 5 fields / non-numeric |
| Invalid geometry (w/h outside (0,1] or center outside [0,1]) | 0 | YOLO cx,cy,w,h |
| Boxes whose center-based extent exceeds image bounds | 0 | cx+/-w/2, cy+/-h/2 |
| Tiny boxes | 3 | area < 0.0001 |
| Extreme aspect-ratio boxes | 166 | ratio > 10.0 or < 0.1 |
| Same-class near-duplicate box pairs | 3 | IoU >= 0.9 |
| Cross-class overlapping box pairs | 0 | IoU >= 0.8 |

Examples (first few per heuristic):

**extreme_aspect_boxes**

- `Batch 1/labels/test/hjmdmqxm.txt` - {"line": 4, "class_id": 2, "aspect": 12.5}
- `Batch 1/labels/train/1a6365c6_.txt` - {"line": 1, "class_id": 2, "aspect": 12.91}
- `Batch 1/labels/train/1a6365c6_.txt` - {"line": 2, "class_id": 2, "aspect": 14.2}
- `Batch 1/labels/train/27f695a9_.txt` - {"line": 1, "class_id": 2, "aspect": 12.14}
- `Batch 1/labels/train/2b187c4d_.txt` - {"line": 1, "class_id": 2, "aspect": 10.2}
- `Batch 1/labels/train/37eb3a21_.txt` - {"line": 1, "class_id": 28, "aspect": 19.85}
- `Batch 1/labels/train/3bfd92fb-01fe-44de-8bfd-57766e235575_bestick-hardanger-header.txt` - {"line": 1, "class_id": 3, "aspect": 11.21}
- `Batch 1/labels/train/3e56521e_.txt` - {"line": 1, "class_id": 2, "aspect": 11.88}

**same_class_duplicate_pairs**

- `Batch 1/labels/train/e7d5c42f_.txt` - {"lines": [1, 3], "class_id": 26, "iou": 0.9344}
- `Batch 2/labels/train/137_png_jpg.rf.302fd3a5e6e0e963eb9b22444da910d7.txt` - {"lines": [1, 2], "class_id": 1, "iou": 0.9456}
- `Batch 2/labels/train/35.txt` - {"lines": [1, 2], "class_id": 26, "iou": 0.9106}

**tiny_boxes**

- `Batch 2/labels/train/98_png_jpg.rf.b8f0458a3d4bb6cb065a757fd2c56787.txt` - {"line": 2, "class_id": 1, "area": 4.394e-05}
- `Batch 3/labels/val/nzuuphjz.txt` - {"line": 3, "class_id": 25, "area": 1.23e-06}
- `Batch 4/labels/val/98_png_jpg.rf.6e27c111b1463fac0dcfca905eda6908.txt` - {"line": 2, "class_id": 1, "area": 4.394e-05}

These checks are structural only. They cannot detect a semantically wrong label, a missed object, or two IDs used for the same concept. Those require human inspection of the contact sheets.

Also relevant: **19** label files are shared by multiple image files (duplicate stems / PNG copies). The audit selects only one image per label to avoid duplicate examples.

### File extension vs actual encoding

Some image files are named with an extension that does not match their actual encoding. This audit reads images by content, so it is unaffected, but tools that trust file extensions may misread these files. The files were not renamed.

| Extension | Actual encoding | Count | Examples |
|---|---|---|---|
| .jpg | png | 298 | `Batch 1/images/test/1jmfmh5w.jpg`, `Batch 1/images/test/fcccirkm.jpg`, `Batch 1/images/test/qorcehkv.jpg` |
| .jpg | webp | 13 | `Batch 1/images/train/ba5bc69f_.jpg`, `Batch 1/images/train/db2eab54_.jpg`, `Batch 1/images/train/f2b3bbf9_.jpg` |
| .png | jpg | 9 | `Batch 1/images/train/798859_452.png`, `Batch 1/images/train/798859_493.png`, `Batch 2/images/train/7911_353.png` |
| .jpg | bmp | 4 | `Batch 1/images/train/paxmax_folding_walking_stick_for_impaired_vision_or_blind_people_395330_0_0.jpg`, `Batch 1/images/train/paxmax_folding_walking_stick_for_impaired_vision_or_blind_people_395335_5_0.jpg`, `Batch 4/images/train/paxmax_folding_walking_stick_for_impaired_vision_or_blind_people_395330_0_0.jpg` |
| .webp | jpg | 3 | `Batch 3/images/train/15_9f2628f7-0a6e-40db-a06a-5aa1cc95dee2.webp`, `Batch 3/images/train/A9674866481b64aa5828c862fd0bdf314T.jpg_640x640.jpg.webp`, `Batch 3/images/train/DSC_3857copy.webp` |

## 6. Read-only verification

- Dataset files fingerprinted before and after: **32,233**
- Files modified: **0**
- Dataset unchanged: **True**

## 7. Obstacle-detection note

No class is treated as a pedestrian obstacle in this audit. Whether a class is relevant to assistive navigation depends on what the human reviewer identifies in these images, not on its frequency or appearance alone. A class that looks like an obstacle can be irrelevant in context, and a visually ordinary class can matter.

## 8. Outputs

| File | Contents |
|---|---|
| `class_00_contact_sheet.png` ... `class_34_contact_sheet.png` | One 20-example contact sheet per class |
| `class_index.png` | Combined visual index, one example per class |
| `class_index.html` / `class_index.md` | Clickable class index |
| `class_visual_inventory.csv` | 700 selected examples |
| `visual_audit_report.md` | This report |

## 9. Exact command

```powershell
python "visual_class_audit.py"
```

## 10. Not done (by rule)

- No dataset image or label modified, moved, renamed or deleted.
- No class name inferred or assigned.
- No class treated as an obstacle.
- No class ID, annotation, coordinate or split changed.
- No model trained.
