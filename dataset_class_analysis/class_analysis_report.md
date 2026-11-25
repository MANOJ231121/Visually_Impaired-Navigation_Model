# Class Distribution Analysis - Raw Numeric Class IDs

- **Dataset:** `prepared_dataset`
- **Generated:** 2026-11-07T22:53:03
- **Mode:** read-only analysis; no dataset modification, no training

> Class IDs are reported strictly as numeric IDs (0-34). No class names are invented, inferred, or assumed. Any statement about semantic meaning would require an authoritative class-name mapping, which this dataset does not contain.

## 1. Method and definitions

- Every `*.txt` YOLO label file in the prepared dataset was parsed.
- **Boxes**: one row per valid YOLO annotation line (5 fields).
- **Annotation files**: label files containing at least one box of the class.
- **Images**: image files whose stem matches a label file in the same batch and split. Duplicate-stem pairs (`.jpeg` + `.png`) and the PNG copies of the original WEBP/GIF files share one label, so an image count can exceed the annotation-file count. This is reported, not corrected.
- **Split** comes from the existing directory name (`train`, `val`, `test`); assignments were not changed.

## 2. Dataset totals

| Metric | Value |
|---|---|
| Images | 16,126 |
| Label files | 16,107 |
| Bounding boxes | 23,043 |
| Class IDs present | 35 (0-34) |
| All expected IDs 0-34 present | True |
| Mean boxes per label file | 1.431 |
| Smallest class (boxes) | 525 |
| Largest class (boxes) | 943 |
| Largest / smallest ratio | 1.796 |

## 3. Distribution by split

| Split | Images | Label files | Boxes | Classes present | Min class boxes | Max class boxes | Empty labels |
|---|---|---|---|---|---|---|---|
| train | 12,297 | 12,280 | 17,327 | 35 | 425 | 696 | 0 |
| val | 3,034 | 3,032 | 4,496 | 35 | 78 | 189 | 0 |
| test | 795 | 795 | 1,220 | 35 | 12 | 74 | 0 |

**Caution:** the test split contains only 1,220 boxes across 35 classes (12-74 boxes per class). Per-class test metrics will be noisy. The split was not changed.

## 4. Classes with very few training examples

Primary metric: **boxes in the train split**. Absolute rule: `train_boxes < 100`. Relative train-box quartiles: Q1 = 473.5, median = 491.0, Q3 = 507.5.

**No class has fewer than 100 training boxes.** By this absolute criterion, no class is severely under-represented in the train split.

Classes in the bottom train-box quartile (low tier, `train_boxes <= 473.5`):

| Class ID | Train boxes | Train images | Total boxes |
|---|---|---|---|
| 5 | 465 | 332 | 678 |
| 12 | 461 | 410 | 601 |
| 14 | 458 | 360 | 598 |
| 15 | 468 | 381 | 587 |
| 18 | 473 | 447 | 576 |
| 22 | 441 | 427 | 603 |
| 27 | 459 | 424 | 603 |
| 30 | 425 | 338 | 525 |
| 31 | 450 | 428 | 546 |

Train images per class range from 309 to 531; no class has fewer than 100 train boxes.

## 5. Classes with high representation

Rule: `train_boxes >= Q3 = 507.5`.

| Class ID | Train boxes | Train images | Total boxes | Share of all boxes |
|---|---|---|---|---|
| 0 | 696 | 531 | 943 | 4.092% |
| 1 | 544 | 431 | 748 | 3.246% |
| 4 | 522 | 405 | 695 | 3.016% |
| 6 | 509 | 425 | 643 | 2.79% |
| 10 | 515 | 351 | 658 | 2.856% |
| 17 | 514 | 434 | 721 | 3.129% |
| 19 | 525 | 476 | 720 | 3.125% |
| 20 | 530 | 315 | 730 | 3.168% |
| 24 | 542 | 411 | 751 | 3.259% |

The overall distribution is close to uniform: the largest class has 943 boxes and the smallest 525 (ratio 1.796). High/low tiers here are relative quartile labels, not evidence of severe imbalance.

## 6. Full per-class table

| ID | Total boxes | Train | Val | Test | Images | Label files | % of boxes | Train tier | Rank (total) |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 943 | 696 | 189 | 58 | 717 | 717 | 4.092% | high | 1 |
| 1 | 748 | 544 | 164 | 40 | 583 | 583 | 3.246% | high | 3 |
| 2 | 726 | 502 | 170 | 54 | 486 | 486 | 3.151% | typical | 5 |
| 3 | 675 | 496 | 146 | 33 | 519 | 519 | 2.929% | typical | 14 |
| 4 | 695 | 522 | 140 | 33 | 527 | 522 | 3.016% | high | 10 |
| 5 | 678 | 465 | 174 | 39 | 464 | 464 | 2.942% | low | 12 |
| 6 | 643 | 509 | 106 | 28 | 534 | 534 | 2.79% | high | 18 |
| 7 | 632 | 486 | 117 | 29 | 459 | 459 | 2.743% | typical | 22 |
| 8 | 601 | 474 | 98 | 29 | 553 | 553 | 2.608% | typical | 28 |
| 9 | 677 | 501 | 139 | 37 | 590 | 590 | 2.938% | typical | 13 |
| 10 | 658 | 515 | 115 | 28 | 456 | 456 | 2.856% | high | 16 |
| 11 | 617 | 476 | 115 | 26 | 556 | 556 | 2.678% | typical | 24 |
| 12 | 601 | 461 | 112 | 28 | 528 | 528 | 2.608% | low | 29 |
| 13 | 639 | 483 | 122 | 34 | 481 | 481 | 2.773% | typical | 19 |
| 14 | 598 | 458 | 112 | 28 | 469 | 469 | 2.595% | low | 31 |
| 15 | 587 | 468 | 95 | 24 | 485 | 485 | 2.547% | low | 32 |
| 16 | 632 | 499 | 101 | 32 | 602 | 602 | 2.743% | typical | 23 |
| 17 | 721 | 514 | 159 | 48 | 588 | 588 | 3.129% | high | 7 |
| 18 | 576 | 473 | 81 | 22 | 545 | 545 | 2.5% | low | 33 |
| 19 | 720 | 525 | 151 | 44 | 647 | 647 | 3.125% | high | 8 |
| 20 | 730 | 530 | 149 | 51 | 437 | 437 | 3.168% | high | 4 |
| 21 | 668 | 494 | 136 | 38 | 540 | 540 | 2.899% | typical | 15 |
| 22 | 603 | 441 | 129 | 33 | 582 | 582 | 2.617% | low | 26 |
| 23 | 633 | 494 | 101 | 38 | 568 | 568 | 2.747% | typical | 21 |
| 24 | 751 | 542 | 169 | 40 | 565 | 565 | 3.259% | high | 2 |
| 25 | 689 | 498 | 144 | 47 | 503 | 503 | 2.99% | typical | 11 |
| 26 | 703 | 506 | 158 | 39 | 532 | 532 | 3.051% | typical | 9 |
| 27 | 603 | 459 | 117 | 27 | 559 | 559 | 2.617% | low | 27 |
| 28 | 610 | 477 | 111 | 22 | 522 | 522 | 2.647% | typical | 25 |
| 29 | 639 | 480 | 126 | 33 | 559 | 559 | 2.773% | typical | 20 |
| 30 | 525 | 425 | 88 | 12 | 422 | 420 | 2.278% | low | 35 |
| 31 | 546 | 450 | 78 | 18 | 522 | 513 | 2.369% | low | 34 |
| 32 | 722 | 491 | 157 | 74 | 410 | 410 | 3.133% | typical | 6 |
| 33 | 601 | 490 | 92 | 19 | 571 | 564 | 2.608% | typical | 30 |
| 34 | 653 | 483 | 135 | 35 | 606 | 606 | 2.834% | typical | 17 |

## 7. Images containing multiple classes

| Metric | Value |
|---|---|
| Label files with >= 2 classes | 2,148 (13.336%) |
| Images with >= 2 classes | 2,152 (13.345%) |
| Train label files with >= 2 classes | 1633 |
| Val label files with >= 2 classes | 410 |
| Test label files with >= 2 classes | 105 |

Distribution of distinct classes per label file:

| Distinct classes | Label files |
|---|---|
| 1 | 13,959 |
| 2 | 1,799 |
| 3 | 297 |
| 4 | 44 |
| 5 | 8 |

## 8. Class co-occurrence

A co-occurrence pair is counted once per label file that contains both class IDs. Pairs not observed are omitted from the CSV.

- Pairs observed: **257** of 595 possible pairs.

Top 15 co-occurring pairs:

| Class i | Class j | Label files | Images | Train | Val | Test |
|---|---|---|---|---|---|---|
| 31 | 33 | 261 | 265 | 218 | 33 | 10 |
| 8 | 16 | 154 | 154 | 110 | 34 | 10 |
| 11 | 34 | 129 | 129 | 94 | 29 | 6 |
| 3 | 9 | 111 | 111 | 90 | 16 | 5 |
| 6 | 9 | 91 | 91 | 68 | 19 | 4 |
| 0 | 22 | 89 | 89 | 49 | 27 | 13 |
| 5 | 9 | 73 | 73 | 48 | 22 | 3 |
| 0 | 30 | 67 | 67 | 63 | 4 | 0 |
| 14 | 24 | 65 | 65 | 50 | 14 | 1 |
| 0 | 7 | 60 | 60 | 51 | 8 | 1 |
| 0 | 14 | 60 | 60 | 44 | 13 | 3 |
| 24 | 26 | 59 | 59 | 37 | 16 | 6 |
| 5 | 6 | 57 | 57 | 41 | 14 | 2 |
| 8 | 34 | 57 | 57 | 42 | 12 | 3 |
| 0 | 15 | 53 | 53 | 45 | 6 | 2 |

Full pair-level data: `class_cooccurrence.csv`. The heatmap in `class_distribution.png` shows all observed pairs.

## 9. Data quality observations

| Observation | Count |
|---|---|
| Malformed label lines (not counted as boxes) | 0 |
| Empty label files | 0 |
| Boxes whose center-based extent exceeds image bounds | 0 |
| Label files without a matching image | 0 |
| Images without a matching label | 0 |
| Label files shared by multiple images | 19 |

Image counts count every image file that maps to a label. Duplicate-stem pairs and PNG copies of original WEBP/GIF files share one label, so they increase image counts but not label or box counts.

Originals unchanged during analysis: **True** (32,233 source files fingerprinted before and after).

## 10. Relevance to pedestrian-obstacle detection

This dataset exposes only numeric class IDs. Because there is no authoritative ID -> name mapping, this analysis **cannot** state which classes correspond to curbs, poles, bollards, vehicles, pedestrians, street furniture, or any other obstacle category. No obstacle class is selected or excluded here. Any obstacle-specific class choice must wait for a verified class-name mapping or a documented box-level visual audit.

Counts alone also do not demonstrate real-world coverage: a class can be well represented in this dataset and still be irrelevant to pedestrian navigation, and vice versa.

## 11. Outputs

| File | Contents |
|---|---|
| `class_analysis_report.md` | This report |
| `class_distribution.csv` | 35 class rows with split-level counts |
| `class_summary.json` | Machine-readable totals, tiers, co-occurrence, validation |
| `class_distribution.png` | Distribution and co-occurrence charts |
| `class_cooccurrence.csv` | 257 observed class pairs |

## 12. Exact command

```powershell
python "class_analysis.py"
```

## 13. Not done (by rule)

- No dataset file modified, moved, renamed, or deleted.
- No label, class ID, box coordinate, or split changed.
- No class name invented or inferred.
- No obstacle class selected.
- No model trained.
