# Dataset Verification Report

- **Dataset root:** `final batches`
- **Generated:** 2026-11-07T00:25:01
- **Mode:** read-only (no file in the dataset was modified, moved or deleted)
- **Runtime:** 43.88 s

---

## 0. Executive summary

- **16,122 image files** and **16,107 label files** (23,043 boxes) across 4 batches.
- Image/label difference: **+15**.
- **0 empty label files**, **0 label files containing malformed lines** (0 malformed lines).
- Class IDs observed: **[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34]** (min 0, max 34).
- **No metadata files found** (`data.yaml`, `classes.txt`, `obj.names`, README...). **Class names cannot be established from the dataset** — only numeric IDs are verifiable.
- Exact-duplicate image groups: **727**; groups spanning different splits: **222**.
- Among identical-image pairs: **7 class conflicts** (same box, different ID) and **177 geometry mismatches**.
- Likely leakage groups by Roboflow base-name: **639** (1,431 files), **292** already span splits.
- **1 images in formats Ultralytics cannot load** (['.gif']).

### 0.1 Pass / fail summary

| Requirement | Result |
|---|---|
| Folder structure (images/labels × train/val/test × 4 batches) | **PASS** — 12/12 split folders present |
| Annotation syntax (5 tokens, YOLO order) | **PASS** — 16,107 files, 0 malformed lines |
| Coordinates normalised and inside the frame | **PASS** — 0 out-of-range, 0 non-positive sizes, 0 boxes outside image |
| Every image has a label | **PASS** — 0 missing-label images, 0 orphan labels |
| All 35 class IDs present in all batches/splits | **PASS** — IDs 0–34 everywhere, no gaps |
| Class names available | **FAIL** — no metadata anywhere in or beside the dataset |
| Split integrity (no train/val/test leakage) | **FAIL** — 222 MD5-identical groups and 292 source-image groups already span splits |
| Annotation consistency on identical images | **FAIL (localised)** — 7 class conflicts, 177 geometry mismatches |
| One image per label stem | **FAIL (localised)** — 15 duplicate-stem pairs |
| Ready for training | **NO** — clean a derived copy first (section 10) |

## 1. Total images and labels per batch

| Batch | Images | Labels | Diff | Boxes | Empty labels | Malformed files | Image extensions |
|---|---|---|---|---|---|---|---|
| Batch 1 | 3,959 | 3,959 | +0 | 6,020 | 0 | 0 | .jpeg=31, .jpg=3916, .png=12 |
| Batch 2 | 4,100 | 4,100 | +0 | 5,848 | 0 | 0 | .jpeg=38, .jpg=4035, .png=27 |
| Batch 3 | 4,066 | 4,062 | +4 | 5,846 | 0 | 0 | .gif=1, .jpeg=100, .jpg=3898, .png=64, .webp=3 |
| Batch 4 | 3,997 | 3,986 | +11 | 5,329 | 0 | 0 | .jpeg=56, .jpg=3896, .png=45 |

**Total: 16,122 images / 16,107 labels / 23,043 boxes.**

### 1.1 Per-split breakdown

| Batch | Split | Image files | Image stems | Label files | Matched stems | Image files w/o own label | Missing label (stem) | Orphan labels | Boxes |
|---|---|---|---|---|---|---|---|---|---|
| Batch 1 | train | 2944 | 2944 | 2944 | 2944 | 0 | 0 | 0 | 4339 |
| Batch 1 | val | 797 | 797 | 797 | 797 | 0 | 0 | 0 | 1317 |
| Batch 1 | test | 218 | 218 | 218 | 218 | 0 | 0 | 0 | 364 |
| Batch 2 | train | 3176 | 3176 | 3176 | 3176 | 0 | 0 | 0 | 4531 |
| Batch 2 | val | 749 | 749 | 749 | 749 | 0 | 0 | 0 | 1074 |
| Batch 2 | test | 175 | 175 | 175 | 175 | 0 | 0 | 0 | 243 |
| Batch 3 | train | 3018 | 3016 | 3016 | 3016 | 2 | 0 | 0 | 4252 |
| Batch 3 | val | 843 | 841 | 841 | 841 | 2 | 0 | 0 | 1269 |
| Batch 3 | test | 205 | 205 | 205 | 205 | 0 | 0 | 0 | 325 |
| Batch 4 | train | 3155 | 3144 | 3144 | 3144 | 11 | 0 | 0 | 4205 |
| Batch 4 | val | 645 | 645 | 645 | 645 | 0 | 0 | 0 | 836 |
| Batch 4 | test | 197 | 197 | 197 | 197 | 0 | 0 | 0 | 288 |

## 2. Available class names and their sources

**NONE.** No `data.yaml`, `classes.txt`, `obj.names`, README, JSON or CSV file exists anywhere under the dataset root. The search was recursive and explicitly excluded the `labels/` trees, so annotation files were not mistaken for metadata.

Files adjacent to (but outside) the dataset root were also inspected:

- `dataset_analysis.py` (2,489 bytes)
- `verify_dataset.py` (90,675 bytes)

None of these is a class-name list; the `.py` file contains only counting logic. **No class names exist anywhere in or next to this dataset.**

**Limitation (explicit):** it is therefore *impossible* to establish which food/object category each numeric class ID represents. Any name table used from here on would be an assumption, not verified metadata. Class IDs below are reported strictly as numbers.

### 2.1 Circumstantial evidence from filenames (NOT class metadata)

The table below counts how many image filenames contain a keyword. This is a domain sanity check only: a filename does **not** tell us which class ID an object has, and these counts must never be turned into a class list.

| Filename keyword | Images with keyword |
|---|---|
| bed | 75 |
| bus | 69 |
| oven | 68 |
| manhole | 66 |
| waste | 65 |
| toothbrush | 62 |
| sign | 60 |
| barrier | 51 |
| toilet | 46 |
| chair | 41 |
| traffic | 41 |
| table | 35 |
| stair | 33 |
| bottle | 21 |
| knife | 20 |

The spread of keywords suggests a **heterogeneous object dataset** (restroom signs, street furniture, appliances, furniture, vehicles, waste containers) rather than a food-product dataset. If the project is meant to be food-specific, confirm the intended scope with the dataset owner before training.

## 3. Class IDs found in each batch

Counts are boxes per class (valid lines only).

| Class ID | Batch 1 | Batch 2 | Batch 3 | Batch 4 | Total |
|---|---|---|---|---|---|
| 0 | 247 | 230 | 277 | 189 | 943 |
| 1 | 197 | 186 | 195 | 170 | 748 |
| 2 | 206 | 170 | 192 | 158 | 726 |
| 3 | 185 | 175 | 175 | 140 | 675 |
| 4 | 183 | 173 | 177 | 162 | 695 |
| 5 | 176 | 174 | 177 | 151 | 678 |
| 6 | 165 | 176 | 162 | 140 | 643 |
| 7 | 156 | 164 | 164 | 148 | 632 |
| 8 | 146 | 162 | 147 | 146 | 601 |
| 9 | 180 | 176 | 169 | 152 | 677 |
| 10 | 174 | 167 | 165 | 152 | 658 |
| 11 | 153 | 165 | 153 | 146 | 617 |
| 12 | 165 | 163 | 126 | 147 | 601 |
| 13 | 159 | 174 | 150 | 156 | 639 |
| 14 | 161 | 152 | 138 | 147 | 598 |
| 15 | 148 | 166 | 133 | 140 | 587 |
| 16 | 170 | 150 | 161 | 151 | 632 |
| 17 | 195 | 185 | 182 | 159 | 721 |
| 18 | 159 | 138 | 136 | 143 | 576 |
| 19 | 187 | 184 | 184 | 165 | 720 |
| 20 | 179 | 187 | 205 | 159 | 730 |
| 21 | 180 | 160 | 174 | 154 | 668 |
| 22 | 161 | 143 | 144 | 155 | 603 |
| 23 | 141 | 168 | 171 | 153 | 633 |
| 24 | 189 | 193 | 210 | 159 | 751 |
| 25 | 184 | 164 | 183 | 158 | 689 |
| 26 | 186 | 180 | 177 | 160 | 703 |
| 27 | 143 | 149 | 151 | 160 | 603 |
| 28 | 152 | 160 | 161 | 137 | 610 |
| 29 | 175 | 153 | 153 | 158 | 639 |
| 30 | 136 | 130 | 127 | 132 | 525 |
| 31 | 134 | 136 | 142 | 134 | 546 |
| 32 | 210 | 167 | 182 | 163 | 722 |
| 33 | 153 | 157 | 160 | 131 | 601 |
| 34 | 185 | 171 | 143 | 154 | 653 |

- Distinct class IDs overall: **35**
- Observed range: **0 … 34**
- IDs absent inside that range: **none**
- IDs >= 35: **none** (the claim of 35 classes implies valid IDs 0–34)
- Class balance: least frequent class has **525** boxes, most frequent **943** (ratio **1.80:1**).
- **Interpretation:** a ratio near 1.0 means the classes are almost perfectly balanced, which is atypical for real-world detection data (usually long-tailed) and suggests curated balancing, resampling or synthetic/duplicated sources. It is not an error, but it means validation mAP will not reflect real-world class imbalance.

### 3.1 IDs present per split

| Batch | Split | #IDs | Class IDs |
|---|---|---|---|
| Batch 1 | train | 35 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34] |
| Batch 1 | val | 35 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34] |
| Batch 1 | test | 35 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34] |
| Batch 2 | train | 35 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34] |
| Batch 2 | val | 35 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34] |
| Batch 2 | test | 35 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34] |
| Batch 3 | train | 35 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34] |
| Batch 3 | val | 35 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34] |
| Batch 3 | test | 35 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34] |
| Batch 4 | train | 35 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34] |
| Batch 4 | val | 35 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34] |
| Batch 4 | test | 35 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34] |

## 4. Class-ID consistency comparison

### 4.1 Set comparison

| Pair | Same ID set? | Only in left | Only in right |
|---|---|---|---|
| Batch 1 vs Batch 2 | IDENTICAL | [] | [] |
| Batch 1 vs Batch 3 | IDENTICAL | [] | [] |
| Batch 1 vs Batch 4 | IDENTICAL | [] | [] |
| Batch 2 vs Batch 3 | IDENTICAL | [] | [] |
| Batch 2 vs Batch 4 | IDENTICAL | [] | [] |
| Batch 3 vs Batch 4 | IDENTICAL | [] | [] |
| ALL | IDENTICAL | union=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34] | intersection=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34] |

**Important caveat:** identical *sets* of IDs do not prove identical *semantics*. Two batches can both use 0–34 while assigning different meanings to the same number. Section 4.2 tests this empirically.

### 4.2 Inferred class-ID mapping (evidence from shared-file annotations)

Method: source images are identified by their Roboflow base name (the `_jpg.rf.<hash>` augmentation suffix is stripped). When the same source image appears in two batches with an identical box geometry (coordinates rounded to 6 decimals), the boxes are matched and `(id_A -> id_B)` is tallied. A consistent tally across many images is evidence of a systematic remap between batches; a mixed tally is evidence of inconsistent annotation.

**Batch 1 -> Batch 3** — 67 shared source images with identical coordinates

| Source ID | Best target ID | Support | Total | Confidence | Ambiguous? |
|---|---|---|---|---|---|
| 0 | 0 | 8 | 8 | 1.00 | no |
| 1 | 1 | 5 | 5 | 1.00 | no |
| 2 | 2 | 7 | 7 | 1.00 | no |
| 3 | 3 | 6 | 6 | 1.00 | no |
| 5 | 5 | 8 | 8 | 1.00 | no |
| 6 | 6 | 10 | 10 | 1.00 | no |
| 7 | 7 | 12 | 12 | 1.00 | no |
| 9 | 9 | 1 | 1 | 1.00 | no |
| 12 | 12 | 3 | 3 | 1.00 | no |
| 15 | 15 | 16 | 16 | 1.00 | no |
| 17 | 17 | 1 | 1 | 1.00 | no |
| 18 | 18 | 1 | 1 | 1.00 | no |
| 20 | 20 | 4 | 4 | 1.00 | no |
| 23 | 23 | 5 | 5 | 1.00 | no |
| 24 | 24 | 2 | 2 | 1.00 | no |
| 25 | 25 | 2 | 2 | 1.00 | no |
| 27 | 27 | 4 | 4 | 1.00 | no |
| 28 | 28 | 13 | 13 | 1.00 | no |
| 29 | 29 | 1 | 1 | 1.00 | no |
| 30 | 30 | 1 | 1 | 1.00 | no |

**Batch 1 -> Batch 4** — 37 shared source images with identical coordinates

| Source ID | Best target ID | Support | Total | Confidence | Ambiguous? |
|---|---|---|---|---|---|
| 0 | 0 | 3 | 3 | 1.00 | no |
| 3 | 3 | 1 | 1 | 1.00 | no |
| 4 | 4 | 2 | 2 | 1.00 | no |
| 8 | 8 | 1 | 1 | 1.00 | no |
| 25 | 25 | 1 | 1 | 1.00 | no |
| 30 | 30 | 31 | 31 | 1.00 | no |
| 31 | 31 | 9 | 9 | 1.00 | no |
| 33 | 33 | 6 | 6 | 1.00 | no |

**Batch 2 -> Batch 3** — 35 shared source images with identical coordinates

| Source ID | Best target ID | Support | Total | Confidence | Ambiguous? |
|---|---|---|---|---|---|
| 1 | 1 | 1 | 1 | 1.00 | no |
| 2 | 2 | 12 | 12 | 1.00 | no |
| 3 | 3 | 8 | 8 | 1.00 | no |
| 5 | 5 | 11 | 11 | 1.00 | no |
| 6 | 6 | 7 | 7 | 1.00 | no |
| 9 | 9 | 2 | 2 | 1.00 | no |
| 23 | 23 | 6 | 6 | 1.00 | no |

**Batch 2 -> Batch 4** — 4 shared source images with identical coordinates

| Source ID | Best target ID | Support | Total | Confidence | Ambiguous? |
|---|---|---|---|---|---|
| 4 | 4 | 1 | 1 | 1.00 | no |
| 24 | 14 | 1 | 2 | 0.50 | yes |
| 25 | 25 | 1 | 1 | 1.00 | no |

**Batch 3 -> Batch 4** — 23 shared source images with identical coordinates

| Source ID | Best target ID | Support | Total | Confidence | Ambiguous? |
|---|---|---|---|---|---|
| 2 | 2 | 2 | 2 | 1.00 | no |
| 3 | 3 | 10 | 10 | 1.00 | no |
| 5 | 5 | 1 | 1 | 1.00 | no |
| 6 | 6 | 8 | 8 | 1.00 | no |
| 9 | 9 | 2 | 2 | 1.00 | no |
| 23 | 23 | 3 | 3 | 1.00 | no |
| 24 | 24 | 1 | 1 | 1.00 | no |

### 4.3 Verdict per batch pair

| Batch pair | Matched boxes | Non-identity matches | Ambiguous source IDs | Verdict |
|---|---|---|---|---|
| Batch 1 -> Batch 3 | 110 | 0 | 0 | consistent — identical IDs |
| Batch 1 -> Batch 4 | 54 | 0 | 0 | consistent — identical IDs |
| Batch 2 -> Batch 3 | 47 | 0 | 0 | consistent — identical IDs |
| Batch 2 -> Batch 4 | 4 | 1 | 1 | INCONSISTENT (mixed IDs) |
| Batch 3 -> Batch 4 | 27 | 0 | 0 | consistent — identical IDs |

A *systematic remap* means the same source image carries different numeric IDs in the two batches, but the mapping is one-to-one and repeatable — fixable by an explicit permutation table. *Inconsistent* means the same geometry maps to different IDs un-repeatably — those are label errors.

**1 box match has identical coordinates but a different class ID.** Examples:

- `-0378d390-5ae2-4d09-84bb-387e850f5e5a-png` — Batch 2 id 24 vs Batch 4 id 14

Shared source images whose *coordinates differ* (different augmentation or re-annotation):

| Batch pair | Source images with differing coordinates |
|---|---|
| Batch 1 -> Batch 3 | 27 |
| Batch 1 -> Batch 2 | 44 |
| Batch 2 -> Batch 3 | 60 |
| Batch 1 -> Batch 4 | 26 |
| Batch 3 -> Batch 4 | 141 |
| Batch 2 -> Batch 4 | 209 |

### 4.4 Overall class-ID verdict

- All 4 batches use exactly the same ID set: **0–34**, and every ID appears in every split of every batch.
- Shared-source-image evidence covers **242** matched boxes (1.05% of all boxes); **1** of them carries a different ID in the other batch.
- Independent evidence from byte-identical images: **744 fully consistent pairs**, 7 class conflicts, 177 geometry mismatches (section 8.1.1).

**Conclusion: the batches are *mostly* aligned, but 1 conflicting box match shows that some annotations disagree.** There is no evidence of a full permutation between batches; treat the conflicts as label errors to review (section 8.1.1).

**Limitation:** this proves *numeric consistency*, not *semantic correctness*. Without a class-name list there is no way to prove that ID 7 means the same object category in every batch — only that the batches were annotated consistently with each other.

## 5. Annotation format and coordinate validation

Expected format per line: `class_id x_center y_center width height`, whitespace separated, all four coordinates normalised to [0, 1].

| Check | Result |
|---|---|
| Files with exactly 5 tokens everywhere | 16,107 / 16,107 |
| Malformed lines (any reason) | 0 |
| Files with BOM | 0 |
| Files with CRLF line endings | 838 |
| Files that are not valid UTF-8 | 0 |
| Coordinate values outside [0,1] | 0 |
| Non-positive width/height | 0 |
| Width or height > 1 | 0 |
| Box extends outside the image frame | 0 |
| Warning: tiny boxes (area < 1e-4) | 3 |
| Warning: boxes touching the image edge | 2,844 |
| Warning: extreme aspect ratio (>20:1) | 1 |
| Warning: duplicated identical line in a file | 0 |

No malformed lines detected.

## 6. Empty annotations and image/label pairing

| Batch | Empty label files | Label files | % empty | Image files | Image stems | Image files without own label |
|---|---|---|---|---|---|---|
| Batch 1 | 0 | 3959 | 0.00% | 3959 | 3959 | 0 |
| Batch 2 | 0 | 4100 | 0.00% | 4100 | 4100 | 0 |
| Batch 3 | 0 | 4062 | 0.00% | 4066 | 4062 | 4 |
| Batch 4 | 0 | 3986 | 0.00% | 3997 | 3986 | 11 |

An *empty* label file is a valid YOLO background image only if it is intentional. If these were meant to be annotated, they will teach the model that the objects are absent.

### 6.1 Images without a matching label

None.

### 6.2 Label files without a matching image (orphans)

None.

### 6.5 Several images sharing one stem in the same split

Only one label file can exist per stem, so at least one image is effectively unlabelled.

- 15 stems affected; **0** pairs are byte-identical (format duplicates) and **15** pairs are *different images* that share a label — those 15 images have no annotation of their own and must be reviewed.

- `Batch 3/train/image22`: Batch 3\images\train\image22.jpeg, Batch 3\images\train\image22.png — **different images**
- `Batch 3/train/image24`: Batch 3\images\train\image24.jpeg, Batch 3\images\train\image24.png — **different images**
- `Batch 3/val/image31`: Batch 3\images\val\image31.jpeg, Batch 3\images\val\image31.png — **different images**
- `Batch 3/val/image33`: Batch 3\images\val\image33.jpeg, Batch 3\images\val\image33.png — **different images**
- `Batch 4/train/image30`: Batch 4\images\train\image30.jpeg, Batch 4\images\train\image30.png — **different images**
- `Batch 4/train/image32`: Batch 4\images\train\image32.jpeg, Batch 4\images\train\image32.png — **different images**
- `Batch 4/train/image35`: Batch 4\images\train\image35.jpeg, Batch 4\images\train\image35.png — **different images**
- `Batch 4/train/image37`: Batch 4\images\train\image37.jpeg, Batch 4\images\train\image37.png — **different images**
- `Batch 4/train/image38`: Batch 4\images\train\image38.jpeg, Batch 4\images\train\image38.png — **different images**
- `Batch 4/train/image39`: Batch 4\images\train\image39.jpeg, Batch 4\images\train\image39.png — **different images**
- `Batch 4/train/image43`: Batch 4\images\train\image43.jpeg, Batch 4\images\train\image43.png — **different images**
- `Batch 4/train/image45`: Batch 4\images\train\image45.jpeg, Batch 4\images\train\image45.png — **different images**
- `Batch 4/train/image46`: Batch 4\images\train\image46.jpeg, Batch 4\images\train\image46.png — **different images**
- `Batch 4/train/image47`: Batch 4\images\train\image47.jpeg, Batch 4\images\train\image47.png — **different images**
- `Batch 4/train/image48`: Batch 4\images\train\image48.jpeg, Batch 4\images\train\image48.png — **different images**

## 7. The 16,103 vs 16,107 discrepancy

The reconciliation is exact, and it is a **counting-method artefact, not missing data**:

- Label files (= distinct label stems): **16,107**
- Distinct image stems: **16,107** — every stem has a label (missing-label images: 0).
- Physical image files: **16,122** = 16,107 stems + **15 extra files that share a stem with another image** in the same split (e.g. `image22.jpeg` and `image22.png` — one label can only cover one of them).
- Stems whose only image is `.webp`/`.gif`: **4**

Files involved (excluded by any `.jpg/.jpeg/.png`-only filter):

- `Batch 3/train/15_9f2628f7-0a6e-40db-a06a-5aa1cc95dee2` — Batch 3\images\train\15_9f2628f7-0a6e-40db-a06a-5aa1cc95dee2.webp
- `Batch 3/train/A9674866481b64aa5828c862fd0bdf314T.jpg_640x640.jpg` — Batch 3\images\train\A9674866481b64aa5828c862fd0bdf314T.jpg_640x640.jpg.webp
- `Batch 3/train/ADA-1025-Blue-Mens-Restroom-Wall-Sign-Male-and-ISA-pictos-6x8_580x` — Batch 3\images\train\ADA-1025-Blue-Mens-Restroom-Wall-Sign-Male-and-ISA-pictos-6x8_580x.gif
- `Batch 3/train/DSC_3857copy` — Batch 3\images\train\DSC_3857copy.webp

**Formula:** 16,107 distinct image stems − 4 non-JPG/PNG-only stems = **16,103**, which is the brief's 16,103. The brief's 16,107 is the label count. This is reproduced exactly by an earlier script that (a) filtered image extensions to jpg/jpeg/png and (b) counted stems in a `set()` while counting labels with a glob.

**For a trainer, the real numbers are 16,122 image files vs 16,107 label files (+15).** The 15 surplus files are not "missing labels" in the usual sense; they are duplicate-stem variants (section 6.5). No label belongs to an image that does not exist (orphans: 0).

## 8. Image file audit

| Check | Result |
|---|---|
| Images decoded successfully | 16,122 |
| Corrupt / undecodable | 0 |
| Formats Ultralytics cannot load | 1 |
| Very small images (< 32 px on a side) | 0 |
| Small images (< 100 px on the shortest side) | 13 |
| Pixel dimension range across all images | 76 – 8660 px |
| Exact duplicate groups (MD5) | 727 |
| Files in duplicate groups | 1,545 |
| Duplicate groups spanning more than one split | 222 |
| ... of those, a leak inside a single batch | 82 |
| Duplicate groups spanning more than one batch (any split) | 592 |
| Duplicate pairs with a class conflict (IoU >= 0.5) | 7 |
| Duplicate pairs with a geometry mismatch | 177 |
| Likely leakage groups (Roboflow base name) | 639 |
| ... of which span different splits | 292 |
| Files inside those leakage groups | 1,431 |

**Images in formats Ultralytics cannot load** (must be converted before training):

- `Batch 3\images\train\ADA-1025-Blue-Mens-Restroom-Wall-Sign-Male-and-ISA-pictos-6x8_580x.gif`

**Small images (< 100 px)** are upscaled by YOLO; small objects in them lose detail. 13 file(s); examples:

- `Batch 1\images\train\1hvtpy9w.jpg` (474×97)
- `Batch 1\images\train\798859_208.jpg` (77×220)
- `Batch 1\images\train\798859_242.jpg` (81×218)
- `Batch 1\images\train\images101.jpg` (86×400)
- `Batch 2\images\train\1v7diomf.jpg` (81×218)
- `Batch 2\images\train\images-2025-01-31T153636_161_jpeg.rf.938329010f8ef2aa4a8d87f3636f3f0b.jpg` (76×400)
- `Batch 2\images\val\1tym9oy8.jpg` (89×220)
- `Batch 2\images\val\sez0rt03.jpg` (92×400)
- `Batch 4\images\train\images101.jpg` (86×400)
- `Batch 4\images\train\images344.jpg` (92×400)
- `Batch 4\images\train\images353.jpg` (92×400)
- `Batch 4\images\val\images367.jpg` (83×400)
- `Batch 4\images\val\mjcs17q6.jpg` (77×220)

### 8.1 Exact duplicate images

727 groups of byte-identical images (1,545 files) were found. 222 of them contain images from more than one split, and 82 contain a train/val/test leak inside a single batch. 592 groups span more than one batch.

Full list: `duplicates.csv`.

#### 8.1.1 Annotation consistency among identical images

- File pairs compared: **927**
- Fully consistent (same boxes at IoU >= 0.5, same classes): **744**
- Geometry mismatch (missing / shifted / differently sized boxes): **177**
- Class conflict (box matches at IoU >= 0.5 but the class ID differs): **7**

Because the image bytes are identical, every difference here is an annotation inconsistency, not a visual difference. It is direct label noise.

Examples of class conflicts on identical images:

- `Batch 2\images\train\7911_376.jpg` (id 31) vs `Batch 2\images\val\itmbghqk.jpg` (id 33), IoU 0.884
- `Batch 2\images\train\7911_376.jpg` (id 33) vs `Batch 2\images\val\itmbghqk.jpg` (id 31), IoU 0.824
- `Batch 2\images\train\7911_448.jpeg` (id 4) vs `Batch 3\images\train\image28.jpeg` (id 33), IoU 0.512
- `Batch 2\images\train\7911_461.jpeg` (id 4) vs `Batch 4\images\train\image45.jpeg` (id 31), IoU 0.76
- `Batch 3\images\train\5fg3vng1.jpg` (id 18) vs `Batch 3\images\train\t6d1wkdv.jpg` (id 17), IoU 0.821
- `Batch 3\images\train\cavill-grey-fabric-bed-frame-upholstered-double-laura-james-1.png` (id 33) vs `Batch 4\images\train\cavill-grey-fabric-bed-frame-upholstered-double-laura-james-1.png` (id 18), IoU 0.864
- `Batch 3\images\train\image36.jpeg` (id 4) vs `Batch 4\images\train\image37.jpeg` (id 31), IoU 0.521
- `Batch 3\images\train\image42.jpeg` (id 4) vs `Batch 4\images\train\image43.jpeg` (id 31), IoU 0.539

Examples of duplicate groups:

- MD5 `facb1f1a4e1d…` ×5 at Batch 1/train, Batch 2/train, Batch 3/train, Batch 3/val — consistent
  - `Batch 1\images\train\798859_191.jpg`
  - `Batch 2\images\train\7911_370.jpg`
  - `Batch 3\images\train\ada-sign-women-fs_1.jpg`
  - `Batch 3\images\val\ada-sign-women-fs_1.jpg`
- MD5 `56203de4475a…` ×5 at Batch 1/train, Batch 2/train, Batch 3/train, Batch 3/val — consistent
  - `Batch 1\images\train\798859_225.jpg`
  - `Batch 2\images\train\7911_375.jpg`
  - `Batch 2\images\train\any-use-vector-eps-vectors-icons-set-any-use-men-women-toilet-sign-181062676.jpg`
  - `Batch 3\images\train\any-use-vector-eps-vectors-icons-set-any-use-men-women-toilet-sign-181062676.jpg`
- MD5 `92bec7936550…` ×5 at Batch 1/train, Batch 2/train, Batch 3/train, Batch 3/val, Batch 4/train — consistent
  - `Batch 1\images\train\798859_391.jpg`
  - `Batch 2\images\train\785879412_0340_0340.jpg`
  - `Batch 3\images\train\785879412_0340_0340.jpg`
  - `Batch 3\images\val\785879412_0340_0340.jpg`
- MD5 `ccef824e13bc…` ×4 at Batch 1/train, Batch 2/train, Batch 3/train — consistent
  - `Batch 1\images\train\1dca04c4_.jpg`
  - `Batch 1\images\train\798859_294.jpg`
  - `Batch 2\images\train\7911_164.jpg`
  - `Batch 3\images\train\3517886-toilet-sign-women.jpg`
- MD5 `d1dbe46473c7…` ×4 at Batch 1/train, Batch 3/test, Batch 3/train, Batch 4/train — geometry mismatch
  - `Batch 1\images\train\798859_206.jpg`
  - `Batch 3\images\train\6jqlqazd.jpg`
  - `Batch 3\images\test\6jqlqazd.jpg`
  - `Batch 4\images\train\th175.jpg`
- MD5 `424d5d2e82ee…` ×4 at Batch 1/train, Batch 3/train, Batch 4/train — geometry mismatch
  - `Batch 1\images\train\798859_222.jpg`
  - `Batch 3\images\train\u75fdr1x.jpg`
  - `Batch 4\images\train\th232.jpg`
  - `Batch 4\images\train\u75fdr1x.jpg`
- MD5 `80870b265146…` ×4 at Batch 1/train, Batch 3/train, Batch 4/train — geometry mismatch
  - `Batch 1\images\train\798859_232.jpg`
  - `Batch 1\images\train\a46shdsx.jpg`
  - `Batch 3\images\train\a46shdsx.jpg`
  - `Batch 4\images\train\th264.jpg`
- MD5 `f2ae0ea40140…` ×4 at Batch 1/train, Batch 2/train, Batch 3/train, Batch 4/train — consistent
  - `Batch 1\images\train\798859_298.jpg`
  - `Batch 2\images\train\7911_166.jpg`
  - `Batch 3\images\train\360_F_573770422_FQ7v28Yiv3LAsgAIOIKSoH2yzHuVkPpL.jpg`
  - `Batch 4\images\train\360_F_573770422_FQ7v28Yiv3LAsgAIOIKSoH2yzHuVkPpL.jpg`
- MD5 `817cb9043ffc…` ×4 at Batch 1/train, Batch 2/train, Batch 3/train — consistent
  - `Batch 1\images\train\798859_309.jpg`
  - `Batch 2\images\train\55bee9183418ab4559b19c448318e9c4.jpg`
  - `Batch 2\images\train\7911_218.jpg`
  - `Batch 3\images\train\55bee9183418ab4559b19c448318e9c4.jpg`
- MD5 `96cd29ea2244…` ×4 at Batch 1/train, Batch 3/train, Batch 3/val — consistent
  - `Batch 1\images\train\798859_317.jpg`
  - `Batch 1\images\train\c22136b4_.jpg`
  - `Batch 3\images\train\61EcRje1lCL._AC_UF10001000_QL80_.jpg`
  - `Batch 3\images\val\61EcRje1lCL._AC_UF10001000_QL80_.jpg`
- MD5 `defa42facc33…` ×4 at Batch 1/train, Batch 1/val, Batch 3/train, Batch 4/train — consistent
  - `Batch 1\images\train\798859_337.jpg`
  - `Batch 1\images\val\66cd04db_.jpg`
  - `Batch 3\images\train\90514458-toilet-sign-wc-women.jpg`
  - `Batch 4\images\train\791881_206.jpg`
- MD5 `9e952731649b…` ×4 at Batch 1/train, Batch 2/train, Batch 3/train — consistent
  - `Batch 1\images\train\798859_415.jpg`
  - `Batch 2\images\train\7911_265.jpg`
  - `Batch 3\images\train\1juklona.jpg`
  - `Batch 3\images\train\7D154AN_WEB_600.jpg`
- MD5 `c1a17f5d21cb…` ×3 at Batch 1/train, Batch 2/train, Batch 3/train — consistent
  - `Batch 1\images\train\020-BREAKING-BUTCHER-ORANGE_BLACK.jpg`
  - `Batch 2\images\train\pbvcnqxm.jpg`
  - `Batch 3\images\train\020-BREAKING-BUTCHER-ORANGE_BLACK.jpg`
- MD5 `111659161df2…` ×3 at Batch 1/train, Batch 4/train — geometry mismatch
  - `Batch 1\images\train\0e7e5497_.jpg`
  - `Batch 1\images\train\images515.jpg`
  - `Batch 4\images\train\images515.jpg`
- MD5 `1b2c20d41825…` ×3 at Batch 1/train, Batch 2/train — consistent
  - `Batch 1\images\train\2c49479c_.jpg`
  - `Batch 1\images\train\798859_319.jpg`
  - `Batch 2\images\train\7911_248.jpg`

#### 8.1.2 Where the leaky duplicates sit

| Location pair | Duplicate file pairs |
|---|---|
| Batch 3/train  <->  Batch 3/val | 47 |
| Batch 1/val  <->  Batch 3/train | 20 |
| Batch 1/train  <->  Batch 3/val | 19 |
| Batch 3/val  <->  Batch 4/train | 14 |
| Batch 1/train  <->  Batch 2/val | 14 |
| Batch 3/train  <->  Batch 4/val | 14 |
| Batch 2/train  <->  Batch 3/val | 13 |
| Batch 2/val  <->  Batch 3/train | 13 |
| Batch 1/val  <->  Batch 4/train | 11 |
| Batch 2/val  <->  Batch 4/train | 11 |
| Batch 2/train  <->  Batch 2/val | 9 |
| Batch 4/train  <->  Batch 4/val | 9 |
| Batch 2/train  <->  Batch 4/val | 7 |
| Batch 1/train  <->  Batch 4/val | 6 |
| Batch 1/train  <->  Batch 1/val | 5 |
| Batch 1/val  <->  Batch 2/train | 5 |
| Batch 3/test  <->  Batch 3/train | 4 |
| Batch 3/test  <->  Batch 4/train | 3 |
| Batch 2/train  <->  Batch 4/test | 3 |
| Batch 4/test  <->  Batch 4/train | 3 |
| Batch 1/train  <->  Batch 4/test | 3 |
| Batch 1/test  <->  Batch 3/train | 3 |
| Batch 1/train  <->  Batch 3/test | 2 |
| Batch 2/train  <->  Batch 3/test | 2 |
| Batch 1/test  <->  Batch 1/train | 2 |
| Batch 2/test  <->  Batch 2/train | 2 |
| Batch 1/train  <->  Batch 2/test | 1 |
| Batch 1/val  <->  Batch 3/test | 1 |
| Batch 1/test  <->  Batch 1/val | 1 |
| Batch 1/val  <->  Batch 2/test | 1 |
| Batch 1/test  <->  Batch 3/val | 1 |
| Batch 1/test  <->  Batch 2/train | 1 |
| Batch 1/test  <->  Batch 4/train | 1 |
| Batch 2/test  <->  Batch 4/train | 1 |
| Batch 2/test  <->  Batch 3/train | 1 |
| Batch 3/train  <->  Batch 4/test | 1 |
| Batch 3/test  <->  Batch 3/val | 1 |

### 8.2 Likely train/val/test leakage (Roboflow base name)

639 source-image groups appear more than once; **292 of them already span different splits** and 347 appear in different batches only (these become leakage if the batches are merged).

Differently augmented copies of one source image are not caught by MD5 but still inflate validation scores. Full list: `leakage_groups.csv`.

- `images112` ×3 — splits ['train', 'val'], batches ['Batch 1', 'Batch 3', 'Batch 4']
- `images127` ×4 — splits ['train', 'val'], batches ['Batch 1', 'Batch 2', 'Batch 3']
- `images130` ×3 — splits ['test', 'train'], batches ['Batch 1', 'Batch 3', 'Batch 4']
- `images135` ×2 — splits ['train', 'val'], batches ['Batch 1', 'Batch 4']
- `images138` ×2 — splits ['train', 'val'], batches ['Batch 1', 'Batch 4']
- `images140` ×3 — splits ['train', 'val'], batches ['Batch 1', 'Batch 2']
- `images165` ×4 — splits ['test', 'train'], batches ['Batch 1', 'Batch 2', 'Batch 4']
- `images207` ×3 — splits ['test', 'train'], batches ['Batch 1', 'Batch 4']
- `images224` ×2 — splits ['train', 'val'], batches ['Batch 1', 'Batch 4']
- `images245` ×2 — splits ['test', 'train'], batches ['Batch 1', 'Batch 4']
- `images288` ×2 — splits ['train', 'val'], batches ['Batch 1', 'Batch 4']
- `images352` ×2 — splits ['train', 'val'], batches ['Batch 1']
- `images362` ×2 — splits ['train', 'val'], batches ['Batch 1', 'Batch 2']
- `images427` ×2 — splits ['test', 'train'], batches ['Batch 1']
- `0vk6txp5` ×2 — splits ['train', 'val'], batches ['Batch 1', 'Batch 3']
- `2bq7etbu` ×2 — splits ['train', 'val'], batches ['Batch 1', 'Batch 3']
- `4vvqxe8s` ×2 — splits ['train', 'val'], batches ['Batch 1', 'Batch 3']
- `5a98xudn` ×2 — splits ['train', 'val'], batches ['Batch 1', 'Batch 3']
- `5k95x9j2` ×2 — splits ['train', 'val'], batches ['Batch 1', 'Batch 3']
- `b5wizdaf` ×2 — splits ['train', 'val'], batches ['Batch 1', 'Batch 3']
- `cbgq13zh` ×2 — splits ['train', 'val'], batches ['Batch 1', 'Batch 3']
- `ggk7h3yt` ×2 — splits ['train', 'val'], batches ['Batch 1', 'Batch 3']
- `images13` ×3 — splits ['train', 'val'], batches ['Batch 1', 'Batch 2', 'Batch 4']
- `images19` ×4 — splits ['train', 'val'], batches ['Batch 1', 'Batch 2', 'Batch 3', 'Batch 4']
- `images247` ×2 — splits ['train', 'val'], batches ['Batch 1', 'Batch 2']
- `images249` ×3 — splits ['test', 'train', 'val'], batches ['Batch 1', 'Batch 2', 'Batch 3']
- `images286` ×2 — splits ['train', 'val'], batches ['Batch 1', 'Batch 4']
- `images291` ×3 — splits ['train', 'val'], batches ['Batch 1', 'Batch 2', 'Batch 3']
- `images308` ×2 — splits ['train', 'val'], batches ['Batch 1', 'Batch 2']
- `images32` ×3 — splits ['test', 'train', 'val'], batches ['Batch 1', 'Batch 3', 'Batch 4']
- `images335` ×2 — splits ['train', 'val'], batches ['Batch 1', 'Batch 2']
- `images4` ×2 — splits ['train', 'val'], batches ['Batch 1', 'Batch 4']
- `images401` ×2 — splits ['train', 'val'], batches ['Batch 1', 'Batch 3']
- `images415` ×2 — splits ['train', 'val'], batches ['Batch 1', 'Batch 3']
- `images49` ×2 — splits ['train', 'val'], batches ['Batch 1', 'Batch 4']
- `images50` ×3 — splits ['train', 'val'], batches ['Batch 1', 'Batch 2', 'Batch 4']
- `images52` ×3 — splits ['train', 'val'], batches ['Batch 1', 'Batch 3', 'Batch 4']
- `images66` ×3 — splits ['test', 'train', 'val'], batches ['Batch 1', 'Batch 3', 'Batch 4']
- `l52312rs` ×2 — splits ['train', 'val'], batches ['Batch 1', 'Batch 3']
- `xngmzwin` ×2 — splits ['train', 'val'], batches ['Batch 1', 'Batch 3']
- ... and 252 more cross-split groups

## 9. Potential dataset problems (ranked)

| Severity | Problem | Detail |
|---|---|---|
| **Critical** | No class-name metadata at all | Class IDs cannot be mapped to names. Any names used downstream are assumptions. |
| **Critical** | Exact duplicate images across splits (leakage) | 222 MD5-identical groups span splits, 82 of them inside a single batch (train/val/test). |
| **High** | Same image annotated with different classes | 7 byte-identical image pairs have a box that matches at IoU >= 0.5 but carries a different class ID. |
| **High** | Same image annotated with different boxes | 177 byte-identical image pairs have missing / shifted / differently sized boxes. |
| **High** | Augmented copies leak across splits | 292 source-image groups already span different splits; validation mAP is therefore optimistic. |
| **Medium** | Cross-batch class-ID conflicts | 1 box match has identical coordinates but different class IDs in different batches (localised label error, not a remap). |
| **Medium** | Unsupported image formats | 1 image (.gif) cannot be loaded by Ultralytics. |
| **Medium** | Very small boxes | 3 boxes have area < 1e-4; consider higher input resolution or excluding them. |
| **Medium** | Two images share one stem in the same split | 15 stems have 2 image files but only 1 label; 15 of the pairs are *different images*, so that many images have no annotation of their own. |
| **Low** | Possible domain mismatch (circumstantial) | Filenames repeatedly mention bed (75), bus (69), oven (68), manhole (66), waste (65). This is not class metadata, but if the project targets food products, confirm that this heterogeneous object dataset is the intended data. |
| **Low** | Images only in .webp/.gif | 4 stems have no JPG/PNG version; extension-filtered scripts miss them. |

## 10. Recommendations for the next stage

1. **Do not train on this layout yet.** Freeze one authoritative class list (id → name) and one `data.yaml` first. No metadata exists, so it must come from the dataset owner; it cannot be derived from the annotation files.
2. **Resolve the cross-batch class semantics before merging.** Section 4.3 gives the verdict per batch pair and 1 box match already shows identical geometry with different IDs. If a pair is a *systematic remap*, apply an explicit permutation table in a derived copy. If it is *inconsistent*, treat those boxes as label errors.
3. **Fix the 15 duplicate-stem pairs** (section 6.5): keep exactly one image per stem per split, or rename the second image and give it its own label. Right now one image of each pair can never be trained on.
4. **Handle the 4 stems that exist only as `.webp`/`.gif`** (section 7). Convert them to JPG/PNG in the derived copy; `.gif` cannot be loaded by Ultralytics at all.
5. **Remove leakage before splitting.** Section 8.2 groups images by source (Roboflow base name): 639 groups contain more than one copy, and 292 already span splits. Split by *group*, not by file, then re-check.
6. **Review the near-uniform class distribution** (section 3). Every class has a similar box count, which is atypical for real-world food data and suggests deliberate balancing or synthetic/duplicated sources. Confirm the provenance so accuracy claims are defensible.
7. **Keep tiny boxes in mind** (3 boxes with area < 1e-4, 1 extreme aspect ratio). Either train at higher input resolution (e.g. 640 → 960) or document their exclusion.
8. **Do all cleaning in a new directory**, never in `final batches`. Log every change (old file, line, new file, line) so the process is auditable and reproducible.
9. **Re-run `verify_dataset.py` on the cleaned copy** and require: 0 hard errors, 0 duplicate stems, 0 cross-split leakage groups, all images decodable, and exactly one label per image.
10. **Only then** build the merged dataset with a documented, seeded split. Keep the original batches untouched as the source of truth.
11. **Class names:** if the owner cannot provide them, state in the project write-up that names are unknown/unverified and report classes by ID only.

## Appendix A — Empty label files

None.

## Appendix B — Malformed label files (full list)

None.

## Appendix C — Method notes

- Images are matched to labels by *exact* filename stem inside the same `batch/split`. Case-insensitive matches are reported separately because they break on Linux.
- A label file is counted as **empty** when it contains no non-blank lines. Whitespace-only files therefore count as empty.
- A line is **hard-invalid** if it does not have exactly five tokens, the class token is not a non-negative integer, a coordinate is not finite, any coordinate lies outside [0,1], width/height is <= 0, or the resulting box leaves the [0,1] frame by more than 1e-6.
- Tiny boxes, edge-touching boxes, extreme aspect ratios and duplicated lines are recorded as **warnings**; the line is still accepted.
- Class IDs are counted only from accepted lines; raw first-token counts are also captured in the JSON (`raw_class_tokens_by_batch`) for cross-checking.
- Mapping inference uses only stems that appear in two batches with an identical multiset of rounded coordinates; it is *evidence*, not proof, and is reported with support counts.
- Leakage detection normalises Roboflow names (`name_jpg.rf.<32 hex>`) to `name` and groups on that; it catches differently augmented copies of the same source image.
