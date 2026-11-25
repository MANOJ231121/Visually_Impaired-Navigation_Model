# Annotation Conflict Review

- **Dataset root:** `final batches`
- **Generated:** 2026-11-07T13:00:27
- **Mode:** read-only — no image, label, class ID, split, or batch was modified.
- **Inputs:** `verification_report.md`, `verification_summary.json`, and the original files.
- **Runtime:** 362.77 s

---

## 0. Executive summary

**Five findings matter most:**

1. **The 292 figure is not a usable leakage count — it fails in both directions.** Of the 292 cross-split Roboflow base-name groups, **174 are pure filename collisions** (same generic name, different picture), while **136 of the 222 byte-identical cross-split groups have members with different base names** and are therefore invisible to a name-based method. The reliable union of MD5 + perceptual matching is **236 cross-split source groups** (223 containing exact duplicates, 13 near-duplicate-only).
2. **The geometry mismatches are mostly one annotation convention, not random error.** Of the 119 same-count/all-unmatched pairs, 111 contain a single object; in 95 of them the smaller box is ≥90% contained in the larger (median containment 0.983), and 91 differ by more than 2× in area. Across batches, the tighter box belongs to Batch 1 in 81% of comparisons and to Batch 4 in only 15% — a systematic tight-vs-loose gradient.
3. **All 7 class-ID conflicts sit in five IDs** (4, 17, 18, 31, 33); 8 boxes are affected. No class ID can be corrected from the files alone.
4. **All 15 duplicate stems pair a JPEG with a PNG whose pixels and dimensions differ.** The existing label aligns better with the JPEG in **15/15** cases (edge-fit heuristic), so the PNG copies are effectively unlabelled.
5. **The 4 WEBP/GIF-only images are all single-frame and every box is inside the image.** Converting a copy to PNG is mechanical and label-preserving.

| Focus area | Count | Verdict |
|---|---|---|
| Class-ID conflicts (7 file pairs / box-level) | 8 | **Manual review** — no auto-change permitted |
| Geometry mismatches (file pairs) | 177 | **Manual review** — cannot know the correct box |
| Duplicate-stem pairs (different images, one label) | 15 | **Manual annotation** for the unlabelled twin |
| Images only in WEBP/GIF | 4 | **Automatically fixable** by conversion of a copy |
| Exact-duplicate groups crossing splits | 222 | **76 safe to dedupe**, 146 need a canonical-label decision |
| Source-image groups crossing splits (corrected) | 236 | **79 safe**, 157 need review |

**Nothing was changed.** All recommendations below are proposals pending your approval.

## 1. Method and scope

- **Images are matched by MD5 first.** Byte-identical images are guaranteed to show the same pixels, so *every* annotation difference on them is a labelling inconsistency, not a visual one.
- **Near-duplicates are detected with a 64-bit dHash over 8 dihedral transforms** (0/90/180/270° × mirror), applied only to images that share a Roboflow base name. This catches augmented copies and — critically — rejects filename collisions.
- **Box matching** is greedy by IoU (class-agnostic), then class equality is checked on each matched pair. `IoU >= 0.5` is treated as the same object.
- **Edge-fit** for duplicate stems is a heuristic: the fraction of the box-border pixels that coincide with strong image gradients. It hints which image a box was drawn on; it is **not** ground truth.
- **Thresholds:** same source if `dHash <= 10` and aspect-ratio ratio `<= 1.03`, or `dHash <= 6` and aspect ratio `<= 1.10`. `11–16` is flagged as *possible* and never auto-merged.

## 2. Class-ID conflicts (7 file pairs)

7 file pairs contain **8 box-level class conflicts**: the box geometry matches at IoU ≥ 0.5, but the class ID differs. They involve only **class IDs 4, 17, 18, 31, 33**. Every pair is a byte-identical (MD5) copy of the same source image, so the pixels cannot justify two different IDs.

| ID | A location | A file | B location | B file | Conflicting boxes |
|---|---|---|---|---|---|
| C01 | Batch 2/train | 7911_376.jpg | Batch 2/val | itmbghqk.jpg | 31 ↔ 33 @ IoU 0.8836; 33 ↔ 31 @ IoU 0.8243 |
| C02 | Batch 2/train | 7911_448.jpeg | Batch 3/train | image28.jpeg | 4 ↔ 33 @ IoU 0.5123 |
| C03 | Batch 2/train | 7911_461.jpeg | Batch 4/train | image45.jpeg | 4 ↔ 31 @ IoU 0.7604 |
| C04 | Batch 3/train | 5fg3vng1.jpg | Batch 3/train | t6d1wkdv.jpg | 18 ↔ 17 @ IoU 0.821 |
| C05 | Batch 3/train | cavill-grey-fabric-bed-frame-upholstered-double-laura-james-1.png | Batch 4/train | cavill-grey-fabric-bed-frame-upholstered-double-laura-james-1.png | 33 ↔ 18 @ IoU 0.8637 |
| C06 | Batch 3/train | image36.jpeg | Batch 4/train | image37.jpeg | 4 ↔ 31 @ IoU 0.5212 |
| C07 | Batch 3/train | image42.jpeg | Batch 4/train | image43.jpeg | 4 ↔ 31 @ IoU 0.5385 |

### 2.1 Structural pattern

| Class pair | Conflicting boxes |
|---|---|
| 4 ↔ 31 | 3 |
| 31 ↔ 33 | 2 |
| 4 ↔ 33 | 1 |
| 17 ↔ 18 | 1 |
| 18 ↔ 33 | 1 |

Every conflict is confined to the five IDs **4, 17, 18, 31, 33**. That is strong evidence that these five classes are **visually or semantically close** (or that two of them are duplicates). Which ID is correct **cannot be decided from the files** — it needs the authoritative class list and, ideally, the annotator's intent.

**Case-specific reading:**

- **C01 (31 ↔ 33) is a clean reciprocal swap.** Both objects exist in both labels; the IDs are exchanged between them. This is the signature of two class lists whose 31/33 entries are in opposite order, not of an annotator choosing the wrong object.
- **C05 (33 ↔ 18)** is the same bed photo exported in Batch 3 and Batch 4. The bed box is identical within 5%, so only the class assignment changed between exports.
- **C02, C06, C07 (4 ↔ 31/33)** and **C03 (4 ↔ 31)** pair a large ``4`` box with a smaller ``31``/``33`` box (IoU 0.51–0.76). This is consistent with a whole-object vs. component/type distinction, or with a renamed class between batches.
- **C04 (18 ↔ 17)** is the smallest semantic gap and stays inside one batch, so it is most likely a single mis-click or a duplicated class definition.

### 2.2 Visual comparisons

Each conflict has a three-panel image (A annotated, B annotated, overlay with B dashed) and a zoomed pair crop:

- **C01** — `visualizations/class_conflicts/C01_7911_376__itmbghqk.png`  ·  zoom: `visualizations/class_conflicts/C01_7911_376__itmbghqk_zoom.png`
- **C02** — `visualizations/class_conflicts/C02_7911_448__image28.png`  ·  zoom: `visualizations/class_conflicts/C02_7911_448__image28_zoom.png`
- **C03** — `visualizations/class_conflicts/C03_7911_461__image45.png`  ·  zoom: `visualizations/class_conflicts/C03_7911_461__image45_zoom.png`
- **C04** — `visualizations/class_conflicts/C04_5fg3vng1__t6d1wkdv.png`  ·  zoom: `visualizations/class_conflicts/C04_5fg3vng1__t6d1wkdv_zoom.png`
- **C05** — `visualizations/class_conflicts/C05_cavill-grey-fabric-bed-frame-upholstered-d__cavill-grey-fabric-bed-frame-upholstered-d.png`  ·  zoom: `visualizations/class_conflicts/C05_cavill-grey-fabric-bed-frame-upholstered-d__cavill-grey-fabric-bed-frame-upholstered-d_zoom.png`
- **C06** — `visualizations/class_conflicts/C06_image36__image37.png`  ·  zoom: `visualizations/class_conflicts/C06_image36__image37_zoom.png`
- **C07** — `visualizations/class_conflicts/C07_image42__image43.png`  ·  zoom: `visualizations/class_conflicts/C07_image42__image43_zoom.png`

### 2.3 Assessment

- **Automatically fixable:** none. Changing a class ID would invent ground truth.
- **Manual inspection:** all 7 pairs. A human must decide which ID describes the object, using the class definitions.
- **Cannot resolve without the owner:** the meaning of IDs 4, 17, 18, 31, 33 — and whether any of them are duplicate classes.

## 3. Bounding-box geometry mismatches (177 file pairs)

All 177 pairs are **byte-identical images** with differing boxes, so the pixels cannot explain the difference. Sub-patterns:

| Pattern | Pairs | Meaning |
|---|---|---|
| Same number of boxes, **none** matches at IoU 0.5 | 119 | Boxes were drawn around different extents (e.g. object vs. part of it) |
| Same number of boxes, **some** match | 10 | One or more objects annotated differently, others agree |
| Different number of boxes | 48 | An object is missing/extra in one of the two labels |

Severity distribution by best remaining IoU:

| Best unmatched IoU | Pairs | Interpretation |
|---|---|---|
| 0.30 – 0.50 | 101 | Near-miss: likely the same object, boundary disagreement |
| 0.10 – 0.30 | 33 | Weak overlap: box shifted or resized substantially |
| < 0.10 | 43 | No overlap: likely a different object or a missing/extra box |

### 3.1 What the differences actually are

- **111 of the 119 all-unmatched pairs contain a single box.** In every one of them the smaller box is at least half inside the larger; in **95** it is ≥90% inside (median containment **0.983**). **91** pairs differ by more than 2× in area. These are the **same object annotated at different tightness**, not different objects.
- **The tightness follows a batch gradient.** Among the **87 cross-batch single-box comparisons**, the tighter box belongs to:

| Batch | Comparisons | Tighter box | Share |
|---|---|---|---|
| Batch 1 | 52 | 42 | 81% |
| Batch 2 | 37 | 24 | 65% |
| Batch 3 | 32 | 13 | 41% |
| Batch 4 | 53 | 8 | 15% |

Batch 1 is systematically the tightest and Batch 4 the loosest. That is the signature of a **different annotation/export convention per batch**, not of random human error — which also means the mismatch is fixable by a single policy decision once the owner says whether boxes should be tight or loose.
- **The 8 multi-box all-unmatched pairs are not a block transform:** only 0 show a consistent delta across all boxes. Those need per-object review.
- **48 pairs differ in box count.** The most frequent unmatched classes are 4 (12 on the A side), 5 (11 on A / 5 on B), 8 (6 on A) and 14 (6 on A). This is a general recall difference, not one broken class.

### 3.2 Examples

The 36 most severe pairs have contact sheets in `visualizations/geometry_sheets/`. Every pair has an individual three-panel image in `visualizations/geometry/` named `Gxxxx_<fileA>__<fileB>.png`.

| ID | A file | B file | boxes A/B | matched ≥0.5 | unmatched | best unmatched IoU |
|---|---|---|---|---|---|---|
| G0124 | Batch 2/train/31_525.jpg | Batch 2/val/x6icumdq.jpg | 8/2 | 1 | 8 | 0.0 |
| G0040 | Batch 1/train/ufmae056.jpg | Batch 2/train/7911_670.jpg | 6/6 | 0 | 6 | 0.4922 |
| G0041 | Batch 2/train/7911_670.jpg | Batch 3/train/ufmae056.jpg | 6/6 | 0 | 6 | 0.4922 |
| G0075 | Batch 1/train/3501658e_.jpg | Batch 2/val/nj9u6ghj.jpg | 7/4 | 2 | 5 | 0.0456 |
| G0155 | Batch 3/train/89iqksu1.jpg | Batch 4/train/89iqksu1.jpg | 1/6 | 1 | 5 | 0.0 |
| G0165 | Batch 3/train/image24.png | Batch 4/train/image32.png | 1/4 | 0 | 5 | 0.0 |
| G0163 | Batch 3/train/image22.jpeg | Batch 3/train/image23.jpeg | 4/1 | 0 | 4 | 0.0155 |
| G0164 | Batch 3/train/image22.png | Batch 4/train/791881_357.png | 4/1 | 0 | 4 | 0.2305 |
| G0116 | Batch 1/val/03f73911_.jpg | Batch 3/test/34_130.jpg | 3/4 | 2 | 3 | 0.0 |
| G0148 | Batch 2/val/2rhni6w7.jpg | Batch 4/train/18_488.jpg | 4/2 | 1 | 3 | 0.4649 |
| G0067 | Batch 1/train/01466d66_.jpg | Batch 1/train/520375e6_.jpg | 3/1 | 0 | 3 | 0.048 |
| G0117 | Batch 1/val/a064c15c_.jpg | Batch 3/train/rehksnyg.jpg | 3/1 | 0 | 3 | 0.022 |

### 3.3 Assessment

- **Automatically fixable:** none. The *convention* can be standardized, but choosing which of the two boxes is right is a semantic decision.
- **Manual inspection:** all 177. Prioritise (a) the 48 count mismatches, (b) the 8 multi-box all-unmatched pairs, then (c) the single-box tightness pairs (which share one root cause).
- **Cannot resolve without the owner:** whether boxes should be tight (object boundary) or loose (full extent / context), and which batch's convention is authoritative.

## 4. Duplicate-stem image pairs (15)

In each case **two different images share one filename stem** in the same split, so only one `.txt` can exist. All 15 pairs are confirmed to be *different images* (MD5 and perceptual hash differ).

| ID | Location | Stem | Images | Sizes | dHash | Edge-fit | Better edge-fit |
|---|---|---|---|---|---|---|---|
| D01 | Batch 3/train | image22 | Batch 3/images/train/image22.jpeg / Batch 3/images/train/image22.png | 194×259 / 194×259 | 31 | 0.285 / 0.127 | image22.jpeg |
| D02 | Batch 3/train | image24 | Batch 3/images/train/image24.jpeg / Batch 3/images/train/image24.png | 225×225 / 225×224 | 28 | 0.425 / 0.049 | image24.jpeg |
| D03 | Batch 3/val | image31 | Batch 3/images/val/image31.jpeg / Batch 3/images/val/image31.png | 225×225 / 225×225 | 20 | 0.345 / 0.188 | image31.jpeg |
| D04 | Batch 3/val | image33 | Batch 3/images/val/image33.jpeg / Batch 3/images/val/image33.png | 153×330 / 225×225 | 24 | 0.234 / 0.059 | image33.jpeg |
| D05 | Batch 4/train | image30 | Batch 4/images/train/image30.jpeg / Batch 4/images/train/image30.png | 275×183 / 225×224 | 24 | 0.257 / 0.021 | image30.jpeg |
| D06 | Batch 4/train | image32 | Batch 4/images/train/image32.jpeg / Batch 4/images/train/image32.png | 285×177 / 225×224 | 24 | 0.249 / 0.080 | image32.jpeg |
| D07 | Batch 4/train | image35 | Batch 4/images/train/image35.jpeg / Batch 4/images/train/image35.png | 196×257 / 225×225 | 13 | 0.511 / 0.028 | image35.jpeg |
| D08 | Batch 4/train | image37 | Batch 4/images/train/image37.jpeg / Batch 4/images/train/image37.png | 225×225 / 225×225 | 19 | 0.244 / 0.018 | image37.jpeg |
| D09 | Batch 4/train | image38 | Batch 4/images/train/image38.jpeg / Batch 4/images/train/image38.png | 225×225 / 225×225 | 17 | 0.237 / 0.035 | image38.jpeg |
| D10 | Batch 4/train | image39 | Batch 4/images/train/image39.jpeg / Batch 4/images/train/image39.png | 199×254 / 225×225 | 28 | 0.451 / 0.010 | image39.jpeg |
| D11 | Batch 4/train | image43 | Batch 4/images/train/image43.jpeg / Batch 4/images/train/image43.png | 225×225 / 255×198 | 22 | 0.304 / 0.117 | image43.jpeg |
| D12 | Batch 4/train | image45 | Batch 4/images/train/image45.jpeg / Batch 4/images/train/image45.png | 225×225 / 261×193 | 11 | 0.177 / 0.035 | image45.jpeg |
| D13 | Batch 4/train | image46 | Batch 4/images/train/image46.jpeg / Batch 4/images/train/image46.png | 210×240 / 313×161 | 24 | 0.295 / 0.175 | image46.jpeg |
| D14 | Batch 4/train | image47 | Batch 4/images/train/image47.jpeg / Batch 4/images/train/image47.png | 225×225 / 252×200 | 20 | 0.202 / 0.032 | image47.jpeg |
| D15 | Batch 4/train | image48 | Batch 4/images/train/image48.jpeg / Batch 4/images/train/image48.png | 201×251 / 364×138 | 25 | 0.275 / 0.178 | image48.jpeg |

**Pattern:** in **15/15** pairs the edge-fit score favours the JPEG (the PNG score is near zero in most cases). The dHash distances are 11–31 and the dimensions often differ, so the PNG is neither a byte copy nor a trivially resized copy; it is most likely a resized/derived variant, but the files cannot prove that. The existing label belongs to the JPEG in every case, leaving the PNG with no annotation of its own.

Visuals: `visualizations/duplicate_stems/`. Each shows both images with the *same* label drawn on top, plus the edge-fit score.

### 4.1 Assessment

- **Automatically fixable (structural only):** rename the second image so each image can have its own label. This does **not** create the missing annotation.
- **Manual inspection:** decide which image the existing label was drawn for (the edge-fit column is a hint), then either annotate the other image or exclude it.
- **Cannot resolve without the owner:** whether both images are intended to be in the dataset. If they are near-duplicates with one label, the honest options are annotate or drop — never reuse the same box blindly.

## 5. Images available only as WEBP / GIF (4)

| ID | Location | Format | Size | Frames | Mode | Boxes | Boxes in bounds | Ultralytics |
|---|---|---|---|---|---|---|---|---|
| F01 | Batch 3/images/train/15_9f2628f7-0a6e-40db-a06a-5aa1cc95dee2.webp | .webp | 800×800 | 1 | RGB | 2 | yes | yes |
| F02 | Batch 3/images/train/A9674866481b64aa5828c862fd0bdf314T.jpg_640x640.jpg.webp | .webp | 480×640 | 1 | RGB | 1 | yes | yes |
| F03 | Batch 3/images/train/ADA-1025-Blue-Mens-Restroom-Wall-Sign-Male-and-ISA-pictos-6x8_580x.gif | .gif | 580×761 | 1 | P | 1 | yes | **NO** |
| F04 | Batch 3/images/train/DSC_3857copy.webp | .webp | 827×1267 | 1 | RGB | 1 | yes | yes |

Visuals: `visualizations/format_only/`.

All four are **single-frame** images and every box is inside the frame, so a PNG conversion loses no annotation information. The GIF uses palette mode (`P`); converting it to RGB PNG is sufficient.

### 5.1 Assessment

- **Automatically fixable:** yes. Copy each file and convert to PNG (for the GIF, take its first frame). Keep the original untouched. This changes no annotation and no class ID.
- **Manual inspection:** confirm the GIF is not animated and that the conversion preserves the frame the label was drawn on.
- **Cannot resolve without the owner:** whether these four images belong in the final dataset.

## 6. Exact-duplicate groups crossing splits (222)

Of the 727 MD5 groups, **222 contain images from more than one split**. They involve 475 physical files.

| Annotation status | All groups | Cross-split groups | Action |
|---|---|---|---|
| Byte-identical labels | 197 | 76 | Safe to keep one copy and assign a single split |
| Minor boundary variance (IoU ≥ 0.9) | 105 | 25 | Low risk; pick one canonical label, document the choice |
| Geometry variance (IoU 0.5–0.9) | 275 | 83 | Manual: choose the better box |
| Strong mismatch (IoU < 0.5) | 96 | 23 | Manual: annotations disagree fundamentally |
| Different box count | 47 | 14 | Manual: object missing/extra |
| Class conflict | 7 | 1 | Manual: class IDs disagree (do not auto-pick) |

Full list with locations, category, worst IoU and every file: `duplicate_group_actions.csv`.

**Deduplication potential:** removing all but one file per MD5 group would delete 818 redundant files (727 groups). Of these, 253 are in cross-split groups. **This is not proposed as automatic**: only the byte-identical-label subset is safe without a review.

### 6.1 Assessment

- **Automatically fixable:** dedupe the **byte-identical-label** groups (197 overall / 76 cross-split), keeping one copy. The split assignment needs a documented group-aware policy.
- **Manual inspection:** every group with any annotation difference (530 overall).
- **Cannot resolve without the owner:** which split a shared source image should belong to, and which annotation is canonical when copies disagree.

## 7. Source-image groups crossing splits — corrected count

### 7.1 The filename-collision problem

The earlier report grouped images by Roboflow base name and reported **292 cross-split groups**. Re-checking the pictures shows that the base-name heuristic fails in **two independent directions**, so 292 cannot be corrected by simple subtraction:

1. **False positives — generic filenames.** Of the 292 cross-split base-name groups, **174 are pure collisions**: `images13`, `images19`, `images112`, `image43`, numeric names, etc. Same name, unrelated picture. Only **103** are pure source groups and **15** are mixed.
2. **False negatives — renamed copies.** **136 of the 222 byte-identical cross-split groups contain files with different base names** (e.g. `7911_376.jpg` and `itmbghqk.jpg`), so a name-based method cannot represent them at all.

Globally, every multi-file Roboflow base-name group was checked perceptually. Of **683** such groups:

| Base-name group outcome | Groups |
|---|---|
| **Pure collision** — name matches, pictures are unrelated | 357 |
| **Pure source group** — all files are the same source image | 300 |
| **Mixed** — some files match, others only share the name | 26 |

- Base-name groups with **no** perceptual match: **298** (minimum dHash > 16).

| Base name | Files | Min dHash | Files (truncated) |
|---|---|---|---|
| `images427` | 2 | 21 | images427.jpg · images427.jpg |
| `images85` | 2 | 20 | images85.jpg · images85.jpg |
| `images90` | 3 | 21 | images90.jpg · images90.jpg · images90.jpg |
| `images11` | 2 | 24 | images11.jpg · images11.jpg |
| `images110` | 3 | 23 | images110.jpg · images110.jpg · images110.jpg |
| `images114` | 3 | 23 | images114.jpg · images114.jpg · images114.jpg |
| `images116` | 2 | 17 | images116.jpg · images116.jpg |
| `images119` | 2 | 30 | images119.jpg · images119.jpg |
| `images12` | 3 | 18 | images12.jpg · images12.jpg · images12.jpg |
| `images124` | 2 | 23 | images124.jpg · images124.jpg |
| `images125` | 3 | 23 | images125.jpg · images125.jpg · images125.jpg |
| `images126` | 2 | 28 | images126.jpg · images126.jpg |

A dHash distance near 32 of 64 is the random-image baseline; the earlier base-name grouping would have produced many false leakage alarms.

### 7.2 Corrected source grouping

After combining **exact MD5 equality** with **perceptual matching**, the dataset has **762 source-image groups** with more than one file. **236 of them cross splits** and **615 cross batches**.

Of the 236 cross-split source groups, **223 contain exact duplicates** (these include all 222 MD5 groups found earlier, sometimes combined into one larger source group) and **13 are near-duplicate-only** (dHash match, different bytes). **236 is therefore the reliable leakage count** — not 292 (base-name groups), and not 222 (MD5 only).

| Annotation status | Groups | Cross-split groups | Action |
|---|---|---|---|
| Byte-identical labels | 201 | 79 | Safe to merge into one split unit |
| Minor boundary variance | 113 | 29 | Low risk; document canonical label |
| Geometry variance | 282 | 85 | Manual: choose box |
| Strong mismatch | 109 | 27 | Manual |
| Different box count | 48 | 15 | Manual |
| Class conflict | 9 | 1 | Manual: do not auto-pick an ID |

Full list: `leakage_group_actions.csv`. Each group carries every member file, its locations, and whether it contains an exact duplicate.

**Recommended split unit:** the source group, not the file. A group-aware split guarantees that no source image appears in two splits. This is a mechanical, label-preserving operation; the only open choice is the split ratio, which is a policy decision.

### 7.3 Assessment

- **Automatically fixable:** group-aware re-splitting (no label change) — automatically computable, but it will change split membership.
- **Manual inspection:** groups whose members disagree on annotations, and the 11–16 dHash band (possible leakage).
- **Cannot resolve without the owner:** desired train/val/test ratio and whether the original split must be preserved.

## 8. Fix classification

### 8.1 Automatically fixable (no semantic guessing)

| Issue | Count | Proposed mechanical action |
|---|---|---|
| WEBP/GIF-only images | 4 | Copy + convert to PNG (GIF: first frame); keep originals |
| Exact-duplicate groups with byte-identical labels | 197 | Keep one copy per group; record the mapping |
| Source groups with byte-identical labels | 201 | Collapse to one split unit; record the mapping |
| Group-aware re-split (all groups) | 762 | Assign whole source groups to splits with a seeded ratio |

### 8.2 Requires manual inspection

| Issue | Count | Why a human is needed |
|---|---|---|
| Class-ID conflicts | 8 | Cannot know which ID is correct |
| Geometry mismatches | 177 | Cannot know which box is correct |
| Duplicate stems | 15 | Must decide which image was annotated / annotate the other |
| Exact groups with differing labels | 530 | Canonical annotation must be chosen |
| Source groups with differing labels | 561 | Canonical annotation must be chosen |
| Possible leakage (dHash 11–16) | 71 | Verify the images are really the same source |

### 8.3 Cannot be resolved without dataset-owner information

| Question | Why the files cannot answer it |
|---|---|
| Authoritative class list (id → name) | No metadata exists anywhere in or beside the dataset |
| Are IDs 4 / 17 / 18 / 31 / 33 distinct classes? | The 7 conflicts cluster on these IDs; names are unknown |
| Which annotation is canonical when copies disagree? | Both annotations are plausible; no tie-breaker in the data |
| Should duplicate source images be kept (oversampling)? | Depends on the intended data design |
| Desired split ratio / split policy | Not encoded in the dataset |
| Do the 4 WEBP/GIF images belong in the final dataset? | No provenance information |
| Are the 15 duplicate-stem twins both intended? | No provenance information |

## 9. Proposed remediation plan (pending your approval)

Ordered so that nothing semantic is guessed:

1. **Freeze the owner's class list and `data.yaml`.** No class ID will be changed before this.
2. **Copy the dataset to a new working directory.** The original `final batches/` stays read-only forever.
3. **Convert the 4 WEBP/GIF images** to PNG in the copy and verify the annotations visually.
4. **Deduplicate only the byte-identical-label groups**, keeping one file and logging every removal in a CSV.
5. **Resolve the 15 duplicate stems**: choose the image the label fits, then annotate or drop the twin (owner/annotator decision).
6. **Review the 7 class conflicts and 177 geometry mismatches** using the generated visuals; record a decision per case in a signed-off table.
7. **Re-split by source group** (MD5 + perceptual), not by file, using a documented seed.
8. **Re-run `verify_dataset.py` and this review** on the cleaned copy; require zero cross-split source groups, zero duplicate stems, and zero conflicts before training.

## 10. Limitations

- **dHash is not a proof.** Flips/rotations are handled, but heavy crop, blur or occlusion can raise the distance; 11–16 is explicitly left as *possible* rather than merged.
- **Edge-fit is a heuristic.** It highlights which image a box aligns with; it does not establish the correct annotation.
- **No class names exist.** Every statement in this report is about numeric IDs only.
- **No files were changed.** The visualisations are derived copies stored outside the dataset.

## Appendix A — Output inventory

| File | Contents |
|---|---|
| `annotation_conflicts.csv` | 7 class-conflict pairs + 177 geometry pairs, with area-ratio and containment diagnostics |
| `annotation_conflicts.json` | Structured version of every finding |
| `duplicate_stem_pairs.csv` | 15 pairs with sizes, hashes, edge-fit |
| `format_only_images.csv` | 4 WEBP/GIF-only images |
| `duplicate_group_actions.csv` | 727 MD5 groups with category and action |
| `leakage_group_actions.csv` | 762 source groups (corrected leakage list) |
| `possible_source_pairs.csv` | 71 pairs in the 11–16 dHash band |
| `visualizations/class_conflicts/` | 3-panel comparisons + zoomed crops |
| `visualizations/geometry/` | 177 three-panel comparisons |
| `visualizations/geometry_sheets/` | contact sheets for the 36 most severe pairs |
| `visualizations/duplicate_stems/` | same-label-on-both-images comparisons |
| `visualizations/format_only/` | previews with annotations |

## Appendix B — What was NOT done

- No class ID was changed.
- No image or label was deleted, moved, renamed or overwritten.
- No batch was merged and no model was trained.
- No class name was invented.
