# Dataset Preparation Plan — DRY RUN

- **Dataset root:** `final batches`
- **Analysis inputs:** `dataset_verification`
- **Plan output:** `dataset_preparation_plan`
- **Proposed prepared root (NOT created):** `prepared_dataset`
- **Generated:** 2026-11-07T15:17:21
- **Mode:** DRY RUN — no image, label, class ID, split, or batch was modified.

---

## 0. Executive summary

The dataset has **16,122 images**, **16,107 labels** and **23,043 boxes**. Every numeric class ID is preserved (IDs **0–34**, 35 IDs).

The plan proposes only mechanical, reversible actions for the prepared copy:

| Proposed action | Count | Needs approval |
|---|---|---|
| Convert WEBP/GIF images to PNG (copy + label verbatim) | 4 | no |
| Resolve duplicate-stem pairs (keep edge-fit winner, quarantine twin) | 15 | yes |
| Omit byte-identical copies inside one split (119 groups) | 121 | no |
| Defer byte-identical copies that cross splits | 76 | yes |
| Defer byte-identical copies blocked by a broader annotation conflict | 2 | yes |
| Flag exact-duplicate annotation conflicts | 530 | yes |
| Flag broader source-group annotation conflicts | 561 | yes |
| Flag class-conflict pairs | 7 | yes |
| Flag geometry-conflict pairs | 177 | yes |
| Flag cross-split source groups | 236 | yes |
| Flag possible near-duplicate pairs (dHash 11–16) | 71 | yes |

**Nothing has been executed.** The script only reads the dataset and writes the four plan files in `dataset_preparation_plan/`. The proposed prepared dataset does not exist yet.

## 1. Inputs and method

- **Primary input:** `annotation_conflicts.json` (the conflict review) and `verification_summary.json`.
- **Corrected source grouping:** `leakage_group_actions.csv`. This is the **762-group MD5 + perceptual** grouping, of which **236 cross splits**. The earlier filename-only 292 figure is **not** used.
- **Byte-identical groups:** `duplicate_group_actions.csv` / `exact_groups` (727 groups; 222 cross-split).
- **Exact duplicate labels:** a group is deduplicable only when every member label is byte-identical (`identical_label_bytes`), the group stays inside one split, and it is not part of a broader source group with a conflicting annotation. Otherwise the group is deferred for manual review.
- **Conflict policy:** any group whose annotations differ is flagged for manual review; no box or class ID is changed by this plan.
- **Action precedence per file:** duplicate-stem quarantine > PNG conversion > annotation-conflict review > exact-duplicate omission > cross-split review > normal copy.

## 2. Issues found

| Issue | Count | Detail |
|---|---|---|
| Images only in WEBP/GIF | 4 | 3 WebP + 1 GIF, all Batch 3/train |
| Duplicate-stem pairs | 15 | Two different images share one stem and one label |
| Exact-duplicate groups (MD5) | 727 | 222 cross-split |
|   – byte-identical labels | 197 | 76 cross-split |
|   – conflicting annotations | 530 | 146 cross-split |
| Source-image groups (corrected) | 762 | 236 cross-split; 561 have conflicting annotations |
| Class-ID conflict pairs | 7 | 8 box-level conflicts in 5 class IDs |
| Geometry mismatch pairs | 177 | Byte-identical images, different boxes |
| Possible near-duplicate pairs | 71 | dHash 11–16; never auto-merged |

### 2.1 Exact-duplicate categories

| Category | All groups | Cross-split |
|---|---|---|
| identical_label_bytes | 197 | 76 |
| minor_boundary_variance | 105 | 25 |
| geometry_variance | 275 | 83 |
| strong_mismatch | 96 | 23 |
| count_mismatch | 47 | 14 |
| class_conflict | 7 | 1 |

### 2.2 Corrected source-group categories

| Category | All groups | Cross-split |
|---|---|---|
| identical_label_bytes | 201 | 79 |
| minor_boundary_variance | 113 | 29 |
| geometry_variance | 282 | 85 |
| strong_mismatch | 109 | 27 |
| count_mismatch | 48 | 15 |
| class_conflict | 9 | 1 |

## 3. Proposed actions

### 3.1 WEBP / GIF → PNG

These four images are copied to PNG and their labels are copied verbatim. The originals stay untouched.

| Issue | Source image | Format | Proposed PNG destination |
|---|---|---|---|
| CONV-001 | Batch 3/images/train/15_9f2628f7-0a6e-40db-a06a-5aa1cc95dee2.webp | webp | prepared_dataset/Batch 3/images/train/15_9f2628f7-0a6e-40db-a06a-5aa1cc95dee2.png |
| CONV-002 | Batch 3/images/train/A9674866481b64aa5828c862fd0bdf314T.jpg_640x640.jpg.webp | webp | prepared_dataset/Batch 3/images/train/A9674866481b64aa5828c862fd0bdf314T.jpg_640x640.jpg.png |
| CONV-003 | Batch 3/images/train/ADA-1025-Blue-Mens-Restroom-Wall-Sign-Male-and-ISA-pictos-6x8_580x.gif | gif | prepared_dataset/Batch 3/images/train/ADA-1025-Blue-Mens-Restroom-Wall-Sign-Male-and-ISA-pictos-6x8_580x.png |
| CONV-004 | Batch 3/images/train/DSC_3857copy.webp | webp | prepared_dataset/Batch 3/images/train/DSC_3857copy.png |

### 3.2 Duplicate-stem pairs

In every case the edge-fit heuristic favours the JPEG. The plan keeps that JPEG with the existing label and quarantines the PNG twin. **This is a recommendation requiring approval**, because edge-fit is a heuristic, not proof.

| Issue | Stem | Kept (edge-fit) | Quarantined twin | Approval |
|---|---|---|---|---|
| STEM-001 | image22 | image22.jpeg | image22.png | yes |
| STEM-002 | image24 | image24.jpeg | image24.png | yes |
| STEM-003 | image31 | image31.jpeg | image31.png | yes |
| STEM-004 | image33 | image33.jpeg | image33.png | yes |
| STEM-005 | image30 | image30.jpeg | image30.png | yes |
| STEM-006 | image32 | image32.jpeg | image32.png | yes |
| STEM-007 | image35 | image35.jpeg | image35.png | yes |
| STEM-008 | image37 | image37.jpeg | image37.png | yes |
| STEM-009 | image38 | image38.jpeg | image38.png | yes |
| STEM-010 | image39 | image39.jpeg | image39.png | yes |
| STEM-011 | image43 | image43.jpeg | image43.png | yes |
| STEM-012 | image45 | image45.jpeg | image45.png | yes |
| STEM-013 | image46 | image46.jpeg | image46.png | yes |
| STEM-014 | image47 | image47.jpeg | image47.png | yes |
| STEM-015 | image48 | image48.jpeg | image48.png | yes |

### 3.3 Exact duplicates with byte-identical labels

| Decision | Groups | Copies | Action |
|---|---|---|---|
| Within one split | 121 | 123 | 121 proposed omissions across 119 groups; 2 deferred because the broader source group conflicts |
| Crosses splits | 76 | 76 | Deferred: keep all copies until the split policy is approved |
| Blocked by broader annotation conflict | — | 2 | Deferred: the source group has a conflicting annotation |

Total: **197 byte-identical groups**, **199 redundant copies**. Only **121** are proposed for omission now; **78** are deferred.

### 3.4 Groups with conflicting annotations

| Category | Groups | Proposed action |
|---|---|---|
| Byte-identical exact groups | 530 | Manual review; keep files; no dedupe and no edit |
| Broader source groups | 561 | Manual review; keep files; choose canonical annotation |
| Class-conflict pairs | 7 | Manual review; choose the correct class ID |
| Geometry-conflict pairs | 177 | Manual review; choose the canonical box / add missing object |

The exact groups are a byte-identical subset of the broader source groups, so the two rows overlap. Both are listed because they answer different questions: *same bytes, different label* vs. *same source image, different label*.

### 3.5 Cross-split source-image groups

**236** corrected source groups cross splits. **79** have byte-identical labels and **157** have conflicting annotations. All are kept unchanged in this plan; the later split step must assign each whole group to one split.

### 3.6 Possible near-duplicate pairs

**71** pairs have dHash 11–16. They are flagged for visual confirmation and are **not** merged or deduplicated.

## 4. Destination layout

The prepared copy would mirror the original structure under the proposed prepared root:

```text
prepared_dataset/
├── Batch <n>/images/<split>/<file>        # copied images
├── Batch <n>/labels/<split>/<stem>.txt    # copied labels (verbatim)
└── _review/unlabelled_duplicate_stems/    # quarantined duplicate-stem twins
```

No destination is inside the original dataset. Every destination path is unique, and the plan does not create or overwrite any file.

## 5. Validation

| Check | Result |
|---|---|
| Every referenced source file exists | PASS |
| Every image has a label | PASS |
| Every label has an image | PASS |
| Image count matches verification summary | PASS |
| Label count matches verification summary | PASS |
| Box count matches verification summary | PASS |
| Class IDs match verification summary | PASS |
| No label action modifies content | PASS |
| No destination collision | PASS |
| No destination inside the original root | PASS |
| No destination equals an original path | PASS |
| No destination already exists | PASS |
| All conflicting annotations require approval | PASS |
| Corrected source grouping used | PASS |
| Source-group count matches leakage CSV | PASS |
| Dry-run only; originals untouched | PASS |

- Class IDs: **[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34]** (35 IDs, no change).
- Total boxes: **23,043**.
- Corrected source groups: **762** total / **236** cross-split.
- Manifest entries: **32,229** (16,122 images + 16,107 labels).
- Label content operations: `{'verbatim_copy': 15986, 'omitted': 121}`.

## 6. Risks and unresolved decisions

### 6.1 Risks

- **Edge-fit is a heuristic.** The duplicate-stem recommendation (keep JPEG, quarantine PNG) is based on border-gradient alignment, not on annotator intent. A human must verify each of the 15 pairs.
- **Cross-split deduplication is deferred.** Removing a byte-identical copy from another split changes split membership, which is explicitly out of scope until the split policy is approved.
- **Broader source groups can block exact dedupe.** Two within-split byte-identical copies are deferred because their source group contains a conflicting annotation.
- **PNG conversion of the GIF** takes the first frame. The file is single-frame according to the analysis, but a human should confirm the converted frame matches the label.
- **No class names exist.** All statements are about numeric IDs only.

### 6.2 Unresolved decisions (cannot be answered from the files)

1. Authoritative class list (`id -> name`).
2. Correct class ID for the 7 class-conflict pairs.
3. Canonical box for the 177 geometry-conflict pairs.
4. Canonical annotation for the 530 exact / 561 source conflict groups.
5. Whether both images of each duplicate-stem pair belong in the dataset.
6. Desired train/validation/test ratio and whether the original splits must be preserved.
7. Whether the 4 WEBP/GIF images belong in the final dataset.
8. Whether duplicate source images should be kept for oversampling.

## 7. Approval gate

Approval is required before any action is implemented. Once the plan as a whole is approved, the mechanical subset (4 conversions + 121 within-split dedupe omissions) may be executed first; the 15 duplicate stems, 530 exact conflicts, 561 source conflicts, 7 class conflicts, 177 geometry conflicts and 236 cross-split groups still need their own sign-off.

**Do not execute until each of these is signed off:**

- the class list and canonical `data.yaml`;
- the 15 duplicate-stem choices;
- the canonical annotation for every conflict group;
- the split ratio / group-aware split policy;
- whether the 4 WEBP/GIF images stay in the dataset.

## 8. Exact command

```powershell
python "prepare_dataset.py"
```

Outputs are written only to:

`dataset_preparation_plan`

## Appendix A — Output inventory

| File | Contents |
|---|---|
| `preparation_plan.md` | This report |
| `preparation_actions.csv` | 1798 issue-level proposed actions |
| `preparation_summary.json` | Counts, proposed actions, validation results |
| `preparation_manifest.json` | 32,229 file-level entries (one per image and label) |

## Appendix B — Strict rules respected

- Original dataset not modified.
- No original file deleted, renamed, moved or overwritten.
- No class ID changed.
- No class name invented.
- No bounding box auto-corrected.
- No class conflict auto-resolved.
- No model trained.
- No split membership changed.
- No object-category assumption made.
