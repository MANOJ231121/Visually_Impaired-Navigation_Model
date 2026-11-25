# Dataset Staging Copy — Report

- **Source:** `final batches`
- **Destination:** `prepared_dataset`
- **Generated:** 2026-11-07T22:42:40
- **Mode:** copy + format conversion only; no training, no splitting.

## 1. What was done

- Created the destination directory.
- Copied **all 16,122 original images** and **all 16,107 original labels**, preserving Batch/split structure.
- Created **4** PNG copies of the WEBP/GIF images. The original WEBP/GIF files were also copied, so no image is omitted.
- Result: **16,126 images** and **16,107 labels** in the staging copy.
- Preserved all duplicate images, duplicate-stem pairs, conflicting annotations and original split membership.

## 2. Conversions

| ID | Source | Format | Dimensions | PNG destination | Pixels preserved | Annotation unchanged | Status |
|---|---|---|---|---|---|---|---|
| CONV-001 | Batch 3/images/train/15_9f2628f7-0a6e-40db-a06a-5aa1cc95dee2.webp | JPEG | 800x800 | Batch 3/images/train/15_9f2628f7-0a6e-40db-a06a-5aa1cc95dee2.png | True | True | verified |
| CONV-002 | Batch 3/images/train/A9674866481b64aa5828c862fd0bdf314T.jpg_640x640.jpg.webp | JPEG | 480x640 | Batch 3/images/train/A9674866481b64aa5828c862fd0bdf314T.jpg_640x640.jpg.png | True | True | verified |
| CONV-003 | Batch 3/images/train/ADA-1025-Blue-Mens-Restroom-Wall-Sign-Male-and-ISA-pictos-6x8_580x.gif | GIF | 580x761 | Batch 3/images/train/ADA-1025-Blue-Mens-Restroom-Wall-Sign-Male-and-ISA-pictos-6x8_580x.png | True | True | verified |
| CONV-004 | Batch 3/images/train/DSC_3857copy.webp | JPEG | 827x1267 | Batch 3/images/train/DSC_3857copy.png | True | True | verified |

Each converted image reuses the same stem-based label file, copied verbatim. No annotation content was changed.

Detected source encodings: GIF, JPEG. Three files with a `.webp` extension actually contain JPEG data; the `.gif` is a genuine GIF. All four were decoded and re-encoded losslessly as PNG.

## 3. Duplicate stems

**15** pre-existing duplicate-stem pairs were preserved exactly: both images copied, the single shared label copied once. This is not resolved and is listed in `unresolved_issues.csv`.

## 4. Validation

| Check | Result | Value |
|---|---|---|
| Source / destination image counts | PASS | 16,122 source images + 4 conversions = 16,126 |
| Source / destination label counts | PASS | 16,107 = 16,107 |
| All source images present | PASS |  |
| All source labels present | PASS |  |
| Converted PNGs present | PASS |  |
| No unexpected files | PASS |  |
| No silent omissions | PASS |  |
| Label bytes unchanged | PASS |  |
| Class IDs unchanged | PASS | IDs 0-34 (35) |
| Bounding-box coordinates unchanged | PASS |  |
| All destination images decodable | PASS | 16,126 decoded, 0 failures |
| Converted dimensions preserved | PASS |  |
| Converted pixels preserved | PASS |  |
| Converted annotations unchanged | PASS |  |
| Converted boxes in-bounds (informational) | PASS | 0 pre-existing out-of-bounds label(s) |
| No original file modified | PASS |  |
| No overwrites | PASS | 0 |

## 5. Unresolved issues

| Issue | Category | Count | Blocking |
|---|---|---|---|
| STAGE-DUPSTEM-D01 | duplicate_stem_shared_label | 1 | yes |
| STAGE-DUPSTEM-D02 | duplicate_stem_shared_label | 1 | yes |
| STAGE-DUPSTEM-D03 | duplicate_stem_shared_label | 1 | yes |
| STAGE-DUPSTEM-D04 | duplicate_stem_shared_label | 1 | yes |
| STAGE-DUPSTEM-D05 | duplicate_stem_shared_label | 1 | yes |
| STAGE-DUPSTEM-D06 | duplicate_stem_shared_label | 1 | yes |
| STAGE-DUPSTEM-D07 | duplicate_stem_shared_label | 1 | yes |
| STAGE-DUPSTEM-D08 | duplicate_stem_shared_label | 1 | yes |
| STAGE-DUPSTEM-D09 | duplicate_stem_shared_label | 1 | yes |
| STAGE-DUPSTEM-D10 | duplicate_stem_shared_label | 1 | yes |
| STAGE-DUPSTEM-D11 | duplicate_stem_shared_label | 1 | yes |
| STAGE-DUPSTEM-D12 | duplicate_stem_shared_label | 1 | yes |
| STAGE-DUPSTEM-D13 | duplicate_stem_shared_label | 1 | yes |
| STAGE-DUPSTEM-D14 | duplicate_stem_shared_label | 1 | yes |
| STAGE-DUPSTEM-D15 | duplicate_stem_shared_label | 1 | yes |
| STAGE-CONV-DUP-CONV-001 | original_and_png_copy_both_staged | 1 | yes |
| STAGE-CONV-DUP-CONV-002 | original_and_png_copy_both_staged | 1 | yes |
| STAGE-CONV-DUP-CONV-003 | original_and_png_copy_both_staged | 1 | yes |
| STAGE-CONV-DUP-CONV-004 | original_and_png_copy_both_staged | 1 | yes |
| STAGE-CONFLICT-EXACT | exact_annotation_conflict | 530 | yes |
| STAGE-CONFLICT-SOURCE | source_annotation_conflict | 561 | yes |
| STAGE-CONFLICT-CLASS | class_id_conflict_pairs | 7 | yes |
| STAGE-CONFLICT-GEOMETRY | geometry_conflict_pairs | 177 | yes |
| STAGE-CROSS-SPLIT | cross_split_source_groups | 236 | yes |
| STAGE-NEAR-DUP | possible_near_duplicate_pairs | 71 | no |

Full detail: `unresolved_issues.csv`.

## 6. Output files

| File | Contents |
|---|---|
| `staging_report.md` | This report |
| `staging_manifest.csv` | 32,233 file entries |
| `staging_summary.json` | Counts, conversions, validation |
| `conversion_log.csv` | 4 conversion records |
| `unresolved_issues.csv` | 25 unresolved issue records |

## 7. Exact command

```powershell
# Initial staging copy (only runs when the destination is empty):
python "stage_dataset.py"

# Re-verify the existing staging copy without copying or overwriting:
python "stage_dataset.py" --validate-only
```

## 8. Not done (by rule)

- No model training.
- No train/validation/test changes.
- No image omitted or quarantined.
- No class ID, box coordinate or annotation changed.
- No conflict resolved.
- No class name invented.

**All automated checks passed: True.**
