import hashlib
import os
import shutil
import sys
from pathlib import Path
from PIL import Image

PROJECT_ROOT = Path(r"C:\Users\DELL\.gemini\antigravity\scratch\Object-Detection-Model")
BATCHES_ROOT = PROJECT_ROOT / "final batches"
DATASET_DIR = PROJECT_ROOT / "dataset"

def hash_file(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()

def main():
    print("Initializing clean dataset build with leakage prevention...", flush=True)

    # Prepare directories
    for split in ["train", "val", "test"]:
        (DATASET_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (DATASET_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    # 1. First gather and process TEST split
    print("\n--- Processing TEST split ---", flush=True)
    test_hashes = set()
    test_count = 0
    test_boxes = 0

    batches = sorted([b for b in BATCHES_ROOT.iterdir() if b.is_dir()])
    for b in batches:
        b_prefix = f"{b.name.lower().replace(' ', '')}_"
        img_dir = b / "images" / "test"
        lbl_dir = b / "labels" / "test"
        if not img_dir.exists(): continue

        for img in sorted(img_dir.glob("*")):
            if img.suffix.lower() not in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
                continue
            h = hash_file(img)
            if h in test_hashes:
                continue
            test_hashes.add(h)

            dest_img_name = f"{b_prefix}{img.name}"
            dest_img_path = DATASET_DIR / "images" / "test" / dest_img_name
            shutil.copy2(img, dest_img_path)

            dest_lbl_name = f"{b_prefix}{img.stem}.txt"
            dest_lbl_path = DATASET_DIR / "labels" / "test" / dest_lbl_name
            src_lbl = lbl_dir / f"{img.stem}.txt"
            if src_lbl.exists():
                shutil.copy2(src_lbl, dest_lbl_path)
                with open(dest_lbl_path, "r", encoding="utf-8", errors="ignore") as lf:
                    lines = [line.strip() for line in lf if line.strip()]
                    test_boxes += len(lines)
            else:
                dest_lbl_path.touch()

            test_count += 1

    print(f"Test split built: {test_count} images, {test_boxes} boxes.", flush=True)

    # 2. Process VAL split (excluding any test image hashes)
    print("\n--- Processing VAL split ---", flush=True)
    val_hashes = set()
    val_count = 0
    val_boxes = 0

    for b in batches:
        b_prefix = f"{b.name.lower().replace(' ', '')}_"
        img_dir = b / "images" / "val"
        lbl_dir = b / "labels" / "val"
        if not img_dir.exists(): continue

        for img in sorted(img_dir.glob("*")):
            if img.suffix.lower() not in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
                continue
            h = hash_file(img)
            if h in test_hashes:
                # Exclude to prevent leakage
                continue
            if h in val_hashes:
                continue
            val_hashes.add(h)

            dest_img_name = f"{b_prefix}{img.name}"
            dest_img_path = DATASET_DIR / "images" / "val" / dest_img_name
            shutil.copy2(img, dest_img_path)

            dest_lbl_name = f"{b_prefix}{img.stem}.txt"
            dest_lbl_path = DATASET_DIR / "labels" / "val" / dest_lbl_name
            src_lbl = lbl_dir / f"{img.stem}.txt"
            if src_lbl.exists():
                shutil.copy2(src_lbl, dest_lbl_path)
                with open(dest_lbl_path, "r", encoding="utf-8", errors="ignore") as lf:
                    lines = [line.strip() for line in lf if line.strip()]
                    val_boxes += len(lines)
            else:
                dest_lbl_path.touch()

            val_count += 1

    print(f"Val split built: {val_count} images, {val_boxes} boxes.", flush=True)

    # 3. Process TRAIN split (excluding any val or test hashes, and intra-train duplicates)
    print("\n--- Processing TRAIN split ---", flush=True)
    train_hashes = set()
    train_count = 0
    train_boxes = 0

    for b in batches:
        b_prefix = f"{b.name.lower().replace(' ', '')}_"
        img_dir = b / "images" / "train"
        lbl_dir = b / "labels" / "train"
        if not img_dir.exists(): continue

        for img in sorted(img_dir.glob("*")):
            if img.suffix.lower() not in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
                continue
            h = hash_file(img)
            if h in test_hashes or h in val_hashes:
                # Prevent cross-split data leakage!
                continue
            if h in train_hashes:
                continue
            train_hashes.add(h)

            dest_img_name = f"{b_prefix}{img.name}"
            dest_img_path = DATASET_DIR / "images" / "train" / dest_img_name
            shutil.copy2(img, dest_img_path)

            dest_lbl_name = f"{b_prefix}{img.stem}.txt"
            dest_lbl_path = DATASET_DIR / "labels" / "train" / dest_lbl_name
            src_lbl = lbl_dir / f"{img.stem}.txt"
            if src_lbl.exists():
                shutil.copy2(src_lbl, dest_lbl_path)
                with open(dest_lbl_path, "r", encoding="utf-8", errors="ignore") as lf:
                    lines = [line.strip() for line in lf if line.strip()]
                    train_boxes += len(lines)
            else:
                dest_lbl_path.touch()

            train_count += 1
            if train_count % 1000 == 0:
                print(f"Processed {train_count} train images...", flush=True)

    print(f"Train split built: {train_count} images, {train_boxes} boxes.", flush=True)

    total_images = train_count + val_count + test_count
    total_boxes = train_boxes + val_boxes + test_boxes
    print(f"\n==========================================")
    print(f"FINAL DATASET SUMMARY:")
    print(f"Train: {train_count} images | {train_boxes} boxes")
    print(f"Val:   {val_count} images | {val_boxes} boxes")
    print(f"Test:  {test_count} images | {test_boxes} boxes")
    print(f"Total: {total_images} images | {total_boxes} boxes")
    print(f"==========================================")

if __name__ == "__main__":
    main()
