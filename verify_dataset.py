import os
import sys
from pathlib import Path
from PIL import Image
import yaml

def verify():
    project_root = Path(__file__).resolve().parent
    data_yaml_path = project_root / "data.yaml"

    if not data_yaml_path.exists():
        print(f"ERROR: {data_yaml_path} does not exist!")
        return False

    with open(data_yaml_path, "r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)

    nc = data_cfg.get("nc", 35)
    names = data_cfg.get("names", {})
    print(f"data.yaml nc: {nc}, names count: {len(names)}")

    root_path = project_root / data_cfg.get("path", "dataset")
    if not root_path.exists():
        # Check if dataset is in current dir directly
        if (project_root / "images").exists():
            root_path = project_root
        else:
            print(f"ERROR: Dataset root {root_path} does not exist!")
            return False

    print(f"Dataset root: {root_path}")

    splits = ["train", "val", "test"]
    stats = {}
    corrupted_images = []
    invalid_labels = []
    class_counts = {i: 0 for i in range(nc)}
    all_image_names = {}

    for split in splits:
        img_rel = data_cfg.get(split, f"images/{split}")
        # Strip path prefix if needed
        if img_rel.startswith(data_cfg.get("path", "") + "/"):
            img_rel = img_rel[len(data_cfg.get("path", "")) + 1:]
        img_dir = (root_path / img_rel).resolve()
        lbl_dir = (root_path / img_rel.replace("images", "labels")).resolve()

        if not img_dir.exists():
            print(f"WARNING: Image dir does not exist: {img_dir}")
            continue

        images = [p for p in img_dir.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]]
        labels = [p for p in lbl_dir.iterdir() if p.suffix.lower() == ".txt"] if lbl_dir.exists() else []

        print(f"Split [{split}]: {len(images)} images, {len(labels)} label files found.")

        split_boxes = 0
        split_corrupt = 0
        split_invalid = 0

        for img_path in images:
            # Check data leakage
            base_name = img_path.name
            if base_name in all_image_names:
                print(f"LEAKAGE DETECTED: {base_name} exists in both {all_image_names[base_name]} and {split}!")
            else:
                all_image_names[base_name] = split

            # Check label
            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            if lbl_path.exists():
                try:
                    with open(lbl_path, "r", encoding="utf-8") as lf:
                        lines = lf.readlines()
                    for line_idx, line in enumerate(lines):
                        parts = line.strip().split()
                        if not parts:
                            continue
                        if len(parts) != 5:
                            invalid_labels.append((str(lbl_path), line_idx, f"Invalid format: {line.strip()}"))
                            split_invalid += 1
                            continue
                        cid, xc, yc, w, h = parts
                        cid = int(cid)
                        xc, yc, w, h = float(xc), float(yc), float(w), float(h)

                        if cid < 0 or cid >= nc:
                            invalid_labels.append((str(lbl_path), line_idx, f"Class ID {cid} out of range [0, {nc-1}]"))
                            split_invalid += 1
                        if not (0 <= xc <= 1 and 0 <= yc <= 1 and 0 < w <= 1 and 0 < h <= 1):
                            invalid_labels.append((str(lbl_path), line_idx, f"BBox out of [0, 1]: {xc}, {yc}, {w}, {h}"))
                            split_invalid += 1

                        if 0 <= cid < nc:
                            class_counts[cid] += 1
                        split_boxes += 1
                except Exception as e:
                    invalid_labels.append((str(lbl_path), -1, str(e)))
                    split_invalid += 1

        stats[split] = {
            "images": len(images),
            "labels": len(labels),
            "boxes": split_boxes,
            "corrupt": split_corrupt,
            "invalid": split_invalid,
        }

    print("\n--- DATASET VERIFICATION SUMMARY ---")
    for split, s in stats.items():
        print(f"Split {split:5s} | Images: {s['images']:6d} | Labels: {s['labels']:6d} | Boxes: {s['boxes']:6d} | Invalid: {s['invalid']}")

    active_classes = {c: count for c, count in class_counts.items() if count > 0}
    print(f"Active classes with annotations: {len(active_classes)} / {nc}")
    print(f"Total invalid labels: {len(invalid_labels)}")
    print(f"Total corrupted images: {len(corrupted_images)}")
    return len(invalid_labels) == 0 and len(corrupted_images) == 0

if __name__ == "__main__":
    success = verify()
    sys.exit(0 if success else 1)
