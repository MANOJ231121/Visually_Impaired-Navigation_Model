import os
from pathlib import Path
from collections import Counter
PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_PATH = PROJECT_ROOT / "final batches"

total_images = 0
total_labels = 0
class_distribution = Counter()

for batch in sorted(DATASET_PATH.iterdir()):

    if not batch.is_dir():
        continue

    print(f"\n{'='*45}")
    print(f"Analyzing: {batch.name}")
    print(f"{'='*45}")

    for split in ["train", "val", "test"]:

        image_dir = batch / "images" / split
        label_dir = batch / "labels" / split

        if not image_dir.exists() or not label_dir.exists():
            print(f"{split}: Missing folder")
            continue

        images = {
            p.stem for p in image_dir.iterdir()
            if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
        }

        labels = {
            p.stem for p in label_dir.glob("*.txt")
        }

        missing_labels = images - labels
        missing_images = labels - images

        total_images += len(images)
        total_labels += len(labels)

        print(f"\n{split.upper()}")
        print("Images:", len(images))
        print("Labels:", len(labels))
        print("Images without labels:", len(missing_labels))
        print("Labels without images:", len(missing_images))

        for label_file in label_dir.glob("*.txt"):

            with open(label_file, "r") as f:

                for line in f:
                    values = line.strip().split()

                    if not values:
                        continue

                    if len(values) != 5:
                        print("Invalid annotation:", label_file.name)
                        continue

                    try:
                        cls = int(values[0])
                        coords = list(map(float, values[1:]))

                        if cls < 0 or not all(
                            0 <= x <= 1 for x in coords
                        ):
                            print("Invalid values:", label_file.name)
                            continue

                        class_distribution[cls] += 1

                    except ValueError:
                        print("Invalid data:", label_file.name)

print("\n========== FINAL SUMMARY ==========")
print("Total images:", total_images)
print("Total label files:", total_labels)
print("Class distribution:", dict(class_distribution))
print("Total classes:", len(class_distribution))
