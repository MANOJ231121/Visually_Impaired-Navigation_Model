import hashlib
from pathlib import Path
from collections import defaultdict

root = Path(r"C:\Users\DELL\.gemini\antigravity\scratch\Object-Detection-Model\final batches")
split_hashes = defaultdict(lambda: defaultdict(list))

print("Scanning files and computing hashes across splits...", flush=True)
for b in sorted(root.iterdir()):
    if not b.is_dir(): continue
    for split in ["train", "val", "test"]:
        img_dir = b / "images" / split
        if not img_dir.exists(): continue
        for img in img_dir.glob("*"):
            if img.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
                h = hashlib.md5(img.read_bytes()).hexdigest()
                split_hashes[split][h].append(img)

print("Scan complete. Checking for cross-split overlap (data leakage)...", flush=True)
train_hashes = set(split_hashes["train"].keys())
val_hashes = set(split_hashes["val"].keys())
test_hashes = set(split_hashes["test"].keys())

train_val_overlap = train_hashes & val_hashes
train_test_overlap = train_hashes & test_hashes
val_test_overlap = val_hashes & test_hashes

print(f"Unique images in train: {len(train_hashes)}", flush=True)
print(f"Unique images in val:   {len(val_hashes)}", flush=True)
print(f"Unique images in test:  {len(test_hashes)}", flush=True)
print(f"Train / Val overlap:    {len(train_val_overlap)} images", flush=True)
print(f"Train / Test overlap:   {len(train_test_overlap)} images", flush=True)
print(f"Val / Test overlap:     {len(val_test_overlap)} images", flush=True)
