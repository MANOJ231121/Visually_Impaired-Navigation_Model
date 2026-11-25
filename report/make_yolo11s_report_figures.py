"""Build YOLO11s figures for PROJECT_REPORT_UPDATED.md.

Sources are the final high-accuracy checkpoint and the cleaned validation split.
Only the four finished figures are written to report/figures/; no prediction
images, validation runs or caches are added to the repository.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report" / "figures"
WEIGHTS = ROOT / "models" / "YOLO11s" / "best.pt"
NAMES = ROOT / "model_training" / "class_names.json"
VAL_IMAGES = ROOT / "model_training" / "dataset" / "images" / "val"
VAL_LABELS = ROOT / "model_training" / "dataset" / "labels" / "val"

TILE_W = 480
TILE_H = 320
COLS = 3
ROWS = 2
PREFERRED_IDS = (0, 1, 4, 20, 29, 34, 24, 23, 27, 32)


def load_names() -> dict[int, str]:
    raw = json.loads(NAMES.read_text(encoding="utf-8"))
    return {int(key): str(value) for key, value in raw.items() if key.isdigit()}


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for family in ("Arial", "DejaVu Sans"):
        try:
            path = font_manager.findfont(
                font_manager.FontProperties(family=family), fallback_to_default=False
            )
            return ImageFont.truetype(path, size)
        except (ValueError, OSError):
            continue
    return ImageFont.load_default()


def candidate_images(limit: int = 220) -> list[Path]:
    """Validation images whose labels contain at least one object."""
    images: list[Path] = []
    for label in sorted(VAL_LABELS.glob("*.txt")):
        if not any(line.strip() for line in label.read_text(encoding="utf-8").splitlines()):
            continue
        for suffix in (".jpg", ".jpeg", ".png"):
            image = VAL_IMAGES / f"{label.stem}{suffix}"
            if image.exists():
                images.append(image)
                break
        if len(images) >= limit:
            break
    return images


def colour(class_id: int) -> tuple[int, int, int]:
    palette = (
        (46, 204, 113), (52, 152, 219), (231, 76, 60), (155, 89, 182),
        (241, 196, 15), (26, 188, 156), (230, 126, 34), (149, 165, 166),
    )
    return palette[class_id % len(palette)]


def draw_tile(image_path: Path, box, class_id: int, confidence: float, name: str) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    line = colour(class_id)
    for offset in range(3):
        draw.rectangle(
            (x1 - offset, y1 - offset, x2 + offset, y2 + offset),
            outline=line,
            width=2,
        )
    label = f"ID {class_id:02d}  {name}  {confidence:.2f}"
    small = font(max(14, image.width // 42))
    text_box = draw.textbbox((0, 0), label, font=small)
    text_w = text_box[2] - text_box[0]
    text_h = text_box[3] - text_box[1]
    label_x = max(0, min(x1, image.width - text_w - 8))
    label_y = max(0, y1 - text_h - 8)
    draw.rectangle(
        (label_x, label_y, label_x + text_w + 8, label_y + text_h + 6),
        fill=(18, 18, 18),
    )
    draw.text((label_x + 4, label_y + 2), label, fill=(255, 255, 255), font=small)

    image.thumbnail((TILE_W, TILE_H), Image.LANCZOS)
    tile = Image.new("RGB", (TILE_W, TILE_H), (245, 245, 245))
    tile.paste(image, ((TILE_W - image.width) // 2, (TILE_H - image.height) // 2))
    return tile


def sample_sheet(model: YOLO, names: dict[int, str], images: list[Path]) -> None:
    chosen: dict[int, tuple[Path, object, float]] = {}
    for start in range(0, len(images), 32):
        batch = images[start : start + 32]
        results = model.predict(
            source=[str(path) for path in batch],
            conf=0.35,
            iou=0.45,
            imgsz=640,
            verbose=False,
        )
        for image_path, result in zip(batch, results):
            if result.boxes is None or len(result.boxes) == 0:
                continue
            best = max(result.boxes, key=lambda item: float(item.conf[0]))
            class_id = int(best.cls[0])
            confidence = float(best.conf[0])
            if class_id in names and class_id not in chosen and confidence >= 0.35:
                box = tuple(float(value) for value in best.xyxy[0])
                chosen[class_id] = (image_path, box, confidence)
        if len(chosen) >= len(PREFERRED_IDS):
            break

    ordered = [class_id for class_id in PREFERRED_IDS if class_id in chosen]
    ordered += [class_id for class_id in chosen if class_id not in ordered]
    selected = ordered[: COLS * ROWS]
    if len(selected) < COLS * ROWS:
        raise RuntimeError(f"only {len(selected)} distinct YOLO11s detections found")

    header = 62
    footer = 42
    sheet = Image.new("RGB", (COLS * TILE_W, header + ROWS * TILE_H + footer), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    draw.text(
        (18, 16),
        "YOLO11s validation images - predictions from models/YOLO11s/best.pt",
        fill=(20, 20, 20),
        font=font(24),
    )
    for index, class_id in enumerate(selected):
        image_path, box, confidence = chosen[class_id]
        tile = draw_tile(image_path, box, class_id, confidence, names[class_id])
        x = (index % COLS) * TILE_W
        y = header + (index // COLS) * TILE_H
        sheet.paste(tile, (x, y))
    draw.text(
        (18, header + ROWS * TILE_H + 10),
        "Real validation photographs; display names from class_names.json, numeric IDs are authoritative.",
        fill=(80, 80, 80),
        font=font(16),
    )
    path = OUT / "yolo11s_dataset_samples.jpg"
    sheet.save(path, quality=90, subsampling=0, optimize=True)
    summary = ", ".join(
        f"{class_id:02d} {chosen[class_id][2]:.2f}" for class_id in selected
    )
    print(f"{path.name}: {sheet.size}; classes/confidences: {summary}")


def copied_result_figures() -> None:
    for source, target in (
        (ROOT / "models" / "YOLO11s" / "results.png", "yolo11s_results.png"),
        (ROOT / "models" / "YOLO11s" / "confusion_matrix.png", "yolo11s_val_confusion_matrix.png"),
    ):
        shutil.copy2(source, OUT / target)
        print(f"{target}: copied from {source.relative_to(ROOT)}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    copied_result_figures()
    model = YOLO(str(WEIGHTS))
    names = load_names()
    sample_sheet(model, names, candidate_images())


if __name__ == "__main__":
    main()
