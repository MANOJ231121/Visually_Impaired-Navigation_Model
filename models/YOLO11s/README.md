# YOLO11s Object Detection Model

This directory contains the high-accuracy trained **Ultralytics YOLO11s (Small)** model for the Visually Impaired Navigation Model project, trained on the custom 35-class dataset.

---

## Model Information

- **Architecture:** Ultralytics YOLO11s (`yolo11s.pt`)
- **Parameters:** ~9.4 Million
- **Model Checkpoint Size:** 18.3 MB (under 100 MB, Git LFS not required)

---

## Dataset Details

- **Dataset:** Visually Impaired Navigation 35-class dataset
- **Classes:** 35 classes (`class_00` to `class_34`)
- **Total Images:** 15,303 (11,526 train, 2,982 val, 795 test)
- **Total Annotations:** 21,945 bounding boxes
- **Data Leakage:** 0 duplicate images across splits

---

## Training Configuration

- **Epochs:** 60 total epochs (55 fine-tuning epochs with Cosine Annealing)
- **Image Size (`imgsz`):** 640 × 640
- **Batch Size:** 8
- **Cosine Learning Rate Decay:** Enabled (`cos_lr=True`)
- **Close Mosaic:** 10 epochs
- **Optimizer:** Auto / AdamW
- **Device:** `cuda:0`

---

## Hardware & Environment

- **GPU:** NVIDIA GeForce RTX 3050 Laptop GPU (4,096 MiB VRAM)
- **CUDA Version:** 12.1
- **Python:** 3.12.0
- **PyTorch:** 2.5.1+cu121
- **Ultralytics:** 8.4.32

---

## Evaluation Metrics (Validation Split - 2,982 Images)

| Metric | Measured Value |
| :--- | :--- |
| **Precision (P)** | **0.8786** (87.86%) |
| **Recall (R)** | **0.8178** (81.78%) |
| **mAP50** | **0.8719** (87.19%) |
| **mAP50-95** | **0.6800** (68.00%) |

---

## Artifacts in this Directory

- `best.pt`: Trained model weights (18.3 MB)
- `results.png`: Full training loss curves and metric progression
- `confusion_matrix.png`: Normalized confusion matrix across all 35 classes
- `README.md`: Training documentation and metrics report

---

## Inference Instructions

```python
from ultralytics import YOLO

model = YOLO("models/YOLO11s/best.pt")
results = model.predict(source="path/to/image.jpg", conf=0.25, save=True)
```
