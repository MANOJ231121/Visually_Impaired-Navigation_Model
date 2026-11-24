# YOLO11m Object Detection Model

This directory contains the trained **Ultralytics YOLO11m (Medium)** model for the Visually Impaired Navigation Model project, trained on the custom 35-class dataset.

---

## Model Information

- **Architecture:** Ultralytics YOLO11m (`yolo11m.pt`)
- **Parameters:** ~20.1 Million (20,079,993 parameters)
- **Layers:** 232 layers
- **Model Checkpoint Size:** 40.5 MB (optimizer stripped for production)

---

## Dataset Details

- **Dataset:** Visually Impaired Navigation 35-class dataset
- **Classes:** 35 classes (`class_00` to `class_34`)
- **Total Images:** 15,303
- **Total Annotations:** 21,945 bounding boxes
- **Split Breakdown:**
  - **Train:** 11,526 images (16,317 bounding boxes)
  - **Validation:** 2,982 images (4,408 bounding boxes)
  - **Test:** 795 images (1,220 bounding boxes)
- **Data Leakage:** 0 duplicate images across splits

---

## Training Configuration

- **Base Weights:** `yolo11m.pt` (transfer learning)
- **Epochs:** 1
- **Image Size (`imgsz`):** 640 × 640
- **Batch Size:** 4 (optimized for 4GB VRAM)
- **Optimizer:** AdamW / Auto (`lr0=0.01`, `lrf=0.01`, `momentum=0.937`, `weight_decay=0.0005`)
- **Automatic Mixed Precision (AMP):** Enabled
- **Patience:** 20
- **Device:** `cuda:0`

---

## Hardware & Environment

- **GPU:** NVIDIA GeForce RTX 3050 Laptop GPU (4,096 MiB VRAM)
- **Peak VRAM Usage:** ~2.34 GB VRAM
- **CUDA Version:** 12.1
- **Python:** 3.12.0
- **PyTorch:** 2.5.1+cu121
- **Ultralytics:** 8.4.32
- **Training Time:** 14.1 minutes (844.5 seconds at ~3.8 it/s)

---

## Evaluation Metrics (Validation Split - 2,982 Images, 4,408 Instances)

| Metric | Measured Value |
| :--- | :--- |
| **Precision (P)** | **0.539** (53.9%) |
| **Recall (R)** | **0.471** (47.1%) |
| **mAP50** | **0.500** (50.0%) |
| **mAP50-95** | **0.317** (31.7%) |
| **Box Loss** | 1.1968 |
| **Class Loss** | 2.4107 |
| **DFL Loss** | 1.5494 |

---

## Artifacts in this Directory

- `best.pt`: Trained YOLO11m model weights (40.5 MB)
- `results.png`: Training loss curves and metric progression
- `confusion_matrix.png`: Normalized confusion matrix across all 35 classes
- `README.md`: Training documentation and metrics report

---

## Inference Instructions

```bash
# Python
from ultralytics import YOLO

model = YOLO("models/YOLO11m/best.pt")
results = model.predict(source="path/to/image.jpg", conf=0.25, save=True)
```
