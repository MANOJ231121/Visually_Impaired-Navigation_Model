# YOLO11s Object Detection Model

This directory contains the trained **Ultralytics YOLO11s (Small)** model for the Visually Impaired Navigation Model project, trained on the custom 35-class dataset.

---

## Model Information

- **Architecture:** Ultralytics YOLO11s (`yolo11s.pt`)
- **Parameters:** ~9.4 Million
- **Model Checkpoint Size:** 19.2 MB (optimizer stripped for production)

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

- **Base Weights:** `yolo11s.pt` (transfer learning)
- **Epochs:** 5
- **Image Size (`imgsz`):** 640 × 640
- **Batch Size:** 8
- **Optimizer:** MuSGD (`lr0=0.01`, `lrf=0.01`, `momentum=0.937`, `weight_decay=0.0005`)
- **Patience:** 20
- **Device:** `cuda:0`

---

## Hardware & Environment

- **GPU:** NVIDIA GeForce RTX 3050 Laptop GPU (4,096 MiB VRAM)
- **CUDA Version:** 12.1
- **Python:** 3.12.0
- **PyTorch:** 2.5.1+cu121
- **Ultralytics:** 8.4.32
- **Training Time:** 51.3 minutes (~3.0 it/s)

---

## Evaluation Metrics

Measured on the validation and test splits:

| Split | Images | Instances | Precision (P) | Recall (R) | mAP50 | mAP50-95 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Validation (`val`)** | 2,982 | 4,408 | **0.6527** (65.27%) | **0.6070** (60.70%) | **0.6514** (65.14%) | **0.4408** (44.08%) |
| **Test (`test`)** | 795 | 1,220 | **0.6266** (62.66%) | **0.6088** (60.88%) | **0.6427** (64.27%) | **0.4476** (44.76%) |

---

## Artifacts in this Directory

- `best.pt`: Trained model weights (19.2 MB)
- `results.png`: Training and validation loss curves and metrics progression
- `confusion_matrix.png`: Normalized confusion matrix across all 35 classes
- `README.md`: Training documentation and metrics report

---

## Inference Instructions

```bash
# Python
from ultralytics import YOLO

model = YOLO("models/YOLO11s/best.pt")
results = model.predict(source="path/to/image.jpg", conf=0.25, save=True)
```
