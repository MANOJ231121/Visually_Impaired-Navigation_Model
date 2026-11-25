#!/usr/bin/env python3
"""
train_yolo11m.py
================

Train Ultralytics YOLO11m (Medium) on the 35-class dataset.
Optimized for GPUs with 4GB+ VRAM (batch=4, amp=True).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = PROJECT_ROOT / "data.yaml"
DEFAULT_PROJECT = PROJECT_ROOT / "runs" / "detect"

try:
    import torch
    AUTO_DEVICE = "0" if torch.cuda.is_available() else "cpu"
except ImportError:
    AUTO_DEVICE = "cpu"

TRAIN_CONFIG: dict[str, object] = {
    "epochs": 5,            # default passes over training set
    "imgsz": 640,           # square input resolution
    "batch": 4,             # batch size 4 safe for 4GB VRAM
    "patience": 20,
    "workers": 0,           # 0 avoids Windows multiprocessing crashes
    "cache": False,
    "fraction": 1.0,
    "seed": 0,
    "deterministic": True,
    "optimizer": "auto",
    "lr0": 0.01,
    "lrf": 0.01,
    "momentum": 0.937,
    "weight_decay": 0.0005,
    "warmup_epochs": 1.0,
    "warmup_momentum": 0.8,
    "warmup_bias_lr": 0.1,
    "box": 7.5,
    "cls": 0.5,
    "dfl": 1.5,
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "degrees": 0.0,
    "translate": 0.1,
    "scale": 0.5,
    "shear": 0.0,
    "perspective": 0.0,
    "flipud": 0.0,
    "fliplr": 0.5,
    "mosaic": 1.0,
    "mixup": 0.0,
    "cutmix": 0.0,
    "copy_paste": 0.0,
    "erasing": 0.4,
    "auto_augment": "randaugment",
    "close_mosaic": 10,
    "amp": True,
    "plots": True,
    "val": True,
    "save": True,
    "single_cls": False,
    "rect": False,
    "cos_lr": False,
    "exist_ok": False,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train YOLO11m on the project dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", default="yolo11m.pt", help="Pretrained model path or name.")
    parser.add_argument("--data", default=str(DEFAULT_DATA), help="Path to data.yaml.")
    parser.add_argument("--epochs", type=int, default=TRAIN_CONFIG["epochs"])
    parser.add_argument("--imgsz", type=int, default=TRAIN_CONFIG["imgsz"])
    parser.add_argument("--batch", type=int, default=TRAIN_CONFIG["batch"])
    parser.add_argument("--patience", type=int, default=TRAIN_CONFIG["patience"])
    parser.add_argument("--device", default=AUTO_DEVICE)
    parser.add_argument("--workers", type=int, default=TRAIN_CONFIG["workers"])
    parser.add_argument("--seed", type=int, default=TRAIN_CONFIG["seed"])
    parser.add_argument("--project", default=str(DEFAULT_PROJECT))
    parser.add_argument("--name", default="yolo11m_custom")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--fraction", type=float, default=TRAIN_CONFIG["fraction"])
    parser.add_argument("--exist-ok", action="store_true")
    return parser


def check_dataset(data_path: Path) -> None:
    if not data_path.is_file():
        raise SystemExit(f"ERROR: dataset config not found: {data_path}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    data_path = Path(args.data).resolve()
    check_dataset(data_path)

    project_dir = Path(args.project).resolve()
    project_dir.mkdir(parents=True, exist_ok=True)

    from ultralytics import YOLO

    model_source = args.model
    if not Path(model_source).is_file():
        local_candidate = PROJECT_ROOT / model_source
        if local_candidate.is_file():
            model_source = str(local_candidate)

    print(f"Loading YOLO model from: {model_source}")
    model = YOLO(model_source)

    train_kwargs = dict(TRAIN_CONFIG)
    train_kwargs.update({
        "data": str(data_path),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "patience": args.patience,
        "device": args.device,
        "workers": args.workers,
        "seed": args.seed,
        "project": str(project_dir),
        "name": args.name,
        "resume": args.resume,
        "cache": args.cache,
        "amp": not args.no_amp,
        "fraction": args.fraction,
        "exist_ok": args.exist_ok,
    })

    print("=" * 60)
    print("Starting YOLO11m Training")
    print(f"Model:    {model_source}")
    print(f"Data:     {data_path}")
    print(f"Epochs:   {args.epochs}")
    print(f"Batch:    {args.batch}")
    print(f"Device:   {args.device}")
    print(f"Project:  {project_dir / args.name}")
    print("=" * 60)

    results = model.train(**train_kwargs)
    print("Training complete!")
    return results


if __name__ == "__main__":
    main()
