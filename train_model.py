"""
train_model.py — Galaxy Scan Animal Detector Fine-Tuner
========================================================
Fine-tunes YOLOv8m on the downloaded COCO animal dataset.
Run AFTER download_dataset.py has completed.

Usage:
    python train_model.py

Output:
    models/animal_detector.pt   ← best weights (load in app.py)
    runs/                       ← training logs and metrics
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any, Optional

# ── Configuration ─────────────────────────────────────────────────────────────
BASE_MODEL   = "yolov8m.pt"      # medium model — best accuracy/speed balance
DATASET_YAML = Path("dataset") / "animals.yaml"
OUTPUT_DIR   = Path("models")
BEST_WEIGHTS = OUTPUT_DIR / "animal_detector.pt"

EPOCHS      = 50       # increase for better accuracy (100 recommended with GPU)
BATCH       = 16       # reduce to 8 if you run out of VRAM
IMG_SIZE    = 640      # standard YOLO input size
WORKERS     = 4        # dataloader workers (set 0 on Windows if errors occur)
PROJECT     = "runs"
RUN_NAME    = "animal_v1"

# ── Import YOLO ───────────────────────────────────────────────────────────────
try:
    from ultralytics import YOLO  # type: ignore[import]
except ImportError:
    print("[ERROR] ultralytics not installed. Run:  pip install ultralytics")
    raise


def _check_prerequisites() -> bool:
    """Validate dataset and model availability."""
    ok = True

    if not DATASET_YAML.exists():
        print(f"[ERROR] Dataset YAML not found: {DATASET_YAML}")
        print("        Run  python download_dataset.py  first.")
        ok = False

    train_imgs = Path("dataset") / "images" / "train"
    if not train_imgs.exists() or not any(train_imgs.iterdir()):
        print(f"[ERROR] Training images not found: {train_imgs}")
        print("        Run  python download_dataset.py  first.")
        ok = False

    return ok


def _copy_best_weights(run_dir: Path) -> Optional[Path]:
    """Copy best.pt from the Ultralytics run directory to models/."""
    best: Path = run_dir / "weights" / "best.pt"
    if not best.exists():
        print(f"[WARN] Best weights not found at {best}")
        return None

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy(best, BEST_WEIGHTS)
    print(f"\n✅  Best weights saved → {BEST_WEIGHTS}")
    return BEST_WEIGHTS


def main() -> None:
    print("=" * 60)
    print("  Galaxy Scan — YOLOv8m Animal Detector Fine-Tuner")
    print("=" * 60)

    if not _check_prerequisites():
        return

    print(f"\n  Base model  : {BASE_MODEL}")
    print(f"  Dataset     : {DATASET_YAML}")
    print(f"  Epochs      : {EPOCHS}")
    print(f"  Batch size  : {BATCH}")
    print(f"  Image size  : {IMG_SIZE}")
    print("\n  Training started — this will take 30–120 min depending on hardware.")
    print("  Press Ctrl+C to stop early (partial weights will be saved).\n")

    model: Any = YOLO(BASE_MODEL)

    results: Any = model.train(
        data=str(DATASET_YAML),
        epochs=EPOCHS,
        batch=BATCH,
        imgsz=IMG_SIZE,
        workers=WORKERS,
        project=PROJECT,
        name=RUN_NAME,
        exist_ok=True,
        # ── Augmentation (built-in YOLO augmentations) ────────────────
        augment=True,
        mosaic=1.0,       # mosaic augmentation (combines 4 images)
        mixup=0.1,        # MixUp augmentation
        flipud=0.0,       # no vertical flip (animals are upright)
        fliplr=0.5,       # horizontal flip
        degrees=10.0,     # random rotation ±10°
        scale=0.5,        # random scale ±50%
        hsv_h=0.015,      # HSV hue jitter
        hsv_s=0.7,        # HSV saturation jitter
        hsv_v=0.4,        # HSV value jitter
        # ── Optimiser settings ────────────────────────────────────────
        optimizer="AdamW",
        lr0=0.001,
        weight_decay=0.0005,
        warmup_epochs=3,
        # ── Misc ──────────────────────────────────────────────────────
        val=True,
        save=True,
        save_period=10,   # checkpoint every 10 epochs
        plots=True,
        verbose=True,
        device="",        # auto-select GPU if available, else CPU
    )

    # Locate run directory and copy best weights
    run_dir = Path(PROJECT) / RUN_NAME
    _copy_best_weights(run_dir)

    print("\n" + "=" * 60)
    print("  Training complete!")
    print(f"  Best model   → {BEST_WEIGHTS}")
    print(f"  Training run → {run_dir}")
    print("\n  Now restart app.py — it will auto-load your fine-tuned model.")
    print("=" * 60)


if __name__ == "__main__":
    main()
