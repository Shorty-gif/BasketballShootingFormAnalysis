"""
training/train_model.py
-----------------------
Trains YOLOv8 to detect basketball shooting form.

Before running:
    1. python convert_annotations.py     (generates labels/ folders)
    2. python training/train_model.py    (trains the model)
"""

import shutil
from pathlib import Path
from ultralytics import YOLO

# Project root is two levels up from training/train_model.py
PROJECT_ROOT = Path(__file__).parent.parent


def train() -> None:
    # yolov8n.pt lives at the project root (visible in your VS Code explorer)
    weights = PROJECT_ROOT / "yolov8n.pt"
    data    = PROJECT_ROOT / "data.yaml"

    model = YOLO(str(weights))

    results = model.train(
        data=str(data),
        epochs=50,
        imgsz=640,
        batch=16,       # lower to 8 if you run out of memory
        patience=10,    # early stopping: quit if val loss stalls for 10 epochs
        save=True,
        project=str(PROJECT_ROOT / "runs" / "detect"),
        name="train",
        exist_ok=True,  # safe to re-run without crashing
        verbose=True,
    )

    # Auto-copy best.pt into models/ to match your project structure
    best_src = Path(results.save_dir) / "weights" / "best.pt"
    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(exist_ok=True)
    shutil.copy(best_src, models_dir / "best.pt")

    print("\nTraining complete.")
    print(f"  runs/detect/train/weights/best.pt")
    print(f"  models/best.pt  (copy)")


if __name__ == "__main__":
    train()