"""
training/evaluate_model.py
--------------------------
Evaluates the trained model on the validation set.

Run after training:
    python training/evaluate_model.py

Key metrics:
    mAP50      > 0.70 is a solid result for a single-class detector
    mAP50-95   > 0.50 is good
    precision  how many detections were correct
    recall     how many real shooting poses were found
"""

from pathlib import Path
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).parent.parent

# Prefer models/best.pt (your project structure), fall back to runs/
WEIGHTS = PROJECT_ROOT / "models" / "best.pt"
if not WEIGHTS.exists():
    WEIGHTS = PROJECT_ROOT / "runs" / "detect" / "train" / "weights" / "best.pt"

DATA = PROJECT_ROOT / "data.yaml"


def evaluate() -> None:
    if not WEIGHTS.exists():
        print(f"No weights found. Train the model first.")
        return

    print(f"Evaluating: {WEIGHTS}\n")
    model = YOLO(str(WEIGHTS))

    metrics = model.val(data=str(DATA), imgsz=640, verbose=True)

    print("\n── Results ─────────────────────────────────")
    print(f"  mAP50:     {metrics.box.map50:.4f}")
    print(f"  mAP50-95:  {metrics.box.map:.4f}")
    print(f"  Precision: {metrics.box.mp:.4f}")
    print(f"  Recall:    {metrics.box.mr:.4f}")
    print("─────────────────────────────────────────────")


if __name__ == "__main__":
    evaluate()