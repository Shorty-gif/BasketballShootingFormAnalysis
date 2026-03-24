"""
inference/frame_inference.py
"""
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Dict
from ultralytics import YOLO

_model: Optional[YOLO] = None

def _get_model() -> YOLO:
    global _model
    if _model is None:
        _model = YOLO("yolov8n-pose.pt")
    return _model

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from inference.image_inference import (
    extract_metrics, draw_overlay, kp_conf,
    pick_shooter, ensure_color, KP,
)


def analyse_frame(frame: np.ndarray) -> Optional[Dict]:
    """
    Runs YOLOv8-pose on a single BGR frame.
    Uses tiered confidence fallback so external / webcam frames always work.
    Returns metrics dict with 'annotated_frame' key, or None if no person found.
    """
    model = _get_model()
    frame = ensure_color(frame)

    # Resize if very large
    h, w = frame.shape[:2]
    if w > 1280:
        scale = 1280 / w
        frame = cv2.resize(frame, (1280, int(h * scale)))

    kp_norm, kp_raw, results = None, None, None
    for conf in [0.45, 0.30, 0.20, 0.10]:
        results = model(frame, verbose=False, conf=conf)
        kp_norm, kp_raw = pick_shooter(results, frame.shape)
        if kp_norm is not None:
            break

    if kp_norm is None:
        return None

    metrics = extract_metrics(kp_norm, kp_raw)

    try:
        from analysis.ml_model import BasketballMLPredictor
        predictor     = BasketballMLPredictor()
        metrics["ml"] = predictor.predict(metrics)
    except Exception as e:
        metrics["ml"] = {"error": str(e), "knn_matches": [], "knn_summary": ""}

    annotated = results[0].plot()
    annotated = draw_overlay(annotated, metrics)
    metrics["annotated_frame"] = annotated
    return metrics