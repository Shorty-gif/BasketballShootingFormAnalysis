"""
vision/draw_predictions.py
---------------------------
Drawing utilities for overlaying pose analysis results on frames.
Used by image_inference and video_inference.
"""

import cv2
import numpy as np
from typing import Dict, Tuple, List

# Colour palette
GREEN  = (50,  220, 100)
AMBER  = (50,  200, 255)
RED    = (50,   80, 230)
WHITE  = (255, 255, 255)
DARK   = (15,   15,  15)


def score_to_color(score: int) -> Tuple:
    if score >= 85:  return GREEN
    if score >= 50:  return AMBER
    return RED


def draw_score_bar(frame: np.ndarray, x: int, y: int,
                   width: int, score: int, label: str) -> None:
    """Draws a labelled horizontal score bar."""
    bar_w = int(width * score / 100)
    cv2.rectangle(frame, (x, y),      (x + width, y + 10), (40, 40, 40), -1)
    cv2.rectangle(frame, (x, y),      (x + bar_w, y + 10), score_to_color(score), -1)
    cv2.putText(frame, label + " " + str(score),
                (x, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.38, WHITE, 1)


def draw_joint_angles(frame: np.ndarray,
                      keypoints_pixel: np.ndarray,
                      metrics: Dict) -> np.ndarray:
    """
    Draws angle labels at the actual joint locations on the frame.
    keypoints_pixel: (17, 2) array in pixel coords (not normalised).
    """
    h, w = frame.shape[:2]

    angle_joints = {
        "elbow_angle":    8,   # r_elbow
        "knee_angle":     14,  # r_knee
        "shoulder_angle": 6,   # r_shoulder
    }

    for metric_key, joint_idx in angle_joints.items():
        if metric_key not in metrics:
            continue
        val  = metrics[metric_key]
        sc   = metrics.get("scores", {}).get(metric_key, 50)
        col  = score_to_color(sc)

        px = int(keypoints_pixel[joint_idx][0])
        py = int(keypoints_pixel[joint_idx][1])

        if px <= 0 or py <= 0:
            continue

        label = f"{val:.0f}deg"
        # Offset label slightly so it doesn't overlap the joint dot
        cv2.putText(frame, label, (px + 6, py - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, col, 1)
        cv2.circle(frame, (px, py), 4, col, -1)

    return frame


def draw_skeleton_overlay(frame: np.ndarray,
                           keypoints_pixel: np.ndarray,
                           scores: Dict) -> np.ndarray:
    """
    Draws a colour-coded skeleton on top of the frame.
    Bones are green (good score) / amber / red (poor score).
    keypoints_pixel: (17, 2) pixel coords.
    """
    # Connections: (kp_a, kp_b, metric_key_or_None)
    connections = [
        (5,  6,  None),               # shoulders
        (5,  7,  None),               # l_shoulder - l_elbow
        (7,  9,  None),               # l_elbow - l_wrist
        (6,  8,  "elbow_angle"),      # r_shoulder - r_elbow
        (8,  10, "elbow_angle"),      # r_elbow - r_wrist
        (5,  11, None),               # l_shoulder - l_hip
        (6,  12, "shoulder_angle"),   # r_shoulder - r_hip
        (11, 12, None),               # hips
        (11, 13, None),               # l_hip - l_knee
        (13, 15, None),               # l_knee - l_ankle
        (12, 14, "knee_angle"),       # r_hip - r_knee
        (14, 16, "knee_angle"),       # r_knee - r_ankle
    ]

    for a, b, metric_key in connections:
        pa = tuple(keypoints_pixel[a].astype(int))
        pb = tuple(keypoints_pixel[b].astype(int))
        if pa[0] <= 0 or pa[1] <= 0 or pb[0] <= 0 or pb[1] <= 0:
            continue
        sc  = scores.get(metric_key, 75) if metric_key else 75
        col = score_to_color(sc)
        cv2.line(frame, pa, pb, col, 2)

    return frame