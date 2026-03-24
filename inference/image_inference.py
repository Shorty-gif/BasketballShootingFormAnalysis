"""
inference/image_inference.py  —  v3
Fixes:
  - Phase detection now requires wrist above shoulder to confirm shooting
  - Sideways detection: auto-detects camera angle and adjusts scoring
  - Shoulder score bug fixed (was using wrong key in scores dict)
  - Standing-still guard: won't call "follow_through" unless arm is raised
"""

import cv2
import json
import argparse
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from ultralytics import YOLO

model = YOLO("yolov8n-pose.pt")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
OUTPUT_IMAGES    = Path("outputs/annotated_images")
OUTPUT_METRICS   = Path("outputs/prediction_results")

KP = {
    "nose":0,"l_eye":1,"r_eye":2,"l_ear":3,"r_ear":4,
    "l_shoulder":5,"r_shoulder":6,
    "l_elbow":7,"r_elbow":8,
    "l_wrist":9,"r_wrist":10,
    "l_hip":11,"r_hip":12,
    "l_knee":13,"r_knee":14,
    "l_ankle":15,"r_ankle":16,
}

# Phase-specific standards. Shoulder = arm elevation from horizontal (0-90°).
PHASE_STANDARDS = {
    "loading": {
        "elbow_angle":    (85,  115, 70,  135),
        "knee_angle":     (95,  125, 80,  145),
        "shoulder_angle": (30,  55,  15,  65),
        "wrist_elbow_v":  (0,   35,  0,   55),
    },
    "set": {
        "elbow_angle":    (80,  100, 65,  118),
        "knee_angle":     (100, 130, 85,  150),
        "shoulder_angle": (48,  72,  32,  82),
        "wrist_elbow_v":  (0,   22,  0,   38),
    },
    "release": {
        "elbow_angle":    (78,  108, 65,  128),
        "knee_angle":     (125, 175, 108, 180),
        "shoulder_angle": (58,  88,  42,  90),
        "wrist_elbow_v":  (0,   28,  0,   45),
    },
    "follow_through": {
        "elbow_angle":    (138, 180, 118, 180),
        "knee_angle":     (148, 180, 128, 180),
        "shoulder_angle": (62,  90,  48,  90),
        "wrist_elbow_v":  (0,   38,  0,   58),
    },
    "not_shooting": {   # standing still / idle
        "elbow_angle":    (80,  100, 60,  120),
        "knee_angle":     (160, 180, 140, 180),
        "shoulder_angle": (0,   30,  0,   45),
        "wrist_elbow_v":  (0,   50,  0,   70),
    },
}

COACHING_CUES = {
    "elbow_angle": {
        "perfect":  "Elbow angle is ideal for this phase",
        "too_low":  "Elbow too collapsed — will restrict wrist snap",
        "too_high": "Elbow flaring — reduces accuracy and arc",
    },
    "knee_angle": {
        "perfect":  "Knee bend is ideal — legs driving the shot correctly",
        "too_low":  "Over-bent knees — losing power transfer upward",
        "too_high": "Legs too straight — shot relies on arm strength only",
    },
    "shoulder_angle": {
        "perfect":  "Arm elevation is ideal for this phase — good arc",
        "too_low":  "Arm too low — raise your elbow higher for better arc",
        "too_high": "Arm at max elevation — normal at follow-through",
    },
    "wrist_elbow_v": {
        "perfect":  "Wrist stacked above elbow — clean release axis",
        "too_low":  None,
        "too_high": "Wrist not above elbow — ball will drift sideways",
    },
}

# ── Math ───────────────────────────────────────────────────────────────────────

def angle_between(a, b, c) -> float:
    ba    = np.array(a) - np.array(b)
    bc    = np.array(c) - np.array(b)
    denom = np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8
    return float(np.degrees(np.arccos(np.clip(np.dot(ba, bc) / denom, -1.0, 1.0))))

def vertical_angle(p1, p2) -> float:
    dx = float(p2[0] - p1[0])
    dy = float(p2[1] - p1[1])
    return float(np.degrees(np.arctan2(abs(dx), abs(dy) + 1e-8)))

def kp_conf(kp_raw: np.ndarray, idx: int) -> float:
    if kp_raw.ndim == 2 and kp_raw.shape[1] >= 3:
        return float(kp_raw[idx, 2])
    return 1.0

# ── Grayscale fix ──────────────────────────────────────────────────────────────

def ensure_color(frame: np.ndarray) -> np.ndarray:
    if frame is None:
        return frame
    is_gray = False
    if len(frame.shape) == 2:
        is_gray = True
    elif frame.shape[2] == 1:
        is_gray = True
    elif (abs(int(frame[:,:,0].mean()) - int(frame[:,:,1].mean())) < 3 and
          abs(int(frame[:,:,1].mean()) - int(frame[:,:,2].mean())) < 3):
        is_gray = True
    if is_gray:
        gray     = frame[:,:,0] if len(frame.shape) == 3 else frame
        clahe    = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    return frame

# ── Camera angle detection ─────────────────────────────────────────────────────

def detect_camera_angle(kp: np.ndarray, kp_raw: np.ndarray) -> str:
    """
    Detects if the player is facing the camera (front-on) or sideways.
    Uses shoulder width vs hip width ratio.
    Front-on: both shoulders visible and wide apart.
    Sideways: one shoulder hidden, narrow shoulder span.
    Returns: 'front', 'sideways', or 'unknown'
    """
    r_sh_conf = kp_conf(kp_raw, KP["r_shoulder"])
    l_sh_conf = kp_conf(kp_raw, KP["l_shoulder"])

    # If one shoulder is very low confidence, likely sideways
    if min(r_sh_conf, l_sh_conf) < 0.25:
        return "sideways"

    # Shoulder span in normalised coords
    sh_span = abs(float(kp[KP["r_shoulder"]][0]) - float(kp[KP["l_shoulder"]][0]))

    # Hip span
    r_hip_conf = kp_conf(kp_raw, KP["r_hip"])
    l_hip_conf = kp_conf(kp_raw, KP["l_hip"])
    if min(r_hip_conf, l_hip_conf) > 0.25:
        hip_span = abs(float(kp[KP["r_hip"]][0]) - float(kp[KP["l_hip"]][0]))
        # Front-on: shoulder span ≈ hip span, both > 0.08 of frame width
        if sh_span > 0.08 and hip_span > 0.06:
            return "front"
        if sh_span < 0.06:
            return "sideways"

    if sh_span < 0.05:
        return "sideways"
    return "front"

# ── Shooting detection ─────────────────────────────────────────────────────────

def is_shooting_pose(kp: np.ndarray, kp_raw: np.ndarray,
                     elbow_angle: float, knee_angle: float) -> bool:
    """
    FIX #6: Guard against standing-still false positives.
    Returns True only if the person appears to actually be shooting.

    Requirements (any one of these):
      - Wrist is above the shoulder (clear shooting position)
      - Elbow is raised above shoulder level
      - Knee is significantly bent (loading for a shot)
    """
    side = detect_side(kp, kp_raw)
    if side == "right":
        wrist_y   = float(kp[KP["r_wrist"]][1])
        shoulder_y = float(kp[KP["r_shoulder"]][1])
        elbow_y    = float(kp[KP["r_elbow"]][1])
    else:
        wrist_y    = float(kp[KP["l_wrist"]][1])
        shoulder_y = float(kp[KP["l_shoulder"]][1])
        elbow_y    = float(kp[KP["l_elbow"]][1])

    # In image coords: lower y = higher in frame
    wrist_above_shoulder = wrist_y < shoulder_y - 0.03
    elbow_above_shoulder = elbow_y < shoulder_y + 0.02
    knee_bent            = knee_angle < 155

    return wrist_above_shoulder or elbow_above_shoulder or knee_bent

# ── Phase detection ────────────────────────────────────────────────────────────

def detect_phase(kp: np.ndarray, kp_raw: np.ndarray,
                 knee_angle: float, elbow_angle: float) -> str:
    """
    FIX #2 + #6: Robust phase detection.
    - Checks wrist position relative to shoulder (not just normalised y)
    - Guards against standing-still → follow_through false positive
    - Uses both arm and leg signals together
    """
    side = detect_side(kp, kp_raw)
    if side == "right":
        wrist_y    = float(kp[KP["r_wrist"]][1])
        shoulder_y = float(kp[KP["r_shoulder"]][1])
        elbow_y    = float(kp[KP["r_elbow"]][1])
    else:
        wrist_y    = float(kp[KP["l_wrist"]][1])
        shoulder_y = float(kp[KP["l_shoulder"]][1])
        elbow_y    = float(kp[KP["l_elbow"]][1])

    # In image coords: lower y = higher up in frame
    wrist_above_shoulder = wrist_y < shoulder_y - 0.03
    wrist_near_shoulder  = abs(wrist_y - shoulder_y) < 0.10
    elbow_raised         = elbow_y < shoulder_y + 0.04

    # FIX #6: Not shooting at all — arm down, legs straight
    if not wrist_above_shoulder and not elbow_raised and knee_angle > 158:
        return "not_shooting"

    # Follow-through: arm fully extended AND wrist above or near shoulder
    if elbow_angle > 135 and (wrist_above_shoulder or wrist_near_shoulder):
        return "follow_through"

    # Release: wrist clearly above shoulder, arm still somewhat bent
    if wrist_above_shoulder and elbow_angle <= 135:
        return "release"

    # Loading: knee bent significantly
    if knee_angle < 128 and elbow_raised:
        return "loading"

    # Set: elbow raised, not yet releasing
    if elbow_raised and wrist_near_shoulder:
        return "set"

    # Fallback
    if wrist_above_shoulder:
        return "release"
    return "loading"

def detect_side(kp: np.ndarray, kp_raw: np.ndarray) -> str:
    r_conf = kp_conf(kp_raw, KP["r_wrist"])
    l_conf = kp_conf(kp_raw, KP["l_wrist"])
    if r_conf < 0.25 and l_conf >= 0.25:
        return "left"
    if l_conf < 0.25 and r_conf >= 0.25:
        return "right"
    return "right" if kp[KP["r_wrist"]][1] <= kp[KP["l_wrist"]][1] else "left"

# ── Scoring ────────────────────────────────────────────────────────────────────

def score_metric(val, ideal, acceptable, lower_is_better=False) -> int:
    if lower_is_better:
        _, imax = ideal; _, amax = acceptable
        if val <= imax:
            return int(85 + 15 * max(0.0, 1.0 - val / (imax + 1e-8)))
        elif val <= amax:
            return int(50 + 35 * (amax - val) / (amax - imax + 1e-8))
        return max(0, int(50 * (amax * 2 - val) / (amax + 1e-8)))
    imin, imax = ideal; amin, amax = acceptable
    if imin <= val <= imax:
        c = (imin + imax) / 2; sp = (imax - imin) / 2 + 1e-8
        return int(85 + 15 * max(0.0, 1.0 - abs(val - c) / sp))
    elif amin <= val <= amax:
        if val < imin: return int(50 + 35 * (val - amin) / (imin - amin + 1e-8))
        return int(50 + 35 * (amax - val) / (amax - imax + 1e-8))
    m = 25.0
    if val < amin: return max(0, int(50 * (val - (amin - m)) / m))
    return max(0, int(50 * ((amax + m) - val) / m))

# ── Metrics extraction ─────────────────────────────────────────────────────────

def extract_metrics(kp: np.ndarray, kp_raw: np.ndarray) -> Dict:
    side = detect_side(kp, kp_raw)
    if side == "right":
        sh, el, wr, hp, kn, an = (KP["r_shoulder"], KP["r_elbow"], KP["r_wrist"],
                                   KP["r_hip"], KP["r_knee"], KP["r_ankle"])
    else:
        sh, el, wr, hp, kn, an = (KP["l_shoulder"], KP["l_elbow"], KP["l_wrist"],
                                   KP["l_hip"], KP["l_knee"], KP["l_ankle"])

    elbow_angle    = angle_between(kp[sh], kp[el], kp[wr])
    knee_angle     = angle_between(kp[hp], kp[kn], kp[an])

    # Arm elevation from horizontal (0=arm down/sideways, 90=arm straight up)
    sh_pt = kp[sh]; el_pt = kp[el]
    dx    = float(el_pt[0] - sh_pt[0])
    dy    = float(el_pt[1] - sh_pt[1])
    shoulder_angle = float(np.clip(
        abs(np.degrees(np.arctan2(-dy, abs(dx) + 1e-8))), 0, 90
    ))

    wrist_elbow_v  = vertical_angle(kp[el], kp[wr])
    hip_knee_align = float(abs(kp[hp][0] - kp[kn][0]))

    # FIX #2 + #6: use robust phase detection
    phase = detect_phase(kp, kp_raw, knee_angle, elbow_angle)

    # FIX #7: detect camera angle — sideways shots have different shoulder geometry
    cam_angle = detect_camera_angle(kp, kp_raw)

    std = PHASE_STANDARDS.get(phase, PHASE_STANDARDS["set"])

    # FIX #4 + #7: use consistent key names in scores dict
    # FIX #7: widen shoulder tolerance for sideways shots
    sh_ideal = std["shoulder_angle"][:2]
    sh_acc   = std["shoulder_angle"][2:]
    if cam_angle == "sideways":
        # Sideways: shoulder elevation measurement is less reliable
        # Widen acceptable range by 15° each side
        sh_ideal = (max(0, sh_ideal[0]-15), min(90, sh_ideal[1]+15))
        sh_acc   = (max(0, sh_acc[0]-20),   min(90, sh_acc[1]+20))

    scores = {
        "elbow_angle":          score_metric(elbow_angle,    std["elbow_angle"][:2],    std["elbow_angle"][2:]),
        "knee_angle":           score_metric(knee_angle,     std["knee_angle"][:2],     std["knee_angle"][2:]),
        "shoulder_angle":       score_metric(shoulder_angle, sh_ideal,                  sh_acc),
        "wrist_elbow_vertical": score_metric(wrist_elbow_v,  std["wrist_elbow_v"][:2],  std["wrist_elbow_v"][2:], True),
        "hip_knee_alignment":   score_metric(hip_knee_align, (0, 0.08),                 (0, 0.20),                True),
    }

    weights = {
        "elbow_angle":          0.30,
        "knee_angle":           0.20,
        "shoulder_angle":       0.20,
        "wrist_elbow_vertical": 0.15,
        "hip_knee_alignment":   0.15,
    }
    overall = int(sum(scores[k] * weights[k] for k in weights))

    def get_cue(metric, val, ideal, lb=False):
        cues = COACHING_CUES.get(metric, {})
        if lb:
            _, imax = ideal
            return cues.get("perfect","Good") if val <= imax else cues.get("too_high","Needs work")
        imin, imax = ideal
        if imin <= val <= imax: return cues.get("perfect","Good")
        if val < imin:          return cues.get("too_low") or cues.get("too_high","Needs work")
        return cues.get("too_high","Needs work")

    feedback = {
        "elbow_angle":          get_cue("elbow_angle",          elbow_angle,    std["elbow_angle"][:2]),
        "knee_angle":           get_cue("knee_angle",           knee_angle,     std["knee_angle"][:2]),
        "shoulder_angle":       get_cue("shoulder_angle",       shoulder_angle, sh_ideal),
        "wrist_elbow_vertical": get_cue("wrist_elbow_v",        wrist_elbow_v,  std["wrist_elbow_v"][:2], True),
        "hip_knee_alignment":   get_cue("hip_knee_align",       hip_knee_align, (0, 0.08),                True),
    }

    grade = ("A" if overall>=88 else "B" if overall>=75 else
             "C" if overall>=60 else "D" if overall>=45 else "F")

    return {
        "side":                 side,
        "phase":                phase,
        "camera_angle":         cam_angle,
        "elbow_angle":          round(elbow_angle, 2),
        "knee_angle":           round(knee_angle, 2),
        "shoulder_angle":       round(shoulder_angle, 2),
        "wrist_elbow_vertical": round(wrist_elbow_v, 2),
        "hip_knee_alignment":   round(hip_knee_align, 4),
        "wrist_height_norm":    round(float(kp[wr][1]), 4),
        "scores":               scores,
        "overall_score":        overall,
        "grade":                grade,
        "feedback":             feedback,
        "is_shooting":          phase != "not_shooting",
    }

# ── Shooter selection ──────────────────────────────────────────────────────────

def pick_shooter(results, frame_shape: Tuple) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Selects the primary shooter with three fallback tiers:
      Tier 1 (strict)  : bbox > 6%, arm conf > 0.20  — dataset images, crowded courts
      Tier 2 (lenient) : bbox > 1.5%, arm conf > 0.10 — external photos, wide shots
      Tier 3 (any)     : at least one body keypoint visible — last resort
    This ensures external/personal images always get analysed.
    """
    if results[0].keypoints is None or len(results[0].keypoints.xyn) == 0:
        return None, None

    img_h, img_w = frame_shape[:2]
    img_area     = img_h * img_w

    def _score_candidates(min_bbox_frac: float, min_arm_conf: float):
        candidates = []
        for i in range(len(results[0].keypoints.xyn)):
            kp_n = results[0].keypoints.xyn[i].cpu().numpy()
            kp_r = results[0].keypoints.data[i].cpu().numpy()
            if kp_n.shape[0] != 17: continue

            if i < len(results[0].boxes):
                box       = results[0].boxes[i].xyxy[0].cpu().numpy()
                bbox_frac = ((box[2]-box[0])*(box[3]-box[1])) / (img_area + 1e-8)
            else:
                bbox_frac = 1.0   # no box info → don't penalise

            if bbox_frac < min_bbox_frac: continue

            arm_joints = [KP["r_elbow"], KP["r_wrist"],
                          KP["l_elbow"], KP["l_wrist"]]
            arm_confs  = [kp_conf(kp_r, j) for j in arm_joints]
            if max(arm_confs) < min_arm_conf: continue

            wrist_score = 1.0 - min(float(kp_n[KP["r_wrist"]][1]),
                                     float(kp_n[KP["l_wrist"]][1]))
            cx          = ((box[0]+box[2])/2)/img_w if i < len(results[0].boxes) else 0.5
            centrality  = 1.0 - abs(cx - 0.5) * 2
            size_score  = min(1.0, bbox_frac * 8)
            total       = (wrist_score*0.50 + max(arm_confs)*0.25 +
                           centrality*0.15  + size_score*0.10)
            candidates.append((total, i, kp_n, kp_r))

        if not candidates: return None, None
        candidates.sort(key=lambda x: x[0], reverse=True)
        _, _, kp_norm, kp_raw = candidates[0]
        return kp_norm, kp_raw

    # Tier 1 — strict (filters crowd in dataset images)
    result = _score_candidates(min_bbox_frac=0.06, min_arm_conf=0.20)
    if result[0] is not None: return result

    # Tier 2 — lenient (catches subjects in wide-angle or high-res external photos)
    result = _score_candidates(min_bbox_frac=0.015, min_arm_conf=0.10)
    if result[0] is not None: return result

    # Tier 3 — any person with any visible keypoints (single-subject photos)
    best_idx, best_total = 0, -1.0
    for i in range(len(results[0].keypoints.xyn)):
        kp_n = results[0].keypoints.xyn[i].cpu().numpy()
        kp_r = results[0].keypoints.data[i].cpu().numpy()
        if kp_n.shape[0] != 17: continue
        body_confs = [kp_conf(kp_r, j) for j in range(17)]
        total      = float(np.mean(body_confs))
        if total > best_total:
            best_total = total; best_idx = i

    if best_total > 0.05:
        kp_norm = results[0].keypoints.xyn[best_idx].cpu().numpy()
        kp_raw  = results[0].keypoints.data[best_idx].cpu().numpy()
        return kp_norm, kp_raw

    return None, None


def _find_shooter_idx(results, frame_shape: Tuple) -> int:
    """Returns index of the shooter in results."""
    img_h, img_w = frame_shape[:2]
    img_area     = img_h * img_w
    best_idx, best_score = 0, -1.0
    for i in range(len(results[0].keypoints.xyn)):
        kp_n = results[0].keypoints.xyn[i].cpu().numpy()
        kp_r = results[0].keypoints.data[i].cpu().numpy()
        if kp_n.shape[0] != 17: continue
        if i < len(results[0].boxes):
            box = results[0].boxes[i].xyxy[0].cpu().numpy()
            bf  = ((box[2]-box[0])*(box[3]-box[1])) / img_area
        else:
            bf = 1.0
        if bf < 0.06: continue
        arm_confs   = [kp_conf(kp_r,j) for j in [KP["r_elbow"],KP["r_wrist"],
                                                   KP["l_elbow"],KP["l_wrist"]]]
        wrist_score = 1.0 - min(float(kp_n[KP["r_wrist"]][1]),float(kp_n[KP["l_wrist"]][1]))
        total       = wrist_score*0.5 + max(arm_confs)*0.3 + min(1.0,bf*8)*0.2
        if total > best_score:
            best_score = total; best_idx = i
    return best_idx


def _draw_only_shooter(results, shooter_idx: int, frame: np.ndarray) -> np.ndarray:
    annotated = frame.copy()
    if results[0].keypoints is None: return annotated
    kp_pixel = results[0].keypoints.xy[shooter_idx].cpu().numpy()
    kp_raw   = results[0].keypoints.data[shooter_idx].cpu().numpy()
    connections = [
        (KP["l_shoulder"],KP["r_shoulder"]),(KP["l_shoulder"],KP["l_elbow"]),
        (KP["l_elbow"],KP["l_wrist"]),(KP["r_shoulder"],KP["r_elbow"]),
        (KP["r_elbow"],KP["r_wrist"]),(KP["l_shoulder"],KP["l_hip"]),
        (KP["r_shoulder"],KP["r_hip"]),(KP["l_hip"],KP["r_hip"]),
        (KP["l_hip"],KP["l_knee"]),(KP["l_knee"],KP["l_ankle"]),
        (KP["r_hip"],KP["r_knee"]),(KP["r_knee"],KP["r_ankle"]),
        (KP["nose"],KP["l_shoulder"]),(KP["nose"],KP["r_shoulder"]),
    ]
    for a, b in connections:
        pa = tuple(kp_pixel[a].astype(int)); pb = tuple(kp_pixel[b].astype(int))
        if (pa[0]>0 and pa[1]>0 and pb[0]>0 and pb[1]>0
                and kp_conf(kp_raw,a)>0.2 and kp_conf(kp_raw,b)>0.2):
            cv2.line(annotated, pa, pb, (0,255,120), 2)
    for j in range(17):
        px,py = int(kp_pixel[j][0]), int(kp_pixel[j][1])
        if px>0 and py>0 and kp_conf(kp_raw,j)>0.2:
            color = (0,200,255) if j in [KP["r_wrist"],KP["l_wrist"]] else (255,100,0)
            cv2.circle(annotated,(px,py),5,color,-1)
    return annotated

# ── Overlay ────────────────────────────────────────────────────────────────────

def score_color(s: int) -> Tuple:
    if s >= 85: return (50, 220, 100)
    if s >= 60: return (50, 200, 255)
    return (50, 80, 230)

def draw_overlay(frame: np.ndarray, m: Dict) -> np.ndarray:
    h, w    = frame.shape[:2]
    pw      = min(320, w // 2)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0,0), (pw,240), (10,10,20), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    overall = m["overall_score"]
    bar_w   = int((pw-20) * overall / 100)
    cv2.rectangle(frame,(10,8),(pw-10,22),(35,35,50),-1)
    cv2.rectangle(frame,(10,8),(10+bar_w,22),score_color(overall),-1)
    cv2.putText(frame,"Score: "+str(overall)+"/100  ["+m["grade"]+"]",
                (10,18),cv2.FONT_HERSHEY_SIMPLEX,0.48,(255,255,255),1)

    phase = m["phase"]
    cam   = m.get("camera_angle","")
    phase_colors = {"loading":(0,180,255),"set":(0,255,160),
                    "release":(80,255,80),"follow_through":(200,255,0),
                    "not_shooting":(150,150,150)}
    phase_label = phase.replace("_"," ") + (" [side view]" if cam=="sideways" else "")
    cv2.putText(frame,"Phase: "+phase_label,(10,36),
                cv2.FONT_HERSHEY_SIMPLEX,0.45,phase_colors.get(phase,(200,200,200)),1)
    cv2.putText(frame,"Side:  "+m["side"],(10,52),
                cv2.FONT_HERSHEY_SIMPLEX,0.44,(180,180,180),1)

    # FIX #4: use correct score keys
    rows = [
        ("Elbow",    m["elbow_angle"],         m["scores"]["elbow_angle"],          "deg"),
        ("Knee",     m["knee_angle"],           m["scores"]["knee_angle"],           "deg"),
        ("Shoulder", m["shoulder_angle"],       m["scores"]["shoulder_angle"],       "deg"),
        ("W/E vert", m["wrist_elbow_vertical"], m["scores"]["wrist_elbow_vertical"], "deg"),
    ]
    y = 70
    for label, val, sc, unit in rows:
        cv2.putText(frame, label+": "+f"{val:.1f}"+unit+"  ["+str(sc)+"]",
                    (10,y),cv2.FONT_HERSHEY_SIMPLEX,0.42,score_color(sc),1)
        y += 18

    ml = m.get("ml", {})
    matches = ml.get("knn_matches", [])
    if matches:
        best = matches[0]
        cv2.putText(frame,"Like: "+best["name"]+" ("+str(best["similarity"])+"%)",
                    (10,170),cv2.FONT_HERSHEY_SIMPLEX,0.40,(200,220,255),1)

    worst    = min(m["scores"], key=lambda k: m["scores"][k])
    hint     = m["feedback"].get(worst,"")
    words, line, lines = hint.split(),"",[]
    for ww in words:
        if len(line)+len(ww)+1>45: lines.append(line); line=ww
        else: line+=(" " if line else "")+ww
    if line: lines.append(line)
    y = 188
    for ln in lines[:2]:
        cv2.putText(frame,ln,(10,y),cv2.FONT_HERSHEY_SIMPLEX,0.37,(255,220,80),1)
        y += 14
    return frame

# ── Single image ───────────────────────────────────────────────────────────────

def process_image(image_path: Path) -> Optional[Dict]:
    frame = cv2.imread(str(image_path))
    if frame is None:
        print("  [skip] Cannot read: " + image_path.name)
        return None

    frame = ensure_color(frame)

    # Resize very large images (phone photos etc.) to max 1280px wide
    # so bbox_frac calculations are consistent and inference is faster
    h, w = frame.shape[:2]
    if w > 1280:
        scale = 1280 / w
        frame = cv2.resize(frame, (1280, int(h * scale)))

    # Try four confidence levels — ensures external images always work
    kp_norm, kp_raw, results = None, None, None
    for conf in [0.45, 0.30, 0.20, 0.10]:
        results  = model(frame, verbose=False, conf=conf)
        kp_norm, kp_raw = pick_shooter(results, frame.shape)
        if kp_norm is not None:
            break

    if kp_norm is None:
        # Absolute last resort: run on resized smaller version
        small    = cv2.resize(frame, (640, int(frame.shape[0] * 640 / frame.shape[1])))
        results  = model(small, verbose=False, conf=0.10)
        kp_norm, kp_raw = pick_shooter(results, small.shape)
        if kp_norm is None:
            # Write original with "no pose" label so user gets feedback
            out = frame.copy()
            cv2.putText(out, "No person detected — try a clearer image",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 80, 255), 2)
            cv2.imwrite(str(OUTPUT_IMAGES / image_path.name), out)
            return None
        frame = small   # use the small frame for annotation

    shooter_idx = _find_shooter_idx(results, frame.shape)
    annotated   = _draw_only_shooter(results, shooter_idx, frame)
    metrics     = extract_metrics(kp_norm, kp_raw)

    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from analysis.ml_model import BasketballMLPredictor
        predictor     = BasketballMLPredictor()
        metrics["ml"] = predictor.predict(metrics)
    except Exception as e:
        metrics["ml"] = {"error": str(e), "knn_matches": [], "knn_summary": ""}

    annotated = draw_overlay(annotated, metrics)
    metrics["image"] = image_path.name
    cv2.imwrite(str(OUTPUT_IMAGES / image_path.name), annotated)
    return metrics

# ── Batch ──────────────────────────────────────────────────────────────────────

def run_on_folder(folder: Path, max_images: int = 200) -> List[Dict]:
    images = sorted([p for p in folder.iterdir()
                     if p.suffix.lower() in IMAGE_EXTENSIONS])
    if not images: print("No images found in "+str(folder)); return []
    if len(images) > max_images: images = images[:max_images]
    print("Found "+str(len(images))+" images\n")
    all_metrics = []
    for i,p in enumerate(images,1):
        m   = process_image(p)
        tag = "["+str(i).rjust(4)+"/"+str(len(images))+"] "+p.name
        if m:
            all_metrics.append(m)
            print(tag+"  score="+str(m["overall_score"])+"  phase="+m["phase"])
        else:
            print(tag+"  -> no shooter")
    return all_metrics

def print_and_save_summary(all_metrics: List[Dict], source_label: str):
    if not all_metrics: print("\nNo pose data."); return
    from collections import Counter
    OUTPUT_METRICS.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_METRICS / "metrics_summary.json"
    summary = {
        "source": source_label, "total_images": len(all_metrics),
        "avg_overall_score": round(float(np.mean([m["overall_score"] for m in all_metrics])),1),
        "phases": dict(Counter(m["phase"] for m in all_metrics)),
    }
    with open(out,"w") as f:
        json.dump({"summary":summary,"per_image":all_metrics},f,indent=2)
    print("\nAvg score: "+str(summary["avg_overall_score"])+"/100")
    print("Saved: "+str(out))

def main():
    OUTPUT_IMAGES.mkdir(parents=True, exist_ok=True)
    OUTPUT_METRICS.mkdir(parents=True, exist_ok=True)
    parser = argparse.ArgumentParser()
    group  = parser.add_mutually_exclusive_group()
    group.add_argument("--folder", type=str)
    group.add_argument("--image",  type=str)
    parser.add_argument("--max", type=int, default=200)
    args = parser.parse_args()
    if args.image:
        p = Path(args.image)
        if not p.exists(): print("Not found: "+str(p)); return
        m = process_image(p)
        if m: print_and_save_summary([m], str(p))
        else: print("No shooter detected.")
    else:
        folder = Path(args.folder) if args.folder else Path("dataset/test/images")
        if not folder.exists(): print("Not found: "+str(folder)); return
        print_and_save_summary(run_on_folder(folder,args.max), str(folder))

if __name__ == "__main__":
    main()