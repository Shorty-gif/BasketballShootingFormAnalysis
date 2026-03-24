"""
analysis/pose_analysis.py
--------------------------
High-level pose analysis layer.
Sits between raw keypoint extraction (image_inference.py)
and the Gemini coaching layer (main.py).

Provides:
  - symmetry analysis (left vs right side balance)
  - shooting arm dominance detection
  - phase sequence validation
  - red-flag detection (biomechanical risk patterns)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple


# ── Symmetry ───────────────────────────────────────────────────────────────────

def analyse_symmetry(kp: np.ndarray) -> Dict:
    """
    kp: normalised (17, 2) keypoints array.
    Returns left/right symmetry scores and hip/shoulder tilt.
    """
    # Shoulder tilt: angle of shoulder line from horizontal
    r_shoulder = kp[6]
    l_shoulder = kp[5]
    dy = float(r_shoulder[1] - l_shoulder[1])
    dx = float(r_shoulder[0] - l_shoulder[0]) + 1e-8
    shoulder_tilt_deg = float(np.degrees(np.arctan2(abs(dy), abs(dx))))

    # Hip tilt
    r_hip = kp[12]
    l_hip = kp[11]
    dy_h  = float(r_hip[1] - l_hip[1])
    dx_h  = float(r_hip[0] - l_hip[0]) + 1e-8
    hip_tilt_deg = float(np.degrees(np.arctan2(abs(dy_h), abs(dx_h))))

    # Spine alignment: midpoint of shoulders vs midpoint of hips (x deviation)
    shoulder_mid_x = float((r_shoulder[0] + l_shoulder[0]) / 2)
    hip_mid_x      = float((r_hip[0]      + l_hip[0])      / 2)
    spine_lean     = float(abs(shoulder_mid_x - hip_mid_x))

    return {
        "shoulder_tilt_deg": round(shoulder_tilt_deg, 2),
        "hip_tilt_deg":      round(hip_tilt_deg,      2),
        "spine_lean_norm":   round(spine_lean,         4),
        "flags": {
            "shoulder_tilt": shoulder_tilt_deg > 12,
            "hip_tilt":      hip_tilt_deg      > 10,
            "spine_lean":    spine_lean        > 0.08,
        },
    }


# ── Red-flag detection ─────────────────────────────────────────────────────────

def detect_red_flags(metrics: Dict) -> List[str]:
    """
    Detects biomechanical patterns that significantly harm shot quality.
    Returns a list of flag strings (empty = clean form).
    """
    flags = []

    elbow   = metrics.get("elbow_angle",        0)
    knee    = metrics.get("knee_angle",          0)
    shoulder= metrics.get("shoulder_angle",      0)
    wrist_v = metrics.get("wrist_elbow_vertical",0)
    hip_al  = metrics.get("hip_knee_alignment",  0)
    wrist_h = metrics.get("wrist_height_norm",   1)

    # Chicken wing: elbow flares more than 20 deg outside ideal
    if elbow > 120:
        flags.append("CHICKEN_WING: Elbow angle " + f"{elbow:.0f}deg — arm flaring badly")

    # Flat release: wrist not high enough at release
    if metrics.get("phase") == "release" and wrist_h > 0.55:
        flags.append("FLAT_RELEASE: Wrist too low at release point")

    # No knee bend: straight-leg shot
    if knee > 160:
        flags.append("NO_LEG_DRIVE: Knee angle " + f"{knee:.0f}deg — almost fully straight")

    # Sidearm / off-axis release
    if wrist_v > 40:
        flags.append("SIDEARM_RISK: Wrist/elbow axis " + f"{wrist_v:.0f}deg off vertical")

    # Hip drift
    if hip_al > 0.20:
        flags.append("HIP_DRIFT: Hips shifted sideways — " + f"{hip_al:.3f} norm units")

    # Low shoulder (flat arc)
    if shoulder < 30:
        flags.append("LOW_ARC: Shoulder elevation only " + f"{shoulder:.0f}deg — shot will be flat")

    return flags


# ── Phase sequence validation ──────────────────────────────────────────────────

def validate_phase_sequence(phases: List[str]) -> Dict:
    """
    Checks whether a sequence of phase labels (from multiple frames) follows
    the expected shooting motion order: loading → set → release → follow_through.
    Returns a verdict and any anomalies detected.
    """
    expected = ["loading", "set", "release", "follow_through"]
    seen     = []
    for p in phases:
        if not seen or seen[-1] != p:
            seen.append(p)

    # Remove unknowns
    seen = [p for p in seen if p in expected]

    anomalies = []
    for i in range(len(seen) - 1):
        curr_idx = expected.index(seen[i])   if seen[i]   in expected else -1
        next_idx = expected.index(seen[i+1]) if seen[i+1] in expected else -1
        if next_idx < curr_idx:
            anomalies.append("Phase went backward: " + seen[i] + " → " + seen[i+1])

    coverage = len(set(seen) & set(expected))

    return {
        "observed_sequence": seen,
        "phase_coverage":    coverage,          # how many of 4 phases were captured
        "anomalies":         anomalies,
        "clean_sequence":    len(anomalies) == 0,
    }


# ── Full pose report ───────────────────────────────────────────────────────────

def full_pose_report(metrics: Dict, kp: Optional[np.ndarray] = None) -> Dict:
    """
    Combines metric scores, red flags, and symmetry into a single report dict.
    kp is the (17,2) normalised keypoint array — optional but enables symmetry.
    """
    from analysis.shooting_metrics import score_frame

    scoring  = score_frame(metrics)
    red_flags = detect_red_flags(metrics)

    report = {
        "overall_score": scoring["overall_score"],
        "grade":         scoring["grade"],
        "scores":        scoring["scores"],
        "feedback":      scoring["feedback"],
        "priority_cue":  scoring["priority_cue"],
        "red_flags":     red_flags,
        "phase":         metrics.get("phase", "unknown"),
        "side":          metrics.get("side",  "unknown"),
    }

    if kp is not None and kp.shape == (17, 2):
        report["symmetry"] = analyse_symmetry(kp)

    return report


# ── Session summary for Gemini ─────────────────────────────────────────────────

def build_gemini_prompt(session_data: Dict, per_image: List[Dict]) -> str:
    """
    Builds the Gemini prompt from aggregated session data.
    Called by main.py.
    """
    avg    = session_data.get("avg_overall", 0)
    phases = session_data.get("phase_distribution", {})
    issues = session_data.get("top_issues", [])
    cons   = session_data.get("consistency", {})
    recs   = session_data.get("recommendations", [])
    per_m  = session_data.get("per_metric", {})

    # Sample a few highest and lowest scoring frames for context
    sorted_frames = sorted(per_image, key=lambda x: x.get("overall_score", 0))
    worst_frames  = sorted_frames[:3]
    best_frames   = sorted_frames[-3:]

    def frame_summary(f: Dict) -> str:
        return (
            "  score=" + str(f.get("overall_score", "?")) +
            "  phase=" + f.get("phase", "?") +
            "  elbow=" + f"{f.get('elbow_angle', 0):.1f}deg" +
            "  knee=" +  f"{f.get('knee_angle',  0):.1f}deg" +
            "  shoulder=" + f"{f.get('shoulder_angle', 0):.1f}deg"
        )

    lines = [
        "You are an expert NBA-level basketball shooting coach.",
        "Analyse the following biomechanical data from a player's shooting session.",
        "",
        "SESSION OVERVIEW",
        "  Total frames analysed: " + str(session_data.get("total_frames", 0)),
        "  Average form score:    " + str(avg) + "/100",
        "  Phase distribution:    " + str(phases),
        "",
        "JOINT ANGLE AVERAGES (with consistency — % frames in ideal range)",
    ]

    for key, stats in per_m.items():
        consistency = cons.get(key, 0)
        lines.append(
            "  " + key.ljust(26) +
            "avg=" + str(stats["mean"]) +
            "  std=" + str(stats["std"]) +
            "  ideal=" + str(stats.get("ideal_range", "?")) +
            "  consistency=" + str(consistency) + "%"
        )

    lines += [
        "",
        "RECURRING ISSUES (most common across all frames)",
    ]
    for item in issues:
        lines.append("  • " + item["issue"] + " (" + str(item["count"]) + " frames)")

    lines += [
        "",
        "WORST 3 FRAMES",
    ]
    for f in worst_frames:
        lines.append(frame_summary(f))

    lines += [
        "",
        "BEST 3 FRAMES",
    ]
    for f in best_frames:
        lines.append(frame_summary(f))

    lines += [
        "",
        "Please provide:",
        "1. Overall assessment of this player's shooting mechanics (2–3 sentences)",
        "2. The single most important thing to fix RIGHT NOW",
        "3. A specific drill or exercise to fix it",
        "4. Two secondary improvements to work on after the main issue is fixed",
        "5. One thing the player is already doing well — genuine encouragement",
        "",
        "Keep the tone like a real coach — direct, specific, encouraging.",
        "No generic advice. Every point must reference the actual numbers above.",
        "Total response: under 250 words.",
    ]

    return "\n".join(lines)