"""
analysis/shooting_metrics.py
------------------------------
Biomechanical evaluation of basketball shooting form.
Compares measured joint angles against NBA coaching / sports science standards
and produces structured scores, flags, and coaching cues per image and in aggregate.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np


# ── Reference standards ────────────────────────────────────────────────────────
#
# Sources: NBA coaching manuals, peer-reviewed biomechanics research
# (Okazaki & Rodacki 2012; Nakano et al. 2020; Hayes 2018)
#
# Format: (ideal_min, ideal_max, acceptable_min, acceptable_max)

STANDARDS = {
    "elbow_angle": {
        "ideal":      (80,  100),
        "acceptable": (70,  115),
        "weight":      0.30,
        "name":        "Elbow angle",
        "unit":        "deg",
        "cues": {
            "perfect":    "Elbow perfectly tucked — textbook release mechanics",
            "too_low":    "Elbow too collapsed — restricts wrist snap at release",
            "too_high":   "Chicken wing elbow — arm flares, reducing arc and accuracy",
        },
    },
    "knee_angle": {
        "ideal":      (100, 130),
        "acceptable": (90,  150),
        "weight":      0.20,
        "name":        "Knee bend",
        "unit":        "deg",
        "cues": {
            "perfect":    "Ideal knee bend — legs are powering the shot cleanly",
            "too_low":    "Over-bent knees — energy leaks into stabilisation, not the shot",
            "too_high":   "Legs too straight — shot relies on arms only, loses range",
        },
    },
    "shoulder_angle": {
        "ideal":      (45,  75),
        "acceptable": (35,  90),
        "weight":      0.20,
        "name":        "Shoulder elevation",
        "unit":        "deg",
        "cues": {
            "perfect":    "Shoulder angle drives a high arc — optimal trajectory",
            "too_low":    "Shoulder too low — ball will release flat, low arc",
            "too_high":   "Shoulder over-elevated — creates tension and side-spin risk",
        },
    },
    "wrist_elbow_vertical": {
        "ideal":      (0,   20),
        "acceptable": (0,   35),
        "weight":      0.15,
        "name":        "Wrist–elbow vertical",
        "unit":        "deg",
        "cues": {
            "perfect":    "Wrist stacked above elbow — ideal for a clean, straight release",
            "too_low":    None,
            "too_high":   "Wrist not above elbow — ball will drift sideways at release",
        },
    },
    "hip_knee_alignment": {
        "ideal":      (0.00, 0.08),
        "acceptable": (0.00, 0.18),
        "weight":      0.15,
        "name":        "Hip–knee alignment",
        "unit":        "",
        "cues": {
            "perfect":    "Hips and knees aligned — full kinetic chain engaged",
            "too_low":    None,
            "too_high":   "Hips drifting sideways — breaks the power chain from legs",
        },
    },
}


# ── Scoring ────────────────────────────────────────────────────────────────────

def score_metric(value: float, ideal: Tuple, acceptable: Tuple,
                 lower_is_better: bool = False) -> int:
    """
    Returns 0–100 score for a single metric value.
    - 85–100 : inside ideal range
    - 50–84  : inside acceptable range only
    -  0–49  : outside acceptable range
    """
    if lower_is_better:
        _, imax = ideal
        _, amax = acceptable
        if value <= imax:
            return int(85 + 15 * max(0.0, 1.0 - value / (imax + 1e-8)))
        elif value <= amax:
            return int(50 + 35 * (amax - value) / (amax - imax + 1e-8))
        else:
            return max(0, int(50 * (amax * 2.0 - value) / (amax + 1e-8)))
    else:
        imin, imax = ideal
        amin, amax = acceptable
        if imin <= value <= imax:
            centre = (imin + imax) / 2.0
            spread = (imax - imin) / 2.0 + 1e-8
            return int(85 + 15 * max(0.0, 1.0 - abs(value - centre) / spread))
        elif amin <= value <= amax:
            if value < imin:
                return int(50 + 35 * (value - amin) / (imin - amin + 1e-8))
            else:
                return int(50 + 35 * (amax - value) / (amax - imax + 1e-8))
        else:
            margin = 30.0
            if value < amin:
                return max(0, int(50 * (value - (amin - margin)) / margin))
            else:
                return max(0, int(50 * ((amax + margin) - value) / margin))


def score_frame(metrics: Dict) -> Dict:
    """
    Takes a metrics dict (from image_inference.extract_metrics) and returns
    a full scoring report with per-metric scores, overall score, and coaching cues.
    """
    lower_is_better_keys = {"hip_knee_alignment", "wrist_elbow_vertical"}
    scores   = {}
    feedback = {}

    for key, std in STANDARDS.items():
        if key not in metrics:
            continue
        val  = metrics[key]
        lb   = key in lower_is_better_keys
        sc   = score_metric(val, std["ideal"], std["acceptable"], lb)
        scores[key] = sc

        imin, imax = std["ideal"]
        amin, amax = std["acceptable"]
        if lb:
            _, imax_ = std["ideal"]
            _, amax_ = std["acceptable"]
            if val <= imax_:
                cue = std["cues"]["perfect"]
            else:
                cue = std["cues"]["too_high"] or std["cues"]["perfect"]
        else:
            if imin <= val <= imax:
                cue = std["cues"]["perfect"]
            elif val < amin:
                cue = std["cues"]["too_low"] or std["cues"]["perfect"]
            else:
                cue = std["cues"]["too_high"] or std["cues"]["perfect"]
        feedback[key] = cue

    # Weighted overall
    total_w = sum(STANDARDS[k]["weight"] for k in scores)
    overall = int(sum(scores[k] * STANDARDS[k]["weight"] for k in scores) / total_w)

    # Priority coaching cue = worst-scoring metric
    worst_key = min(scores, key=lambda k: scores[k])
    priority_cue = feedback[worst_key]

    # Grade
    if overall >= 88:    grade = "A"
    elif overall >= 75:  grade = "B"
    elif overall >= 60:  grade = "C"
    elif overall >= 45:  grade = "D"
    else:                grade = "F"

    return {
        "overall_score":  overall,
        "grade":          grade,
        "scores":         scores,
        "feedback":       feedback,
        "priority_cue":   priority_cue,
        "worst_metric":   worst_key,
    }


# ── Aggregate analysis ─────────────────────────────────────────────────────────

def aggregate_session(all_metrics: List[Dict]) -> Dict:
    """
    Aggregates per-frame metrics across a full image set / session.
    Returns averages, consistency scores, phase distribution, and top issues.
    """
    if not all_metrics:
        return {}

    from collections import Counter

    angle_keys = ["elbow_angle", "knee_angle", "shoulder_angle",
                  "wrist_elbow_vertical", "hip_knee_alignment"]

    aggregated: Dict = {
        "total_frames":     len(all_metrics),
        "avg_overall":      0.0,
        "grade_distribution": {},
        "phase_distribution": {},
        "per_metric": {},
        "top_issues": [],
        "consistency": {},
        "recommendations": [],
    }

    # Overall score stats
    overall_scores = [m.get("overall_score", 0) for m in all_metrics]
    aggregated["avg_overall"] = round(float(np.mean(overall_scores)), 1)

    # Grade distribution
    grades = []
    for sc in overall_scores:
        if sc >= 88:    grades.append("A")
        elif sc >= 75:  grades.append("B")
        elif sc >= 60:  grades.append("C")
        elif sc >= 45:  grades.append("D")
        else:           grades.append("F")
    aggregated["grade_distribution"] = dict(Counter(grades))

    # Phase distribution
    phases = [m.get("phase", "unknown") for m in all_metrics]
    aggregated["phase_distribution"] = dict(Counter(phases))

    # Per-metric averages + consistency (lower std = more consistent)
    for key in angle_keys:
        vals = [m[key] for m in all_metrics if key in m]
        if not vals:
            continue
        std_ref = STANDARDS.get(key, {})
        ideal   = std_ref.get("ideal", (0, 180))
        aggregated["per_metric"][key] = {
            "mean":        round(float(np.mean(vals)), 2),
            "std":         round(float(np.std(vals)),  2),
            "min":         round(float(np.min(vals)),  2),
            "max":         round(float(np.max(vals)),  2),
            "ideal_range": ideal,
        }
        # Consistency: what % of frames are inside ideal range
        if STANDARDS.get(key):
            imin, imax = STANDARDS[key]["ideal"]
            in_ideal = sum(1 for v in vals if imin <= v <= imax)
            aggregated["consistency"][key] = round(100 * in_ideal / len(vals), 1)

    # Top 3 recurring issues (from worst-metric per frame)
    issues = []
    for m in all_metrics:
        if "scores" in m:
            worst = min(m["scores"], key=lambda k: m["scores"][k])
            if "feedback" in m:
                issues.append(m["feedback"].get(worst, ""))
    top_issues = Counter(issues).most_common(3)
    aggregated["top_issues"] = [{"issue": i, "count": c} for i, c in top_issues]

    # Session recommendations
    recs = []
    for key, cons in aggregated["consistency"].items():
        if cons < 50:
            cue = STANDARDS[key]["cues"]["too_high"] or STANDARDS[key]["cues"]["too_low"]
            if cue:
                recs.append("Work on " + STANDARDS[key]["name"] + ": " + cue)
    aggregated["recommendations"] = recs[:3]   # top 3

    return aggregated


# ── Quick CLI for standalone use ───────────────────────────────────────────────

if __name__ == "__main__":
    import json, sys
    if len(sys.argv) < 2:
        print("Usage: python analysis/shooting_metrics.py outputs/prediction_results/metrics_summary.json")
        sys.exit(0)

    with open(sys.argv[1]) as f:
        data = json.load(f)

    per_image = data.get("per_image", [])
    agg       = aggregate_session(per_image)

    print("\n══ Session Report ══════════════════════════════════")
    print("  Frames analysed:  " + str(agg["total_frames"]))
    print("  Avg form score:   " + str(agg["avg_overall"]) + "/100")
    print("  Grades:           " + str(agg["grade_distribution"]))
    print("  Phases:           " + str(agg["phase_distribution"]))
    print("\n  Consistency (% frames in ideal range):")
    for k, v in agg["consistency"].items():
        bar = "█" * int(v / 5) + "░" * (20 - int(v / 5))
        print("    " + k.ljust(24) + bar + "  " + str(v) + "%")
    print("\n  Top issues:")
    for item in agg["top_issues"]:
        print("    • " + item["issue"] + " (" + str(item["count"]) + " frames)")
    print("\n  Recommendations:")
    for r in agg["recommendations"]:
        print("    → " + r)
    print("════════════════════════════════════════════════════")