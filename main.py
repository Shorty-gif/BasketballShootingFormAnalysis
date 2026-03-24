"""
main.py
--------
AI Basketball Coach — entry point.

Usage:
    python main.py                                           # test set
    python main.py --folder dataset/train/images
    python main.py --folder dataset/test/images --gemini
    python main.py --image dataset/test/images/shot.jpg --gemini

Set your Gemini API key:
    export GEMINI_API_KEY="your_key_here"
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path

# ── Gemini — new SDK ───────────────────────────────────────────────────────────
try:
    from google import genai as google_genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

sys.path.insert(0, str(Path(__file__).parent))

from inference.image_inference import run_on_folder, process_image, print_and_save_summary
from inference.image_inference import OUTPUT_METRICS
from analysis.shooting_metrics import aggregate_session
from analysis.pose_analysis    import build_gemini_prompt

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

# Models tried in order — flash-lite uses least quota, best for free tier
GEMINI_MODELS = [
    "gemini-1.5-flash",
    "gemini-1.5-flash-8b",
    "gemini-2.0-flash",
]


def get_gemini_coaching(session_data: dict, per_image: list) -> str:
    if not GEMINI_AVAILABLE:
        return ("google-genai not installed.\n"
                "Run: pip install google-genai --break-system-packages")

    api_key = GEMINI_API_KEY
    if not api_key:
        return ("No Gemini API key found.\n"
                "Set it with: export GEMINI_API_KEY='your_key_here'")

    try:
        client     = google_genai.Client(api_key=api_key)
        prompt     = build_gemini_prompt(session_data, per_image)
        last_error = ""

        for model_name in GEMINI_MODELS:
            try:
                print("Trying model: " + model_name + "...")
                response = client.models.generate_content(
                    model=model_name, contents=prompt
                )
                print("Success with: " + model_name)
                return response.text
            except Exception as e:
                last_error = str(e)
                if "429" in last_error or "quota" in last_error.lower() or \
                   "resource_exhausted" in last_error.lower():
                    print("  Quota hit on " + model_name + " — trying next model...")
                    time.sleep(2)
                    continue
                else:
                    break   # non-quota error, don't retry

        # All models failed
        if "429" in last_error or "quota" in last_error.lower():
            return (
                "Gemini quota exceeded on the free tier.\n\n"
                "Free tier limits: ~15 requests/minute, 1,500/day.\n"
                "Solutions:\n"
                "  1. Wait 60 seconds and try again\n"
                "  2. Create a new project at aistudio.google.com and get a fresh key\n"
                "  3. Enable billing on your Google Cloud project for higher limits\n"
                "     (still free up to generous daily limits with billing enabled)"
            )
        return "Gemini error: " + last_error

    except Exception as e:
        return "Gemini error: " + str(e)


def main():
    parser = argparse.ArgumentParser(
        description="AI Basketball Coach — pose analysis + Gemini coaching"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--folder", type=str,
                       help="Folder of images (default: dataset/test)")
    group.add_argument("--image",  type=str,
                       help="Single image file")
    parser.add_argument("--max",    type=int, default=4000)
    parser.add_argument("--gemini", action="store_true",
                        help="Run Gemini AI coaching after analysis")
    args = parser.parse_args()

    print("=" * 54)
    print("  AI Basketball Coach — Pose Analysis")
    print("=" * 54 + "\n")

    if args.image:
        img_path = Path(args.image)
        if not img_path.exists():
            print("File not found: " + str(img_path)); return
        print("Analysing: " + img_path.name + "\n")
        m = process_image(img_path)
        if not m:
            print("No person detected."); return
        all_metrics = [m]
        print_and_save_summary(all_metrics, str(img_path))
    else:
        folder = Path(args.folder) if args.folder else Path("dataset/test")
        if not folder.exists():
            print("Folder not found: " + str(folder)); return
        print("Folder: " + str(folder) + "\n")
        all_metrics = run_on_folder(folder, max_images=args.max)
        print_and_save_summary(all_metrics, str(folder))

    if not all_metrics:
        print("No metrics collected."); return

    print("\nAggregating session data...")
    session = aggregate_session(all_metrics)

    print("\n── Session Report ────────────────────────────────────")
    print("  Frames with pose:   " + str(session["total_frames"]))
    print("  Avg form score:     " + str(session["avg_overall"]) + "/100")
    print("  Grade distribution: " + str(session.get("grade_distribution", {})))
    print("  Phase distribution: " + str(session.get("phase_distribution", {})))
    print("\n  Consistency (% of frames inside ideal range):")
    for key, pct in session.get("consistency", {}).items():
        bar = "█" * int(pct/5) + "░" * (20-int(pct/5))
        print("    " + key.ljust(26) + bar + "  " + str(pct) + "%")
    print("\n  Top issues:")
    for item in session.get("top_issues", []):
        print("    • " + item["issue"] + " (" + str(item["count"]) + " frames)")
    print("──────────────────────────────────────────────────────")

    OUTPUT_METRICS.mkdir(parents=True, exist_ok=True)
    session_path = OUTPUT_METRICS / "session_aggregate.json"
    with open(session_path, "w") as f:
        json.dump(session, f, indent=2)
    print("\nSession saved to: " + str(session_path))

    if args.gemini:
        print("\nCalling Gemini...")
        coaching = get_gemini_coaching(session, all_metrics)
        print("\n" + "=" * 54)
        print("  Gemini Coaching Feedback")
        print("=" * 54)
        print(coaching)
        print("=" * 54)
        coaching_path = OUTPUT_METRICS / "coaching_feedback.txt"
        coaching_path.write_text(coaching)
        print("\nCoaching saved to: " + str(coaching_path))
    else:
        print("\nTip: add --gemini to get AI coaching feedback.")
        print("     Set key: export GEMINI_API_KEY='your_key_here'")


if __name__ == "__main__":
    main()