"""
inference/video_inference.py
-----------------------------
CLI video analysis — saves a fully annotated output video.

Usage:
    python inference/video_inference.py --video /path/to/shot.mp4
    python inference/video_inference.py --webcam
    python inference/video_inference.py --video shot.mp4 --gemini
"""

import cv2
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict

sys.path.insert(0, str(Path(__file__).parent.parent))

from inference.frame_inference import analyse_frame
from analysis.shooting_metrics import aggregate_session

OUTPUT_VID  = Path("outputs/annotated_video")
METRICS_DIR = Path("outputs/prediction_results")


def run_video(source, use_gemini: bool = False,
              frame_skip: int = 3, max_frames: int = 500) -> List[Dict]:

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("ERROR: Cannot open: " + str(source))
        sys.exit(1)

    if OUTPUT_VID.exists() and not OUTPUT_VID.is_dir():
        OUTPUT_VID.unlink()
    OUTPUT_VID.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_name = Path(source).stem + "_analyzed.mp4" if isinstance(source, str) else "webcam_analyzed.mp4"
    out_path = OUTPUT_VID / out_name

    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps, (width, height)
    )

    print(f"Input:  {source}  ({width}x{height}  {fps:.0f}fps  {total} frames)")
    print(f"Output: {out_path}")
    print("Press ESC to stop early.\n")

    # Skip first 10% — usually setup/walking
    skip = max(0, int(total * 0.10))
    for _ in range(skip):
        cap.read()

    all_metrics: List[Dict] = []
    frame_count = 0
    analysed    = 0

    while cap.isOpened() and analysed < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_skip == 0:
            result = analyse_frame(frame)
            analysed += 1

            if result:
                phase       = result.get("phase", "")
                is_shooting = result.get("is_shooting", True)
                ann         = result.pop("annotated_frame", frame)

                if is_shooting:
                    all_metrics.append(result)

                writer.write(ann)
            else:
                # No pose — write original with label
                lbl = frame.copy()
                cv2.putText(lbl, "No pose detected", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100,100,100), 2)
                writer.write(lbl)

            if analysed % 30 == 0:
                print(f"  Frame {frame_count}  analysed={analysed}  "
                      f"shooting={len(all_metrics)}")
        else:
            writer.write(frame)   # keep original for non-sampled frames

        frame_count += 1

        # Show preview window
        cv2.imshow("AI Basketball Coach — ESC to quit", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            print("Stopped early.")
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    print(f"\n── Complete ──────────────────────────────────────")
    print(f"  Frames processed:  {frame_count}")
    print(f"  Shooting frames:   {len(all_metrics)}")
    print(f"  Annotated video:   {out_path}")

    if all_metrics:
        session = aggregate_session(all_metrics)
        print(f"  Avg form score:    {session.get('avg_overall', 0)}/100")

        out_json = METRICS_DIR / "video_session.json"
        with open(out_json, "w") as f:
            json.dump({"session": session, "frames": all_metrics}, f, indent=2)
        print(f"  Metrics JSON:      {out_json}")

        if use_gemini:
            try:
                from main import get_gemini_coaching
                from analysis.pose_analysis import build_gemini_prompt
                coaching = get_gemini_coaching(session, all_metrics)
                print("\n── Gemini Coaching ──────────────────────────────")
                print(coaching)
                cpath = METRICS_DIR / "video_coaching.txt"
                cpath.write_text(coaching)
            except Exception as e:
                print("Gemini error: " + str(e))

    print("──────────────────────────────────────────────────")
    return all_metrics


def main():
    parser = argparse.ArgumentParser(description="AI Basketball Coach — video analysis")
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--video",  type=str, help="Path to video file")
    group.add_argument("--webcam", action="store_true", help="Use webcam")
    parser.add_argument("--gemini",     action="store_true")
    parser.add_argument("--skip",       type=int, default=3,   help="Analyse every Nth frame (default 3)")
    parser.add_argument("--max-frames", type=int, default=500, help="Max frames to analyse")
    args = parser.parse_args()
    run_video(0 if args.webcam else args.video,
              use_gemini=args.gemini,
              frame_skip=args.skip,
              max_frames=args.max_frames)

if __name__ == "__main__":
    main()