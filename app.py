"""
app.py  —  AI Basketball Coach  |  Streamlit Frontend
------------------------------------------------------
Place this file at the PROJECT ROOT (same level as main.py).

Run with:
    streamlit run app.py

Three-tab UI:
  Tab 1 — Image Upload  : upload player photos → pose + form score
  Tab 2 — Video Upload  : upload video → auto extract frames → analyse each
  Tab 3 — Live Webcam   : real-time pose via OpenCV (no extra packages needed)
"""

import os
import sys
import cv2
import json
import time
import tempfile
import threading
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict

import streamlit as st

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from inference.image_inference import process_image, OUTPUT_METRICS, OUTPUT_IMAGES
from inference.frame_inference import analyse_frame
from analysis.shooting_metrics import aggregate_session
from analysis.pose_analysis     import build_gemini_prompt

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Basketball Coach",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styles ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@400;600;700;800;900&family=Barlow:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Barlow', sans-serif;
    background-color: #080810;
    color: #e8e4dc;
}

/* ── Hero ── */
.hero {
    background: linear-gradient(160deg, #0d0d1a 0%, #111120 60%, #0a0a14 100%);
    border-bottom: 1px solid rgba(255,107,26,0.2);
    padding: 2.5rem 2rem 2rem;
    text-align: center;
    margin: -1rem -1rem 2rem -1rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -60px; left: 50%;
    transform: translateX(-50%);
    width: 400px; height: 200px;
    background: radial-gradient(ellipse, rgba(255,107,26,0.12) 0%, transparent 70%);
    pointer-events: none;
}
.hero-title {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 3.4rem;
    font-weight: 900;
    letter-spacing: -0.01em;
    color: #fff;
    line-height: 1;
    margin: 0;
    text-transform: uppercase;
}
.hero-title span { color: #ff6b1a; }
.hero-sub {
    font-size: 0.78rem;
    font-weight: 400;
    color: #55556a;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    margin-top: 0.6rem;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: #0e0e1c;
    border-radius: 14px;
    padding: 5px;
    gap: 4px;
    border: 1px solid #1a1a2e;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 1.05rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: #4a4a6a;
    background: transparent;
    border-radius: 10px;
    padding: 0.65rem 1.6rem;
    border: none;
    transition: all 0.15s;
}
.stTabs [aria-selected="true"] {
    background: #ff6b1a !important;
    color: #fff !important;
    box-shadow: 0 4px 20px rgba(255,107,26,0.35);
}
.stTabs [data-baseweb="tab-panel"] { padding-top: 1.8rem; }

/* ── Cards ── */
.card {
    background: #0e0e1c;
    border: 1px solid #1a1a2e;
    border-radius: 18px;
    padding: 1.6rem;
    margin-bottom: 1.2rem;
}
.card-title {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #ff6b1a;
    margin: 0 0 1.1rem;
}

/* ── Score display ── */
.big-score {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 6rem;
    font-weight: 900;
    line-height: 1;
    text-align: center;
    letter-spacing: -0.03em;
}
.score-denom {
    text-align: center;
    font-size: 0.8rem;
    color: #44446a;
    font-weight: 500;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-top: -0.3rem;
}
.grade-pill {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 1rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    padding: 0.22rem 1.1rem;
    border-radius: 999px;
    display: block;
    text-align: center;
    width: fit-content;
    margin: 0.6rem auto 0;
    text-transform: uppercase;
}
.phase-tag {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 0.88rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    padding: 0.2rem 0.9rem;
    border-radius: 999px;
    text-transform: uppercase;
    display: inline-block;
}

/* ── Metric bars ── */
.m-wrap { margin-bottom: 0.85rem; }
.m-head {
    display: flex;
    justify-content: space-between;
    font-size: 0.72rem;
    font-weight: 500;
    letter-spacing: 0.07em;
    text-transform: uppercase;
    color: #55556a;
    margin-bottom: 0.3rem;
}
.m-val { color: #aaa; font-weight: 600; }
.m-track {
    background: #1a1a2e;
    border-radius: 5px;
    height: 7px;
    overflow: hidden;
}
.m-fill { height: 100%; border-radius: 5px; transition: width 0.4s; }

/* ── Alert / feedback boxes ── */
.priority-box {
    background: rgba(255,107,26,0.07);
    border-left: 3px solid #ff6b1a;
    border-radius: 0 10px 10px 0;
    padding: 0.8rem 1rem;
    margin-top: 0.8rem;
    font-size: 0.87rem;
    color: #d4b8a0;
    line-height: 1.55;
}
.flag-box {
    background: rgba(255,50,50,0.06);
    border: 1px solid rgba(255,60,60,0.18);
    border-radius: 8px;
    padding: 0.45rem 0.85rem;
    margin: 0.3rem 0;
    font-size: 0.8rem;
    color: #ff8888;
}
.gemini-box {
    background: rgba(42,255,106,0.04);
    border: 1px solid rgba(42,255,106,0.12);
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    margin-top: 1.2rem;
    font-size: 0.9rem;
    line-height: 1.75;
    color: #b8e8c8;
    white-space: pre-wrap;
}
.info-box {
    background: rgba(56,189,248,0.05);
    border: 1px solid rgba(56,189,248,0.15);
    border-radius: 10px;
    padding: 0.7rem 1rem;
    font-size: 0.85rem;
    color: #7dd3f8;
    margin-bottom: 1rem;
}

/* ── Upload zone ── */
section[data-testid="stFileUploadDropzone"] {
    background: #0e0e1c !important;
    border: 2px dashed #1e1e32 !important;
    border-radius: 16px !important;
    transition: border-color 0.2s;
}
section[data-testid="stFileUploadDropzone"]:hover {
    border-color: #ff6b1a !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #ff6b1a, #e55c0f);
    color: #fff !important;
    font-family: 'Barlow Condensed', sans-serif !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.65rem 1.8rem !important;
    width: 100%;
    box-shadow: 0 4px 20px rgba(255,107,26,0.25);
    transition: all 0.15s !important;
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 28px rgba(255,107,26,0.38) !important;
}

/* ── Progress bar ── */
.stProgress > div > div > div { background: #ff6b1a !important; }

/* ── Metrics (st.metric) ── */
[data-testid="metric-container"] {
    background: #0e0e1c;
    border: 1px solid #1a1a2e;
    border-radius: 14px;
    padding: 1rem 1.2rem;
}
[data-testid="metric-container"] label {
    font-size: 0.7rem !important;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #55556a !important;
}
[data-testid="stMetricValue"] {
    font-family: 'Barlow Condensed', sans-serif !important;
    font-size: 1.8rem !important;
    font-weight: 800 !important;
    color: #fff !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0a0a14 !important;
    border-right: 1px solid #1a1a2e;
}
[data-testid="stSidebar"] .block-container { padding-top: 2rem; }

/* ── Misc ── */
hr { border-color: #1a1a2e !important; }
footer, #MainMenu { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-title">🏀 AI <span>Basketball</span> Coach</div>
  <div class="hero-sub">YOLOv8 Pose Estimation · Biomechanical Scoring · Gemini AI Feedback</div>
</div>
""", unsafe_allow_html=True)


# ── Utility functions ──────────────────────────────────────────────────────────

def score_col(s: int) -> str:
    if s >= 85: return "#2aff6a"
    if s >= 60: return "#ffb830"
    return "#ff4444"

def grade_col(g: str) -> str:
    return {"A":"#2aff6a","B":"#7aee44","C":"#ffb830","D":"#ff7a1a","F":"#ff4444"}.get(g,"#888")

def phase_col(p: str) -> str:
    return {"loading":"#38bdf8","set":"#2aff6a","release":"#ffb830","follow_through":"#c084fc"}.get(p,"#777")

def get_grade(score: int) -> str:
    if score >= 88: return "A"
    if score >= 75: return "B"
    if score >= 60: return "C"
    if score >= 45: return "D"
    return "F"

def render_score_panel(m: Dict):
    """Full scoring panel for one analysed frame."""
    overall  = m.get("overall_score", 0)
    grade    = m.get("grade", get_grade(overall))
    phase    = m.get("phase", "unknown")
    side     = m.get("side",  "unknown")
    scores   = m.get("scores", {})
    feedback = m.get("feedback", {})

    col_score, col_metrics = st.columns([1, 2], gap="medium")

    with col_score:
        st.markdown(f"""
        <div class="card" style="text-align:center;min-height:260px;display:flex;
             flex-direction:column;align-items:center;justify-content:center">
          <div class="card-title">Form Score</div>
          <div class="big-score" style="color:{score_col(overall)}">{overall}</div>
          <div class="score-denom">out of 100</div>
          <div class="grade-pill" style="background:{grade_col(grade)}18;
               color:{grade_col(grade)};border:1px solid {grade_col(grade)}40">
            Grade &nbsp;{grade}
          </div>
          <div style="margin-top:1.1rem">
            <span class="phase-tag" style="background:{phase_col(phase)}18;
                  color:{phase_col(phase)};border:1px solid {phase_col(phase)}40">
              {phase.replace("_"," ").title()}
            </span>
          </div>
          <div style="margin-top:0.6rem;font-size:0.75rem;color:#44446a;
               letter-spacing:0.08em;text-transform:uppercase">
            {'Right arm' if side=='right' else 'Left arm'}
          </div>
        </div>
        """, unsafe_allow_html=True)

    with col_metrics:
        metrics_def = [
            ("elbow_angle",          f"Elbow angle",          m.get("elbow_angle",0),        "° (ideal 80–100°)"),
            ("knee_angle",           f"Knee bend",            m.get("knee_angle",0),         "° (ideal 100–130°)"),
            ("shoulder_angle",       f"Shoulder elevation",   m.get("shoulder_angle",0),     "° (ideal 45–75°)"),
            ("wrist_elbow_vertical", f"Wrist/elbow vertical", m.get("wrist_elbow_vertical",0),"° off vertical"),
            ("hip_knee_alignment",   f"Hip–knee alignment",   m.get("hip_knee_alignment",0), ""),
        ]
        bars_html = '<div class="card"><div class="card-title">Joint Analysis</div>'
        for key, label, val, unit in metrics_def:
            sc  = scores.get(key, 50)
            fmt = f"{val:.3f}" if key == "hip_knee_alignment" else f"{val:.1f}"
            bars_html += f"""
            <div class="m-wrap">
              <div class="m-head">
                <span>{label}</span>
                <span class="m-val">{fmt}{unit}</span>
              </div>
              <div class="m-track">
                <div class="m-fill" style="width:{sc}%;background:{score_col(sc)}"></div>
              </div>
            </div>"""
        bars_html += "</div>"
        st.markdown(bars_html, unsafe_allow_html=True)

    # Priority coaching cue
    if feedback and scores:
        worst = min(scores, key=lambda k: scores[k])
        cue   = feedback.get(worst, "")
        if cue:
            st.markdown(f'<div class="priority-box">💡 <strong>Priority:</strong> {cue}</div>',
                        unsafe_allow_html=True)

    # Red flags
    from analysis.pose_analysis import detect_red_flags
    flags = detect_red_flags(m)
    if flags:
        st.markdown("**⚠️ Form Alerts**")
        for flag in flags:
            st.markdown(f'<div class="flag-box">⚡ {flag}</div>', unsafe_allow_html=True)


def render_session_summary(session: Dict):
    st.markdown("---")
    st.markdown("### 📊 Session Summary")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Avg Score",       str(session.get("avg_overall", 0)) + " / 100")
    c2.metric("Frames Analysed", str(session.get("total_frames", 0)))
    grades = session.get("grade_distribution", {})
    top_g  = max(grades, key=grades.get) if grades else "—"
    c3.metric("Most Common Grade", top_g)
    phases = session.get("phase_distribution", {})
    dom_p  = max(phases, key=phases.get) if phases else "—"
    c4.metric("Dominant Phase", dom_p.replace("_", " ").title())

    st.markdown("**Consistency — % of frames inside ideal range**")
    for key, pct in session.get("consistency", {}).items():
        label = key.replace("_", " ").title()
        col_a, col_b = st.columns([4, 1])
        with col_a:
            st.progress(int(pct))
        with col_b:
            st.markdown(f"<div style='color:{score_col(int(pct))};font-weight:700;"
                        f"font-size:0.9rem'>{pct}%</div>", unsafe_allow_html=True)
        st.caption(label)

    issues = session.get("top_issues", [])
    if issues:
        st.markdown("**Recurring issues across all frames:**")
        for item in issues:
            st.markdown(f"- {item['issue']} *({item['count']} frames)*")


def call_gemini(session: Dict, all_metrics: List[Dict]) -> Optional[str]:
    api_key = st.session_state.get("gemini_key", "").strip()
    if not api_key:
        return None

    try:
        from google import genai as gai
        client = gai.Client(api_key=api_key)
        prompt = build_gemini_prompt(session, all_metrics)

        # Try models in order — flash-lite uses the least quota
        models_to_try = [
            "gemini-2.5-flash",
            "gemini-2.5-flash-8b",
            "gemini-2.0-flash",
        ]
        last_error = ""
        for model_name in models_to_try:
            try:
                response = client.models.generate_content(
                    model=model_name, contents=prompt
                )
                return response.text
            except Exception as e:
                last_error = str(e)
                err_lower  = last_error.lower()
                # Only try next model on quota/rate errors
                if "429" in last_error or "resource_exhausted" in err_lower or "quota" in err_lower:
                    import time
                    time.sleep(2)   # small pause before retry
                    continue
                else:
                    # Non-quota error — don't retry with other models
                    break

        # All models failed — give helpful message
        if "429" in last_error or "resource_exhausted" in last_error.lower():
            return (
                "⚠️ Gemini quota exceeded on the free tier.\n\n"
                "The free tier allows ~15 requests/minute and 1,500/day. "
                "Wait 60 seconds and try again, or:\n"
                "1. Go to https://aistudio.google.com\n"
                "2. Create a new project and generate a fresh API key\n"
                "3. Paste the new key in the sidebar\n\n"
                "Alternatively, enable billing on your Google Cloud project "
                "for higher limits."
            )
        return "Gemini error: " + last_error

    except Exception as e:
        return "Gemini error: " + str(e)




def render_ml_panel(m: Dict):
    """Renders XGBoost score + k-NN pro player matches."""
    ml = m.get("ml", {})
    if not ml:
        return

    xgb_score   = ml.get("xgb_score")
    knn_matches = ml.get("knn_matches", [])

    # Always try to show k-NN even if XGBoost not trained
    if not knn_matches:
        # Try running k-NN directly if ml dict has no matches
        try:
            from analysis.ml_model import BasketballMLPredictor
            predictor  = BasketballMLPredictor()
            result     = predictor.predict(m)
            knn_matches = result.get("knn_matches", [])
            xgb_score   = result.get("xgb_score") or xgb_score
            ml["knn_matches"] = knn_matches
            ml["knn_summary"] = result.get("knn_summary","")
        except Exception:
            pass

    if xgb_score is None and not knn_matches:
        return

    st.markdown("---")
    col_xgb, col_knn = st.columns([1, 2], gap="medium")

    with col_xgb:
        if xgb_score is not None:
            st.markdown(f"""
            <div class="card" style="text-align:center">
              <div class="card-title">XGBoost Score</div>
              <div class="big-score" style="color:{score_col(int(xgb_score))};font-size:4rem">
                {xgb_score:.0f}
              </div>
              <div class="score-denom">ML-predicted / 100</div>
              <div style="margin-top:0.8rem;font-size:0.82rem;color:#aaa;line-height:1.5">
                {ml.get("xgb_explanation","")}
              </div>
            </div>
            """, unsafe_allow_html=True)

    with col_knn:
        if knn_matches:
            st.markdown(
                '<div class="card"><div class="card-title">Closest Pro Player Matches</div>',
                unsafe_allow_html=True
            )
            summary = ml.get("knn_summary","")
            if summary:
                st.markdown(f'<div class="priority-box">{summary}</div>',
                            unsafe_allow_html=True)
            for match in knn_matches[:3]:
                sim = match["similarity"]
                col = score_col(int(sim))
                advice_html = ""
                for adv in match.get("advice", []):
                    advice_html += f'<div style="font-size:0.75rem;color:#888;margin-top:0.2rem">• {adv}</div>'
                st.markdown(f"""
                <div style="margin-bottom:0.9rem;padding-bottom:0.9rem;
                     border-bottom:1px solid #1a1a2e">
                  <div style="display:flex;justify-content:space-between;
                       align-items:center;margin-bottom:0.3rem">
                    <span style="font-family:Barlow Condensed,sans-serif;
                          font-size:1rem;font-weight:700;color:#e8e4dc">
                      {match["rank"]}. {match["name"]}
                    </span>
                    <span style="font-size:0.8rem;font-weight:700;color:{col}">
                      {sim}% match
                    </span>
                  </div>
                  <div style="font-size:0.75rem;color:#55556a;
                       text-transform:uppercase;letter-spacing:0.06em">
                    {match["style"]} &nbsp;·&nbsp; {match["team"]}
                  </div>
                  {advice_html}
                </div>
                """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    key_val = st.text_input(
        "Gemini API Key",
        type="password",
        placeholder="Paste your key...",
        help="Free key at aistudio.google.com",
    )
    if key_val:
        st.session_state["gemini_key"] = key_val
        st.success("Key saved ✓")

    st.markdown("---")
    st.markdown("""
**Get a free Gemini key:**
1. Visit [aistudio.google.com](https://aistudio.google.com)
2. Sign in with Google
3. Click **Get API key**
4. Paste above
""")
    st.markdown("---")
    st.markdown("**Stack**")
    st.caption("YOLOv8n-pose · OpenCV · Streamlit · Gemini 2.0 Flash")


# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_img, tab_vid, tab_cam = st.tabs([
    "📷  Image Upload",
    "🎬  Video Upload",
    "📹  Live Webcam",
])


# ═══════════════════════════════════════════════════════════════════
# TAB 1 — IMAGE UPLOAD
# ═══════════════════════════════════════════════════════════════════
with tab_img:
    st.markdown("#### Upload player photos")
    st.markdown(
        '<div class="info-box">Upload one or more shooting images. '
        'The system detects body keypoints, calculates joint angles, '
        'and scores the form against NBA biomechanical standards.</div>',
        unsafe_allow_html=True,
    )

    uploaded_imgs = st.file_uploader(
        "Drop images here or click to browse",
        type=["jpg", "jpeg", "png", "webp", "bmp"],
        accept_multiple_files=True,
        key="img_up",
    )

    gemini_img = st.checkbox("Get Gemini AI coaching after analysis", key="gem_img")

    if uploaded_imgs:
        if st.button("▶  Analyse Images", key="btn_img"):
            OUTPUT_IMAGES.mkdir(parents=True, exist_ok=True)
            all_metrics = []
            prog        = st.progress(0)
            status      = st.empty()

            for i, f in enumerate(uploaded_imgs):
                status.markdown(f"Analysing **{f.name}** ({i+1}/{len(uploaded_imgs)})…")

                with tempfile.NamedTemporaryFile(
                    suffix=Path(f.name).suffix, delete=False
                ) as tmp:
                    tmp.write(f.read())
                    tmp_path = Path(tmp.name)

                metrics = process_image(tmp_path)
                tmp_path.unlink(missing_ok=True)
                prog.progress((i + 1) / len(uploaded_imgs))

                st.markdown(f"---\n#### {f.name}")
                if metrics:
                    all_metrics.append(metrics)
                    ann = OUTPUT_IMAGES / f.name
                    if ann.exists():
                        st.image(str(ann), use_container_width=True)
                    render_score_panel(metrics)
                    render_ml_panel(metrics)
                else:
                    st.warning(f"No person detected in **{f.name}**")

            prog.empty()
            status.empty()

            # Session summary for multiple images
            if len(all_metrics) > 1:
                session = aggregate_session(all_metrics)
                render_session_summary(session)
            else:
                session = aggregate_session(all_metrics) if all_metrics else {}

            # Gemini coaching — works for single image OR multiple
            if gemini_img and all_metrics:
                if st.session_state.get("gemini_key"):
                    with st.spinner("Calling Gemini AI coach…"):
                        coaching = call_gemini(session, all_metrics)
                    if coaching:
                        st.markdown("### 🤖 Gemini Coaching Feedback")
                        st.markdown(f'<div class="gemini-box">{coaching}</div>',
                                    unsafe_allow_html=True)
                else:
                    st.info("Paste your Gemini API key in the sidebar to get coaching.")


# ═══════════════════════════════════════════════════════════════════
# TAB 2 — VIDEO UPLOAD
# ═══════════════════════════════════════════════════════════════════
with tab_vid:
    st.markdown("#### Upload a shooting video")
    st.markdown(
        '<div class="info-box">Upload an mp4, mov, or avi file. '
        'Frames are extracted automatically using OpenCV. '
        'Pose estimation runs on each sampled frame.</div>',
        unsafe_allow_html=True,
    )

    uploaded_vid = st.file_uploader(
        "Drop a video here",
        type=["mp4", "mov", "avi", "mkv"],
        accept_multiple_files=False,
        key="vid_up",
    )

    col1, col2 = st.columns(2)
    with col1:
        frame_skip = st.slider(
            "Analyse every Nth frame",
            min_value=1, max_value=30, value=5,
            help="5 = balanced speed/accuracy. 1 = max accuracy but slow.",
        )
    with col2:
        max_frames = st.slider(
            "Max frames to analyse",
            min_value=20, max_value=600, value=150,
        )

    gemini_vid = st.checkbox("Get Gemini AI coaching after analysis", key="gem_vid")

    if uploaded_vid:
        if st.button("▶  Extract Frames & Analyse", key="btn_vid"):

            # Write to temp file — OpenCV needs a real file path
            suffix = Path(uploaded_vid.name).suffix or ".mp4"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(uploaded_vid.read())
                tmp_path = Path(tmp.name)

            cap = cv2.VideoCapture(str(tmp_path))

            if not cap.isOpened():
                st.error(
                    "Could not open the video file. "
                    "Please try re-exporting as H.264 mp4."
                )
                tmp_path.unlink(missing_ok=True)
            else:
                total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps    = cap.get(cv2.CAP_PROP_FPS) or 30
                dur    = total / fps
                width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                st.markdown(
                    f'<div class="info-box">'
                    f'<strong>{uploaded_vid.name}</strong> &nbsp;·&nbsp; '
                    f'{total} frames &nbsp;·&nbsp; {fps:.0f} fps &nbsp;·&nbsp; '
                    f'{dur:.1f}s &nbsp;·&nbsp; {width}×{height}'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                all_metrics: List[Dict] = []
                sample_frames = []   # list of (frame_idx, rgb_frame, metrics_or_None)

                prog    = st.progress(0)
                status  = st.empty()

                frame_idx = 0
                analysed  = 0

                # Skip first 10% of video (usually setup/walking)
                skip_frames = max(0, int(total * 0.10))
                for _ in range(skip_frames):
                    cap.read()
                    frame_idx += 1

                # Set up annotated video writer
                OUTPUT_VID = Path("outputs/annotated_video")
                if OUTPUT_VID.exists() and not OUTPUT_VID.is_dir():
                    OUTPUT_VID.unlink()   # remove stale file from earlier run
                OUTPUT_VID.mkdir(parents=True, exist_ok=True)
                out_vid_name = Path(uploaded_vid.name).stem + "_analyzed.mp4"
                out_vid_path = OUTPUT_VID / out_vid_name
                writer = cv2.VideoWriter(
                    str(out_vid_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps, (width, height)
                )

                # Read all frames, write annotated video, collect metrics
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    if frame_idx % frame_skip == 0:
                        result = analyse_frame(frame)
                        analysed += 1

                        if result:
                            phase       = result.get("phase", "")
                            is_shooting = result.get("is_shooting", True)
                            ann_frame   = result.pop("annotated_frame", frame)

                            if is_shooting:
                                all_metrics.append(result)
                                if len(sample_frames) < 8 and phase in ["release","follow_through","set"]:
                                    rgb = cv2.cvtColor(ann_frame, cv2.COLOR_BGR2RGB)
                                    sample_frames.append((frame_idx, rgb, result))
                            else:
                                result.pop("annotated_frame", None)

                            writer.write(ann_frame)
                        else:
                            # No pose detected — write original frame with "no pose" label
                            no_pose = frame.copy()
                            cv2.putText(no_pose, "No pose detected", (10, 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100,100,100), 2)
                            writer.write(no_pose)

                        prog.progress(min(analysed / max(max_frames, 1), 1.0))
                        status.markdown(
                            f"Frame **{frame_idx}** &nbsp;·&nbsp; "
                            f"Analysed **{analysed}** &nbsp;·&nbsp; "
                            f"Shooting frames: **{len(all_metrics)}**"
                        )

                        if analysed >= max_frames:
                            break
                    else:
                        # Write unskipped frames as-is to keep video timing correct
                        writer.write(frame)

                    frame_idx += 1

                writer.release()

                cap.release()
                tmp_path.unlink(missing_ok=True)
                prog.empty()
                status.empty()

                st.success(
                    f"✓ Done — {analysed} frames checked, "
                    f"shooting detected in **{len(all_metrics)}** frames."
                )

                # Offer annotated video download
                if out_vid_path.exists():
                    st.markdown("#### 🎬 Download Annotated Video")
                    with open(out_vid_path, "rb") as vf:
                        st.download_button(
                            label="⬇  Download " + out_vid_name,
                            data=vf,
                            file_name=out_vid_name,
                            mime="video/mp4",
                            use_container_width=True,
                        )
                    st.caption("Video saved to: outputs/annotated_video/" + out_vid_name)

                # Sample frame grid
                if sample_frames:
                    st.markdown("#### Sample Annotated Frames")
                    cols = st.columns(min(4, len(sample_frames)))
                    for ci, (fidx, rgb, met) in enumerate(sample_frames[:4]):
                        with cols[ci]:
                            st.image(rgb, caption=f"Frame {fidx}", use_container_width=True)
                            if met:
                                sc = met.get("overall_score", 0)
                                st.markdown(
                                    f"<div style='text-align:center;"
                                    f"color:{score_col(sc)};font-family:Barlow Condensed,"
                                    f"sans-serif;font-size:1.1rem;font-weight:700'>"
                                    f"Score {sc}/100</div>",
                                    unsafe_allow_html=True,
                                )

                if all_metrics:
                    session = aggregate_session(all_metrics)
                    render_session_summary(session)

                    # Best frame
                    best = max(all_metrics, key=lambda x: x.get("overall_score", 0))
                    st.markdown("#### Best Detected Frame")
                    render_score_panel(best)
                    render_ml_panel(best)

                    # Save JSON
                    OUTPUT_METRICS.mkdir(parents=True, exist_ok=True)
                    out = OUTPUT_METRICS / "video_session.json"
                    with open(out, "w") as jf:
                        json.dump({"session": session, "frames": all_metrics}, jf, indent=2)

                    if gemini_vid:
                        if st.session_state.get("gemini_key"):
                            with st.spinner("Calling Gemini…"):
                                coaching = call_gemini(session, all_metrics)
                            if coaching:
                                st.markdown("### 🤖 Gemini Coaching Feedback")
                                st.markdown(f'<div class="gemini-box">{coaching}</div>',
                                            unsafe_allow_html=True)
                                cpath = OUTPUT_METRICS / "video_coaching.txt"
                                with open(cpath, "w") as cf:
                                    cf.write(coaching)
                        else:
                            st.info("Paste your Gemini API key in the sidebar.")
                else:
                    st.warning(
                        "No pose detected in any frame. "
                        "Make sure the player is clearly visible and well-lit."
                    )


# ═══════════════════════════════════════════════════════════════════
# TAB 3 — LIVE WEBCAM  (OpenCV, no extra packages needed)
# ═══════════════════════════════════════════════════════════════════
with tab_cam:
    st.markdown("#### Real-time pose estimation")
    st.markdown(
        '<div class="info-box">'
        'Captures frames from your webcam using OpenCV and runs pose estimation live. '
        'Click <strong>Start Camera</strong>, aim at a player, and watch the scores update.'
        '</div>',
        unsafe_allow_html=True,
    )

    col_ctrl, col_info = st.columns([1, 2])

    with col_ctrl:
        start_cam  = st.button("▶  Start Camera",  key="start_cam")
        stop_cam   = st.button("⏹  Stop Camera",   key="stop_cam")
        cam_index  = st.number_input("Camera index", min_value=0, max_value=5,
                                     value=0, step=1,
                                     help="0 = built-in, 1+ = external")

    with col_info:
        st.markdown("""
**How it works:**
1. Click **Start Camera** — allow browser access if prompted
2. Point the camera at yourself or a player in shooting position
3. Pose keypoints, joint angles, and form score appear below
4. Click **Stop Camera** when done
""")

    # Frame display + metrics placeholder
    frame_ph   = st.empty()
    metrics_ph = st.empty()
    session_ph = st.empty()

    # Session state for webcam
    if "cam_running" not in st.session_state:
        st.session_state["cam_running"] = False
    if "cam_metrics" not in st.session_state:
        st.session_state["cam_metrics"] = []

    if stop_cam:
        st.session_state["cam_running"] = False

    if start_cam:
        st.session_state["cam_running"] = True
        st.session_state["cam_metrics"] = []

        cap = cv2.VideoCapture(int(cam_index))
        if not cap.isOpened():
            st.error(
                f"Cannot open camera {int(cam_index)}. "
                "Try a different camera index or check permissions."
            )
            st.session_state["cam_running"] = False
        else:
            frame_count = 0
            st.info("Camera running… click **Stop Camera** to end the session.")

            while st.session_state["cam_running"]:
                ret, frame = cap.read()
                if not ret:
                    st.warning("Lost camera feed.")
                    break

                # Analyse every 3rd frame for performance
                if frame_count % 3 == 0:
                    result = analyse_frame(frame)
                    if result:
                        ann = result.pop("annotated_frame")
                        rgb = cv2.cvtColor(ann, cv2.COLOR_BGR2RGB)
                        st.session_state["cam_metrics"].append(result)
                    else:
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        result = None

                    # Display annotated frame
                    frame_ph.image(rgb, channels="RGB", use_container_width=True)

                    # Display latest metrics
                    if result:
                        with metrics_ph.container():
                            render_score_panel(result)

                frame_count += 1
                time.sleep(0.03)   # ~30 fps cap

            cap.release()
            st.session_state["cam_running"] = False
            st.success(
                f"Session ended — {len(st.session_state['cam_metrics'])} frames with pose data."
            )

            # Session summary after webcam
            cam_data = st.session_state["cam_metrics"]
            if len(cam_data) > 5:
                with session_ph.container():
                    session = aggregate_session(cam_data)
                    render_session_summary(session)

                    if st.session_state.get("gemini_key"):
                        with st.spinner("Calling Gemini…"):
                            coaching = call_gemini(session, cam_data)
                        if coaching:
                            st.markdown("### 🤖 Gemini Coaching Feedback")
                            st.markdown(
                                f'<div class="gemini-box">{coaching}</div>',
                                unsafe_allow_html=True,
                            )