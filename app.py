from __future__ import annotations
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent


def _set_workspace_page(destination: str) -> None:
    """Update the sidebar radio value from a widget callback."""
    st.session_state["workspace_page"] = destination
sys.path.insert(0, str(ROOT / "src"))
from fairxplain.config import DATA_FILE, MODELS_DIR, REPORTS_DIR, FIGURES_DIR
from fairxplain.data import load_dataset
from fairxplain.features import engineer_features
from fairxplain.counterfactual import nearest_observed_contrast
from fairxplain.explain import local_explanation
from fairxplain.evidence import retrieve_feature_evidence

st.set_page_config(page_title="FairXplain | Student Success Intelligence", page_icon="🎓", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
:root {
  --fx-bg:#07111f; --fx-panel:#101e31; --fx-panel-2:#14263d;
  --fx-line:#263d59; --fx-teal:#4dd6c4; --fx-blue:#77a9f7;
  --fx-text:#eaf1fb; --fx-muted:#a9bad0;
}
html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
.stApp { background:linear-gradient(135deg,#081321 0%,#0b1728 55%,#0a1422 100%); color:var(--fx-text); }
.block-container { max-width:1420px; padding-top:2.0rem; padding-bottom:3rem; }
[data-testid="stHeader"] { display:none !important; height:0 !important; visibility:hidden !important; }
[data-testid="stToolbar"], [data-testid="stAppDeployButton"] {
  display:none !important; visibility:hidden !important; height:0 !important;
}
[data-testid="stMetricValue"] {
  color:#f3f7ff; font-size:clamp(1.25rem, 2.1vw, 2rem) !important;
  white-space:normal !important; overflow-wrap:anywhere;
}
[data-testid="stMetricLabel"] { white-space:normal !important; overflow-wrap:anywhere; }
#MainMenu, footer { visibility:hidden; }
h1,h2,h3,h4 { font-family:'Manrope',sans-serif !important; letter-spacing:-.025em; color:var(--fx-text); }
h2 { font-size:1.55rem !important; }
h3 { font-size:1.15rem !important; }
p, li, label, .stMarkdown { color:var(--fx-text); }
[data-testid="stSidebar"] { background:#0c192a; border-right:1px solid #243a55; }
[data-testid="stSidebar"] * { color:#dce7f5 !important; }
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color:#aebfd4 !important; }
[data-testid="stSidebar"] [role="radiogroup"] label { background:transparent; border-radius:10px; padding:5px 9px; }
[data-testid="stSidebar"] [role="radiogroup"] label:hover { background:#172b43; }
[data-testid="stSidebar"] [role="radiogroup"] label p { color:#dce7f5 !important; font-weight:500; }
[data-testid="stSidebar"] [data-baseweb="select"] > div { background:#17283e; border:1px solid #35506e; }
[data-testid="stMetric"] { background:linear-gradient(145deg,#14263d,#101d30); border:1px solid #2a425f; padding:14px 16px; border-radius:15px; }
[data-testid="stMetricValue"] { color:#f3f7ff; }
[data-testid="stMetricLabel"] { color:#b7c7dc; }
.stButton>button, .stDownloadButton>button, [data-testid="stFormSubmitButton"] button {
  border-radius:10px; border:1px solid #3d7d8b;
  background:linear-gradient(100deg,#167e7c,#285e9d); color:#fff; font-weight:700;
}
.stButton>button:hover, .stDownloadButton>button:hover { border-color:#65d7ca; color:#fff; }
.fx-hero { display:grid; grid-template-columns:minmax(0,1.6fr) minmax(240px,.8fr); gap:24px; align-items:center;
  padding:27px 30px; border-radius:22px; border:1px solid #2b496a;
  background:linear-gradient(115deg,rgba(23,65,88,.96),rgba(20,37,65,.97));
  box-shadow:0 16px 40px rgba(0,0,0,.20); }
.fx-kicker { color:#73e1d3; font-size:.72rem; font-weight:800; text-transform:uppercase; letter-spacing:.15em; margin-bottom:10px; }
.fx-hero-title { color:#f2f6ff; font-family:'Manrope',sans-serif; font-size:clamp(1.75rem,3.1vw,2.7rem); line-height:1.12; font-weight:800; letter-spacing:-.04em; margin:0 0 12px 0; }
.fx-sub { color:#c1d0e3; font-size:1rem; line-height:1.65; margin:0; max-width:760px; }
.fx-hero-side { background:rgba(7,18,33,.34); border:1px solid rgba(177,207,236,.18); border-radius:16px; padding:18px 19px; }
.fx-side-title { color:#cfe2f4; text-transform:uppercase; letter-spacing:.12em; font-weight:800; font-size:.7rem; margin-bottom:12px; }
.fx-step { display:flex; gap:11px; align-items:center; padding:8px 0; color:#eef5ff; font-weight:600; font-size:.92rem; border-bottom:1px solid rgba(198,220,243,.11); }
.fx-step:last-child { border-bottom:0; padding-bottom:0; }
.fx-step-no { display:inline-flex; align-items:center; justify-content:center; width:27px; height:27px; border-radius:8px; background:#244d69; color:#7ce0d3; font-size:.76rem; font-weight:800; }
.fx-kpi { min-height:112px; height:100%; background:linear-gradient(145deg,#14263d,#101d30); border:1px solid #2a425f; border-radius:15px; padding:17px 18px; }
.fx-kpi-label { color:#aebfd4; font-size:.78rem; font-weight:700; letter-spacing:.045em; text-transform:uppercase; margin-bottom:10px; }
.fx-kpi-value { color:#f3f7ff; font-family:'Manrope',sans-serif; font-size:2rem; line-height:1.05; font-weight:800; white-space:nowrap; }
.fx-kpi-help { color:#8fa7c1; font-size:.74rem; margin-top:7px; }
.fx-panel { background:rgba(16,30,49,.78); border:1px solid #2a415d; border-radius:17px; padding:18px 19px; height:100%; }
.fx-panel-title { color:#e9f1fb; font-family:'Manrope',sans-serif; font-size:1.08rem; font-weight:800; margin-bottom:4px; }
.fx-muted { color:#a9bad0; font-size:.85rem; line-height:1.5; }
.fx-result-number { color:#72e0d2; font-family:'Manrope',sans-serif; font-size:2.15rem; line-height:1.1; font-weight:800; }
.fx-pill { display:inline-block; padding:5px 9px; margin:7px 6px 0 0; border-radius:20px; background:rgba(75,145,170,.17); border:1px solid rgba(109,187,204,.28); color:#cde8ef; font-size:.74rem; font-weight:700; }
.fx-note { border-left:3px solid #4dd6c4; background:#12283a; padding:12px 15px; border-radius:8px; color:#c9d8ec; line-height:1.55; }
[data-testid="stDataFrame"] { border:1px solid #2b415e; border-radius:12px; overflow:hidden; }
small, .stCaption { color:#9fb2c9 !important; }

.fx-topbar { display:flex; align-items:center; justify-content:space-between; gap:16px; padding:8px 4px 18px; border-bottom:1px solid rgba(119,169,247,.20); margin-bottom:22px; }
.fx-brand { font-family:'Manrope',sans-serif; font-size:1.45rem; font-weight:800; color:#f3f6ff; letter-spacing:-.04em; }
.fx-brand span { color:#a78bfa; }
.fx-brand small { display:block; color:#91a9c4; font-family:'DM Sans',sans-serif; font-size:.68rem; letter-spacing:.12em; text-transform:uppercase; }
.fx-topnav { color:#cbd8eb; font-size:.86rem; font-weight:600; }
.fx-hero-main { display:grid; grid-template-columns:minmax(0,1.05fr) minmax(300px,.95fr); gap:22px; align-items:center; padding:30px; border:1px solid #28456b; border-radius:24px; background:radial-gradient(circle at 72% 45%,rgba(20,100,210,.23),transparent 38%),linear-gradient(120deg,#0b1830 0%,#102544 52%,#101b35 100%); box-shadow:0 22px 55px rgba(0,0,0,.28); overflow:hidden; }
.fx-hero-main .fx-hero-title { font-size:clamp(2.2rem,4.5vw,4rem); line-height:1.03; }
.fx-hero-main .fx-hero-title span { background:linear-gradient(90deg,#b79cff,#64d9ef); -webkit-background-clip:text; color:transparent; }
.fx-hero-art { width:100%; border-radius:18px; border:1px solid rgba(112,171,255,.3); box-shadow:0 12px 35px rgba(0,0,0,.3); }
.fx-actions { display:flex; gap:12px; flex-wrap:wrap; margin-top:22px; }
.fx-section-kicker { color:#6edbd4; font-size:.72rem; text-transform:uppercase; letter-spacing:.14em; font-weight:800; }
.fx-metric-ribbon { background:linear-gradient(110deg,#0e1d36,#132b4b); border:1px solid #2b4b76; border-radius:18px; padding:15px 18px; margin-top:20px; }
.fx-mini-label { color:#9fb4d1; font-size:.74rem; font-weight:700; text-transform:uppercase; letter-spacing:.06em; }
.fx-mini-value { color:#f3f7ff; font-family:'Manrope',sans-serif; font-size:1.7rem; font-weight:800; line-height:1.2; }
.fx-mini-help { color:#9eb2cd; font-size:.72rem; }
@media (max-width:900px) { .fx-hero-main { grid-template-columns:1fr; padding:20px; } .fx-topbar { flex-wrap:wrap; } }
@media (max-width:900px) { .fx-hero { grid-template-columns:1fr; padding:22px; } .fx-hero-title { font-size:1.8rem; } }

/* Reference-matched FairXplain home screen */
.fx-home-shell { max-width: 1500px; margin: 0 auto; }
.fx-home-nav { display:flex; align-items:center; justify-content:space-between; gap:12px;
  padding:4px 0 18px; margin-bottom:22px; border-bottom:1px solid rgba(119,169,247,.22); }
.fx-home-logo { font-family:'Manrope',sans-serif; font-size:1.7rem; line-height:1.05;
  font-weight:800; letter-spacing:-.045em; color:#f5f7ff; white-space:nowrap; }
.fx-home-logo .grad { background:linear-gradient(90deg,#b58cff,#5bd9ef);
  -webkit-background-clip:text; background-clip:text; color:transparent; }
.fx-home-logo small { display:block; color:#a9bad0; font:600 .67rem 'DM Sans',sans-serif;
  letter-spacing:.13em; text-transform:uppercase; margin-left:49px; margin-top:4px; }
.fx-home-nav [data-testid="stButton"] button { min-height:42px; padding:0 10px;
  background:transparent; border:1px solid transparent; color:#d9e4f7; font-weight:600;
  white-space:nowrap; box-shadow:none; }
.fx-home-nav [data-testid="stButton"] button:hover { border-color:#7756e8;
  background:rgba(112,83,222,.16); color:#fff; }
.fx-home-hero { position:relative; display:grid; grid-template-columns:1.02fr 1.18fr;
  gap:20px; align-items:center; padding:28px 10px 24px; min-height:430px; }
.fx-home-badge { display:inline-flex; align-items:center; gap:9px; padding:9px 17px;
  border-radius:30px; background:linear-gradient(100deg,rgba(26,43,76,.88),rgba(19,30,56,.72));
  border:1px solid rgba(126,145,255,.18); color:#dce6ff; font-weight:600; font-size:.9rem; }
.fx-home-title { margin:24px 0 14px; color:#f8f9ff; font:800 clamp(3.1rem,5.3vw,5.5rem)/.98 'Manrope',sans-serif;
  letter-spacing:-.065em; }
.fx-home-title span { background:linear-gradient(95deg,#c18bff 0%,#8c83ff 45%,#55d8ef 100%);
  -webkit-background-clip:text; background-clip:text; color:transparent; }
.fx-home-description { color:#e0e8f8; font-size:1.13rem; line-height:1.48; max-width:610px; }
.fx-home-description strong { color:#c6a4ff; }
.fx-home-features { display:flex; gap:10px; flex-wrap:wrap; margin:22px 0 24px; }
.fx-home-feature { display:flex; align-items:center; gap:10px; min-width:145px; padding:12px 14px;
  border-radius:17px; background:linear-gradient(145deg,rgba(18,35,62,.94),rgba(10,22,42,.92));
  border:1px solid rgba(94,126,190,.18); box-shadow:0 8px 25px rgba(0,0,0,.18); }
.fx-home-feature .ico { font-size:1.55rem; }
.fx-home-feature b { display:block; color:#f7f8ff; font-size:.98rem; }
.fx-home-feature small { display:block; color:#b8c6dc; font-size:.74rem; line-height:1.3; }
.fx-home-cta [data-testid="stButton"] button { min-height:52px; border-radius:14px;
  background:linear-gradient(105deg,#8a45f4,#4e65ef); border:1px solid #a88bff;
  color:white; font-weight:800; box-shadow:0 8px 25px rgba(102,74,240,.28); }
.fx-home-cta [data-testid="stButton"]:nth-child(2) button { background:rgba(9,19,38,.7);
  border:1.5px solid #a88bff; box-shadow:none; }
.fx-home-art img { width:100%; max-height:500px; object-fit:contain; border-radius:20px;
  filter:drop-shadow(0 18px 35px rgba(0,90,255,.20)); }
.fx-home-metrics { display:grid; grid-template-columns:repeat(4,1fr); gap:0;
  padding:18px 12px; margin:12px 0 30px; border:1px solid rgba(69,106,177,.45);
  border-radius:22px; background:linear-gradient(110deg,rgba(13,27,52,.96),rgba(18,39,72,.96));
  box-shadow:0 12px 35px rgba(0,0,0,.22); }
.fx-home-metric { display:flex; align-items:center; gap:16px; padding:6px 20px;
  border-right:1px solid rgba(111,139,202,.35); }
.fx-home-metric:last-child { border-right:0; }
.fx-home-metric .ico { font-size:2.2rem; color:#7eaaff; }
.fx-home-metric strong { display:block; color:#f6f7ff; font:800 1.9rem 'Manrope',sans-serif; }
.fx-home-metric span { color:#e0e8f8; font-size:.88rem; }
.fx-home-wave { height:42px; margin-top:12px; opacity:.8;
  background:radial-gradient(ellipse at 12% 100%,rgba(143,66,255,.28),transparent 34%),
             radial-gradient(ellipse at 90% 100%,rgba(0,188,255,.25),transparent 35%); }
@media(max-width:1050px) {
 .fx-home-hero { grid-template-columns:1fr; }
 .fx-home-art img { max-height:390px; }
 .fx-home-metric { padding:8px; gap:8px; }
 .fx-home-metric strong { font-size:1.35rem; }
}
@media(max-width:720px) {
 .fx-home-metrics { grid-template-columns:repeat(2,1fr); }
 .fx-home-metric:nth-child(2) { border-right:0; }
 .fx-home-metric { border-bottom:1px solid rgba(111,139,202,.25); }
 .fx-home-metric:nth-child(n+3) { border-bottom:0; }
}
</style>
""", unsafe_allow_html=True)


def load_artifacts(task: str):
    task_dir = MODELS_DIR / task
    model_path = task_dir / "pipeline.joblib"
    meta_path = task_dir / "model_metadata.json"
    enc_path = task_dir / "label_encoder.joblib"
    if not (model_path.exists() and meta_path.exists() and enc_path.exists()):
        return None
    return joblib.load(model_path), json.loads(meta_path.read_text(encoding="utf-8")), joblib.load(enc_path)


@st.cache_data(show_spinner=False)
def load_df():
    if DATA_FILE.exists():
        return load_dataset(DATA_FILE)
    return None


def kpi(label, value, delta=None):
    st.metric(label, value, delta)


def _read_metrics(task: str):
    path = REPORTS_DIR / task / "final_test_metrics.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _kpi_card(label: str, value: str, helper: str):
    st.markdown(
        f'<div class="fx-kpi"><div class="fx-kpi-label">{label}</div>'
        f'<div class="fx-kpi-value">{value}</div><div class="fx-kpi-help">{helper}</div></div>',
        unsafe_allow_html=True,
    )


def render_overview(df):
    # Hide the sidebar only on the landing page; all research workspace pages retain it.
    st.markdown("""
    <style>
      [data-testid="stSidebar"] {display:none;}
      .block-container {max-width:1540px;padding-top:1.0rem;padding-left:3.4rem;padding-right:3.4rem;}
      [data-testid="stHeader"] {height:0;background:transparent;}
      [data-testid="stToolbar"], [data-testid="stAppDeployButton"] {display:none;}
    </style>
    """, unsafe_allow_html=True)

    destinations = [
        ("⌂  Home", "Overview"), ("▥  Predict", "Student prediction"),
        ("◇  Explain", "Explainable AI"), ("⚖  Audit", "Fairness audit"),
        ("↗  Analyse", "Model analytics"), ("▤  Dataset", "Dataset explorer"),
        ("ⓘ  About", "Research notes"),
    ]
    nav = st.columns([2.05, 1, 1, 1, 1, 1, 1, 1], gap="small")
    with nav[0]:
        st.markdown(
            '<div class="fx-home-logo">🎓 Fair<span class="grad">Xplain</span>'
            '<small>Explain · Audit · Trust</small></div>',
            unsafe_allow_html=True,
        )
    for col, (label, destination) in zip(nav[1:], destinations):
        with col:
            st.button(
                label,
                key=f"home_nav_{destination}",
                width="stretch",
                on_click=_set_workspace_page,
                args=(destination,),
            )

    art = ROOT / "figures" / "fairxplain_hero_art.png"
    left, right = st.columns([1.02, 1.18], gap="large")
    with left:
        st.markdown(
            '<div class="fx-home-badge">🎓 &nbsp; AI for Fair and Transparent Education</div>'
            '<div class="fx-home-title">Fair<span>Xplain</span></div>'
            '<div class="fx-home-description">An Evidence-Grounded Framework for Auditing the '
            '<strong>Fairness and Faithfulness</strong> of AI Explanations in Digital Learning.</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="fx-home-features">'
            '<div class="fx-home-feature"><span class="ico">🎯</span><div><b>Predict</b><small>Student outcomes</small></div></div>'
            '<div class="fx-home-feature"><span class="ico">💡</span><div><b>Explain</b><small>With SHAP &amp; profile comparisons</small></div></div>'
            '<div class="fx-home-feature"><span class="ico">⚖️</span><div><b>Audit</b><small>Fairness &amp; model reliability</small></div></div>'
            '</div>',
            unsafe_allow_html=True,
        )
        cta1, cta2 = st.columns(2, gap="small")
        with cta1:
            st.button(
                "▶  Try a Prediction   →",
                key="hero_predict",
                width="stretch",
                on_click=_set_workspace_page,
                args=("Student prediction",),
            )
        with cta2:
            st.button(
                "▣  View Research   →",
                key="hero_research",
                width="stretch",
                on_click=_set_workspace_page,
                args=("Research notes",),
            )
    with right:
        if art.exists():
            st.markdown('<div class="fx-home-art">', unsafe_allow_html=True)
            st.image(str(art), width="stretch")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.warning("Hero artwork not found. Keep figures/fairxplain_hero_art.png beside the app.")

    if df is None:
        st.warning("Dataset not found. Place the official UCI data.csv file in data/raw/data.csv, then run the training command.")
        return

    records = len(df)

    left, right = st.columns([1.2, 1], gap="large")
    with left:
        st.markdown('<div class="fx-panel-title">Student outcome distribution</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="fx-muted">Distribution of {records:,} student records across the three outcome classes.</div>',
            unsafe_allow_html=True,
        )
        counts = df["Target"].value_counts().rename_axis("Outcome").reset_index(name="Students")
        order = [x for x in ["Dropout", "Enrolled", "Graduate"] if x in counts["Outcome"].tolist()]
        if order:
            counts["Outcome"] = pd.Categorical(counts["Outcome"], categories=order, ordered=True)
            counts = counts.sort_values("Outcome")
        fig = px.bar(
            counts, x="Outcome", y="Students", color="Outcome", text="Students",
            template="plotly_dark",
            color_discrete_sequence=["#a78bfa", "#55c7e8", "#4dd6c4"],
        )
        fig.update_traces(textposition="outside", cliponaxis=False, marker_line_width=0)
        fig.update_layout(
            height=320, showlegend=False, paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=10, r=10, t=18, b=10),
            font=dict(color="#dce7f5", family="DM Sans"),
            xaxis=dict(title=None, showgrid=False),
            yaxis=dict(title="Number of students", gridcolor="rgba(160,185,210,.13)"),
        )
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    with right:
        st.markdown('<div class="fx-panel-title">Prediction · Explanation · Audit</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="fx-muted">One research workflow for inspecting model outputs and their limitations.</div>',
            unsafe_allow_html=True,
        )
        for title, desc in [
            ("01  Predict", "Classify student outcomes with the selected machine-learning pipeline."),
            ("02  Explain", "Inspect global and individual feature contributions using SHAP."),
            ("03  Audit", "Review subgroup metrics and feature-perturbation sensitivity."),
        ]:
            st.markdown(
                f'<div class="fx-step"><span class="fx-step-no">{title[:2]}</span>'
                f'<div><b>{title[4:]}</b><br><span class="fx-muted">{desc}</span></div></div>',
                unsafe_allow_html=True,
            )
    st.markdown('<div class="fx-home-wave"></div>', unsafe_allow_html=True)


def render_prediction(task):
    artifacts = load_artifacts(task)
    if artifacts is None:
        st.info(f"No trained {task} model found. Run: python scripts/train.py --task {task}")
        return
    model, meta, label_encoder = artifacts
    schema = meta["input_schema"]
    task_title = {'multiclass':'Three-class outcome', 'binary':'Dropout vs all other outcomes', 'resolved_binary':'Dropout vs Graduate (Enrolled excluded)'}[task]
    st.markdown(f"### {task_title} prediction")
    st.caption("Enter a student profile based on the fields available in the dataset. Default values are training-data medians or modes, not a real student's values.")
    categories = list(schema)
    groups = {
        "Admission & background": [c for c in categories if not c.startswith("Curricular units") and c not in ["Unemployment rate", "Inflation rate", "GDP"]],
        "Semester 1 performance": [c for c in categories if "1st sem" in c],
        "Semester 2 performance": [c for c in categories if "2nd sem" in c],
        "Economic context": [c for c in categories if c in ["Unemployment rate", "Inflation rate", "GDP"]],
    }
    input_values = {}
    with st.form(f"predict_form_{task}"):
        for title, cols in groups.items():
            if not cols: continue
            with st.expander(title, expanded=(title == "Admission & background")):
                c1,c2 = st.columns(2)
                for i, col in enumerate(cols):
                    spec = schema[col]
                    container = c1 if i % 2 == 0 else c2
                    with container:
                        if spec["kind"] == "categorical":
                            options = spec.get("values", [])
                            default = spec.get("default", options[0] if options else 0)
                            try: ix = options.index(default)
                            except ValueError: ix = 0
                            input_values[col] = st.selectbox(col, options, index=ix, key=f"{task}_{col}") if options else default
                        else:
                            lo, hi, default = spec["min"], spec["max"], spec["median"]
                            if not np.isfinite(lo): lo = 0.0
                            if not np.isfinite(hi): hi = max(lo + 1, default + 1)
                            if hi <= lo: hi = lo + 1.0
                            integerish = abs(default - round(default)) < 1e-8 and abs(hi-round(hi)) < 1e-8
                            input_values[col] = st.number_input(col, min_value=float(lo), max_value=float(hi), value=float(min(max(default,lo),hi)), step=1.0 if integerish else 0.1, key=f"{task}_{col}")
        submitted = st.form_submit_button("Generate outcome prediction", width="stretch")
    if submitted:
        raw = pd.DataFrame([input_values])[meta["raw_feature_columns"]]
        engineered = engineer_features(raw)
        encoded = int(model.predict(engineered)[0])
        prediction = str(label_encoder.inverse_transform([encoded])[0])
        st.session_state[f"last_input_{task}"] = raw
        st.session_state[f"last_prediction_{task}"] = prediction
        st.session_state[f"last_engineered_{task}"] = engineered
        proba = model.predict_proba(engineered)[0] if hasattr(model, "predict_proba") else None
        st.success(f"Predicted outcome: {prediction}")
        if proba is not None:
            prob_df = pd.DataFrame({"Outcome": label_encoder.classes_, "Probability": proba})
            st.plotly_chart(px.bar(prob_df, x="Outcome", y="Probability", color="Outcome", text=prob_df["Probability"].map(lambda x:f"{x:.1%}"), template="plotly_dark", color_discrete_sequence=["#42d6c4","#7aa8ff","#f3b56b"]), width="stretch")
        st.subheader("Individual feature attribution")
        local = local_explanation(model, engineered, class_index=encoded)
        if local is not None and not local.empty:
            st.caption("Top local SHAP contributions for the predicted class. Encoded values are shown where one-hot encoding expands a category.")
            st.dataframe(local, width="stretch", hide_index=True)
            evidence = retrieve_feature_evidence(local["feature"].head(15).tolist())
            if evidence:
                st.markdown("**Retrieved field evidence (UCI feature dictionary)**")
                for item in evidence:
                    with st.expander(item["topic"]):
                        st.write(item["definition"])
                        st.caption(item["boundary"])
                        st.markdown(f"[UCI dataset documentation]({item['source']})")
        else:
            st.info("A local SHAP explanation is not available for this selected estimator. See the global explanation and permutation-importance artifacts on the Explainable AI page.")
        st.markdown('<div class="fx-note">Predicted probabilities are model scores, not guaranteed outcomes. Feature attributions are associations, not causes. This model is not a clinical or administrative decision system.</div>', unsafe_allow_html=True)
    last = st.session_state.get(f"last_input_{task}")
    pred = st.session_state.get(f"last_prediction_{task}")
    if last is not None and pred:
        st.subheader("Observed contrast profiles")
        st.caption("Nearest records from another observed outcome class. These are descriptive examples, not causal interventions or recommendations to change a protected attribute.")
        reference_path = ROOT / "data" / "processed" / task / "train_reference.csv"
        if reference_path.exists():
            reference = pd.read_csv(reference_path)
            classes = [c for c in label_encoder.classes_ if c != pred]
            desired = st.selectbox("Compare against outcome", classes, key=f"contrast_{task}")
            contrasts = nearest_observed_contrast(last, reference, desired)
            if not contrasts.empty:
                safe_cols = [c for c in contrasts.columns if c not in {"Gender", "Nacionality", "Age at enrollment", "Marital status", "International", "Scholarship holder", "Tuition fees up to date", "Displaced", "Educational special needs", "Debtor"}]
                st.dataframe(contrasts[safe_cols].head(3), width="stretch")
            else:
                st.info("No sufficiently similar reference profile was available.")


def render_model_analytics(task: str):
    """Display held-out and validation metrics for the shared workspace task."""
    report_dir = REPORTS_DIR / task
    metrics_path = report_dir / "final_test_metrics.json"
    comp_path = report_dir / "validation_model_comparison.csv"
    if not metrics_path.exists():
        st.info(f"No trained results for {task}. Run the training script first.")
        return

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    task_labels = {
        "multiclass": "Three-class outcome (Dropout / Enrolled / Graduate)",
        "binary": "Dropout vs. non-Dropout",
        "resolved_binary": "Dropout vs. Graduate (Enrolled excluded)",
    }
    st.caption(f"Selected evaluation task: {task_labels.get(task, task)}")

    if task == "resolved_binary":
        st.markdown(
            '<div class="fx-note"><strong>Task definition:</strong> This experiment '
            'distinguishes Dropout from Graduate and excludes Enrolled records. '
            'The metrics below use the held-out test subset for this task '
            '(726 records).</div>',
            unsafe_allow_html=True,
        )
    elif task == "binary":
        st.markdown(
            '<div class="fx-note"><strong>Task definition:</strong> Dropout is '
            'compared with the combined non-Dropout group. These results belong '
            'to this binary experiment only.</div>',
            unsafe_allow_html=True,
        )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi("Test accuracy", f"{metrics['accuracy']:.2%}")
    with c2:
        kpi("Macro F1", f"{metrics['macro_f1']:.2%}")
    with c3:
        kpi("Balanced accuracy", f"{metrics['balanced_accuracy']:.2%}")
    with c4:
        kpi("Selected model", metrics["selected_model"])

    st.caption(
        f"Held-out test size: {metrics.get('n_test', 'not recorded')}. "
        "The final test partition is held out from model selection."
    )

    percentage_columns = [
        "accuracy", "balanced_accuracy", "macro_precision", "macro_recall",
        "macro_f1", "weighted_f1",
    ]
    if comp_path.exists():
        comp = pd.read_csv(comp_path)
        st.subheader("Validation comparison (model selection only)")
        comp_format = {
            col: "{:.2%}" for col in percentage_columns if col in comp.columns
        }
        st.dataframe(
            comp.style.format(comp_format, na_rep="—"),
            width="stretch",
            hide_index=True,
        )
        if "accuracy" in comp.columns and "model" in comp.columns:
            fig = px.bar(
                comp.sort_values("accuracy"),
                x="accuracy",
                y="model",
                orientation="h",
                color="macro_f1" if "macro_f1" in comp.columns else "accuracy",
                template="plotly_dark",
                color_continuous_scale="Teal",
                labels={"accuracy": "Validation accuracy", "model": "Model"},
            )
            fig.update_xaxes(tickformat=".0%")
            st.plotly_chart(fig, width="stretch")

    cm_path = report_dir / "confusion_matrix.csv"
    if cm_path.exists():
        cm = pd.read_csv(cm_path, index_col=0)
        st.subheader("Untouched test confusion matrix")
        st.dataframe(cm, width="stretch")

    per_class = report_dir / "classification_report.csv"
    if per_class.exists():
        st.subheader("Per-class test metrics")
        per_class_df = pd.read_csv(per_class)
        class_format = {
            col: "{:.2%}" for col in ["precision", "recall", "f1-score"]
            if col in per_class_df.columns
        }
        st.dataframe(
            per_class_df.style.format(class_format, na_rep="—"),
            width="stretch",
            hide_index=True,
        )


def render_explainability(task):
    st.subheader("Model explanations")
    fig_path = FIGURES_DIR / task / "shap_global_importance.png"
    imp_path = REPORTS_DIR / task / "shap_global_importance.csv"
    if fig_path.exists():
        st.image(str(fig_path), width="stretch")
    elif imp_path.exists():
        st.dataframe(pd.read_csv(imp_path).head(20), width="stretch")
    else:
        st.info("SHAP artefacts are not generated yet. After training, run: python scripts/create_explanations.py")
    faithfulness_path = REPORTS_DIR / task / "faithfulness_perturbation_audit.csv"
    faithfulness_summary = REPORTS_DIR / task / "faithfulness_summary.json"
    if faithfulness_path.exists():
        st.subheader("Perturbation-based faithfulness check")
        st.caption("Validation-only sensitivity audit: features are permuted one at a time to measure changes in predictive accuracy. This measures model reliance; it is not a causal test or proof of explanation correctness.")
        st.dataframe(pd.read_csv(faithfulness_path).head(15), width="stretch")
    if faithfulness_summary.exists():
        st.json(json.loads(faithfulness_summary.read_text(encoding="utf-8")))
    st.markdown("**Evidence-grounded explanation policy**")
    st.write("Explanations combine model attribution values with definitions retrieved from the UCI feature dictionary. They describe predictive associations, not causes. No claim is made that changing a feature would change the real-world outcome.")
    st.write("Counterfactual profile comparisons use nearest observed examples from the training partition and deliberately omit selected protected/sensitive fields from the displayed contrast.")


def render_fairness(task):
    """Show overall metrics and subgroup results for the same selected task."""
    report_dir = REPORTS_DIR / task
    metrics = _read_metrics(task)
    audit_path = report_dir / "fairness_group_metrics.csv"
    summary_path = report_dir / "fairness_summary.json"

    task_labels = {
        "multiclass": "Three-class outcome (Dropout / Enrolled / Graduate)",
        "binary": "Dropout vs. non-Dropout",
        "resolved_binary": "Dropout vs. Graduate (Enrolled excluded)",
    }
    st.caption(f"Selected evaluation task: {task_labels.get(task, task)}")

    if metrics:
        st.subheader("Overall held-out test performance")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            kpi("Test accuracy", f"{metrics['accuracy']:.2%}")
        with c2:
            kpi("Macro F1", f"{metrics['macro_f1']:.2%}")
        with c3:
            kpi("Balanced accuracy", f"{metrics['balanced_accuracy']:.2%}")
        with c4:
            kpi("Selected model", metrics["selected_model"])

    st.subheader("Subgroup-level performance")
    st.caption(
        "Each row reports performance within one demographic or background group. "
        "These subgroup values are expected to differ from the overall test metrics above."
    )

    if not audit_path.exists():
        st.info("Fairness audit is generated automatically after model training.")
        return

    audit = pd.read_csv(audit_path)
    if audit.empty:
        st.info("No groups met the minimum support threshold (20).")
    else:
        percent_cols = [
            col for col in audit.columns
            if col in {
                "accuracy", "macro_f1", "predicted_dropout_rate",
                "recall_Dropout", "recall_Enrolled", "recall_Graduate",
                "recall_Not dropout",
            }
        ]
        st.dataframe(
            audit.style.format({col: "{:.2%}" for col in percent_cols}, na_rep="—"),
            width="stretch",
            hide_index=True,
        )
        attr = st.selectbox("Audit attribute", sorted(audit["attribute"].unique()))
        subset = audit[audit["attribute"] == attr]
        fig = px.bar(
            subset,
            x="group",
            y="accuracy",
            color="macro_f1",
            template="plotly_dark",
            title=f"Group-wise performance: {attr}",
            labels={"accuracy": "Subgroup accuracy", "group": "Group"},
        )
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig, width="stretch")

    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        st.subheader("Summary of subgroup gaps")
        st.caption(summary.get("status", "Descriptive subgroup summary"))
        rows = []
        for attribute, values in summary.items():
            if attribute == "status" or not isinstance(values, dict):
                continue
            rows.append({
                "Attribute": attribute,
                "Groups": values.get("groups"),
                "Accuracy gap": values.get("accuracy_gap_max_minus_min"),
                "Macro F1 gap": values.get("macro_f1_gap_max_minus_min"),
                "Predicted dropout-rate gap": values.get("predicted_dropout_rate_gap"),
            })
        if rows:
            gap_df = pd.DataFrame(rows)
            gap_cols = ["Accuracy gap", "Macro F1 gap", "Predicted dropout-rate gap"]
            st.dataframe(
                gap_df.style.format({col: "{:.2%}" for col in gap_cols}, na_rep="—"),
                width="stretch",
                hide_index=True,
            )

    st.markdown(
        '<div class="fx-note">Subgroup comparisons are descriptive. Differences '
        'are not causal findings or a legal compliance assessment; group sizes '
        'and class balance should be considered when interpreting them.</div>',
        unsafe_allow_html=True,
    )


def render_dataset(df):
    if df is None:
        st.info("Place data.csv in data/raw/ to enable the dataset explorer.")
        return
    st.subheader("Dataset explorer")
    st.write(f"{len(df):,} records • {df.shape[1]-1} predictors • {df['Target'].nunique()} outcomes")
    st.dataframe(df.head(100), width="stretch")
    counts = df["Target"].value_counts().rename_axis("Target").reset_index(name="Count")
    st.plotly_chart(px.pie(counts, names="Target", values="Count", hole=.48, template="plotly_dark"), width="stretch")
    num = df.select_dtypes(include=np.number)
    if not num.empty:
        col = st.selectbox("Numeric feature", list(num.columns))
        st.plotly_chart(px.histogram(df, x=col, color="Target", marginal="box", template="plotly_dark"), width="stretch")


def main():
    st.sidebar.markdown("## 🎓 FairXplain")
    st.sidebar.markdown("<div style='color:#91a9c4;font-size:.82rem;margin-top:-8px;margin-bottom:18px;'>Student outcome intelligence</div>", unsafe_allow_html=True)
    task = st.sidebar.selectbox(
        "Shared evaluation task",
        ["multiclass", "binary", "resolved_binary"],
        index=2,
        format_func=lambda x: {
            "multiclass": "Three-class outcome",
            "binary": "Dropout vs all other outcomes",
            "resolved_binary": "Dropout vs Graduate (Enrolled excluded)",
        }[x],
        help="The selected task is used consistently across prediction, analytics, explanations and fairness audit.",
    )
    st.sidebar.caption("One shared task selection is used across all analysis pages.")
    if "workspace_page" not in st.session_state:
        st.session_state["workspace_page"] = "Overview"
    page = st.sidebar.radio("Workspace", ["Overview", "Student prediction", "Model analytics", "Explainable AI", "Fairness audit", "Dataset explorer", "Research notes"], key="workspace_page")
    st.sidebar.markdown("---")
    st.sidebar.caption("UCI Student Dropout & Academic Success • FairXplain research prototype")
    df = load_df()
    if page == "Overview":
        render_overview(df)
    elif page == "Student prediction":
        render_prediction(task)
    elif page == "Model analytics":
        render_model_analytics(task)
    elif page == "Explainable AI":
        render_explainability(task)
    elif page == "Fairness audit":
        render_fairness(task)
    elif page == "Dataset explorer":
        render_dataset(df)
    else:
        st.subheader("Research protocol & limitations")
        report = ROOT / "docs" / "research_protocol.md"
        st.markdown(report.read_text(encoding="utf-8") if report.exists() else "Research protocol not found.")


if __name__ == "__main__":
    main()
