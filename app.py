"""Streamlit interface for ᕓ𐌉𐌕𐌀 𐌀𐌉."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import streamlit as st

from src.animations import render_animated_hero
from src.config import (
    APP_SUBTITLE,
    APP_TITLE,
    APP_VERSION,
    CLASSIFICATION_REPORT_PATH,
    CONFUSION_MATRIX_PATH,
    HISTORY_PATH,
    METRICS_PATH,
    MODEL_PATH,
    ensure_runtime_directories,
    load_class_names,
    load_model_metadata,
)
from src.database import PredictionDatabase
from src.disease_info import library_conditions
from src.model import ModelUnavailableError, load_trained_model
from src.prediction import predict_image
from src.preprocessing import ImageValidationError, load_image
from src.quality_checker import assess_image_quality
from src.report import build_text_report

APP_ROOT = Path(__file__).resolve().parent
HERO_ARTWORK_PATH = APP_ROOT / "assets" / "vita-ai-diagnostic-hero.jpg"
LOGGER = logging.getLogger(__name__)

st.set_page_config(
    page_title=f"{APP_TITLE} · Plant Health Screening",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="auto",
)


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --forest-950: #082b24;
            --forest-900: #0d352d;
            --forest-800: #16493c;
            --forest-700: #24614f;
            --sage-100: #e9f0e7;
            --sage-50: #f3f7f1;
            --lime-400: #cfe16c;
            --lime-300: #dce987;
            --paper: #fafbf7;
            --surface: #ffffff;
            --ink: #14231d;
            --muted: #617069;
            --line: #dce5dc;
            --warning: #8a6418;
            --radius-lg: 24px;
            --radius-md: 16px;
            --shadow-sm: 0 8px 24px rgba(12, 52, 43, .055);
            --shadow-md: 0 18px 50px rgba(12, 52, 43, .09);
        }
        html { scroll-behavior: smooth; }
        body, .stApp {
            font-family: Inter, "Segoe UI", system-ui, -apple-system, sans-serif;
        }
        .stApp {
            background:
                radial-gradient(circle at 88% -4%, rgba(207,225,108,.16), transparent 31rem),
                radial-gradient(circle at 5% 42%, rgba(110,163,139,.09), transparent 26rem),
                linear-gradient(180deg, #fbfcf8 0%, #f4f7f2 100%);
            color: var(--ink);
        }
        [data-testid="stAppViewContainer"] > .main .block-container {
            max-width: 1240px;
            padding-top: 1.25rem;
            padding-bottom: 4rem;
        }
        [data-testid="stHeader"] { background: transparent; }
        [data-testid="stSidebar"] {
            background:
                radial-gradient(circle at 18% 5%, rgba(207,225,108,.13), transparent 15rem),
                linear-gradient(180deg, var(--forest-950), var(--forest-900));
            border-right: 1px solid rgba(255,255,255,.08);
        }
        [data-testid="stSidebar"] > div:first-child { padding-top: 1rem; }
        [data-testid="stSidebar"] * { color: #f1f5ef; }
        [data-testid="stSidebar"] div[role="radiogroup"] label {
            border: 1px solid transparent;
            border-radius: 13px;
            padding: .48rem .65rem;
            margin-bottom: .18rem;
            transition: background-color .16s ease, border-color .16s ease;
        }
        [data-testid="stSidebar"] div[role="radiogroup"] label:hover {
            background: rgba(255,255,255,.06);
            border-color: rgba(255,255,255,.08);
        }
        [data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
            background: rgba(207,225,108,.14);
            border-color: rgba(207,225,108,.28);
            box-shadow: inset 3px 0 0 var(--lime-400);
        }
        .vita-brand {
            display: flex;
            align-items: center;
            gap: .78rem;
            padding: .4rem .1rem 1rem;
        }
        .vita-brand-mark {
            display: grid;
            place-items: center;
            width: 46px;
            height: 46px;
            border: 1px solid rgba(220,233,135,.55);
            border-radius: 15px;
            color: var(--lime-300);
            background: rgba(255,255,255,.055);
            font-size: 1.22rem;
            font-weight: 850;
            box-shadow: inset 0 0 18px rgba(207,225,108,.08);
        }
        .vita-brand-name { color: #fff; font-size: 1.04rem; font-weight: 800; letter-spacing: .02em; }
        .vita-brand-sub { color: #b9c9c0; font-size: .72rem; line-height: 1.35; }
        .vita-sidebar-label {
            margin: .25rem 0 .45rem;
            color: #91a69c;
            font-size: .68rem;
            font-weight: 800;
            letter-spacing: .14em;
            text-transform: uppercase;
        }
        .vita-kicker {
            color: var(--forest-700);
            font-size: .76rem;
            font-weight: 800;
            letter-spacing: .16em;
            text-transform: uppercase;
            margin-bottom: .4rem;
        }
        .vita-card {
            position: relative;
            overflow: hidden;
            background: rgba(255,255,255,.9);
            border: 1px solid var(--line);
            border-radius: var(--radius-lg);
            padding: 1.35rem 1.4rem;
            height: 100%;
            box-shadow: var(--shadow-sm);
        }
        .vita-card::after {
            content: "";
            position: absolute;
            width: 94px;
            height: 94px;
            right: -44px;
            bottom: -48px;
            border-radius: 50%;
            background: rgba(207,225,108,.13);
        }
        .vita-card h3 { margin: 0 0 .55rem; color: var(--forest-900); font-size: 1rem; }
        .vita-card p { margin: 0; color: var(--muted); line-height: 1.58; }
        .vita-step {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 32px;
            height: 32px;
            margin-bottom: .9rem;
            border-radius: 10px;
            background: var(--sage-100);
            color: var(--forest-800);
            font-size: .78rem;
            font-weight: 850;
        }
        .vita-trust-strip {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 1px;
            overflow: hidden;
            margin: .25rem 1.5rem 1.35rem;
            border: 1px solid var(--line);
            border-radius: 17px;
            background: var(--line);
            box-shadow: var(--shadow-sm);
        }
        .vita-trust-item { background: rgba(255,255,255,.94); padding: .85rem 1rem; }
        .vita-trust-value { display: block; color: var(--forest-900); font-size: 1rem; font-weight: 850; }
        .vita-trust-label { color: var(--muted); font-size: .72rem; }
        .vita-section-head { margin: 2rem 0 .85rem; }
        .vita-section-head small {
            color: var(--forest-700);
            font-size: .68rem;
            font-weight: 850;
            letter-spacing: .14em;
            text-transform: uppercase;
        }
        .vita-section-head h2 {
            margin: .22rem 0 .35rem;
            color: var(--forest-950);
            font-size: clamp(1.45rem, 3vw, 2.05rem);
            letter-spacing: -.025em;
        }
        .vita-section-head p { max-width: 48rem; margin: 0; color: var(--muted); }
        .vita-workflow {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: .8rem;
            margin: .25rem 0 1.25rem;
            padding: .85rem 1rem;
            border: 1px solid var(--line);
            border-radius: var(--radius-md);
            background: rgba(255,255,255,.72);
            color: var(--muted);
            font-size: .8rem;
        }
        .vita-workflow strong { color: var(--forest-800); }
        .vita-workflow-arrow { color: #9aaba2; }
        .vita-empty {
            padding: 2.2rem;
            border: 1px dashed #b9c9bd;
            border-radius: var(--radius-lg);
            background: rgba(255,255,255,.62);
            text-align: center;
            color: var(--muted);
        }
        .vita-result {
            background:
                radial-gradient(circle at 92% 20%, rgba(207,225,108,.24), transparent 10rem),
                linear-gradient(135deg, #e8f1e8, #f9faf4);
            border: 1px solid #c7d8ca;
            border-left: 5px solid var(--forest-700);
            border-radius: var(--radius-lg);
            padding: 1.4rem 1.55rem;
            box-shadow: var(--shadow-sm);
        }
        .vita-result .condition { color: var(--forest-900); font-size: 1.75rem; font-weight: 850; letter-spacing: -.025em; }
        .vita-result .crop { color: var(--muted); font-size: .9rem; text-transform: uppercase; letter-spacing: .12em; }
        .vita-disclaimer {
            background: #fff9e9;
            border: 1px solid #ead9aa;
            border-left: 4px solid #c39430;
            color: #5d4918;
            border-radius: var(--radius-md);
            padding: 1rem 1.05rem;
            font-size: .9rem;
            line-height: 1.55;
        }
        .model-ready { color: var(--lime-300); font-weight: 750; }
        .model-missing { color: #ffd591; font-weight: 700; }
        .vita-badge {
            display: inline-block;
            border-radius: 999px;
            padding: .22rem .58rem;
            margin: 0 .35rem .65rem 0;
            font-size: .72rem;
            font-weight: 800;
            letter-spacing: .04em;
            text-transform: uppercase;
        }
        .vita-badge-ai { background: var(--sage-100); color: var(--forest-900); }
        .vita-badge-reference { background: #eef0e8; color: #59635d; }
        .vita-badge-category { background: #f4f0d8; color: #655d27; }
        [data-testid="stMetric"] {
            background: rgba(255,255,255,.88);
            border: 1px solid var(--line);
            border-radius: var(--radius-md);
            padding: .95rem 1.05rem;
            box-shadow: 0 6px 20px rgba(12,52,43,.035);
        }
        [data-testid="stMetricValue"] { color: var(--forest-900); font-weight: 820; }
        .stButton > button, .stDownloadButton > button {
            border-radius: 999px;
            font-weight: 750;
            min-height: 2.8rem;
            border-color: var(--forest-700);
            transition: transform .14s ease, box-shadow .14s ease, background-color .14s ease;
        }
        .stButton > button:hover, .stDownloadButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 8px 22px rgba(13,53,45,.12);
        }
        .stButton > button[kind="primary"] {
            background: var(--forest-900);
            color: white;
        }
        .stButton > button:focus-visible, .stDownloadButton > button:focus-visible,
        input:focus-visible, textarea:focus-visible {
            outline: 3px solid rgba(207,225,108,.55) !important;
            outline-offset: 2px;
        }
        [data-testid="stFileUploaderDropzone"] {
            min-height: 170px;
            border: 1.5px dashed #77998a;
            border-radius: var(--radius-lg);
            background: rgba(255,255,255,.75);
        }
        [data-testid="stExpander"] {
            overflow: hidden;
            border-color: var(--line);
            border-radius: var(--radius-md);
            background: rgba(255,255,255,.7);
        }
        [data-baseweb="tab-list"] { gap: .35rem; }
        [data-baseweb="tab"] {
            border-radius: 999px;
            padding-left: 1rem;
            padding-right: 1rem;
        }
        [data-baseweb="tab"][aria-selected="true"] { background: var(--sage-100); }
        [data-testid="stDataFrame"] { border-radius: var(--radius-md); overflow: hidden; }
        .vita-principle {
            min-height: 148px;
            padding: 1.2rem;
            border: 1px solid var(--line);
            border-radius: var(--radius-md);
            background: rgba(255,255,255,.8);
        }
        .vita-principle strong { display: block; margin-bottom: .4rem; color: var(--forest-900); }
        .vita-principle span { color: var(--muted); font-size: .9rem; line-height: 1.55; }
        footer { visibility: hidden; }
        #MainMenu { visibility: hidden; }
        [data-testid="stToolbar"] { display: none !important; }
        [data-testid="stDecoration"] { display: none !important; }
        button[data-testid="stBaseButton-headerNoPadding"] { color: var(--forest-900); }
        @media (prefers-reduced-motion: reduce) {
            html { scroll-behavior: auto; }
            *, *::before, *::after { transition: none !important; }
        }
        button[aria-label*="fullscreen" i],
        button[title*="fullscreen" i] {
            display: none !important;
        }
        @media (max-width: 1100px) {
            .vita-trust-strip { grid-template-columns: repeat(2, 1fr); margin-inline: .5rem; }
            .vita-workflow { align-items: flex-start; flex-direction: column; }
            .vita-workflow-arrow { display: none; }
        }
        @media (max-width: 700px) {
            [data-testid="stAppViewContainer"] > .main .block-container { padding-inline: 1rem; }
            .vita-trust-strip { grid-template-columns: 1fr 1fr; }
            .vita-trust-item { padding: .75rem; }
            .vita-card { padding: 1.15rem; border-radius: 18px; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def database() -> PredictionDatabase:
    return PredictionDatabase()


@st.cache_resource(show_spinner="Loading the CNN model…")
def model_bundle():
    if not MODEL_PATH.exists():
        from download_model import install_model

        install_model()
    labels = load_class_names()
    metadata = load_model_metadata()
    model = load_trained_model()
    output_classes = int(model.output_shape[-1])
    if output_classes != len(labels):
        raise ModelUnavailableError(
            f"Model/label mismatch: model has {output_classes} outputs but labels contain {len(labels)} classes."
        )
    return model, labels, metadata


def model_status() -> tuple[bool, str]:
    try:
        model_bundle()
        return True, "CNN ready"
    except ModelUnavailableError as exc:
        return False, f"Model needs attention: {exc}"
    except Exception:
        LOGGER.exception("Unexpected model initialization failure")
        return False, "Model needs attention. Check the server logs for details."


def hero(kicker: str, heading: str, body: str, *, artwork: bool = False) -> None:
    render_animated_hero(
        kicker,
        heading,
        body,
        artwork_path=HERO_ARTWORK_PATH if artwork else None,
    )


def section_heading(eyebrow: str, heading: str, body: str) -> None:
    st.markdown(
        f"""
        <div class="vita-section-head">
            <small>{eyebrow}</small>
            <h2>{heading}</h2>
            <p>{body}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def navigate_to(page: str) -> None:
    st.session_state["page"] = page


def sidebar() -> str:
    st.sidebar.markdown(
        f"""
        <div class="vita-brand">
            <div class="vita-brand-mark">V</div>
            <div>
                <div class="vita-brand-name">{APP_TITLE}</div>
                <div class="vita-brand-sub">Explainable plant-health AI</div>
            </div>
        </div>
        <div class="vita-sidebar-label">Workspace</div>
        """,
        unsafe_allow_html=True,
    )
    page = st.sidebar.radio(
        "Navigate",
        (
            "Home",
            "Screen a leaf",
            "History & analytics",
            "Model performance",
            "Disease library",
            "About & limitations",
        ),
        key="page",
        label_visibility="collapsed",
    )
    st.sidebar.divider()
    ready, message = model_status()
    css_class = "model-ready" if ready else "model-missing"
    st.sidebar.markdown(f'<div class="{css_class}">● {message}</div>', unsafe_allow_html=True)
    st.sidebar.caption(f"App {APP_VERSION} · Images are not stored")
    return page


def render_home() -> None:
    hero(
        "Explainable plant intelligence",
        APP_TITLE,
        f"{APP_SUBTITLE}. Screen a leaf, inspect the model's evidence, and act with appropriate caution.",
        artwork=True,
    )
    st.markdown(
        """
        <div class="vita-trust-strip" aria-label="System capabilities">
            <div class="vita-trust-item"><span class="vita-trust-value">38</span><span class="vita-trust-label">AI-supported classes</span></div>
            <div class="vita-trust-item"><span class="vita-trust-value">70</span><span class="vita-trust-label">Knowledge references</span></div>
            <div class="vita-trust-item"><span class="vita-trust-value">Grad-CAM</span><span class="vita-trust-label">Visual explanation</span></div>
            <div class="vita-trust-item"><span class="vita-trust-value">Private</span><span class="vita-trust-label">Images are not stored</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    action, learn = st.columns(2)
    with action:
        st.button(
            "Start a leaf screening",
            type="primary",
            use_container_width=True,
            on_click=navigate_to,
            args=("Screen a leaf",),
        )
    with learn:
        st.button(
            "Explore disease library",
            use_container_width=True,
            on_click=navigate_to,
            args=("Disease library",),
        )

    section_heading(
        "How it works",
        "A transparent three-step workflow",
        "Every result begins with image-quality checks and ends with confidence-aware guidance.",
    )
    left, middle, right = st.columns(3)
    with left:
        st.markdown(
            '<div class="vita-card"><span class="vita-step">01</span><h3>Capture clearly</h3><p>Upload or photograph one leaf. '
            "Vita AI checks focus, lighting, contrast, dimensions, and visible detail before inference.</p></div>",
            unsafe_allow_html=True,
        )
    with middle:
        st.markdown(
            '<div class="vita-card"><span class="vita-step">02</span><h3>Inspect the evidence</h3><p>Review the crop, condition, '
            "confidence level, alternatives, and the regions that most influenced the CNN.</p></div>",
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(
            '<div class="vita-card"><span class="vita-step">03</span><h3>Act responsibly</h3><p>Read conservative prevention guidance '
            "and understand when professional confirmation is needed—without false certainty.</p></div>",
            unsafe_allow_html=True,
        )

    section_heading(
        "Capture guide",
        "Give the model a useful image",
        "A strong photograph reduces avoidable uncertainty before the screening begins.",
    )
    cols = st.columns(4)
    tips = (
        ("01 · Natural light", "Use bright, even light without glare."),
        ("02 · One main leaf", "Keep the symptomatic leaf large in frame."),
        ("03 · Sharp focus", "Focus on spots, lesions, or discoloration."),
        ("04 · Calm background", "Reduce clutter behind the leaf."),
    )
    for column, (title, text) in zip(cols, tips, strict=True):
        with column:
            st.markdown(f"**{title}**  \n{text}")
    st.write("")
    st.markdown(
        '<div class="vita-disclaimer">Vita AI predicts patterns learned from the PlantVillage classes. '
        "Field images, uncommon crops, mixed diseases, pests, and nutrient deficiencies may produce unreliable results. "
        "Always verify important decisions with a qualified agricultural professional.</div>",
        unsafe_allow_html=True,
    )


def render_quality(quality) -> None:
    st.markdown("#### Image-quality check")
    cols = st.columns(4)
    cols[0].metric("Overall score", f"{quality.score:.0f}/100")
    cols[1].metric("Brightness", f"{quality.brightness:.0f}/255")
    cols[2].metric("Contrast", f"{quality.contrast:.1f}")
    cols[3].metric("Sharpness", f"{quality.sharpness:.0f}")
    for warning in quality.warnings:
        st.warning(warning)
    if not quality.warnings:
        st.success("The image passed the basic quality checks.")


def persist_prediction(result, quality, metadata) -> int:
    alternatives = [f"{item.crop} — {item.condition}" for item in result.alternatives]
    return database().save_prediction(
        crop_name=result.primary.crop,
        disease_name=result.primary.condition,
        health_status=result.primary.status,
        confidence=result.primary.confidence,
        alternatives=alternatives,
        image_quality_score=quality.score,
        inference_time_ms=result.inference_time_ms,
        model_version=metadata.model_version,
    )


def render_prediction_result(
    result, quality, prediction_id: int, model, labels, metadata, image
) -> None:
    primary = result.primary
    status_icon = "✓" if primary.status == "Healthy" else "!"
    st.markdown(
        f"""
        <div class="vita-result">
            <div class="crop">{primary.crop} · {primary.status}</div>
            <div class="condition">{status_icon} {primary.condition}</div>
            <div><strong>{primary.confidence:.1%} confidence</strong> · {result.confidence_level} confidence</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.info(result.guidance)
    metrics = st.columns(3)
    metrics[0].metric("Confidence", f"{primary.confidence:.1%}")
    metrics[1].metric("Image quality", f"{quality.score:.0f}/100")
    metrics[2].metric("Inference", f"{result.inference_time_ms:.0f} ms")

    tab_details, tab_explain, tab_alternatives = st.tabs(
        ("What this may mean", "Why the model looked here", "Top predictions")
    )
    with tab_details:
        info = result.disease_info
        st.markdown(f"**Overview**  \n{info.description}")
        st.markdown(f"**Common visible symptoms**  \n{info.symptoms}")
        if primary.confidence >= 0.55:
            st.markdown(f"**General prevention**  \n{info.prevention}")
        else:
            st.warning(
                "Prevention guidance is hidden because confidence is low. Try another image first."
            )
        st.warning(info.expert_warning)
    with tab_explain:
        if primary.confidence < 0.55:
            st.info(
                "Explainability is withheld for this low-confidence result; capture another image."
            )
        else:
            try:
                from src.gradcam import explain_prediction

                class_index = labels.index(primary.label)
                overlay = explain_prediction(model, image, class_index, metadata)
                c1, c2 = st.columns(2)
                c1.image(image, caption="Original image", use_container_width=True)
                c2.image(
                    overlay,
                    caption="Grad-CAM: warmer areas influenced this prediction more",
                    use_container_width=True,
                )
                st.caption(
                    "Grad-CAM is an explanation of model attention, not a segmentation of infected tissue."
                )
            except Exception:
                LOGGER.exception("Grad-CAM generation failed")
                st.info("Grad-CAM is unavailable for this model build.")
    with tab_alternatives:
        rows = [
            {
                "Rank": rank,
                "Crop": item.crop,
                "Condition": item.condition,
                "Confidence": f"{item.confidence:.1%}",
            }
            for rank, item in enumerate((primary, *result.alternatives), start=1)
        ]
        st.dataframe(rows, hide_index=True, use_container_width=True)

    report = build_text_report(result, quality, metadata.model_version)
    st.download_button(
        "Download screening report",
        data=report.encode("utf-8"),
        file_name=f"vita-ai-screening-{prediction_id}.txt",
        mime="text/plain",
        use_container_width=True,
    )
    st.markdown("#### Was this screening useful?")
    feedback_cols = st.columns(3)
    choices = (
        ("Correct", "Appears correct"),
        ("Incorrect", "Appears incorrect"),
        ("Unsure", "I’m unsure"),
    )
    for column, (value, label) in zip(feedback_cols, choices, strict=True):
        if column.button(label, key=f"feedback-{prediction_id}-{value}", use_container_width=True):
            database().update_feedback(prediction_id, value)
            st.success("Anonymous feedback saved. Thank you.")


def render_screening() -> None:
    hero(
        "Guided screening",
        "Screen a leaf",
        "Capture one clear leaf. Vita AI validates the image before running the CNN and presenting explainable evidence.",
    )
    st.markdown(
        """
        <div class="vita-workflow" aria-label="Screening workflow">
            <span><strong>01</strong> Select a source</span><span class="vita-workflow-arrow">→</span>
            <span><strong>02</strong> Pass quality checks</span><span class="vita-workflow-arrow">→</span>
            <span><strong>03</strong> Run CNN screening</span><span class="vita-workflow-arrow">→</span>
            <span><strong>04</strong> Review evidence</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    section_heading(
        "Image input",
        "Choose a clear leaf photograph",
        "For best results, keep one leaf prominent and make the affected area easy to see.",
    )
    source_mode = st.radio("Image source", ("Upload an image", "Use camera"), horizontal=True)
    source = (
        st.file_uploader("Choose a JPG, JPEG, or PNG image", type=("jpg", "jpeg", "png"))
        if source_mode == "Upload an image"
        else st.camera_input("Take a close, well-lit leaf photo")
    )
    if source is None:
        st.markdown(
            '<div class="vita-empty"><strong>Ready when you are</strong><br>Your image is processed in memory and is not stored in prediction history.</div>',
            unsafe_allow_html=True,
        )
        return
    try:
        image = load_image(source)
    except ImageValidationError as exc:
        st.error(str(exc))
        return
    preview, assessment = st.columns((1, 1.1), gap="large")
    with preview:
        st.image(
            image,
            caption=f"Selected image · {image.width} × {image.height}",
            use_container_width=True,
        )
    quality = assess_image_quality(image)
    with assessment:
        render_quality(quality)
    if not quality.acceptable:
        st.error("This image is not suitable for screening. Please capture a clearer replacement.")
        return
    analyze = st.button("Analyze leaf", type="primary", use_container_width=True)
    if not analyze:
        return
    try:
        model, labels, metadata = model_bundle()
        with st.spinner("Analyzing visible patterns…"):
            result = predict_image(model, image, labels, metadata)
        prediction_id = persist_prediction(result, quality, metadata)
        st.session_state["last_prediction_id"] = prediction_id
        render_prediction_result(result, quality, prediction_id, model, labels, metadata, image)
    except ModelUnavailableError as exc:
        st.error(str(exc))
        st.code("python download_model.py\nstreamlit run app.py", language="bash")
    except Exception:
        LOGGER.exception("Unexpected screening failure")
        st.error("Analysis stopped safely. Please try another image or contact the app maintainer.")


def render_history() -> None:
    hero(
        "Private by design",
        "History & analytics",
        "Only anonymous prediction metadata is stored. Uploaded leaf images are never written to the database.",
    )
    data = database().analytics()
    summary = data["summary"]
    cols = st.columns(5)
    cols[0].metric("Screenings", int(summary["total"] or 0))
    cols[1].metric("Average confidence", f"{float(summary['average_confidence'] or 0):.1%}")
    cols[2].metric("Healthy", int(summary["healthy"] or 0))
    cols[3].metric("Diseased", int(summary["diseased"] or 0))
    cols[4].metric("Low confidence", int(summary["low_confidence"] or 0))
    if int(summary["total"] or 0) == 0:
        st.markdown(
            '<div class="vita-empty"><strong>No screening history yet</strong><br>Complete a leaf screening to build private, anonymous analytics on this device.</div>',
            unsafe_allow_html=True,
        )
        st.button(
            "Start the first screening",
            type="primary",
            on_click=navigate_to,
            args=("Screen a leaf",),
        )
        return
    left, right = st.columns(2)
    with left:
        st.subheader("Screenings by crop")
        st.bar_chart(data["by_crop"], x="crop_name", y="count", color="#3b735d")
    with right:
        st.subheader("Most frequent conditions")
        st.bar_chart(data["top_conditions"], x="disease_name", y="count", color="#9aaa45")
    st.subheader("Recent screenings")
    rows = database().recent_predictions(200)
    for row in rows:
        row["confidence"] = f"{float(row['confidence']):.1%}"
        row["image_quality_score"] = f"{float(row['image_quality_score']):.0f}/100"
    visible = [
        "created_at",
        "crop_name",
        "disease_name",
        "health_status",
        "confidence",
        "image_quality_score",
        "feedback",
    ]
    st.dataframe(
        [{key: row[key] for key in visible} for row in rows],
        hide_index=True,
        use_container_width=True,
    )
    with st.expander("Privacy controls"):
        st.warning("Clearing history permanently deletes all locally stored prediction metadata.")
        confirm = st.checkbox("I understand that this cannot be undone.")
        if st.button("Clear local prediction history", disabled=not confirm):
            count = database().clear_history()
            st.success(f"Deleted {count} local prediction record(s).")
            st.rerun()


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def render_performance() -> None:
    hero(
        "Evidence, not decoration",
        "Model performance",
        "This page displays metrics produced by evaluate.py. It never substitutes invented scores when evaluation artifacts are absent.",
    )
    metrics = _load_json(METRICS_PATH)
    history = _load_json(HISTORY_PATH)
    if metrics is None:
        st.info(
            "No verified evaluation artifacts are installed yet. Train or install a model, then run "
            "`python evaluate.py --data <test-folder>`."
        )
        st.code(
            "python train.py --data data/plantvillage --epochs 12\n"
            "python evaluate.py --data data/plantvillage/test",
            language="bash",
        )
        return
    st.markdown(
        '<div class="vita-disclaimer"><strong>Evaluation context:</strong> These results measure the controlled PlantVillage test split. They do not guarantee the same accuracy in farms, gardens, or mixed field conditions.</div>',
        unsafe_allow_html=True,
    )
    st.write("")
    cols = st.columns(5)
    displayed = (
        ("Test accuracy", "accuracy"),
        ("Top-3 accuracy", "top_3_accuracy"),
        ("Macro precision", "macro_precision"),
        ("Macro recall", "macro_recall"),
        ("Macro F1", "macro_f1"),
    )
    for column, (label, key) in zip(cols, displayed, strict=True):
        value = metrics.get(key)
        column.metric(label, "—" if value is None else f"{float(value):.2%}")
    if history:
        st.subheader("Training history")
        keys = [key for key in ("accuracy", "val_accuracy", "loss", "val_loss") if key in history]
        if keys:
            st.line_chart({key: history[key] for key in keys})
    if CONFUSION_MATRIX_PATH.exists():
        st.subheader("Confusion matrix")
        st.image(str(CONFUSION_MATRIX_PATH), use_container_width=True)
    if CLASSIFICATION_REPORT_PATH.exists():
        import pandas as pd

        st.subheader("Per-class report")
        st.dataframe(
            pd.read_csv(CLASSIFICATION_REPORT_PATH), hide_index=True, use_container_width=True
        )
    st.caption(
        "Dataset metrics may overestimate field reliability because PlantVillage images commonly use controlled backgrounds."
    )


def render_library() -> None:
    labels = load_class_names()
    items = library_conditions(labels)
    ai_supported = [item for item in items if item.model_supported]
    reference_only = [item for item in items if not item.model_supported]
    hero(
        "Curated plant-health knowledge",
        "Disease library",
        f"Explore {len(items)} crop and condition references. AI-supported entries match the installed "
        "CNN; reference-only entries broaden education without overstating model capability.",
    )

    metrics = st.columns(4)
    metrics[0].metric("Entries", len(items))
    metrics[1].metric("AI classes", len(ai_supported))
    metrics[2].metric("References", len(reference_only))
    metrics[3].metric("Crops", len({item.crop for item in items}))
    st.info(
        "AI-supported means the installed model has a matching output class. Reference-only entries are "
        "educational and cannot be selected as CNN predictions."
    )

    crops = sorted({item.crop for item in items})
    categories = sorted({item.category for item in items})
    search_col, crop_col, category_col = st.columns((1.5, 1, 1))
    with search_col:
        search = st.text_input(
            "Search library", placeholder="Crop, condition, symptom, or category"
        )
    with crop_col:
        selected_crop = st.selectbox("Crop", ("All crops", *crops))
    with category_col:
        selected_category = st.selectbox("Category", ("All categories", *categories))
    support_filter = st.radio(
        "Model coverage",
        ("All entries", "AI-supported only", "Reference-only"),
        horizontal=True,
    )

    query = search.casefold().strip()
    filtered = [
        item
        for item in items
        if (selected_crop == "All crops" or item.crop == selected_crop)
        and (selected_category == "All categories" or item.category == selected_category)
        and (support_filter != "AI-supported only" or item.model_supported)
        and (support_filter != "Reference-only" or not item.model_supported)
        and (
            not query
            or query
            in f"{item.crop} {item.condition} {item.category} {item.description} {item.symptoms}".casefold()
        )
    ]
    st.caption(f"Showing {len(filtered)} of {len(items)} library entries")
    if not filtered:
        st.warning("No entries match these filters. Clear the search or broaden the filters.")
        return

    for item in filtered:
        coverage = "AI-supported" if item.model_supported else "Reference-only"
        prefix = "●" if item.model_supported else "○"
        with st.expander(f"{prefix} {item.crop} · {item.condition}"):
            coverage_class = "vita-badge-ai" if item.model_supported else "vita-badge-reference"
            st.markdown(
                f'<span class="vita-badge {coverage_class}">{coverage}</span>'
                f'<span class="vita-badge vita-badge-category">{item.category}</span>',
                unsafe_allow_html=True,
            )
            st.write(item.description)
            st.markdown(f"**Common visible symptoms**  \n{item.symptoms}")
            st.markdown(f"**Conservative prevention guidance**  \n{item.prevention}")
            if not item.model_supported:
                st.caption(
                    "Knowledge-base entry only — the current CNN cannot predict this condition. "
                    "Use local diagnostic services for confirmation."
                )

    st.divider()
    with st.expander("Library standards and authoritative references"):
        st.markdown(
            "Guidance follows integrated pest-management principles: prevention, clean planting material, "
            "sanitation, resistant varieties, monitoring, and confirmation by qualified local professionals. "
            "Recommendations deliberately exclude pesticide products and dosages."
        )
        st.markdown(
            "- [FAO integrated pest-management principles]"
            "(https://www.fao.org/pest-and-pesticide-management/ipm/principles-and-practices/en/)\n"
            "- [American Phytopathological Society: plant disease diagnosis]"
            "(https://www.apsnet.org/edcenter/Pages/PlantDiseaseDiagnosis.aspx)\n"
            "- [USDA APHIS citrus disease guidance]"
            "(https://www.aphis.usda.gov/plant-pests-diseases/citrus-diseases)"
        )


def render_about() -> None:
    hero(
        "Responsible AI",
        "About & limitations",
        "A transparent academic decision-support system designed to demonstrate a correct CNN workflow.",
    )
    principles = st.columns(3)
    principle_copy = (
        (
            "Confidence before certainty",
            "The interface changes its guidance when confidence is low and never presents a screening as a confirmed diagnosis.",
        ),
        (
            "Evidence before decoration",
            "Performance figures come from evaluation artifacts, while Grad-CAM is labeled as model attention—not diseased tissue.",
        ),
        (
            "Privacy by default",
            "Uploaded images remain in memory. Only anonymous screening metadata and optional feedback are stored locally.",
        ),
    )
    for column, (title, copy) in zip(principles, principle_copy, strict=True):
        with column:
            st.markdown(
                f'<div class="vita-principle"><strong>{title}</strong><span>{copy}</span></div>',
                unsafe_allow_html=True,
            )

    section_heading(
        "System boundary",
        "What Vita AI does—and what it cannot promise",
        "A responsible interface makes the limits as easy to understand as the capabilities.",
    )
    left, right = st.columns(2, gap="large")
    with left:
        st.subheader("What the system does")
        st.markdown(
            """
            - Converts genuine JPEG/PNG uploads to RGB.
            - Preserves the 224 × 224 × 3 spatial tensor for CNN inference.
            - Checks blur, exposure, dimensions, contrast, and visible detail.
            - Returns top-three predictions with confidence-aware wording.
            - Generates Grad-CAM for models trained by this project.
            - Stores anonymous metadata and optional feedback in SQLite.
            """
        )
    with right:
        st.subheader("What it cannot guarantee")
        st.markdown(
            """
            - It cannot confirm a laboratory diagnosis.
            - It may fail on unsupported crops, mixed conditions, or non-leaf images.
            - PlantVillage performance does not equal field performance.
            - Visual symptoms may also reflect insects, nutrients, weather, or physical damage.
            - It does not provide pesticide products, dosages, or regulatory advice.
            """
        )
    st.subheader("Privacy")
    st.write(
        "No account, name, phone number, email, or location is requested. Images are processed in memory "
        "and not stored. History contains prediction metadata only and can be cleared locally."
    )
    st.subheader("Model lineage")
    st.write(
        "The included training pipeline uses MobileNetV2 transfer learning. A separately downloaded model "
        "must retain its own source, license, label order, preprocessing mode, and evaluation evidence in "
        "`models/model_metadata.json`."
    )


def main() -> None:
    ensure_runtime_directories()
    inject_styles()
    page = sidebar()
    routes = {
        "Home": render_home,
        "Screen a leaf": render_screening,
        "History & analytics": render_history,
        "Model performance": render_performance,
        "Disease library": render_library,
        "About & limitations": render_about,
    }
    routes[page]()


if __name__ == "__main__":
    main()
