"""Streamlit interface for ᕓ𐌉𐌕𐌀 𐌀𐌉."""

from __future__ import annotations

import json
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
    MODEL_VERSION,
    ensure_runtime_directories,
    load_class_names,
    load_model_metadata,
)
from src.database import PredictionDatabase
from src.disease_info import supported_conditions
from src.model import ModelUnavailableError, load_trained_model
from src.prediction import predict_image
from src.preprocessing import ImageValidationError, load_image
from src.quality_checker import assess_image_quality
from src.report import build_text_report

APP_ROOT = Path(__file__).resolve().parent
LOGO_PATH = APP_ROOT / "assets" / "vita-ai-logo.png"
ICON_PATH = APP_ROOT / "assets" / "vita-ai-icon.png"

st.set_page_config(
    page_title=f"{APP_TITLE} · Plant Health Screening",
    page_icon=str(ICON_PATH) if ICON_PATH.exists() else "🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --forest: #173f35;
            --leaf: #3b735d;
            --sage: #dfeadd;
            --lime: #d5e76b;
            --paper: #f7f8f2;
            --ink: #17201c;
            --muted: #65706a;
            --line: #d9dfd7;
        }
        .stApp {
            background:
                radial-gradient(circle at 90% 4%, rgba(213,231,107,.18), transparent 24rem),
                linear-gradient(180deg, #f9faf5 0%, #f3f6ef 100%);
            color: var(--ink);
        }
        [data-testid="stSidebar"] {
            background: #173f35;
            border-right: 1px solid rgba(255,255,255,.08);
        }
        [data-testid="stSidebar"] * { color: #f4f6ee; }
        [data-testid="stSidebar"] div[role="radiogroup"] label {
            border-radius: 12px;
            padding: .42rem .65rem;
        }
        [data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
            background: rgba(213,231,107,.16);
        }
        .vita-kicker {
            color: var(--leaf);
            font-size: .76rem;
            font-weight: 800;
            letter-spacing: .16em;
            text-transform: uppercase;
            margin-bottom: .4rem;
        }
        .vita-hero {
            padding: 2.1rem 2.2rem;
            border-radius: 24px;
            color: #f9fbf5;
            background:
                radial-gradient(circle at 90% 15%, rgba(213,231,107,.32), transparent 12rem),
                linear-gradient(135deg, #173f35 0%, #2f6753 100%);
            box-shadow: 0 22px 55px rgba(23,63,53,.14);
            margin-bottom: 1.2rem;
        }
        .vita-hero h1 {
            color: #fff;
            font-size: clamp(2rem, 5vw, 4.5rem);
            line-height: .95;
            letter-spacing: -.04em;
            margin: 0 0 .8rem;
        }
        .vita-hero p { color: #e8efe6; max-width: 48rem; font-size: 1.08rem; margin: 0; }
        .vita-card {
            background: rgba(255,255,255,.78);
            border: 1px solid var(--line);
            border-radius: 18px;
            padding: 1.15rem 1.2rem;
            height: 100%;
            box-shadow: 0 10px 30px rgba(31,60,48,.05);
        }
        .vita-card h3 { margin-top: 0; color: var(--forest); }
        .vita-result {
            background: linear-gradient(135deg, #e8f0e4, #f6f8ed);
            border: 1px solid #cbd9c8;
            border-left: 6px solid #3b735d;
            border-radius: 18px;
            padding: 1.25rem 1.4rem;
        }
        .vita-result .condition { color: var(--forest); font-size: 1.7rem; font-weight: 800; }
        .vita-result .crop { color: var(--muted); font-size: .9rem; text-transform: uppercase; letter-spacing: .12em; }
        .vita-disclaimer {
            background: #fff8e7;
            border: 1px solid #ead8aa;
            color: #584716;
            border-radius: 14px;
            padding: .9rem 1rem;
            font-size: .9rem;
        }
        .model-ready { color: #d5e76b; font-weight: 700; }
        .model-missing { color: #ffd591; font-weight: 700; }
        [data-testid="stMetric"] {
            background: rgba(255,255,255,.72);
            border: 1px solid var(--line);
            border-radius: 16px;
            padding: .85rem 1rem;
        }
        .stButton > button, .stDownloadButton > button {
            border-radius: 999px;
            font-weight: 750;
            border-color: #3b735d;
        }
        .stButton > button[kind="primary"] {
            background: #173f35;
            color: white;
        }
        [data-testid="stFileUploaderDropzone"] {
            border: 1.5px dashed #7a9b8d;
            border-radius: 18px;
            background: rgba(255,255,255,.65);
        }
        @media (max-width: 700px) {
            .vita-hero { padding: 1.5rem; border-radius: 18px; }
            .vita-hero h1 { font-size: 2.6rem; }
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
    if not MODEL_PATH.exists():
        return False, "Model not installed"
    try:
        model_bundle()
        return True, "CNN ready"
    except Exception as exc:
        return False, f"Model needs attention: {exc}"


def hero(kicker: str, heading: str, body: str) -> None:
    render_animated_hero(kicker, heading, body)


def sidebar() -> str:
    if LOGO_PATH.exists():
        st.sidebar.image(str(LOGO_PATH), width=104)
    st.sidebar.markdown(f"## {APP_TITLE}")
    st.sidebar.caption("Explainable plant-health screening")
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
        "Plant health, explained",
        APP_TITLE,
        f"{APP_SUBTITLE}. Upload a clear leaf image to receive a confidence-aware educational screening.",
    )
    left, middle, right = st.columns(3)
    with left:
        st.markdown(
            '<div class="vita-card"><h3>01 · Screen</h3><p>Upload or capture one clear leaf. '
            "The app checks focus, lighting, contrast, size, and visible detail first.</p></div>",
            unsafe_allow_html=True,
        )
    with middle:
        st.markdown(
            '<div class="vita-card"><h3>02 · Understand</h3><p>See the predicted crop, condition, '
            "confidence level, top alternatives, and the image regions that influenced the CNN.</p></div>",
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(
            '<div class="vita-card"><h3>03 · Act carefully</h3><p>Read conservative prevention guidance '
            "and know when to seek help—without pesticide dosages or false certainty.</p></div>",
            unsafe_allow_html=True,
        )
    st.write("")
    st.subheader("How to take a useful photo")
    cols = st.columns(4)
    tips = (
        ("Natural light", "Use bright, even light without glare."),
        ("One main leaf", "Keep the symptomatic leaf large in the frame."),
        ("Sharp focus", "Hold steady and focus on spots or discoloration."),
        ("Simple background", "Reduce clutter behind the leaf when possible."),
    )
    for column, (title, text) in zip(cols, tips):
        with column:
            st.markdown(f"**{title}**  \n{text}")
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


def render_prediction_result(result, quality, prediction_id: int, model, labels, metadata, image) -> None:
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
            st.warning("Prevention guidance is hidden because confidence is low. Try another image first.")
        st.warning(info.expert_warning)
    with tab_explain:
        if primary.confidence < 0.55:
            st.info("Explainability is withheld for this low-confidence result; capture another image.")
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
            except Exception as exc:
                st.info(f"Grad-CAM is unavailable for this model build: {exc}")
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
    choices = (("Correct", "Appears correct"), ("Incorrect", "Appears incorrect"), ("Unsure", "I’m unsure"))
    for column, (value, label) in zip(feedback_cols, choices):
        if column.button(label, key=f"feedback-{prediction_id}-{value}", use_container_width=True):
            database().update_feedback(prediction_id, value)
            st.success("Anonymous feedback saved. Thank you.")


def render_screening() -> None:
    hero(
        "Disease detection",
        "Screen a leaf",
        "Start with image quality, then run the CNN only when the image contains enough usable detail.",
    )
    source_mode = st.radio("Image source", ("Upload an image", "Use camera"), horizontal=True)
    source = (
        st.file_uploader("Choose a JPG, JPEG, or PNG image", type=("jpg", "jpeg", "png"))
        if source_mode == "Upload an image"
        else st.camera_input("Take a close, well-lit leaf photo")
    )
    if source is None:
        st.info("Your image stays in memory for analysis and is not saved to prediction history.")
        return
    try:
        image = load_image(source)
    except ImageValidationError as exc:
        st.error(str(exc))
        return
    preview, assessment = st.columns((1, 1.1), gap="large")
    with preview:
        st.image(image, caption=f"Selected image · {image.width} × {image.height}", use_container_width=True)
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
    except Exception as exc:
        st.error(f"Analysis stopped safely: {exc}")


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
        st.info("No screenings have been recorded yet.")
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
    st.dataframe([{key: row[key] for key in visible} for row in rows], hide_index=True, use_container_width=True)
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
    cols = st.columns(5)
    displayed = (
        ("Test accuracy", "accuracy"),
        ("Top-3 accuracy", "top_3_accuracy"),
        ("Macro precision", "macro_precision"),
        ("Macro recall", "macro_recall"),
        ("Macro F1", "macro_f1"),
    )
    for column, (label, key) in zip(cols, displayed):
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
        st.dataframe(pd.read_csv(CLASSIFICATION_REPORT_PATH), hide_index=True, use_container_width=True)
    st.caption(
        "Dataset metrics may overestimate field reliability because PlantVillage images commonly use controlled backgrounds."
    )


def render_library() -> None:
    hero(
        "Supported knowledge",
        "Disease library",
        "Browse the crop and condition classes understood by the default 38-class PlantVillage model.",
    )
    labels = load_class_names()
    items = supported_conditions(labels)
    crops = sorted({item.crop for item in items})
    selected = st.selectbox("Filter by crop", ("All crops", *crops))
    search = st.text_input("Search conditions", placeholder="e.g. early blight")
    filtered = [
        item
        for item in items
        if (selected == "All crops" or item.crop == selected)
        and (not search or search.casefold() in f"{item.crop} {item.condition}".casefold())
    ]
    st.caption(f"{len(filtered)} supported class(es)")
    for item in filtered:
        with st.expander(f"{item.crop} · {item.condition}"):
            st.write(item.description)
            st.markdown(f"**Common symptoms:** {item.symptoms}")
            st.markdown(f"**General prevention:** {item.prevention}")


def render_about() -> None:
    hero(
        "Responsible AI",
        "About & limitations",
        "A transparent academic decision-support system designed to demonstrate a correct CNN workflow.",
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
