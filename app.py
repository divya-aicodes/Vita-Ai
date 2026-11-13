"""Vita AI Streamlit interface."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from vita_ai.config import (  # noqa: E402
    APP_NAME,
    APP_VERSION,
    DISTRIBUTIONS,
    ENVIRONMENTS,
    ML_ARTIFACTS_DIR,
    SYMPTOMS,
)
from vita_ai.image_quality import ImageValidationError, assess_photo, decode_image  # noqa: E402
from vita_ai.knowledge import crops, load_conditions  # noqa: E402
from vita_ai.ml.registry import MODEL_SPECS  # noqa: E402
from vita_ai.ml.results import load_leaderboard  # noqa: E402
from vita_ai.report import build_report  # noqa: E402
from vita_ai.screening import screen_case  # noqa: E402
from vita_ai.storage import list_screenings, save_feedback, save_screening  # noqa: E402

st.set_page_config(
    page_title="Vita AI · Plant Health Screening",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      :root { --ink: #163126; --green: #286444; --mint: #eef6f0; --line: #d7e5db; }
      .stApp { background: #fbfcfa; color: var(--ink); }
      [data-testid="stSidebar"] { background: #f1f6f2; border-right: 1px solid var(--line); }
      .hero { background: #eef6f0; border: 1px solid #d2e3d7; border-radius: 18px;
              padding: 2rem 2.2rem; margin-bottom: 1.25rem; }
      .hero h1 { color: #173f2a; margin: 0 0 .35rem; font-size: 2.55rem; }
      .hero p { color: #456052; max-width: 760px; font-size: 1.05rem; margin: 0; }
      .eyebrow { color: #2f6d4a; font-weight: 700; letter-spacing: .08em;
                 text-transform: uppercase; font-size: .75rem; }
      .notice { border-left: 4px solid #c28a25; background: #fff8e8; padding: .85rem 1rem;
                border-radius: 8px; margin: .8rem 0 1.1rem; color: #604817; }
      .card { border: 1px solid var(--line); border-radius: 14px; padding: 1rem 1.1rem;
              background: white; min-height: 150px; }
      .card h3 { color: #214d35; margin-top: 0; }
      .score { font-variant-numeric: tabular-nums; color: #286444; font-weight: 700; }
      div[data-testid="stMetric"] { background: white; border: 1px solid var(--line);
              padding: .75rem 1rem; border-radius: 12px; }
      .small-note { color: #5d7166; font-size: .88rem; }
      .status-good { color: #1f6a3c; font-weight: 700; }
      .status-usable { color: #8a631b; font-weight: 700; }
      .status-poor { color: #a23c32; font-weight: 700; }
      .stButton > button, .stDownloadButton > button { border-radius: 9px; }
    </style>
    """,
    unsafe_allow_html=True,
)


def page_header(title: str, subtitle: str) -> None:
    st.markdown(
        f'<div class="hero"><div class="eyebrow">{APP_NAME} · prototype {APP_VERSION}</div>'
        f"<h1>{title}</h1><p>{subtitle}</p></div>",
        unsafe_allow_html=True,
    )


def render_home() -> None:
    page_header(
        "Plant-health screening that knows when to stop",
        "Vita AI combines a private photo-quality check with transparent symptom matching. "
        "It shows its evidence, avoids invented accuracy claims, and abstains when the case "
        "is unclear.",
    )
    st.markdown(
        '<div class="notice"><strong>Prototype boundary:</strong> Vita AI is decision support, '
        "not a disease diagnosis. The match score is a rule match—not model confidence or "
        "accuracy.</div>",
        unsafe_allow_html=True,
    )

    conditions = load_conditions()
    metric_cols = st.columns(4)
    metric_cols[0].metric("Supported crops", len(crops()))
    metric_cols[1].metric("Curated patterns", len(conditions))
    metric_cols[2].metric("Images retained", "0")
    metric_cols[3].metric("Unverified claims", "0")

    st.subheader("How a screening works")
    columns = st.columns(3)
    cards = [
        (
            "1 · Check the photo",
            "The image is decoded safely and assessed for exposure, focus, resolution, and "
            "visible leaf coverage.",
        ),
        (
            "2 · Match evidence",
            "Crop, visible symptoms, location on the plant, weather, and broad colour cues are "
            "scored transparently.",
        ),
        (
            "3 · Act cautiously",
            "The app shows a shortlist and safe next steps—or explicitly returns no reliable "
            "single match.",
        ),
    ]
    for column, (title, body) in zip(columns, cards, strict=True):
        column.markdown(
            f'<div class="card"><h3>{title}</h3><p>{body}</p></div>', unsafe_allow_html=True
        )

    st.subheader("Designed for honest iteration")
    st.write(
        "Anonymous metadata and optional user feedback are stored locally in SQLite. That "
        "creates a path to a future validated vision model without pretending this prototype "
        "has field accuracy evidence."
    )


def photo_panel(uploaded_file):
    if uploaded_file is None:
        st.info("Upload one clear leaf photo. The file is processed in memory and is not saved.")
        return None, None
    try:
        image = decode_image(uploaded_file)
        assessment = assess_photo(image)
    except ImageValidationError as exc:
        st.error(str(exc))
        return None, None

    left, right = st.columns([1.25, 1])
    left.image(image, caption="Local preview", use_container_width=True)
    right.markdown(
        "**Photo status:** "
        f"<span class='status-{assessment.status}'>{assessment.status.title()}</span>",
        unsafe_allow_html=True,
    )
    right.progress(assessment.quality_score, text=f"Capture quality {assessment.quality_score:.0%}")
    c1, c2 = right.columns(2)
    c1.metric("Resolution", f"{assessment.width} × {assessment.height}")
    c2.metric("Leaf coverage", f"{assessment.leaf_coverage:.0%}")
    if assessment.issues:
        right.caption("Improve the photo")
        for issue in assessment.issues:
            right.write(f"• {issue}")
    else:
        right.success("No major capture-quality issue detected.")
    right.caption("Colour measurements are weak supporting cues and never diagnose a condition.")
    return image, assessment


def render_result(result, case_id: str, payload: dict) -> None:
    if result.status == "screened":
        st.success(result.headline)
    elif result.status in {"needs_review", "needs_better_photo"}:
        st.warning(result.headline)
    else:
        st.info(result.headline)
    st.write(result.explanation)

    if result.candidates:
        st.subheader("Transparent shortlist")
        table = [
            {
                "Candidate": item.name,
                "Type": item.category.title(),
                "Rule match": f"{item.match_score:.0%}",
                "Evidence matched": ", ".join(item.matched_evidence) or "Limited",
                "Urgency": item.urgency.title(),
            }
            for item in result.candidates
        ]
        st.dataframe(table, hide_index=True, use_container_width=True)

        for index, candidate in enumerate(result.candidates, start=1):
            with st.expander(f"{index}. {candidate.name} · {candidate.match_score:.0%} rule match"):
                st.write(candidate.overview)
                st.markdown("**Matched evidence**")
                st.write(", ".join(candidate.matched_evidence) or "No strong symptom evidence")
                st.markdown("**Useful checks that were not reported**")
                st.write(", ".join(candidate.missing_evidence) or "None")
                st.markdown("**Safe next steps**")
                for step in candidate.next_steps:
                    st.write(f"- {step}")
                st.markdown("**Prevention**")
                for step in candidate.prevention:
                    st.write(f"- {step}")

    report = build_report(case_id=case_id, result=result, **payload)
    st.download_button(
        "Download screening report",
        data=report,
        file_name=f"vita-ai-{case_id[:8]}.txt",
        mime="text/plain",
        use_container_width=True,
    )

    st.markdown("#### Was this screening useful?")
    feedback_cols = st.columns(3)
    choices = [("Helpful", "helpful"), ("Incorrect", "incorrect"), ("Unsure", "unsure")]
    for column, (label, verdict) in zip(feedback_cols, choices, strict=True):
        if column.button(label, key=f"feedback-{case_id}-{verdict}", use_container_width=True):
            save_feedback(case_id, verdict)
            st.toast("Feedback saved locally. Thank you.")


def render_screening() -> None:
    page_header(
        "Start a new screening",
        "Use one representative leaf, answer only what you can observe, and treat the result "
        "as a shortlist.",
    )
    uploaded_file = st.file_uploader(
        "Leaf photo",
        type=["jpg", "jpeg", "png", "webp"],
        help="Maximum 8 MB. The image is processed in memory and is not stored.",
    )
    _, photo = photo_panel(uploaded_file)

    symptom_by_label = {label: key for key, label in SYMPTOMS.items()}
    environment_by_label = {label: key for key, label in ENVIRONMENTS.items()}
    distribution_by_label = {label: key for key, label in DISTRIBUTIONS.items()}

    with st.form("screening-form"):
        left, right = st.columns(2)
        crop = left.selectbox("Known crop", crops())
        selected_symptom_labels = left.multiselect(
            "Visible symptoms",
            list(symptom_by_label),
            help="Select only visible evidence. Do not guess.",
        )
        distribution_label = left.selectbox("Where symptoms appear", list(distribution_by_label))
        selected_environment_labels = right.multiselect(
            "Recent conditions",
            list(environment_by_label),
        )
        notes = right.text_area(
            "Optional notes",
            max_chars=1000,
            placeholder="When symptoms started, how quickly they spread, watering changes…",
        )
        consent = st.checkbox(
            "I understand this is a screening prototype, not a diagnosis or treatment prescription."
        )
        submitted = st.form_submit_button(
            "Screen this case", type="primary", use_container_width=True
        )

    if submitted:
        if uploaded_file is None or photo is None:
            st.error("Upload a valid leaf photo before screening.")
            return
        if not consent:
            st.error("Please acknowledge the prototype boundary before continuing.")
            return
        symptom_keys = [symptom_by_label[item] for item in selected_symptom_labels]
        environment_keys = [environment_by_label[item] for item in selected_environment_labels]
        distribution = distribution_by_label[distribution_label]
        result = screen_case(
            crop=crop,
            symptoms=symptom_keys,
            environments=environment_keys,
            distribution=distribution,
            photo=photo,
        )
        payload = {
            "crop": crop,
            "symptoms": symptom_keys,
            "environments": environment_keys,
            "distribution": distribution,
            "notes": notes,
        }
        case_id = save_screening(result=result, **payload)
        st.session_state["latest_screening"] = (result, case_id, payload)

    if "latest_screening" in st.session_state:
        result, case_id, payload = st.session_state["latest_screening"]
        st.divider()
        render_result(result, case_id, payload)


def render_history() -> None:
    page_header(
        "Local screening history",
        "Only anonymous case metadata and optional feedback are retained. Uploaded images are "
        "never saved.",
    )
    rows = list_screenings()
    if not rows:
        st.info("No screenings have been saved on this device yet.")
        return
    st.metric("Saved cases", len(rows))
    display = []
    for row in rows:
        display.append(
            {
                "Created": row["created_at"].replace("T", " ")[:19] + " UTC",
                "Crop": row["crop"],
                "Outcome": row["status"].replace("_", " ").title(),
                "Primary candidate": row["primary_condition"] or "None",
                "Rule match": f"{row['match_score']:.0%}"
                if row["match_score"] is not None
                else "—",
                "Photo": (row["photo_status"] or "—").title(),
                "Feedback": (row["verdict"] or "—").title(),
            }
        )
    st.dataframe(display, hide_index=True, use_container_width=True)
    with st.expander("Inspect latest stored metadata"):
        latest = dict(rows[0])
        latest["symptoms_json"] = json.loads(latest["symptoms_json"])
        latest["environments_json"] = json.loads(latest["environments_json"])
        latest.pop("result_json", None)
        st.json(latest)


def render_library() -> None:
    page_header(
        "Plant-health knowledge library",
        "Browse the curated patterns used by the transparent matching engine.",
    )
    crop_filter = st.selectbox("Crop", ("All", *crops()))
    conditions = [
        item for item in load_conditions() if crop_filter == "All" or item["crop"] == crop_filter
    ]
    for item in conditions:
        with st.expander(f"{item['crop']} · {item['name']} · {item['category'].title()}"):
            st.write(item["overview"])
            st.markdown("**Typical observable evidence**")
            st.write(", ".join(SYMPTOMS.get(key, key) for key in item["symptoms"]))
            st.markdown("**Safe next steps**")
            for step in item["next_steps"]:
                st.write(f"- {step}")


def render_model_lab() -> None:
    page_header(
        "Five-model evaluation lab",
        "Train every candidate on the same data split, then select the winner using untouched "
        "test-set macro F1, balanced accuracy, and measured inference time.",
    )
    st.dataframe(
        [
            {
                "Model": spec.display_name,
                "Input": f"{spec.image_size} × {spec.image_size}",
                "Role": spec.purpose,
            }
            for spec in MODEL_SPECS
        ],
        hide_index=True,
        use_container_width=True,
    )

    leaderboard_path = ML_ARTIFACTS_DIR / "leaderboard.json"
    try:
        leaderboard = load_leaderboard(leaderboard_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        st.error(f"The generated leaderboard could not be read: {exc}")
        return
    if leaderboard is None:
        st.info(
            "No trained artifacts are installed yet. The repository contains the reproducible "
            "training and evaluation pipeline, but it will not display invented performance."
        )
        st.code(
            "python scripts/train_models.py --data-dir data/plant-disease",
            language="powershell",
        )
        return

    rows = []
    for item in leaderboard["models"]:
        rows.append(
            {
                "Model": item["model"],
                "Accuracy": f"{item['accuracy']:.2%}",
                "Macro F1": f"{item['macro_f1']:.2%}",
                "Balanced accuracy": f"{item['balanced_accuracy']:.2%}",
                "Latency": f"{item['inference_ms_per_image']:.1f} ms/image",
                "Size": f"{item['model_size_mb']:.1f} MB",
            }
        )
    st.success(f"Selected by test macro F1: {rows[0]['Model']}")
    st.dataframe(rows, hide_index=True, use_container_width=True)
    st.caption(
        "These values describe only the configured held-out test set. They are not automatically "
        "valid for field photographs, new regions, or unsupported crops."
    )


def render_about() -> None:
    page_header(
        "About this prototype",
        "A from-scratch Python implementation focused on truthful evidence and a safe path to "
        "validation.",
    )
    st.subheader("What it does now")
    st.write(
        "Vita AI validates a leaf photo, measures capture quality and broad colour cues, "
        "combines those with structured symptom evidence, ranks curated condition patterns, "
        "and stores anonymous local feedback."
    )
    st.subheader("What it deliberately does not claim")
    st.write(
        "The repository includes a five-model training and comparison pipeline, but does not "
        "claim a trained classifier is validated until real artifacts and held-out results are "
        "present. Rule-match percentages remain transparent evidence scores, not probabilities. "
        "The app never recommends pesticide products or doses."
    )
    st.subheader("Production path")
    st.markdown(
        "1. Collect consented, expert-labelled field images from target regions.\n"
        "2. Split by farm, plant, and capture session to prevent leakage.\n"
        "3. Train an out-of-distribution-aware vision model.\n"
        "4. Calibrate probabilities on a locked field validation set.\n"
        "5. Publish per-crop sensitivity, specificity, abstention rate, and known limitations."
    )


with st.sidebar:
    st.markdown("## 🌿 Vita AI")
    st.caption("Transparent plant-health screening")
    page = st.radio(
        "Navigate", ("Overview", "New screening", "History", "Knowledge", "Model lab", "About")
    )
    st.divider()
    st.caption("Photos stay in memory. Metadata stays on this device.")

if page == "Overview":
    render_home()
elif page == "New screening":
    render_screening()
elif page == "History":
    render_history()
elif page == "Knowledge":
    render_library()
elif page == "Model lab":
    render_model_lab()
else:
    render_about()
