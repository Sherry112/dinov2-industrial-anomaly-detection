from __future__ import annotations

import io
import json
import time
from pathlib import Path

import pandas as pd
import streamlit as st
import torch
from PIL import Image, UnidentifiedImageError
from torch import nn

from dino_anomaly.inference import predict_pil_images
from dino_anomaly.model import DinoAnomalyDetector


ROOT = Path(__file__).resolve().parent
ARTIFACT_ROOT = ROOT / "artifacts" / "all_categories"
SUMMARY_PATH = ARTIFACT_ROOT / "summary.json"
MODEL_NAME = "dinov2_vits14"
IMAGE_SIZE = 224
FEATURE_LAYERS = (8, 11)
SAMPLE_LICENSE_URL = "https://creativecommons.org/licenses/by-nc-sa/4.0/"
SAMPLE_SOURCE_URL = "https://www.mvtec.com/research-teaching/datasets/mvtec-ad"


def sample(
    category: str,
    name: str,
    relative_path: str,
    ground_truth: str,
    defect_type: str,
) -> dict[str, object]:
    return {
        "name": name,
        "path": ROOT / "dataset" / "archive" / category / "test" / relative_path,
        "ground_truth": ground_truth,
        "defect_type": defect_type,
    }


SAMPLE_CATALOG = {
    "bottle": (
        sample("bottle", "Sample A", "good/000.png", "normal", "good"),
        sample("bottle", "Sample B", "broken_large/000.png", "anomaly", "broken large"),
    ),
    "cable": (
        sample("cable", "Sample A", "good/000.png", "normal", "good"),
        sample("cable", "Sample B", "bent_wire/000.png", "anomaly", "bent wire"),
    ),
    "capsule": (
        sample("capsule", "Sample A", "good/000.png", "normal", "good"),
        sample("capsule", "Sample B", "crack/000.png", "anomaly", "crack"),
    ),
    "carpet": (
        sample("carpet", "Sample A", "good/000.png", "normal", "good"),
        sample("carpet", "Sample B", "color/000.png", "anomaly", "color"),
    ),
    "grid": (
        sample("grid", "Sample A", "good/009.png", "normal", "good"),
        sample("grid", "Sample B", "bent/000.png", "anomaly", "bent"),
    ),
    "hazelnut": (
        sample("hazelnut", "Sample A", "good/000.png", "normal", "good"),
        sample("hazelnut", "Sample B", "crack/000.png", "anomaly", "crack"),
    ),
    "leather": (
        sample("leather", "Sample A", "good/000.png", "normal", "good"),
        sample("leather", "Sample B", "color/000.png", "anomaly", "color"),
    ),
    "metal_nut": (
        sample("metal_nut", "Sample A", "good/000.png", "normal", "good"),
        sample("metal_nut", "Sample B", "bent/000.png", "anomaly", "bent"),
    ),
    "pill": (
        sample("pill", "Sample A", "good/000.png", "normal", "good"),
        sample("pill", "Sample B", "color/001.png", "anomaly", "color"),
    ),
    "screw": (
        sample("screw", "Sample A", "good/000.png", "normal", "good"),
        sample("screw", "Sample B", "scratch_neck/008.png", "anomaly", "scratch neck"),
    ),
    "tile": (
        sample("tile", "Sample A", "good/000.png", "normal", "good"),
        sample("tile", "Sample B", "crack/000.png", "anomaly", "crack"),
    ),
    "toothbrush": (
        sample("toothbrush", "Sample A", "good/000.png", "normal", "good"),
        sample("toothbrush", "Sample B", "defective/000.png", "anomaly", "defective"),
    ),
    "transistor": (
        sample("transistor", "Sample A", "good/000.png", "normal", "good"),
        sample("transistor", "Sample B", "good/005.png", "normal", "good"),
        sample("transistor", "Sample C", "bent_lead/000.png", "anomaly", "bent lead"),
        sample("transistor", "Sample D", "cut_lead/000.png", "anomaly", "cut lead"),
        sample("transistor", "Sample E", "damaged_case/000.png", "anomaly", "damaged case"),
        sample("transistor", "Sample F", "misplaced/000.png", "anomaly", "misplaced"),
    ),
    "wood": (
        sample("wood", "Sample A", "good/000.png", "normal", "good"),
        sample("wood", "Sample B", "color/000.png", "anomaly", "color"),
    ),
    "zipper": (
        sample("zipper", "Sample A", "good/000.png", "normal", "good"),
        sample("zipper", "Sample B", "broken_teeth/000.png", "anomaly", "broken teeth"),
    ),
}


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def available_categories() -> list[str]:
    if not ARTIFACT_ROOT.exists():
        return []
    return sorted(
        path.parent.name for path in ARTIFACT_ROOT.glob("*/detector.pt")
    )


@st.cache_resource(show_spinner=False)
def load_encoder() -> tuple[nn.Module, torch.device]:
    """Load one frozen backbone shared by every category detector."""
    device = resolve_device()
    encoder = torch.hub.load(
        "facebookresearch/dinov2",
        MODEL_NAME,
        trust_repo=True,
        skip_validation=True,
    ).to(device)
    encoder.eval()
    encoder.requires_grad_(False)
    return encoder, device


@st.cache_resource(show_spinner=False)
def load_detector(category: str) -> tuple[DinoAnomalyDetector, str]:
    """Attach one category's lightweight state to the shared backbone."""
    encoder, device = load_encoder()
    checkpoint = ARTIFACT_ROOT / category / "detector.pt"
    detector = DinoAnomalyDetector(
        model_name=MODEL_NAME,
        device=device,
        image_size=IMAGE_SIZE,
        layers=FEATURE_LAYERS,
        encoder=encoder,
    )
    state = detector.load(checkpoint)
    if state.category != category:
        raise ValueError(
            f"Checkpoint category is '{state.category}', not '{category}'"
        )
    return detector, str(device)


def read_uploaded_images(uploaded_files) -> tuple[list[str], list[Image.Image]]:
    names, images = [], []
    for uploaded_file in uploaded_files:
        try:
            image = Image.open(io.BytesIO(uploaded_file.getvalue())).convert("RGB")
        except (UnidentifiedImageError, OSError) as error:
            raise ValueError(f"Could not read {uploaded_file.name} as an image") from error
        names.append(uploaded_file.name)
        images.append(image)
    return names, images


def load_summary() -> dict:
    if not SUMMARY_PATH.exists():
        return {}
    return json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))


def available_samples(category: str) -> list[dict[str, object]]:
    return [
        item for item in SAMPLE_CATALOG.get(category, ()) if item["path"].is_file()
    ]


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        .block-container {max-width: 1180px; padding-top: 2rem;}
        [data-testid="stMetric"] {
            background: rgba(125, 125, 125, 0.08);
            border: 1px solid rgba(125, 125, 125, 0.18);
            border-radius: 0.75rem;
            padding: 0.8rem;
        }
        .result-normal {color: #16a34a; font-weight: 700;}
        .result-anomaly {color: #dc2626; font-weight: 700;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_inference_page(category: str) -> None:
    st.subheader("Inspect images")
    st.write(
        "Upload up to eight images. The detector compares their DINOv2 patch "
        "features with the selected category's normal-feature coreset."
    )
    input_source = st.radio(
        "Choose an input source",
        ["Sample gallery", "Upload images"],
        horizontal=True,
    )

    names: list[str]
    images: list[Image.Image]
    ground_truths: list[dict[str, str] | None]
    if input_source == "Sample gallery":
        samples = available_samples(category)
        if not samples:
            st.warning(f"No bundled samples are available for `{category}`.")
            return
        sample_lookup = {str(sample["name"]): sample for sample in samples}
        default_samples = (
            ["Sample A", "Sample C"]
            if category == "transistor"
            else ["Sample A", "Sample B"]
        )
        selected_names = st.multiselect(
            "Choose samples",
            options=list(sample_lookup),
            default=default_samples,
            max_selections=6,
            help="Ground-truth labels are revealed after inference.",
        )
        selected_samples = [sample_lookup[name] for name in selected_names]
        if selected_samples:
            preview_columns = st.columns(min(3, len(selected_samples)))
            for index, sample in enumerate(selected_samples):
                preview_columns[index % len(preview_columns)].image(
                    str(sample["path"]),
                    caption=str(sample["name"]),
                    width="stretch",
                )
        st.caption(
            "Sample images: MVTec AD, © MVTec Software GmbH, "
            "CC BY-NC-SA 4.0. Non-commercial educational demonstration."
        )
        st.markdown(
            f"[Dataset source]({SAMPLE_SOURCE_URL}) · "
            f"[License]({SAMPLE_LICENSE_URL})"
        )
        st.caption(
            "Gallery examples are curated for interaction and are not an "
            "unbiased evaluation set. See the Benchmark tab for results over "
            "all test images."
        )
        if not selected_samples:
            st.info("Choose at least one gallery image to enable analysis.")
            return
        names = [str(sample["name"]) for sample in selected_samples]
        images = [Image.open(sample["path"]).convert("RGB") for sample in selected_samples]
        ground_truths = [
            {
                "label": str(sample["ground_truth"]),
                "defect_type": str(sample["defect_type"]),
            }
            for sample in selected_samples
        ]
    else:
        uploaded_files = st.file_uploader(
            "Upload product images",
            type=["png", "jpg", "jpeg", "bmp", "tif", "tiff", "webp"],
            accept_multiple_files=True,
            help="For meaningful results, use images captured similarly to the MVTec category.",
        )
        if len(uploaded_files) > 8:
            st.warning("Please select no more than eight images at once.")
            uploaded_files = uploaded_files[:8]
        if not uploaded_files:
            st.info("Upload one or more images to enable analysis.")
            return
        try:
            names, images = read_uploaded_images(uploaded_files)
        except ValueError as error:
            st.error(str(error))
            return
        ground_truths = [None] * len(images)

    if not st.button("Analyze images", type="primary", width="stretch"):
        return

    try:
        load_started = time.perf_counter()
        with st.spinner("Loading DINOv2 and analyzing image patches…"):
            detector, device = load_detector(category)
            load_seconds = time.perf_counter() - load_started
            inference_started = time.perf_counter()
            predictions = predict_pil_images(detector, images, IMAGE_SIZE)
            inference_seconds = time.perf_counter() - inference_started
    except Exception as error:
        st.error(f"Inference failed: {error}")
        return

    st.caption(
        f"Running on `{device}` · image threshold `{detector.image_threshold:.4f}` "
        f"· pixel threshold `{detector.pixel_threshold:.4f}` · inference "
        f"`{inference_seconds:.2f}s` ({inference_seconds / len(images):.2f}s/image) "
        f"· detector access `{load_seconds:.2f}s`"
    )
    report = []
    for name, prediction, ground_truth in zip(names, predictions, ground_truths):
        st.divider()
        st.markdown(f"### {name}")
        image_col, overlay_col, heatmap_col = st.columns(3)
        image_col.image(prediction.resized_image, caption="Input", width="stretch")
        overlay_col.image(prediction.overlay, caption="Anomaly overlay", width="stretch")
        heatmap_col.image(prediction.heatmap, caption="Anomaly heatmap", width="stretch")

        metric_col, decision_col = st.columns([1, 2])
        metric_col.metric(
            "Anomaly score",
            f"{prediction.score:.4f}",
            delta=f"{prediction.score - detector.image_threshold:+.4f} vs threshold",
            delta_color="inverse",
        )
        if prediction.prediction == "anomaly":
            decision_col.error("Decision: ANOMALY — manual inspection recommended")
        else:
            decision_col.success("Decision: NORMAL for this category")
        if ground_truth is not None:
            matches = prediction.prediction == ground_truth["label"]
            status = "correct" if matches else "incorrect"
            decision_col.caption(
                f"Ground truth: **{ground_truth['label'].upper()}** "
                f"({ground_truth['defect_type']}) · prediction {status}"
            )

        result = {
                "filename": name,
                "category": category,
                "score": prediction.score,
                "image_threshold": detector.image_threshold,
                "pixel_threshold": detector.pixel_threshold,
                "prediction": prediction.prediction,
                "model": MODEL_NAME,
                "feature_layers": list(FEATURE_LAYERS),
            }
        if ground_truth is not None:
            result["ground_truth"] = ground_truth["label"]
            result["defect_type"] = ground_truth["defect_type"]
            result["prediction_correct"] = prediction.prediction == ground_truth["label"]
        report.append(result)

    st.download_button(
        "Download prediction report",
        data=json.dumps(report, indent=2) + "\n",
        file_name=f"{category}_predictions.json",
        mime="application/json",
        width="stretch",
    )
    st.warning(
        "This is a research demo, not a production quality-control decision. "
        "Different cameras, crops, backgrounds, or lighting can raise the score."
    )


def render_benchmark_page() -> None:
    st.subheader("MVTec AD benchmark")
    summary = load_summary()
    if not summary:
        st.info("No benchmark summary is available.")
        return

    macro = summary["macro_average"]
    metric_columns = st.columns(4)
    metric_columns[0].metric("Image AUROC", f"{macro['image_auroc']:.3f}")
    metric_columns[1].metric("Image AP", f"{macro['image_average_precision']:.3f}")
    metric_columns[2].metric("Pixel AUROC", f"{macro['pixel_auroc']:.3f}")
    metric_columns[3].metric("Image F1", f"{macro['image_f1']:.3f}")

    rows = summary["categories"]
    frame = pd.DataFrame(rows).set_index("category")
    chart_columns = ["image_auroc", "pixel_auroc"]
    st.markdown("#### Ranking quality by category")
    st.bar_chart(frame[chart_columns], height=420)

    display_columns = [
        "image_auroc",
        "image_average_precision",
        "pixel_auroc",
        "pixel_average_precision",
        "image_f1",
    ]
    st.markdown("#### Detailed results")
    st.dataframe(
        frame[display_columns].rename(
            columns={
                "image_auroc": "Image AUROC",
                "image_average_precision": "Image AP",
                "pixel_auroc": "Pixel AUROC",
                "pixel_average_precision": "Pixel AP",
                "image_f1": "Image F1",
            }
        ),
        width="stretch",
    )
    st.caption(
        "Macro averages give every category equal weight. Thresholds were "
        "calibrated on held-out normal images before test evaluation."
    )


def render_method_page() -> None:
    st.subheader("How the detector works")
    st.markdown(
        """
        1. **Normal-only fitting:** 90% of each category's normal training images
           supply reference features.
        2. **DINOv2 representation:** patch tokens from transformer blocks 8 and
           11 are normalized and fused.
        3. **Coreset compression:** greedy k-center selection retains 2,048
           representative normal patches.
        4. **Calibration:** the held-out 10% normal split defines image and pixel
           anomaly thresholds without using test labels.
        5. **Inference:** each uploaded patch is scored by cosine distance to its
           nearest normal reference. Patch scores form the heatmap; the most
           anomalous patches determine the image score.
        """
    )
    st.markdown("#### Known limitations")
    st.markdown(
        """
        - Internet images can differ substantially from the benchmark domain.
        - The 16×16 patch grid limits very fine defect boundaries.
        - A normal-only threshold controls false alarms but cannot encode the
          real business cost of missed defects.
        - The demo is intended for education and portfolio review, not factory
          deployment.
        """
    )


def main() -> None:
    st.set_page_config(
        page_title="DINOv2 Visual Inspector",
        page_icon="🔍",
        layout="wide",
    )
    inject_styles()
    st.title("DINOv2 Visual Inspector")
    st.caption(
        "Normal-only industrial anomaly detection with localized heatmaps"
    )

    categories = available_categories()
    if not categories:
        st.error(
            "No fitted detector checkpoints were found under "
            "`artifacts/all_categories/<category>/detector.pt`."
        )
        st.stop()

    default_index = categories.index("transistor") if "transistor" in categories else 0
    with st.sidebar:
        st.header("Detector")
        category = st.selectbox("MVTec category", categories, index=default_index)
        st.caption(f"Checkpoint: `artifacts/all_categories/{category}/detector.pt`")
        st.markdown("---")
        st.markdown("**Model:** DINOv2 ViT-S/14")
        st.markdown("**Feature blocks:** 8 and 11")
        st.markdown("**Input:** 224 × 224")
        st.markdown("**Memory:** 2,048 coreset patches")
        st.markdown("**Runtime:** one shared encoder")

    inference_tab, benchmark_tab, method_tab = st.tabs(
        ["Inference", "Benchmark", "Method & limitations"]
    )
    with inference_tab:
        render_inference_page(category)
    with benchmark_tab:
        render_benchmark_page()
    with method_tab:
        render_method_page()


if __name__ == "__main__":
    main()
