from pathlib import Path

from streamlit.testing.v1 import AppTest

from dino_anomaly.cli import MVTEC_CATEGORIES
from streamlit_app import ARTIFACT_ROOT, SAMPLE_CATALOG, load_detector


def test_every_category_has_published_samples_and_checkpoint():
    assert set(SAMPLE_CATALOG) == set(MVTEC_CATEGORIES)
    for category, samples in SAMPLE_CATALOG.items():
        assert len(samples) >= 2
        assert {sample["ground_truth"] for sample in samples} == {"normal", "anomaly"}
        assert all(sample["path"].is_file() for sample in samples)
        assert (ARTIFACT_ROOT / category / "detector.pt").is_file()


def test_streamlit_app_initial_view_has_no_exceptions():
    app_path = Path(__file__).parents[1] / "streamlit_app.py"
    app = AppTest.from_file(app_path, default_timeout=10).run()

    assert not app.exception
    assert app.title[0].value == "DINOv2 Visual Inspector"
    assert app.selectbox[0].value == "transistor"
    assert len(app.tabs) == 3
    assert app.radio[0].value == "Sample gallery"
    assert app.multiselect[0].value == ["Sample A", "Sample C"]


def test_sample_gallery_runs_normal_and_anomaly_inference():
    app_path = Path(__file__).parents[1] / "streamlit_app.py"
    app = AppTest.from_file(app_path, default_timeout=30).run()
    app.button[0].click().run(timeout=30)

    ground_truth_captions = [
        item.value for item in app.caption if "Ground truth" in item.value
    ]
    assert not app.exception
    assert len(app.success) == 1
    assert len(app.error) == 1
    assert ground_truth_captions == [
        "Ground truth: **NORMAL** (good) · prediction correct",
        "Ground truth: **ANOMALY** (bent lead) · prediction correct",
    ]


def test_gallery_updates_when_category_changes():
    app_path = Path(__file__).parents[1] / "streamlit_app.py"
    app = AppTest.from_file(app_path, default_timeout=10).run()
    app.selectbox[0].select("bottle").run()

    assert not app.exception
    assert app.multiselect[0].value == ["Sample A", "Sample B"]


def test_category_detectors_share_the_cached_encoder():
    transistor, _ = load_detector("transistor")
    bottle, _ = load_detector("bottle")

    assert transistor is not bottle
    assert transistor.encoder is bottle.encoder
