import json
from pathlib import Path

import numpy as np
import pytest
import torch

from vision_lib import CLASSES, SmallCNN, augment, gradcam, make_resnet, to_tensor

ROOT = Path(__file__).resolve().parents[1]


def test_to_tensor_shapes_and_normalisation():
    x = np.full((2, 64, 64), 255, np.uint8)
    t = to_tensor(x, 96)
    assert t.shape == (2, 3, 96, 96) and torch.allclose(t[0, 0, 0, 0], torch.tensor((1 - 0.449) / 0.226))


def test_augment_preserves_content():
    t = torch.arange(3 * 8 * 8, dtype=torch.float32).reshape(3, 8, 8)
    a = augment(t, torch.Generator().manual_seed(0))
    assert a.shape == t.shape
    assert torch.equal(a.flatten().sort().values, t.flatten().sort().values)


def test_models_output_seven_classes():
    x = torch.zeros(2, 3, 64, 64)
    assert SmallCNN()(x).shape == (2, len(CLASSES))
    assert make_resnet(False)(torch.zeros(2, 3, 96, 96)).shape == (2, len(CLASSES))


def test_gradcam_returns_normalised_map():
    torch.manual_seed(0)
    cam, cls = gradcam(make_resnet(False), torch.randn(1, 3, 96, 96))
    assert cam.shape == (96, 96) and 0 <= cam.min() and cam.max() <= 1 + 1e-6 and 0 <= cls < 7


def test_board_scan_covers_whole_image():
    from board_scan import scan
    res = scan(np.zeros((130, 100), np.uint8), SmallCNN().eval(), "small_cnn_scratch", stride=32)
    ys = {w["y"] for w in res["windows"]}; xs = {w["x"] for w in res["windows"]}
    assert max(ys) == 130 - 64 and max(xs) == 100 - 64 and res["heatmap"].shape == (130, 100)


def test_template_baseline_flags_a_difference():
    from evaluate import template_scores
    temp = np.zeros((2, 64, 64), np.uint8)
    test = temp.copy(); test[1, 20:30, 20:30] = 255
    s = template_scores({"test": test, "template": temp}, k=5, t=150)
    assert s[0] == 0 and s[1] > 0


@pytest.mark.skipif(not (ROOT / "reports" / "metrics.json").exists(), reason="run src/evaluate.py")
def test_metrics_file_is_consistent():
    m = json.loads((ROOT / "reports" / "metrics.json").read_text())
    for name, c in m["classifiers"].items():
        cm = np.array(c["confusion_matrix"])
        assert cm.sum() == sum(m["test_counts"].values())
        assert abs(np.trace(cm) / cm.sum() - c["accuracy"]) < 1e-3


@pytest.mark.skipif(not (ROOT / "models" / "small_cnn_scratch.pt").exists(), reason="run src/train.py")
def test_model_loader_finds_trained_weights():
    from board_scan import available_model
    name, m = available_model()
    assert name in ("resnet18_imagenet", "small_cnn_scratch")


@pytest.mark.skipif(not (ROOT / "downloads" / "DeepPCB").exists(), reason="run src/prepare_patches.py")
def test_app_renders_a_board():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120).run()
    assert not at.exception and at.metric[0].label == "Windows over threshold"
