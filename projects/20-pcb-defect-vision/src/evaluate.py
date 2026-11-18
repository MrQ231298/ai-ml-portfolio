"""Score everything once on the official DeepPCB test split.

1. Classifiers (from train.py): per-class precision/recall/F1, confusion matrix.
2. Golden-template baseline: pixel difference between the tested patch and the
   aligned defect-free template. No learning; threshold chosen on validation.
3. PatchCore-lite (Roth et al., 2022, simplified): anomaly detection that sees
   ONLY defect-free training patches. Frozen ImageNet ResNet-18 features from
   layer2+layer3, a random-subsampled memory bank, score = max nearest-neighbour
   distance over locations. Threshold chosen on validation.
4. Grad-CAM grid for the ResNet-18 classifier.
"""

from __future__ import annotations

import json
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from scipy.ndimage import uniform_filter  # noqa: E402
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,  # noqa: E402
                             precision_recall_fscore_support, roc_auc_score)

from train import predict  # noqa: E402
from vision_lib import CLASSES, INPUT_SIZE, ROOT, SmallCNN, gradcam, load_split, make_resnet, to_tensor  # noqa: E402

REPORTS = ROOT / "reports"
SEED = 20


def load_model(name):
    m = SmallCNN() if name == "small_cnn_scratch" else make_resnet(False)
    m.load_state_dict(torch.load(ROOT / "models" / f"{name}.pt", map_location="cpu"))
    return m.eval()


def binary_at(scores_val, y_val, scores_test, y_test):
    """Pick the threshold that maximises F1 (defect = positive) on validation, apply it to test."""
    cands = np.unique(np.quantile(scores_val, np.linspace(0, 1, 401)))
    f1s = [f1_score(y_val, scores_val >= t) for t in cands]
    t = float(cands[int(np.argmax(f1s))])
    pred = scores_test >= t
    tp = int((pred & y_test).sum()); fp = int((pred & ~y_test).sum()); fn = int((~pred & y_test).sum())
    return {"threshold": round(t, 6), "auroc": round(roc_auc_score(y_test, scores_test), 4),
            "precision": round(tp / max(tp + fp, 1), 4), "recall": round(tp / max(tp + fn, 1), 4),
            "false_alarm_rate": round(fp / max(int((~y_test).sum()), 1), 4),
            "f1": round(f1_score(y_test, pred), 4), "missed_defects": fn, "false_alarms": fp}


TEMPLATE_GRID = [(k, t) for k in (1, 3, 5) for t in (40, 60, 100, 150)]


def template_scores(split, k=3, t=60):
    """Fraction of pixels whose k x k-smoothed absolute difference from the template exceeds t grey levels."""
    a = uniform_filter(split["test"].astype(np.float32), size=(1, k, k))
    b = uniform_filter(split["template"].astype(np.float32), size=(1, k, k))
    return (np.abs(a - b) > t).mean((1, 2))


def template_baseline(tr, va, te):
    """Pick (smoothing, grey-level threshold) by AUROC on the TRAIN split (no learning, 10 board groups),
    then the decision threshold on validation like every other method."""
    grid = {f"k{k}_t{t}": round(roc_auc_score(tr["label"] > 0, template_scores(tr, k, t)), 4) for k, t in TEMPLATE_GRID}
    k, t = TEMPLATE_GRID[int(np.argmax(list(grid.values())))]
    sv, st_ = template_scores(va, k, t), template_scores(te, k, t)
    out = binary_at(sv, va["label"] > 0, st_, te["label"] > 0)
    groups = np.array([x.split("/")[0] for x in te["source"]])
    ok = te["label"] == 0
    fa = {g: round(float((st_[ok & (groups == g)] >= out["threshold"]).mean()), 4) for g in sorted(set(groups))}
    return {**out, "smoothing": k, "grey_threshold": t, "val_auroc": round(roc_auc_score(va["label"] > 0, sv), 4),
            "test_false_alarm_rate_by_group": fa, "train_auroc_grid": grid}


class FeatureExtractor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.m = make_resnet(True).eval()

    @torch.no_grad()
    def forward(self, x):
        m = self.m
        x = m.maxpool(m.relu(m.bn1(m.conv1(x))))
        l2 = m.layer2(m.layer1(x))
        l3 = m.layer3(l2)
        l2 = F.avg_pool2d(l2, 3, stride=2, padding=1)   # 12x12 -> 6x6, with local context
        l3 = F.avg_pool2d(l3, 3, stride=1, padding=1)
        f = torch.cat([l2, l3], 1)                      # (N, 384, 6, 6)
        return f.permute(0, 2, 3, 1).reshape(len(f), -1, f.shape[1])


def embed(fx, X, bs=256):
    return torch.cat([fx(to_tensor(X[i:i + bs], 96)) for i in range(0, len(X), bs)])


def patchcore(train, val, test, bank_size=10000):
    t0 = time.time()
    fx = FeatureExtractor()
    ok = train["test"][train["label"] == 0]
    bank = embed(fx, ok).reshape(-1, 384)
    g = torch.Generator().manual_seed(SEED)
    bank = bank[torch.randperm(len(bank), generator=g)[:bank_size]]

    def score(X):
        e = embed(fx, X)
        out = []
        for i in range(0, len(e), 128):
            q = e[i:i + 128]
            d = torch.cdist(q.reshape(-1, 384), bank).min(1).values.reshape(len(q), -1)
            out.append(d.max(1).values)
        return torch.cat(out).numpy()

    return score(val["test"]), score(test["test"]), {"memory_bank": int(len(bank)), "ok_train_patches": int(len(ok)),
                                                       "seconds": round(time.time() - t0, 1)}


def classifier_report(y, p):
    pred = p.argmax(1)
    pr, rc, f1, sup = precision_recall_fscore_support(y, pred, labels=range(len(CLASSES)), zero_division=0)
    return {"accuracy": round(accuracy_score(y, pred), 4), "macro_f1": round(f1_score(y, pred, average="macro"), 4),
            "per_class": {c: {"precision": round(pr[i], 4), "recall": round(rc[i], 4), "f1": round(f1[i], 4),
                              "support": int(sup[i])} for i, c in enumerate(CLASSES)},
            "confusion_matrix": confusion_matrix(y, pred, labels=range(len(CLASSES))).tolist()}


def plot_confusion(cm, title, path):
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm / cm.sum(1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
    for i in range(len(cm)):
        for j in range(len(cm)):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8, color="white" if i == j else "black")
    ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=45, ha="right"); ax.set_yticks(range(len(CLASSES)), CLASSES)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(title)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def gradcam_grid(model, test, path, per_class=2):
    rng = np.random.default_rng(SEED)
    picks = [i for c in range(1, len(CLASSES)) for i in rng.choice(np.where(test["label"] == c)[0], per_class, replace=False)]
    fig, axes = plt.subplots(2, len(picks) // 2, figsize=(2 * len(picks) // 2, 4.4))
    for ax, i in zip(axes.T.ravel(), picks):
        x = to_tensor(test["test"][i:i + 1], 96)
        cam, cls = gradcam(model, x)
        ax.imshow(F.interpolate(torch.from_numpy(test["test"][i:i + 1]).float()[None], size=(96, 96))[0, 0], cmap="gray")
        ax.imshow(cam, cmap="jet", alpha=0.4)
        ok = "" if cls == test["label"][i] else " (wrong)"
        ax.set_title(f"{CLASSES[test['label'][i]]}\n-> {CLASSES[cls]}{ok}", fontsize=8); ax.axis("off")
    fig.suptitle("Grad-CAM, ResNet-18 (layer3), DeepPCB test patches: true -> predicted", fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def main():
    torch.set_num_threads(4)
    tr, va, te = load_split("train"), load_split("val"), load_split("test")
    yv, yt = va["label"] > 0, te["label"] > 0
    res = {"test_counts": {c: int((te["label"] == i).sum()) for i, c in enumerate(CLASSES)},
           "training": json.loads((REPORTS / "training_log.json").read_text())}
    res["classifiers"] = {}
    groups = np.array([x.split("/")[0] for x in te["source"]])
    trainval_groups = {x.split("/")[0] for x in np.concatenate([tr["source"], va["source"]])}
    for name in ("small_cnn_scratch", "resnet18_imagenet"):
        m = load_model(name)
        size = INPUT_SIZE[name]
        t0 = time.time()
        pt = predict(m, to_tensor(te["test"], size))
        ms = (time.time() - t0) / len(pt) * 1000
        pv = predict(m, to_tensor(va["test"], size))
        rep = classifier_report(te["label"], pt)
        rep["defect_vs_ok"] = binary_at(1 - pv[:, 0], yv, 1 - pt[:, 0], yt)
        rep["cpu_ms_per_patch_batched"] = round(ms, 3)
        seen = np.isin(groups, sorted(trainval_groups))
        rep["by_board_group_seen_in_training"] = {
            "seen": {"patches": int(seen.sum()), "macro_f1": round(f1_score(te["label"][seen], pt[seen].argmax(1), average="macro"), 4)},
            "unseen": {"patches": int((~seen).sum()), "macro_f1": round(f1_score(te["label"][~seen], pt[~seen].argmax(1), average="macro"), 4)}}
        res["classifiers"][name] = rep
        plot_confusion(rep["confusion_matrix"], f"{name}: test confusion matrix", REPORTS / f"confusion_{name}.png")
        print(name, rep["accuracy"], rep["macro_f1"], rep["defect_vs_ok"], flush=True)
    res["golden_template_baseline"] = template_baseline(tr, va, te)
    print("template", {k: v for k, v in res["golden_template_baseline"].items() if k != "train_auroc_grid"}, flush=True)
    sv, st_, info = patchcore(tr, va, te)
    res["patchcore_lite"] = {**binary_at(sv, yv, st_, yt), **info}
    print("patchcore", res["patchcore_lite"], flush=True)
    res["test_board_groups"] = sorted(set(groups))
    res["trainval_board_groups"] = sorted(trainval_groups)
    res["test_groups_unseen_in_trainval"] = sorted(set(groups) - trainval_groups)
    gradcam_grid(load_model("resnet18_imagenet"), te, REPORTS / "gradcam_grid.png")
    (REPORTS / "metrics.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
