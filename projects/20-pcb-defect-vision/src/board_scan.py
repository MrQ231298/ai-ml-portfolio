"""Sliding-window scan of a whole board image with a trained patch classifier."""

from __future__ import annotations

import numpy as np
import torch

from vision_lib import CLASSES, INPUT_SIZE, ROOT, SmallCNN, make_resnet, to_tensor

P = 64


def available_model() -> tuple[str, torch.nn.Module]:
    """ResNet-18 if its weights exist (not committed: 45 MB), else the committed small CNN."""
    for name in ("resnet18_imagenet", "small_cnn_scratch"):
        f = ROOT / "models" / f"{name}.pt"
        if f.exists():
            m = make_resnet(False) if name == "resnet18_imagenet" else SmallCNN()
            m.load_state_dict(torch.load(f, map_location="cpu"))
            return name, m.eval()
    raise FileNotFoundError("no model in models/: run src/train.py")


def scan(img: np.ndarray, model: torch.nn.Module, name: str, stride: int = 32) -> dict:
    """img: uint8 greyscale (H, W) >= 64x64. Returns per-window class probabilities and a defect heatmap."""
    h, w = img.shape
    ys = list(range(0, h - P + 1, stride)) or [0]
    xs = list(range(0, w - P + 1, stride)) or [0]
    if ys[-1] != h - P:
        ys.append(h - P)
    if xs[-1] != w - P:
        xs.append(w - P)
    coords = [(y, x) for y in ys for x in xs]
    patches = np.stack([img[y:y + P, x:x + P] for y, x in coords])
    with torch.no_grad():
        probs = torch.cat([torch.softmax(model(to_tensor(patches[i:i + 256], INPUT_SIZE[name])), 1)
                           for i in range(0, len(patches), 256)]).numpy()
    heat, cnt = np.zeros((h, w), np.float32), np.zeros((h, w), np.float32)
    for (y, x), p in zip(coords, probs):
        heat[y:y + P, x:x + P] += 1 - p[0]
        cnt[y:y + P, x:x + P] += 1
    windows = [{"y": y, "x": x, "defect_prob": float(1 - p[0]), "class": CLASSES[int(p.argmax())]}
               for (y, x), p in zip(coords, probs)]
    return {"heatmap": heat / np.maximum(cnt, 1), "windows": windows}
