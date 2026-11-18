"""Shared model, data and Grad-CAM code."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18

ROOT = Path(__file__).resolve().parents[1]
PATCHES = ROOT / "data" / "patches"
CLASSES = ["ok", "open", "short", "mousebite", "spur", "copper", "pin-hole"]
MEAN, STD = 0.449, 0.226  # ImageNet grey-level mean / std
INPUT = 96
# CPU budget: the scratch CNN sees native 64 px patches, ResNet-18 sees them upscaled to 96 px
INPUT_SIZE = {"small_cnn_scratch": 64, "resnet18_imagenet": 96}


def load_split(name: str) -> dict:
    d = np.load(PATCHES / f"{name}.npz")
    return {k: d[k] for k in d.files}


def to_tensor(x: np.ndarray, size: int = INPUT) -> torch.Tensor:
    """uint8 (N, 64, 64) -> float (N, 3, size, size), normalised."""
    t = torch.from_numpy(x).float().div(255.0).unsqueeze(1)
    if size != t.shape[-1]:
        t = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)
    t = (t - MEAN) / STD
    return t.repeat(1, 3, 1, 1)


def augment(t: torch.Tensor, g: torch.Generator) -> torch.Tensor:
    """Random flips and 90-degree rotations (PCB defects have no preferred orientation)."""
    if torch.rand(1, generator=g) < 0.5:
        t = t.flip(-1)
    if torch.rand(1, generator=g) < 0.5:
        t = t.flip(-2)
    k = int(torch.randint(0, 4, (1,), generator=g))
    return torch.rot90(t, k, dims=(-2, -1))


class SmallCNN(nn.Module):
    """From-scratch baseline: four conv blocks, about 0.19 M parameters."""

    def __init__(self, n_classes=len(CLASSES)):
        super().__init__()
        chans = [3, 32, 64, 96, 128]
        layers = []
        for a, b in zip(chans, chans[1:]):
            layers += [nn.Conv2d(a, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU(), nn.MaxPool2d(2)]
        self.features = nn.Sequential(*layers)
        self.head = nn.Linear(128, n_classes)

    def forward(self, x):
        return self.head(self.features(x).mean((-2, -1)))


def make_resnet(pretrained: bool = True, n_classes=len(CLASSES)) -> nn.Module:
    m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    m.fc = nn.Linear(m.fc.in_features, n_classes)
    return m


def gradcam(model: nn.Module, x: torch.Tensor, target: int | None = None, layer: str = "layer3") -> tuple[np.ndarray, int]:
    """Grad-CAM (Selvaraju et al., 2017) for one input of shape (1, 3, H, W)."""
    acts, grads = {}, {}
    mod = dict(model.named_modules())[layer]
    h1 = mod.register_forward_hook(lambda m, i, o: acts.__setitem__("a", o))
    h2 = mod.register_full_backward_hook(lambda m, gi, go: grads.__setitem__("g", go[0]))
    model.eval()
    x = x.clone().requires_grad_(True)
    out = model(x)
    cls = int(out.argmax(1)) if target is None else target
    model.zero_grad()
    out[0, cls].backward()
    h1.remove(); h2.remove()
    w = grads["g"].mean((-2, -1), keepdim=True)
    cam = F.relu((w * acts["a"]).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    cam = cam / (cam.max() + 1e-8)
    return cam.detach().numpy(), cls
