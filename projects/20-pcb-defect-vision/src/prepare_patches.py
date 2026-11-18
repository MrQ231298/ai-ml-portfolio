"""Fetch DeepPCB (MIT licence) at a pinned commit and cut 64x64 patches.

For every annotated defect a patch centred on its box is cut from the TESTED
image (class = defect type) together with the same window from the aligned
TEMPLATE image (used by the golden-template baseline). Defect-free ("ok")
patches are random windows that do not touch any defect box. The official
DeepPCB split is kept: trainval.txt -> train/val (by board group), test.txt -> test.
"""

from __future__ import annotations

import json
import random
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "downloads" / "DeepPCB"
COMMIT = "08e98c4db5922613fb97176eb3d6497d48260cb1"
OUT = ROOT / "data" / "patches"
CLASSES = ["ok", "open", "short", "mousebite", "spur", "copper", "pin-hole"]  # DeepPCB ids 1..6 -> 1..6
P = 64
OK_PER_IMAGE = 3
SEED = 20


def fetch():
    if not REPO.exists():
        REPO.parent.mkdir(exist_ok=True)
        subprocess.run(["git", "clone", "-q", "https://github.com/tangsanli5201/DeepPCB.git", str(REPO)], check=True)
    subprocess.run(["git", "-C", str(REPO), "checkout", "-q", COMMIT], check=True)


def crop(img: np.ndarray, cx: int, cy: int) -> np.ndarray:
    h, w = img.shape
    x0 = int(np.clip(cx - P // 2, 0, w - P))
    y0 = int(np.clip(cy - P // 2, 0, h - P))
    return img[y0:y0 + P, x0:x0 + P]


def parse_list(name):
    rows = []
    for line in (REPO / "PCBData" / name).read_text().split("\n"):
        if line.strip():
            img, ann = line.split()
            rows.append((img, ann))
    return rows


def build():
    fetch()
    rng = random.Random(SEED)
    split_rows = {"trainval": parse_list("trainval.txt"), "test": parse_list("test.txt")}
    groups = sorted({r[0].split("/")[0] for r in split_rows["trainval"]})
    val_groups = set(groups[-2:])  # last two board groups of trainval become validation
    meta = {"source": "https://github.com/tangsanli5201/DeepPCB", "commit": COMMIT, "licence": "MIT",
            "citation": "Tang, S., He, F., Huang, X., Yang, J. (2019). Online PCB Defect Detector On A New PCB Defect Dataset. arXiv:1902.06197",
            "patch": P, "classes": CLASSES, "val_groups": sorted(val_groups), "counts": {}}
    for split, rows in split_rows.items():
        tests, temps, labels, info = [], [], [], []
        for img_rel, ann_rel in rows:
            base = (REPO / "PCBData" / img_rel).with_suffix("")
            test = np.array(Image.open(f"{base}_test.jpg").convert("L"))
            temp = np.array(Image.open(f"{base}_temp.jpg").convert("L"))
            boxes = []
            for line in (REPO / "PCBData" / ann_rel).read_text().split("\n"):
                parts = line.replace(",", " ").split()
                if len(parts) == 5:
                    x1, y1, x2, y2, c = map(int, parts)
                    boxes.append((x1, y1, x2, y2, c))
            group = img_rel.split("/")[0]
            sub = "test" if split == "test" else ("val" if group in val_groups else "train")
            for x1, y1, x2, y2, c in boxes:
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                tests.append(crop(test, cx, cy)); temps.append(crop(temp, cx, cy)); labels.append(c)
                info.append((sub, img_rel, cx, cy))
            h, w = test.shape
            made, tries = 0, 0
            while made < OK_PER_IMAGE and tries < 200:
                tries += 1
                cx, cy = rng.randint(P // 2, w - P // 2), rng.randint(P // 2, h - P // 2)
                if any(cx + P // 2 > x1 - 4 and cx - P // 2 < x2 + 4 and cy + P // 2 > y1 - 4 and cy - P // 2 < y2 + 4
                       for x1, y1, x2, y2, _ in boxes):
                    continue
                tests.append(crop(test, cx, cy)); temps.append(crop(temp, cx, cy)); labels.append(0)
                info.append((sub, img_rel, cx, cy)); made += 1
        subs = np.array([i[0] for i in info])
        for sub in sorted(set(subs)):
            m = subs == sub
            OUT.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(OUT / f"{sub}.npz", test=np.stack(tests)[m], template=np.stack(temps)[m],
                                label=np.array(labels)[m],
                                source=np.array([f"{i[1]}@{i[2]},{i[3]}" for i, k in zip(info, m) if k]))
            meta["counts"][sub] = {CLASSES[c]: int((np.array(labels)[m] == c).sum()) for c in range(len(CLASSES))}
    (ROOT / "data" / "patch_metadata.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta["counts"], indent=1))


if __name__ == "__main__":
    build()
