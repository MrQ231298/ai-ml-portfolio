"""Train the from-scratch CNN and the ImageNet-pretrained ResNet-18 on DeepPCB patches.

Model selection uses the validation board groups only; the official test split
is scored once at the end by evaluate.py.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

from vision_lib import INPUT_SIZE, ROOT, SmallCNN, augment, load_split, make_resnet, to_tensor


def predict(model, X, bs=256):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            out.append(torch.softmax(model(X[i:i + bs]), 1))
    return torch.cat(out).numpy()


def train(name, model, Xtr, ytr, Xva, yva, epochs, lr, bs=128, seed=20):
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * ((len(Xtr) + bs - 1) // bs))
    # class weights: inverse frequency, so 'ok' (most common) does not dominate
    counts = torch.bincount(ytr, minlength=7).float()
    loss_fn = nn.CrossEntropyLoss(weight=(counts.sum() / (7 * counts)))
    hist, best, best_state = [], -1, None
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr), generator=g)
        tot = 0.0
        for i in range(0, len(Xtr), bs):
            idx = perm[i:i + bs]
            xb = torch.stack([augment(x, g) for x in Xtr[idx]])
            loss = loss_fn(model(xb), ytr[idx])
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot += loss.item() * len(idx)
        pv = predict(model, Xva).argmax(1)
        f1 = f1_score(yva.numpy(), pv, average="macro")
        hist.append({"epoch": ep + 1, "train_loss": round(tot / len(Xtr), 4), "val_macro_f1": round(f1, 4),
                     "elapsed_s": round(time.time() - t0, 1)})
        print(name, hist[-1], flush=True)
        if f1 > best:
            best, best_state = f1, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, {"history": hist, "best_val_macro_f1": round(best, 4), "train_seconds": round(time.time() - t0, 1),
                   "params": int(sum(p.numel() for p in model.parameters()))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=6)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    tr, va = load_split("train"), load_split("val")
    ytr, yva = torch.from_numpy(tr["label"]).long(), torch.from_numpy(va["label"]).long()
    log = {}
    (ROOT / "models").mkdir(exist_ok=True)
    for name, model, lr in (("small_cnn_scratch", SmallCNN(), 3e-3), ("resnet18_imagenet", make_resnet(True), 1e-3)):
        size = INPUT_SIZE[name]
        Xtr, Xva = to_tensor(tr["test"], size), to_tensor(va["test"], size)
        model, info = train(name, model, Xtr, ytr, Xva, yva, a.epochs, lr)
        info["input_size"] = size
        torch.save(model.state_dict(), ROOT / "models" / f"{name}.pt")
        log[name] = info
    log["threads"] = a.threads
    (ROOT / "reports" / "training_log.json").write_text(json.dumps(log, indent=2))


if __name__ == "__main__":
    main()
