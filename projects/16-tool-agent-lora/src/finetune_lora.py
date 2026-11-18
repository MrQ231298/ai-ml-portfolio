"""LoRA fine-tuning of SmolLM2-360M-Instruct to emit tool plans, on CPU.

Loss is computed on the assistant (plan) tokens only. 10% of the training
tasks are held back as a validation split for the loss curve; the test sets
are never touched here.
"""

from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from agent import ADAPTER_DIR, BASE_MODEL, plan_messages, target_text

ROOT = Path(__file__).resolve().parents[1]
CFG = {"r": 16, "alpha": 32, "dropout": 0.05, "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
       "lr": 3e-4, "epochs": 2, "batch_size": 4, "grad_accum": 2, "warmup_steps": 10, "seed": 16,
       "val_fraction": 0.1, "threads": 8}


def encode(tok, task):
    msgs = plan_messages(task["question"], few_shot=False)
    prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    full = prompt + target_text(task["plan"]) + tok.eos_token
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    ids = tok(full, add_special_tokens=False)["input_ids"]
    labels = [-100] * len(p_ids) + ids[len(p_ids):]
    return ids, labels


def batches(rows, bs, pad):
    for i in range(0, len(rows), bs):
        chunk = rows[i:i + bs]
        n = max(len(r[0]) for r in chunk)
        ids = torch.tensor([r[0] + [pad] * (n - len(r[0])) for r in chunk])
        lab = torch.tensor([r[1] + [-100] * (n - len(r[1])) for r in chunk])
        att = torch.tensor([[1] * len(r[0]) + [0] * (n - len(r[0])) for r in chunk])
        yield ids, att, lab


def evaluate_loss(model, rows, pad):
    model.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for ids, att, lab in batches(rows, 8, pad):
            out = model(input_ids=ids, attention_mask=att, labels=lab)
            k = int((lab[:, 1:] != -100).sum())
            tot += out.loss.item() * k
            n += k
    model.train()
    return tot / n


def main():
    torch.manual_seed(CFG["seed"])
    torch.set_num_threads(CFG["threads"])
    rng = random.Random(CFG["seed"])
    tasks = [json.loads(l) for l in open(ROOT / "data" / "tasks_train.jsonl", encoding="utf-8")]
    rng.shuffle(tasks)
    n_val = int(len(tasks) * CFG["val_fraction"])
    val_t, tr_t = tasks[:n_val], tasks[n_val:]
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    tr = [encode(tok, t) for t in tr_t]
    va = [encode(tok, t) for t in val_t]
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.float32)
    model = get_peft_model(model, LoraConfig(r=CFG["r"], lora_alpha=CFG["alpha"], lora_dropout=CFG["dropout"],
                                             target_modules=CFG["target_modules"], task_type="CAUSAL_LM"))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=CFG["lr"], weight_decay=0.0)
    steps_per_epoch = math.ceil(len(tr) / (CFG["batch_size"] * CFG["grad_accum"]))
    total_steps = steps_per_epoch * CFG["epochs"]
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / CFG["warmup_steps"]) * max(0.0, (total_steps - s) / total_steps))
    log = {"val_loss_before": evaluate_loss(model, va, pad), "curve": []}
    print("val loss before", log["val_loss_before"], "trainable", trainable, "of", total)
    t0 = time.time()
    step = 0
    model.train()
    for ep in range(CFG["epochs"]):
        rng.shuffle(tr)
        tr_sorted = tr  # random order; padding cost is acceptable at this size
        micro = 0
        running = []
        for ids, att, lab in batches(tr_sorted, CFG["batch_size"], pad):
            loss = model(input_ids=ids, attention_mask=att, labels=lab).loss / CFG["grad_accum"]
            loss.backward()
            running.append(loss.item() * CFG["grad_accum"])
            micro += 1
            if micro % CFG["grad_accum"] == 0:
                opt.step(); sched.step(); opt.zero_grad(); step += 1
                if step % 10 == 0:
                    print(f"epoch {ep} step {step}/{total_steps} train_loss {sum(running[-20:]) / len(running[-20:]):.4f} {time.time() - t0:.0f}s", flush=True)
        if micro % CFG["grad_accum"]:
            opt.step(); sched.step(); opt.zero_grad(); step += 1
        vl = evaluate_loss(model, va, pad)
        log["curve"].append({"epoch": ep + 1, "step": step, "train_loss_last50": sum(running[-50:]) / len(running[-50:]),
                             "val_loss": vl, "elapsed_s": round(time.time() - t0, 1)})
        print("epoch", ep + 1, "val loss", vl, flush=True)
    log.update({"config": CFG, "base_model": BASE_MODEL, "train_examples": len(tr), "val_examples": len(va),
                "trainable_params": trainable, "total_params": total, "optimizer_steps": step,
                "train_seconds": round(time.time() - t0, 1), "device": "cpu",
                "cpu_threads": CFG["threads"]})
    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(ADAPTER_DIR))
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "finetune_log.json").write_text(json.dumps(log, indent=2))
    print(json.dumps({k: v for k, v in log.items() if k != "curve"}, indent=2))


if __name__ == "__main__":
    main()
