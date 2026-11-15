"""
Build text for every SYNTHETIC invoice (digital + simulated scan/OCR),
train the candidate-ranking model on development layouts, and evaluate
three extractors field by field.

Splits
------
- dev-train : 70% of layouts A, B, C (stratified by layout, seed 42)
- dev-test  : the other 30% of A, B, C
- unseen-D  : all 60 invoices in held-out layout D (never used for
              rules or training)
Each test split is scored on digital PDF text and on OCR text from a
simulated scan.

Extractors: rules, ml (candidate ranking), hybrid (rules, falling back
to ML when a rule finds nothing).

    python src/evaluate.py            # uses cached OCR text if present
    python src/evaluate.py --re-ocr   # re-run OCR (about 2 s per page)
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_lib import (  # noqa: E402
    FIELDS, CandidateModel, RuleExtractor, arithmetic_ok, norm_value, ocr_text, pdf_text,
)

ROOT = Path(__file__).resolve().parents[1]
PDF_DIR = ROOT / "data" / "invoices_pdf"
OCR_DIR = ROOT / "data" / "ocr_text"
REPORTS = ROOT / "reports"


def hybrid(rule_out: dict, ml_out: dict) -> dict:
    return {f: rule_out[f] if rule_out[f] not in (None, "") else ml_out[f] for f in FIELDS}


def score(preds: list[dict], truths: pd.DataFrame) -> dict:
    per_field = {}
    correct = np.zeros((len(preds), len(FIELDS)), dtype=bool)
    for j, f in enumerate(FIELDS):
        for i, (p, (_, t)) in enumerate(zip(preds, truths.iterrows())):
            correct[i, j] = norm_value(f, p[f]) == norm_value(f, t[f])
        per_field[f] = round(float(correct[:, j].mean()), 4)
    arith = np.array([arithmetic_ok(p) for p in preds])
    all_ok = correct.all(axis=1)
    return {
        "n_docs": len(preds),
        "field_accuracy": per_field,
        "mean_field_accuracy": round(float(correct.mean()), 4),
        "all_fields_correct_rate": round(float(all_ok.mean()), 4),
        "arithmetic_check_pass_rate": round(float(arith.mean()), 4),
        "all_correct_given_check_pass": round(float(all_ok[arith].mean()), 4) if arith.any() else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--re-ocr", action="store_true"); args = ap.parse_args()
    REPORTS.mkdir(exist_ok=True); OCR_DIR.mkdir(exist_ok=True)
    truth = pd.read_csv(ROOT / "data" / "SYNTHETIC_invoice_ground_truth.csv")
    assert (truth.source_flag == "SYNTHETIC_EXAMPLE_DATA").all()

    dev = truth[truth.template != "D"]
    tr, te = train_test_split(dev, test_size=0.3, stratify=dev.template, random_state=42)
    splits = {"dev_test": te, "unseen_D": truth[truth.template == "D"]}
    print(f"train={len(tr)} dev_test={len(te)} unseen_D={len(splits['unseen_D'])}")

    digital = {f: pdf_text(PDF_DIR / f) for f in truth.file}

    def get_ocr(fname: str, i: int) -> str:
        cache = OCR_DIR / (fname.replace(".pdf", ".txt"))
        if cache.exists() and not args.re_ocr:
            return cache.read_text()
        txt = ocr_text(PDF_DIR / fname, seed=i)
        cache.write_text(txt)
        return txt

    ocr = {}
    for i, f in enumerate(pd.concat(splits.values()).file):
        ocr[f] = get_ocr(f, i)
    print("text ready")

    rules = RuleExtractor()
    ml = CandidateModel().fit([digital[f] for f in tr.file], tr[FIELDS].to_dict("records"))
    joblib.dump(ml, ROOT / "models" / "candidate_ranker.joblib")

    results, extracted = {}, []
    for split, df in splits.items():
        for source, texts in (("digital", digital), ("ocr", ocr)):
            outs = {"rules": [], "ml": [], "hybrid": []}
            for f in df.file:
                r, m = rules.extract(texts[f]), ml.extract(texts[f])
                outs["rules"].append(r); outs["ml"].append(m); outs["hybrid"].append(hybrid(r, m))
            for name, preds in outs.items():
                key = f"{split}/{source}/{name}"
                results[key] = score(preds, df)
                print(key, results[key]["mean_field_accuracy"], results[key]["all_fields_correct_rate"])
                for fname, p in zip(df.file, preds):
                    extracted.append({"split": split, "source": source, "extractor": name, "file": fname, **p})
    # ---- few-shot adaptation: label 5 invoices of the new layout D, retrain ----
    d_all = splits["unseen_D"]
    d_few, d_rest = d_all.iloc[:5], d_all.iloc[5:]
    ml_adapt = CandidateModel().fit([digital[f] for f in pd.concat([tr, d_few]).file],
                                    pd.concat([tr, d_few])[FIELDS].to_dict("records"))
    for source, texts in (("digital", digital), ("ocr", ocr)):
        before = [ml.extract(texts[f]) for f in d_rest.file]
        after = [ml_adapt.extract(texts[f]) for f in d_rest.file]
        results[f"unseen_D_rest55/{source}/ml_before_adaptation"] = score(before, d_rest)
        results[f"unseen_D_rest55/{source}/ml_after_5_examples"] = score(after, d_rest)
        print(source, "adapt", results[f"unseen_D_rest55/{source}/ml_before_adaptation"]["mean_field_accuracy"],
              "->", results[f"unseen_D_rest55/{source}/ml_after_5_examples"]["mean_field_accuracy"])
    pd.DataFrame(extracted).to_csv(REPORTS / "extracted_fields.csv", index=False)

    # Example structured output a client would receive (hybrid, digital, unseen layout)
    ex = pd.DataFrame([e for e in extracted if e["split"] == "unseen_D" and e["source"] == "digital" and e["extractor"] == "hybrid"])
    ex["arithmetic_check"] = ex.apply(lambda r: "pass" if arithmetic_ok(r) else "REVIEW", axis=1)
    ex.drop(columns=["split", "source", "extractor"]).to_csv(REPORTS / "sample_output_unseen_D_hybrid.csv", index=False)

    # Plot field accuracy for the hybrid extractor across the four conditions
    fig, ax = plt.subplots(figsize=(10, 4.5))
    conds = [k for k in results if k.endswith("/hybrid") and k.startswith(("dev_test", "unseen_D/"))]
    w = 0.8 / len(conds)
    for i, k in enumerate(conds):
        ax.bar(np.arange(len(FIELDS)) + i * w, [results[k]["field_accuracy"][f] for f in FIELDS], w,
               label=k.replace("/hybrid", ""))
    ax.set_xticks(np.arange(len(FIELDS)) + 0.4 - w / 2); ax.set_xticklabels(FIELDS, rotation=25, ha="right")
    ax.set_ylim(0, 1.05); ax.set_ylabel("exact-match accuracy"); ax.legend(fontsize=8)
    ax.set_title("Hybrid extractor: field accuracy by split and text source (SYNTHETIC invoices)")
    fig.tight_layout(); fig.savefig(REPORTS / "field_accuracy_hybrid.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    names = ["rules", "ml", "hybrid"]
    keys = ["dev_test/digital", "dev_test/ocr", "unseen_D/digital", "unseen_D/ocr"]
    for i, n in enumerate(names):
        ax.bar(np.arange(4) + i * 0.27, [results[f"{k}/{n}"]["mean_field_accuracy"] for k in keys], 0.27, label=n)
    ax.set_xticks(np.arange(4) + 0.27); ax.set_xticklabels(keys)
    ax.set_ylim(0, 1.05); ax.set_ylabel("mean field accuracy"); ax.legend()
    ax.set_title("Rules vs ML vs hybrid (8 fields)")
    fig.tight_layout(); fig.savefig(REPORTS / "extractor_comparison.png", dpi=130); plt.close(fig)

    metrics = {
        "run_at_europe_dublin": datetime.now(ZoneInfo("Europe/Dublin")).isoformat(timespec="seconds"),
        "data": {"invoices": int(len(truth)), "layouts": sorted(truth.template.unique().tolist()),
                 "train_docs": int(len(tr)), "dev_test_docs": int(len(te)), "unseen_D_docs": int((truth.template == "D").sum()),
                 "status": "SYNTHETIC_EXAMPLE_DATA"},
        "ocr_engine": "RapidOCR (rapidocr_onnxruntime 1.2.3) on 150 dpi renders with rotation, blur and noise",
        "results": results,
    }
    (REPORTS / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print("wrote", REPORTS / "metrics.json")


if __name__ == "__main__":
    main()
