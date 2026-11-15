"""
Invoice text extraction and field extraction (rule-based and ML).

Text sources
------------
- digital PDF  -> `pdftotext -layout` (poppler), falling back to pypdf
- scanned page -> rasterise with pdftoppm, add scan noise, OCR with RapidOCR
                  (PP-OCR models via ONNX Runtime; Tesseract could not be
                  installed on the build machine)

Field extractors
----------------
- RuleExtractor: label + regex rules written while looking at the three
  development layouts (A, B, C) only.
- CandidateModel: generates every plausible candidate per field (all money
  amounts, all dates, all ID-like tokens, top-of-page text segments) and
  ranks them with a logistic regression over layout-independent features
  (words to the left, position on the page, amount rank, whether an amount
  equals the sum of two others, VAT-rate ratios, date order).
"""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

import numpy as np

FIELDS = ["vendor_name", "vendor_vat_number", "invoice_number", "invoice_date",
          "due_date", "net_amount", "vat_amount", "total_amount"]
MONEY_FIELDS = ["net_amount", "vat_amount", "total_amount"]
DATE_FIELDS = ["invoice_date", "due_date"]

MONEY_RE = re.compile(r"(?<![\d.])(\d{1,3}(?:,\d{3})*\.\d{2})(?![\d])")
# No leading \b: OCR often glues the label to the number ("numberIE2004976PA").
# (Relaxed after the first error review of OCR output - see README.)
VATNO_RE = re.compile(r"(?<![A-Z0-9])IE\s?\d{7}[A-Z]{1,2}(?![A-Za-z0-9])")
# Spaces are optional because OCR often drops them ("24Sep2025").
DATE_PATTERNS = [
    (re.compile(r"(?<!\d)\d{1,2}/\d{1,2}/\d{4}(?!\d)"), ["%d/%m/%Y"]),
    (re.compile(r"(?<!\d)\d{4}-\d{2}-\d{2}(?!\d)"), ["%Y-%m-%d"]),
    (re.compile(r"(?<!\d)\d{1,2} ?[A-Z][a-z]{2} ?\d{4}(?!\d)"), ["%d%b%Y"]),
    (re.compile(r"[A-Z][a-z]{2,8} ?\d{1,2}, ?\d{4}(?!\d)"), ["%B%d,%Y"]),
]
ID_RE = re.compile(r"(?<![\w/-])([A-Z0-9][A-Z0-9/-]{3,}[0-9])(?![\w/-])")
COMPANY_RE = re.compile(r"(Ltd|Limited|Teoranta|DAC)\b\.?", re.I)


# ------------------------------------------------------------------ text
def pdf_text(pdf: Path) -> str:
    if shutil.which("pdftotext"):
        return subprocess.run(["pdftotext", "-layout", str(pdf), "-"],
                              capture_output=True, text=True, check=True).stdout
    from pypdf import PdfReader
    return "\n".join(p.extract_text() or "" for p in PdfReader(str(pdf)).pages)


_OCR_ENGINE = None


def _engine():
    global _OCR_ENGINE
    if _OCR_ENGINE is None:
        from rapidocr_onnxruntime import RapidOCR  # pip-installable OCR (PP-OCR models, ONNX)
        _OCR_ENGINE = RapidOCR()
    return _OCR_ENGINE


def boxes_to_layout_text(result) -> str:
    """Rebuild reading-order lines from OCR word boxes.

    Boxes whose vertical centres are within half a text height are put on
    the same line; boxes on a line are joined with three spaces (the same
    column-gap convention `pdftotext -layout` produces), so the downstream
    extractors see a similar structure for scanned and digital invoices.
    """
    items = []
    for box, txt, _conf in result or []:
        ys = [p[1] for p in box]; xs = [p[0] for p in box]
        items.append(((min(ys) + max(ys)) / 2, min(xs), max(ys) - min(ys), txt))
    items.sort()
    lines, cur = [], []
    for it in items:
        if cur and abs(it[0] - np.mean([c[0] for c in cur])) > 0.5 * max(it[2], 8):
            lines.append(cur); cur = []
        cur.append(it)
    if cur:
        lines.append(cur)
    return "\n".join("   ".join(c[3] for c in sorted(l, key=lambda c: c[1])) for l in lines)


def ocr_text(pdf: Path, seed: int = 0, dpi: int = 150) -> str:
    """Simulate a scan (rotation, blur, noise) and OCR it with RapidOCR."""
    from PIL import Image, ImageFilter
    from pdf2image import convert_from_path

    rng = np.random.default_rng(seed)
    img = convert_from_path(str(pdf), dpi=dpi)[0].convert("L")
    img = img.rotate(float(rng.uniform(-1.2, 1.2)), expand=True, fillcolor=255)
    img = img.filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(0.3, 0.8))))
    arr = np.asarray(img, dtype=np.float32)
    arr = np.clip(arr + rng.normal(0, 18, arr.shape), 0, 255).astype(np.uint8)
    result, _ = _engine()(np.stack([arr] * 3, axis=-1))
    return clean_ocr(boxes_to_layout_text(result))


def clean_ocr(text: str) -> str:
    """Light, documented OCR clean-up applied before *both* extractors.

    - full-width punctuation (e.g. '：' or '＃') -> ASCII
    - money written with '.' as thousands separator ('2.322.82') -> '2,322.82'
    """
    text = text.translate(str.maketrans({"：": ":", "＃": "#", "，": ",", "（": "(", "）": ")"}))
    return re.sub(r"(?<![\d.,])(\d{1,3})((?:\.\d{3})+)(\.\d{2})(?!\d)",
                  lambda m: m.group(1) + m.group(2).replace(".", ",") + m.group(3), text)


# --------------------------------------------------------------- parsing
def parse_money(s: str) -> float:
    return round(float(s.replace(",", "")), 2)


def parse_date(s: str) -> str | None:
    for rx, fmts in DATE_PATTERNS:
        if rx.fullmatch(s):
            for f in fmts:
                try:
                    return datetime.strptime(s.replace(" ", ""), f).date().isoformat()
                except ValueError:
                    pass
    return None


def norm_value(field_name: str, v) -> str:
    """Normalise a value so extracted and ground-truth values compare fairly."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    if field_name in MONEY_FIELDS:
        return f"{float(v):.2f}"
    s = re.sub(r"\s+", " ", str(v)).strip()
    if field_name in ("vendor_vat_number", "vendor_name"):
        # Vendor names are compared case- and space-insensitively, because
        # OCR routinely drops the spaces between words.
        s = s.replace(" ", "")
    return s.lower() if field_name == "vendor_name" else s


# ---------------------------------------------------------------- rules
class RuleExtractor:
    """Hand-written rules based on the development layouts A, B and C."""

    LABELS = {
        "invoice_number": [r"Invoice\s*No\.?\s*:?", r"Invoice\s*#\s*:?", r"\bRef\s*:"],
        "invoice_date": [r"Invoice\s*Date\s*:?", r"(?<!\w)Date\s*:", r"Issued\s*:"],
        "due_date": [r"Due\s*Date\s*:?", r"Payment\s*due\s*:?", r"(?<!\w)Due\s*:"],
        "net_amount": [r"Subtotal\s*:?", r"Net\s*amount", r"Goods\s*total"],
        "vat_amount": [r"VAT\s*@\s*[\d.]+%\s*:?", r"VAT\s*\([\d.]+%\)", r"Tax\s+[\d.]+%"],
        "total_amount": [r"Total\s*Due\s*:?", r"Amount\s*payable", r"Grand\s*Total"],
    }

    def extract(self, text: str) -> dict:
        out = {f: None for f in FIELDS}
        m = VATNO_RE.search(text)
        out["vendor_vat_number"] = m.group(0).replace(" ", "") if m else None
        for line in text.splitlines()[:6]:
            for seg in re.split(r"\s{3,}|\|", line):
                if COMPANY_RE.search(seg):
                    out["vendor_name"] = seg.strip()
                    break
            if out["vendor_name"]:
                break
        for f, labels in self.LABELS.items():
            for lab in labels:
                if f in MONEY_FIELDS:
                    m = re.search(lab + r"[^\d\n]{0,60}?" + MONEY_RE.pattern, text, re.I)
                    if m:
                        out[f] = parse_money(m.group(1)); break
                elif f in DATE_FIELDS:
                    m = re.search(lab + r"\s*([^\n]{6,20})", text, re.I)
                    if m:
                        for rx, _ in DATE_PATTERNS:
                            dm = rx.search(m.group(1))
                            if dm and parse_date(dm.group(0)):
                                out[f] = parse_date(dm.group(0)); break
                    if out[f]:
                        break
                else:
                    m = re.search(lab + r"\s*([A-Z0-9][A-Z0-9/-]{3,})", text, re.I)
                    if m:
                        out[f] = m.group(1); break
        if out["total_amount"] is None:  # generic fallback: largest amount on the page
            amts = [parse_money(x) for x in MONEY_RE.findall(text)]
            out["total_amount"] = max(amts) if amts else None
        return out


# ----------------------------------------------------------- candidates
@dataclass
class Cand:
    kind: str
    value: str
    line: int
    x: float
    ctx: str
    feats: dict = field(default_factory=dict)


def _ctx_words(left: str, n: int = 4) -> str:
    words = re.findall(r"[a-z#@%]+", left.lower())
    return " ".join(words[-n:])


def candidates(text: str) -> dict[str, list[Cand]]:
    lines = text.splitlines()
    nl = max(len(lines), 1)
    money, dates, ids, segs = [], [], [], []
    for i, line in enumerate(lines):
        L = max(len(line), 1)
        dspans = []
        for rx, _ in DATE_PATTERNS:
            for m in rx.finditer(line):
                iso = parse_date(m.group(0))
                if iso:
                    dates.append(Cand("date", iso, i, m.start() / L, _ctx_words(line[: m.start()])))
                    dspans.append((m.start(), m.end()))
        for m in MONEY_RE.finditer(line):
            money.append(Cand("money", f"{parse_money(m.group(1)):.2f}", i, m.start() / L,
                              _ctx_words(line[: m.start()])))
        vat_spans = [(m.start(), m.end()) for m in VATNO_RE.finditer(line)]
        for m in ID_RE.finditer(line):
            if any(a <= m.start() < b for a, b in dspans + vat_spans):
                continue
            ids.append(Cand("id", m.group(1), i, m.start() / L, _ctx_words(line[: m.start()])))
        if i < 8:
            pos = 0
            for seg in re.split(r"(\s{3,}|\|)", line):
                if seg.strip() and not re.fullmatch(r"\s{3,}|\|", seg):
                    segs.append(Cand("seg", seg.strip(), i, pos / L, _ctx_words(line[:pos])))
                pos += len(seg)

    vals = np.array([float(c.value) for c in money]) if money else np.array([])
    uniq = sorted(set(vals.tolist()), reverse=True)
    for c in money:
        v = float(c.value)
        others = vals[vals != v]
        is_sum = any(abs(a + b - v) < 0.015 for j, a in enumerate(others) for b in others[j + 1:]) if len(others) > 1 else False
        is_part = any(abs(v + o - t) < 0.015 for o in others for t in others if t > v and t > o)
        vat_like = any(abs(v - r * o) <= 0.01 for o in others for r in (0.23, 0.135, 0.09))
        has_vat = any(abs(o - r * v) <= 0.01 for o in others for r in (0.23, 0.135, 0.09))
        c.feats = {"rank": uniq.index(v) / max(len(uniq) - 1, 1), "is_max": float(v == uniq[0]),
                   "is_sum": float(is_sum), "is_part": float(is_part), "vat_like": float(vat_like),
                   "has_vat_of_it": float(has_vat), "count": float((vals == v).sum())}
    dvals = sorted({c.value for c in dates})
    for c in dates:
        c.feats = {"is_earliest": float(c.value == dvals[0]), "is_latest": float(c.value == dvals[-1]),
                   "n_distinct": float(len(dvals))}
    for c in ids:
        c.feats = {"has_alpha": float(bool(re.search(r"[A-Z]", c.value))), "has_dash": float("-" in c.value),
                   "has_slash": float("/" in c.value), "len": len(c.value) / 12,
                   "digits_only": float(c.value.isdigit())}
    for c in segs:
        c.feats = {"company_word": float(bool(COMPANY_RE.search(c.value))), "n_words": len(c.value.split()) / 5,
                   "has_digit": float(bool(re.search(r"\d", c.value))), "upper": float(c.value.isupper())}
    for group in (money, dates, ids, segs):
        for c in group:
            c.feats.update({"rel_line": c.line / nl, "x": c.x})
    return {"money": money, "date": dates, "id": ids, "seg": segs}


FIELD_KIND = {"vendor_name": "seg", "invoice_number": "id", "invoice_date": "date", "due_date": "date",
              "net_amount": "money", "vat_amount": "money", "total_amount": "money"}


def cand_dict(c: Cand) -> dict:
    d = dict(c.feats)
    for w in c.ctx.split():
        d[f"ctx={w}"] = 1.0
    last = c.ctx.split()[-1] if c.ctx else "<none>"
    d[f"last={last}"] = 1.0
    return d


class CandidateModel:
    """One logistic-regression ranker per field over candidate features."""

    def __init__(self):
        self.models = {}

    def fit(self, texts: list[str], truths: list[dict]):
        from sklearn.feature_extraction import DictVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline

        for f, kind in FIELD_KIND.items():
            X, y = [], []
            for text, truth in zip(texts, truths):
                gold = norm_value(f, truth[f])
                for c in candidates(text)[kind]:
                    X.append(cand_dict(c)); y.append(int(norm_value(f, c.value) == gold))
            if len(set(y)) < 2:  # degenerate: every candidate is (or none is) correct
                self.models[f] = None
                continue
            self.models[f] = make_pipeline(DictVectorizer(), LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced")).fit(X, y)
        return self

    def extract(self, text: str) -> dict:
        out = {f: None for f in FIELDS}
        m = VATNO_RE.search(text)
        out["vendor_vat_number"] = m.group(0).replace(" ", "") if m else None
        cands = candidates(text)
        for f, kind in FIELD_KIND.items():
            cs = cands[kind]
            if cs and self.models[f] is None:
                out[f] = cs[0].value
            elif cs:
                p = self.models[f].predict_proba([cand_dict(c) for c in cs])[:, 1]
                out[f] = cs[int(np.argmax(p))].value
        return out


def arithmetic_ok(rec: dict) -> bool:
    try:
        return abs(float(rec["net_amount"]) + float(rec["vat_amount"]) - float(rec["total_amount"])) < 0.015
    except (TypeError, ValueError):
        return False
