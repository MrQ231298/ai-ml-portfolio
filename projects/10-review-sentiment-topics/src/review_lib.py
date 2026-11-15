"""Shared loading / labelling for the review analysis project."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "womens_clothing_reviews.csv"
LABELS = ["negative", "neutral", "positive"]
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def rating_to_label(r: int) -> str:
    """1-2 stars -> negative, 3 -> neutral, 4-5 -> positive."""
    return "negative" if r <= 2 else ("neutral" if r == 3 else "positive")


def load_reviews(path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    df = df.dropna(subset=["Review Text"]).copy()
    df["Title"] = df["Title"].fillna("")
    df["text"] = (df["Title"] + ". " + df["Review Text"]).str.strip(". ").str.strip()
    df["label"] = df["Rating"].map(rating_to_label)
    df = df.rename(columns={"Department Name": "department", "Class Name": "class_name",
                            "Recommended IND": "recommended", "Rating": "rating", "Age": "age"})
    return df.reset_index(drop=True)
