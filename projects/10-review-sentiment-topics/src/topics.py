"""
Topic modelling and an insights report for the clothing reviews.

Method
------
- TF-IDF (unigrams + bigrams, English stop words plus generic review
  words such as "love", "dress", "retailer") on review text.
- NMF topic models for k in {6, 8, 10, 12, 15}.
- k chosen by mean NPMI coherence of each topic's top-10 words, computed
  from document co-occurrence in the corpus itself.
- Stability: the chosen k is re-fitted with 3 further random seeds; for
  each topic we take the best Jaccard overlap of top-10 words with any
  topic from the other run, and report the mean.
- Each review is assigned its dominant topic. The report gives, per topic:
  share of reviews, mean rating, share of negative (1-2 star) reviews,
  recommendation rate, lift of negativity vs the whole corpus, and the
  departments where it is most common.

    python src/topics.py
"""

from __future__ import annotations

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
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer, TfidfVectorizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from review_lib import ROOT, load_reviews  # noqa: E402

REPORTS, MODELS = ROOT / "reports", ROOT / "models"
K_GRID = [6, 8, 10, 12, 15]
EXTRA_STOP = {"love", "loved", "like", "dress", "top", "shirt", "retailer", "just", "really", "wear", "wore",
              "bought", "ordered", "order", "im", "ive", "dont", "didnt", "got", "did", "does", "item", "piece",
              "would", "one", "also", "much", "well", "get", "will", "could", "even", "still", "though", "way"}


def npmi_coherence(top_idx: list[int], X_bin) -> float:
    """Mean normalised PMI over word pairs, from document co-occurrence."""
    n = X_bin.shape[0]
    sub = X_bin[:, top_idx]
    df = np.asarray(sub.sum(axis=0)).ravel() / n
    co = (sub.T @ sub).toarray() / n
    vals = []
    for i in range(len(top_idx)):
        for j in range(i + 1, len(top_idx)):
            pij = co[i, j]
            if pij <= 0:
                vals.append(-1.0); continue
            vals.append(np.log(pij / (df[i] * df[j])) / -np.log(pij))
    return float(np.mean(vals))


def top_terms(model, vocab, n=10):
    return [[vocab[i] for i in np.argsort(-comp)[:n]] for comp in model.components_]


def main() -> None:
    REPORTS.mkdir(exist_ok=True); MODELS.mkdir(exist_ok=True)
    df = load_reviews()
    stop = list(ENGLISH_STOP_WORDS | EXTRA_STOP)
    vec = TfidfVectorizer(stop_words=stop, ngram_range=(1, 2), min_df=15, max_df=0.4,
                          token_pattern=r"(?u)\b[a-z][a-z]+\b", lowercase=True)
    X = vec.fit_transform(df["Review Text"])
    vocab = np.array(vec.get_feature_names_out())
    X_bin = (X > 0).astype(np.float32).tocsc()
    print("vocab", len(vocab))

    sweep = {}
    models = {}
    for k in K_GRID:
        m = NMF(n_components=k, init="nndsvda", random_state=0, max_iter=400).fit(X)
        idx = [list(np.argsort(-c)[:10]) for c in m.components_]
        coh = float(np.mean([npmi_coherence(t, X_bin) for t in idx]))
        sweep[k] = {"mean_npmi": round(coh, 4), "reconstruction_err": round(float(m.reconstruction_err_), 2)}
        models[k] = m
        print(k, sweep[k])
    k_best = max(sweep, key=lambda k: sweep[k]["mean_npmi"])
    model = models[k_best]
    terms = top_terms(model, vocab)

    # stability across seeds
    jac = []
    for seed in (1, 2, 3):
        alt = NMF(n_components=k_best, init="random", random_state=seed, max_iter=600).fit(X)
        alt_terms = top_terms(alt, vocab)
        for t in terms:
            jac.append(max(len(set(t) & set(a)) / len(set(t) | set(a)) for a in alt_terms))
    stability = round(float(np.mean(jac)), 4)

    W = model.transform(X)
    df["topic"] = W.argmax(axis=1)
    df["topic_weight"] = W.max(axis=1)
    overall_neg = float((df.label == "negative").mean())
    rows = []
    for t in range(k_best):
        d = df[df.topic == t]
        dept = d.department.value_counts(normalize=True).head(2)
        rows.append({
            "topic": t, "top_terms": ", ".join(terms[t][:8]), "reviews": int(len(d)),
            "share": round(len(d) / len(df), 4), "mean_rating": round(float(d.rating.mean()), 3),
            "negative_share": round(float((d.label == "negative").mean()), 4),
            "negative_lift": round(float((d.label == "negative").mean() / overall_neg), 3),
            "recommend_rate": round(float(d.recommended.mean()), 4),
            "top_departments": "; ".join(f"{k} {v:.0%}" for k, v in dept.items()),
            "npmi": round(npmi_coherence(list(np.argsort(-model.components_[t])[:10]), X_bin), 4),
        })
    tab = pd.DataFrame(rows).sort_values("negative_lift", ascending=False)
    tab.to_csv(REPORTS / "topic_summary.csv", index=False)

    # example snippets: most representative negative review per topic (short)
    examples = {}
    for t in range(k_best):
        d = df[(df.topic == t) & (df.label == "negative")].sort_values("topic_weight", ascending=False)
        if len(d):
            txt = d.iloc[0]["Review Text"]
            examples[t] = (txt[:220] + "...") if len(txt) > 220 else txt

    # plots
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(tab.share * 100, tab.negative_share * 100, s=80, color="#2b6cb0")
    for _, r in tab.iterrows():
        ax.annotate(f"T{r.topic}: {', '.join(r.top_terms.split(', ')[:3])}", (r.share * 100, r.negative_share * 100),
                    fontsize=7, xytext=(4, 3), textcoords="offset points")
    ax.axhline(overall_neg * 100, color="grey", ls=":", label=f"corpus negative share {overall_neg:.1%}")
    ax.set_xlabel("% of reviews (dominant topic)"); ax.set_ylabel("% of topic's reviews that are 1-2 stars")
    ax.set_title(f"NMF topics (k={k_best}): size vs negativity"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(REPORTS / "topics_negativity.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(list(sweep), [v["mean_npmi"] for v in sweep.values()], marker="o")
    ax.axvline(k_best, color="k", ls="--", lw=1)
    ax.set_xlabel("number of topics k"); ax.set_ylabel("mean NPMI (top-10 words)"); ax.set_title("Topic coherence sweep")
    fig.tight_layout(); fig.savefig(REPORTS / "coherence_sweep.png", dpi=130); plt.close(fig)

    joblib.dump({"vectorizer": vec, "nmf": model, "terms": terms}, MODELS / "topics_nmf.joblib")

    # ---- insights report (all numbers computed above) ----
    sent = json.loads((REPORTS / "sentiment_metrics.json").read_text()) if (REPORTS / "sentiment_metrics.json").exists() else None
    lines = [
        "# Insights report — clothing reviews (auto-generated)",
        "",
        f"Generated by `src/topics.py` on {datetime.now(ZoneInfo('Europe/Dublin')):%d %B %Y %H:%M} (Europe/Dublin).",
        f"Data: {len(df):,} real, anonymised reviews (Women's E-Commerce Clothing Reviews, CC0).",
        "Every figure below is computed from the data by the script; nothing is typed in by hand.",
        "",
        "## Headline numbers",
        "",
        f"- Mean rating: {df.rating.mean():.2f} / 5; recommendation rate {df.recommended.mean():.1%}.",
        f"- Negative (1–2 star) share: {overall_neg:.1%}; neutral (3 star): {(df.label == 'neutral').mean():.1%}.",
    ]
    by_dept = df.groupby("department").agg(n=("rating", "size"), mean_rating=("rating", "mean"),
                                           neg=("label", lambda s: (s == "negative").mean())).sort_values("neg", ascending=False)
    lines.append("- Most negative department: "
                 f"{by_dept.index[0]} ({by_dept.neg.iloc[0]:.1%} negative, n={by_dept.n.iloc[0]:,}); least negative: "
                 f"{by_dept.index[-1]} ({by_dept.neg.iloc[-1]:.1%}, n={by_dept.n.iloc[-1]:,}).")
    if sent:
        b = sent["best_model"]
        lines.append(f"- Sentiment classifier ({b}): test macro-F1 {sent['test'][b]['macro_f1']:.3f}, "
                     f"neutral-class F1 {sent['test'][b]['per_class_f1']['neutral']:.3f}.")
    lines += ["", f"## Topics (NMF, k = {k_best}, chosen by NPMI coherence; seed-stability Jaccard {stability:.2f})", "",
              "| Topic | Top terms | Share | Mean rating | Negative share | Negativity lift | Recommend | Top departments |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |"]
    for _, r in tab.iterrows():
        lines.append(f"| T{r.topic} | {r.top_terms} | {r.share:.1%} | {r.mean_rating:.2f} | {r.negative_share:.1%} | "
                     f"{r.negative_lift:.2f}x | {r.recommend_rate:.1%} | {r.top_departments} |")
    worst = tab.iloc[0]; best_t = tab.iloc[-1]
    lines += ["", "## What a merchandising team could take from this", "",
              f"1. **Biggest complaint theme:** topic T{worst.topic} ({worst.top_terms.split(', ')[0]}, "
              f"{', '.join(worst.top_terms.split(', ')[1:4])}) is {worst.negative_lift:.1f}x as likely to be a 1–2 star review "
              f"as the average review and covers {worst.share:.1%} of reviews.",
              f"2. **Strongest praise theme:** topic T{best_t.topic} ({', '.join(best_t.top_terms.split(', ')[:4])}) has the lowest "
              f"negative share ({best_t.negative_share:.1%}) and a {best_t.recommend_rate:.0%} recommendation rate.",
              "3. Topics are *dominant-topic* assignments from a bag-of-words model; they show where to read "
              "reviews first, not proven causes of returns.", "",
              "## One representative negative review per topic (verbatim, truncated)", ""]
    for t in tab.topic:
        if t in examples:
            lines.append(f"- **T{t}:** “{examples[t]}”")
    (REPORTS / "insights_report.md").write_text("\n".join(lines) + "\n")

    out = {
        "run_at_europe_dublin": datetime.now(ZoneInfo("Europe/Dublin")).isoformat(timespec="seconds"),
        "vocab_size": int(len(vocab)), "k_sweep": sweep, "k_best": k_best,
        "seed_stability_mean_jaccard": stability, "topics": tab.to_dict(orient="records"),
        "overall_negative_share": round(overall_neg, 4),
        "department_negative_share": by_dept.neg.round(4).to_dict(),
    }
    (REPORTS / "topic_metrics.json").write_text(json.dumps(out, indent=2))
    print("k_best", k_best, "stability", stability)
    print(tab[["topic", "top_terms", "share", "negative_share", "negative_lift"]].to_string(index=False))


if __name__ == "__main__":
    main()
