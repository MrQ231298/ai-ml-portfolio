"""
Customer segmentation: scaling, k selection, stability, GMM comparison,
an RFM-score baseline, profiles, and validation on the following 90 days.

    python src/segment.py   -> reports/metrics.json, reports/*.csv, reports/*.png, models/segmenter.joblib
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score,
                             roc_auc_score, silhouette_score)
from sklearn.mixture import GaussianMixture

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from seg_lib import FEATURES, preprocess  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA, REPORTS, MODELS = ROOT / "data", ROOT / "reports", ROOT / "models"
SEED = 42
K_RANGE = range(2, 11)
BUSINESS_K = range(3, 9)            # pre-specified: 3-8 segments are usable by a marketing team
MIN_STABILITY = 0.75                # pre-specified: mean bootstrap ARI must reach this


def bootstrap_ari(Z, k, reps=30, frac=0.8, rng=None):
    """Fit on two overlapping 80% subsamples; compare labels on the shared customers."""
    rng = rng or np.random.default_rng(SEED)
    n = len(Z)
    out = []
    for r in range(reps):
        a = rng.choice(n, int(frac * n), replace=False)
        b = rng.choice(n, int(frac * n), replace=False)
        ka = KMeans(k, n_init=10, random_state=r).fit(Z[a])
        kb = KMeans(k, n_init=10, random_state=1000 + r).fit(Z[b])
        common = np.intersect1d(a, b)
        out.append(adjusted_rand_score(ka.predict(Z[common]), kb.predict(Z[common])))
    return float(np.mean(out)), float(np.std(out))


def rfm_scores(df):
    """Classic rule-based RFM: quintile scores 1-5 and common named segments."""
    r = pd.qcut(df["recency_days"].rank(method="first"), 5, labels=[5, 4, 3, 2, 1]).astype(int)
    f = pd.qcut(df["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    m = pd.qcut(df["monetary"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    fm = (f + m) / 2
    seg = np.select([(r >= 4) & (fm >= 4), (r >= 3) & (fm >= 3), (r >= 4) & (fm < 3), (r <= 2) & (fm >= 3), (r <= 2) & (fm < 3)],
                    ["Champions", "Loyal", "New / promising", "At risk", "Lost / hibernating"], "Needs attention")
    return pd.DataFrame({"R": r, "F": f, "M": m, "rfm_score": r + f + m, "rfm_segment": seg}, index=df.index)


def name_segments(prof: pd.DataFrame) -> dict[int, str]:
    """Readable names from the profile medians.

    The first five rules were written before the first run. The high-cancellation rule was added
    after the first run showed a small cluster (about 2% of customers) whose mean cancel rate was ~0.4;
    without it that cluster was labelled "Cooling off", which hid its defining trait.
    """
    names = {}
    mon_rank = prof["monetary"].rank(ascending=False)
    for c, r in prof.iterrows():
        if mon_rank[c] == 1:
            names[c] = "VIP / wholesale-scale"
        elif r["cancel_rate"] >= 0.25:
            names[c] = "High-cancellation"
        elif r["recency_days"] > 120 and r["frequency"] <= 1.5:
            names[c] = "Lapsed one-off"
        elif r["tenure_days"] < 60:
            names[c] = "New this quarter"
        elif r["recency_days"] <= 45 and r["frequency"] >= 3:
            names[c] = "Active regulars"
        elif r["recency_days"] > 90:
            names[c] = "Cooling off"
        else:
            names[c] = "Occasional"
    # make names unique
    seen = {}
    for c in sorted(names):
        n = names[c]; seen[n] = seen.get(n, 0) + 1
        if seen[n] > 1:
            names[c] = f"{n} ({seen[n]})"
    return names


def main() -> None:
    df = pd.read_csv(DATA / "customer_features_calibration.csv", index_col="customer_id")
    meta = json.loads((DATA / "source_metadata.json").read_text())
    pre = preprocess().fit(df[FEATURES].values)
    Z = pre.transform(df[FEATURES].values)

    # --- k selection grid
    rng = np.random.default_rng(SEED)
    grid = []
    for k in K_RANGE:
        km = KMeans(k, n_init=20, random_state=SEED).fit(Z)
        gm = GaussianMixture(k, covariance_type="full", n_init=3, random_state=SEED).fit(Z)
        stab, stab_sd = bootstrap_ari(Z, k, rng=rng)
        grid.append({"k": k, "inertia": km.inertia_, "silhouette": silhouette_score(Z, km.labels_, sample_size=3000, random_state=SEED),
                     "calinski_harabasz": calinski_harabasz_score(Z, km.labels_), "davies_bouldin": davies_bouldin_score(Z, km.labels_),
                     "bootstrap_ari_mean": stab, "bootstrap_ari_sd": stab_sd, "gmm_bic": gm.bic(Z),
                     "kmeans_vs_gmm_ari": adjusted_rand_score(km.labels_, gm.predict(Z))})
    grid = pd.DataFrame(grid)
    grid.round(4).to_csv(REPORTS / "k_selection.csv", index=False)
    eligible = grid[grid.k.isin(BUSINESS_K) & (grid.bootstrap_ari_mean >= MIN_STABILITY)]
    rule = "highest silhouette among k in 3-8 with mean bootstrap ARI >= 0.75"
    if eligible.empty:
        eligible = grid[grid.k.isin(BUSINESS_K)]
        rule += " (no k met the stability bar; fell back to highest silhouette in 3-8)"
    k_best = int(eligible.sort_values("silhouette", ascending=False).iloc[0]["k"])
    km = KMeans(k_best, n_init=50, random_state=SEED).fit(Z)
    labels = km.labels_
    seeds_ari = [adjusted_rand_score(labels, KMeans(k_best, n_init=10, random_state=s).fit_predict(Z)) for s in range(10)]

    # --- profiles
    prof = df.assign(segment=labels).groupby("segment").agg(
        customers=("frequency", "size"), recency_days=("recency_days", "median"), frequency=("frequency", "median"),
        monetary=("monetary", "median"), tenure_days=("tenure_days", "median"), avg_basket_value=("avg_basket_value", "median"),
        distinct_products=("distinct_products", "median"), cancel_rate=("cancel_rate", "mean"),
        revenue_share=("monetary", "sum"), future_90d_purchase_rate=("future_90d_purchased", "mean"),
        future_90d_revenue_mean=("future_90d_revenue", "mean"), future_90d_revenue_share=("future_90d_revenue", "sum"))
    prof["revenue_share"] /= df["monetary"].sum()
    prof["future_90d_revenue_share"] /= df["future_90d_revenue"].sum()
    prof["customer_share"] = prof["customers"] / len(df)
    names = name_segments(prof)
    prof.insert(0, "name", prof.index.map(names))
    prof.round(4).to_csv(REPORTS / "segment_profiles.csv")

    # --- RFM baseline and hold-out validation
    rfm = rfm_scores(df)
    seg_names = pd.Series(labels, index=df.index).map(names)
    y = df["future_90d_purchased"].astype(int).values
    # score = calibration-period purchase rate of the customer's group (no hold-out information used)
    def group_score(groups):
        return groups.map(pd.Series(df["frequency"].values, index=df.index).groupby(groups).mean()).values
    auc = {"kmeans_segment_mean_frequency": float(roc_auc_score(y, group_score(seg_names))),
           "rfm_segment_mean_frequency": float(roc_auc_score(y, group_score(rfm["rfm_segment"]))),
           "rfm_total_score": float(roc_auc_score(y, rfm["rfm_score"])),
           "recency_only": float(roc_auc_score(y, -df["recency_days"]))}
    eta2 = {}
    for name, g in [("kmeans", seg_names), ("rfm_rules", rfm["rfm_segment"])]:
        v = df["future_90d_revenue"]
        eta2[name] = float(((v.groupby(g).transform("mean") - v.mean()) ** 2).sum() / ((v - v.mean()) ** 2).sum())
    cross = pd.crosstab(seg_names, rfm["rfm_segment"])
    cross.to_csv(REPORTS / "kmeans_vs_rfm_crosstab.csv")
    out = df.join(rfm).assign(segment=labels, segment_name=seg_names)
    out.to_csv(REPORTS / "customer_segments.csv")

    # --- plots
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    ax[0].plot(grid.k, grid.silhouette, marker="o"); ax[0].set_title("Silhouette (higher better)")
    ax[1].plot(grid.k, grid.bootstrap_ari_mean, marker="o"); ax[1].axhline(MIN_STABILITY, ls="--", c="grey"); ax[1].set_title("Bootstrap stability (ARI)")
    ax[2].plot(grid.k, grid.gmm_bic, marker="o"); ax[2].set_title("GMM BIC (lower better)")
    for a in ax:
        a.axvline(k_best, c="#d1495b", lw=0.8); a.set_xlabel("k")
    fig.tight_layout(); fig.savefig(REPORTS / "k_selection.png", dpi=120); plt.close(fig)
    pca = PCA(2, random_state=SEED).fit(Z); P = pca.transform(Z)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for c in sorted(set(labels)):
        ax.scatter(P[labels == c, 0], P[labels == c, 1], s=6, alpha=0.5, label=names[c])
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.0%})"); ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.0%})")
    ax.legend(markerscale=3, fontsize=8); ax.set_title("Segments in PCA space (scaled log features)")
    fig.tight_layout(); fig.savefig(REPORTS / "segments_pca.png", dpi=120); plt.close(fig)
    zprof = pd.DataFrame(km.cluster_centers_, columns=FEATURES, index=[names[c] for c in range(k_best)])
    fig, ax = plt.subplots(figsize=(9, 0.6 * k_best + 1.5))
    im = ax.imshow(zprof.values, cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
    ax.set_xticks(range(len(FEATURES))); ax.set_xticklabels(FEATURES, rotation=30, ha="right"); ax.set_yticks(range(k_best)); ax.set_yticklabels(zprof.index)
    for i in range(k_best):
        for j in range(len(FEATURES)):
            ax.text(j, i, f"{zprof.values[i, j]:.1f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="z-score (centroid)"); ax.set_title("Segment centroids (standardised)")
    fig.tight_layout(); fig.savefig(REPORTS / "segment_heatmap.png", dpi=120); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4))
    pp = prof.sort_values("future_90d_purchase_rate")
    ax.barh(pp["name"], pp["future_90d_purchase_rate"] * 100, color="#00798c")
    ax.set_xlabel("% who bought again in the next 90 days (hold-out)"); ax.set_title("Do segments predict future behaviour?")
    fig.tight_layout(); fig.savefig(REPORTS / "holdout_by_segment.png", dpi=120); plt.close(fig)

    joblib.dump({"preprocess": pre, "kmeans": km, "features": FEATURES, "names": names}, MODELS / "segmenter.joblib")
    metrics = {"data": meta, "features": FEATURES, "scaling": "log1p on six skewed features, cancel_rate clipped to [0,1], then z-score",
               "k_grid": grid.round(4).to_dict("records"), "selection_rule": rule, "k_selected": k_best,
               "selected": grid[grid.k == k_best].round(4).to_dict("records")[0],
               "seed_stability_ari_mean": float(np.mean(seeds_ari)), "seed_stability_ari_min": float(np.min(seeds_ari)),
               "segment_names": {int(k): v for k, v in names.items()},
               "profiles": prof.round(4).reset_index().to_dict("records"),
               "holdout_validation": {"buyers_share": float(y.mean()), "auc_predict_purchase_next_90d": auc,
                                      "eta_squared_future_revenue": eta2},
               "kmeans_vs_rfm_rules_ari": float(adjusted_rand_score(seg_names, rfm["rfm_segment"]))}
    (REPORTS / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    print(grid.round(3).to_string()); print("k =", k_best, rule)
    print(prof.round(3).to_string()); print(json.dumps(metrics["holdout_validation"], indent=1), metrics["kmeans_vs_rfm_rules_ari"], np.mean(seeds_ari))


if __name__ == "__main__":
    main()
