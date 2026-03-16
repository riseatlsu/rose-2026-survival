"""
test_rate_features.py

Tests whether using rate-based features (per month) instead of raw cumulative
counts reduces reverse causality and changes feature importance ranking.

Reverse causality concern:
  Raw counts (total forks, total commits) correlate with time_to_event because
  repos that survived longer had more time to accumulate them.
  Rates (forks/month, commits/month) control for repo age.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance

PROJECT_ROOT = Path(__file__).parent.parent
INPUT_FILE   = PROJECT_ROOT / "out" / "survival_dataset_complete.csv"
OUT_DIR      = PROJECT_ROOT / "out" / "test_rate_features"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
COX_ALPHA    = 1e-2


# ── helpers ──────────────────────────────────────────────────────────────────
def load_df():
    df = pd.read_csv(INPUT_FILE)

    def to_bin(col):
        return df[col].map(lambda x: 1 if str(x).lower() in ("true", "1", "yes") else 0)

    df["has_readme_bin"]           = to_bin("has_readme")
    df["has_contributing_bin"]     = to_bin("has_contributing")
    df["has_code_of_conduct_bin"]  = to_bin("has_code_of_conduct")
    df["has_pr_template_bin"]      = to_bin("has_pr_template")
    df["has_issue_template_bin"]   = to_bin("has_issue_template")
    df["has_newcomer_labels_bin"]  = to_bin("has_newcomer_labels")
    df["is_organization"]          = (df["owner_type"] == "Organization").astype(int)

    # Raw cumulative counts (log1p)
    df["raw_commits"]       = np.log1p(df["commits_count"].fillna(0))
    df["raw_contributors"]  = np.log1p(df["contributors_count"].fillna(0))
    df["raw_forks"]         = np.log1p(df["Number of forks"].fillna(0))

    # Rate-based features (log1p)
    df["rate_commits"]      = np.log1p(df["Average number of commits per month"].fillna(0))
    df["rate_newcomers"]    = np.log1p(df["Average number of newcomers per month"].fillna(0))
    df["rate_forks"]        = np.log1p(df["Average number of forks per month"].fillna(0))

    df = df.dropna(subset=["time_to_event_months", "event_dead"])
    df = df[df["time_to_event_months"] > 0].copy()
    return df


def make_surv(df):
    return Surv.from_arrays(
        event=df["event_dead"].astype(bool).values,
        time=df["time_to_event_months"].values,
    )


def fit_and_importance(df, features, labels, title, filename):
    X = df[features].fillna(0).copy()
    numeric = [f for f in features if f.startswith("raw_") or f.startswith("rate_")]
    if numeric:
        X[numeric] = StandardScaler().fit_transform(X[numeric])

    y = make_surv(df)

    cox = CoxPHSurvivalAnalysis(alpha=COX_ALPHA)
    cox.fit(X, y)
    c_index = cox.score(X, y)

    print(f"\n{'='*55}")
    print(f"{title}  —  C-index: {c_index:.4f}")
    print(f"{'='*55}")
    for f, coef in zip(features, cox.coef_):
        hr = np.exp(coef)
        print(f"  {labels[f]:30s}  HR={hr:.3f}  coef={coef:.4f}")

    # reverse causality check
    print(f"\n  Correlação com time_to_event:")
    for f in features:
        r = df[f].corr(df["time_to_event_months"])
        print(f"    {labels[f]:30s}  r={r:.3f}")

    # permutation importance
    def cindex_scorer(est, X_, y_):
        risk = est.predict(X_)
        return concordance_index_censored(y_["event"], y_["time"], risk)[0]

    perm = permutation_importance(
        cox, X, y,
        scoring=cindex_scorer,
        n_repeats=20,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    imp_df = pd.DataFrame({
        "Feature":          [labels[f] for f in features],
        "Importance (mean)": perm.importances_mean.round(6),
        "Importance (std)":  perm.importances_std.round(6),
    }).sort_values("Importance (mean)", ascending=False)

    print(f"\n  Permutation importance:")
    for _, row in imp_df.iterrows():
        print(f"    {row['Feature']:30s}  {row['Importance (mean)']:+.6f}")

    # plot
    fig, ax = plt.subplots(figsize=(8, max(4, len(imp_df) * 0.55)))
    sorted_imp = imp_df.sort_values("Importance (mean)", ascending=True)
    colors = ["#d62728" if v < 0 else "#1f77b4" for v in sorted_imp["Importance (mean)"]]
    ax.barh(sorted_imp["Feature"], sorted_imp["Importance (mean)"],
            xerr=sorted_imp["Importance (std)"], color=colors, alpha=0.85,
            error_kw={"elinewidth": 1.5})
    ax.axvline(0, color="gray", ls="--", lw=1)
    ax.set_xlabel("Permutation Importance (Δ C-index)")
    ax.set_title(f"{title}\n(C-index={c_index:.3f}, alpha={COX_ALPHA})")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    path = OUT_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: {path}")

    imp_df["C-index"] = round(c_index, 4)
    imp_df.to_csv(OUT_DIR / filename.replace(".png", ".csv"), index=False)

    return imp_df, c_index


# ── main ─────────────────────────────────────────────────────────────────────
df = load_df()

BINARY = [
    "has_readme_bin", "has_contributing_bin", "has_code_of_conduct_bin",
    "has_pr_template_bin", "has_issue_template_bin",
    "has_newcomer_labels_bin", "is_organization",
]
LABELS_BINARY = {
    "has_readme_bin":          "Has README",
    "has_contributing_bin":    "Has CONTRIBUTING",
    "has_code_of_conduct_bin": "Has Code of Conduct",
    "has_pr_template_bin":     "Has PR Template",
    "has_issue_template_bin":  "Has Issue Template",
    "has_newcomer_labels_bin": "Has Newcomer Labels",
    "is_organization":         "Is Organization",
}

# Model A: raw cumulative counts (current script 19)
FEATURES_RAW = BINARY + ["raw_commits", "raw_contributors", "raw_forks"]
LABELS_RAW = {**LABELS_BINARY,
    "raw_commits":      "Commits (log)",
    "raw_contributors": "Contributors (log)",
    "raw_forks":        "Forks (log)",
}

# Model B: rate-based features
FEATURES_RATE = BINARY + ["rate_commits", "rate_newcomers", "rate_forks"]
LABELS_RATE = {**LABELS_BINARY,
    "rate_commits":    "Commits/month (log)",
    "rate_newcomers":  "Newcomers/month (log)",
    "rate_forks":      "Forks/month (log)",
}

# Model C: mixed — raw contributors + rates for forks/commits
FEATURES_MIX = BINARY + ["raw_contributors", "rate_commits", "rate_forks"]
LABELS_MIX = {**LABELS_BINARY,
    "raw_contributors": "Contributors (log)",
    "rate_commits":     "Commits/month (log)",
    "rate_forks":       "Forks/month (log)",
}

imp_raw,  c_raw  = fit_and_importance(df, FEATURES_RAW,  LABELS_RAW,  "Model A — Raw counts",        "importance_raw.png")
imp_rate, c_rate = fit_and_importance(df, FEATURES_RATE, LABELS_RATE, "Model B — Rate features",     "importance_rate.png")
imp_mix,  c_mix  = fit_and_importance(df, FEATURES_MIX,  LABELS_MIX,  "Model C — Mixed (rate+raw)",  "importance_mix.png")

print(f"\n{'='*55}")
print(f"SUMMARY")
print(f"{'='*55}")
print(f"  Model A (raw counts) C-index : {c_raw:.4f}")
print(f"  Model B (rates)      C-index : {c_rate:.4f}")
print(f"  Model C (mixed)      C-index : {c_mix:.4f}")

# ── Comparison plots ──────────────────────────────────────────────────────────

# 1. Side-by-side importance bars (top 5 features per model)
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
models = [
    (imp_raw,  f"Model A — Raw counts\n(C-index={c_raw:.3f})",   "#1f77b4"),
    (imp_rate, f"Model B — Rate features\n(C-index={c_rate:.3f})", "#ff7f0e"),
    (imp_mix,  f"Model C — Mixed\n(C-index={c_mix:.3f})",         "#2ca02c"),
]
for ax, (imp, title, color) in zip(axes, models):
    top = imp.sort_values("Importance (mean)", ascending=True).tail(7)
    ax.barh(top["Feature"], top["Importance (mean)"],
            xerr=top["Importance (std)"], color=color, alpha=0.8,
            error_kw={"elinewidth": 1.5})
    ax.axvline(0, color="gray", ls="--", lw=1)
    ax.set_xlabel("Permutation Importance (Δ C-index)")
    ax.set_title(title, fontsize=11)
    ax.grid(True, axis="x", alpha=0.3)

fig.suptitle("Feature Importance Comparison — Raw vs Rate vs Mixed", fontsize=13, y=1.02)
fig.tight_layout()
path = OUT_DIR / "comparison_importance.png"
fig.savefig(path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {path}")

# 2. Reverse causality: correlation with time_to_event for numeric features
numeric_features = {
    "Model A\n(raw)":   {"Commits (log)": "raw_commits",
                          "Contributors (log)": "raw_contributors",
                          "Forks (log)": "raw_forks"},
    "Model B\n(rates)": {"Commits/month": "rate_commits",
                          "Newcomers/month": "rate_newcomers",
                          "Forks/month": "rate_forks"},
    "Model C\n(mixed)": {"Contributors (log)": "raw_contributors",
                          "Commits/month": "rate_commits",
                          "Forks/month": "rate_forks"},
}

fig, ax = plt.subplots(figsize=(11, 5))
x = np.arange(3)
width = 0.22
colors_feat = ["#1f77b4", "#ff7f0e", "#2ca02c"]

for i, (model_label, feat_map) in enumerate(numeric_features.items()):
    corrs = [df[col].corr(df["time_to_event_months"]) for col in feat_map.values()]
    feat_labels = list(feat_map.keys())
    offset = (i - 1) * width
    bars = ax.bar(x + offset, corrs, width, label=model_label,
                  color=colors_feat[i], alpha=0.8)
    for bar, val in zip(bars, corrs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.2f}", ha="center", va="bottom", fontsize=8)

ax.set_xticks(x)
ax.set_xticklabels(["Feature 1\n(commits)", "Feature 2\n(contributors/newcomers)", "Feature 3\n(forks)"])
ax.set_ylabel("Pearson r with time_to_event")
ax.set_title("Reverse Causality Check\nCorrelation of numeric features with time-to-event\n(lower = less reverse causality)")
ax.axhline(0, color="black", lw=0.8)
ax.legend()
ax.grid(True, axis="y", alpha=0.3)
fig.tight_layout()
path = OUT_DIR / "comparison_reverse_causality.png"
fig.savefig(path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {path}")

# 3. C-index comparison bar
fig, ax = plt.subplots(figsize=(6, 4))
labels_ci = ["Model A\n(raw counts)", "Model B\n(rates)", "Model C\n(mixed)"]
cvals = [c_raw, c_rate, c_mix]
bars = ax.bar(labels_ci, cvals, color=["#1f77b4", "#ff7f0e", "#2ca02c"], alpha=0.85)
for bar, val in zip(bars, cvals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
            f"{val:.4f}", ha="center", va="bottom", fontsize=10)
ax.set_ylim(0.80, 0.91)
ax.set_ylabel("C-index")
ax.set_title("C-index by Model")
ax.grid(True, axis="y", alpha=0.3)
fig.tight_layout()
path = OUT_DIR / "comparison_cindex.png"
fig.savefig(path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {path}")

print(f"\nOutputs saved to: {OUT_DIR}")
