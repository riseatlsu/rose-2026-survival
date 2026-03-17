"""
19_survival_analysis.py

Survival analysis of ROS GitHub repositories.
Reference: Ait et al. (2022) MSR — "An Empirical Study on the Survival Rate of GitHub Projects"

Outputs:
  out/survival_analysis/tables/
    dataset_summary.csv
    survival_probabilities.csv
    logrank_tests.csv
    cox_model_results.csv
    cox_permutation_importance.csv
    rsf_feature_importance.csv

  out/survival_analysis/plots/
    km_overall.png
    km_by_owner_type.png
    km_by_community_size.png
    hazard_ratios.png
    cox_feature_importance.png
    feature_importance.png
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

from scipy.stats import norm as _norm
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.compare import compare_survival
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

# =========================
# CONFIG
# =========================
PROJECT_ROOT = Path(__file__).parent.parent
INPUT_FILE   = PROJECT_ROOT / "out" / "survival_dataset_complete.csv"
OUT_DIR      = PROJECT_ROOT / "out" / "survival_analysis"
TABLES_DIR   = OUT_DIR / "tables"
PLOTS_DIR    = OUT_DIR / "plots" / "paper"

STUDY_END_DATE      = datetime(2026, 3, 8) 
DEAD_THRESHOLD_DAYS = 180                   
TIMEPOINTS_MONTHS   = [12, 24, 36, 48, 60]
RANDOM_STATE        = 42
ALPHA               = 0.05
TEST_SIZE           = 0.25  # shared train/test split ratio for Cox and RSF

COLORS = {
    "Tier 1":  "#d62728",
    "Tier 2":  "#2ca02c",
    "Tier 3":  "#9467bd",
    "Overall": "#333333",
}
TIER_ORDER = ["Tier 1", "Tier 2", "Tier 3"]

# =========================
# SETUP
# =========================
TABLES_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 150,
})


# =========================
# DATA LOADING
# =========================
def load_data():
    df = pd.read_csv(INPUT_FILE)
    df["days_since_last_activity"] = pd.to_numeric(df["days_since_last_activity"], errors="coerce")
    df["event_dead"] = (df["days_since_last_activity"] > DEAD_THRESHOLD_DAYS).astype(int)

    # Recalculate time-to-event
    df["created_at_dt"]    = pd.to_datetime(df["created_at"], errors="coerce")
    df["last_activity_dt"] = pd.to_datetime(df["last_activity_date"], errors="coerce")

    def recalc_tte(row):
        if pd.isna(row["created_at_dt"]) or pd.isna(row["last_activity_dt"]):
            return None
        if row["event_dead"] == 1:
            return max(1, (row["last_activity_dt"] - row["created_at_dt"]).days) / 30.44
        return (STUDY_END_DATE - row["created_at_dt"]).days / 30.44

    df["time_to_event_months"] = df.apply(recalc_tte, axis=1)
    df = df.dropna(subset=["time_to_event_months", "event_dead"])
    df = df[df["time_to_event_months"] > 0].copy()

    # Community tiers (IQR-based on full dataset)
    q1 = df["contributors_count"].quantile(0.25)
    q3 = df["contributors_count"].quantile(0.75)

    def assign_tier(n):
        if n <= q1:   return "Tier 1"
        if n <= q3:   return "Tier 2"
        return "Tier 3"

    df["community_tier"]  = df["contributors_count"].apply(assign_tier)
    df["is_organization"] = (df["owner_type"] == "Organization").astype(int)

    def to_bool_int(col):
        return df[col].map(lambda x: 1 if str(x).lower() in ("true", "1", "yes") else 0)

    df["has_readme_bin"]           = to_bool_int("has_readme")
    df["has_contributing_bin"]     = to_bool_int("has_contributing")
    df["has_code_of_conduct_bin"]  = to_bool_int("has_code_of_conduct")
    df["has_pr_template_bin"]      = to_bool_int("has_pr_template")
    df["has_issue_template_bin"]   = to_bool_int("has_issue_template")
    df["has_newcomer_labels_bin"]  = to_bool_int("has_newcomer_labels")

    df["commits_per_month"]   = df["Average number of commits per month"].fillna(0)
    df["newcomers_per_month"] = df["Average number of newcomers per month"].fillna(0)
    df["forks_per_month"]     = df["Average number of forks per month"].fillna(0)
    df["contributors_count"]  = df["contributors_count"].fillna(0)

    # Stars excluded: high collinearity with forks (r≈0.78);    
    n = len(df)
    n_dead = int(df["event_dead"].sum())
    print(f"Dataset: {n} repos | Dead: {n_dead} ({100*n_dead/n:.1f}%) | "
          f"Threshold: {DEAD_THRESHOLD_DAYS}d ({DEAD_THRESHOLD_DAYS/30.44:.0f} months)")
    print(f"Community tiers  Q1={q1:.0f}  Q3={q3:.0f}")
    for t in TIER_ORDER:
        sub = df[df["community_tier"] == t]
        print(f"  {t}: {len(sub):3d} repos  |  death rate: {100*sub['event_dead'].mean():.1f}%")

    return df, q1, q3


# =========================
# HELPERS
# =========================
def make_surv_array(df):
    return Surv.from_arrays(
        event=df["event_dead"].astype(bool).values,
        time=df["time_to_event_months"].values,
    )


def km_at_timepoints(time, survival_prob, timepoints):
    results = {}
    for t in timepoints:
        mask = time <= t
        results[t] = float(survival_prob[mask][-1]) if mask.any() else 1.0
    return results


def logrank_test(df, group_col):
    sub = df[df[group_col].notna()].copy()
    y   = make_surv_array(sub)
    try:
        chisq, pval = compare_survival(y, sub[group_col].values)
        return chisq, pval, sorted(sub[group_col].unique())
    except Exception as e:
        print(f"  Log-rank test failed for {group_col}: {e}")
        return None, None, []


def format_pval(p):
    if p is None:  return "N/A"
    if p < 0.001:  return "< 0.001 ***"
    if p < 0.01:   return f"{p:.3f} **"
    if p < 0.05:   return f"{p:.3f} *"
    return f"{p:.3f}"


# =========================
# 1. KAPLAN-MEIER CURVES
# =========================
def plot_km_overall(df):
    print("\n--- KM Overall ---")
    time_km, surv_km = kaplan_meier_estimator(
        df["event_dead"].astype(bool), df["time_to_event_months"]
    )
    pts  = km_at_timepoints(time_km, surv_km, TIMEPOINTS_MONTHS)
    info = "  ".join([f"S({t}m)={v:.2f}" for t, v in pts.items()])

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.step(time_km, surv_km, where="post", color=COLORS["Overall"], lw=2, label="All repos")
    ax.axhline(0.5, color="gray", ls="--", lw=1, alpha=0.6, label="S(t)=0.50")
    ax.set_xlabel(f"Time (months)\n{info}", fontsize=9)
    ax.set_ylabel("Survival probability S(t)")
    ax.set_title("Kaplan-Meier: All ROS Repositories")
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = PLOTS_DIR / "km_overall.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return time_km, surv_km, pts


def plot_km_by_group(df, group_col, title, filename, color_map=None, group_order=None):
    """Generic KM plot by group with log-rank p-value."""
    print(f"\n--- KM by {group_col} ---")
    groups    = [g for g in (group_order or []) if g in df[group_col].unique()] \
                or sorted(df[group_col].unique())
    chisq, pval, _ = logrank_test(df, group_col)

    fig, ax = plt.subplots(figsize=(8, 5))
    for g in groups:
        sub   = df[df[group_col] == g]
        t_km, s_km = kaplan_meier_estimator(sub["event_dead"].astype(bool), sub["time_to_event_months"])
        color = (color_map or {}).get(g)
        ax.step(t_km, s_km, where="post", lw=2, color=color,
                label=f"{g} (n={len(sub)}, deaths={int(sub['event_dead'].sum())})")

    ax.set_title(f"{title}\nLog-rank p={format_pval(pval)}")
    ax.set_xlabel("Time (months)")
    ax.set_ylabel("Survival probability S(t)")
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0)
    ax.axhline(0.5, color="gray", ls="--", lw=1, alpha=0.5)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = PLOTS_DIR / filename
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return chisq, pval


# =========================
# 2. LOG-RANK TESTS TABLE
# =========================
def run_logrank_tests(df):
    print("\n--- Log-Rank Tests ---")
    tests = [
        ("owner_type",             "Owner Type (Org vs User)"),
        ("community_tier",         "Community Tier (1/2/3)"),
        ("has_readme_bin",         "Has README"),
        ("has_contributing_bin",   "Has CONTRIBUTING"),
        ("has_code_of_conduct_bin","Has Code of Conduct"),
        ("has_pr_template_bin",    "Has PR Template"),
        ("has_issue_template_bin", "Has Issue Template"),
        ("has_newcomer_labels_bin","Has Newcomer Labels"),
    ]
    rows = []
    for col, label in tests:
        if col not in df.columns:
            continue
        chisq, pval, groups = logrank_test(df, col)
        chisq_str = f"{chisq:.3f}" if chisq is not None else 'N/A'
        print(f"  {label}: χ²={chisq_str}, p={format_pval(pval)}")
        rows.append({
            "Factor":             label,
            "Column":             col,
            "Groups":             str(groups),
            "N Groups":           len(groups),
            "Chi-squared":        round(chisq, 4) if chisq else None,
            "P-value":            round(pval,  4) if pval  else None,
            "Significant (p<0.05)": (pval < ALPHA) if pval else False,
        })

    result_df = pd.DataFrame(rows)
    path = TABLES_DIR / "logrank_tests.csv"
    result_df.to_csv(path, index=False)
    print(f"  Saved: {path}")
    return result_df


# =========================
# 3. COX PROPORTIONAL HAZARDS
# =========================
COX_BINARY_FEATURES = [
    "has_readme_bin",
    "has_contributing_bin",
    # "has_code_of_conduct_bin",  # EXCLUDED: only 24 repos (3.0%), 1 death → quasi-separation
    "has_pr_template_bin",
    # "has_issue_template_bin",   # EXCLUDED: only 10 repos (1.2%), 0 deaths → perfect separation
    "has_newcomer_labels_bin",
    "is_organization",
]
# Activity rates to capture project engagement independent of lifespan.
# contributors_count EXCLUDED to avoid confounding with newcomers_per_month
# stars_per_month excluded: high collinearity with forks (r=0.86).
# Pipeline applies median imputation + StandardScaler
COX_NUMERIC_FEATURES = [
    "commits_per_month",
    "newcomers_per_month",
    "forks_per_month",
]
COX_ALPHA = 1e-2  # L2 regularization — stabilizes rare/quasi-separated features

COX_FEATURE_LABELS = {
    "commits_per_month":       "Commits/month",
    "newcomers_per_month":     "Newcomers/month",
    "forks_per_month":         "Forks/month",
    "has_readme_bin":          "Has README",
    "has_contributing_bin":    "Has CONTRIBUTING",
    "has_pr_template_bin":     "Has PR Template",
    "has_newcomer_labels_bin": "Has Newcomer Labels",
    "is_organization":         "Is Organization",
}


def fit_cox(df):
    """Fit L2-regularized Cox PH model.
    Trains on 75%, evaluates on stratified 25% held-out test set.
    Permutation importance computed on test set to avoid in-sample bias."""
    print("\n--- Cox PH Model (train/test split) ---")

    binary       = [f for f in COX_BINARY_FEATURES  if f in df.columns]
    numeric      = [f for f in COX_NUMERIC_FEATURES if f in df.columns]
    all_features = numeric + binary

    X = df[all_features].fillna(0).copy()
    y = make_surv_array(df)

    event_vec   = df["event_dead"].astype(int)
    min_class   = min(int((event_vec == 0).sum()), int((event_vec == 1).sum()))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=event_vec if min_class >= 2 else None,
    )
    print(f"  Train: {len(X_train)} ({int(y_train['event'].sum())} events) | "
          f"Test: {len(X_test)} ({int(y_test['event'].sum())} events)")

    preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler()),
        ]), numeric),
        ("bin", "passthrough", binary),
    ])

    cox = Pipeline([
        ("preprocess", preprocessor),
        ("model",      CoxPHSurvivalAnalysis(alpha=COX_ALPHA)),
    ])
    cox.fit(X_train, y_train)

    c_index_train = cox.score(X_train, y_train)
    c_index_test  = cox.score(X_test,  y_test)
    print(f"  C-index (train): {c_index_train:.4f}")
    print(f"  C-index (test) : {c_index_test:.4f}")

    # Hazard ratios
    coefs = cox.named_steps["model"].coef_
    rows  = []
    for f, coef in zip(all_features, coefs):
        label = COX_FEATURE_LABELS.get(f, f)
        hr    = float(np.exp(coef))
        rows.append({
            "Feature":         label,
            "Coef":            round(float(coef), 4),
            "HR":              round(hr, 4),
            "log_HR":          round(float(coef), 4),
            "Direction":       "increased hazard" if hr > 1 else "decreased hazard",
            "C-index (train)": round(c_index_train, 4),
            "C-index (test)":  round(c_index_test,  4),
        })
        print(f"  {label:28s}  coef={coef:.4f}  HR={hr:.3f}")

    result_df = pd.DataFrame(rows)
    result_df.to_csv(TABLES_DIR / "cox_model_results.csv", index=False)
    print(f"  Saved: {TABLES_DIR / 'cox_model_results.csv'}")

    # Permutation importance on test set
    def cindex_scorer(est, Xp, yp):
        return concordance_index_censored(yp["event"], yp["time"], est.predict(Xp))[0]

    perm = permutation_importance(
        cox, X_test, y_test,
        scoring=cindex_scorer,
        n_repeats=20,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    perm_df = pd.DataFrame({
        "Feature":           [COX_FEATURE_LABELS.get(f, f) for f in all_features],
        "Importance (mean)": perm.importances_mean.round(6),
        "Importance (std)":  perm.importances_std.round(6),
        "C-index (test)":    round(c_index_test, 4),
    }).sort_values("Importance (mean)", ascending=False)
    perm_df.to_csv(TABLES_DIR / "cox_permutation_importance.csv", index=False)
    print(f"  Saved: {TABLES_DIR / 'cox_permutation_importance.csv'}")

    # Feature importance plot
    fig, ax = plt.subplots(figsize=(7, max(4, len(perm_df) * 0.55)))
    sp      = perm_df.sort_values("Importance (mean)", ascending=True)
    colors  = ["#d62728" if v < 0 else "#1f77b4" for v in sp["Importance (mean)"]]
    ax.barh(sp["Feature"], sp["Importance (mean)"], xerr=sp["Importance (std)"],
            color=colors, alpha=0.85, error_kw={"elinewidth": 1.5})
    ax.set_xlabel("Permutation Importance (Δ C-index)", fontsize=17)
    ax.tick_params(axis='both', labelsize=15)
    ax.axvline(0, color="gray", ls="--", lw=1)
    ax.grid(True, axis="x", alpha=0.3)
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color('black')
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "cox_feature_importance.png", dpi=300, bbox_inches="tight")
    fig.savefig(PLOTS_DIR / "cox_feature_importance.svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {PLOTS_DIR / 'cox_feature_importance.png'}")
    print(f"  Saved: {PLOTS_DIR / 'cox_feature_importance.svg'}")

    return result_df, all_features, c_index_test


N_BOOT_HR = 500  # bootstrap resamples for HR confidence intervals

def plot_hazard_ratios(df, c_index_test):
    """Publication forest plot — one row per feature, sorted by HR.
    Uses sksurv CoxPHSurvivalAnalysis fitted on the full dataset.
    95% CIs computed via bootstrap percentile (N=N_BOOT_HR).
    Wald p-values derived from bootstrap SE: z = coef / SE, p = 2*(1-Phi(|z|)).
    Binary: HR for present vs absent.  Continuous: HR per 1-SD change in raw units."""
    print("\n--- Plotting Hazard Ratios (forest plot) ---")

    PROT  = "#2166ac"
    RISK  = "#c0392b"
    GRAY  = "#888888"
    FS    = 10

    numeric  = [f for f in COX_NUMERIC_FEATURES if f in df.columns]
    binary   = [f for f in COX_BINARY_FEATURES  if f in df.columns]
    features = numeric + binary  # consistent order with fit_cox

    def build_pipeline():
        pre = ColumnTransformer([
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler",  StandardScaler()),
            ]), numeric),
            ("bin", "passthrough", binary),
        ])
        return Pipeline([("preprocess", pre),
                         ("model", CoxPHSurvivalAnalysis(alpha=COX_ALPHA))])

    # ── fit on full dataset ───────────────────────────────────────
    X    = df[features].fillna(0)
    y    = make_surv_array(df)
    pipe = build_pipeline()
    pipe.fit(X, y)
    coefs_full = pipe.named_steps["model"].coef_

    # ── bootstrap for CIs and SE ──────────────────────────────────
    print(f"  Bootstrap CIs (N={N_BOOT_HR}) ...")
    rng        = np.random.default_rng(RANDOM_STATE)
    boot_coefs = np.full((N_BOOT_HR, len(features)), np.nan)
    for i in range(N_BOOT_HR):
        idx  = rng.integers(0, len(X), size=len(X))
        X_b  = X.iloc[idx]
        y_b  = y[idx]
        try:
            m = build_pipeline()
            m.fit(X_b, y_b)
            boot_coefs[i] = m.named_steps["model"].coef_
        except Exception:
            boot_coefs[i] = coefs_full  # fallback on degenerate sample

    ci_lo_coef = np.nanpercentile(boot_coefs, 2.5,  axis=0)
    ci_hi_coef = np.nanpercentile(boot_coefs, 97.5, axis=0)
    boot_se    = np.nanstd(boot_coefs, axis=0)
    z_scores   = np.where(boot_se > 0, coefs_full / boot_se, np.nan)
    pvals      = 2 * (1 - _norm.cdf(np.abs(z_scores)))

    records = []
    for j, col in enumerate(features):
        label  = COX_FEATURE_LABELS.get(col, col)
        hr     = float(np.exp(coefs_full[j]))
        ci_lo  = float(np.exp(ci_lo_coef[j]))
        ci_hi  = float(np.exp(ci_hi_coef[j]))
        pval   = float(pvals[j]) if np.isfinite(pvals[j]) else None
        is_bin = col in COX_BINARY_FEATURES
        n_pos  = int((df[col] == 1).sum()) if is_bin else None
        records.append(dict(label=label, hr=hr, ci_lo=ci_lo, ci_hi=ci_hi,
                            pval=pval, is_bin=is_bin, n_pos=n_pos))

    # save bootstrap HR results to CSV
    hr_df = pd.DataFrame([{
        "Feature":    r["label"],
        "HR":         round(r["hr"],    4),
        "CI_lo":      round(r["ci_lo"], 4),
        "CI_hi":      round(r["ci_hi"], 4),
        "p-value":    round(r["pval"],  4) if r["pval"] is not None else None,
        "Significant": r["pval"] is not None and r["pval"] < 0.05,
    } for r in records])
    hr_df.to_csv(TABLES_DIR / "cox_hazard_ratios.csv", index=False)
    print(f"  Saved: {TABLES_DIR / 'cox_hazard_ratios.csv'}")

    # sort by HR ascending (most protective at top)
    records.sort(key=lambda r: r["hr"])
    n = len(records)

    fig, (ax_L, ax_R) = plt.subplots(
        1, 2, figsize=(12, max(5, n * 0.52 + 1.6)),
        gridspec_kw={"width_ratios": [1.7, 1.0], "wspace": 0.02}
    )

    # left panel: text
    ax_L.set_xlim(0, 1)
    ax_L.set_ylim(-0.7, n + 0.3)
    ax_L.axis("off")

    # right panel: forest plot
    valid = [v for r in records for v in [r["ci_lo"], r["ci_hi"]]
             if np.isfinite(v) and v > 0]
    xlo = max(0.05, min(valid) * 0.55)
    xhi = min(50,   max(valid) * 2.0)
    ax_R.set_xscale("log")
    ax_R.set_xlim(xlo, xhi)
    ax_R.set_ylim(-0.7, n + 0.3)
    ax_R.axvline(1, color="#333", ls="--", lw=1.1, zorder=3)

    # ── column headers ────────────────────────────────────────────
    hy = n + 0.05
    ax_L.text(0.01, hy, "Feature",      fontsize=FS, fontweight="bold", va="bottom")
    ax_L.text(0.60, hy, "HR (95% CI)",  fontsize=FS, fontweight="bold", va="bottom")
    ax_R.text(1.04, 1.0, "p-value",     fontsize=FS, fontweight="bold", va="bottom",
              ha="left", transform=ax_R.transAxes)
    ax_L.axhline(n - 0.2, color="#aaa", lw=0.8)
    ax_R.axhline(n - 0.2, color="#aaa", lw=0.8)

    # ── rows ──────────────────────────────────────────────────────
    for i, rec in enumerate(records):
        y     = i
        color = RISK if rec["hr"] > 1 else PROT
        sig   = rec["pval"] is not None and rec["pval"] < 0.05

        # alternating row background (both panels)
        bg = "#f5f5f5" if i % 2 == 0 else "white"
        ax_L.axhspan(y - 0.45, y + 0.45, color=bg, zorder=0)
        ax_R.axhspan(y - 0.45, y + 0.45, color=bg, zorder=0)

        # ── left panel text ───────────────────────────────────────
        # feature label — bold if significant
        fw = "bold" if sig else "normal"
        ax_L.text(0.01, y, rec["label"], fontsize=FS, va="center",
                  fontweight=fw, color="#111")

        # N annotation for binary
        if rec["is_bin"] and rec["n_pos"] is not None:
            ax_L.text(0.53, y, f"(N={rec['n_pos']})", fontsize=8.5,
                      va="center", color=GRAY, ha="right")

        # HR (CI) text
        ci_str = f"{rec['hr']:.2f} ({rec['ci_lo']:.2f}–{rec['ci_hi']:.2f})"
        ax_L.text(0.60, y, ci_str, fontsize=8.8, va="center", color=color)

        # ── right panel: CI bar + marker ─────────────────────────
        ax_R.plot([rec["ci_lo"], rec["ci_hi"]], [y, y],
                  color=color, lw=1.6, alpha=0.7, zorder=2, solid_capstyle="round")
        ax_R.plot([rec["ci_lo"], rec["ci_lo"]], [y - 0.12, y + 0.12],
                  color=color, lw=1.6, zorder=2)
        ax_R.plot([rec["ci_hi"], rec["ci_hi"]], [y - 0.12, y + 0.12],
                  color=color, lw=1.6, zorder=2)
        ax_R.scatter(rec["hr"], y, color=color, s=55, marker="D",
                     zorder=5, linewidths=0)

        # ── p-value (outside right panel) ────────────────────────
        yrel = (y + 0.5) / (n + 0.3 + 0.7)
        ax_R.text(1.04, yrel, format_pval(rec["pval"]), fontsize=8.5,
                  va="center", ha="left", transform=ax_R.transAxes,
                  color="#111" if sig else GRAY)

    # ── axes styling ──────────────────────────────────────────────
    ax_R.set_xlabel("Hazard Ratio (log scale)", fontsize=FS)
    ax_R.set_yticks([])
    ax_R.tick_params(axis="x", labelsize=8.5)
    ax_R.grid(True, axis="x", alpha=0.2, linestyle=":", zorder=0)
    for sp in ["top", "right", "left"]:
        ax_R.spines[sp].set_visible(False)
    ax_L.set_yticks([])

    # footer
    n_ev = int(df["event_dead"].sum())
    fig.text(0.5, 0.005,
             f"N = {len(df)} repositories  ·  Events (dead): {n_ev}  ·  "
             f"C-index (test set) = {c_index_test:.3f}  ·  L2 regularization α = {COX_ALPHA}",
             ha="center", fontsize=8, color=GRAY, style="italic")

    fig.tight_layout(rect=(0, 0.02, 1, 1))
    path = PLOTS_DIR / "hazard_ratios.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# =========================
# 4. RANDOM SURVIVAL FOREST
# =========================
def run_rsf(df):
    """RSF with stratified train/test split"""
    print("\n--- Random Survival Forest (train/test split) ---")

    numeric  = [f for f in COX_NUMERIC_FEATURES if f in df.columns]
    binary   = [f for f in COX_BINARY_FEATURES  if f in df.columns]
    features = numeric + binary

    X = df[features].fillna(0)
    y = make_surv_array(df)

    event_vec = df["event_dead"].astype(int)
    min_class = min(int((event_vec == 0).sum()), int((event_vec == 1).sum()))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=event_vec if min_class >= 2 else None,
    )

    rsf = RandomSurvivalForest(
        n_estimators=200,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features="sqrt",
        oob_score=True,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rsf.fit(X_train, y_train)

    c_index_oob  = rsf.oob_score_
    c_index_test = concordance_index_censored(
        y_test["event"], y_test["time"], rsf.predict(X_test)
    )[0]
    print(f"  C-index (OOB,  train): {c_index_oob:.4f}")
    print(f"  C-index (test set)   : {c_index_test:.4f}")

    def cindex_scorer(est, Xp, yp):
        return concordance_index_censored(yp["event"], yp["time"], est.predict(Xp))[0]

    perm = permutation_importance(
        rsf, X_test, y_test,
        scoring=cindex_scorer,
        n_repeats=15,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    importances = pd.DataFrame({
        "Feature":           [COX_FEATURE_LABELS.get(f, f) for f in features],
        "Importance (mean)": perm.importances_mean,
        "Importance (std)":  perm.importances_std,
        "C-index":           c_index_test,
    }).sort_values("Importance (mean)", ascending=False)

    importances.to_csv(TABLES_DIR / "rsf_feature_importance.csv", index=False)
    print(f"  Saved: {TABLES_DIR / 'rsf_feature_importance.csv'}")

    fig, ax = plt.subplots(figsize=(7, max(4, len(importances) * 0.55)))
    si = importances.sort_values("Importance (mean)", ascending=True)
    ax.barh(si["Feature"], si["Importance (mean)"], xerr=si["Importance (std)"],
            color="#1f77b4", alpha=0.8, error_kw={"elinewidth": 1.5})
    ax.set_xlabel("Permutation Importance (Δ C-index)")
    ax.set_title(f"RSF Feature Importance (test set)\n(C-index={c_index_test:.3f})")
    ax.axvline(0, color="gray", ls="--", lw=1)
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "feature_importance.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {PLOTS_DIR / 'feature_importance.png'}")

    return importances


# =========================
# 5. SUMMARY TABLES
# =========================
def build_survival_probabilities(df):
    """S(t) at TIMEPOINTS_MONTHS for overall, owner type, and community tiers."""
    print("\n--- Survival Probabilities Table ---")
    rows = []

    def add_row(label, group, sub_df):
        t_km, s_km = kaplan_meier_estimator(
            sub_df["event_dead"].astype(bool), sub_df["time_to_event_months"]
        )
        pts = km_at_timepoints(t_km, s_km, TIMEPOINTS_MONTHS)
        row = {"Group": label, "Subgroup": group,
               "N": len(sub_df), "Dead": int(sub_df["event_dead"].sum())}
        for tp, sv in pts.items():
            row[f"S({tp}m)"] = round(sv, 3)
        rows.append(row)

    add_row("Overall", "All", df)
    for ot in ["Organization", "User"]:
        add_row("Owner Type", ot, df[df["owner_type"] == ot])
    for t in TIER_ORDER:
        add_row("Community Tier", t, df[df["community_tier"] == t])

    result = pd.DataFrame(rows)
    result.to_csv(TABLES_DIR / "survival_probabilities.csv", index=False)
    print(f"  Saved: {TABLES_DIR / 'survival_probabilities.csv'}")

    print(f"\n  {'Group':<35} {'N':>5} {'S(12m)':>7} {'S(24m)':>7} {'S(36m)':>7} {'S(48m)':>7} {'S(60m)':>7}")
    print("  " + "-" * 80)
    for _, r in result.iterrows():
        print(f"  {r['Subgroup']:<35} {r['N']:>5} "
              f"{r.get('S(12m)','-'):>7} {r.get('S(24m)','-'):>7} "
              f"{r.get('S(36m)','-'):>7} {r.get('S(48m)','-'):>7} "
              f"{r.get('S(60m)','-'):>7}")
    return result


def build_dataset_summary(df, q1, q3):
    """High-level dataset_summary.csv."""
    print("\n--- Dataset Summary ---")
    rows = [
        {"Metric": "Total Repositories",              "Value": len(df)},
        {"Metric": "Dead (event=1)",                  "Value": int(df["event_dead"].sum())},
        {"Metric": "Alive/Censored (event=0)",        "Value": int((df["event_dead"] == 0).sum())},
        {"Metric": "Death Rate (%)",                  "Value": round(100 * df["event_dead"].mean(), 1)},
        {"Metric": "Median Time to Event (months)",   "Value": round(df["time_to_event_months"].median(), 1)},
        {"Metric": "Mean Time to Event (months)",     "Value": round(df["time_to_event_months"].mean(), 1)},
        {"Metric": "Max Time to Event (months)",      "Value": round(df["time_to_event_months"].max(), 1)},
        {"Metric": "Organizations",                   "Value": int((df["owner_type"] == "Organization").sum())},
        {"Metric": "Users",                           "Value": int((df["owner_type"] == "User").sum())},
        {"Metric": "Community Tier 1 (≤ Q1)",         "Value": int((df["community_tier"] == "Tier 1").sum())},
        {"Metric": "Community Tier 2 (Q1–Q3)",        "Value": int((df["community_tier"] == "Tier 2").sum())},
        {"Metric": "Community Tier 3 (> Q3)",         "Value": int((df["community_tier"] == "Tier 3").sum())},
        {"Metric": "Q1 contributors",                 "Value": q1},
        {"Metric": "Q3 contributors",                 "Value": q3},
        {"Metric": "Has README",                      "Value": int(df["has_readme_bin"].sum())},
        {"Metric": "Has CONTRIBUTING",                "Value": int(df["has_contributing_bin"].sum())},
        {"Metric": "Has Code of Conduct",             "Value": int(df["has_code_of_conduct_bin"].sum())},
        {"Metric": "Has PR Template",                 "Value": int(df["has_pr_template_bin"].sum())},
        {"Metric": "Has Issue Template",              "Value": int(df["has_issue_template_bin"].sum())},
        {"Metric": "Has Newcomer Labels",             "Value": int(df["has_newcomer_labels_bin"].sum())},
    ]
    result = pd.DataFrame(rows)
    result.to_csv(TABLES_DIR / "dataset_summary.csv", index=False)
    print(f"  Saved: {TABLES_DIR / 'dataset_summary.csv'}")
    return result


# =========================
# MAIN
# =========================
def main():
    print("=" * 65)
    print("SURVIVAL ANALYSIS: ROS GITHUB REPOSITORIES")
    print(f"  Study end date : {STUDY_END_DATE.date()}")
    print(f"  Dead threshold : {DEAD_THRESHOLD_DAYS} days")
    print("=" * 65)

    df, q1, q3 = load_data()

    # --- Kaplan-Meier ---
    print("\n" + "=" * 65)
    print("KAPLAN-MEIER ANALYSIS")
    print("=" * 65)
    plot_km_overall(df)
    plot_km_by_group(df, "owner_type",    "Kaplan-Meier: Owner Type",
                    "km_by_owner_type.png",    color_map={"Organization": "#1f77b4", "User": "#ff7f0e"},
                    group_order=["Organization", "User"])
    plot_km_by_group(df, "community_tier", "Kaplan-Meier: Community Tier",
                    "km_by_community_size.png", color_map=COLORS, group_order=TIER_ORDER)

    # --- Log-Rank Tests ---
    print("\n" + "=" * 65)
    print("LOG-RANK TESTS")
    print("=" * 65)
    run_logrank_tests(df)

    # --- Cox PH ---
    print("\n" + "=" * 65)
    print("COX PROPORTIONAL HAZARDS MODEL")
    print("=" * 65)
    _, _, c_index_test = fit_cox(df)
    plot_hazard_ratios(df, c_index_test)

    # --- RSF ---
    print("\n" + "=" * 65)
    print("RANDOM SURVIVAL FOREST")
    print("=" * 65)
    run_rsf(df)

    # --- Summary Tables ---
    print("\n" + "=" * 65)
    print("SUMMARY TABLES")
    print("=" * 65)
    build_survival_probabilities(df)
    build_dataset_summary(df, q1, q3)

    # --- Final report ---
    print("\n" + "=" * 65)
    print("OUTPUTS")
    print("=" * 65)
    for label, d in [("Tables", TABLES_DIR), ("Plots", PLOTS_DIR)]:
        files = sorted(d.glob("*.csv" if label == "Tables" else "*.png"))
        print(f"\n{label} ({len(files)}):")
        for f in files:
            print(f"  {f.relative_to(PROJECT_ROOT)}")
    print("\nDone.")


if __name__ == "__main__":
    main()
