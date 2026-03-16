# survival_new_dataset.py — Survival analysis for survival_dataset.csv
# Adapted from survival.py to work with pre-computed survival features

import numpy as np
import pandas as pd
from pathlib import Path

from sksurv.util import Surv
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from lifelines import KaplanMeierFitter
from lifelines import CoxPHFitter
from lifelines.statistics import logrank_test
try:
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
import matplotlib.pyplot as plt

# =========================
# 0) CONFIG
# =========================
PROJECT_ROOT = Path(__file__).parent.parent
CSV_PATH = PROJECT_ROOT / "out" / "survival_dataset_complete.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.25

RSF_N_ESTIMATORS = 200

# Cox regularization
COX_ALPHA = 1e-2

# OneHot category control
OHE_MIN_FREQUENCY = 3
OHE_MAX_CATEGORIES = 30


# =========================
# 1) IO HELPERS
# =========================
def read_csv_smart(path: str) -> pd.DataFrame:
    for enc in ("utf-8", "cp1252", "latin1"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            pass
    return pd.read_csv(path, encoding="latin1", encoding_errors="replace")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.replace(" ", "_")
        .str.replace("-", "_")
    )
    return df


def ensure_bool01(df: pd.DataFrame, col: str) -> None:
    if col not in df.columns:
        return
    if df[col].dtype == bool:
        df[col] = df[col].astype(int)
        return
    df[col] = (
        df[col]
        .astype(str)
        .str.strip()
        .str.lower()
        .replace({"true": "1", "false": "0", "yes": "1", "no": "0", "nan": np.nan})
    )
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)


def km_survival_at(time: np.ndarray, surv: np.ndarray, t_point: float) -> float:
    idx = np.searchsorted(time, t_point, side="right") - 1
    return float(surv[idx]) if idx >= 0 else 1.0


# =========================
# 2) LOAD + CLEAN
# =========================
df = read_csv_smart(CSV_PATH)
df = normalize_columns(df)

print("\n=== Columns in dataset ===")
print(df.columns.tolist())

# Check required columns
required = ["time_to_event_months", "event_dead"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns in CSV: {missing}")

# Ensure boolean columns
for bcol in [
    "has_readme",
    "has_contributing",
    "has_code_of_conduct",
    "has_newcomer_labels",
]:
    ensure_bool01(df, bcol)

# Clean data
df["time_months"] = pd.to_numeric(df["time_to_event_months"], errors="coerce")
df["event"] = pd.to_numeric(df["event_dead"], errors="coerce").fillna(0).astype(int)

df = df.dropna(subset=["time_months"]).copy()
df = df[df["time_months"] > 0].copy()

# Convert event to boolean for sksurv
df["event_dead"] = df["event"].astype(bool)

print("\n=== Dataset summary ===")
print("N repos:", len(df))
print(f"Dead count (event=1):", int(df["event_dead"].sum()))
print("Dead rate:", float(df["event_dead"].mean()))
print("\ntime_months summary:\n", df["time_months"].describe())

if "owner_type" in df.columns:
    print("\nDead by owner_type:\n", df.groupby("owner_type")["event_dead"].sum())
    print("\nDead rate by owner_type:\n", df.groupby("owner_type")["event_dead"].mean())

y = Surv.from_arrays(event=df["event_dead"].values, time=df["time_months"].values)


# =========================
# 3) KAPLAN–MEIER
# =========================
print("\n=== Kaplan–Meier overall (S@6/12/24 months) ===")
t = df["time_months"].values
e = df["event_dead"].values

time_km, surv_km = kaplan_meier_estimator(e, t)

s6 = km_survival_at(time_km, surv_km, 6)
s12 = km_survival_at(time_km, surv_km, 12)
s24 = km_survival_at(time_km, surv_km, 24)

print(f"Overall | n={len(df):4d} | S(6mo)={s6:.3f}  S(12mo)={s12:.3f}  S(24mo)={s24:.3f}")

if "owner_type" in df.columns:
    print("\n=== Kaplan–Meier by owner_type ===")
    for v in sorted(df["owner_type"].unique()):
        mask = (df["owner_type"] == v).values
        if mask.sum() < 5:
            continue

        t = df.loc[mask, "time_months"].values
        e = df.loc[mask, "event_dead"].values

        time, surv = kaplan_meier_estimator(e, t)

        s6 = km_survival_at(time, surv, 6)
        s12 = km_survival_at(time, surv, 12)
        s24 = km_survival_at(time, surv, 24)

        print(f"{v:>12} | n={mask.sum():4d} | S(6mo)={s6:.3f}  S(12mo)={s12:.3f}  S(24mo)={s24:.3f}")

    # Log-Rank Test
    print("\n=== Log-Rank Test (Organization vs User) ===")
    mask_org = df["owner_type"] == "Organization"
    mask_user = df["owner_type"] == "User"
    
    results = logrank_test(
        df.loc[mask_org, "time_months"],
        df.loc[mask_user, "time_months"],
        event_observed_A=df.loc[mask_org, "event_dead"],
        event_observed_B=df.loc[mask_user, "event_dead"]
    )
    print(f"Test statistic: {results.test_statistic:.3f}")
    print(f"p-value: {results.p_value:.4f}")
    if results.p_value < 0.05:
        print("→ SIGNIFICANT difference between Organization and User survival curves")
    else:
        print("→ No significant difference")


# =========================
# 4) FEATURES
# =========================
# Using RATES/FREQUENCIES instead of cumulative counts
# to avoid survival bias (Robinson et al. critique of "Cheating Death")
# Rates normalize by project age, allowing fair comparison between
# young and old projects

candidate_num = [
    # Rate/average features (normalized by time) - PREFERRED
    "Average_number_of_commits_per_month",
    "Average_number_of_newcomers_per_month",
    "Average_number_of_forks_per_month",
    "Average_number_of_stars_per_month",
    # Time metrics (already normalized)
    # "Average_time_to_close_a_pull_request_(days)",
    # "Average_time_to_close_an_issue_(days)",
    # Documentation (binary - no bias)
    "has_readme",
    "has_contributing",
    "has_code_of_conduct",
    "has_pr_template",
    "has_issue_template",
    "has_newcomer_labels",
    # "Size_of_README_(KB)",
    # "Size_of_CONTRIBUTING_(KB)",
]

candidate_cat = [
    "owner_type",
    "License",
]

num_cols = [c for c in candidate_num if c in df.columns]
cat_cols = [c for c in candidate_cat if c in df.columns]

print(f"\nNumeric features: {num_cols}")
print(f"Categorical features: {cat_cols}")

for c in num_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

for c in cat_cols:
    df[c] = df[c].astype(str).fillna("NA")

X = df[num_cols + cat_cols].copy()


# =========================
# 4b) COX WITH LIFELINES (proper HR interpretation with CI and p-values)
# =========================
print("\n" + "="*70)
print("=== CoxPH with lifelines (interpretable HRs with CI & p-values) ===")
print("="*70)

# Prepare data for lifelines (no transformation - raw values for interpretation)
cox_df = df[num_cols + ["time_months", "event_dead"]].copy()

# Add owner_type as binary
if "owner_type" in df.columns:
    cox_df["is_user"] = (df["owner_type"] == "User").astype(int)

# Fits Cox model
cph = CoxPHFitter(penalizer=0.01)  # L2 regularization
cph.fit(cox_df, duration_col="time_months", event_col="event_dead")

print("\n--- Summary with Hazard Ratios, 95% CI, and p-values ---")
print(cph.summary[["coef", "exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]].to_string())

print("\n--- Significant features (p < 0.05) ---")
significant = cph.summary[cph.summary["p"] < 0.05][["exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]]
if len(significant) > 0:
    print(significant.to_string())
else:
    print("No features with p < 0.05")

# Check Proportional Hazards assumption
print("\n--- Proportional Hazards Test (Schoenfeld residuals) ---")
try:
    ph_test = cph.check_assumptions(cox_df, p_value_threshold=0.05, show_plots=False)
    print("PH assumption check completed (see warnings above if any)")
except Exception as e:
    print(f"PH test error: {e}")

# Concordance index
print(f"\nC-index (lifelines, full data): {cph.concordance_index_:.3f}")

# Multicollinearity check (VIF)
print("\n--- Variance Inflation Factor (multicollinearity) ---")

if HAS_STATSMODELS:
    # Only numeric cols for VIF
    vif_df = cox_df[num_cols].dropna()
    if len(vif_df) > 0:
        vif_data = pd.DataFrame()
        vif_data["feature"] = num_cols
        vif_data["VIF"] = [variance_inflation_factor(vif_df.values, i) for i in range(len(num_cols))]
        print(vif_data.to_string(index=False))
        high_vif = vif_data[vif_data["VIF"] > 5]
        if len(high_vif) > 0:
            print("\n⚠️  WARNING: Features with VIF > 5 (potential multicollinearity):")
            print(high_vif.to_string(index=False))
else:
    print("(statsmodels not installed - skipping VIF calculation)")


# =========================
# 4c) STRATIFIED COX (fix PH violations for binary variables)
# =========================
print("\n" + "="*70)
print("=== Stratified CoxPH (fixing PH violations) ===")
print("="*70)
print("Stratifying on: has_newcomer_labels, has_contributing, has_readme")

# Variables that violated PH and are binary -> use as strata
# Variables that are continuous and violated -> need different approach
strata_vars = ["has_newcomer_labels", "has_contributing", "has_readme"]

# Non-stratified covariates (continuous ones)
cox_df_strat = df[num_cols + ["time_months", "event_dead"]].copy()
if "owner_type" in df.columns:
    cox_df_strat["is_user"] = (df["owner_type"] == "User").astype(int)

# Add strata columns
for sv in strata_vars:
    cox_df_strat[sv] = df[sv].astype(int)

# Fit stratified Cox
cph_strat = CoxPHFitter(penalizer=0.01)
cph_strat.fit(
    cox_df_strat, 
    duration_col="time_months", 
    event_col="event_dead",
    strata=strata_vars
)

print("\n--- Stratified Cox Summary ---")
print(cph_strat.summary[["coef", "exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]].to_string())

print(f"\nC-index (stratified): {cph_strat.concordance_index_:.3f}")

# Check PH again on stratified model
print("\n--- PH Test on Stratified Model ---")
try:
    cph_strat.check_assumptions(cox_df_strat, p_value_threshold=0.05, show_plots=False)
except Exception as e:
    print(f"Note: {e}")


# =========================
# 4d) AFT MODEL (alternative that doesn't assume PH)
# =========================
from lifelines import WeibullAFTFitter, LogNormalAFTFitter

print("\n" + "="*70)
print("=== AFT Models (Accelerated Failure Time - no PH assumption) ===")
print("="*70)

# Prepare data for AFT
aft_df = df[num_cols + ["time_months", "event_dead"]].copy()
if "owner_type" in df.columns:
    aft_df["is_user"] = (df["owner_type"] == "User").astype(int)

# Weibull AFT
print("\n--- Weibull AFT ---")
waft = WeibullAFTFitter(penalizer=0.01)
waft.fit(aft_df, duration_col="time_months", event_col="event_dead")

print("Coefficients (positive = longer survival, negative = shorter survival):")
print(waft.summary[["coef", "exp(coef)", "p"]].head(15).to_string())
print(f"\nC-index (Weibull AFT): {waft.concordance_index_:.3f}")
print(f"AIC: {waft.AIC_:.1f}")

# Log-Normal AFT
print("\n--- LogNormal AFT ---")
lnaft = LogNormalAFTFitter(penalizer=0.01)
lnaft.fit(aft_df, duration_col="time_months", event_col="event_dead")

print("Coefficients (positive = longer survival, negative = shorter survival):")
print(lnaft.summary[["coef", "exp(coef)", "p"]].head(15).to_string())
print(f"\nC-index (LogNormal AFT): {lnaft.concordance_index_:.3f}")
print(f"AIC: {lnaft.AIC_:.1f}")

# Compare models
print("\n" + "="*70)
print("=== Model Comparison ===")
print("="*70)
print(f"{'Model':<25} {'C-index':>10} {'AIC':>12}")
print("-" * 50)
print(f"{'CoxPH (basic)':<25} {cph.concordance_index_:>10.3f} {'N/A':>12}")
print(f"{'CoxPH (stratified)':<25} {cph_strat.concordance_index_:>10.3f} {'N/A':>12}")
print(f"{'Weibull AFT':<25} {waft.concordance_index_:>10.3f} {waft.AIC_:>12.1f}")
print(f"{'LogNormal AFT':<25} {lnaft.concordance_index_:>10.3f} {lnaft.AIC_:>12.1f}")
print("\n→ Lower AIC = better fit. Higher C-index = better discrimination.")

# Best model interpretation
best_aft = waft if waft.AIC_ < lnaft.AIC_ else lnaft
best_name = "Weibull" if waft.AIC_ < lnaft.AIC_ else "LogNormal"
print(f"\n→ Best AFT model: {best_name} (AIC = {best_aft.AIC_:.1f})")

print("\n--- Final Interpretation (from best AFT model) ---")
print("Acceleration Factors (AF): AF > 1 = longer survival, AF < 1 = shorter survival")
af_summary = best_aft.summary[["coef", "exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]]
# Filter to mu_ parameters only (not lambda/sigma)
af_summary = af_summary[af_summary.index.get_level_values(0) == "mu_"]
af_summary = af_summary.sort_values("p")
print(af_summary.to_string())


# =========================
# 5) TRAIN/TEST SPLIT
# =========================
n_events = int(df["event_dead"].sum())
n_total = len(df)
n_censored = n_total - n_events

print("\n=== Split feasibility check ===")
print("Events:", n_events, "| Censored:", n_censored, "| Total:", n_total)

if n_events == 0 or n_censored == 0:
    print("\n[STOP] CoxPH/RSF cannot be trained (all censored or all events).")
    raise SystemExit(0)

stratify_vec = df["event_dead"].astype(int)
min_class = min(int((stratify_vec == 0).sum()), int((stratify_vec == 1).sum()))
use_stratify = min_class >= 2

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=stratify_vec if use_stratify else None,
)

print("\n=== Split check ===")
print("Train dead:", int(y_train["event"].sum()), "of", len(y_train))
print("Test  dead:", int(y_test["event"].sum()), "of", len(y_test))


# =========================
# 6) PREPROCESSING (Cox-stable)
# =========================
def safe_log1p(Xarr):
    Xarr = np.asarray(Xarr, dtype=float)
    Xarr = np.nan_to_num(Xarr, nan=0.0, posinf=0.0, neginf=0.0)
    Xarr = np.clip(Xarr, 0.0, None)
    return np.log1p(Xarr)

numeric_pipe = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("log1p", FunctionTransformer(safe_log1p, validate=False)),
    ("scaler", StandardScaler()),
])

try:
    ohe = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False,
        min_frequency=OHE_MIN_FREQUENCY,
        max_categories=OHE_MAX_CATEGORIES,
    )
except TypeError:
    ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

categorical_pipe = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", ohe),
])

if cat_cols:
    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, num_cols),
            ("cat", categorical_pipe, cat_cols),
        ],
        remainder="drop",
    )
else:
    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, num_cols),
        ],
        remainder="drop",
    )


# =========================
# 7) COXPH
# =========================
cox = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", CoxPHSurvivalAnalysis(alpha=COX_ALPHA)),
])

cox.fit(X_train, y_train)
cox_risk = cox.predict(X_test)
cox_cindex = concordance_index_censored(y_test["event"], y_test["time"], cox_risk)[0]

print("\n=== CoxPH ===")
print(f"alpha: {COX_ALPHA}")
print(f"Features used: {len(num_cols)} numeric/boolean + {len(cat_cols)} categorical")
print(f"C-index (test): {cox_cindex:.3f}")

# Extract Hazard Ratios
cox_model = cox.named_steps["model"]
coefs = cox_model.coef_

# Get feature names after transformation
feature_names = []
# Numeric features (same names after transform)
feature_names.extend(num_cols)
# Categorical features (one-hot encoded)
if cat_cols:
    ohe_fitted = cox.named_steps["preprocess"].named_transformers_["cat"].named_steps["onehot"]
    cat_feature_names = ohe_fitted.get_feature_names_out(cat_cols)
    feature_names.extend(cat_feature_names)

hazard_ratios = np.exp(coefs)

hr_df = pd.DataFrame({
    "feature": feature_names,
    "coef": coefs,
    "hazard_ratio": hazard_ratios,
    "HR_interpretation": ["↑ risk" if hr > 1 else "↓ risk" for hr in hazard_ratios]
}).sort_values("hazard_ratio", ascending=False)

print("\n=== Hazard Ratios (CoxPH) ===")
print("HR > 1: increases risk of death | HR < 1: decreases risk (protective)")
print(hr_df.to_string(index=False))


# =========================
# 8) RANDOM SURVIVAL FOREST
# =========================
rsf = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", RandomSurvivalForest(
        n_estimators=RSF_N_ESTIMATORS,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features="sqrt",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )),
])

rsf.fit(X_train, y_train)
rsf_risk = rsf.predict(X_test)
rsf_cindex = concordance_index_censored(y_test["event"], y_test["time"], rsf_risk)[0]

print("\n=== RSF ===")
print(f"n_estimators: {RSF_N_ESTIMATORS}")
print(f"C-index (test): {rsf_cindex:.3f}")

print("\nDone with models.")


# =========================
# 9) PLOTS
# =========================
from sklearn.inspection import permutation_importance
import os

# Create output directory for plots
PLOT_DIR = "survival_plots"
os.makedirs(PLOT_DIR, exist_ok=True)

# KM curve - overall
plt.figure(figsize=(10, 6))
kmf = KaplanMeierFitter()
kmf.fit(df["time_months"], event_observed=df["event_dead"], label="All repositories")
kmf.plot_survival_function(ci_show=True)

plt.xlabel("Time (months)")
plt.ylabel("Estimated survival probability S(t)")
plt.title("Kaplan–Meier Survival Curve (95% CI)")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/km_overall.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {PLOT_DIR}/km_overall.png")

# KM by owner_type
if "owner_type" in df.columns:
    plt.figure(figsize=(10, 6))
    for v in sorted(df["owner_type"].dropna().unique()):
        mask = df["owner_type"] == v
        if mask.sum() < 5:
            continue
        kmf = KaplanMeierFitter()
        kmf.fit(
            df.loc[mask, "time_months"],
            event_observed=df.loc[mask, "event_dead"],
            label=f"{v} (n={mask.sum()})"
        )
        kmf.plot_survival_function(ci_show=True)

    plt.xlabel("Time (months)")
    plt.ylabel("Estimated survival probability S(t)")
    plt.title("Kaplan–Meier Survival by owner_type (95% CI)")
    plt.ylim(0, 1.02)
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/km_by_owner_type.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {PLOT_DIR}/km_by_owner_type.png")

# Event time distribution
plt.figure(figsize=(10, 6))
dead = df[df["event_dead"]]
if len(dead) > 0:
    plt.hist(dead["time_months"], bins=25, edgecolor='black', alpha=0.7)
    plt.xlabel("Event time (months) [dead repos only]")
    plt.ylabel("Count")
    plt.title("Distribution of event times (when repos died)")
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/event_time_dist.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {PLOT_DIR}/event_time_dist.png")

# Hazard Ratios plot
plt.figure(figsize=(12, 10))
hr_sorted = hr_df.sort_values("hazard_ratio")
colors = ['green' if hr < 1 else 'red' for hr in hr_sorted["hazard_ratio"]]
plt.barh(hr_sorted["feature"], hr_sorted["hazard_ratio"], color=colors, alpha=0.7)
plt.axvline(x=1, color='black', linestyle='--', linewidth=1.5, label='HR=1 (no effect)')
plt.xlabel("Hazard Ratio")
plt.title("Hazard Ratios (CoxPH)\nGreen: Protective (HR<1) | Red: Risk factor (HR>1)")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/hazard_ratios.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {PLOT_DIR}/hazard_ratios.png")

# Permutation importance
def perm_importance_rawcols(pipeline, X_test, y_test, raw_cols, n_repeats=15, random_state=42):
    def cindex_scorer(estimator, X, y):
        risk = estimator.predict(X)
        return concordance_index_censored(y["event"], y["time"], risk)[0]

    result = permutation_importance(
        pipeline,
        X_test[raw_cols],
        y_test,
        scoring=cindex_scorer,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1,
    )
    imp = pd.DataFrame({
        "feature": raw_cols,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    }).sort_values("importance_mean", ascending=False)
    return imp

raw_cols = list(X.columns)

# Cox importance
cox_imp = perm_importance_rawcols(cox, X_test, y_test, raw_cols)
print("\n=== Permutation importance (CoxPH, test) ===")
print(cox_imp.to_string(index=False))

plt.figure(figsize=(12, 10))
top = cox_imp[::-1]
plt.barh(top["feature"], top["importance_mean"], xerr=top["importance_std"], alpha=0.7)
plt.xlabel("Permutation importance (Δ C-index)")
plt.title("Feature importance (CoxPH) — test set")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/cox_importance.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {PLOT_DIR}/cox_importance.png")

# RSF importance
rsf_imp = perm_importance_rawcols(rsf, X_test, y_test, raw_cols)
print("\n=== Permutation importance (RSF, test) ===")
print(rsf_imp.to_string(index=False))

plt.figure(figsize=(12, 10))
top = rsf_imp[::-1]
plt.barh(top["feature"], top["importance_mean"], xerr=top["importance_std"], alpha=0.7)
plt.xlabel("Permutation importance (Δ C-index)")
plt.title("Feature importance (RSF) — test set")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/rsf_importance.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {PLOT_DIR}/rsf_importance.png")

print(f"\nAll plots saved to '{PLOT_DIR}/' folder.")
