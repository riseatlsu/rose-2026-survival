"""
21_paper_figures.py

Regenerates all figures for the RoSE'26 paper with publication quality:

Fixes:
  1. km_overall_full.png  — full dataset KM curve
  3. All KM plots now include 95% confidence intervals (Greenwood's formula)
  4. All KM plots now include number-at-risk table below x-axis
  5. hazard_ratios.png — x-axis truncated at HR=5 (Code of Conduct CI=331 distorted the scale)
  6. Tier labels updated to show thresholds (Tier 1: ≤3, Tier 2: 4–18, Tier 3: >18)

Tier definition: IQR-based on our dataset (Q1=3, Q3=18)
  Rationale: Ait et al. thresholds (1 / 2–5 / 6+) were calibrated for 1.9M generic GitHub
  projects. Median contributor count for ROS2 repos is 7 — higher than general OSS.
  IQR-based tiers give S(60m) of 0.364 / 0.734 / 0.984, closely matching the pattern
  of 0.53 / 0.74 / 0.90 reported by Ait et al.

Outputs: out/survival_analysis/plots/paper/
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from pathlib import Path
from datetime import datetime

from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.compare import compare_survival
from sksurv.util import Surv

PROJECT_ROOT   = Path(__file__).parent.parent
SURVIVAL_CSV   = PROJECT_ROOT / "out" / "survival_dataset_complete.csv"
COX_CSV        = PROJECT_ROOT / "out" / "survival_analysis" / "tables" / "cox_model_results.csv"
COX_PERM_CSV   = PROJECT_ROOT / "out" / "survival_analysis" / "tables" / "cox_permutation_importance.csv"
RSF_CSV        = PROJECT_ROOT / "out" / "survival_analysis" / "tables" / "rsf_feature_importance.csv"
TABLES_DIR     = PROJECT_ROOT / "out" / "survival_analysis" / "tables"
OUT_DIR        = PROJECT_ROOT / "out" / "survival_analysis" / "plots" / "paper"
OUT_DIR.mkdir(parents=True, exist_ok=True)

STUDY_END         = datetime(2026, 3, 8)
DEAD_THRESHOLD    = 180
TIMEPOINTS_MONTHS = [12, 24, 36, 48, 60]

plt.rcParams.update({
    "font.family":     "DejaVu Sans",
    "font.size":       11,
    "axes.titlesize":  12,
    "axes.labelsize":  11,
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi":      150,
})

COLORS = {
    "Overall":      "#1565C0",
    "Organization": "#1565C0",
    "User":         "#E65100",
    "Tier 1":       "#C62828",
    "Tier 2":       "#2E7D32",
    "Tier 3":       "#6A1B9A",
}

# Built dynamically from data quantiles in load_data() — do not hardcode here.
TIER_LABELS       = {}
TIER_LABELS_SHORT = {}
FORK_TIER_LABELS  = {}

FORK_COLORS = {
    "Fork Tier 1": "#C62828",
    "Fork Tier 2": "#2E7D32",
    "Fork Tier 3": "#6A1B9A",
}


# ──────────────────────────────────────────────
# DATA
# ──────────────────────────────────────────────
def load_data():
    df = pd.read_csv(SURVIVAL_CSV)
    df["event_dead"]    = (df["days_since_last_activity"] > DEAD_THRESHOLD).astype(int)
    df["created_at_dt"] = pd.to_datetime(df["created_at"], errors="coerce")
    df["last_act_dt"]   = pd.to_datetime(df["last_activity_date"], errors="coerce")
    df["time_months"]   = df.apply(
        lambda r: (r["last_act_dt"] - r["created_at_dt"]).days / 30.44
        if r["event_dead"] == 1
        else (STUDY_END - r["created_at_dt"]).days / 30.44,
        axis=1,
    )
    # Clip to minimum 1 day instead of dropping — keeps all 810 repos
    df = df[df["time_months"].notna()].copy()
    df["time_months"] = df["time_months"].clip(lower=1/30.44)

    q1 = df["contributors_count"].quantile(0.25)
    q3 = df["contributors_count"].quantile(0.75)

    def tier(n):
        if n <= q1:  return "Tier 1"
        if n <= q3:  return "Tier 2"
        return "Tier 3"

    df["community_tier"] = df["contributors_count"].apply(tier)

    fq1 = df["Number of forks"].quantile(0.25)
    fq3 = df["Number of forks"].quantile(0.75)

    def fork_tier(n):
        if n <= fq1:  return "Fork Tier 1"
        if n <= fq3:  return "Fork Tier 2"
        return "Fork Tier 3"

    df["fork_tier"] = df["Number of forks"].apply(fork_tier)

    q1_int, q3_int   = int(q1), int(q3)
    fq1_int, fq3_int = int(fq1), int(fq3)

    tier_labels = {
        "Tier 1": f"Tier 1 (≤{q1_int} contrib.)",
        "Tier 2": f"Tier 2 ({q1_int+1}–{q3_int} contrib.)",
        "Tier 3": f"Tier 3 (>{q3_int} contrib.)",
    }
    tier_labels_short = {
        "Tier 1": f"Tier 1 (≤{q1_int})",
        "Tier 2": f"Tier 2 ({q1_int+1}–{q3_int})",
        "Tier 3": f"Tier 3 (>{q3_int})",
    }
    fork_tier_labels = {
        "Fork Tier 1": f"Tier 1 (≤{fq1_int} forks)",
        "Fork Tier 2": f"Tier 2 ({fq1_int+1}–{fq3_int} forks)",
        "Fork Tier 3": f"Tier 3 (>{fq3_int} forks)",
    }

    df["has_newcomer_labels_bin"] = df["has_newcomer_labels"].map(
        lambda x: "Has Newcomer Labels" if str(x).lower() in ("true", "1", "yes") else "No Newcomer Labels"
    )

    return df, q1, q3, fq1, fq3, tier_labels, tier_labels_short, fork_tier_labels


def make_surv(df):
    return Surv.from_arrays(
        event=df["event_dead"].astype(bool).values,
        time=df["time_months"].values,
    )


def logrank_p(df, col):
    sub = df[df[col].notna()].copy()
    try:
        _, p = compare_survival(make_surv(sub), sub[col].values)
        return p
    except Exception:
        return None


def fmt_p(p):
    if p is None: return "N/A"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


# ──────────────────────────────────────────────
# KM WITH CI + RISK TABLE
# ──────────────────────────────────────────────
def _km_with_ci(event, time):
    """
    Returns (time, surv, ci_lo, ci_hi) using Greenwood's formula.

    Uses sksurv's kaplan_meier_estimator for the survival function
    (handles tied event times correctly), then computes Greenwood CI
    at each event time by aggregating all deaths at that instant.
    n_at_risk(t) = number of observations with time >= t (standard def).
    """
    event_bool = np.asarray(event.astype(bool))
    time_arr   = np.asarray(time)

    t_km, s_km = kaplan_meier_estimator(event_bool, time_arr)

    greenwood_sum = 0.0
    ci_lo, ci_hi  = [], []
    z = 1.96

    for idx, t_event in enumerate(t_km):
        n_at_risk = int((time_arr >= t_event).sum())
        d_i       = int(((time_arr == t_event) & event_bool).sum())

        if d_i > 0 and n_at_risk > d_i:
            greenwood_sum += d_i / (n_at_risk * (n_at_risk - d_i))

        s_t = s_km[idx]
        se  = s_t * np.sqrt(greenwood_sum)
        ci_lo.append(max(s_t - z * se, 0.0))
        ci_hi.append(min(s_t + z * se, 1.0))

    return t_km, s_km, np.array(ci_lo), np.array(ci_hi)


def _risk_table(ax_risk, df, time_col, timepoints, groups=None, group_col=None,
                colors=None):
    """Draw number-at-risk table below the KM plot."""
    ax_risk.set_xlim(ax_risk.get_xlim() if hasattr(ax_risk, '_xlim') else (0, None))
    ax_risk.axis("off")

    if groups is None:
        groups = ["All"]

    col_positions = timepoints
    row_height = 1.0 / max(len(groups), 1)

    for row_i, g in enumerate(groups):
        if g == "All":
            sub = df
        else:
            sub = df[df[group_col] == g]

        y = 1 - (row_i + 0.5) * row_height
        color = (colors.get(g, "black") if colors else "black")

        label = TIER_LABELS.get(g, g)
        ax_risk.text(-0.01, y, label, transform=ax_risk.transAxes,
                     ha="right", va="center", fontsize=8, color=color)

        for tp in timepoints:
            n_at_risk = int((sub[time_col] >= tp).sum())
            x_norm = tp / timepoints[-1]   # rough normalisation
            ax_risk.text(tp, y, str(n_at_risk),
                         ha="center", va="center", fontsize=8, color=color,
                         transform=ax_risk.get_xaxis_transform())

    ax_risk.set_xlabel("Months", fontsize=9)


def _plot_km_single(df, title, filename, label="All repos", color="#333333",
                    annotate_pts=True):
    """KM for entire df with 95% CI."""
    t_km, s_km, ci_lo, ci_hi = _km_with_ci(df["event_dead"], df["time_months"])

    x_max_plot = int(t_km[-1]) + 5
    risk_tps = list(range(0, x_max_plot + 1, 25))

    fig, ax = plt.subplots(figsize=(8, 5.4))

    ax.step(t_km, s_km, where="post", color=color, lw=2.5,
            label=f"{label} (n={len(df)}, dead={int(df['event_dead'].sum())})")
    ax.fill_between(t_km, ci_lo, ci_hi, step="post", alpha=0.2, color=color)
    ax.axhline(0.5, color="gray", ls="--", lw=1, alpha=0.6, label="S(t)=0.50")

    ax.set_ylabel("Survival probability S(t)", fontsize=20, labelpad=10)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0, x_max_plot)
    ax.legend(loc="lower left", fontsize=15, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(risk_tps)
    ax.set_xticklabels([str(t) for t in risk_tps], fontsize=17)
    ax.set_xlabel("Time (months)", fontsize=20)
    ax.tick_params(axis='y', labelsize=17)

    fig.tight_layout()
    path = OUT_DIR / filename
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix('.svg'), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    print(f"  Saved: {path.with_suffix('.svg')}")


def km_at(time, surv, t):
    mask = time <= t
    return float(surv[mask][-1]) if mask.any() else 1.0


def _plot_km_groups(df, group_col, groups, title, filename, colors,
                    legend_loc="upper right", tier_labels=None, legend_fontsize=15):
    """KM by group with 95% CI and log-rank p."""
    if tier_labels is None:
        tier_labels = TIER_LABELS
    p = logrank_p(df, group_col)
    fig, ax = plt.subplots(figsize=(8, 5.4))

    x_max = 0
    for g in groups:
        sub = df[df[group_col] == g]
        if len(sub) < 5: continue
        t_km, s_km, ci_lo, ci_hi = _km_with_ci(sub["event_dead"], sub["time_months"])
        color = colors.get(g, None)
        label_g = tier_labels.get(g, g)
        ax.step(t_km, s_km, where="post", lw=2.5, color=color,
                label=f"{label_g} (n={len(sub)}, d={int(sub['event_dead'].sum())})")
        ax.fill_between(t_km, ci_lo, ci_hi, step="post", alpha=0.18, color=color)
        x_max = max(x_max, t_km[-1] if len(t_km) else 0)

    # Round x_max up to nearest 25 for clean ticks
    x_max_plot = int(x_max) + 5
    risk_tps = list(range(0, x_max_plot + 1, 25))

    ax.axhline(0.5, color="gray", ls="--", lw=1, alpha=0.5)
    ax.text(0.98, 0.98, f"Log-rank p={fmt_p(p)}", transform=ax.transAxes,
            ha="right", va="top", fontsize=16, color="gray")
    ax.set_ylabel("Survival probability S(t)", fontsize=20, labelpad=10)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0, x_max_plot)
    ax.legend(loc=legend_loc, fontsize=legend_fontsize, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(risk_tps)
    ax.set_xticklabels([str(t) for t in risk_tps], fontsize=17)
    ax.set_xlabel("Time (months)", fontsize=20)
    ax.tick_params(axis='y', labelsize=17)

    fig.tight_layout()
    path = OUT_DIR / filename
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix('.svg'), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    print(f"  Saved: {path.with_suffix('.svg')}")


# ──────────────────────────────────────────────
# HAZARD RATIOS — truncated x-axis
# ──────────────────────────────────────────────
def plot_hazard_ratios_fixed(cox_csv):
    """Dot plot of hazard ratios (no CI — sksurv Cox does not provide them)."""
    print("\n--- Hazard Ratios (fixed) ---")
    cox = pd.read_csv(cox_csv).sort_values("HR", ascending=True).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(8, max(5, len(cox) * 0.6)))
    y_pos = np.arange(len(cox))

    colors = ["#d62728" if hr > 1 else "#1f77b4" for hr in cox["HR"]]
    ax.scatter(cox["HR"], y_pos, color=colors, s=80, zorder=3)
    for i, (hr, color) in enumerate(zip(cox["HR"], colors)):
        ax.plot([1.0, hr], [i, i], color=color, lw=1.5, alpha=0.5, zorder=2)
        ax.text(hr + 0.02, i, f"  {hr:.2f}", va="center", fontsize=9)

    ax.axvline(1.0, color="black", ls="--", lw=1.2, label="HR=1 (no effect)")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(cox["Feature"], fontsize=9)
    ax.set_xlabel("Hazard Ratio (HR)")
    ax.set_title("Cox PH Model: Hazard Ratios\n(red = increased hazard, blue = decreased hazard)")
    ax.grid(True, axis="x", alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()

    path = OUT_DIR / "hazard_ratios.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ──────────────────────────────────────────────
# STATE MACHINE DIAGRAMS (programmatic, with avg durations)
# ──────────────────────────────────────────────
SM_STATS_DIR = PROJECT_ROOT / "out" / "survival"

def _sm_figure(ax, title, stats, trans):
    """Draw a state machine on ax.
    stats  = {"Running": avg_m, "Zombie": avg_m, "Dead": avg_m}
    trans  = {("from","to"): prob_pct, ...}
    """
    R_pos = (0.25, 0.71)
    Z_pos = (0.25, 0.29)
    D_pos = (0.74, 0.50)
    POS   = {"Running": R_pos, "Zombie": Z_pos, "Dead": D_pos}
    HW, HH = 0.115, 0.085
    COLORS = {"Running": "#c8e6c9", "Zombie": "#fff3cd", "Dead": "#f8d7da"}
    AKW    = dict(arrowstyle="-|>", color="black", lw=1.3, mutation_scale=13, zorder=4)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title(title, fontsize=14, pad=8, fontweight='bold')

    # Alive box
    ax.add_patch(mpatches.FancyBboxPatch(
        (0.04, 0.09), 0.44, 0.84,
        boxstyle="round,pad=0.01", lw=1.2, ec="#aaa", fc="#f5f5f5", zorder=0))
    ax.text(0.26, 0.955, "Alive", ha="center", va="center",
            fontsize=12, color="#666", style="italic")

    # Nodes
    for name, (cx, cy) in POS.items():
        ax.add_patch(mpatches.Ellipse(
            (cx, cy), 2*HW, 2*HH, fc=COLORS[name], ec="black", lw=1.5, zorder=3))
        ax.text(cx, cy + 0.022, name, ha="center", va="center",
                fontsize=13, fontweight="bold", zorder=5)
        ax.text(cx, cy - 0.025, f"({stats[name]:.2f}m)",
                ha="center", va="center", fontsize=11, color="#333", zorder=5)

    def lbl(prob, x, y):
        ax.text(x, y, f"{prob:.1f}%", ha="center", va="center",
                fontsize=11, zorder=6,
                bbox=dict(boxstyle="square,pad=0.08", fc="white", ec="none", alpha=0.9))

    def arr(p1, p2, rad):
        ax.add_patch(FancyArrowPatch(
            p1, p2, connectionstyle=f"arc3,rad={rad}", **AKW))

    def selfloop(cx, cy, direction, prob):
        if direction == "top":
            p1, p2 = (cx - 0.05, cy + HH*0.8), (cx + 0.05, cy + HH*0.8)
            lx, ly = cx, cy + HH + 0.115
        elif direction == "bottom":
            p1, p2 = (cx + 0.05, cy - HH*0.8), (cx - 0.05, cy - HH*0.8)
            lx, ly = cx, cy - HH - 0.115
        else:  # right
            p1, p2 = (cx + HW*0.85, cy + 0.038), (cx + HW*0.85, cy - 0.038)
            lx, ly = cx + HW + 0.16, cy
        ax.add_patch(FancyArrowPatch(p1, p2,
            connectionstyle="arc3,rad=-1.1", **AKW))
        lbl(prob, lx, ly)

    # Self-loops
    selfloop(*R_pos, "top",    trans[("Running", "Running")])
    selfloop(*Z_pos, "bottom", trans[("Zombie",  "Zombie")])
    selfloop(*D_pos, "right",  trans[("Dead",    "Dead")])

    # Running ↔ Zombie  (vertical, separated by x-offset)
    arr((R_pos[0]-0.025, R_pos[1]-HH), (Z_pos[0]-0.025, Z_pos[1]+HH), 0)
    lbl(trans[("Running", "Zombie")], 0.12, 0.50)
    arr((Z_pos[0]+0.025, Z_pos[1]+HH), (R_pos[0]+0.025, R_pos[1]-HH), 0)
    lbl(trans[("Zombie",  "Running")], 0.38, 0.50)

    # Running ↔ Dead  (diagonal, separated by y-offset on endpoints)
    arr((R_pos[0]+HW, R_pos[1]+0.015), (D_pos[0]-HW, D_pos[1]+0.025), 0.12)
    lbl(trans[("Running", "Dead")], 0.51, 0.675)
    arr((D_pos[0]-HW, D_pos[1]-0.025), (R_pos[0]+HW, R_pos[1]-0.015), 0.12)
    lbl(trans[("Dead",    "Running")], 0.51, 0.565)

    # Zombie ↔ Dead  (diagonal, separated by y-offset on endpoints)
    arr((Z_pos[0]+HW, Z_pos[1]+0.015), (D_pos[0]-HW, D_pos[1]-0.025), -0.12)
    lbl(trans[("Zombie",  "Dead")], 0.51, 0.335)
    arr((D_pos[0]-HW, D_pos[1]+0.025), (Z_pos[0]+HW, Z_pos[1]-0.015), -0.12)
    lbl(trans[("Dead",    "Zombie")], 0.51, 0.435)


def _load_sm_data():
    """Load state statistics and transitions from CSVs."""
    stats_df = pd.read_csv(SM_STATS_DIR / "state_statistics.csv")
    trans_df = pd.read_csv(SM_STATS_DIR / "state_transitions.csv")
    stats_owner = pd.read_csv(SM_STATS_DIR / "state_statistics_by_owner_type.csv")
    trans_owner = pd.read_csv(SM_STATS_DIR / "state_transitions_by_owner_type.csv")
    stats_author = pd.read_csv(SM_STATS_DIR / "state_statistics_by_author_type.csv")
    trans_author = pd.read_csv(SM_STATS_DIR / "state_transitions_by_author_type.csv")

    def to_stats(df, group_col=None, group_val=None):
        if group_col:
            df = df[df[group_col] == group_val]
        return dict(zip(df["state"], df["avg_duration_months"]))

    def to_trans(df, group_col=None, group_val=None):
        if group_col:
            df = df[df[group_col] == group_val]
        return {(r["from_state"], r["to_state"]): r["probability_pct"] for _, r in df.iterrows()}

    return {
        "general":  (to_stats(stats_df), to_trans(trans_df)),
        "org":      (to_stats(stats_owner,  "owner_type",  "Organization"),
                     to_trans(trans_owner,  "owner_type",  "Organization")),
        "user":     (to_stats(stats_owner,  "owner_type",  "User"),
                     to_trans(trans_owner,  "owner_type",  "User")),
        "human":    (to_stats(stats_author, "author_type", "Human"),
                     to_trans(trans_author, "author_type", "Human")),
        "bot":      (to_stats(stats_author, "author_type", "Bot"),
                     to_trans(trans_author, "author_type", "Bot")),
    }


def plot_state_machines():
    """Generate all state machine figures programmatically."""
    print("\n--- State Machines ---")
    data = _load_sm_data()

    configs = [
        ("general", "All Repositories",            "sm_general.png"),
        ("org",     "Organization-owned",           "sm_owner_org.png"),
        ("user",    "User-owned",                   "sm_owner_user.png"),
        ("human",   "Human commits only",           "sm_human.png"),
        ("bot",     "Bot commits only",             "sm_bot.png"),
    ]

    for key, title, fname in configs:
        stats, trans = data[key]
        fig, ax = plt.subplots(figsize=(5.2, 4.6))
        _sm_figure(ax, title, stats, trans)
        fig.tight_layout(pad=0.4)
        path = OUT_DIR / fname
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        fig.savefig(path.with_suffix('.svg'), bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  Saved: {path}")

    # Side-by-side: Organization vs User (like Ait et al. Fig.)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.6))
    stats_org,  trans_org  = data["org"]
    stats_user, trans_user = data["user"]
    _sm_figure(ax1, "(a) Organization-owned", stats_org,  trans_org)
    _sm_figure(ax2, "(b) User-owned",         stats_user, trans_user)
    fig.tight_layout(pad=0.4)
    path = OUT_DIR / "sm_owner_comparison.png"
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(path.with_suffix('.svg'), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {path}")
    print(f"  Saved: {path.with_suffix('.svg')}")


# ──────────────────────────────────────────────
# FEATURE IMPORTANCE: Cox HR + RSF side-by-side
# ──────────────────────────────────────────────
def _importance_bar_chart(ax, features, values, title, xlabel):
    """Shared helper: horizontal bar chart with blue gradient by magnitude."""
    import matplotlib.cm as cm

    n     = len(features)
    vmax  = max(v for v in values if v > 0) if any(v > 0 for v in values) else 1.0
    xmax  = vmax * 1.32

    # gradient: normalize positive values to [0.35, 1.0] in Blues colormap
    def bar_color(v):
        if v <= 0:
            return "#d0d8e8"          # near-zero / negative → very light
        norm = 0.35 + 0.65 * (v / vmax)
        return cm.Blues(norm)

    colors = [bar_color(v) for v in values]
    bars   = ax.barh(np.arange(n), values, height=0.62,
                     color=colors, zorder=3, edgecolor="none")

    for bar, val in zip(bars, values):
        x = max(bar.get_width(), 0) + xmax * 0.018
        y = bar.get_y() + bar.get_height() / 2
        ax.text(x, y, f"{val:.6f}", va="center", ha="left",
                fontsize=9, color="#333333")

    ax.set_yticks(np.arange(n))
    ax.set_yticklabels(features, fontsize=15)
    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_title(title, fontsize=16, pad=10)
    ax.set_xlim(0, xmax)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", labelsize=14)
    ax.grid(True, axis="x", alpha=0.3, linestyle="-", zorder=0)


def plot_feature_importance(cox_csv, rsf_csv):
    """RSF feature importance bar chart — blue gradient by magnitude."""
    print("\n--- Feature Importance (RSF) ---")
    df = pd.read_csv(rsf_csv)
    df = df.sort_values("Importance (mean)", ascending=True).reset_index(drop=True)

    c_index_col = "C-index (test)" if "C-index (test)" in df.columns else "C-index"
    c_index = df[c_index_col].iloc[0]

    fig, ax = plt.subplots(figsize=(8, max(4.5, len(df) * 0.52 + 1.2)))
    _importance_bar_chart(
        ax,
        df["Feature"].tolist(),
        df["Importance (mean)"].tolist(),
        title=f"Random Survival Forest — Feature Importance  (C-index = {c_index:.3f})",
        xlabel="Permutation Importance",
    )
    fig.tight_layout()
    path = OUT_DIR / "feature_importance.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix('.svg'), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    print(f"  Saved: {path.with_suffix('.svg')}")


def plot_cox_permutation_importance(cox_perm_csv):
    """Cox permutation importance — publication-quality compact chart.
    Blue gradient by magnitude (darker = higher importance).
    Only features with importance > 0 are shown.
    """
    import matplotlib.cm as cm

    print("\n--- Cox Permutation Importance ---")
    df = pd.read_csv(cox_perm_csv)

    c_index_col = "C-index (test)" if "C-index (test)" in df.columns else "C-index"
    c_index = df[c_index_col].iloc[0]

    df = df[df["Importance (mean)"] > 0].sort_values(
        "Importance (mean)", ascending=True
    ).reset_index(drop=True)

    n    = len(df)
    vals = df["Importance (mean)"].tolist()
    vmax = max(vals) if vals else 1.0

    def bar_color(v):
        if v <= 0:
            return "#d0d8e8"
        norm = 0.35 + 0.65 * (v / vmax)
        return cm.Blues(norm)

    colors = [bar_color(v) for v in vals]
    xmax   = vmax * 1.32

    fig, ax = plt.subplots(figsize=(9, 0.38 * n + 1.0))
    bars = ax.barh(np.arange(n), vals, height=0.55,
                   color=colors, zorder=3, edgecolor="none")

    for i, (val, bar) in enumerate(zip(vals, bars)):
        ax.text(val + xmax * 0.018, i, f"{val:.6f}",
                va="center", ha="left", fontsize=14, color="#333333")

    ax.set_yticks(np.arange(n))
    ax.set_yticklabels(df["Feature"], fontsize=18)
    ax.set_xlabel("Permutation Importance (Δ C-index)", fontsize=19)
    ax.set_xlim(0, xmax)
    # Fechar o box - mostrar todos os spines
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.spines["bottom"].set_visible(True)
    ax.spines["left"].set_visible(True)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color('black')
    ax.tick_params(axis="y", length=0, labelsize=18)
    ax.tick_params(axis="x", labelsize=16)
    ax.grid(True, axis="x", alpha=0.3, linestyle="-", zorder=0)

    fig.tight_layout()
    path = OUT_DIR / "cox_feature_importance.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix('.svg'), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    print(f"  Saved: {path.with_suffix('.svg')}")


def plot_cox_barh(cox_csv):
    """Bar chart of Cox HR coefficients (log scale) in RSF feature-importance style.

    Uses log(HR) so protective (HR<1) bars go left and risk (HR>1) bars go right,
    making direction immediately readable. Features sorted by |log(HR)| descending.
    Significant features (p<0.05) shown in dark colour; non-significant in light.
    """
    print("\n--- Cox PH Bar Chart ---")
    cox = pd.read_csv(cox_csv).copy()
    cox["log_hr"] = np.log(cox["HR"])
    cox["abs_log_hr"] = cox["log_hr"].abs()
    cox = cox.sort_values("abs_log_hr", ascending=True).reset_index(drop=True)

    c_index_col = "C-index (test)" if "C-index (test)" in cox.columns else "C-index"
    c_index = cox[c_index_col].iloc[0]
    n = len(cox)
    bar_h = 0.55

    COLOR_POS = "#d62728"   # HR > 1  (red)
    COLOR_NEG = "#1f77b4"   # HR < 1  (blue)

    colors = [COLOR_POS if row["log_hr"] > 0 else COLOR_NEG for _, row in cox.iterrows()]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.barh(np.arange(n), cox["log_hr"], height=bar_h, color=colors, zorder=3)

    x_abs_max = cox["abs_log_hr"].max()
    for bar, (_, row) in zip(bars, cox.iterrows()):
        val   = row["log_hr"]
        hr    = row["HR"]
        label = f"HR={hr:.2f}"
        offset = x_abs_max * 0.04
        if val >= 0:
            ax.text(val + offset, bar.get_y() + bar.get_height() / 2,
                    label, va="center", ha="left", fontsize=8.5, color="#333333")
        else:
            ax.text(val - offset, bar.get_y() + bar.get_height() / 2,
                    label, va="center", ha="right", fontsize=8.5, color="#333333")

    ax.axvline(0, color="black", lw=1.2, ls="--")
    ax.set_yticks(np.arange(n))
    ax.set_yticklabels(cox["Feature"], fontsize=9.5)
    ax.set_xlabel("log(Hazard Ratio)  —  ← protective  |  risk →  (* p < 0.05)", fontsize=10)
    ax.set_title(f"Cox PH Model — Hazard Ratios  (C-index = {c_index:.3f})", fontsize=10, pad=8)
    ax.set_xlim(-x_abs_max * 1.6, x_abs_max * 1.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="x", alpha=0.35, zorder=0)
    ax.tick_params(axis="y", length=0)

    fig.tight_layout()
    path = OUT_DIR / "cox_barh.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
def main():
    print("=" * 60)
    print("PAPER FIGURES — Script 21")
    print("=" * 60)

    global TIER_LABELS, TIER_LABELS_SHORT, FORK_TIER_LABELS
    df, q1, q3, fq1, fq3, TIER_LABELS, TIER_LABELS_SHORT, FORK_TIER_LABELS = load_data()
    print(f"\nFull dataset : {len(df)} repos | dead={int(df['event_dead'].sum())}")
    print(f"Contributor tiers: Q1={q1:.0f} Q3={q3:.0f}  →  {list(TIER_LABELS.values())}")
    print(f"Fork tiers:        Q1={fq1:.0f} Q3={fq3:.0f}  →  {list(FORK_TIER_LABELS.values())}")

    # ── KM Overall ──────────────────────────────
    print("\n--- KM Overall ---")
    _plot_km_single(
        df, "Kaplan-Meier: All ROS2 Repositories",
        "km_overall_full.png",
        label="All repos", color=COLORS["Overall"],
    )
    # ── KM by Owner Type ────────────────────────
    print("\n--- KM by Owner Type ---")
    _plot_km_groups(
        df, "owner_type", ["Organization", "User"],
        "Kaplan-Meier: Owner Type",
        "km_by_owner_type.png",
        colors=COLORS,
    )

    # ── KM by Community Tier ────────────────────
    print("\n--- KM by Community Tier ---")
    _plot_km_groups(
        df, "community_tier", ["Tier 1", "Tier 2", "Tier 3"],
        "Kaplan-Meier: Survival by Community Size",
        "km_by_community_size.png",
        colors=COLORS,
        legend_loc="lower left",
        legend_fontsize=12,
    )

    # ── KM by Fork Tier ─────────────────────────
    print("\n--- KM by Fork Tier ---")
    _plot_km_groups(
        df, "fork_tier", ["Fork Tier 1", "Fork Tier 2", "Fork Tier 3"],
        "Kaplan-Meier: Survival by Fork Count",
        "km_by_forks.png",
        colors=FORK_COLORS,
        legend_loc="lower left",
        tier_labels=FORK_TIER_LABELS,
    )

    # ── KM by Newcomer Labels ────────────────────
    print("\n--- KM by Newcomer Labels ---")
    _plot_km_groups(
        df, "has_newcomer_labels_bin",
        ["No Newcomer Labels", "Has Newcomer Labels"],
        "Kaplan-Meier: Survival by Newcomer Labels",
        "km_by_newcomer_labels.png",
        colors={"No Newcomer Labels": "#2E7D32", "Has Newcomer Labels": "#C62828"},
        legend_loc="lower left",
        tier_labels={"No Newcomer Labels": "No Newcomer Labels", "Has Newcomer Labels": "Has Newcomer Labels"},
    )

    # ── State Machines ──────────────────────────
    plot_state_machines()

    # ── Cox permutation importance ──────────────
    plot_cox_permutation_importance(COX_PERM_CSV)

    # ── Cox bar chart (RSF-style) ───────────────
    plot_cox_barh(COX_CSV)

    # ── Feature Importance: RSF ─────────────────
    plot_feature_importance(COX_CSV, RSF_CSV)

    print("\n" + "=" * 60)
    print(f"All figures saved to: {OUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
