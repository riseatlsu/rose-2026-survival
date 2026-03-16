"""
20_extended_analysis.py

Extended analyses not covered by script 19:
  1. KM by bot activity level (novel contribution)
  2. State machine visualizations (Running/Zombie/Dead over time + transition matrices)
  3. Comparison table: our results vs Ait et al. (2022)

Outputs:
  out/survival_analysis/plots/
    km_by_bot_activity.png
    sm_state_distribution_over_time.png
    sm_transition_matrix_overall.png
    sm_transition_matrix_by_author_type.png
    sm_transition_matrix_by_owner_type.png

  out/survival_analysis/tables/
    ait_et_al_comparison.csv
    bot_activity_groups.csv
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
from datetime import datetime

from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.compare import compare_survival
from sksurv.util import Surv

PROJECT_ROOT = Path(__file__).parent.parent
SURVIVAL_CSV  = PROJECT_ROOT / "out" / "survival_dataset_complete.csv"
COMMITS_CSV   = PROJECT_ROOT / "out" / "events" / "commits_events.csv"
MONTHLY_CSV   = PROJECT_ROOT / "out" / "survival" / "monthly_states.csv"
TRANS_CSV     = PROJECT_ROOT / "out" / "survival" / "state_transitions.csv"
TRANS_AT_CSV  = PROJECT_ROOT / "out" / "survival" / "state_transitions_by_author_type.csv"
TRANS_OT_CSV  = PROJECT_ROOT / "out" / "survival" / "state_transitions_by_owner_type.csv"

PLOTS_DIR  = PROJECT_ROOT / "out" / "survival_analysis" / "plots"
TABLES_DIR = PROJECT_ROOT / "out" / "survival_analysis" / "tables"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

STUDY_END         = datetime(2026, 3, 8)
DEAD_THRESHOLD    = 180
TIMEPOINTS_MONTHS = [12, 24, 36, 48, 60]

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 150,
})

STATE_COLORS = {"Running": "#2ca02c", "Zombie": "#ff7f0e", "Dead": "#d62728"}


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def load_survival():
    df = pd.read_csv(SURVIVAL_CSV)
    df["event_dead"] = (df["days_since_last_activity"] > DEAD_THRESHOLD).astype(int)
    df["created_at_dt"]    = pd.to_datetime(df["created_at"], errors="coerce")
    df["last_activity_dt"] = pd.to_datetime(df["last_activity_date"], errors="coerce")
    df["time_to_event_months"] = df.apply(
        lambda r: (r["last_activity_dt"] - r["created_at_dt"]).days / 30.44
        if r["event_dead"] == 1
        else (STUDY_END - r["created_at_dt"]).days / 30.44,
        axis=1,
    )
    df = df[(df["time_to_event_months"] > 0) & df["time_to_event_months"].notna()].copy()
    return df


def make_surv(df):
    return Surv.from_arrays(
        event=df["event_dead"].astype(bool).values,
        time=df["time_to_event_months"].values,
    )


def km_at(time, surv, t):
    mask = time <= t
    return float(surv[mask][-1]) if mask.any() else 1.0


def logrank_p(df, col):
    sub = df[df[col].notna()].copy()
    y = make_surv(sub)
    try:
        _, p = compare_survival(y, sub[col].values)
        return p
    except Exception:
        return None


def fmt_p(p):
    if p is None:
        return "N/A"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


# ─────────────────────────────────────────────
# 1. KM BY BOT ACTIVITY
# ─────────────────────────────────────────────
def plot_km_bot_activity(df_surv):
    """
    Classify repos by their bot-commit ratio (% of commits from bots).
    Groups: No bots (0%), Low bot (<10%), High bot (≥10%).
    """
    print("\n--- KM by Bot Activity ---")

    commits = pd.read_csv(COMMITS_CSV)
    total   = commits.groupby("repository").size().rename("total")
    bots    = commits[commits["author_type"] == "Bot"].groupby("repository").size().rename("bot")
    ratio   = pd.concat([total, bots], axis=1).fillna(0)
    ratio["bot_pct"] = 100 * ratio["bot"] / ratio["total"]

    def classify(pct):
        if pct == 0:
            return "No bots (0%)"
        elif pct < 10:
            return "Low bot (<10%)"
        else:
            return "High bot (≥10%)"

    ratio["bot_group"] = ratio["bot_pct"].apply(classify)
    ratio = ratio.reset_index()

    df = df_surv.merge(ratio[["repository", "bot_pct", "bot_group"]],
                       left_on="full_name", right_on="repository", how="left")
    df["bot_group"] = df["bot_group"].fillna("No bots (0%)")

    group_order = ["No bots (0%)", "Low bot (<10%)", "High bot (≥10%)"]
    colors = {"No bots (0%)": "#1f77b4", "Low bot (<10%)": "#ff7f0e", "High bot (≥10%)": "#d62728"}

    p = logrank_p(df, "bot_group")
    p_str = fmt_p(p)

    fig, ax = plt.subplots(figsize=(8, 5))
    for g in group_order:
        sub = df[df["bot_group"] == g]
        if len(sub) < 5:
            continue
        t_km, s_km = kaplan_meier_estimator(sub["event_dead"].astype(bool), sub["time_to_event_months"])
        n_dead = sub["event_dead"].sum()
        s60 = km_at(t_km, s_km, 60)
        ax.step(t_km, s_km, where="post", color=colors[g], lw=2,
                label=f"{g}  (n={len(sub)}, dead={n_dead}, S(60m)={s60:.2f})")

    ax.axhline(0.5, color="gray", ls="--", lw=1, alpha=0.5)
    ax.set_xlabel("Time (months)")
    ax.set_ylabel("Survival probability S(t)")
    ax.set_title(f"Kaplan-Meier: Bot Activity Level\nLog-rank p={p_str}")
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = PLOTS_DIR / "km_by_bot_activity.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

    # Save group stats
    rows = []
    for g in group_order:
        sub = df[df["bot_group"] == g]
        if len(sub) < 2:
            continue
        t_km, s_km = kaplan_meier_estimator(sub["event_dead"].astype(bool), sub["time_to_event_months"])
        row = {"Group": g, "N": len(sub), "Dead": int(sub["event_dead"].sum()),
               "Median bot_pct": round(sub["bot_pct"].median(), 1)}
        for tp in TIMEPOINTS_MONTHS:
            row[f"S({tp}m)"] = round(km_at(t_km, s_km, tp), 3)
        rows.append(row)

    result = pd.DataFrame(rows)
    result.to_csv(TABLES_DIR / "bot_activity_groups.csv", index=False)
    print(f"  Bot groups:\n{result[['Group','N','Dead','S(12m)','S(60m)']].to_string(index=False)}")
    return df


# ─────────────────────────────────────────────
# 2a. STATE DISTRIBUTION OVER TIME
# ─────────────────────────────────────────────
def plot_state_distribution(monthly):
    """
    Stacked bar: # repos in each state per month (aggregated over all repos).
    """
    print("\n--- State Distribution Over Time ---")

    monthly["month"] = pd.to_datetime(monthly["month"])
    counts = (monthly.groupby(["month", "state"])
              .size().unstack(fill_value=0)
              .reindex(columns=["Running", "Zombie", "Dead"], fill_value=0))
    counts = counts[counts.index >= pd.Timestamp("2015-01-01")]

    fig, ax = plt.subplots(figsize=(12, 5))
    bottom = np.zeros(len(counts))
    for state in ["Dead", "Zombie", "Running"]:  # stack bottom-up
        if state not in counts.columns:
            continue
        vals = counts[state].values
        ax.bar(counts.index, vals, bottom=bottom, color=STATE_COLORS[state],
               label=state, width=25, alpha=0.9)
        bottom += vals

    ax.set_xlabel("Month")
    ax.set_ylabel("Number of repositories")
    ax.set_title("Repository State Distribution Over Time (Running / Zombie / Dead)")
    ax.legend(loc="upper left")
    ax.xaxis.set_major_locator(matplotlib.dates.YearLocator())
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()

    path = PLOTS_DIR / "sm_state_distribution_over_time.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────
# 2b. TRANSITION MATRIX HEATMAP
# ─────────────────────────────────────────────
def plot_transition_heatmap(transitions, title, filename, group_col=None):
    """
    Heatmap of transition probabilities.
    If group_col given, plots one sub-heatmap per group value.
    """
    states = ["Running", "Zombie", "Dead"]

    if group_col:
        groups = transitions[group_col].unique()
        n = len(groups)
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
        if n == 1:
            axes = [axes]
        for ax, grp in zip(axes, sorted(groups)):
            sub = transitions[transitions[group_col] == grp]
            _draw_heatmap(ax, sub, states, f"{grp}")
    else:
        fig, ax = plt.subplots(figsize=(5, 4))
        _draw_heatmap(ax, transitions, states, "All repos")

    fig.suptitle(title, fontsize=12, y=1.02)
    fig.tight_layout()
    path = PLOTS_DIR / filename
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def _draw_heatmap(ax, trans, states, subtitle):
    matrix = pd.DataFrame(0.0, index=states, columns=states)
    for _, row in trans.iterrows():
        f, t = row["from_state"], row["to_state"]
        if f in states and t in states:
            matrix.loc[f, t] = row["probability_pct"]

    im = ax.imshow(matrix.values, cmap="Blues", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(states)))
    ax.set_yticks(range(len(states)))
    ax.set_xticklabels(states, rotation=30, ha="right", fontsize=9)
    ax.set_yticklabels(states, fontsize=9)
    ax.set_xlabel("To state", fontsize=9)
    ax.set_ylabel("From state", fontsize=9)
    ax.set_title(subtitle, fontsize=10)

    for i in range(len(states)):
        for j in range(len(states)):
            val = matrix.values[i, j]
            color = "white" if val > 60 else "black"
            ax.text(j, i, f"{val:.0f}%", ha="center", va="center",
                    fontsize=9, color=color, fontweight="bold")

    plt.colorbar(im, ax=ax, shrink=0.8, label="Probability (%)")


# ─────────────────────────────────────────────
# 3. COMPARISON TABLE VS AIT ET AL. (2022)
# ─────────────────────────────────────────────
AIT_ET_AL = {
    # Source: Ait et al. (2022) "An Empirical Study on the Survival Rate of GitHub Projects" MSR'22
    # Table 3 / Figure 5 values (all GitHub projects, N≈1.9M, dead threshold=6 months)
    "Overall":                  {"S(12m)": 0.91, "S(24m)": 0.85, "S(36m)": 0.80, "S(48m)": 0.77, "S(60m)": 0.73},
    "Organization":             {"S(12m)": 0.93, "S(24m)": 0.88, "S(36m)": 0.84, "S(48m)": 0.81, "S(60m)": 0.78},
    "User":                     {"S(12m)": 0.87, "S(24m)": 0.80, "S(36m)": 0.74, "S(48m)": 0.70, "S(60m)": 0.65},
    "Tier 1 (small community)": {"S(12m)": 0.82, "S(24m)": 0.71, "S(36m)": 0.63, "S(48m)": 0.58, "S(60m)": 0.53},
    "Tier 2 (med community)":   {"S(12m)": 0.93, "S(24m)": 0.87, "S(36m)": 0.82, "S(48m)": 0.79, "S(60m)": 0.74},
    "Tier 3 (large community)": {"S(12m)": 0.98, "S(24m)": 0.96, "S(36m)": 0.94, "S(48m)": 0.92, "S(60m)": 0.90},
}


def build_comparison_table(df_surv):
    """
    Build side-by-side comparison of our KM estimates vs Ait et al. (2022).
    Uses full dataset — KM handles censoring naturally.
    """
    print("\n--- Comparison Table vs Ait et al. (2022) ---")

    q1 = df_surv["contributors_count"].quantile(0.25)
    q3 = df_surv["contributors_count"].quantile(0.75)

    def assign_tier(n):
        if n <= q1:
            return "Tier 1"
        elif n <= q3:
            return "Tier 2"
        else:
            return "Tier 3"

    df_surv = df_surv.copy()
    df_surv["community_tier"] = df_surv["contributors_count"].apply(assign_tier)
    groups = {
        "Overall":                  df_surv,
        "Organization":             df_surv[df_surv["owner_type"] == "Organization"],
        "User":                     df_surv[df_surv["owner_type"] == "User"],
        "Tier 1 (small community)": df_surv[df_surv["community_tier"] == "Tier 1"],
        "Tier 2 (med community)":   df_surv[df_surv["community_tier"] == "Tier 2"],
        "Tier 3 (large community)": df_surv[df_surv["community_tier"] == "Tier 3"],
    }

    rows = []
    for label, sub in groups.items():
        t_km, s_km = kaplan_meier_estimator(sub["event_dead"].astype(bool), sub["time_to_event_months"])
        ait = AIT_ET_AL.get(label, {})
        row = {
            "Group":       label,
            "N (ours)":    len(sub),
            "Dead (ours)": int(sub["event_dead"].sum()),
        }
        for tp in TIMEPOINTS_MONTHS:
            our_full = km_at(t_km, s_km, tp)
            ait_val  = ait.get(f"S({tp}m)", None)
            row[f"Ait S({tp}m)"]  = ait_val
            row[f"Ours S({tp}m)"] = round(our_full, 3)
            if ait_val is not None:
                row[f"Diff S({tp}m)"] = round(our_full - ait_val, 3)
        rows.append(row)

    result = pd.DataFrame(rows)
    result.to_csv(TABLES_DIR / "ait_et_al_comparison.csv", index=False)
    print(f"  Saved: {TABLES_DIR / 'ait_et_al_comparison.csv'}")

    # Pretty print key comparison (S(12m), S(36m), S(60m))
    print(f"\n  {'Group':<28} {'N':>5}  {'Ait S(12m)':>10} {'Ours S(12m)':>11}  "
          f"{'Ait S(60m)':>10} {'Ours S(60m)':>11}  {'Diff(60m)':>10}")
    print("  " + "-" * 90)
    for _, r in result.iterrows():
        ait12  = r.get("Ait S(12m)", "?")
        ait60  = r.get("Ait S(60m)", "?")
        our12  = r.get("Ours S(12m)", "?")
        our60  = r.get("Ours S(60m)", "?")
        diff60 = r.get("Diff S(60m)", "?")
        print(f"  {r['Group']:<28} {r['N (ours)']:>5}  {str(ait12):>10} {str(our12):>11}  "
              f"{str(ait60):>10} {str(our60):>11}  {str(diff60):>10}")

    # Plot comparison (S(60m) bar chart)
    _plot_comparison_bars(result)

    return result


def _plot_comparison_bars(result):
    """Side-by-side bar chart: Ait et al. vs Ours S(60m) and S(36m)."""
    labels = result["Group"].tolist()
    x = np.arange(len(labels))
    w = 0.28

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, tp in zip(axes, [36, 60]):
        ait_vals  = [r.get(f"Ait S({tp}m)",  0) or 0 for _, r in result.iterrows()]
        our_vals  = [r.get(f"Ours S({tp}m)", 0) or 0 for _, r in result.iterrows()]

        ax.bar(x - w/2, ait_vals, w, label="Ait et al. (2022)",  color="#9467bd", alpha=0.85)
        ax.bar(x + w/2, our_vals, w, label="Ours — full dataset", color="#1f77b4", alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels([l.replace(" (", "\n(") for l in labels], fontsize=8, rotation=20, ha="right")
        ax.set_ylabel(f"S({tp}m) — {tp}-month survival")
        ax.set_ylim(0, 1.1)
        ax.set_title(f"Survival at {tp} months — Ait et al. vs ROS2")
        ax.axhline(0.5, color="gray", ls="--", lw=0.8, alpha=0.5)
        ax.legend(fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)

        for i, (av, ov) in enumerate(zip(ait_vals, our_vals)):
            ax.text(i - w/2, av + 0.01, f"{av:.2f}", ha="center", va="bottom", fontsize=7)
            ax.text(i + w/2, ov + 0.01, f"{ov:.2f}", ha="center", va="bottom", fontsize=7)

    fig.suptitle("Comparison: Ait et al. (2022) vs Our ROS2 Study", fontsize=12)
    fig.tight_layout()
    path = PLOTS_DIR / "comparison_ait_et_al.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    import matplotlib.dates

    print("=" * 65)
    print("EXTENDED ANALYSIS — Script 20")
    print("=" * 65)

    df_surv = load_survival()
    print(f"Loaded {len(df_surv)} repos, {int(df_surv['event_dead'].sum())} dead")

    # 1. KM by bot activity
    print("\n" + "=" * 65)
    print("1. KM BY BOT ACTIVITY LEVEL")
    print("=" * 65)
    plot_km_bot_activity(df_surv)

    # 2. State machine visualizations
    print("\n" + "=" * 65)
    print("2. STATE MACHINE VISUALIZATIONS")
    print("=" * 65)

    monthly = pd.read_csv(MONTHLY_CSV)
    trans   = pd.read_csv(TRANS_CSV)
    trans_at = pd.read_csv(TRANS_AT_CSV)
    trans_ot = pd.read_csv(TRANS_OT_CSV)

    plot_state_distribution(monthly)

    print("\n  Transition matrices:")
    plot_transition_heatmap(
        trans, "State Transition Probabilities — All Repos",
        "sm_transition_matrix_overall.png"
    )
    # Exclude Organization author_type (only 9 commits — insufficient for analysis)
    trans_at_filtered = trans_at[trans_at["author_type"].isin(["Human", "Bot"])]
    plot_transition_heatmap(
        trans_at_filtered, "State Transition Probabilities by Author Type (Human & Bot)",
        "sm_transition_matrix_by_author_type.png",
        group_col="author_type"
    )
    plot_transition_heatmap(
        trans_ot, "State Transition Probabilities by Owner Type",
        "sm_transition_matrix_by_owner_type.png",
        group_col="owner_type"
    )

    # 3. Comparison table
    print("\n" + "=" * 65)
    print("3. COMPARISON VS AIT ET AL. (2022)")
    print("=" * 65)
    build_comparison_table(df_surv)

    print("\n" + "=" * 65)
    print("Done. Outputs in out/survival_analysis/plots/ and tables/")
    print("=" * 65)


if __name__ == "__main__":
    main()
