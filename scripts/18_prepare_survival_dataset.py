"""
18_prepare_survival_dataset.py

Prepara o dataset de sobrevivência extraindo datas dos JSONs.

Para cada repositório:
1. Extrair created_at de general_info.json
2. Extrair última data de QUALQUER atividade (commits, issues, PRs, comments, reviews)
3. Calcular time_to_event e event

Following Ait et al. (2022): a repository is considered inactive (event=1) only if
it shows a complete absence of ANY activity (commits, issues opened, PRs opened,
issue comments, PR reviews) for more than 180 consecutive days before the study end.
Repositories with any such activity within 180 days are active (event=0, censored),
even if the activity is non-code (issues, comments) — this corresponds to the
"Zombie" state in Ait et al., which they classify as still Alive.

Configurações:
- DEAD_THRESHOLD_DAYS = 180 (sem QUALQUER atividade = morto)
- STUDY_END_DATE = "2026-03-08"

Colunas criadas:
- first_activity_date: data de criação do repo
- last_commit_date: data do último commit (para state machine)
- last_activity_date: data da última QUALQUER atividade (commits + issues + PRs + comments + reviews)
- study_end_date: fim do estudo (2026-03-08)
- days_since_last_activity: dias desde última atividade até fim do estudo
- event_dead: 1 se morto (days_since_last_activity > 180), 0 se censurado
- time_to_event_days: dias desde criação até morte ou censoring
- time_to_event_months: time_to_event_days / 30.44
"""

import json
import pandas as pd
from datetime import datetime
from pathlib import Path

# =========================
# CONFIG
# =========================
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = SCRIPT_DIR / "data" / "ros_robotics_data"
INPUT_FILE = PROJECT_ROOT / "out" / "survival_repo_dataset.csv"
OUTPUT_FILE = PROJECT_ROOT / "out" / "survival_dataset_complete.csv"

DEAD_THRESHOLD_DAYS = 180  # 6 months without activity = dead (matches Ait et al. 2022)
STUDY_END_DATE = datetime(2026, 3, 8)

# =========================
# HELPERS
# =========================
def load_json_safe(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def parse_date(date_str):
    if not date_str:
        return None
    try:
        s = str(date_str)
        if "T" in s:
            return datetime.fromisoformat(s.replace("Z", "+00:00")).replace(tzinfo=None)
        return datetime.strptime(s[:10], "%Y-%m-%d")
    except Exception:
        return None


def find_repo_dir(owner, name):
    """Locate repo directory using multiple naming conventions."""
    candidates = [
        f"{owner}__{name}",
        f"{owner.lower()}__{name.lower()}",
        f"{owner.replace('-', '_')}__{name.replace('-', '_')}",
    ]
    for pattern in candidates:
        p = DATA_DIR / pattern
        if p.exists():
            return p

    # Fuzzy match (case-insensitive)
    owner_l, name_l = owner.lower(), name.lower()
    for d in DATA_DIR.iterdir():
        if not d.is_dir():
            continue
        parts = d.name.split("__")
        if len(parts) == 2 and parts[0].lower() == owner_l and parts[1].lower() == name_l:
            return d
    return None


def _extract_dates_from_list(items, *date_fields):
    """Extract all non-None dates from a list of dicts given field names."""
    dates = []
    if not isinstance(items, list):
        return dates
    for item in items:
        if not isinstance(item, dict):
            continue
        for field in date_fields:
            val = item.get(field)
            if val:
                d = parse_date(val)
                if d:
                    dates.append(d)
    return dates


def get_repo_dates(owner, name):
    """Return (first_activity, last_commit, last_any_activity, created_at) from JSON files.

    last_commit: most recent commit date (for state machine / zombie detection)
    last_any_activity: most recent date across commits, issues, PRs, comments, reviews
                       (for Ait et al. inactivity definition)
    """
    repo_dir = find_repo_dir(owner, name)
    if repo_dir is None:
        return None, None, None, None

    # created_at and pushed_at from general_info.json
    general = load_json_safe(repo_dir / "general_info.json")
    created_at = pushed_at = None
    if general and "data" in general:
        created_at = parse_date(general["data"].get("created_at"))
        pushed_at = parse_date(general["data"].get("pushed_at"))

    all_dates = []

    # Commits
    commits_data = load_json_safe(repo_dir / "commits.json")
    commit_dates = []
    if commits_data and "data" in commits_data and isinstance(commits_data["data"], list):
        commit_dates = [
            parse_date(c.get("date"))
            for c in commits_data["data"]
            if isinstance(c, dict) and c.get("date")
        ]
        commit_dates = [d for d in commit_dates if d is not None]
    all_dates.extend(commit_dates)

    # Issues (created_at, closed_at)
    issues_data = load_json_safe(repo_dir / "issues.json")
    if issues_data:
        items = issues_data.get("data", issues_data) if isinstance(issues_data, dict) else issues_data
        all_dates.extend(_extract_dates_from_list(items, "created_at", "closed_at"))

    # Pull requests (created_at, merged_at, closed_at)
    prs_data = load_json_safe(repo_dir / "pull_requests.json")
    if prs_data:
        items = prs_data.get("data", prs_data) if isinstance(prs_data, dict) else prs_data
        all_dates.extend(_extract_dates_from_list(items, "created_at", "merged_at", "closed_at"))

    # Issue comments (created_at)
    ic_data = load_json_safe(repo_dir / "issue_comments.json")
    if ic_data:
        items = ic_data.get("data", ic_data) if isinstance(ic_data, dict) else ic_data
        all_dates.extend(_extract_dates_from_list(items, "created_at"))

    # PR reviews (submitted_at)
    pr_reviews_data = load_json_safe(repo_dir / "pr_reviews.json")
    if pr_reviews_data:
        items = pr_reviews_data.get("data", pr_reviews_data) if isinstance(pr_reviews_data, dict) else pr_reviews_data
        all_dates.extend(_extract_dates_from_list(items, "submitted_at"))

    # Derive summary dates
    first_commit = min(commit_dates) if commit_dates else None
    last_commit = max(commit_dates) if commit_dates else None
    last_any_activity = max(all_dates) if all_dates else pushed_at

    first_activity = first_commit or created_at

    return first_activity, last_commit, last_any_activity, created_at


# =========================
# MAIN
# =========================
def main():
    print("=" * 60)
    print("PREPARING SURVIVAL DATASET")
    print(f"  Study end date  : {STUDY_END_DATE.date()}")
    print(f"  Dead threshold  : {DEAD_THRESHOLD_DAYS} days")
    print("=" * 60)

    df = pd.read_csv(INPUT_FILE)
    print(f"\nLoaded {len(df)} rows from {INPUT_FILE.name}")

    # ---- Extract dates from JSONs ----
    first_activities, last_commits, last_any_activities, created_dates = [], [], [], []
    not_found = 0

    for i, row in df.iterrows():
        if (i + 1) % 100 == 0:
            print(f"  Processing {i + 1}/{len(df)} ...")

        owner = str(row.get("Owner", ""))
        name = str(row.get("Name", ""))
        first_act, last_commit, last_any, created = get_repo_dates(owner, name)

        if first_act is None and last_any is None:
            not_found += 1

        first_activities.append(first_act)
        last_commits.append(last_commit)
        last_any_activities.append(last_any)
        created_dates.append(created)

    print(f"\n  Repos with JSON data   : {len(df) - not_found}/{len(df)}")
    print(f"  Repos without JSON data: {not_found}/{len(df)}")

    # ---- Build survival columns ----
    df["first_activity_date"] = first_activities
    df["last_commit_date"] = last_commits
    df["last_activity_date"] = last_any_activities   # commits + issues + PRs + comments + reviews
    df["created_at"] = created_dates
    df["study_end_date"] = STUDY_END_DATE

    df["days_since_last_activity"] = df["last_activity_date"].apply(
        lambda x: (STUDY_END_DATE - x).days if pd.notna(x) else None
    )
    df["inactive_months_before_study_end"] = df["days_since_last_activity"] / 30.44

    df["repository_age_days"] = df["created_at"].apply(
        lambda x: (STUDY_END_DATE - x).days if pd.notna(x) else None
    )

    # Event: 1 = dead (no activity for > threshold), 0 = censored (still active)
    df["event_dead"] = (df["days_since_last_activity"] > DEAD_THRESHOLD_DAYS).astype(int)

    def calc_time_to_event(row):
        if pd.isna(row["created_at"]) or pd.isna(row["last_activity_date"]):
            return None
        if row["event_dead"] == 1:
            # Time from creation to last observed activity (death time).
            # Use max(1, days) so sub-day differences don't yield 0.
            return max(1, (row["last_activity_date"] - row["created_at"]).days)
        else:
            # Censored: observed until study end
            return (STUDY_END_DATE - row["created_at"]).days

    df["time_to_event_days"] = df.apply(calc_time_to_event, axis=1)
    df["time_to_event_months"] = df["time_to_event_days"] / 30.44

    # ---- Summary ----
    valid = df["time_to_event_days"].notna() & (df["time_to_event_days"] > 0)
    valid_df = df[valid]

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total repos           : {len(df)}")
    print(f"Valid survival rows   : {valid.sum()}")
    print(f"Dead   (event=1)      : {valid_df['event_dead'].sum()}  "
          f"({100 * valid_df['event_dead'].mean():.1f}%)")
    print(f"Alive  (censored=0)   : {(valid_df['event_dead'] == 0).sum()}  "
          f"({100 * (1 - valid_df['event_dead'].mean()):.1f}%)")
    print(f"Median time to event  : {valid_df['time_to_event_months'].median():.1f} months")
    print(f"Max time to event     : {valid_df['time_to_event_months'].max():.1f} months")

    # ---- Save ----
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved to: {OUTPUT_FILE}")

    new_cols = [
        "first_activity_date", "last_commit_date", "last_activity_date", "created_at",
        "study_end_date", "days_since_last_activity", "inactive_months_before_study_end",
        "repository_age_days", "event_dead", "time_to_event_days", "time_to_event_months",
    ]
    print("\nNew columns added:")
    for c in new_cols:
        print(f"  - {c}")


if __name__ == "__main__":
    main()
