"""
16_build_state_machine.py

Build state machine representation for repository evolution following Ait et al. (2022).
Discretizes time into monthly intervals and assigns states based on activity.

States (mutually exclusive per month):
  - Running: Month with at least 1 commit by a human developer
  - Zombie:  Month with activity (issues, PRs, comments, reviews) but no human commits
  - Dead:    Month with no registered activity

Definitions (adjusted to match Ait et al.):
  - Survival criterion: Activity in the last 6 months of study period
  - Death moment: Timestamp of last activity for dead projects

Outputs:
  - out/survival/monthly_states.csv                    (repository, month, state)
  - out/survival/evolution_paths.csv                   (repository, path, final_state)
  - out/survival/state_transitions.csv                 (from_state, to_state, count, probability)
  - out/survival/state_statistics.csv                  (state, avg_duration_months, total_months)
  - out/survival/survival_dataset.csv                  (main dataset for scikit-survival)
  - out/survival/monthly_states_by_author_type.csv     (author_type, month, state)
  - out/survival/state_transitions_by_author_type.csv  (author_type, from_state, to_state, count, probability)
  - out/survival/state_statistics_by_author_type.csv   (author_type, state, avg_duration_months, ...)
  - out/survival/monthly_states_by_owner_type.csv      (owner_type, month, state)
  - out/survival/state_transitions_by_owner_type.csv   (owner_type, from_state, to_state, count, probability)
  - out/survival/state_statistics_by_owner_type.csv    (owner_type, state, avg_duration_months, ...)

References:
  - Ait et al. (2022) "An Empirical Study on the Survival Rate of GitHub Projects"
"""

import os
import csv
from datetime import datetime
from dateutil.relativedelta import relativedelta
from collections import defaultdict
from typing import Dict, List, Tuple, Any

# =========================
# CONFIG
# =========================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

EVENTS_DIR = os.path.join(PROJECT_ROOT, "out/events")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "out/survival")
FILTERED_REPOS = os.path.join(PROJECT_ROOT, "out/survival_repo_dataset.csv")

# Study period
STUDY_END = datetime(2026, 3, 8)
SURVIVAL_THRESHOLD_MONTHS = 6

# States
RUNNING = "Running"
ZOMBIE  = "Zombie"
DEAD    = "Dead"

# =========================
# HELPERS
# =========================
def parse_month(ts: str) -> str:
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m")
    except:
        return None

def month_diff(month1: str, month2: str) -> int:
    d1 = datetime.strptime(month1, "%Y-%m")
    d2 = datetime.strptime(month2, "%Y-%m")
    return (d2.year - d1.year) * 12 + (d2.month - d1.month)

def generate_month_range(start_month: str, end_month: str) -> List[str]:
    months = []
    current = datetime.strptime(start_month, "%Y-%m")
    end = datetime.strptime(end_month, "%Y-%m")
    while current <= end:
        months.append(current.strftime("%Y-%m"))
        current += relativedelta(months=1)
    return months

def compress_path(states: List[str]) -> str:
    if not states:
        return ""
    compressed = []
    current_state = states[0]
    count = 1
    for state in states[1:]:
        if state == current_state:
            count += 1
        else:
            compressed.append(f"{current_state}({count}m)")
            current_state = state
            count = 1
    compressed.append(f"{current_state}({count}m)")
    return "-".join(compressed)

def write_transitions(rows: List[Dict], filepath: str, extra_fields: List[str] = []):
    fieldnames = extra_fields + ["from_state", "to_state", "count", "probability_pct"]
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def write_statistics(rows: List[Dict], filepath: str, extra_fields: List[str] = []):
    fieldnames = extra_fields + ["state", "avg_duration_months", "total_occurrences", "total_months"]
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def compute_transitions_and_stats(
    group_sequences: Dict[str, List[List[str]]],
    group_key: str
) -> Tuple[List[Dict], List[Dict]]:
    """
    group_sequences:
      {
        group_value: [
          [state1, state2, state3, ...],   # repo 1
          [state1, state2, state3, ...],   # repo 2
          ...
        ]
      }
    """
    transitions_out = []
    stats_out = []

    for group_val in sorted(group_sequences.keys()):
        sequences = group_sequences[group_val]

        transitions = defaultdict(int)
        state_durations = defaultdict(list)

        for states in sequences:
            if not states:
                continue

            # transitions within the same repo only
            for i in range(len(states) - 1):
                transitions[(states[i], states[i + 1])] += 1

            # durations within the same repo only
            current_state = states[0]
            duration = 1
            for state in states[1:]:
                if state == current_state:
                    duration += 1
                else:
                    state_durations[current_state].append(duration)
                    current_state = state
                    duration = 1
            state_durations[current_state].append(duration)

        for from_state in [RUNNING, ZOMBIE, DEAD]:
            outbound = {k: v for k, v in transitions.items() if k[0] == from_state}
            total = sum(outbound.values())
            for (f, t), count in outbound.items():
                prob = (count / total * 100) if total > 0 else 0
                transitions_out.append({
                    group_key: group_val,
                    "from_state": f,
                    "to_state": t,
                    "count": count,
                    "probability_pct": round(prob, 2)
                })

        for state in [RUNNING, ZOMBIE, DEAD]:
            durations = state_durations.get(state, [])
            avg = sum(durations) / len(durations) if durations else 0
            stats_out.append({
                group_key: group_val,
                "state": state,
                "avg_duration_months": round(avg, 2),
                "total_occurrences": len(durations),
                "total_months": sum(durations)
            })

    return transitions_out, stats_out

def normalize_author_type(author_type: str) -> str:
    """
    Normalize raw GitHub API author types into analysis-friendly categories.

    Raw GitHub API values:
      - User         → Human
      - Bot          → Bot
      - Organization → Organization (kept separate for grouped analysis)
      - Unknown      → Human (deleted accounts were real users; bots are always labeled)
      - missing      → Human (same reasoning)

    Never returns 'Unknown'.
    """
    if not author_type:
        return "Human"

    value = str(author_type).strip()

    mapping = {
        "Human":        "Human",         # already normalized by 15_build_event_tables.py
        "User":         "Human",         # raw GitHub API value
        "Bot":          "Bot",
        "Organization": "Organization",  # kept as own group
        "Unknown":      "Human",         # deleted accounts → treat as human
    }
    return mapping.get(value, "Human")

# =========================
# LOAD EVENTS
# =========================
def load_events() -> Tuple[Dict, Dict, set]:
    """Load events and organize by repository and month."""

    commits        = defaultdict(lambda: defaultdict(list))
    other_activity = defaultdict(lambda: defaultdict(list))
    all_repos      = set()

    commits_file = os.path.join(EVENTS_DIR, "commits_events.csv")
    if os.path.exists(commits_file):
        with open(commits_file, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                month = parse_month(row["timestamp"])
                if month:
                    repo = row["repository"]
                    all_repos.add(repo)
                    commits[repo][month].append({
                        "author": row["author"],
                        "author_type": normalize_author_type(row.get("author_type"))
                    })

    for event_type, event_file in [
        ("issue",   "issues_events.csv"),
        ("pr",      "pull_requests_events.csv"),
        ("comment", "comments_events.csv"),
        ("review",  "reviews_events.csv"),
    ]:
        path = os.path.join(EVENTS_DIR, event_file)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    month = parse_month(row["timestamp"])
                    if month:
                        repo = row["repository"]
                        all_repos.add(repo)
                        other_activity[repo][month].append({
                            "type": event_type,
                            "author": row["author"],
                            "author_type": normalize_author_type(row.get("author_type"))
                        })
    
    debug_commit_types = defaultdict(int)
    for repo in commits:
        for month in commits[repo]:
            for c in commits[repo][month]:
                debug_commit_types[c.get("author_type", "MISSING")] += 1

    print("\n[DEBUG] Commit author_type distribution in load_events():")
    for k, v in sorted(debug_commit_types.items(), key=lambda x: (-x[1], x[0])):
        print(f"  {k}: {v}")

    return commits, other_activity, all_repos

def determine_monthly_state(commits_in_month: List[Dict], other_in_month: List[Dict]) -> str:
    human_commits = [c for c in commits_in_month if c.get("author_type") == "Human"]
    if human_commits:
        return RUNNING
    if commits_in_month or other_in_month:
        return ZOMBIE
    return DEAD
# =========================
# MAIN STATE MACHINE
# =========================
def build_state_machine():
    print("=" * 60)
    print("Building State Machine for Survival Analysis")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\nLoading events...")
    commits, other_activity, all_repos = load_events()
    print(f"  Found {len(all_repos)} repositories with events")

    repo_metadata = {}
    with open(FILTERED_REPOS, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            full_name = row.get("full_name", f"{row['Owner']}/{row['Name']}")
            repo_metadata[full_name] = row

    study_end_month  = STUDY_END.strftime("%Y-%m")
    survival_cutoff  = (STUDY_END - relativedelta(months=SURVIVAL_THRESHOLD_MONTHS)).strftime("%Y-%m")

    print(f"  Study end: {study_end_month}")
    print(f"  Survival cutoff: {survival_cutoff} (last {SURVIVAL_THRESHOLD_MONTHS} months)")

    monthly_states  = []
    evolution_paths = []
    transitions     = defaultdict(int)
    state_durations = defaultdict(list)
    survival_data   = []

    for repo in sorted(all_repos):
        all_months = set(commits[repo].keys()) | set(other_activity[repo].keys())
        if not all_months:
            continue

        first_month          = min(all_months)
        last_activity_month  = max(all_months)
        month_range          = generate_month_range(first_month, study_end_month)

        states = []
        for month in month_range:
            state = determine_monthly_state(
                commits[repo].get(month, []),
                other_activity[repo].get(month, [])
            )
            states.append(state)
            monthly_states.append({"repository": repo, "month": month, "state": state})

        for i in range(len(states) - 1):
            transitions[(states[i], states[i + 1])] += 1

        if states:
            current_state = states[0]
            duration = 1
            for state in states[1:]:
                if state == current_state:
                    duration += 1
                else:
                    state_durations[current_state].append(duration)
                    current_state = state
                    duration = 1
            state_durations[current_state].append(duration)

        is_alive    = last_activity_month >= survival_cutoff
        final_state = states[-1] if states else DEAD

        evolution_paths.append({
            "repository":          repo,
            "path":                compress_path(states),
            "final_state":         final_state,
            "is_alive":            is_alive,
            "first_month":         first_month,
            "last_activity_month": last_activity_month,
            "lifespan_months":     len(month_range)
        })

        time_to_event = month_diff(first_month,
                                   last_activity_month if not is_alive else study_end_month)
        meta = repo_metadata.get(repo, {})

        survival_data.append({
            "repository":          repo,
            "time_months":         max(1, time_to_event),
            "event":               0 if is_alive else 1,
            "is_alive":            is_alive,
            "first_month":         first_month,
            "last_activity_month": last_activity_month,
            "owner_type":          meta.get("owner_type") or "",
            "has_readme":          meta.get("has_readme", "False") == "True",
            "has_contributing":    meta.get("has_contributing", "False") == "True",
            "has_code_of_conduct": meta.get("has_code_of_conduct", "False") == "True",
            "has_newcomer_labels": meta.get("has_newcomer_labels", "False") == "True",
            "contributors_count":  int(meta.get("contributors_count", 0) or 0),
            "commits_count":       int(meta.get("commits_count", 0) or 0),
            "stars":               int(meta.get("Number of stars", 0) or 0),
            "forks":               int(meta.get("Number of forks", 0) or 0),
        })

    # --- Write outputs ---
    print("\nWriting outputs...")

    with open(os.path.join(OUTPUT_DIR, "monthly_states.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["repository", "month", "state"])
        writer.writeheader()
        writer.writerows(monthly_states)
    print(f"  monthly_states.csv: {len(monthly_states)} rows")

    with open(os.path.join(OUTPUT_DIR, "evolution_paths.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["repository", "path", "final_state", "is_alive",
                                               "first_month", "last_activity_month", "lifespan_months"])
        writer.writeheader()
        writer.writerows(evolution_paths)
    print(f"  evolution_paths.csv: {len(evolution_paths)} rows")

    transition_stats = []
    for from_state in [RUNNING, ZOMBIE, DEAD]:
        outbound = {k: v for k, v in transitions.items() if k[0] == from_state}
        total = sum(outbound.values())
        for (f, t), count in outbound.items():
            prob = (count / total * 100) if total > 0 else 0
            transition_stats.append({
                "from_state": f, "to_state": t,
                "count": count, "probability_pct": round(prob, 2)
            })
    write_transitions(transition_stats, os.path.join(OUTPUT_DIR, "state_transitions.csv"))
    print(f"  state_transitions.csv: {len(transition_stats)} rows")

    state_stats = []
    for state in [RUNNING, ZOMBIE, DEAD]:
        durations = state_durations.get(state, [])
        avg = sum(durations) / len(durations) if durations else 0
        state_stats.append({
            "state": state,
            "avg_duration_months": round(avg, 2),
            "total_occurrences":   len(durations),
            "total_months":        sum(durations)
        })
    write_statistics(state_stats, os.path.join(OUTPUT_DIR, "state_statistics.csv"))
    print(f"  state_statistics.csv: {len(state_stats)} rows")

    survival_fieldnames = [
        "repository", "time_months", "event", "is_alive",
        "first_month", "last_activity_month",
        "owner_type", "has_readme", "has_contributing",
        "has_code_of_conduct", "has_newcomer_labels",
        "contributors_count", "commits_count", "stars", "forks"
    ]
    with open(os.path.join(OUTPUT_DIR, "survival_dataset.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=survival_fieldnames)
        writer.writeheader()
        writer.writerows(survival_data)
    print(f"  survival_dataset.csv: {len(survival_data)} rows")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    alive_count = sum(1 for d in survival_data if d["is_alive"])
    dead_count  = len(survival_data) - alive_count
    if survival_data:
        print(f"\nSurvival Status:")
        print(f"  Alive: {alive_count} ({100*alive_count/len(survival_data):.1f}%)")
        print(f"  Dead:  {dead_count}  ({100*dead_count/len(survival_data):.1f}%)")
    print(f"\nState Statistics:")
    for stat in state_stats:
        print(f"  {stat['state']}: avg {stat['avg_duration_months']:.2f}m, {stat['total_occurrences']} occurrences")
    print(f"\nTransition Probabilities:")
    for t in sorted(transition_stats, key=lambda x: (x["from_state"], -x["probability_pct"])):
        print(f"  {t['from_state']} → {t['to_state']}: {t['probability_pct']:.1f}%")
    print("\n" + "=" * 60)

# =========================
# STATE MACHINE BY AUTHOR TYPE (Human vs Bot)
# =========================

def determine_monthly_state_for_group(commits_in_month: List[Dict], other_in_month: List[Dict]) -> str:
    if commits_in_month:
        return RUNNING
    if other_in_month:
        return ZOMBIE
    return DEAD

def build_state_machine_by_author_type():
    """
    Builds state machine grouping monthly activity by normalized author type.

    Raw API values:
      - User         → Human
      - Bot          → Bot
      - Organization → Organization
      - Unknown      → Human (deleted accounts; see normalize_author_type)

    Analysis groups (detected dynamically from data):
      - Human
      - Bot
      - Organization  (only if present in events)

    'Unknown' is rejected at runtime — normalize_author_type() must resolve it upstream.
    """
    print("\n" + "=" * 60)
    print("Building State Machine by Author Type")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    commits, other_activity, all_repos = load_events()

    study_end_month = STUDY_END.strftime("%Y-%m")

    # Detect which author groups actually exist in the loaded events.
    # Order is fixed; only groups present in data are included.
    # "Unknown" is never accepted — normalize_author_type() guarantees it won't appear.
    types_in_data: set = set()
    for repo_months in commits.values():
        for month_events in repo_months.values():
            for c in month_events:
                types_in_data.add(c.get("author_type"))
    for repo_months in other_activity.values():
        for month_events in repo_months.values():
            for e in month_events:
                types_in_data.add(e.get("author_type"))

    if "Unknown" in types_in_data:
        raise ValueError(
            "author_type 'Unknown' found in events — fix upstream (09b / 15) before running."
        )

    # Organization author_type excluded: only 9 commits across 9 repos — insufficient for analysis.
    # Owner type=Organization (repo ownership) is analyzed separately in the owner_type machine.
    author_groups = [g for g in ["Human", "Bot"] if g in types_in_data]
    print(f"  Author groups detected in data: {author_groups}")

    group_repo_states: Dict[str, List[List[str]]] = defaultdict(list)
    monthly_rows = []

    for repo in sorted(all_repos):
        all_months = set(commits[repo].keys()) | set(other_activity[repo].keys())
        if not all_months:
            continue

        first_month = min(all_months)
        month_range = generate_month_range(first_month, study_end_month)

        for author_group in author_groups:
            repo_states_for_group = []

            for month in month_range:
                commits_in_month = commits[repo].get(month, [])
                other_in_month = other_activity[repo].get(month, [])

                filtered_commits = [
                    c for c in commits_in_month
                    if c.get("author_type") == author_group
                ]
                filtered_other = [
                    e for e in other_in_month
                    if e.get("author_type") == author_group
                ]

                state = determine_monthly_state_for_group(filtered_commits, filtered_other)
                repo_states_for_group.append(state)

                monthly_rows.append({
                    "author_type": author_group,
                    "repository": repo,
                    "month": month,
                    "state": state
                })

            group_repo_states[author_group].append(repo_states_for_group)

    with open(os.path.join(OUTPUT_DIR, "monthly_states_by_author_type.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["author_type", "repository", "month", "state"])
        writer.writeheader()
        writer.writerows(monthly_rows)
    print(f"  monthly_states_by_author_type.csv: {len(monthly_rows)} rows")

    transitions_rows, stats_rows = compute_transitions_and_stats(group_repo_states, "author_type")

    write_transitions(
        transitions_rows,
        os.path.join(OUTPUT_DIR, "state_transitions_by_author_type.csv"),
        extra_fields=["author_type"]
    )
    print(f"  state_transitions_by_author_type.csv: {len(transitions_rows)} rows")

    write_statistics(
        stats_rows,
        os.path.join(OUTPUT_DIR, "state_statistics_by_author_type.csv"),
        extra_fields=["author_type"]
    )
    print(f"  state_statistics_by_author_type.csv: {len(stats_rows)} rows")

# =========================
# STATE MACHINE BY OWNER TYPE (Organization vs Individual)
# =========================
def build_state_machine_by_owner_type():
    """
    Builds state machine grouping repositories by owner_type.
    For each owner_type, computes transitions and statistics using
    per-repository state sequences.
    """
    print("\n" + "=" * 60)
    print("Building State Machine by Owner Type (Organization vs Individual)")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    commits, other_activity, all_repos = load_events()

    repo_owner_type: Dict[str, str] = {}
    with open(FILTERED_REPOS, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            full_name = row.get("full_name", f"{row['Owner']}/{row['Name']}")
            repo_owner_type[full_name] = row.get("owner_type", "").strip()

    # Restrict to repos present in the filtered dataset (others were excluded upstream, e.g. time_to_event=0).
    excluded_from_dataset = [repo for repo in all_repos if repo not in repo_owner_type]
    if excluded_from_dataset:
        print(f"  [INFO] Skipping {len(excluded_from_dataset)} repo(s) not in survival dataset: "
              + ", ".join(sorted(excluded_from_dataset)))
    all_repos = {repo for repo in all_repos if repo in repo_owner_type}

    # Fail fast if any remaining repo has empty owner_type (data quality issue).
    missing_owner_type = [repo for repo in all_repos if not repo_owner_type.get(repo)]
    if missing_owner_type:
        raise ValueError(
            f"owner_type missing for {len(missing_owner_type)} repo(s) — "
            f"fix upstream before running:\n  " + "\n  ".join(sorted(missing_owner_type))
        )

    study_end_month = STUDY_END.strftime("%Y-%m")

    group_repo_states: Dict[str, List[List[str]]] = defaultdict(list)
    monthly_rows = []

    for repo in sorted(all_repos):
        all_months = set(commits[repo].keys()) | set(other_activity[repo].keys())
        if not all_months:
            continue

        owner_type = repo_owner_type[repo]
        first_month = min(all_months)
        month_range = generate_month_range(first_month, study_end_month)

        repo_states = []

        for month in month_range:
            state = determine_monthly_state(
                commits[repo].get(month, []),
                other_activity[repo].get(month, [])
            )
            repo_states.append(state)

            monthly_rows.append({
                "owner_type": owner_type,
                "repository": repo,
                "month": month,
                "state": state
            })

        group_repo_states[owner_type].append(repo_states)

    with open(os.path.join(OUTPUT_DIR, "monthly_states_by_owner_type.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["owner_type", "repository", "month", "state"])
        writer.writeheader()
        writer.writerows(monthly_rows)
    print(f"  monthly_states_by_owner_type.csv: {len(monthly_rows)} rows")

    transitions_rows, stats_rows = compute_transitions_and_stats(group_repo_states, "owner_type")

    write_transitions(
        transitions_rows,
        os.path.join(OUTPUT_DIR, "state_transitions_by_owner_type.csv"),
        extra_fields=["owner_type"]
    )
    print(f"  state_transitions_by_owner_type.csv: {len(transitions_rows)} rows")

    write_statistics(
        stats_rows,
        os.path.join(OUTPUT_DIR, "state_statistics_by_owner_type.csv"),
        extra_fields=["owner_type"]
    )
    print(f"  state_statistics_by_owner_type.csv: {len(stats_rows)} rows")

# =========================
# ENTRY POINT
# =========================
if __name__ == "__main__":
    build_state_machine()
    build_state_machine_by_author_type()
    build_state_machine_by_owner_type()