"""
apply_exclusion_criteria.py

Apply exclusion criteria from the research design to filter ROS packages dataset.

Exclusion Criteria:
1. Archived repositories
2. Too new repositories: Created in the last 6 months
3. Non-software/content-only repositories: No programming languages or very small size
4. Duplicates/forks: Exclude forked repositories
5. Not hosted on GitHub: Already satisfied by input dataset

Inputs:
  - out/final_repo_dataset.csv

Outputs:
  - out/filtered_repo_dataset.csv        (all repos after filtering)
  - out/exclusion_summary.csv            (all excluded repos with reasons)
  - out/exclusion_statistics.txt         (summary statistics)

Notes:
  - This script documents HOW MANY projects were excluded for EACH criterion
  - A repository can be excluded for multiple reasons (tracked separately)
"""

import os
import csv
import json
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from typing import Dict, Any, List, Tuple, Set

# =========================
# CONFIG
# =========================
INPUT_CSV = "out/final_repo_dataset.csv"
DATA_ROOT = "scripts/data/ros_robotics_data"
OUTPUT_FILTERED = "out/filtered_repo_dataset.csv"
OUTPUT_EXCLUDED = "out/exclusion_summary.csv"
OUTPUT_STATS = "out/exclusion_statistics.txt"

# Exclusion thresholds
INACTIVITY_MONTHS = 6
MIN_REPO_SIZE_KB = 1  # Minimum repo size to consider it software
MIN_LANGUAGES = 1     # Must have at least 1 programming language

CUTOFF_DATE = datetime(2026, 3, 3, 0, 0, 0)

# =========================
# HELPERS
# =========================
def safe_read_json(path: str) -> Any:
    try:
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def is_snapshot(obj: Any) -> bool:
    return isinstance(obj, dict) and "_meta" in obj and "data" in obj

def snapshot_data(obj: Any) -> Any:
    if is_snapshot(obj):
        return obj.get("data")
    return obj

def read_snapshot_file(repo_dir: str, filename: str) -> Any:
    obj = safe_read_json(os.path.join(repo_dir, filename))
    if obj is None:
        return None
    return snapshot_data(obj)

def parse_iso_date(date_str: str) -> datetime:
    """Parse ISO format date string"""
    try:
        # Handle both "2023-01-15T10:30:00Z" and "2023-01-15T10:30:00+00:00"
        if date_str.endswith("Z"):
            return datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        return datetime.fromisoformat(date_str)
    except Exception:
        return None

def is_within_months(date_str: str, months: int) -> bool:
    """Check if date is within N months from today"""
    if not date_str:
        return False
    
    date = parse_iso_date(date_str)
    if not date:
        return False
    
    # Make both datetimes timezone-naive for comparison
    if date.tzinfo is not None:
        date = date.replace(tzinfo=None)
    
    cutoff = CUTOFF_DATE - relativedelta(months=months)
    return date >= cutoff

def get_last_commit_date(owner: str, repo: str) -> str:
    """Get the most recent commit date from commits.json"""
    repo_dir = os.path.join(DATA_ROOT, f"{owner}__{repo}")
    commits = read_snapshot_file(repo_dir, "commits.json") or []
    
    if not isinstance(commits, list) or len(commits) == 0:
        return None
    
    # Commits are typically in reverse chronological order, but let's be safe
    latest = None
    for c in commits:
        date = c.get("date")
        if date:
            if latest is None:
                latest = date
            else:
                # Compare dates (ISO format)
                if date > latest:
                    latest = date
    
    return latest

def count_languages(languages_str: str) -> int:
    """Count number of programming languages from semicolon-separated string"""
    if not languages_str or not isinstance(languages_str, str):
        return 0
    langs = [l.strip() for l in languages_str.split(";") if l.strip()]
    return len(langs)

# =========================
# EXCLUSION CRITERIA FUNCTIONS
# =========================
def check_archived(row: Dict[str, Any]) -> Tuple[bool, str]:
    """Check if repository is archived"""
    # This would require reading from general_info.json
    # For now, archived flag might be in the dataset if available
    owner = row.get("Owner", "").strip()
    repo = row.get("Name", "").strip()
    
    if not owner or not repo:
        return False, None
    
    repo_dir = os.path.join(DATA_ROOT, f"{owner}__{repo}")
    general_info = read_snapshot_file(repo_dir, "general_info.json")
    
    if general_info and general_info.get("archived"):
        return True, "archived"
    
    return False, None

def check_inactive(row: Dict[str, Any]) -> Tuple[bool, str]:
    """Check if repository is inactive (no commits in last 6 months)"""
    owner = row.get("Owner", "").strip()
    repo = row.get("Name", "").strip()
    
    if not owner or not repo:
        return False, None
    
    last_commit = get_last_commit_date(owner, repo)
    
    if last_commit and not is_within_months(last_commit, INACTIVITY_MONTHS):
        return True, f"inactive_{INACTIVITY_MONTHS}mo"
    
    return False, None

def check_non_software(row: Dict[str, Any]) -> Tuple[bool, str]:
    """Check if repository is non-software (content-only)"""
    reasons = []
    
    # Check 1: No programming languages
    languages_str = row.get("languages", "").strip()
    num_languages = count_languages(languages_str)
    
    if num_languages < MIN_LANGUAGES:
        reasons.append("no_programming_languages")
    
    # Check 2: Very small repository size (likely documentation/content only)
    try:
        size_kb = int(row.get("Repository Size", 0) or 0)
        if size_kb < MIN_REPO_SIZE_KB:
            reasons.append("too_small")
    except (ValueError, TypeError):
        pass
    
    # Check 3: Keywords in description suggesting content-only
    description = (row.get("Description", "") or "").lower()
    content_keywords = [
        "documentation", "tutorial", "webinar", "slides",
        "template", "example data", "benchmark dataset",
        "dataset", "data repository", "archive"
    ]
    
    # Only flag if description explicitly mentions content/docs only
    if description and any(kw in description for kw in content_keywords):
        # But only if also no languages detected
        if num_languages == 0:
            reasons.append("content_only")
    
    if reasons:
        return True, ";".join(reasons)
    
    return False, None

def check_fork(row: Dict[str, Any]) -> Tuple[bool, str]:
    """Check if repository is a fork"""
    owner = row.get("Owner", "").strip()
    repo = row.get("Name", "").strip()
    
    if not owner or not repo:
        return False, None
    
    repo_dir = os.path.join(DATA_ROOT, f"{owner}__{repo}")
    general_info = read_snapshot_file(repo_dir, "general_info.json")
    
    if general_info and general_info.get("fork"):
        return True, "is_fork"
    
    return False, None

def get_first_commit_date(owner: str, repo: str) -> str:
    """Get the oldest commit date from commits.json"""
    repo_dir = os.path.join(DATA_ROOT, f"{owner}__{repo}")
    commits = read_snapshot_file(repo_dir, "commits.json") or []
    
    if not isinstance(commits, list) or len(commits) == 0:
        return None
    
    # Find the oldest commit date
    oldest = None
    for c in commits:
        commit_date_str = None
        
        # Handle multiple date formats:
        # 1. Simplified format: {date: "...", author: "..."}
        # 2. Nested format: {commit: {author: {date: "..."}}}
        # 3. Alternative format: {author_date: "..."}
        
        if 'date' in c:
            commit_date_str = c['date']
        elif 'author_date' in c:
            commit_date_str = c['author_date']
        elif 'commit' in c and isinstance(c['commit'], dict):
            if 'author' in c['commit'] and isinstance(c['commit']['author'], dict):
                commit_date_str = c['commit']['author'].get('date')
        
        if commit_date_str:
            # Compare dates (ISO format strings compare correctly)
            if oldest is None or commit_date_str < oldest:
                oldest = commit_date_str
    
    return oldest

def check_too_new(row: Dict[str, Any]) -> Tuple[bool, str]:
    """Check if repository's first commit was recently (within last N months)"""
    owner = row.get("Owner", "").strip()
    repo = row.get("Name", "").strip()
    
    if not owner or not repo:
        return False, None
    
    first_commit = get_first_commit_date(owner, repo)
    
    if first_commit and is_within_months(first_commit, INACTIVITY_MONTHS):
        return True, f"created_recently_{INACTIVITY_MONTHS}mo"
    
    return False, None

# =========================
# MAIN FILTERING LOGIC
# =========================
def apply_exclusions(row: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Apply all exclusion criteria to a single repository.
    Returns (is_excluded, list_of_exclusion_reasons)
    """
    exclusions = []
    
    # Check archived
    excluded, reason = check_archived(row)
    if excluded and reason:
        exclusions.append(reason)
    
    # Check too new
    excluded, reason = check_too_new(row)
    if excluded and reason:
        exclusions.append(reason)
    
    # Check non-software
    excluded, reason = check_non_software(row)
    if excluded and reason:
        exclusions.append(reason)
    
    # Check fork
    excluded, reason = check_fork(row)
    if excluded and reason:
        exclusions.append(reason)
    
    is_excluded = len(exclusions) > 0
    return is_excluded, exclusions

def main():
    # Load input CSV
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"Input file not found: {INPUT_CSV}")
    
    with open(INPUT_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    total_repos = len(rows)
    print(f"[INFO] Processing {total_repos} repositories...")
    
    # Apply exclusions
    included_rows = []
    excluded_rows = []
    
    # Statistics counters
    exclusion_counts = {}  # criterion -> count
    
    for i, row in enumerate(rows, start=1):
        if i % 50 == 0:
            print(f"[PROGRESS] {i}/{total_repos}")
        
        is_excluded, reasons = apply_exclusions(row)
        
        if is_excluded:
            # Add exclusion column to row
            row["exclusion_reasons"] = ";".join(reasons)
            excluded_rows.append(row)
            
            # Track statistics
            for reason in reasons:
                exclusion_counts[reason] = exclusion_counts.get(reason, 0) + 1
        else:
            included_rows.append(row)
    
    # Write filtered dataset
    os.makedirs(os.path.dirname(OUTPUT_FILTERED) or ".", exist_ok=True)
    
    fieldnames = list(rows[0].keys()) if rows else []
    if "exclusion_reasons" not in fieldnames:
        fieldnames.append("exclusion_reasons")
    
    with open(OUTPUT_FILTERED, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(included_rows)
    
    # Write excluded repos summary
    excluded_fieldnames = fieldnames.copy()
    with open(OUTPUT_EXCLUDED, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=excluded_fieldnames)
        writer.writeheader()
        writer.writerows(excluded_rows)
    
    # Write statistics
    with open(OUTPUT_STATS, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("EXCLUSION CRITERIA STATISTICS\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"Total repositories in input:      {total_repos}\n")
        f.write(f"Included repositories:            {len(included_rows)} ({100*len(included_rows)/total_repos:.1f}%)\n")
        f.write(f"Excluded repositories:            {len(excluded_rows)} ({100*len(excluded_rows)/total_repos:.1f}%)\n")
        f.write("\n")
        
        f.write("EXCLUSION BREAKDOWN (by criterion):\n")
        f.write("-" * 70 + "\n")
        
        # Sort by count (descending)
        sorted_exclusions = sorted(exclusion_counts.items(), key=lambda x: x[1], reverse=True)
        for criterion, count in sorted_exclusions:
            pct = 100 * count / total_repos
            f.write(f"  {criterion:.<40} {count:>5} ({pct:>5.1f}%)\n")
        
        f.write("\n")
        f.write("NOTES:\n")
        f.write("-" * 70 + "\n")
        f.write(f"  - created_recently_{INACTIVITY_MONTHS}mo: Repository created in last {INACTIVITY_MONTHS} months\n")
        f.write(f"  - archived: Repository marked as archived on GitHub\n")
        f.write(f"  - is_fork: Repository is a fork of another repository\n")
        f.write(f"  - no_programming_languages: No detected programming languages\n")
        f.write(f"  - too_small: Repository size < {MIN_REPO_SIZE_KB} KB\n")
        f.write(f"  - content_only: Appears to be documentation/content repository\n")
        f.write("\nNote: A repository can be excluded for multiple reasons.\n")
        f.write("=" * 70 + "\n")
    
    # Print summary
    print("\n" + "=" * 70)
    print("EXCLUSION SUMMARY")
    print("=" * 70)
    print(f"Total repositories:     {total_repos}")
    print(f"Included:               {len(included_rows)} ({100*len(included_rows)/total_repos:.1f}%)")
    print(f"Excluded:               {len(excluded_rows)} ({100*len(excluded_rows)/total_repos:.1f}%)")
    print("\nExclusion by criterion:")
    for criterion, count in sorted_exclusions:
        pct = 100 * count / total_repos
        print(f"  {criterion:.<40} {count:>5} ({pct:>5.1f}%)")
    print("=" * 70)
    
    print(f"\n[OK] Filtered dataset saved to: {OUTPUT_FILTERED}")
    print(f"[OK] Exclusion summary saved to: {OUTPUT_EXCLUDED}")
    print(f"[OK] Statistics saved to: {OUTPUT_STATS}")

if __name__ == "__main__":
    main()
