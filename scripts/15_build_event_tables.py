"""
15_build_event_tables.py

Build event tables for survival analysis following Ait et al. (2022) methodology.
Extracts events from repository JSON files into 5 CSV tables with canonical schema.

Event Tables (5 columns each):
  - timestamp: Event creation time (ISO format)
  - repository: Full name (owner/repo)
  - author: GitHub login
  - author_type: 'Human', 'Bot', or 'Organization' (never 'Unknown')
  - unique_id: Asset identifier (e.g., commit SHA)

Tables generated:
  - out/events/commits_events.csv
  - out/events/issues_events.csv
  - out/events/pull_requests_events.csv
  - out/events/comments_events.csv
  - out/events/reviews_events.csv

Inputs:
  - out/filtered_repo_dataset.csv (list of repos to process)
  - scripts/data/ros_robotics_data/{owner}__{repo}/*.json

References:
  - Ait et al. (2022) "An Empirical Study on the Survival Rate of GitHub Projects"
"""

import os
import csv
import json
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

# =========================
# CONFIG
# =========================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# Use survival dataset (includes inactive repos for death events)
INPUT_CSV = os.path.join(PROJECT_ROOT, "out/survival_repo_dataset.csv")
DATA_DIR = os.path.join(SCRIPT_DIR, "data/ros_robotics_data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "out/events")

# =========================
# HELPERS
# =========================
def get_author_type(item: dict, username: str) -> str:
    """
    Classify author type using the GitHub API 'type' field exclusively.
    No regex heuristics — API is the single source of truth (Ait et al. 2022).

    GitHub API values and their mapping:
      'Bot'          → 'Bot'
      'User'         → 'Human'
      'Organization' → 'Organization'
      'Unknown'      → 'Human'  (account deleted; was a real contributor)
      missing/None   → 'Human'  (no linked GitHub account; unverified email commit)

    Returns: 'Human', 'Bot', or 'Organization' — never 'Unknown'.

    Pre-condition: 09b_update_author_types.py must have been run to populate
    the author_type field in all JSON files before calling this script.
    """
    api_type = item.get("author_type")

    if api_type == "Bot":
        return "Bot"
    if api_type == "Organization":
        return "Organization"
    # "User", "Unknown", None, or any unrecognised value → Human
    return "Human"

def safe_read_json(path: str) -> Any:
    """Safely read JSON file, handling snapshots."""
    try:
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            return None
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            # Handle snapshot format
            if isinstance(data, dict) and "data" in data:
                return data["data"]
            return data
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return None

def get_repo_dir(owner: str, repo: str) -> str:
    """Get repository data directory path."""
    return os.path.join(DATA_DIR, f"{owner}__{repo}")

def parse_timestamp(ts: str) -> Optional[str]:
    """Parse and normalize timestamp to ISO format."""
    if not ts:
        return None
    try:
        # Handle various formats
        if ts.endswith("Z"):
            ts = ts.replace("Z", "+00:00")
        dt = datetime.fromisoformat(ts)
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    except:
        return None

# =========================
# EVENT EXTRACTORS
# =========================
def extract_commits(repo_dir: str, full_name: str) -> List[Dict]:
    """Extract commit events from commits.json."""
    events = []
    data = safe_read_json(os.path.join(repo_dir, "commits.json"))
    
    if not data or not isinstance(data, list):
        return events
    
    for commit in data:
        author_login = commit.get("author_login") or commit.get("author", "unknown")
        timestamp = parse_timestamp(commit.get("date"))
        
        if timestamp:
            events.append({
                "timestamp": timestamp,
                "repository": full_name,
                "author": author_login,
                "author_type": get_author_type(commit, author_login),
                "unique_id": commit.get("sha", "")
            })
    
    return events

def extract_issues(repo_dir: str, full_name: str) -> List[Dict]:
    """Extract issue events from issues.json."""
    events = []
    data = safe_read_json(os.path.join(repo_dir, "issues.json"))
    
    if not data or not isinstance(data, list):
        return events
    
    for issue in data:
        # Get author from user object or direct field
        user = issue.get("user", {})
        if isinstance(user, dict):
            author_login = user.get("login", "unknown")
        else:
            author_login = issue.get("author") or str(user) if user else "unknown"
        
        timestamp = parse_timestamp(issue.get("created_at"))
        
        if timestamp:
            events.append({
                "timestamp": timestamp,
                "repository": full_name,
                "author": author_login,
                "author_type": get_author_type(issue, author_login),
                "unique_id": str(issue.get("number", ""))
            })
    
    return events

def extract_pull_requests(repo_dir: str, full_name: str) -> List[Dict]:
    """Extract pull request events from pull_requests.json."""
    events = []
    data = safe_read_json(os.path.join(repo_dir, "pull_requests.json"))
    
    if not data or not isinstance(data, list):
        return events
    
    for pr in data:
        # Get author from user object
        user = pr.get("user", {})
        if isinstance(user, dict):
            author_login = user.get("login", "unknown")
        else:
            author_login = pr.get("author") or str(user) if user else "unknown"
        
        timestamp = parse_timestamp(pr.get("created_at"))
        
        if timestamp:
            events.append({
                "timestamp": timestamp,
                "repository": full_name,
                "author": author_login,
                "author_type": get_author_type(pr, author_login),
                "unique_id": str(pr.get("number", ""))
            })
    
    return events

def extract_comments(repo_dir: str, full_name: str) -> List[Dict]:
    """Extract comment events from issue_comments.json."""
    events = []
    
    # Read from dedicated comments file (created by 17_fetch_comments_and_reviews.py)
    comments_file = os.path.join(repo_dir, "issue_comments.json")
    data = safe_read_json(comments_file)
    
    if not data or not isinstance(data, list):
        return events
    
    for comment in data:
        author_login = comment.get("author") or "unknown"
        timestamp = parse_timestamp(comment.get("created_at"))
        
        if timestamp:
            events.append({
                "timestamp": timestamp,
                "repository": full_name,
                "author": author_login,
                "author_type": get_author_type(comment, author_login),
                "unique_id": str(comment.get("id", ""))
            })
    
    return events

def extract_reviews(repo_dir: str, full_name: str) -> List[Dict]:
    """Extract code review events from pr_reviews.json."""
    events = []
    
    # Read from dedicated reviews file (created by 17_fetch_comments_and_reviews.py)
    reviews_file = os.path.join(repo_dir, "pr_reviews.json")
    data = safe_read_json(reviews_file)
    
    if not data or not isinstance(data, list):
        return events
    
    for review in data:
        author_login = review.get("author") or "unknown"
        timestamp = parse_timestamp(review.get("submitted_at"))
        
        if timestamp:
            events.append({
                "timestamp": timestamp,
                "repository": full_name,
                "author": author_login,
                "author_type": get_author_type(review, author_login),
                "unique_id": f"PR{review.get('pr_number', '')}-{review.get('state', '')}"
            })
    
    return events

# =========================
# MAIN
# =========================
def main():
    print("=" * 60)
    print("Building Event Tables for Survival Analysis")
    print("Following Ait et al. (2022) methodology")
    print("=" * 60)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load filtered repos
    repos = []
    with open(INPUT_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            repos.append({
                "name": row["Name"],
                "owner": row["Owner"],
                "full_name": row.get("full_name", f"{row['Owner']}/{row['Name']}")
            })
    
    print(f"Processing {len(repos)} repositories...")
    
    # Initialize event lists
    all_commits = []
    all_issues = []
    all_prs = []
    all_comments = []
    all_reviews = []
    
    # Process each repository
    for i, repo in enumerate(repos):
        owner = repo["owner"]
        name = repo["name"]
        full_name = repo["full_name"]
        repo_dir = get_repo_dir(owner, name)
        
        if not os.path.exists(repo_dir):
            print(f"  [{i+1}/{len(repos)}] Skipping {full_name} - no data directory")
            continue
        
        # Extract events
        commits = extract_commits(repo_dir, full_name)
        issues = extract_issues(repo_dir, full_name)
        prs = extract_pull_requests(repo_dir, full_name)
        comments = extract_comments(repo_dir, full_name)
        reviews = extract_reviews(repo_dir, full_name)
        
        all_commits.extend(commits)
        all_issues.extend(issues)
        all_prs.extend(prs)
        all_comments.extend(comments)
        all_reviews.extend(reviews)
        
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(repos)}] Processed {full_name}")
    
    # Write event tables
    fieldnames = ["timestamp", "repository", "author", "author_type", "unique_id"]
    
    def write_events(filename: str, events: List[Dict]):
        path = os.path.join(OUTPUT_DIR, filename)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            # Sort by timestamp
            sorted_events = sorted(events, key=lambda x: x["timestamp"])
            writer.writerows(sorted_events)
        print(f"  Written {filename}: {len(events)} events")
    
    print("\nWriting event tables...")
    write_events("commits_events.csv", all_commits)
    write_events("issues_events.csv", all_issues)
    write_events("pull_requests_events.csv", all_prs)
    write_events("comments_events.csv", all_comments)
    write_events("reviews_events.csv", all_reviews)
    
    # Summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total commits:        {len(all_commits)}")
    print(f"Total issues:         {len(all_issues)}")
    print(f"Total pull requests:  {len(all_prs)}")
    print(f"Total comments:       {len(all_comments)}")
    print(f"Total reviews:        {len(all_reviews)}")
    
    # Bot statistics
    human_commits = sum(1 for e in all_commits if e["author_type"] == "Human")
    bot_commits = sum(1 for e in all_commits if e["author_type"] == "Bot")
    print(f"\nCommit breakdown:")
    print(f"  Human:  {human_commits} ({100*human_commits/len(all_commits):.1f}%)" if all_commits else "  Human: 0")
    print(f"  Bot:    {bot_commits} ({100*bot_commits/len(all_commits):.1f}%)" if all_commits else "  Bot: 0")
    
    print("\nEvent tables saved to:", OUTPUT_DIR)
    print("=" * 60)

if __name__ == "__main__":
    main()
