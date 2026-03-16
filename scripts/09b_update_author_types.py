"""
09b_update_author_types.py

Updates existing commits.json files with author_type field from GitHub API.
This follows Ait et al. (2022) methodology using the GitHub API's native bot flag.

The GitHub API returns a "type" field for each user which can be:
  - "User": Regular human user
  - "Bot": Automated bot account  
  - "Organization": Organization account

This script:
1. Reads existing commits.json files
2. Collects unique author_logins
3. Fetches user type from GitHub API
4. Updates commits with author_type field
5. Saves updated commits.json

References:
  - Ait et al. (2022) "An Empirical Study on the Survival Rate of GitHub Projects"
    Section 3.2: "SourceCred relies on the GitHub API, which uniquely identify 
    users by their username... the API flags this kind of users [bots]"
"""

import os
import sys
import json
import time
import csv
from pathlib import Path
from datetime import datetime, timezone

import requests
from dotenv import load_dotenv

# =========================
# CONFIG
# =========================
load_dotenv()

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = SCRIPT_DIR / "data" / "ros_robotics_data"
INPUT_CSV = PROJECT_ROOT / "out" / "survival_repo_dataset.csv"

TOKEN = os.getenv("GITHUB_TOKEN")
if not TOKEN:
    raise ValueError("GitHub token not found. Set GITHUB_TOKEN in your environment (.env).")

HEADERS = {
    "Authorization": f"Bearer {TOKEN}",
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
}

REQUEST_SLEEP = 0.1  # seconds between API calls
CACHE_FILE = SCRIPT_DIR / "data" / "author_type_cache.json"

# =========================
# HELPERS
# =========================
def load_cache():
    """Load cached author types to avoid redundant API calls."""
    if CACHE_FILE.exists():
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_cache(cache):
    """Save author type cache."""
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)

def fetch_user_type(login, cache):
    """
    Fetch user type from GitHub API.
    Returns: "User", "Bot", "Organization", or None if not found.
    """
    if not login:
        return None
    
    # Check cache first
    if login in cache:
        return cache[login]
    
    url = f"https://api.github.com/users/{login}"
    try:
        time.sleep(REQUEST_SLEEP)
        response = requests.get(url, headers=HEADERS, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            user_type = data.get("type", "User")
            cache[login] = user_type
            return user_type
        elif response.status_code == 404:
            # User not found (deleted account, etc.)
            cache[login] = "Unknown"
            return "Unknown"
        elif response.status_code == 403:
            # Rate limit - wait and retry
            reset_time = int(response.headers.get("X-RateLimit-Reset", 0))
            wait_time = max(reset_time - time.time(), 60)
            print(f"  Rate limited. Waiting {wait_time:.0f}s...")
            time.sleep(wait_time)
            return fetch_user_type(login, cache)
        else:
            print(f"  Warning: API returned {response.status_code} for {login}")
            return None
    except Exception as e:
        print(f"  Error fetching user {login}: {e}")
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
    
    # Fuzzy match
    owner_l, name_l = owner.lower(), name.lower()
    for d in DATA_DIR.iterdir():
        if not d.is_dir():
            continue
        parts = d.name.split("__")
        if len(parts) == 2 and parts[0].lower() == owner_l and parts[1].lower() == name_l:
            return d
    return None

def update_commits_json(repo_dir, cache):
    """
    Update commits.json with author_type field.
    Returns: (updated_count, total_count, unique_authors)
    """
    commits_file = repo_dir / "commits.json"
    if not commits_file.exists():
        return 0, 0, set()
    
    with open(commits_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    commits = data.get("data", [])
    if not commits:
        return 0, 0, set()
    
    unique_authors = set()
    updated = 0
    
    for commit in commits:
        author_login = commit.get("author_login")
        if author_login:
            unique_authors.add(author_login)
        
        # Skip if already has author_type
        if "author_type" in commit:
            continue
        
        # Fetch and set author type
        if author_login:
            author_type = fetch_user_type(author_login, cache)
            commit["author_type"] = author_type or "Unknown"
        else:
            commit["author_type"] = "Unknown"
        updated += 1
    
    # Update metadata
    if "_meta" not in data:
        data["_meta"] = {}
    data["_meta"]["author_types_updated_at"] = datetime.now(timezone.utc).isoformat()
    
    # Save updated file
    with open(commits_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    return updated, len(commits), unique_authors

def update_issues_json(repo_dir, cache):
    """Update issues.json with author_type field."""
    issues_file = repo_dir / "issues.json"
    if not issues_file.exists():
        return 0, 0
    
    with open(issues_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    issues = data.get("data", [])
    if not issues:
        return 0, 0
    
    updated = 0
    for issue in issues:
        if "author_type" in issue:
            continue
        author = issue.get("author")
        if author:
            author_type = fetch_user_type(author, cache)
            issue["author_type"] = author_type or "Unknown"
        else:
            issue["author_type"] = "Unknown"
        updated += 1
    
    if "_meta" not in data:
        data["_meta"] = {}
    data["_meta"]["author_types_updated_at"] = datetime.now(timezone.utc).isoformat()
    
    with open(issues_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    return updated, len(issues)

def update_prs_json(repo_dir, cache):
    """Update pull_requests.json with author_type field."""
    prs_file = repo_dir / "pull_requests.json"
    if not prs_file.exists():
        return 0, 0
    
    with open(prs_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    prs = data.get("data", [])
    if not prs:
        return 0, 0
    
    updated = 0
    for pr in prs:
        if "author_type" in pr:
            continue
        author = pr.get("user") or pr.get("author")
        if author:
            author_type = fetch_user_type(author, cache)
            pr["author_type"] = author_type or "Unknown"
        else:
            pr["author_type"] = "Unknown"
        updated += 1
    
    if "_meta" not in data:
        data["_meta"] = {}
    data["_meta"]["author_types_updated_at"] = datetime.now(timezone.utc).isoformat()
    
    with open(prs_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    return updated, len(prs)

# =========================
# MAIN
# =========================
def main():
    print("=" * 60)
    print("UPDATING AUTHOR TYPES FROM GITHUB API")
    print("Following Ait et al. (2022) methodology")
    print("=" * 60)
    
    # Load repos
    repos = []
    with open(INPUT_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            repos.append({
                "owner": row.get("Owner", ""),
                "name": row.get("Name", ""),
            })
    
    print(f"\nFound {len(repos)} repositories to process")
    
    # Load cache
    cache = load_cache()
    print(f"Loaded {len(cache)} cached author types")
    
    # Process each repo
    total_commits_updated = 0
    total_commits = 0
    total_issues_updated = 0
    total_issues = 0
    total_prs_updated = 0
    total_prs = 0
    all_authors = set()
    
    for i, repo in enumerate(repos):
        owner, name = repo["owner"], repo["name"]
        repo_dir = find_repo_dir(owner, name)
        
        if not repo_dir:
            continue
        
        if (i + 1) % 50 == 0:
            print(f"  Processing {i + 1}/{len(repos)}: {owner}/{name}")
            save_cache(cache)  # Save progress
        
        # Update commits
        c_updated, c_total, authors = update_commits_json(repo_dir, cache)
        total_commits_updated += c_updated
        total_commits += c_total
        all_authors.update(authors)
        
        # Update issues
        i_updated, i_total = update_issues_json(repo_dir, cache)
        total_issues_updated += i_updated
        total_issues += i_total
        
        # Update PRs
        p_updated, p_total = update_prs_json(repo_dir, cache)
        total_prs_updated += p_updated
        total_prs += p_total
    
    # Final save
    save_cache(cache)
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\nCommits: {total_commits_updated}/{total_commits} updated")
    print(f"Issues:  {total_issues_updated}/{total_issues} updated")
    print(f"PRs:     {total_prs_updated}/{total_prs} updated")
    print(f"\nUnique authors encountered: {len(all_authors)}")
    print(f"Author types cached: {len(cache)}")
    
    # Count by type
    type_counts = {}
    for author, atype in cache.items():
        type_counts[atype] = type_counts.get(atype, 0) + 1
    
    print("\nAuthor types distribution:")
    for atype, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"  {atype}: {count} ({100*count/len(cache):.1f}%)")
    
    print(f"\nCache saved to: {CACHE_FILE}")
    print("\nDone!")

if __name__ == "__main__":
    main()
