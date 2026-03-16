"""
09b_update_author_types_all.py

Same as 09b_update_author_types.py but processes ALL repository directories
in ros_robotics_data/ — not just repos in survival_repo_dataset.csv.

Use this to ensure complete data integrity across the entire collected dataset,
including repos that were excluded from the survival analysis.

Skips commits/issues/PRs that already have author_type set (idempotent).
Uses the same cache file as 09b to avoid redundant API calls.
"""

import os
import json
import time
import csv
from pathlib import Path
from datetime import datetime, timezone

import requests
from dotenv import load_dotenv

load_dotenv()

SCRIPT_DIR = Path(__file__).parent
DATA_DIR   = SCRIPT_DIR / "data" / "ros_robotics_data"
CACHE_FILE = SCRIPT_DIR / "data" / "author_type_cache.json"

TOKEN = os.getenv("GITHUB_TOKEN")
if not TOKEN:
    raise ValueError("GITHUB_TOKEN not found. Set GITHUB_TOKEN in your environment (.env).")

HEADERS = {
    "Authorization": f"Bearer {TOKEN}",
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
}

REQUEST_SLEEP = 0.1


def load_cache():
    if CACHE_FILE.exists():
        with open(CACHE_FILE) as f:
            return json.load(f)
    return {}


def save_cache(cache):
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)


def fetch_user_type(login, cache):
    if not login:
        return None
    if login in cache:
        return cache[login]
    url = f"https://api.github.com/users/{login}"
    try:
        time.sleep(REQUEST_SLEEP)
        r = requests.get(url, headers=HEADERS, timeout=30)
        if r.status_code == 200:
            user_type = r.json().get("type", "User")
            cache[login] = user_type
            return user_type
        elif r.status_code == 404:
            cache[login] = "Unknown"
            return "Unknown"
        elif r.status_code == 403:
            reset = int(r.headers.get("X-RateLimit-Reset", 0))
            wait  = max(reset - time.time(), 60)
            print(f"  Rate limited — waiting {wait:.0f}s...")
            time.sleep(wait)
            return fetch_user_type(login, cache)
        else:
            print(f"  Warning: {r.status_code} for {login}")
            return None
    except Exception as e:
        print(f"  Error fetching {login}: {e}")
        return None


def update_commits(repo_dir, cache):
    path = repo_dir / "commits.json"
    if not path.exists(): return 0, 0
    with open(path) as f: raw = json.load(f)
    commits = raw.get("data", raw)
    if not isinstance(commits, list): return 0, 0
    updated = 0
    for c in commits:
        if "author_type" in c and c["author_type"] is not None: continue
        login = c.get("author_login")
        c["author_type"] = fetch_user_type(login, cache) if login else "Unknown"
        updated += 1
    if updated:
        if isinstance(raw, dict) and "data" in raw:
            raw["_meta"]["author_types_updated_at"] = datetime.now(timezone.utc).isoformat()
            raw["data"] = commits
            with open(path, "w") as f: json.dump(raw, f, indent=2, ensure_ascii=False)
        else:
            with open(path, "w") as f: json.dump(commits, f, indent=2, ensure_ascii=False)
    return updated, len(commits)


def update_issues(repo_dir, cache):
    path = repo_dir / "issues.json"
    if not path.exists(): return 0, 0
    with open(path) as f: raw = json.load(f)
    issues = raw.get("data", raw)
    if not isinstance(issues, list): return 0, 0
    updated = 0
    for item in issues:
        if "author_type" in item and item["author_type"] is not None: continue
        login = item.get("author")
        item["author_type"] = fetch_user_type(login, cache) if login else "Unknown"
        updated += 1
    if updated:
        if isinstance(raw, dict) and "data" in raw:
            raw["_meta"]["author_types_updated_at"] = datetime.now(timezone.utc).isoformat()
            raw["data"] = issues
            with open(path, "w") as f: json.dump(raw, f, indent=2, ensure_ascii=False)
        else:
            with open(path, "w") as f: json.dump(issues, f, indent=2, ensure_ascii=False)
    return updated, len(issues)


def update_prs(repo_dir, cache):
    path = repo_dir / "pull_requests.json"
    if not path.exists(): return 0, 0
    with open(path) as f: raw = json.load(f)
    prs = raw.get("data", raw)
    if not isinstance(prs, list): return 0, 0
    updated = 0
    for item in prs:
        if "author_type" in item and item["author_type"] is not None: continue
        login = item.get("user") or item.get("author")
        item["author_type"] = fetch_user_type(login, cache) if login else "Unknown"
        updated += 1
    if updated:
        if isinstance(raw, dict) and "data" in raw:
            raw["_meta"]["author_types_updated_at"] = datetime.now(timezone.utc).isoformat()
            raw["data"] = prs
            with open(path, "w") as f: json.dump(raw, f, indent=2, ensure_ascii=False)
        else:
            with open(path, "w") as f: json.dump(prs, f, indent=2, ensure_ascii=False)
    return updated, len(prs)


def main():
    print("=" * 60)
    print("UPDATING AUTHOR TYPES — ALL REPOS (including non-dataset)")
    print("=" * 60)

    repo_dirs = sorted(d for d in DATA_DIR.iterdir() if d.is_dir())
    print(f"\nFound {len(repo_dirs)} repo directories")

    cache = load_cache()
    print(f"Loaded {len(cache)} cached author types\n")

    total_c = total_c_upd = 0
    total_i = total_i_upd = 0
    total_p = total_p_upd = 0

    for idx, repo_dir in enumerate(repo_dirs):
        c_upd, c_tot = update_commits(repo_dir, cache)
        i_upd, i_tot = update_issues(repo_dir, cache)
        p_upd, p_tot = update_prs(repo_dir, cache)

        total_c     += c_tot;   total_c_upd += c_upd
        total_i     += i_tot;   total_i_upd += i_upd
        total_p     += p_tot;   total_p_upd += p_upd

        if (idx + 1) % 50 == 0:
            print(f"  [{idx+1}/{len(repo_dirs)}] {repo_dir.name}")
            save_cache(cache)

    save_cache(cache)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Commits: {total_c_upd:,} updated / {total_c:,} total")
    print(f"Issues:  {total_i_upd:,} updated / {total_i:,} total")
    print(f"PRs:     {total_p_upd:,} updated / {total_p:,} total")
    print(f"Cache:   {len(cache):,} unique logins")

    from collections import Counter
    dist = Counter(cache.values())
    print("\nAuthor type distribution in cache:")
    for t, n in dist.most_common():
        print(f"  {t:<15} {n:,}")


if __name__ == "__main__":
    main()
