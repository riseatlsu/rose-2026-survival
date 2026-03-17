"""
09f_refetch_incomplete_repo.py

Re-fetches specific JSON files that have empty/null data for a target repo.
Designed for introlab/rtabmap whose collection partially failed.

Files re-fetched only if their current data is empty/null:
  - languages.json    (needed for exclusion criteria — was {})
  - owner_info.json   (needed for owner_type — was null)
  - pull_requests.json (needed for events — was [])
  - contributors.json  (needed for contributors_count — was [])
  - forks.json         (needed for forks stats — was [])
  - weekly_commit_activity.json (needed for commit stats — was [])

After running this script:
    1. python3 scripts/10_build_final_repo_dataset.py  (rebuild dataset — may include repo now)
    2. python3 scripts/09b_update_author_types.py      (enrich new PR author_types)
    3. python3 scripts/15_build_event_tables.py
    4. python3 scripts/16_build_state_machine.py       (rebuild state-machine tables)
"""

import os
import json
import time
import requests
from pathlib import Path
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

SCRIPT_DIR    = Path(__file__).parent
DATA_DIR      = SCRIPT_DIR / "data" / "ros_robotics_data"
REQUEST_SLEEP = 0.2
PER_PAGE      = 100

# Target repo — change if reusing for other repos
TARGET_DIR    = DATA_DIR / "introlab__rtabmap"
TARGET_OWNER  = "introlab"
TARGET_REPO   = "rtabmap"

TOKEN = os.getenv("GITHUB_TOKEN")
if not TOKEN:
    raise ValueError("GITHUB_TOKEN not found in environment.")

HEADERS = {
    "Authorization": f"Bearer {TOKEN}",
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
}


def utc_now():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def fetch_rest(url, params=None):
    time.sleep(REQUEST_SLEEP)
    r = requests.get(url, headers=HEADERS, params=params, timeout=30)
    if r.status_code == 200:
        return r.json()
    if r.status_code == 403:
        reset = int(r.headers.get("X-RateLimit-Reset", 0))
        wait  = max(reset - time.time(), 60)
        print(f"  Rate limited — waiting {wait:.0f}s...")
        time.sleep(wait)
        return fetch_rest(url, params)
    print(f"  [{r.status_code}] {url}")
    return None


def is_empty(data):
    return data is None or data == {} or data == []


def load_snapshot(path):
    with open(path) as f:
        return json.load(f)


def save_snapshot(path, data, meta_extra=None):
    existing = load_snapshot(path)
    existing["_meta"]["fetched_at"]   = utc_now()
    existing["_meta"]["owner"]        = TARGET_OWNER
    existing["_meta"]["refetched_by"] = "09f_refetch_incomplete_repo.py"
    if meta_extra:
        existing["_meta"].update(meta_extra)
    existing["data"] = data
    with open(path, "w") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)


def fetch_languages():
    url = f"https://api.github.com/repos/{TARGET_OWNER}/{TARGET_REPO}/languages"
    return fetch_rest(url) or {}


def fetch_owner_info():
    url = f"https://api.github.com/users/{TARGET_OWNER}"
    r = fetch_rest(url)
    if r:
        return {
            "login":        r.get("login"),
            "type":         r.get("type"),
            "public_repos": r.get("public_repos"),
            "followers":    r.get("followers"),
            "company":      r.get("company"),
        }
    return None


def fetch_pull_requests():
    url = f"https://api.github.com/repos/{TARGET_OWNER}/{TARGET_REPO}/pulls"
    out, page = [], 1
    while True:
        r = fetch_rest(url, params={"state": "all", "per_page": PER_PAGE, "page": page})
        if not r or not isinstance(r, list) or len(r) == 0:
            break
        for pr in r:
            out.append({
                "id":         pr.get("id"),
                "number":     pr.get("number"),
                "state":      pr.get("state"),
                "title":      pr.get("title"),
                "created_at": pr.get("created_at"),
                "closed_at":  pr.get("closed_at"),
                "merged_at":  pr.get("merged_at"),
                "user":       (pr.get("user") or {}).get("login"),
            })
        print(f"    PRs page {page}: {len(r)} (total: {len(out)})")
        page += 1
    return out


def fetch_contributors():
    url = f"https://api.github.com/repos/{TARGET_OWNER}/{TARGET_REPO}/contributors"
    out, page = [], 1
    while True:
        r = fetch_rest(url, params={"per_page": PER_PAGE, "page": page})
        if not r or not isinstance(r, list) or len(r) == 0:
            break
        for c in r:
            out.append({"login": c.get("login"), "contributions": c.get("contributions")})
        page += 1
    return out


def fetch_forks():
    url = f"https://api.github.com/repos/{TARGET_OWNER}/{TARGET_REPO}/forks"
    out, page = [], 1
    while True:
        r = fetch_rest(url, params={"per_page": PER_PAGE, "page": page})
        if not r or not isinstance(r, list) or len(r) == 0:
            break
        for f in r:
            out.append({
                "forked_at": f.get("created_at"),
                "owner":     (f.get("owner") or {}).get("login"),
                "full_name": f.get("full_name"),
                "html_url":  f.get("html_url"),
            })
        print(f"    Forks page {page}: {len(r)} (total: {len(out)})")
        page += 1
    return out


def fetch_weekly_commit_activity(retries=6, delay=10):
    url = f"https://api.github.com/repos/{TARGET_OWNER}/{TARGET_REPO}/stats/commit_activity"
    for attempt in range(retries):
        r = requests.get(url, headers=HEADERS, timeout=30)
        if r.status_code == 200:
            return r.json() or []
        if r.status_code == 202:
            print(f"    Stats computing (202), waiting {delay}s... (attempt {attempt+1}/{retries})")
            time.sleep(delay)
        else:
            print(f"    [{r.status_code}] weekly_commit_activity")
            break
    return []


def main():
    print("=" * 60)
    print(f"Re-fetching incomplete data for {TARGET_OWNER}/{TARGET_REPO}")
    print("=" * 60)

    tasks = [
        ("languages.json",             fetch_languages),
        ("owner_info.json",            fetch_owner_info),
        ("pull_requests.json",         fetch_pull_requests),
        ("contributors.json",          fetch_contributors),
        ("forks.json",                 fetch_forks),
        ("weekly_commit_activity.json", fetch_weekly_commit_activity),
    ]

    for filename, fetcher in tasks:
        path = TARGET_DIR / filename
        if not path.exists():
            print(f"\nSKIP {filename}: file not found")
            continue

        snapshot = load_snapshot(path)
        current_data = snapshot.get("data")

        if not is_empty(current_data):
            size = len(current_data) if isinstance(current_data, (list, dict)) else "?"
            print(f"\nSKIP {filename}: already has data ({size} items)")
            continue

        print(f"\nFetching {filename} ...")
        new_data = fetcher()

        if is_empty(new_data):
            print(f"  WARNING: still empty after re-fetch — check API access")
        else:
            size = len(new_data) if isinstance(new_data, (list, dict)) else "?"
            print(f"  → {size} items fetched")

        save_snapshot(path, new_data)
        print(f"  Saved.")

    print("\n" + "=" * 60)
    print("Done. Verify results, then run:")
    print("  python3 scripts/10_build_final_repo_dataset.py")
    print("  python3 scripts/09b_update_author_types.py")
    print("  python3 scripts/15_build_event_tables.py")
    print("  python3 scripts/16_build_state_machine.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
