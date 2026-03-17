"""
09e_refetch_empty_commits.py

Re-fetches commits.json for repositories where data=[] (empty fetch).
Automatically detects all such repos — currently only introlab/rtabmap.

After re-fetching, run:
    python3 scripts/09b_update_author_types.py   (to enrich with author_type)
    python3 scripts/15_build_event_tables.py     (to rebuild event CSVs)
    python3 scripts/16_build_state_machine.py    (to rebuild state-machine tables)
"""

import os
import json
import time
import requests
from pathlib import Path
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

SCRIPT_DIR   = Path(__file__).parent
DATA_DIR     = SCRIPT_DIR / "data" / "ros_robotics_data"
REQUEST_SLEEP = 0.2
PER_PAGE      = 100

TOKEN = os.getenv("GITHUB_TOKEN")
if not TOKEN:
    raise ValueError("GITHUB_TOKEN not found in environment.")

HEADERS = {
    "Authorization": f"Bearer {TOKEN}",
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
}


def utc_now() -> str:
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


def get_current_owner_repo(repo_dir: Path):
    """Get current owner/repo from general_info.json full_name."""
    path = repo_dir / "general_info.json"
    if not path.exists():
        return None, None
    try:
        with open(path) as f:
            raw = json.load(f)
        info = raw.get("data", raw)
        full_name = info.get("full_name", "") if isinstance(info, dict) else ""
        if "/" in full_name:
            owner, repo = full_name.split("/", 1)
            return owner, repo
    except Exception as e:
        print(f"  Error reading {path}: {e}")
    return None, None


def fetch_commits(owner, repo):
    """
    Fetch all commits from GitHub API (paginated).
    Extracts author_type directly from the API response (author.type field)
    so 09b does not need to be run separately for these commits.
    """
    url  = f"https://api.github.com/repos/{owner}/{repo}/commits"
    out  = []
    page = 1
    while True:
        r = fetch_rest(url, params={"per_page": PER_PAGE, "page": page})
        if not r or not isinstance(r, list) or len(r) == 0:
            break
        for c in r:
            author_obj   = c.get("author") or {}
            author_login = author_obj.get("login")
            author_type  = author_obj.get("type")   # "User", "Bot", "Organization"
            out.append({
                "sha":          c.get("sha"),
                "author":       (c.get("commit") or {}).get("author", {}).get("name"),
                "author_login": author_login,
                "author_type":  author_type,          # included at fetch time
                "date":         (c.get("commit") or {}).get("author", {}).get("date"),
                "message":      (c.get("commit") or {}).get("message"),
            })
        print(f"    page {page}: {len(r)} commits (total so far: {len(out)})")
        page += 1
    return out


def find_targets():
    """
    Find repo directories where commits.json is missing or has data=[].
    Both cases need a re-fetch.
    """
    targets = []
    for repo_dir in sorted(DATA_DIR.iterdir()):
        if not repo_dir.is_dir():
            continue
        path = repo_dir / "commits.json"
        if not path.exists():
            targets.append((repo_dir, "missing"))
            continue
        try:
            with open(path) as f:
                raw = json.load(f)
            data = raw.get("data", raw)
            if isinstance(data, list) and len(data) == 0:
                targets.append((repo_dir, "empty"))
        except:
            pass
    return targets


def default_snapshot_meta(owner, repo, endpoint):
    return {
        "_meta": {
            "fetched_at":  utc_now(),
            "source":      "github_rest",
            "endpoint":    endpoint,
            "owner":       owner,
            "repo":        repo,
            "refetched_by": "09e_refetch_empty_commits.py",
        },
        "data": [],
    }


def main():
    print("=" * 60)
    print("Re-fetching commits for repos with missing/empty commits.json")
    print("=" * 60)

    targets = find_targets()
    if not targets:
        print("\nNo repos with missing/empty commits.json found — nothing to do.")
        return

    print(f"\nFound {len(targets)} repo(s) to re-fetch:")
    for repo_dir, reason in targets:
        print(f"  {repo_dir.name}  ({reason})")

    for repo_dir, reason in targets:
        owner, repo = get_current_owner_repo(repo_dir)
        if not owner:
            print(f"\n  SKIP {repo_dir.name}: cannot determine current owner")
            continue

        print(f"\nFetching {owner}/{repo} ...")
        commits = fetch_commits(owner, repo)
        print(f"  → {len(commits)} commits fetched")

        typed = sum(1 for c in commits if c.get("author_type"))
        print(f"  → {typed} with author_type from API ({len(commits)-typed} without — login=None)")

        if not commits:
            print(f"  WARNING: still 0 commits — repo may be empty or inaccessible")

        commits_path = repo_dir / "commits.json"
        if commits_path.exists():
            with open(commits_path) as f:
                existing = json.load(f)
            existing["_meta"]["fetched_at"]    = utc_now()
            existing["_meta"]["owner"]         = owner
            existing["_meta"]["refetched_by"]  = "09e_refetch_empty_commits.py"
            existing["data"] = commits
        else:
            existing = default_snapshot_meta(owner, repo, "/repos/{owner}/{repo}/commits")
            existing["data"] = commits

        with open(commits_path, "w") as f:
            json.dump(existing, f, indent=2, ensure_ascii=False)

        print(f"  Saved to {commits_path}")

    print("\nDone. Next steps:")
    print("  python3 scripts/09b_update_author_types.py")
    print("  python3 scripts/15_build_event_tables.py")
    print("  python3 scripts/16_build_state_machine.py")


if __name__ == "__main__":
    main()
