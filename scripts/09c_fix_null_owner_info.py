"""
09c_fix_null_owner_info.py

Fixes owner_info.json files where 'data' is null — which happens when a
repository was transferred to a new owner and the original owner account
no longer exists on GitHub.

Strategy:
  1. Scan all repo directories for owner_info.json with data=null
  2. Read general_info.json to get the current full_name (e.g. koide3/fast_gicp)
  3. Extract the current owner login from full_name
  4. Fetch /users/{current_owner} from GitHub API
  5. Overwrite owner_info.json with the resolved data

This ensures owner_type is never empty in the final dataset.
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

TOKEN = os.getenv("GITHUB_TOKEN")
if not TOKEN:
    raise ValueError("GitHub token not found. Set GITHUB_TOKEN in your environment (.env).")

HEADERS = {
    "Authorization": f"Bearer {TOKEN}",
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
}


def fetch_owner_data(login: str) -> dict | None:
    """Fetch /users/{login} from GitHub API. Returns the JSON body or None."""
    url = f"https://api.github.com/users/{login}"
    time.sleep(REQUEST_SLEEP)
    resp = requests.get(url, headers=HEADERS, timeout=30)
    if resp.status_code == 200:
        return resp.json()
    if resp.status_code == 403:
        reset = int(resp.headers.get("X-RateLimit-Reset", 0))
        wait  = max(reset - time.time(), 60)
        print(f"  Rate limited — waiting {wait:.0f}s...")
        time.sleep(wait)
        return fetch_owner_data(login)
    print(f"  API returned {resp.status_code} for {login}")
    return None


def get_current_owner_from_general_info(repo_dir: Path) -> str | None:
    """Extract current owner login from general_info.json full_name field."""
    path = repo_dir / "general_info.json"
    if not path.exists():
        return None
    try:
        with open(path) as f:
            raw = json.load(f)
        info = raw.get("data", raw)
        if not isinstance(info, dict):
            return None
        full_name = info.get("full_name", "")
        if "/" in full_name:
            return full_name.split("/")[0]
    except Exception as e:
        print(f"  Error reading {path}: {e}")
    return None


def main():
    print("=" * 60)
    print("Fixing null owner_info.json entries")
    print("=" * 60)

    fixed   = 0
    skipped = 0
    failed  = 0

    for repo_dir in sorted(DATA_DIR.iterdir()):
        if not repo_dir.is_dir():
            continue

        owner_info_path = repo_dir / "owner_info.json"
        if not owner_info_path.exists():
            continue

        with open(owner_info_path) as f:
            snapshot = json.load(f)

        if snapshot.get("data") is not None:
            skipped += 1
            continue  # Already has data, nothing to fix

        # data is null — resolve using current owner from general_info
        current_owner = get_current_owner_from_general_info(repo_dir)
        if not current_owner:
            print(f"  WARN: cannot determine current owner for {repo_dir.name} — skipping")
            failed += 1
            continue

        print(f"  Fixing {repo_dir.name}: fetching /users/{current_owner} ...")
        owner_data = fetch_owner_data(current_owner)
        if not owner_data:
            print(f"  FAILED: could not fetch owner data for {current_owner}")
            failed += 1
            continue

        # Update snapshot in place, preserving _meta
        snapshot["_meta"]["fetched_at"]    = datetime.now(timezone.utc).isoformat()
        snapshot["_meta"]["owner"]         = current_owner
        snapshot["_meta"]["resolved_from"] = "general_info.full_name (transferred repo)"
        snapshot["data"] = owner_data

        with open(owner_info_path, "w") as f:
            json.dump(snapshot, f, indent=2, ensure_ascii=False)

        print(f"    → owner_type = {owner_data.get('type')} ({current_owner})")
        fixed += 1

    print(f"\nDone. Fixed: {fixed} | Already OK: {skipped} | Failed: {failed}")
    if failed:
        print("  Re-run script 10 after fixing failures manually.")


if __name__ == "__main__":
    main()
