"""
17_fetch_comments_and_reviews.py

Supplementary data collection script to fetch issue comments and PR reviews.
These are required for full Ait et al. (2022) activity analysis.

Outputs (per repo):
  - scripts/data/ros_robotics_data/{owner}__{repo}/issue_comments.json
  - scripts/data/ros_robotics_data/{owner}__{repo}/pr_reviews.json

Estimated API calls per repo:
  - Issue comments: 1-3 calls (paginated)
  - PR reviews: 1-3 calls (paginated)

NOTE: This can take significant time for 467 repos.
With rate limiting, expect ~2-3 hours for full collection.
"""

import os
import sys
import csv
import json
import time
from datetime import datetime, timezone

import requests
from dotenv import load_dotenv

# =========================
# CONFIG
# =========================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

load_dotenv()

TOKEN = os.getenv("GITHUB_TOKEN")
if not TOKEN:
    raise ValueError("GitHub token not found. Set GITHUB_TOKEN in your environment (.env).")

HEADERS_REST = {
    "Authorization": f"Bearer {TOKEN}",
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
}

GRAPHQL_URL = "https://api.github.com/graphql"
HEADERS_GQL = {
    "Authorization": f"Bearer {TOKEN}",
    "Content-Type": "application/json",
}

DATA_DIR = os.path.join(SCRIPT_DIR, "data/ros_robotics_data")
# Use survival dataset to fetch for ALL repos (including inactive)
INPUT_CSV = os.path.join(PROJECT_ROOT, "out/survival_repo_dataset.csv")
REQUEST_SLEEP = 0.3
PER_PAGE = 100

# =========================
# HELPERS
# =========================
def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

def save_snapshot_json(data, filename, meta: dict):
    snapshot = {
        "_meta": {"fetched_at": utc_now_iso(), **(meta or {})},
        "data": data,
    }
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2, ensure_ascii=False, default=str)

def fetch_rest(url, params=None):
    time.sleep(REQUEST_SLEEP)
    r = requests.get(url, headers=HEADERS_REST, params=params)
    if r.status_code == 200:
        return r.json()
    if r.status_code == 403:
        # Rate limit
        reset = r.headers.get("X-RateLimit-Reset")
        if reset:
            wait = int(reset) - int(time.time()) + 5
            print(f"  Rate limited. Waiting {wait}s...")
            time.sleep(max(wait, 60))
            return fetch_rest(url, params)
    print(f"  [ERROR] {r.status_code}: {r.text[:200]}")
    return None

def fetch_gql(query: str, variables: dict):
    time.sleep(REQUEST_SLEEP)
    r = requests.post(GRAPHQL_URL, headers=HEADERS_GQL,
                      json={"query": query, "variables": variables})
    if r.status_code == 200:
        return r.json()
    print(f"  [GQL ERROR] {r.status_code}: {r.text[:200]}")
    return None

# =========================
# FETCH FUNCTIONS
# =========================
def fetch_issue_comments(owner: str, repo: str) -> list:
    """
    Fetch all issue comments for a repository.
    Uses REST API: GET /repos/{owner}/{repo}/issues/comments
    
    Returns list of {id, issue_url, user, created_at, updated_at}
    """
    comments = []
    page = 1
    
    while True:
        url = f"https://api.github.com/repos/{owner}/{repo}/issues/comments"
        params = {"per_page": PER_PAGE, "page": page}
        
        data = fetch_rest(url, params)
        if not data:
            break
        
        for c in data:
            user = c.get("user") or {}
            comments.append({
                "id": c.get("id"),
                "issue_url": c.get("issue_url"),
                "author": user.get("login"),
                "created_at": c.get("created_at"),
                "updated_at": c.get("updated_at"),
            })
        
        if len(data) < PER_PAGE:
            break
        page += 1
    
    return comments

def fetch_pr_reviews(owner: str, repo: str) -> list:
    """
    Fetch all PR reviews for a repository.
    Uses GraphQL to get reviews for all PRs efficiently.
    
    Returns list of {pr_number, author, state, submitted_at}
    """
    query = """
    query ($owner: String!, $repo: String!, $after: String) {
      repository(owner: $owner, name: $repo) {
        pullRequests(first: 50, after: $after) {
          edges {
            node {
              number
              reviews(first: 100) {
                nodes {
                  author { login }
                  state
                  submittedAt
                }
              }
            }
          }
          pageInfo { endCursor hasNextPage }
        }
      }
    }
    """
    
    reviews = []
    after = None
    
    while True:
        data = fetch_gql(query, {"owner": owner, "repo": repo, "after": after})
        if not data or "errors" in data:
            break
        
        try:
            prs = data["data"]["repository"]["pullRequests"]["edges"]
            for pr_edge in prs:
                pr = pr_edge["node"]
                pr_number = pr["number"]
                
                for review in pr.get("reviews", {}).get("nodes", []):
                    if review:
                        reviews.append({
                            "pr_number": pr_number,
                            "author": (review.get("author") or {}).get("login"),
                            "state": review.get("state"),
                            "submitted_at": review.get("submittedAt"),
                        })
            
            page_info = data["data"]["repository"]["pullRequests"]["pageInfo"]
            if page_info["hasNextPage"]:
                after = page_info["endCursor"]
            else:
                break
        except (KeyError, TypeError) as e:
            print(f"  [PARSE ERROR] {e}")
            break
    
    return reviews

# =========================
# MAIN
# =========================
def main():
    print("=" * 60)
    print("Fetching Comments and Reviews")
    print("=" * 60)
    
    # Load repos
    repos = []
    with open(INPUT_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            repos.append({
                "name": row["Name"],
                "owner": row["Owner"],
            })
    
    print(f"Processing {len(repos)} repositories...")
    print("Estimated time: 2-3 hours\n")
    
    total_comments = 0
    total_reviews = 0
    
    for i, repo in enumerate(repos):
        owner = repo["owner"]
        name = repo["name"]
        repo_dir = os.path.join(DATA_DIR, f"{owner}__{name}")
        
        print(f"[{i+1}/{len(repos)}] {owner}/{name}")
        
        if not os.path.exists(repo_dir):
            print(f"  Skipping - no data directory")
            continue
        
        # Check if already fetched
        comments_file = os.path.join(repo_dir, "issue_comments.json")
        reviews_file = os.path.join(repo_dir, "pr_reviews.json")
        
        if os.path.exists(comments_file) and os.path.exists(reviews_file):
            print(f"  Already fetched, skipping")
            continue
        
        # Fetch issue comments
        if not os.path.exists(comments_file):
            comments = fetch_issue_comments(owner, name)
            save_snapshot_json(comments, comments_file, {
                "source": "github_rest",
                "endpoint": "/repos/{owner}/{repo}/issues/comments",
                "owner": owner,
                "repo": name,
            })
            total_comments += len(comments)
            print(f"  Comments: {len(comments)}")
        
        # Fetch PR reviews
        if not os.path.exists(reviews_file):
            reviews = fetch_pr_reviews(owner, name)
            save_snapshot_json(reviews, reviews_file, {
                "source": "github_graphql",
                "endpoint": "repository.pullRequests.reviews",
                "owner": owner,
                "repo": name,
            })
            total_reviews += len(reviews)
            print(f"  Reviews: {len(reviews)}")
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total comments fetched: {total_comments}")
    print(f"Total reviews fetched: {total_reviews}")

if __name__ == "__main__":
    main()
