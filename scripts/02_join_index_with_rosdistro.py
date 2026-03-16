import os
import re
import json
import csv
from typing import Dict, Any, Optional, Tuple

DISTROS = ["humble", "jazzy", "kilted"]
INDEX_DIR = "cache/ros_index"
ROSDISTRO_DIR = "cache/rosdistro"
OUT_DIR = "out"
os.makedirs(OUT_DIR, exist_ok=True)

GITHUB_RE = re.compile(r"https?://github\.com/([^/]+)/([^/#?]+)")

def load_index_packages(distro: str) -> list[str]:
    path = os.path.join(INDEX_DIR, f"data.{distro}.json")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        pkgs = [row.get("package") for row in data if isinstance(row, dict)]
    else:
        pkgs = [row.get("package") for row in data.get("packages", [])]

    pkgs = [p for p in pkgs if p]
    return sorted(set(pkgs))

def load_repo_table(distro: str) -> Dict[str, Dict[str, Any]]:
    path = os.path.join(ROSDISTRO_DIR, f"repo_table.{distro}.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_pkg_to_repo_key_from_release(repo_table: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
    pkg_to_key: Dict[str, str] = {}
    for repo_key, entry in repo_table.items():
        for pkg in entry.get("packages_released", []) or []:
            pkg_to_key.setdefault(pkg, repo_key)
    return pkg_to_key

def choose_best_url(entry: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    for url_type in ("source", "doc", "release"):
        url = entry.get(f"url_{url_type}")
        if url:
            return url, url_type
    return None, None

def parse_github_owner_repo(url: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    if not url:
        return None, None
    m = GITHUB_RE.search(url)
    if not m:
        return None, None
    owner = m.group(1)
    repo = m.group(2)
    if repo.endswith(".git"):
        repo = repo[:-4]
    return owner, repo

def main():
    rows = []

    for d in DISTROS:
        index_pkgs = load_index_packages(d)
        repo_table = load_repo_table(d)
        pkg_to_key_release = build_pkg_to_repo_key_from_release(repo_table)

        resolved_count = 0

        for pkg in index_pkgs:
            repo_key = pkg_to_key_release.get(pkg)
            resolved_via = "rosdistro_release_packages" if repo_key else None

            # Fallback A (rosdistro-only):
            if not repo_key and pkg in repo_table:
                url_tmp, _ = choose_best_url(repo_table[pkg])
                if url_tmp:
                    repo_key = pkg
                    resolved_via = "rosdistro_repo_key_name_match"

            entry = repo_table.get(repo_key) if repo_key else None

            url, url_type = (None, None)
            if entry:
                url, url_type = choose_best_url(entry)

            owner, repo = parse_github_owner_repo(url)
            resolved = bool(url)
            if resolved:
                resolved_count += 1

            rows.append({
                "ros_distro": d,
                "package": pkg,
                "rosdistro_repo_key": repo_key,
                "repo_url": url,
                "repo_url_type": url_type,
                "github_owner": owner,
                "github_repo": repo,
                "resolved": resolved,
                "resolved_via": resolved_via,
            })

        print(f"[OK] joined {d}: index_pkgs={len(index_pkgs)} resolved={resolved_count}")

    csv_path = os.path.join(OUT_DIR, "mapping_packages_to_github.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[OK] saved {csv_path}")

    jsonl_path = os.path.join(OUT_DIR, "mapping_packages_to_github.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[OK] saved {jsonl_path}")

if __name__ == "__main__":
    main()
