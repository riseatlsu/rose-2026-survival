import os
import json
from typing import Dict, Any, Optional

import rosdistro

DISTROS = ["humble", "jazzy", "kilted"]
OUT_DIR = "cache/rosdistro"
os.makedirs(OUT_DIR, exist_ok=True)

def safe_url(repo_obj: Any) -> Optional[str]:
    if repo_obj is None:
        return None
    return getattr(repo_obj, "url", None)

def load_distribution_file(distro_name: str):
    index_url = rosdistro.get_index_url()
    index = rosdistro.get_index(index_url)

    if distro_name not in index.distributions:
        raise ValueError(f"Distro '{distro_name}' not found in rosdistro index: {index_url}")

    dist_file = rosdistro.get_distribution_file(index, distro_name)
    return dist_file

def main():
    for d in DISTROS:
        print(f"[rosdistro] loading distribution.yaml for {d} ...")
        dist_file = load_distribution_file(d)

        #repo_table: URLs per repo_key (distribution.repositories)
        repo_table: Dict[str, Dict[str, Any]] = {}
        for repo_key, repo_entry in dist_file.repositories.items():
            release_repo = getattr(repo_entry, "release_repository", None)
            source_repo  = getattr(repo_entry, "source_repository", None)
            doc_repo     = getattr(repo_entry, "doc_repository", None)

            repo_table[repo_key] = {
                "repo_key": repo_key,
                "url_source": safe_url(source_repo),
                "url_doc": safe_url(doc_repo),
                "url_release": safe_url(release_repo),
                "packages_released": [],
            }

        # (pkg -> object with .repository_name)
        pkg_count = 0
        for pkg_name, pkg_obj in dist_file.release_packages.items():
            repo_key = getattr(pkg_obj, "repository_name", None)
            if repo_key and repo_key in repo_table:
                repo_table[repo_key]["packages_released"].append(pkg_name)
                pkg_count += 1

        repos_with_release = 0
        for repo_key in repo_table:
            pkgs = repo_table[repo_key]["packages_released"]
            if pkgs:
                repos_with_release += 1
                repo_table[repo_key]["packages_released"] = sorted(pkgs)

        out_path = os.path.join(OUT_DIR, f"repo_table.{d}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(repo_table, f, indent=2, ensure_ascii=False)

        print(f"[OK] {d}: repos={len(repo_table)} repos_with_release={repos_with_release} released_pkgs={pkg_count}")
        print(f"[OK] saved {out_path}")

if __name__ == "__main__":
    main()
