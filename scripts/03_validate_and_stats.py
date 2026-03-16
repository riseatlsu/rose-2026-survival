import os
import csv
from collections import Counter, defaultdict

IN_PATH = "out/mapping_packages_to_github.csv"

def main():
    with open(IN_PATH, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    by_distro = defaultdict(list)
    for r in rows:
        by_distro[r["ros_distro"]].append(r)

    for d, rs in by_distro.items():
        n = len(rs)
        resolved = sum(1 for r in rs if r["resolved"].lower() == "true")
        unresolved = n - resolved

        url_types = Counter(r["repo_url_type"] for r in rs if r["repo_url_type"])
        hosts = Counter(("github" if r["github_owner"] else "non_github_or_missing") for r in rs if r["repo_url"])

        print(f"\n=== {d.upper()} ===")
        print(f"packages_total: {n}")
        print(f"resolved_url:   {resolved} ({resolved/n:.3f})")
        print(f"unresolved:     {unresolved} ({unresolved/n:.3f})")
        print(f"url_type_breakdown: {dict(url_types)}")
        print(f"url_host_breakdown: {dict(hosts)}")

if __name__ == "__main__":
    main()
