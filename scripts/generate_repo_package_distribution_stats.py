#!/usr/bin/env python3
"""
Generate statistics about how packages and distributions are distributed
across repositories in the final survival dataset.
"""

import csv
import os
from collections import defaultdict

def main():
    # Load final dataset repos
    survival = set()
    with open("out/survival_repo_dataset.csv", encoding='utf-8') as f:
        for row in csv.DictReader(f):
            owner = row.get('Owner', '').strip()
            name  = row.get('Name', '').strip()
            if owner and name:
                survival.add(f'{owner}/{name}'.lower())

    # Map repo -> packages and repo -> distros
    pkgs_per_repo   = defaultdict(set)
    distros_per_repo = defaultdict(set)

    with open("out/mapping_packages_to_github.csv", encoding='utf-8') as f:
        for row in csv.DictReader(f):
            owner  = row.get('github_owner', '').strip()
            repo   = row.get('github_repo', '').strip()
            pkg    = row.get('package', '').strip()
            distro = row.get('ros_distro', '').strip()
            fn = f'{owner}/{repo}'.lower()
            if fn in survival and pkg:
                pkgs_per_repo[fn].add(pkg)
                distros_per_repo[fn].add(distro)

    total = len(pkgs_per_repo)

    # --- Package multiplicity ---
    multi_pkg  = sum(1 for p in pkgs_per_repo.values() if len(p) > 1)
    single_pkg = total - multi_pkg
    all_pkgs   = set(p for pkgs in pkgs_per_repo.values() for p in pkgs)

    # --- Distribution multiplicity ---
    in_all3 = sum(1 for d in distros_per_repo.values() if len(d) == 3)
    in_2    = sum(1 for d in distros_per_repo.values() if len(d) == 2)
    in_1    = sum(1 for d in distros_per_repo.values() if len(d) == 1)
    multi_distro = in_all3 + in_2

    # --- Per-distro membership ---
    per_distro = defaultdict(int)
    excl_distro = defaultdict(int)
    for fn, distros in distros_per_repo.items():
        for d in distros:
            per_distro[d] += 1
        if len(distros) == 1:
            excl_distro[list(distros)[0]] += 1

    # --- Print summary ---
    print("=" * 60)
    print("REPO-PACKAGE DISTRIBUTION STATISTICS")
    print("=" * 60)
    print(f"\nFinal dataset repos:         {len(survival)}")
    print(f"Repos with distro mapping:   {total}")
    print(f"Unique packages (final set): {len(all_pkgs)}")
    print(f"\nPackage multiplicity:")
    print(f"  Repos with 1 package:      {single_pkg} ({100*single_pkg/total:.1f}%)")
    print(f"  Repos with >1 package:     {multi_pkg}  ({100*multi_pkg/total:.1f}%)")
    print(f"\nDistribution multiplicity:")
    print(f"  Repos in 1 distro only:    {in_1}  ({100*in_1/total:.1f}%)")
    print(f"  Repos in 2 distros:        {in_2}  ({100*in_2/total:.1f}%)")
    print(f"  Repos in all 3 distros:    {in_all3} ({100*in_all3/total:.1f}%)")
    print(f"  Repos in multiple distros: {multi_distro} ({100*multi_distro/total:.1f}%)")
    print(f"\nPer-distro membership (repos counted once per distro):")
    for d in ['humble', 'jazzy', 'kilted']:
        print(f"  {d}: {per_distro[d]} total, {excl_distro[d]} exclusive")

    # --- Save summary CSV ---
    os.makedirs("out", exist_ok=True)
    rows = [
        {"metric": "total_repos_in_dataset",           "value": len(survival)},
        {"metric": "repos_with_distro_mapping",        "value": total},
        {"metric": "unique_packages",                  "value": len(all_pkgs)},
        {"metric": "repos_single_package",             "value": single_pkg},
        {"metric": "repos_single_package_pct",         "value": f"{100*single_pkg/total:.1f}%"},
        {"metric": "repos_multi_package",              "value": multi_pkg},
        {"metric": "repos_multi_package_pct",          "value": f"{100*multi_pkg/total:.1f}%"},
        {"metric": "repos_in_1_distro",                "value": in_1},
        {"metric": "repos_in_2_distros",               "value": in_2},
        {"metric": "repos_in_3_distros",               "value": in_all3},
        {"metric": "repos_in_multiple_distros",        "value": multi_distro},
        {"metric": "repos_in_multiple_distros_pct",    "value": f"{100*multi_distro/total:.1f}%"},
        {"metric": "repos_humble_total",               "value": per_distro['humble']},
        {"metric": "repos_humble_exclusive",           "value": excl_distro['humble']},
        {"metric": "repos_jazzy_total",                "value": per_distro['jazzy']},
        {"metric": "repos_jazzy_exclusive",            "value": excl_distro['jazzy']},
        {"metric": "repos_kilted_total",               "value": per_distro['kilted']},
        {"metric": "repos_kilted_exclusive",           "value": excl_distro['kilted']},
    ]

    out_path = "out/repo_package_distribution_stats.csv"
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "value"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n✅ Saved to: {out_path}")

if __name__ == "__main__":
    main()
