#!/usr/bin/env python3
"""
Generate ROS package statistics table by distribution.
"""

import csv
import os
from collections import defaultdict

def main():
    # Load all packages
    all_packages_by_distro = defaultdict(int)
    total_all_packages = 0

    with open("out/mapping_packages_to_github.csv", encoding='utf-8') as f:
        for row in csv.DictReader(f):
            distro = row.get('ros_distro', '').strip()
            all_packages_by_distro[distro] += 1
            total_all_packages += 1

    # Load packages on GitHub
    github_packages_by_distro = defaultdict(int)
    total_github_packages = 0

    with open("out/diagnostics/resolved_ok.csv", encoding='utf-8') as f:
        for row in csv.DictReader(f):
            distro = row.get('ros_distro', '').strip()
            github_packages_by_distro[distro] += 1
            total_github_packages += 1

    # Load unique repos per distribution
    repos_by_distro = defaultdict(int)
    all_unique_repo_names = set()

    if os.path.exists("out/repos/github_repos_unique_by_distro.csv"):
        with open("out/repos/github_repos_unique_by_distro.csv", encoding='utf-8') as f:
            for row in csv.DictReader(f):
                distro    = row.get('ros_distro', '').strip()
                full_name = row.get('full_name', '').strip().lower()
                if distro:
                    repos_by_distro[distro] += 1
                if full_name:
                    all_unique_repo_names.add(full_name)

    total_repos = len(all_unique_repo_names)

    # Load final dataset repos (after exclusions)
    filtered_repo_names = set()
    filtered_repo_distros = {}  # full_name -> set of distros

    if os.path.exists("out/survival_repo_dataset.csv"):
        with open("out/survival_repo_dataset.csv", encoding='utf-8') as f:
            for row in csv.DictReader(f):
                owner = row.get('Owner', '').strip()
                name  = row.get('Name', '').strip()
                if owner and name:
                    filtered_repo_names.add(f"{owner}/{name}".lower())

    total_filtered_repos = len(filtered_repo_names)

    # Count final repos per distribution + exclusive
    filtered_repos_by_distro = defaultdict(int)
    if os.path.exists("out/repos/github_repos_unique_by_distro.csv"):
        repo_distros_map = defaultdict(set)
        with open("out/repos/github_repos_unique_by_distro.csv", encoding='utf-8') as f:
            for row in csv.DictReader(f):
                distro    = row.get('ros_distro', '').strip()
                full_name = row.get('full_name', '').strip().lower()
                if distro and full_name in filtered_repo_names:
                    filtered_repos_by_distro[distro] += 1
                    repo_distros_map[full_name].add(distro)

    repos_exclusive_by_distro = defaultdict(int)
    total_multi_distro = 0
    for fn, distros in repo_distros_map.items():
        if len(distros) == 1:
            repos_exclusive_by_distro[list(distros)[0]] += 1
        else:
            total_multi_distro += 1

    # Unique packages per distro (before exclusion) and overall
    unique_pkgs_by_distro = defaultdict(set)
    all_unique_pkg_names = set()
    with open("out/mapping_packages_to_github.csv", encoding='utf-8') as f:
        for row in csv.DictReader(f):
            distro = row.get('ros_distro', '').strip()
            pkg    = row.get('package', '').strip()
            if distro and pkg:
                unique_pkgs_by_distro[distro].add(pkg)
                all_unique_pkg_names.add(pkg)

    # Unique packages after exclusion + multi-package repo stats
    pkg_repos = defaultdict(set)
    repo_pkgs = defaultdict(set)
    with open("out/mapping_packages_to_github.csv", encoding='utf-8') as f:
        for row in csv.DictReader(f):
            owner     = row.get('github_owner', '').strip()
            repo_name = row.get('github_repo', '').strip()
            pkg       = row.get('package', '').strip()
            fn = f'{owner}/{repo_name}'.lower()
            if fn in filtered_repo_names and pkg:
                pkg_repos[pkg].add(fn)
                repo_pkgs[fn].add(pkg)

    total_unique_pkgs_after = len(pkg_repos)
    repos_with_multi_pkgs   = sum(1 for pkgs in repo_pkgs.values() if len(pkgs) > 1)
    pkgs_in_multi_pkg_repos = sum(
        1 for pkg, repos in pkg_repos.items()
        if any(len(repo_pkgs[r]) > 1 for r in repos)
    )
    pct_repos_multi  = (repos_with_multi_pkgs / len(repo_pkgs) * 100) if repo_pkgs else 0
    pct_pkgs_in_multi = (pkgs_in_multi_pkg_repos / total_unique_pkgs_after * 100) if total_unique_pkgs_after else 0

    # Build table
    all_distros = sorted(set(all_packages_by_distro.keys()))
    rows_data = []

    for distro in all_distros:
        total  = all_packages_by_distro[distro]
        github = github_packages_by_distro[distro]
        pct    = (github / total * 100) if total > 0 else 0

        rows_data.append({
            'distribution':                   distro,
            'total_packages':                 total,
            'unique_packages':                len(unique_pkgs_by_distro[distro]),
            'packages_on_github':             github,
            'not_on_github':                  total - github,
            'github_percentage':              f"{pct:.1f}%",
            'unique_repositories':            repos_by_distro[distro],
            'repos_after_exclusion':          filtered_repos_by_distro[distro],
            'repos_exclusive_to_distro':      repos_exclusive_by_distro.get(distro, ''),
            'repos_multi_distro':             '',
            'unique_packages_after_exclusion':'',
            'repos_with_multiple_packages':   '',
            'pct_repos_with_multiple_packages': '',
            'packages_in_multi_pkg_repos':    '',
            'pct_packages_in_multi_pkg_repos':'',
        })

    # TOTAL row
    total_not_github = total_all_packages - total_github_packages
    total_pct = (total_github_packages / total_all_packages * 100) if total_all_packages > 0 else 0

    rows_data.append({
        'distribution':                   'TOTAL',
        'total_packages':                 total_all_packages,
        'unique_packages':                len(all_unique_pkg_names),
        'packages_on_github':             total_github_packages,
        'not_on_github':                  total_not_github,
        'github_percentage':              f"{total_pct:.1f}%",
        'unique_repositories':            total_repos,
        'repos_after_exclusion':          total_filtered_repos,
        'repos_exclusive_to_distro':      sum(repos_exclusive_by_distro.values()),
        'repos_multi_distro':             total_multi_distro,
        'unique_packages_after_exclusion':total_unique_pkgs_after,
        'repos_with_multiple_packages':   repos_with_multi_pkgs,
        'pct_repos_with_multiple_packages': f"{pct_repos_multi:.1f}%",
        'packages_in_multi_pkg_repos':    pkgs_in_multi_pkg_repos,
        'pct_packages_in_multi_pkg_repos': f"{pct_pkgs_in_multi:.1f}%",
    })

    # Print summary
    print(f"\n{'Distribution':<15} {'Total Pkg':<12} {'On GitHub':<12} {'Not GitHub':<12} {'GitHub %':<10} {'Unique Repos':<14} {'After Exclusion':<16}")
    print("-" * 95)
    for row in rows_data:
        print(f"{row['distribution']:<15} {row['total_packages']:<12} {row['packages_on_github']:<12} {row['not_on_github']:<12} {row['github_percentage']:<10} {row['unique_repositories']:<14} {row['repos_after_exclusion']:<16}")

    # Write CSV
    os.makedirs("out", exist_ok=True)
    fieldnames = [
        'distribution', 'total_packages', 'unique_packages', 'packages_on_github',
        'not_on_github', 'github_percentage', 'unique_repositories', 'repos_after_exclusion',
        'repos_exclusive_to_distro', 'repos_multi_distro',
        'unique_packages_after_exclusion',
        'repos_with_multiple_packages', 'pct_repos_with_multiple_packages',
        'packages_in_multi_pkg_repos', 'pct_packages_in_multi_pkg_repos',
    ]
    with open("out/ros_package_statistics.csv", 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_data)

    print(f"\n✅ Saved to: out/ros_package_statistics.csv")

if __name__ == "__main__":
    main()
