import os
import csv
from collections import defaultdict, Counter

IN_PATH = "out/diagnostics/resolved_ok.csv"
OUT_DIR = "out/repos"
os.makedirs(OUT_DIR, exist_ok=True)

OUT_REPOS = os.path.join(OUT_DIR, "github_repos_unique.csv")
OUT_REPOS_BY_DISTRO = os.path.join(OUT_DIR, "github_repos_unique_by_distro.csv")

def main():
    rows = list(csv.DictReader(open(IN_PATH, encoding="utf-8")))

    # aggregate per repo
    agg = {}
    distros_per_repo = defaultdict(set)
    packages_per_repo = defaultdict(set)
    via_counter = defaultdict(Counter)
    distro_pkg_counter = defaultdict(Counter)

    for r in rows:
        repo = (r.get("github_owner") or "").strip() + "/" + (r.get("github_repo") or "").strip()
        if repo == "/":
            continue

        distro = (r.get("ros_distro") or "").strip()
        pkg = (r.get("package") or "").strip()
        via = (r.get("resolved_via") or "none").strip()

        distros_per_repo[repo].add(distro)
        packages_per_repo[repo].add(pkg)
        via_counter[repo][via] += 1
        distro_pkg_counter[repo][distro] += 1

        if repo not in agg:
            agg[repo] = {
                "full_name": repo,
                "repo_url": (r.get("repo_url") or "").strip(),
            }

    out = []
    out_by_distro = []

    for repo, base in agg.items():
        distros = sorted(distros_per_repo[repo])
        n_pkgs = len(packages_per_repo[repo])

        # compact via stats
        via = via_counter[repo]
        via_str = ";".join([f"{k}:{v}" for k, v in via.most_common()])

        row = dict(base)
        row.update({
            "distros": "|".join(distros),
            "n_distros": len(distros),
            "n_packages_total": n_pkgs,
            "resolved_via_breakdown": via_str,
        })
        out.append(row)

        for d in distros:
            out_by_distro.append({
                "full_name": repo,
                "ros_distro": d,
                "n_packages_in_distro": distro_pkg_counter[repo][d],
                "repo_url": base["repo_url"],
            })

    out.sort(key=lambda r: int(r["n_packages_total"]), reverse=True)
    out_by_distro.sort(key=lambda r: (r["ros_distro"], r["full_name"]))

    with open(OUT_REPOS, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)

    with open(OUT_REPOS_BY_DISTRO, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out_by_distro[0].keys()))
        w.writeheader()
        w.writerows(out_by_distro)

    print(f"[OK] saved {OUT_REPOS} repos={len(out)}")
    print(f"[OK] saved {OUT_REPOS_BY_DISTRO} rows={len(out_by_distro)}")

if __name__ == "__main__":
    main()
