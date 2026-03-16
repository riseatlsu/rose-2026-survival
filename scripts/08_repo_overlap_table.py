import csv
from collections import defaultdict, Counter

IN_PATH = "out/diagnostics/resolved_ok.csv"
OUT_CSV = "out/repos/repo_overlap_summary.csv"

def main():
    rows = list(csv.DictReader(open(IN_PATH, encoding="utf-8")))

    repo_distros = defaultdict(set)
    for r in rows:
        repo = (r.get("github_owner") or "").strip() + "/" + (r.get("github_repo") or "").strip()
        distro = (r.get("ros_distro") or "").strip()
        if repo != "/" and distro:
            repo_distros[repo].add(distro)

    # Unique repos per distro
    unique_per_distro = Counter()
    for repo, ds in repo_distros.items():
        for d in ds:
            unique_per_distro[d] += 1

    # Exclusives and overlap sizes
    exclusive = Counter()
    overlap_size = Counter()  # 1,2,3 distros
    overlap_sets = Counter()

    for repo, ds in repo_distros.items():
        ds_sorted = tuple(sorted(ds))
        overlap_size[len(ds_sorted)] += 1
        overlap_sets["|".join(ds_sorted)] += 1
        if len(ds_sorted) == 1:
            exclusive[ds_sorted[0]] += 1

    all_distros = sorted({d for ds in repo_distros.values() for d in ds})

    print("=== UNIQUE REPOS PER DISTRO (GitHub-only) ===")
    for d in all_distros:
        print(f"{d}: {unique_per_distro[d]}")

    print("\n=== OVERLAP BY #DISTROS ===")
    for k in sorted(overlap_size):
        print(f"{k} distro(s): {overlap_size[k]} repos")

    print("\n=== EXCLUSIVE (ONLY IN ONE DISTRO) ===")
    for d in all_distros:
        print(f"only_{d}: {exclusive[d]}")

    print("\n=== EXACT SETS ===")
    for k, v in overlap_sets.most_common():
        print(f"{k}: {v}")

    out_rows = []
    out_rows.append({"metric": "unique_repos_total", "value": str(len(repo_distros))})
    for d in all_distros:
        out_rows.append({"metric": f"unique_repos_in_{d}", "value": str(unique_per_distro[d])})
        out_rows.append({"metric": f"exclusive_only_{d}", "value": str(exclusive[d])})

    import os
    os.makedirs("out/repos", exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["metric", "value"])
        w.writeheader()
        w.writerows(out_rows)

    print(f"\n[OK] saved {OUT_CSV}")

if __name__ == "__main__":
    main()
