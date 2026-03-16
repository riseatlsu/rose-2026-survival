import os
import csv
from collections import Counter, defaultdict

IN_PATH = "out/mapping_packages_to_github_with_index_html.csv"
OUT_DIR = "out/diagnostics"
os.makedirs(OUT_DIR, exist_ok=True)

def is_truthy(v: str) -> bool:
    return (v or "").strip().lower() in {"true", "1", "yes"}

def main():
    with open(IN_PATH, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    buckets = defaultdict(list)
    by_distro = defaultdict(list)

    # Categorize and group by distro
    for r in rows:
        repo_key = r["rosdistro_repo_key"] or ""
        url = r["repo_url"] or ""
        owner = r["github_owner"] or ""
        resolved = is_truthy(r["resolved"])

        by_distro[r["ros_distro"]].append(r)

        if not repo_key:
            buckets["missing_repo_key__not_in_rosdistro_release"].append(r)
        elif repo_key and not url:
            buckets["repo_key_but_no_url_in_rosdistro"].append(r)
        elif url and not owner:
            buckets["non_github_url"].append(r)
        elif resolved:
            buckets["resolved_ok"].append(r)
        else:
            buckets["other"].append(r)

    # === DIAG V1: Resolution status per distro ===
    print("=== DIAG V1: Resolution Status (per distro) ===")
    for d, rs in sorted(by_distro.items()):
        c = Counter()
        for r in rs:
            repo_key = r["rosdistro_repo_key"] or ""
            url = r["repo_url"] or ""
            owner = r["github_owner"] or ""
            resolved = is_truthy(r["resolved"])

            if not repo_key:
                c["missing_repo_key__not_in_rosdistro_release"] += 1
            elif repo_key and not url:
                c["repo_key_but_no_url_in_rosdistro"] += 1
            elif url and not owner:
                c["non_github_url"] += 1
            elif resolved:
                c["resolved_ok"] += 1
            else:
                c["other"] += 1

        total = len(rs)
        print(f"\n{d.upper()} total={total}")
        for k, v in c.most_common():
            print(f"  - {k}: {v} ({v/total:.3f})")

    # === DIAG V2: URL coverage + GitHub ===
    print("\n\n=== DIAG V2: URL Coverage + GitHub (per distro) ===")
    for d, rs in sorted(by_distro.items()):
        total = len(rs)

        has_url = sum(1 for r in rs if (r.get("repo_url") or "").strip())
        no_url = total - has_url

        is_github = sum(1 for r in rs if (r.get("github_owner") or "").strip() and (r.get("github_repo") or "").strip())
        non_github = has_url - is_github

        via = Counter((r.get("resolved_via") or "none") for r in rs)

        print(f"\n{d.upper()} total={total}")
        print(f"  has_repo_url: {has_url} ({has_url/total:.3f})")
        print(f"  missing_url:  {no_url} ({no_url/total:.3f})")
        print(f"  github:       {is_github} ({is_github/total:.3f})")
        print(f"  non_github:   {non_github} ({non_github/total:.3f})")
        print(f"  resolved_via: {dict(via)}")

    # Save bucket CSVs
    print("\n\n=== Saving CSV buckets ===")
    for name, rs in buckets.items():
        if not rs:
            continue
        out_path = os.path.join(OUT_DIR, f"{name}.csv")
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rs[0].keys()))
            w.writeheader()
            w.writerows(rs)
        print(f"[OK] saved {out_path} rows={len(rs)}")

if __name__ == "__main__":
    main()
