import os
import csv
from collections import defaultdict

IN_OK = "out/diagnostics/resolved_ok.csv"

def main():
    with open(IN_OK, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    rows = [r for r in rows if (r.get("github_owner") and r.get("github_repo"))]

    by_distro = defaultdict(list)
    for r in rows:
        by_distro[r["ros_distro"]].append(r)

    print(f"[INFO] Resolved packages by distro:")
    for d in sorted(by_distro.keys()):
        print(f"  {d}: {len(by_distro[d])} packages")
    
    print(f"[INFO] Total resolved packages (GitHub): {len(rows)}")

if __name__ == "__main__":
    main()
