import os
import json
import requests

DISTROS = ["humble", "jazzy", "kilted"]
BASE = "https://index.ros.org/search/packages/data.{distro}.json"

OUT_DIR = "cache/ros_index"
os.makedirs(OUT_DIR, exist_ok=True)

def main():
    for d in DISTROS:
        url = BASE.format(distro=d)
        out_path = os.path.join(OUT_DIR, f"data.{d}.json")

        print(f"[GET] {url}")
        r = requests.get(url, timeout=60)
        r.raise_for_status()

        data = r.json()
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        n = len(data) if isinstance(data, list) else len(data.get("packages", []))
        print(f"[OK] saved {out_path} | packages={n}")

if __name__ == "__main__":
    main()
