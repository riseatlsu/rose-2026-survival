import os
import re
import csv
import json
import time
import html as ihtml
from typing import Optional, Dict, Tuple

import requests

IN_CSV = "out/mapping_packages_to_github.csv"
OUT_CSV = "out/mapping_packages_to_github_with_index_html.csv"

CACHE_DIR = "cache/index_html"
os.makedirs(CACHE_DIR, exist_ok=True)
CACHE_JSON = os.path.join(CACHE_DIR, "package_page_cache.json")

REQUEST_DELAY_SEC = 0.35
TIMEOUT_SEC = 30

GITHUB_RE = re.compile(r"https?://github\.com/([^/]+)/([^/#?\s]+)")

def load_cache() -> Dict[str, str]:
    if os.path.exists(CACHE_JSON):
        with open(CACHE_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_cache(cache: Dict[str, str]) -> None:
    with open(CACHE_JSON, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

def fetch_package_page_html(package: str, session: requests.Session, cache: Dict[str, str]) -> Optional[str]:
    if package in cache:
        return cache[package]

    url = f"https://index.ros.org/p/{package}/"
    try:
        r = session.get(url, timeout=TIMEOUT_SEC)
        if r.status_code != 200:
            print(f"[WARN] HTTP {r.status_code} for {url}")
            return None
        html = r.text
        cache[package] = html
        return html
    except Exception as e:
        print(f"[WARN] exception for {url}: {e}")
        return None
    finally:
        time.sleep(REQUEST_DELAY_SEC)

def extract_checkout_uri_for_distro(html: str, distro: str) -> Optional[str]:
    """
    Strategy:
    A) Try to match the HTML structure: Checkout URI ... <a href="URL"> ... VCS Version ... distro
    B) Fallback: strip tags -> plain text and match "Checkout URI <URL> ... VCS Version <distro>"
    """
    distro_l = distro.lower()

    # A) HTML-structure match (href after Checkout URI)
    # This pattern allows tags between labels and fields.
    pat_href = re.compile(
        r"Checkout URI.*?href=\"(https?://[^\"]+)\".*?VCS Version.*?([A-Za-z0-9_\-]+)",
        re.IGNORECASE | re.DOTALL
    )

    matches = pat_href.findall(html)
    for url, vcs_version in matches:
        if vcs_version.lower() == distro_l:
            return ihtml.unescape(url)

    # B) Plain-text fallback
    text = re.sub(r"<[^>]+>", " ", html)        # remove tags
    text = ihtml.unescape(text)
    text = re.sub(r"\s+", " ", text)            # normalize whitespace

    pat_text = re.compile(
        r"Checkout URI\s+(https?://\S+).*?VCS Version\s+([A-Za-z0-9_\-]+)",
        re.IGNORECASE
    )
    matches2 = pat_text.findall(text)
    for url, vcs_version in matches2:
        if vcs_version.lower() == distro_l:
            return url

    # If only one checkout URI exists, return it
    all_urls = [u for (u, _) in matches] + [u for (u, _) in matches2]
    all_urls = list(dict.fromkeys(all_urls))
    if len(all_urls) == 1:
        return all_urls[0]

    return None

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
    cache = load_cache()

    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (compatible; ros-index-mapper/1.0)",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Connection": "keep-alive",
    })

    with open(IN_CSV, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    missing = [r for r in rows if not (r.get("repo_url") or "").strip()]
    print(f"[INFO] rows={len(rows)} missing_repo_url={len(missing)}")

    filled = 0
    tried = 0

    for r in rows:
        if (r.get("repo_url") or "").strip():
            continue

        pkg = r["package"]
        distro = r["ros_distro"]

        tried += 1
        html = fetch_package_page_html(pkg, session, cache)
        if not html:
            continue

        checkout = extract_checkout_uri_for_distro(html, distro)
        if not checkout:
            if filled == 0 and tried <= 10:
                print(f"[DEBUG] no checkout found for {distro}/{pkg}")
            continue

        r["repo_url"] = checkout
        r["repo_url_type"] = "index_checkout_uri"
        r["resolved_via"] = "index_html_checkout_uri"
        r["resolved"] = "True"

        owner, repo = parse_github_owner_repo(checkout)
        r["github_owner"] = owner or ""
        r["github_repo"] = repo or ""

        filled += 1
        if filled % 50 == 0:
            print(f"[OK] filled={filled} (latest: {distro}/{pkg} -> {checkout})")

    save_cache(cache)

    # ensure columns exist
    fieldnames = list(rows[0].keys())
    if "resolved_via" not in fieldnames:
        fieldnames.append("resolved_via")

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"[DONE] tried_missing={tried} filled_total={filled}")
    print(f"[OK] saved {OUT_CSV}")

if __name__ == "__main__":
    main()
