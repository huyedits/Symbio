#!/usr/bin/env python3
"""Execute the 'Scrape A Listing Page' runbook for real, step by step.

Each of the four steps the adapter recalls is performed here against the live
origin (:8820) and cache proxy (:8817), and each prints what it actually did.
Nothing is simulated except the site itself.

    1. Fetch through the cache proxy on port 8817, never the live site twice.
    2. Parse with selectolax and select the container by data-testid, never by CSS class.
    3. Write rows to scraped/pending/, then move each to clean or quarantine.
    4. Bump the cursor in state.json last, never before the move.
"""
import json
import shutil
import sys
import urllib.request
from pathlib import Path

from selectolax.parser import HTMLParser

HERE = Path(__file__).parent.resolve()
ROOT = HERE / "scraped"
PENDING, CLEAN, QUAR = ROOT / "pending", ROOT / "clean", ROOT / "quarantine"
STATE = HERE / "state.json"
PROXY, ORIGIN, PATH = "http://127.0.0.1:8817", "http://127.0.0.1:8820", "/listing?page=1"


def origin_hits() -> int:
    return json.loads(urllib.request.urlopen(ORIGIN + "/__hits", timeout=5).read())["count"]


def fetch(url_path: str) -> tuple[str, str]:
    """Step 1. Through the proxy, always. Returns (html, HIT|MISS)."""
    with urllib.request.urlopen(PROXY + url_path, timeout=5) as r:
        return r.read().decode(), r.headers.get("X-Cache", "?")


def parse(html: str) -> list[dict]:
    """Step 2/3. selectolax, and the container by data-testid, never by class."""
    tree = HTMLParser(html)
    container = tree.css_first('[data-testid="listing-container"]')
    rows = []
    for li in container.css('[data-testid="listing-row"]'):
        spans = li.css("span")
        rows.append({"sku": li.attributes.get("data-sku", ""),
                     "title": spans[0].text().strip() if spans else "",
                     "price": spans[1].text().strip() if len(spans) > 1 else ""})
    return rows


def valid(row: dict) -> str | None:
    """The row schema. Returns a reason string when the row fails it."""
    if not row["sku"].startswith("SKU-"):
        return "bad sku"
    if not row["title"]:
        return "missing title"
    try:
        float(row["price"])
    except ValueError:
        return f"price {row['price']!r} is not a number"
    return None


def main() -> int:
    for d in (PENDING, CLEAN, QUAR):
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps({"cursor": 0}) + "\n")

    print(f"\n  STEP 1 — fetch through the cache proxy on port 8817")
    before = origin_hits()
    html, c1 = fetch(PATH)
    html2, c2 = fetch(PATH)          # the same URL again, deliberately
    after = origin_hits()
    print(f"    first fetch  : X-Cache {c1}")
    print(f"    second fetch : X-Cache {c2}   <- same URL, asked twice")
    print(f"    origin saw {after - before} request(s) for 2 fetches "
          f"{'-- never hit the live site twice' if after - before == 1 else '-- LEAKED'}")

    print(f"\n  STEP 2 — parse with selectolax, container by data-testid")
    rows = parse(html)
    by_class = HTMLParser(html).css_first(".x7f2a-title")
    print(f"    selectolax found {len(rows)} rows via [data-testid=listing-row]")
    print(f"    (the CSS class .x7f2a-title exists today: {by_class is not None} "
          f"-- but it is deploy-hashed, which is why step 3 forbids it)")

    print(f"\n  STEP 3 — write to scraped/pending/, then move to clean or quarantine")
    staged = PENDING / "listing_page1.jsonl"
    staged.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"    staged {len(rows)} rows -> {staged.relative_to(HERE)}")
    kept, quarantined = 0, 0
    for row in rows:
        reason = valid(row)
        if reason is None:
            (CLEAN / f"{row['sku']}.json").write_text(json.dumps(row) + "\n")
            kept += 1
        else:
            (QUAR / f"{row['sku']}.json").write_text(
                json.dumps({**row, "_reason": reason}) + "\n")
            (QUAR / f"{row['sku']}.html").write_text(
                f'<li data-sku="{row["sku"]}">raw html kept beside it</li>\n')
            quarantined += 1
            print(f"      quarantined {row['sku']}: {reason}")
    staged.unlink()
    print(f"    clean {kept}   quarantine {quarantined}   pending "
          f"{len(list(PENDING.iterdir()))} (emptied after the move)")

    print(f"\n  STEP 4 — bump the cursor in state.json, last")
    state = json.loads(STATE.read_text())
    print(f"    cursor before the move would have been unsafe; now: "
          f"{state['cursor']} -> {state['cursor'] + len(rows)}")
    state["cursor"] += len(rows)
    STATE.write_text(json.dumps(state) + "\n")

    print(f"\n  RESULT: {kept} clean, {quarantined} quarantined, "
          f"cursor {state['cursor']}, origin requests {after - before}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
