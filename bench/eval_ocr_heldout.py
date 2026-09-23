"""Held-out check of symbio.ocr.match on pages it was never tuned on.

Truth comes from the DOM, not from OCR: every visible control's bounding box,
rendered at device scale 2 like a Retina capture. Queries are phrased the way
the model asks ("the Log in link"). Icon-only controls (aria-label, no visible
text) are included on purpose: the right answer there is a decline.

Grades per present query:
  HIT      centre inside the target box
  DUP-PICK label occurs on several visible controls and OCR picked one (a guess)
  WRONG    returned a box outside every control carrying that label
  DECLINE  returned nothing

The first run captures every site into $OCR_HELDOUT_DIR (screenshots plus
pages.json); later runs grade the frozen captures, so two matchers can be
compared on identical pages. MATCHER=path/to/other_ocr.py grades that file's
read_text/match instead of symbio.ocr.

Usage: venv/bin/python bench/eval_ocr_heldout.py
"""
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from playwright.sync_api import sync_playwright  # noqa: E402
from symbio import ocr  # noqa: E402

OUT = os.environ.get("OCR_HELDOUT_DIR", "/tmp/ocr_heldout")
SITES = [
    "https://news.ycombinator.com",
    "https://en.wikipedia.org/wiki/Main_Page",
    "https://docs.python.org/3/",
    "https://pypi.org/",
    "https://github.com/huyedits/Symbio",
    "https://developer.mozilla.org/en-US/",
    "https://www.apple.com/",
    "https://duckduckgo.com/",
    "https://www.bbc.com/news",
    "https://www.npmjs.com/",
    "https://www.rust-lang.org/",
    "https://arxiv.org/",
    "https://www.gov.uk/",
    "https://www.craigslist.org/about/sites",
]
DSF = 2
PAD = 4 * DSF

COLLECT = r"""
() => {
  const out = [];
  const sel = 'a, button, input, textarea, select, summary, [role=button], [role=tab], [role=link], [role=menuitem]';
  for (const el of document.querySelectorAll(sel)) {
    const r = el.getBoundingClientRect();
    if (r.width < 4 || r.height < 4) continue;
    if (r.bottom <= 0 || r.right <= 0 || r.top >= innerHeight || r.left >= innerWidth) continue;
    const st = getComputedStyle(el);
    if (st.visibility === 'hidden' || st.display === 'none' || +st.opacity === 0) continue;
    // topmost at its centre, so a covered control is not counted as visible
    const cx = Math.min(innerWidth - 1, Math.max(0, r.left + r.width / 2));
    const cy = Math.min(innerHeight - 1, Math.max(0, r.top + r.height / 2));
    const top = document.elementFromPoint(cx, cy);
    if (!top || !(el === top || el.contains(top) || top.contains(el))) continue;
    const tag = el.tagName.toLowerCase();
    let text = (el.innerText || '').trim().replace(/\s+/g, ' ');
    let icon = false;
    if (!text && (tag === 'input' || tag === 'textarea')) text = (el.value || el.placeholder || '').trim();
    if (!text) { text = (el.getAttribute('aria-label') || el.title || '').trim(); icon = !!text; }
    if (!text) continue;
    const kind = (tag === 'a' || el.getAttribute('role') === 'link') ? 'link'
               : (tag === 'input' && !['submit','button'].includes(el.type)) || tag === 'textarea' ? 'field'
               : el.getAttribute('role') === 'tab' ? 'tab' : 'button';
    out.push({text, kind, icon,
              box: [r.left, r.top, Math.min(r.right, innerWidth), Math.min(r.bottom, innerHeight)]});
  }
  return out;
}
"""


def norm(t):
    return " ".join(t.lower().split())


def capture():
    os.makedirs(OUT, exist_ok=True)
    pages = []
    with sync_playwright() as p:
        browser = p.chromium.launch()
        ctx = browser.new_context(viewport={"width": 1280, "height": 800},
                                  device_scale_factor=DSF, locale="en-US")
        for url in SITES:
            page = ctx.new_page()
            try:
                page.goto(url, timeout=30000, wait_until="load")
                page.wait_for_timeout(1500)
            except Exception as e:
                print("skip", url, e)
                continue
            els = page.evaluate(COLLECT)
            body = norm(page.evaluate("document.body.innerText"))
            name = url.split("//")[1].split("/")[0]
            shot = f"{OUT}/{name}.png"
            page.screenshot(path=shot)
            pages.append({"url": url, "name": name, "shot": shot, "els": els, "body": body})
            page.close()
        browser.close()
    json.dump(pages, open(f"{OUT}/pages.json", "w"))
    return pages


def main():
    import importlib.util
    global ocr
    if os.environ.get("MATCHER"):
        spec = importlib.util.spec_from_file_location("ocr_alt", os.environ["MATCHER"])
        ocr = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ocr)
    rng = random.Random(7)
    try:
        pages = json.load(open(f"{OUT}/pages.json"))
    except FileNotFoundError:
        pages = capture()

    all_labels = [e["text"] for pg in pages for e in pg["els"] if not e["icon"] and 3 <= len(e["text"]) <= 30]
    tally = {"HIT": 0, "DUP-PICK": 0, "WRONG": 0, "DECLINE": 0}
    icon_tally = {"HIT": 0, "DUP-PICK": 0, "WRONG": 0, "DECLINE": 0}
    dup_declined = absent_ok = absent_n = 0
    for pg in pages:
        runs = ocr.read_text(pg["shot"])
        texts = [e for e in pg["els"] if len(e["text"]) <= 40]
        text_els = [e for e in texts if not e["icon"]]
        icon_els = [e for e in texts if e["icon"]]
        chosen = rng.sample(text_els, min(15, len(text_els))) + rng.sample(icon_els, min(5, len(icon_els)))
        print(f"\n{pg['name']}  ({len(pg['els'])} controls, {len(runs)} text runs)")
        for e in chosen:
            q = f"the {e['text']} {e['kind']}"
            same = [o for o in pg["els"] if norm(o["text"]) == norm(e["text"])]
            got = ocr.match(runs, q)

            def inside(box, c):
                return (box[0]*DSF - PAD <= c[0] <= box[2]*DSF + PAD
                        and box[1]*DSF - PAD <= c[1] <= box[3]*DSF + PAD)
            if got is None:
                g = "DECLINE"
                dup_declined += len(same) > 1
            else:
                c = ((got["box"][0] + got["box"][2]) // 2, (got["box"][1] + got["box"][3]) // 2)
                if inside(e["box"], c):
                    g = "HIT" if len(same) == 1 else "DUP-PICK"
                elif any(inside(o["box"], c) for o in same):
                    g = "DUP-PICK"
                else:
                    g = "WRONG"
            (icon_tally if e["icon"] else tally)[g] += 1
            extra = f" -> {got['text'][:30]!r} @{got['box']}" if got else ""
            print(f"  {g:8} {'icon ' if e['icon'] else ''}x{len(same)} {q[:48]}{extra if g != 'HIT' else ''}")
        absent = [lbl for lbl in all_labels if norm(lbl) not in pg["body"]]
        for lbl in rng.sample(absent, min(8, len(absent))):
            q = f"the {lbl} link"
            got = ocr.match(runs, q)
            absent_n += 1
            absent_ok += got is None
            if got:
                print(f"  FALSE    (absent) {q[:48]} -> {got['text'][:30]!r}")
    print(f"\ntext controls: {tally}   (declines on duplicated labels: {dup_declined})")
    print(f"icon-only controls: {icon_tally}")
    print(f"absent labels returned nothing: {absent_ok}/{absent_n}")


if __name__ == "__main__":
    main()
