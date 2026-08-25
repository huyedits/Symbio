#!/usr/bin/env python3
"""The 'live site' the runbook must not hit twice: a listing page on :8820.

Counts its own requests, so 'never hit the live site twice for the same URL in
one run' is a claim the demo can prove rather than assert. Rows are seeded with
two deliberate defects so step 5's clean/quarantine split has real work to do.
"""
import json
from http.server import BaseHTTPRequestHandler, HTTPServer

ROWS = [
    ("SKU-1001", "Groundhog peg set",        "18.50"),
    ("SKU-1002", "Reflective guy line, 4m",   "6.00"),
    ("SKU-1003", "Grey pole segment",        "23.75"),
    ("SKU-1004", "",                         "9.99"),      # missing title
    ("SKU-1005", "Ridge line tensioner",     "not-a-price"),  # bad price
    ("SKU-1006", "Porch groundsheet",        "41.00"),
]
HITS = {"count": 0}

def page() -> bytes:
    items = "\n".join(
        f'    <li data-testid="listing-row" data-sku="{s}">'
        f'<span class="x7f2a-title">{t}</span>'
        f'<span class="x7f2a-price">{p}</span></li>'
        for s, t, p in ROWS)
    # The class names are hashed-looking on purpose: step 3 says select by
    # data-testid, never by CSS class, because classes change per deploy.
    return (f'<html><body><ul data-testid="listing-container">\n{items}\n'
            f'</ul></body></html>').encode()

class H(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/__hits":
            body = json.dumps(HITS).encode()
        else:
            HITS["count"] += 1
            body = page()
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    def log_message(self, *a):
        pass

if __name__ == "__main__":
    print("origin listening on :8820", flush=True)
    HTTPServer(("127.0.0.1", 8820), H).serve_forever()
