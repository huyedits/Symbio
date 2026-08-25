#!/usr/bin/env python3
"""The cache proxy on :8817 that step 1 requires.

Caches by URL for the life of the process. The second fetch of the same URL is
served from memory and the origin never sees it — which is the whole point of
step 1, and is checkable against the origin's own hit counter.
"""
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer

CACHE: dict[str, bytes] = {}
ORIGIN = "http://127.0.0.1:8820"

class H(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/__cache":
            body, state = str(sorted(CACHE)).encode(), "META"
        elif self.path in CACHE:
            body, state = CACHE[self.path], "HIT"
        else:
            body = urllib.request.urlopen(ORIGIN + self.path, timeout=5).read()
            CACHE[self.path] = body
            state = "MISS"
        print(f"  [proxy:8817] {state:4} {self.path}", flush=True)
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.send_header("X-Cache", state)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    def log_message(self, *a):
        pass

if __name__ == "__main__":
    print("cache proxy listening on :8817 -> :8820", flush=True)
    HTTPServer(("127.0.0.1", 8817), H).serve_forever()
