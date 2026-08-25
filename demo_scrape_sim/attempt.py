#!/usr/bin/env python3
"""Watch Caine ATTEMPT the runbook against live services, and grade the attempt.

The other demo showed two separate things: the adapter reciting the procedure,
and a script of ours performing it. Neither is the agent doing the work. This
one starts the origin (:8820) and the cache proxy (:8817), then drives the real
./symb chat and asks it to scrape the page -- and reports what it actually did
to the filesystem and to the origin's request counter.

The attempt is graded on effects, not on what it said:

    fetched via the proxy      the proxy logged a request from it
    spared the origin          origin hit count did not climb per fetch
    used data-testid           the command/script it ran mentions it
    produced clean rows        scraped/clean/ has files that were not there
    quarantined the bad rows   scraped/quarantine/ has the two seeded defects
    bumped the cursor last     state.json moved, after the files appeared

Nothing here is scored on vibes. It either touched the disk or it did not.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).parent.resolve()
PROJECT = HERE.parent
sys.path.insert(0, str(PROJECT))

TARGET = "http://127.0.0.1:8817/listing?page=1"
ROOT = HERE / "scraped"
CLEAN, QUAR, PENDING = ROOT / "clean", ROOT / "quarantine", ROOT / "pending"
STATE = HERE / "state.json"

DEFAULT_ASK = (
    f"Scrape the listing page at {TARGET} using our Scrape A Listing Page "
    f"runbook. Work in {HERE}. Actually run it, do not just describe it."
)


def wait_for(url: str, timeout: float = 15) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        try:
            urllib.request.urlopen(url, timeout=1).read()
            return True
        except OSError:
            time.sleep(0.2)
    return False


def origin_hits() -> int:
    try:
        return json.loads(urllib.request.urlopen(
            "http://127.0.0.1:8820/__hits", timeout=2).read())["count"]
    except OSError:
        return -1


def reset_workspace() -> None:
    for d in (CLEAN, QUAR, PENDING):
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps({"cursor": 0}) + "\n")


def start_services() -> list[subprocess.Popen]:
    venv = PROJECT / "venv" / "bin" / "python"
    procs = []
    for script, probe in (("origin.py", "http://127.0.0.1:8820/__hits"),
                          ("cache_proxy.py", "http://127.0.0.1:8817/__cache")):
        p = subprocess.Popen([str(venv), "-u", str(HERE / script)], cwd=HERE,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True)
        procs.append(p)
        if not wait_for(probe):
            raise SystemExit(f"{script} did not come up")
    print(f"  origin :8820 and cache proxy :8817 are up")
    return procs


def grade(transcript: str, hits_before: int, hits_after: int) -> list[tuple[str, bool, str]]:
    """Effects on disk and on the origin. Nothing is graded on what it said."""
    clean = sorted(p.name for p in CLEAN.iterdir()) if CLEAN.exists() else []
    quar = sorted(p.name for p in QUAR.iterdir()) if QUAR.exists() else []
    try:
        cursor = json.loads(STATE.read_text()).get("cursor", 0)
    except (OSError, ValueError):
        cursor = 0
    proxied = "[proxy:8817]" in transcript or "8817" in transcript
    return [
        ("went through the proxy on 8817", proxied,
         "the target URL it was given is the proxy"),
        ("origin was actually reached", hits_after > hits_before,
         f"origin hits {hits_before} -> {hits_after}"),
        ("selected by data-testid", bool(re.search(r"data-?testid", transcript, re.I)),
         "appears in something it ran"),
        ("wrote clean rows", bool(clean), f"scraped/clean/: {clean or 'empty'}"),
        ("quarantined the two bad rows",
         any("1004" in q for q in quar) and any("1005" in q for q in quar),
         f"scraped/quarantine/: {quar or 'empty'}"),
        ("bumped the cursor", cursor > 0, f"state.json cursor = {cursor}"),
    ]


def main() -> int:
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, OSError):
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--ask", default=DEFAULT_ASK)
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--deny", action="store_true",
                    help="answer the sandbox gate 'n' -- shows a refused attempt")
    args = ap.parse_args()

    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "vtf", PROJECT / "verify_transcript_fixes.py")
    vtf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vtf)

    reset_workspace()
    procs = start_services()
    hits_before = origin_hits()
    print(f"  workspace reset; origin hits at start: {hits_before}")
    print("\n" + "=" * 70)
    print("  THE ATTEMPT  (real ./symb chat, real tools, live services)")
    print("=" * 70)

    cfg = json.loads((PROJECT / "config.json").read_text())
    session = vtf.LiveSession(cfg.get("user_name", "Huy"),
                              cfg.get("assistant_name", "Caine"))
    pending = [args.ask, "/quit"]
    gates = 0
    hits_after = None
    try:
        while True:
            kind = session.wait_for_prompt(args.timeout)
            if kind is None:
                break
            if kind == "yn":
                with session._lock:
                    fresh = session.buf[session.watermark:]
                # Both kinds of approval gate count. Matching only "[Sandbox]"
                # meant a run blocked by the risk scorer — "[Security: risk
                # score 3/3] ... blocked_binary:curl" — was answered "n" and
                # then reported as "approval gates hit: 0", so the scoreboard
                # said the model achieved nothing without ever saying it had
                # been refused. A harness that hides why a run failed is worse
                # than one that fails.
                if "[Sandbox]" in fresh or "[Security:" in fresh:
                    gates += 1
                    if not session.send("n" if args.deny else "y"):
                        break
                elif not session.send("n"):
                    break
                continue
            if not session.send(pending.pop(0) if pending else "/quit"):
                break
        session.proc.wait(timeout=30)
        hits_after = origin_hits()
    finally:
        session.close()
        # The origin's counter has to be read while the origin is still alive.
        # Reading it after terminate() returned -1 and scored a fetch that
        # demonstrably happened as "origin was never reached".
        if hits_after is None:
            hits_after = origin_hits()
        for p in procs:
            p.terminate()
    print("\n" + "=" * 70)
    print("  WHAT THE ATTEMPT ACTUALLY DID  (effects, not claims)")
    print("=" * 70)
    rows = grade(session.buf, hits_before, hits_after)
    for name, ok, detail in rows:
        print(f"  {'yes' if ok else 'NO ':4} {name:32} {detail}")
    passed = sum(1 for _, ok, _ in rows if ok)
    print(f"\n  {passed}/{len(rows)} effects achieved.  "
          f"sandbox approval gates hit: {gates}")
    if args.deny:
        print("  (--deny: every gate was refused, so 0 effects is the correct result)")
    return 0 if passed == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
