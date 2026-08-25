#!/usr/bin/env python3
"""The scraping runbook: recalled from weights, then actually executed.

Starts a listing origin on :8820 and the cache proxy on :8817, asks the trained
adapter to state the procedure with the steps STRIPPED from its prompt, then
performs that procedure for real and checks the two against each other.

The point of the pairing: a model reciting a plausible-sounding runbook proves
nothing. A model reciting one whose every clause is then executed and verified
against live services -- a cache that demonstrably spares the origin, a parser
that finds the container by data-testid, a validator that quarantines the two
bad rows, a cursor that moves last -- is a different claim.

    ./run.py              # recall + execute
    ./run.py --exec-only  # just the runbook, no model
"""
import argparse
import json
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).parent.resolve()
PROJECT = HERE.parent
VENV = PROJECT / "venv" / "bin" / "python"


def wait_for(url: str, timeout: float = 15) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        try:
            urllib.request.urlopen(url, timeout=1).read()
            return True
        except (urllib.error.URLError, OSError):
            time.sleep(0.2)
    return False


def start_services() -> list[subprocess.Popen]:
    procs = []
    for script, probe in (("origin.py", "http://127.0.0.1:8820/__hits"),
                          ("cache_proxy.py", "http://127.0.0.1:8817/__cache")):
        p = subprocess.Popen([str(VENV), "-u", str(HERE / script)], cwd=HERE,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True)
        procs.append(p)
        if not wait_for(probe):
            raise SystemExit(f"{script} did not come up")
        print(f"  started {script}")
    return procs


def recall_from_weights() -> str:
    """Ask the adapter for the procedure, steps stripped from its prompt."""
    from mlx_lm import load, generate
    from symbio import constants
    from symbio.app import dispatch, eval as eval_mod, skills, tooling

    cfg = json.loads((PROJECT / "config.json").read_text())
    entry = dispatch.catalog_entry_for_role("scrape_a_listing_page")
    if entry is None:
        raise SystemExit("No 'scrape_a_listing_page' skill. Run "
                         "../demo_finetune_loop.py first.")
    adapter = constants.adapter_dir_for("scrape_a_listing_page")
    system = skills.build_worker_system_prompt("Scrape A Listing Page")
    model, tok = load(entry["model_name"], adapter_path=str(adapter))
    p = tok.apply_chat_template(
        [{"role": "system", "content": system},
         {"role": "user", "content": "Do 'Scrape A Listing Page'."}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    # Greedy: a regression demo has no business being a dice roll. At the
    # configured temp 0.6 this adapter collapses into a repetition loop about
    # one sample in six (measured), which says nothing about what it learned.
    raw = generate(model, tok, prompt=p,
                   sampler=eval_mod._make_sampler(
                       {**cfg, "agent": {**cfg["agent"], "temperature": 0.0}}),
                   max_tokens=400, verbose=False)
    return tooling.strip_tool_tags(tooling.strip_reasoning_block(raw)).strip()


CLAUSES = {
    "port 8817":     r"8817",
    "selectolax":    r"selectolax",
    "data-testid":   r"data-?testid",
    "pending stage": r"pending",
    "quarantine":    r"quarantine",
    "cursor last":   r"cursor",
}


def main() -> int:
    # Piped or recorded, stdout is a pipe and Python block-buffers it: this
    # script's own prints sit in a buffer while the runbook subprocess flushes
    # on exit, so the execution block lands ABOVE the recall block it is
    # supposed to follow. Fine on a tty, wrong everywhere a demo gets captured.
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, OSError):
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--exec-only", action="store_true")
    args = ap.parse_args()

    recalled = ""
    if not args.exec_only:
        print("\n" + "=" * 70)
        print("  RECALLED FROM WEIGHTS  (procedure stripped from the prompt)")
        print("=" * 70)
        recalled = recall_from_weights()
        for line in recalled.splitlines():
            if line.strip():
                print(f"    {line.strip()}")

    print("\n" + "=" * 70)
    print("  NOW ACTUALLY RUN IT  (live origin :8820, cache proxy :8817)")
    print("=" * 70)
    procs = start_services()
    try:
        rc = subprocess.run([str(VENV), str(HERE / "run_runbook.py")],
                            cwd=HERE).returncode
    finally:
        for p in procs:
            p.terminate()
        for p in procs:
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()
        print("  services stopped")

    if recalled:
        print("\n" + "=" * 70)
        print("  DID THE RECALLED PROCEDURE MATCH THE ONE THAT RAN?")
        print("=" * 70)
        missing = [name for name, pat in CLAUSES.items()
                   if not re.search(pat, recalled, re.I)]
        for name, pat in CLAUSES.items():
            hit = re.search(pat, recalled, re.I)
            print(f"    {'yes' if hit else 'NO ':4} {name}")
        print(f"\n    {len(CLAUSES) - len(missing)}/{len(CLAUSES)} clauses the "
              f"adapter recalled are ones the live run actually performed.")
        return rc or (1 if missing else 0)
    return rc


if __name__ == "__main__":
    sys.exit(main())
