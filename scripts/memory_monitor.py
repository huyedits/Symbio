#!/usr/bin/env python3
"""Simple memory monitor for macOS."""
import subprocess
import sys
import time


INTERVAL = 5  # seconds
WARNING_GB = 2.0
CRITICAL_GB = 1.0


def get_available_gb():
    """Return approximate available memory in GB (free + inactive + speculative)."""
    try:
        out = subprocess.check_output(["vm_stat"], text=True)
        stats = {}
        for line in out.splitlines():
            if ":" in line:
                key, val = line.split(":", 1)
                val = val.strip().replace(".", "")
                try:
                    stats[key.strip()] = int(val)
                except ValueError:
                    pass
        page_size = 16384
        free = stats.get("Pages free", 0)
        inactive = stats.get("Pages inactive", 0)
        speculative = stats.get("Pages speculative", 0)
        total_gb = (free + inactive + speculative) * page_size / (1024 ** 3)
        return total_gb
    except Exception as e:
        print(f"[monitor] error reading memory: {e}", file=sys.stderr)
        return None


def main():
    print("[monitor] Starting memory monitor...")
    sys.stdout.flush()
    while True:
        avail = get_available_gb()
        if avail is not None:
            status = "ok"
            if avail < CRITICAL_GB:
                status = "CRITICAL"
            elif avail < WARNING_GB:
                status = "WARNING"
            print(f"[monitor] available memory: {avail:.2f} GB ({status})")
            sys.stdout.flush()
        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()
