#!/usr/bin/env python3
"""Caine / Symbio legacy entry point.

This file is kept for backward compatibility. It delegates to the modern CLI
in `symbio.app.cli`, so `python main.py --telegram` and `python main.py --train`
continue to work exactly as before, while `symbio` / `symb` expose subcommands.
"""

import sys

from symbio.app.cli import main

if __name__ == "__main__":
    sys.exit(main())
