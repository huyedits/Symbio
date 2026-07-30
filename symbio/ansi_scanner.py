"""Detect and read ANSI-colored terminal text, especially red error output.

This module captures terminal output with ANSI escape codes preserved, then
parses colour regions. The caller gets both the original text and a list of
red segments, so the machine can say "the terminal showed this in red".
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


# Foreground red codes: 31 (normal red), 91 (bright red).
_RED_FG = frozenset({"31", "91"})
# ANSI SGR colour open codes we care about.
_ANSI_COLOR_RE = re.compile(r"\x1b\[(\d+)m")
# Strip all ANSI escape sequences.
_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*m")


@dataclass
class AnsiScanResult:
    """Result of scanning a terminal output string."""

    text: str
    stripped: str
    red_segments: list[str]
    has_red: bool
    error_keywords: list[str]
    looks_bad: bool


def strip_ansi(text: str) -> str:
    """Remove all ANSI escape sequences from text."""
    return _ANSI_ESCAPE_RE.sub("", text)


def extract_red_segments(text: str) -> list[str]:
    """Return plain-text fragments that were rendered in red.

    Handles stacked/redundant resets. Only SGR colour codes are tracked; we
    ignore bold, italic, cursor movement, etc.
    """
    segments: list[str] = []
    current: list[str] = []
    in_red = False

    parts = _ANSI_COLOR_RE.split(text)
    # parts alternates: text, code, text, code, ...
    for i, part in enumerate(parts):
        if i % 2 == 0:
            # Text fragment.
            if in_red:
                current.append(part)
        else:
            # ANSI code number.
            code = part
            if code in _RED_FG:
                in_red = True
            elif code == "0":
                if in_red and current:
                    joined = "".join(current)
                    if joined.strip():
                        segments.append(strip_ansi(joined))
                current = []
                in_red = False
            # Other colour codes are ignored: we only care about red vs reset.

    # Trailing red text without an explicit reset.
    if in_red and current:
        joined = "".join(current)
        if joined.strip():
            segments.append(strip_ansi(joined))

    return [s.strip() for s in segments if s.strip()]


_ERROR_KEYWORDS = (
    "error",
    "fatal",
    "failed",
    "traceback",
    "exception",
    "cannot",
    "could not",
    "permission denied",
    "command not found",
    "no such file",
)


def find_error_keywords(text: str) -> list[str]:
    """Return error-related keywords found in text (case-insensitive)."""
    low = text.lower()
    return [kw for kw in _ERROR_KEYWORDS if kw in low]


def scan_text(text: str) -> AnsiScanResult:
    """Analyse a terminal output string."""
    stripped = strip_ansi(text)
    red_segments = extract_red_segments(text)
    keywords = find_error_keywords(stripped)
    return AnsiScanResult(
        text=text,
        stripped=stripped,
        red_segments=red_segments,
        has_red=bool(red_segments),
        error_keywords=keywords,
        looks_bad=bool(red_segments or keywords),
    )


def run_and_scan(
    args: list[str],
    *,
    cwd: str | Path | None = None,
    timeout: float | None = None,
    env: dict[str, str] | None = None,
) -> tuple[AnsiScanResult, int]:
    """Run a subprocess and scan its combined stdout/stderr for red text."""
    result = subprocess.run(
        args,
        capture_output=True,
        text=False,
        timeout=timeout,
        cwd=str(cwd) if cwd else None,
        env=env,
    )
    raw = result.stdout + b"\n" + result.stderr
    # Decode preserving bytes that produced ANSI codes; replace invalid chars.
    text = raw.decode("utf-8", errors="replace")
    return scan_text(text), result.returncode


def format_scan_report(result: AnsiScanResult, returncode: int | None = None) -> str:
    """Return a human/machine-readable summary of a scan."""
    lines: list[str] = []
    if returncode is not None and returncode != 0:
        lines.append(f"Process exited with code {returncode}.")
    if result.has_red:
        lines.append("Red terminal text detected:")
        for seg in result.red_segments:
            lines.append(f"  - {seg}")
    if result.error_keywords:
        lines.append("Error keywords found: " + ", ".join(result.error_keywords))
    if not lines:
        lines.append("No red text or error keywords detected.")
    return "\n".join(lines)


def scan_lines(lines: Iterable[str]) -> AnsiScanResult:
    """Scan multiple lines (e.g. from a log file)."""
    return scan_text("\n".join(lines))
