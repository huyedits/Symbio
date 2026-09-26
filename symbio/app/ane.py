"""The side models on the Apple Neural Engine, reached from Python.

The 14B headmaster runs on the GPU through MLX and takes most of the memory.
Three jobs do not need it, and each one competed with it for the GPU and for
RAM while it ran:

    ocr      the text on a screenshot, with pixel boxes (Vision's text
             recognizer, pinned to the Neural Engine)
    vision   rectangles, salient objects and scene labels for a screenshot
             (Vision's detectors, on the Neural Engine where they support it)
    decide   a routing decision for a message (think first? chat, a tool, a
             search?) from Apple Intelligence's on-device model, as a filled-in
             schema rather than prose

They live in symbio_ane/ane_helper.swift, a small Swift program compiled on
first use with the command-line tools' swiftc and kept running, one JSON line
in and one out. Kept running because the Neural Engine models load once:
measured on the M4, the first OCR of a screenshot took 7.4 s and every one
after it 121 ms.

Every call degrades to {"ok": False, ...}: no macOS, no swiftc, Apple
Intelligence off, or the helper crashed. Callers keep their old path then.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import select
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

from symbio import constants

SOURCE = Path(__file__).resolve().parent.parent.parent / "symbio_ane" / "ane_helper.swift"
# The first OCR loads the recognizer onto the Neural Engine (7.4 s measured);
# a routing decision must answer within a turn's patience or be skipped.
COLD_TIMEOUT_S = 60.0
DECIDE_TIMEOUT_S = 3.0

_lock = threading.Lock()
_proc: subprocess.Popen | None = None
_build_error: str | None = None


def _binary_path() -> Path:
    digest = hashlib.sha256(SOURCE.read_bytes()).hexdigest()[:12]
    return constants.PROJECT_DIR / "cache" / f"symbio-ane-{digest}"


def helper_binary() -> Path | None:
    """The compiled helper, built from source when the source changed."""
    global _build_error
    if sys.platform != "darwin" or not SOURCE.exists():
        _build_error = "the Neural Engine helper needs macOS"
        return None
    target = _binary_path()
    if target.exists():
        return target
    swiftc = shutil.which("swiftc")
    if not swiftc:
        _build_error = "swiftc not found (install the Xcode command-line tools)"
        return None
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".building")
    try:
        done = subprocess.run([swiftc, "-O", "-parse-as-library", str(SOURCE), "-o", str(temporary)],
                              capture_output=True, text=True, timeout=600)
    except (OSError, subprocess.SubprocessError) as e:
        _build_error = f"building the helper failed: {e}"
        return None
    if done.returncode != 0 or not temporary.exists():
        _build_error = "building the helper failed: " + (done.stderr or done.stdout).strip()[-400:]
        return None
    os.replace(temporary, target)
    for old in target.parent.glob("symbio-ane-*"):
        if old != target:
            try:
                old.unlink()
            except OSError:
                pass
    return target


def _start() -> subprocess.Popen | None:
    global _proc
    if _proc is not None and _proc.poll() is None:
        return _proc
    binary = helper_binary()
    if binary is None:
        return None
    _proc = subprocess.Popen([str(binary)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                             stderr=subprocess.DEVNULL, text=True, bufsize=1)
    return _proc


def request(payload: dict[str, Any], timeout: float = COLD_TIMEOUT_S) -> dict[str, Any]:
    """One request to the helper. Never raises."""
    global _proc
    with _lock:
        proc = _start()
        if proc is None:
            return {"ok": False, "error": _build_error or "the Neural Engine helper is unavailable"}
        try:
            proc.stdin.write(json.dumps(payload) + "\n")
            proc.stdin.flush()
            ready, _, _ = select.select([proc.stdout], [], [], timeout)
            if not ready:
                # A wedged helper would hold every later call hostage too.
                proc.kill()
                _proc = None
                return {"ok": False, "error": f"no answer in {timeout:.0f}s"}
            line = proc.stdout.readline()
            if not line:
                _proc = None
                return {"ok": False, "error": "the helper exited"}
            return json.loads(line)
        except (OSError, ValueError) as e:
            try:
                proc.kill()
            except OSError:
                pass
            _proc = None
            return {"ok": False, "error": f"helper failed: {e}"}


def status() -> dict[str, Any]:
    return request({"op": "status"}, timeout=10)


def ocr(path: str | Path, level: str = "accurate") -> dict[str, Any]:
    return request({"op": "ocr", "path": str(path), "level": level})


def read_crop(path: str | Path, box: tuple[int, int, int, int], pad: int = 6,
              min_conf: float = 0.5) -> str:
    """The text inside one region of a screenshot, or "".

    Enlarged 3x when the region is short: the recognizer misses a lone label
    in small type that it reads cleanly bigger. Low-confidence reads count as
    no text, so a caller falls back to asking the vision model.
    """
    left, top, right, bottom = box
    x, y = max(0, left - pad), max(0, top - pad)
    width, height = (right - left) + 2 * pad, (bottom - top) + 2 * pad
    answer = request({"op": "ocr", "path": str(path), "crop": [x, y, width, height],
                      "upscale": 3.0 if height < 48 else 1.0})
    if not answer.get("ok"):
        return ""
    texts = [str(line.get("text") or "").strip() for line in answer.get("lines") or []
             if float(line.get("conf") or 0) >= min_conf]
    return " ".join(t for t in texts if t)


def vision(path: str | Path, tasks: list[str] | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {"op": "vision", "path": str(path)}
    if tasks:
        payload["tasks"] = tasks
    return request(payload)


_decide_state = {"unavailable_until": 0.0}


def decide(message: str) -> dict[str, Any]:
    """A routing decision from the on-device model, or ok=False quickly.

    Unavailable (Apple Intelligence off) is remembered for ten minutes, so a
    machine without it pays one short round trip, not one per turn.
    """
    if time.monotonic() < _decide_state["unavailable_until"]:
        return {"ok": False, "error": "on-device model unavailable"}
    answer = request({"op": "decide", "message": message}, timeout=DECIDE_TIMEOUT_S)
    if not answer.get("ok") and "status" in answer:
        _decide_state["unavailable_until"] = time.monotonic() + 600
    return answer


def close() -> None:
    global _proc
    with _lock:
        if _proc is not None:
            try:
                _proc.stdin.close()
                _proc.wait(timeout=2)
            except (OSError, subprocess.SubprocessError):
                _proc.kill()
            _proc = None


# ── what the rest of Symbio uses ─────────────────────────────────────────

def enabled(config: dict[str, Any] | None) -> bool:
    """On unless ane.enabled is false; only ever on macOS."""
    return sys.platform == "darwin" and bool(((config or {}).get("ane") or {}).get("enabled", True))


def ocr_elements(result: dict[str, Any]) -> list[dict[str, Any]]:
    """OCR lines in the shape vision's elements have, so the click tools and
    their Retina-scale handling take them unchanged."""
    elements = []
    for line in result.get("lines") or []:
        text = str(line.get("text") or "").strip()
        box = line.get("box") or []
        if not text or len(box) < 4:
            continue
        left, top, width, height = (int(v) for v in box[:4])
        elements.append({"label": text, "conf": float(line.get("conf") or 0.0),
                         "box": (left, top, left + width, top + height),
                         "center": (left + width // 2, top + height // 2)})
    return elements


# A look whose question is about WORDS — what something says, the text of a
# field, an error message — is answered by the text itself. Anything about how
# the screen LOOKS (an icon, an image, a colour, a layout) is not.
_READING = re.compile(
    r"\b(read|reads|say|says|said|saying|text|written|wording|words?|message|"
    r"error|title|heading|label|caption|transcribe|quote)\b|what'?s on|what is on|"
    r"in front|on (?:the|my) screen", re.IGNORECASE)
_LOOKS = re.compile(
    r"\b(colou?r|icon|image|picture|photo|logo|shape|layout|chart|graph|diagram)\b|"
    r"look(?:s)? like", re.IGNORECASE)


def is_reading_question(question: str) -> bool:
    q = question or ""
    return not q.strip() or (bool(_READING.search(q)) and not _LOOKS.search(q))


def text_block(elements: list[dict[str, Any]], limit: int = 60) -> str:
    """One line per piece of text, centre coordinates first."""
    lines = [f"  ({e['center'][0]}, {e['center'][1]})  {e['label']}" for e in elements[:limit]]
    if len(elements) > limit:
        lines.append(f"  … {len(elements) - limit} more lines of text")
    return "\n".join(lines)
