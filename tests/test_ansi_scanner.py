"""Tests for ANSI red-text detection."""

import pytest

from symbio.ansi_scanner import (
    AnsiScanResult,
    extract_red_segments,
    find_error_keywords,
    scan_text,
    strip_ansi,
)


def test_strip_ansi_removes_colors():
    colored = "\x1b[31mred\x1b[0m normal \x1b[91mbright\x1b[0m"
    assert strip_ansi(colored) == "red normal bright"


def test_extract_red_segments_finds_red():
    text = "ok \x1b[31mthis failed\x1b[0m done"
    assert extract_red_segments(text) == ["this failed"]


def test_extract_red_segments_bright_red():
    text = "\x1b[91mbright error\x1b[0m"
    assert extract_red_segments(text) == ["bright error"]


def test_extract_red_segments_no_red():
    text = "\x1b[32mgreen\x1b[0m text"
    assert extract_red_segments(text) == []


def test_extract_red_segments_unclosed_red():
    text = "start \x1b[31mtrailing red"
    assert extract_red_segments(text) == ["trailing red"]


def test_find_error_keywords():
    text = "Fatal: the command failed with a Traceback"
    kws = find_error_keywords(text)
    assert "fatal" in kws
    assert "failed" in kws
    assert "traceback" in kws


def test_scan_text_result():
    text = "\x1b[31mError: file not found\x1b[0m\nDone."
    result = scan_text(text)
    assert isinstance(result, AnsiScanResult)
    assert result.has_red
    assert result.red_segments == ["Error: file not found"]
    assert "error" in result.error_keywords
    assert result.looks_bad


def test_scan_text_no_errors():
    text = "Everything is fine."
    result = scan_text(text)
    assert not result.has_red
    assert not result.error_keywords
    assert not result.looks_bad
