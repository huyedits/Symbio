"""Reading the activity log back.

The log had been written to since day one and never once read — log_path() was
defined and called from nowhere. These pin the parser against the real line
format, including the awkward parts: a `result=` field that runs to end of
line and contains spaces, equals signs and brackets.
"""
from datetime import datetime, timedelta

from symbio.app import local_telemetry as lt

REAL = """\
[2026-08-18 08:40:34] turn user=hello there
[2026-08-18 08:40:35] model model=Qwen/Qwen3-8B-MLX-4bit adapter=True
[2026-08-18 08:41:00] tool name=run_command ok=True result=Shell command exited ok. Output: 15327 [Security alert: MEDIUM-risk action (score 2/3): shell.]
[2026-08-18 08:42:00] tool name=browser_click ok=False result=Could not find 'Sign in' on the page.
[2026-08-18 08:43:00] tool name=browser_click ok=True result=Clicked.
[2026-08-18 08:44:00] tool name=run_command ok=False result=Blocked: the user declined this action earlier in this turn.
not a telemetry line at all
[2026-08-18 08:45:00] train iters=65 ok=True
"""


def write_log(tmp_path, text=REAL):
    p = tmp_path / "activity.txt"
    p.write_text(text, encoding="utf-8")
    return str(p)


def test_it_parses_the_real_line_format(tmp_path):
    events = lt.read_events(path=write_log(tmp_path))
    # The unparseable line is skipped, not fatal.
    assert len(events) == 7
    assert events[0]["kind"] == "turn"
    assert events[0]["user"] == "hello there"


def test_a_result_field_keeps_its_spaces_and_punctuation(tmp_path):
    """result= runs to end of line and contains spaces, colons, brackets and
    even 'score 2/3' — a naive split on '=' or ' ' loses it."""
    events = lt.read_events(path=write_log(tmp_path))
    tool = next(e for e in events if e["kind"] == "tool")
    assert tool["name"] == "run_command"
    assert tool["ok"] == "True"
    assert "Security alert" in tool["result"]
    assert "score 2/3" in tool["result"]


def test_per_tool_success_rates(tmp_path):
    r = lt.summarise(path=write_log(tmp_path))
    assert r["tools"]["browser_click"] == {"calls": 2, "ok": 1}
    assert r["tools"]["run_command"] == {"calls": 2, "ok": 1}
    assert r["turns"] == 1


def test_tools_are_ordered_by_how_much_they_are_used(tmp_path):
    r = lt.summarise(path=write_log(tmp_path))
    calls = [t["calls"] for t in r["tools"].values()]
    assert calls == sorted(calls, reverse=True)


def test_security_alerts_are_surfaced(tmp_path):
    """The reason this reader exists: an alert sat in the file unread."""
    r = lt.summarise(path=write_log(tmp_path))
    assert len(r["alerts"]) == 1
    assert "MEDIUM-risk" in r["alerts"][0]["result"]


def test_blocked_actions_are_counted(tmp_path):
    r = lt.summarise(path=write_log(tmp_path))
    assert len(r["blocked"]) == 1


def test_days_filters_by_timestamp(tmp_path):
    """The fixture above is dated the day this was written, so it cannot be
    used as the 'old' side here — it would still fall inside a 1-day window
    on that day. Use an unambiguously old line instead."""
    recent = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    old = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d %H:%M:%S")
    text = (f"[{old}] tool name=run_command ok=True result=old\n"
            f"[{recent}] tool name=web_search ok=True result=fine\n")
    r = lt.summarise(days=1, path=write_log(tmp_path, text))
    assert list(r["tools"]) == ["web_search"]


def test_a_missing_log_is_not_an_error(tmp_path):
    r = lt.summarise(path=str(tmp_path / "nope.txt"))
    assert r["events"] == 0
    assert "No activity recorded yet" in lt.format_summary(r)


def test_the_summary_names_the_file_it_actually_read(tmp_path):
    """Reporting the default path while summarising a different file names a
    file the numbers did not come from."""
    path = write_log(tmp_path)
    r = lt.summarise(path=path)
    assert r["path"] == path
    assert path in lt.format_summary(r)


def test_no_tool_calls_says_so_plainly(tmp_path):
    path = write_log(tmp_path, "[2026-08-18 08:40:34] turn user=hi\n")
    out = lt.format_summary(lt.summarise(path=path))
    assert "not reaching" in out


def test_the_rendered_summary_shows_rates(tmp_path):
    out = lt.format_summary(lt.summarise(path=write_log(tmp_path)))
    assert "browser_click" in out
    assert "50% ok" in out
