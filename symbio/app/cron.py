"""Scheduled jobs: 5-field cron expressions and one-shot reminders."""

from datetime import datetime, timedelta
from typing import Any

import json

from symbio import constants
from symbio.app import sandbox


def load_cron_jobs() -> list[dict[str, Any]]:
    if not constants.CRON_FILE.exists():
        return []
    try:
        return json.loads(constants.CRON_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []


def save_cron_jobs(jobs: list[dict[str, Any]]):
    constants.CRON_FILE.write_text(json.dumps(jobs, indent=2), encoding="utf-8")


def _cron_field_matches(field: str, value: int, lo: int, hi: int) -> bool:
    for part in field.split(","):
        part = part.strip()
        step = 1
        if "/" in part:
            part, step_str = part.split("/", 1)
            step = int(step_str)
        if part == "*":
            start, end = lo, hi
        elif "-" in part:
            a, b = part.split("-", 1)
            start, end = int(a), int(b)
        else:
            start = end = int(part)
        if start <= value <= end and (value - start) % step == 0:
            return True
    return False


def cron_matches(expr: str, when: datetime) -> bool:
    """Match a 5-field cron expression (minute hour day month weekday,
    weekday 0/7 = Sunday) against a datetime. Raises ValueError on bad fields."""
    fields = expr.split()
    if len(fields) != 5:
        return False
    minute, hour, dom, month, dow = fields
    dow_val = (when.weekday() + 1) % 7
    return (
        _cron_field_matches(minute, when.minute, 0, 59)
        and _cron_field_matches(hour, when.hour, 0, 23)
        and _cron_field_matches(dom, when.day, 1, 31)
        and _cron_field_matches(month, when.month, 1, 12)
        and (
            _cron_field_matches(dow, dow_val, 0, 7)
            or (dow_val == 0 and _cron_field_matches(dow, 7, 0, 7))
        )
    )


def validate_cron_expr(expr: str) -> str | None:
    """Return an error message if expr is not a valid cron expression."""
    if len(expr.split()) != 5:
        return "Schedule must be 'at YYYY-MM-DD HH:MM' or 5 cron fields: minute hour day month weekday."
    try:
        cron_matches(expr, datetime.now())
    except ValueError as e:
        return f"Bad cron expression '{expr}': {e}"
    return None


def parse_one_shot(schedule: str) -> datetime | None:
    """Parse a one-time schedule ('at 2026-07-16 21:30', '21:30', ...).
    A bare time means the next occurrence of that time."""
    s = schedule.strip()
    while s.lower().startswith("at "):
        s = s[3:].strip()
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            pass
    try:
        t = datetime.strptime(s, "%H:%M")
    except ValueError:
        return None
    now = datetime.now()
    target = now.replace(hour=t.hour, minute=t.minute, second=0, microsecond=0)
    if target <= now:
        target += timedelta(days=1)
    return target


def add_cron_job(schedule: str, text: str) -> dict[str, Any]:
    schedule = schedule.strip()
    text = text.strip()
    if not text:
        raise ValueError("Job text is empty.")
    one_shot = parse_one_shot(schedule)
    if one_shot:
        # Normalize to an absolute time so it fires exactly once.
        schedule = f"at {one_shot:%Y-%m-%d %H:%M}"
    else:
        error = validate_cron_expr(schedule)
        if error:
            raise ValueError(error)
    jobs = load_cron_jobs()
    job = {
        "id": max((j.get("id", 0) for j in jobs), default=0) + 1,
        "schedule": schedule,
        "text": text,
        "last_fired": None,
    }
    jobs.append(job)
    save_cron_jobs(jobs)
    return job


def check_due_jobs(config: dict[str, Any], now: datetime | None = None) -> list[str]:
    """Fire all due jobs and return their event messages. One-shot jobs are
    removed after firing; recurring jobs fire at most once per minute."""
    now = now or datetime.now()
    minute_key = now.strftime("%Y-%m-%d %H:%M")
    jobs = load_cron_jobs()
    events: list[str] = []
    remaining: list[dict[str, Any]] = []
    changed = False

    for job in jobs:
        schedule = job.get("schedule", "")
        fire = drop = False
        if schedule.startswith("at "):
            try:
                target = datetime.strptime(schedule[3:], "%Y-%m-%d %H:%M")
                fire = drop = target <= now
            except ValueError:
                events.append(f"Removed job {job.get('id')}: invalid schedule '{schedule}'.")
                drop = True
        else:
            try:
                fire = cron_matches(schedule, now) and job.get("last_fired") != minute_key
            except ValueError:
                events.append(f"Removed job {job.get('id')}: invalid schedule '{schedule}'.")
                drop = True

        if fire:
            job["last_fired"] = minute_key
            text = job.get("text", "")
            if text.startswith("cmd:"):
                shell_cmd = text[4:].strip()
                ok, out = sandbox.run_sandboxed(shell_cmd, config, interactive=False)
                events.append(
                    f"Scheduled job {job.get('id')} ran '{shell_cmd}' "
                    f"({'ok' if ok else 'error'}):\n{out}"
                )
            else:
                events.append(f"Scheduled reminder: {text}")

        if fire or drop:
            changed = True
        if not drop:
            remaining.append(job)

    if changed:
        save_cron_jobs(remaining)
    return events
