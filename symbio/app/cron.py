"""Scheduled jobs: 5-field cron expressions and one-shot reminders.

A job is a REMINDER -- its text goes to whoever is chatting when it comes due
-- unless the user gave it a standing permission when it was made. Then it is
a TASK: `symb watch` runs it as an ordinary turn, with every tool, and the
approval questions that turn raises are answered from that permission rather
than refused for want of anyone to ask. Posting four times a day, a weekly
read of how the posts did -- any recurring work is a task like this; nothing
in here knows what the work is.
"""

import re
import shlex
from datetime import datetime, timedelta
from typing import Any

import json

from symbio import constants
from symbio.app import sandbox, security

# Kinds of action (symbio/guardrails.py) a task may be given in advance.
# Changing settings, schedules, training or deleting memory are never among
# them: an unattended run must not rewrite the rules it runs under.
GRANTABLE = ("publish", "browser", "files", "commands", "desktop")


def site_host(site: str) -> str:
    """'https://www.x.com/home' -> 'x.com'."""
    host = re.sub(r"^[a-z]+://", "", str(site or "").strip().lower()).split("/")[0]
    return host.split(":")[0].removeprefix("www.")


def normalize_grant(allow: Any, sites: Any) -> tuple[list[str], list[str]]:
    """(kinds, hosts) a job may act on unattended; ValueError for anything else."""
    if isinstance(allow, str):
        allow = [a for a in re.split(r"[,\s]+", allow) if a]
    if isinstance(sites, str):
        sites = [s for s in re.split(r"[,\s]+", sites) if s]
    kinds: list[str] = []
    for kind in allow or []:
        kind = str(kind).strip().lower()
        if kind not in GRANTABLE:
            raise ValueError(f"'{kind}' cannot be granted to a scheduled job. It may "
                             f"be given: {', '.join(GRANTABLE)}.")
        if kind not in kinds:
            kinds.append(kind)
    hosts = sorted({site_host(s) for s in sites or [] if site_host(s)})
    if hosts and not kinds:
        raise ValueError("'sites' only narrows what 'allow' grants; give 'allow' too.")
    return kinds, hosts


def is_task(job: dict[str, Any]) -> bool:
    return bool(job.get("allow"))


def describe_grant(kinds: list[str], hosts: list[str]) -> str:
    """The grant in the words of the guardrail switches: 'Post or send, Browse the web, only on x.com'."""
    from symbio import guardrails

    if not kinds:
        return "nothing"
    where = f", only on {', '.join(hosts)}" if hosts else ", on any site" if (
        {"publish", "browser"} & set(kinds)) else ""
    return ", ".join(guardrails.label(k) for k in kinds) + where


def _check_grant_fits(text: str, kinds: list[str]) -> None:
    if kinds and text.startswith(("cmd:", "script:")):
        raise ValueError("A cmd: or script: job runs in the sandbox and takes no grant; "
                         "describe the work as a task instead.")


def _check_script_job(text: str) -> None:
    """A script: job names a script that exists, now rather than at 3am."""
    if text.startswith("script:"):
        from symbio.app import scripts

        name, _args = scripts.split_job_text(text)
        scripts.load_script(name)


def load_cron_jobs() -> list[dict[str, Any]]:
    if not constants.CRON_FILE.exists():
        return []
    try:
        return json.loads(constants.CRON_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []


def _require_ownership(job: dict[str, Any], owner: str | None):
    """Raise if the caller does not own the job. Jobs with no owner are
    treated as legacy/unowned and can be managed by any caller (backwards
    compatibility)."""
    if owner is None:
        return
    job_owner = job.get("owner")
    if job_owner is not None and job_owner != owner:
        raise ValueError(f"Job {job.get('id')} is owned by another session.")


def save_cron_jobs(jobs: list[dict[str, Any]]):
    constants.CRON_FILE.write_text(json.dumps(jobs, indent=2), encoding="utf-8")


def list_cron_jobs() -> list[dict[str, Any]]:
    """Return all scheduled jobs sorted by id."""
    jobs = load_cron_jobs()
    jobs.sort(key=lambda j: j.get("id", 0))
    return jobs


def delete_cron_job(job_id: int, owner: str | None = None) -> dict[str, Any]:
    """Remove the job with the given id. Return the deleted job or raise ValueError.
    If `owner` is provided, only jobs owned by that owner can be deleted."""
    jobs = load_cron_jobs()
    for i, job in enumerate(jobs):
        if job.get("id") == job_id:
            _require_ownership(job, owner)
            removed = jobs.pop(i)
            save_cron_jobs(jobs)
            return removed
    raise ValueError(f"No job with id {job_id}.")


def update_cron_job(
    job_id: int,
    schedule: str | None = None,
    text: str | None = None,
    blocked_commands: set[str] | None = None,
    owner: str | None = None,
    allow: Any = None,
    sites: Any = None,
) -> dict[str, Any]:
    """Edit an existing job's schedule and/or text, and its grant.
    If `owner` is provided, only jobs owned by that owner can be updated.
    `allow` replaces the grant ([] takes it away); None leaves it as it was."""
    jobs = load_cron_jobs()
    job = next((j for j in jobs if j.get("id") == job_id), None)
    if job is None:
        raise ValueError(f"No job with id {job_id}.")
    _require_ownership(job, owner)
    if allow is not None or sites is not None:
        if allow is not None and not allow:
            kinds, hosts = [], []  # the grant taken away, sites with it
        else:
            kinds, hosts = normalize_grant(
                job.get("allow") if allow is None else allow,
                job.get("sites") if sites is None else sites)
        _check_grant_fits((text or job.get("text", "")).strip(), kinds)
        job.pop("allow", None)
        job.pop("sites", None)
        if kinds:
            job["allow"] = kinds
        if hosts:
            job["sites"] = hosts

    new_schedule = (schedule or job.get("schedule", "")).strip()
    new_text = (text or job.get("text", "")).strip()
    if not new_text:
        raise ValueError("Job text is empty.")
    _check_script_job(new_text)
    if new_text.startswith("cmd:"):
        shell_cmd = new_text[4:].strip()
        try:
            args = shlex.split(shell_cmd)
        except ValueError as e:
            raise ValueError(f"Invalid command: {e}")
        if blocked_commands and args and args[0] in blocked_commands:
            raise ValueError(
                f"Cannot schedule blocked command '{args[0]}' in cron — "
                f"interactive approval is not possible when the job fires."
            )

    one_shot = parse_one_shot(new_schedule)
    if one_shot:
        new_schedule = f"at {one_shot:%Y-%m-%d %H:%M}"
    else:
        error = validate_cron_expr(new_schedule)
        if error:
            raise ValueError(error)

    job["schedule"] = new_schedule
    job["text"] = new_text
    save_cron_jobs(jobs)
    return job


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


def add_cron_job(
    schedule: str,
    text: str,
    blocked_commands: set[str] | None = None,
    owner: str | None = None,
    allow: Any = None,
    sites: Any = None,
) -> dict[str, Any]:
    schedule = schedule.strip()
    text = text.strip()
    if not text:
        raise ValueError("Job text is empty.")
    kinds, hosts = normalize_grant(allow, sites)
    _check_grant_fits(text, kinds)
    _check_script_job(text)
    if text.startswith("cmd:"):
        shell_cmd = text[4:].strip()
        try:
            args = shlex.split(shell_cmd)
        except ValueError as e:
            raise ValueError(f"Invalid command: {e}")
        if blocked_commands and args and args[0] in blocked_commands:
            raise ValueError(
                f"Cannot schedule blocked command '{args[0]}' in cron — "
                f"interactive approval is not possible when the job fires."
            )
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
        "owner": owner,
    }
    if kinds:
        job["allow"] = kinds
    if hosts:
        job["sites"] = hosts
    jobs.append(job)
    save_cron_jobs(jobs)
    return job


class Fired(str):
    """A fired job's event text, still a plain string, carrying the job itself.

    The event is what every caller has always printed or handed on; the job
    rides along for the one that needs more -- the supervisor, which runs a
    task as a turn and answers its approval questions from its grant.
    """

    job: dict[str, Any] | None

    def __new__(cls, text: str, job: dict[str, Any] | None = None):
        fired = super().__new__(cls, text)
        fired.job = job
        return fired


def task_turn(job: dict[str, Any]) -> str:
    """What a due task says to the model: the work, and what it may do alone."""
    grant = describe_grant(job.get("allow") or [], job.get("sites") or [])
    return (f"Scheduled task {job.get('id')} is due ({job.get('schedule')}). Do it now, "
            f"with your tools, the way you would if I had just asked you. I'm not here "
            f"to answer questions. For this task you may do this without asking me: "
            f"{grant}. Anything else that needs my approval will be declined.\n\n"
            f"The task: {job.get('text', '')}")


def check_due_jobs(config: dict[str, Any], now: datetime | None = None,
                   include_tasks: bool = True) -> list[str]:
    """Fire all due jobs and return their event messages. One-shot jobs are
    removed after firing; recurring jobs fire at most once per minute.

    `include_tasks=False` leaves tasks alone -- not fired, not marked -- for
    the supervisor, which runs them. A chat session polls this too, and a task
    it fired would only become a line of text at the user's next message, and
    never run."""
    now = now or datetime.now()
    minute_key = now.strftime("%Y-%m-%d %H:%M")
    jobs = load_cron_jobs()
    events: list[str] = []
    remaining: list[dict[str, Any]] = []
    changed = False

    for job in jobs:
        if not include_tasks and is_task(job):
            remaining.append(job)
            continue
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
                # Checked again here, not only where the job was created. This
                # is the one place a stored command actually becomes a running
                # one, and it is reached by jobs written before the guard
                # existed and by anything that edits cron_jobs.json directly —
                # neither of which passed through the tool chokepoint. It fires
                # with interactive=False, so there is nobody to ask either.
                blocked = security.block_reason("run_command", {"cmd": shell_cmd})
                if blocked is not None:
                    # Reported, not run — and then treated exactly like a job
                    # that did fire, so a one-shot still drops. Keeping it
                    # would re-refuse the same command every minute forever.
                    events.append(
                        f"Scheduled job {job.get('id')} was not run: '{shell_cmd}' "
                        f"is refused.\n{blocked}"
                    )
                else:
                    ok, out = sandbox.run_sandboxed(shell_cmd, config, interactive=False)
                    events.append(
                        f"Scheduled job {job.get('id')} ran '{shell_cmd}' "
                        f"({'ok' if ok else 'error'}):\n{out}"
                    )
            elif text.startswith("script:"):
                # A script the model saved, run in the sandbox like
                # execute_code -- no model turn, nothing to approve.
                from symbio.app import scripts

                try:
                    name, args = scripts.split_job_text(text)
                    ok, out = scripts.run_script(name, args, config)
                except ValueError as e:
                    name, ok, out = text[len("script:"):].strip(), False, str(e)
                events.append(Fired(
                    f"Scheduled job {job.get('id')} ran script {name} "
                    f"({'ok' if ok else 'error'}):\n{out}", dict(job)))
            elif is_task(job):
                events.append(Fired(task_turn(job), dict(job)))
            else:
                events.append(Fired(f"Scheduled reminder: {text}", dict(job)))

        if fire or drop:
            changed = True
        if not drop:
            remaining.append(job)

    if changed:
        save_cron_jobs(remaining)
    return events


# ---------- answering a task's questions, with nobody there ----------

_DOMAIN = re.compile(r"\b(?:[a-z0-9-]+\.)+[a-z]{2,}\b", re.IGNORECASE)
# The browser's own question before a site it has not visited: no kind of its
# own (a site is asked about by name), but it is browsing all the same.
_NEW_SITE = re.compile(r"^Open \S+ in Symbio's browser", re.IGNORECASE)


class UnattendedApprover:
    """The answer to each approval question a scheduled task's turn raises.

    Yes when the question is of a kind the user granted this job, on a site
    they named -- a post on x.com for a job allowed to post there. No to
    everything else, and no to anything the harness flagged on the card (the
    text in the box is not what the task said to send). A grant of `publish`
    carries `browser` with it: nothing is posted without opening the page.
    `approve_all` is cron.unattended_approve, the old blanket switch.
    """

    def __init__(self, job: dict[str, Any], approve_all: bool = False) -> None:
        self.kinds = set(job.get("allow") or [])
        if "publish" in self.kinds:
            self.kinds.add("browser")
        self.sites = set(job.get("sites") or [])
        self.approve_all = approve_all
        self.approved: list[str] = []
        self.denied: list[str] = []

    def __call__(self, frame: dict[str, Any]) -> bool:
        card = (frame or {}).get("card") or {}
        headline = str(card.get("headline") or (frame or {}).get("prompt") or "")
        if self.approve_all:
            self.approved.append(headline[:120])
            return True
        kind = card.get("kind")
        named = {site_host(d) for d in _DOMAIN.findall(headline)}
        if kind is None and _NEW_SITE.match(headline):
            kind = "browser"
        if kind not in self.kinds:
            return self._no(f"not granted ({card.get('kind_label') or kind or 'unkeyed'}): "
                            f"{headline[:120]}")
        if card.get("warning"):
            return self._no(f"the harness flagged it: {card['warning']}")
        if self.sites and kind in ("publish", "browser") and named - self.sites:
            return self._no(f"{', '.join(sorted(named - self.sites))} is not one of "
                            f"{', '.join(sorted(self.sites))}")
        self.approved.append(headline[:120])
        return True

    def _no(self, why: str) -> bool:
        self.denied.append(why)
        return False
