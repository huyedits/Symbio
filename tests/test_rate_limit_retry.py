"""A rate limit is the one failure whose fix is the identical call again.

Measured live 2026-08-27 against an API that 429s twice before serving: the
script that received the 429 ran fine, so sounds_like_tool_error was False,
failed_calls never counted it, and the key stayed in executed_calls -- where
the fresh-tool filter drops a reissue as "already done". The model could not
repeat the call at all, and extra tool rounds would not have changed that.
"""
import pytest

from symbio.app import chat, learn


@pytest.mark.parametrize("obs", [
    '429 {"error": "rate limited"}',
    "HTTP 429 from http://x. Retry-After: 7s.",
    "Too Many Requests",
    '{"detail": "slow down, retry shortly"}',
    "503 Service Unavailable",
])
def test_recognises_a_rate_limit(obs):
    assert learn.is_rate_limited(obs)


@pytest.mark.parametrize("obs", [
    "Tool 'run_command' was not approved (risk score 3/3: blocked_import:os)",
    "user denied the request",
])
def test_a_refusal_is_never_a_rate_limit(obs):
    """No wait turns a "no" into a "yes"; retrying only re-prompts the user."""
    assert not learn.is_rate_limited(obs)


@pytest.mark.parametrize("obs", [
    "Python script exited ok.\nOutput:\n3 items found.",
    "the capital of France is Paris",
])
def test_ordinary_observations_are_not_rate_limits(obs):
    assert not learn.is_rate_limited(obs)


@pytest.mark.parametrize("obs,expected", [
    ("HTTP 429 from http://x. Retry-After: 7s.", 7.0),
    ("retry-after 1", 1.0),
    ("429 rate limited", None),
])
def test_retry_after_is_read_when_present(obs, expected):
    assert learn.retry_after_seconds(obs) == expected


def test_rate_limit_budget_exceeds_the_failure_budget():
    """Repeating a rate-limited call is the documented fix, not a loop, so it
    must not draw on the budget for a call that is actually going wrong."""
    assert chat._MAX_RATE_LIMIT_RETRIES > chat._MAX_TOOL_RETRIES


def test_the_wait_is_bounded():
    """Retry-After is the server's number, not a licence to stall the CLI."""
    assert chat._MAX_RATE_LIMIT_WAIT <= 10


def test_default_rounds_clear_a_moved_and_limited_api():
    """410 -> 429 -> 429 -> 200 is four rounds before the first real byte."""
    from symbio.app import config as app_config
    assert app_config.DEFAULT_CONFIG["agent"]["max_tool_rounds"] >= 4
