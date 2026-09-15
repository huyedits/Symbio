"""Credentials must not survive a turn.

An API that wants an Authorization header leaves the model no choice but to
inline the token in the script it writes: run_python_code passes
env={"PATH": ...} only, and `os` is a blocked import, so there is no other way
to supply one. Driving the CLI against such an API on 2026-08-27 produced
exactly that, and the tool loop's automatic mistake-note pipeline wrote the
whole script to notes/mistakes/ -- after the session had been answered "n" at
"Save conversation for training?", which gates save_history_pairs and nothing
else. These tests pin every write boundary that outlives the turn.
"""
import json

import pytest

from symbio import constants
from symbio.app import learn, sessions, training
from symbio.app.tooling import redact_messages, redact_secrets

# The live case, verbatim from the 2026-08-27 run.
LIVE_SCRIPT = (
    "import requests\n"
    "response = requests.get('http://127.0.0.1:8787/items', "
    "headers={'Authorization': 'Bearer sk_live_7Q2'})\n"
    "data = response.json()\n"
)


class Tok:
    def apply_chat_template(self, messages, tokenize=False, **kw):
        return "\n".join(f"<{m['role']}>{m['content']}" for m in messages)


# ---- redact_secrets ----

@pytest.mark.parametrize("text,secret", [
    ("headers={'Authorization': 'Bearer sk_live_7Q2'}", "sk_live_7Q2"),
    ('Authorization: Bearer ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ012345', "ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ012345"),
    ('curl -H "Authorization: Token abc123def456" https://x', "abc123def456"),
    ('api_key="AIzaSyD1234567890abcdefghijklmnopqrstuvw"', "AIzaSyD1234567890abcdefghijklmnopqrstuvw"),
    ('{"access_token": "eyJhbGciOiJIUzI1NiJ9.abc.def"}', "eyJhbGciOiJIUzI1NiJ9.abc.def"),
    ("AWS key AKIAIOSFODNN7EXAMPLE here", "AKIAIOSFODNN7EXAMPLE"),
    ("client_secret=abcdefghijklmnop", "abcdefghijklmnop"),
    ("-----BEGIN RSA PRIVATE KEY-----\nMIIEpQIB\n-----END RSA PRIVATE KEY-----", "MIIEpQIB"),
])
def test_secret_is_removed(text, secret):
    out = redact_secrets(text)
    assert secret not in out
    assert "[redacted]" in out


@pytest.mark.parametrize("text,secret", [
    ("headers={'X-Api-Key': 'abc123XYZ789secret'}", "abc123XYZ789secret"),
    ("aws_secret_access_key = wJalrXUtnFEMI/K7MDENG/bPx", "wJalrXUtnFEMI"),
    ("https://api.x.com/v2?api_key=abc123XYZ789secret&q=1", "abc123XYZ789secret"),
    ("export OPENAI_API_KEY=sk-proj-abcdefghijklmnop", "sk-proj-abcdefghijklmnop"),
    ("the response was eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.abc", "eyJzdWIiOiIxIn0"),
    ("postgres://admin:s3cr3tp4ss@db.host:5432/prod", "s3cr3tp4ss"),
    ("curl https://user:hunter2pass@example.com/api", "hunter2pass"),
    ("https://hooks.slack.com/services/T0/B0/XXXXXXXXXXXXXXXXXXXXXXXX",
     "XXXXXXXXXXXXXXXXXXXXXXXX"),
])
def test_secret_is_removed_in_the_shapes_that_leaked(text, secret):
    """Found by probing the first version: a key name the rule could not see
    (X-Api-Key, aws_secret_access_key), and credentials carried positionally
    rather than after a name (URL userinfo, webhook paths, a bare JWT)."""
    out = redact_secrets(text)
    assert secret not in out
    assert "[redacted]" in out


@pytest.mark.parametrize("text", [
    "generated 42 tokens (5s)",              # the CLI's own spinner
    "the token is short",
    "password: hunter",                      # too short to be a credential
    "Use the Bearer pattern for auth.",      # prose, no credential follows
    "What is the capital of France?",
    "Content-Type: application/json",
    "the tokenizer: apply_chat_template renders it",   # "token" inside a word
    "access_token is the field name you want",
    "Error: token limit exceeded on this model",
    "def get_token(self): return self._token",
    "note: password rotation is quarterly",
    "GET /v2/items HTTP/1.1",
])
def test_ordinary_text_is_untouched(text):
    assert redact_secrets(text) == text


def test_calling_shape_survives_redaction():
    """The sample must still teach that an Authorization header is sent."""
    out = redact_secrets(LIVE_SCRIPT)
    assert "sk_live_7Q2" not in out
    assert "'Authorization': 'Bearer [redacted]'" in out
    assert "requests.get(" in out


def test_redact_messages_keeps_roles():
    out = redact_messages([{"role": "user", "content": "key=sk_live_ABCDEFGH"}])
    assert out[0]["role"] == "user"
    assert "sk_live_ABCDEFGH" not in out[0]["content"]


# ---- the write boundaries ----

def test_training_corpus_is_redacted(tmp_path, monkeypatch):
    train = tmp_path / "train.jsonl"
    monkeypatch.setattr(constants, "TRAIN_FILE", train)
    monkeypatch.setattr(training, "_train_file_for", lambda role=None: train)
    training.append_training_text(
        LIVE_SCRIPT, messages=[{"role": "user", "content": LIVE_SCRIPT}])
    record = json.loads(train.read_text().splitlines()[0])
    assert "sk_live_7Q2" not in record["text"]
    assert "sk_live_7Q2" not in json.dumps(record["messages"])


def test_history_pairs_are_redacted(tmp_path, monkeypatch):
    monkeypatch.setattr(constants, "TRAIN_FILE", tmp_path / "train.jsonl")
    monkeypatch.setattr(constants, "VALID_FILE", tmp_path / "valid.jsonl")
    history = [
        {"role": "user", "content": "pull my open issues"},
        {"role": "assistant", "content": f"<tool_call>{LIVE_SCRIPT}</tool_call>"},
    ]
    assert training.save_history_pairs(history, Tok(), "SYS") == 1
    written = "".join(p.read_text() for p in tmp_path.glob("*.jsonl"))
    assert "sk_live_7Q2" not in written


def test_mistake_note_is_redacted(tmp_path, monkeypatch):
    """The path that bypassed the save prompt in the live run."""
    monkeypatch.setattr(constants, "MISTAKES_DIR", tmp_path)
    path = learn.save_mistake_note(
        original_query="fetch my issues",
        wrong_answer="(a prior tool call failed)",
        correction="(automatic: the next tool call succeeded)",
        correct_answer=LIVE_SCRIPT,
    )
    assert "sk_live_7Q2" not in path.read_text()


def test_session_log_is_redacted(tmp_path, monkeypatch):
    monkeypatch.setattr(constants, "SESSIONS_DIR", tmp_path)
    store = sessions.SessionStore("redaction-test")
    store.log("tool", f"execute_code: {LIVE_SCRIPT}")
    assert "sk_live_7Q2" not in store.path.read_text()
