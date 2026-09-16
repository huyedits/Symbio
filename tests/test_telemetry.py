#!/usr/bin/env python3
"""Unit tests for the telemetry + /feedback feature (no network).

Covers: collect_env excludes PII; consent defaults; local JSONL fallback when
no endpoint; the POST shape (monkeypatched urllib); and the /feedback command
(disabled → blocked + no send)."""
import copy
import json
import os
import tempfile
from pathlib import Path
from unittest import mock

from symbio import constants
from symbio.app import chat, config as app_config, telemetry


def _cfg(**overrides):
    # Build from DEFAULT_CONFIG (not the user's live config.json) so tests are
    # deterministic regardless of whether the user has already run /telemetry on.
    cfg = copy.deepcopy(app_config.DEFAULT_CONFIG)
    tcfg = cfg.setdefault("telemetry", {})
    tcfg.setdefault("enabled", False)
    tcfg.setdefault("consented", False)
    tcfg.setdefault("endpoint", "")
    tcfg.setdefault("shared_secret", "")
    tcfg.setdefault("feedback_enabled", True)
    tcfg.setdefault("ping_daily", True)
    for k, v in overrides.items():
        tcfg[k] = v
    return cfg


def _tmp_telemetry(tmpdir):
    """Redirect the telemetry module's disk paths into a temp dir."""
    td = Path(tmpdir)
    telemetry._TELEMETRY_DIR = td
    telemetry._STATE_FILE = td / "state.json"
    return td


def test_collect_env_excludes_pii():
    cfg = _cfg()
    cfg["user_name"] = "Huy"
    cfg["assistant_name"] = "Caine"
    env = telemetry.collect_env(cfg)
    blob = json.dumps(env).lower()
    assert "huy" not in blob and "caine" not in blob, "names leaked into env"
    assert env["model_name"]  # present
    assert "session_count" not in env  # that's in the payload, not env
    # No field carries message/note/prompt text or paths.
    for k in ("messages", "notes", "prompts", "paths", "history"):
        assert k not in env, f"PII-ish key {k!r} present"


def test_consent_defaults_off():
    cfg = _cfg()
    tcfg = cfg["telemetry"]
    assert tcfg["enabled"] is False
    assert tcfg["consented"] is False


def test_send_local_fallback_writes_txt():
    with tempfile.TemporaryDirectory() as tmp:
        _tmp_telemetry(tmp)
        cfg = _cfg(endpoint="", feedback_enabled=True)
        ok, msg = telemetry.send(
            {"type": "feedback", "text": "hi there", "env": {"os": "Darwin"}}, cfg)
        assert ok, msg
        fb_path = Path(tmp) / "feedback.txt"
        assert fb_path.exists(), "feedback should write a human-readable .txt"
        content = fb_path.read_text()
        assert content.startswith("=== "), content
        assert "hi there" in content
        assert "os=Darwin" in content
        assert content.rstrip().endswith("---"), content
        # No JSON object — it's prose for a PR/Discussion, not machine JSONL.
        assert not content.strip().startswith("{")


def test_send_local_telemetry_writes_ndjson():
    with tempfile.TemporaryDirectory() as tmp:
        _tmp_telemetry(tmp)
        cfg = _cfg(endpoint="")
        ok, msg = telemetry.send(
            {"type": "telemetry", "env": {"os": "Darwin"}, "session_count": 1}, cfg)
        assert ok, msg
        ping_path = Path(tmp) / "pings.jsonl"
        assert ping_path.exists()
        line = json.loads(ping_path.read_text().strip())
        assert line["type"] == "telemetry"
        assert line["session_count"] == 1


def test_send_post_shape_with_secret():
    captured = {}

    class _FakeResp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self, n=-1):
            return b""

    def fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        captured["method"] = req.method
        captured["headers"] = {k.lower(): v for k, v in req.headers.items()}
        captured["data"] = json.loads(req.data.decode("utf-8"))
        return _FakeResp()

    with tempfile.TemporaryDirectory() as tmp:
        _tmp_telemetry(tmp)
        cfg = _cfg(endpoint="https://example.workers.dev/ingest",
                   shared_secret="s3cr3t")
        with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
            ok, msg = telemetry.send(
                {"type": "telemetry", "env": {"os": "Darwin"}, "session_count": 1},
                cfg,
            )
        assert ok, msg
        assert captured["url"] == "https://example.workers.dev/ingest"
        assert captured["method"] == "POST"
        assert captured["headers"]["x-telemetry-secret"] == "s3cr3t"
        assert captured["headers"]["x-telemetry-kind"] == "pings"
        assert captured["headers"]["content-type"] == "application/json"
        assert captured["data"]["type"] == "telemetry"


def test_maybe_daily_ping_skips_when_disabled():
    with tempfile.TemporaryDirectory() as tmp:
        _tmp_telemetry(tmp)
        cfg = _cfg(enabled=False, consented=True, ping_daily=True)
        state = telemetry.load_state()
        result = telemetry.maybe_daily_ping(cfg, state)
        assert result is None  # disabled → no attempt


def test_feedback_command_disabled_blocks_send():
    """A ChatSession with /feedback off must print the disabled message and
    NOT call telemetry.send_feedback on a follow-up /feedback <text>."""
    from test_main_loop import FakeTokenizer

    outputs = []
    with tempfile.TemporaryDirectory() as tmp:
        _tmp_telemetry(tmp)
        cfg = _cfg(feedback_enabled=True)
        # Force the tag index off (same reason as ScriptedSession).
        rag_cfg = cfg.setdefault("rag", {})
        rag_cfg["tag_index_enabled"] = False
        rag_cfg["auto_index_enabled"] = False

        # /feedback off persists via app save_config -> constants.CONFIG_FILE.
        # Redirect CONFIG_FILE to the temp dir so this test never overwrites
        # the user's real config.json with the DEFAULT_CONFIG built by _cfg().
        with mock.patch("symbio.app.telemetry.send_feedback",
                        return_value=(True, "sent")) as fake_send, \
             mock.patch.object(constants, "CONFIG_FILE", Path(tmp) / "config.json"):
            session = chat.ChatSession(
                cfg,
                model=object(),
                tokenizer=FakeTokenizer(),
                adapter_loaded=False,
                generate_fn=lambda *a, **k: "ok",
                stream_fn=None,
                stream_chunk_fn=lambda s: None,
                output_fn=lambda t: outputs.append(t),
                input_fn=lambda p="": "n",
                owner="test",
            )
            # 1. Turn /feedback off.
            session._handle_command("/feedback off")
            assert any("disabled" in o.lower() for o in outputs), outputs
            assert cfg["telemetry"]["feedback_enabled"] is False

            outputs.clear()
            # 2. Try to send feedback while disabled.
            session._handle_command("/feedback hello there")
            assert any("disabled" in o.lower() for o in outputs), outputs
            assert fake_send.call_count == 0, "send_feedback was called while disabled"


def _run():
    test_collect_env_excludes_pii()
    test_consent_defaults_off()
    test_send_local_fallback_writes_txt()
    test_send_local_telemetry_writes_ndjson()
    test_send_post_shape_with_secret()
    test_maybe_daily_ping_skips_when_disabled()
    test_feedback_command_disabled_blocks_send()
    print("test_telemetry: all passed")


if __name__ == "__main__":
    _run()