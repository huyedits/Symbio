"""The terminal skin must stay a skin.

The bracket tags it restyles are a protocol before they are decoration: the
same vocabulary goes into self.history where the model reads it, and this
project's bug history is largely two representations of one fact drifting
apart. So the tests that matter here are the ones that pin what this must
NOT change.
"""

from symbio.app import chat_style as cs


# ---- off a terminal, nothing changes at all ----

_LINES = [
    "  [Tool: browser_open]",
    "  [Observation] Opened browser at http://example.com.",
    "  [Vision] Looking at the browser page...",
    "  [Security: risk score 3/3] Run this command?",
    "  [Mood: neutral]",
    "Just some prose.",
]


def test_piped_output_is_byte_identical():
    """A log file, a grep, and the repo's own pty harnesses all match on
    "[Tool: ...]" and "[Observation] ...". Reshaping those for an audience
    that cannot see colour anyway would break them for nothing."""
    for line in _LINES:
        assert cs.style_line(line, color=False) == line


def test_no_color_env_disables_styling(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    assert cs.colors_enabled() is False


def test_a_stream_that_cannot_be_asked_is_not_a_terminal():
    class _Odd:
        def isatty(self):
            raise OSError("detached")

    assert cs.colors_enabled(_Odd()) is False


# ---- on a terminal, the shape a reader expects ----

def test_a_tool_call_becomes_a_bullet():
    out = cs.style_line("  [Tool: browser_open]", color=True)
    assert cs.BULLET in out
    assert "browser_open" in out
    assert "[Tool:" not in out


def test_an_observation_hangs_off_its_call():
    out = cs.style_line("  [Observation] Opened browser.", color=True)
    assert cs.ELBOW in out
    assert "Opened browser." in out


def test_multiline_output_stays_aligned():
    out = cs.style_line("  [Observation] first\nsecond\nthird", color=True)
    body = out.split("\n")
    assert len(body) == 3
    assert body[1].startswith("     "), body[1]


def test_a_security_label_keeps_its_punctuation():
    """The tag's own colon lives in the second capture group; joining with a
    space rendered it as "Security : risk score 3/3"."""
    out = cs.style_line("  [Security: risk score 3/3] Run this?", color=True)
    assert "Security: risk score 3/3" in out
    assert "Security :" not in out


def test_an_alert_is_not_dimmed_into_the_background():
    alert = cs.style_line("  [Unverified] claims work with no tool call", color=True)
    routine = cs.style_line("  [Dispatch] Headmaster awake.", color=True)
    assert cs._RED in alert
    assert cs._RED not in routine


def test_an_unknown_tag_is_still_shown():
    """A skin over a protocol that keeps growing must not swallow the line
    somebody just added."""
    out = cs.style_line("  [Brand-New-Thing] something happened", color=True)
    assert "something happened" in out


def test_prose_passes_through_untouched():
    assert cs.style_line("Just some prose.", color=True) == "Just some prose."


def test_blank_and_non_strings_survive():
    assert cs.style_line("") == ""
    assert cs.style_line(None) is None


# ---- the prompts the two audiences see ----

def test_the_prompts_are_plain_without_colour():
    assert "\033" not in cs.user_prompt(color=False)
    assert "\033" not in cs.assistant_prefix(color=False)


def test_the_session_keeps_the_parseable_prompt_for_scripted_runs(monkeypatch, tmp_path):
    """tests/verify_transcript_fixes.py and the drive harnesses wait for the literal
    "Huy     : " prompt. A styled prompt handed to a scripted front-end would
    hang them forever."""
    from symbio import constants
    from symbio.app import chat
    from symbio.app import config as app_config

    class FakeTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return ""

    (tmp_path / "adapters").mkdir(parents=True, exist_ok=True)
    (tmp_path / "training_data").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(constants, "ADAPTER_DIR", tmp_path / "adapters")
    monkeypatch.setattr(constants, "DATA_DIR", tmp_path / "training_data")
    monkeypatch.setattr(constants, "LOG_DIR", tmp_path / "logs")
    monkeypatch.setattr(chat, "load", lambda *a, **k: (object(), FakeTokenizer()))

    session = chat.ChatSession(
        app_config.load_config(), model=object(), tokenizer=FakeTokenizer(),
        adapter_loaded=False, output_fn=lambda *a: None,
        input_fn=lambda *a: "", generate_fn=lambda *a, **k: "",
    )
    assert session.user_prompt() == f"{session.config['user_name']:8}: "
    assert session.assistant_prefix().startswith(
        f"{session.config['assistant_name']:8}")


def test_each_line_of_a_multiline_status_is_styled():
    """Seen in a real session: one output_fn call carried two [Learn] lines
    and only the first was styled — the tag patterns end in DOTALL groups, so
    the first match swallowed the rest as its own body:

        · learn Tool mistake captured: ...
          [Learn] 3/5 mistake note(s) collected; training after 2 more.
    """
    msg = ("  [Learn] Tool mistake captured: note.md\n"
           "  [Learn] 3/5 mistake note(s) collected; training after 2 more.")
    out = cs.style_line(msg, color=True)

    assert "[Learn]" not in out, out
    assert out.count("· learn") == 2


def test_an_observation_keeps_its_body_under_one_elbow():
    """The exception: a tool's multi-line output is one observation, not
    several status lines, and hangs off a single elbow."""
    out = cs.style_line("  [Observation] line one\nline two\nline three",
                        color=True)
    assert out.count(cs.ELBOW) == 1
    assert out.split("\n")[1].startswith("     ")
