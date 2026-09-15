"""`/voice` at the chat prompt.

Nothing here speaks and nothing here writes the developer's config: `say` is a
recorder and CONFIG_FILE points at tmp_path. A suite that makes noise gets run
with the volume off, and one that edits your real settings gets run once.
"""
from types import SimpleNamespace

import pytest

from symbio import constants
from symbio.app import tts

VOICE_LISTING = (
    "Fred                en_US    # I sure like being inside this fancy computer.\n"
    "Bells               en_US    # Time flies when you are having fun.\n"
    "Karen               en_AU    # Hello! My name is Karen.\n"
    "Daniel              en_GB    # Hello! My name is Daniel.\n"
    "Samantha            en_US    # Hello! My name is Samantha.\n"
    "Moira               en_IE    # Hello! My name is Moira.\n"
    "Flo (English (UK))  en_GB    # Hello! My name is Flo.\n"
)


@pytest.fixture
def voice(monkeypatch, tmp_path):
    """A chat shell with /voice wired up, a fixed voice install, and a
    throwaway config file."""
    from symbio.app.chat_commands import CommandsMixin

    spoken: list[list[str]] = []

    class _Listing:
        stdout = VOICE_LISTING
        returncode = 0

    class _Fake:
        def __init__(self, cmd, **_kw):
            spoken.append(cmd)

        def poll(self):
            return None

        def terminate(self):
            pass

    monkeypatch.setattr(tts, "_voices_cache", None)
    monkeypatch.setattr(tts, "_system_rate_cache", (None,))
    monkeypatch.setattr(tts, "_system_locale", lambda: "en_AU")
    monkeypatch.setattr(tts.shutil, "which", lambda _n: "/usr/bin/say")
    monkeypatch.setattr(tts.subprocess, "run", lambda *_a, **_k: _Listing())
    monkeypatch.setattr(tts.subprocess, "Popen", _Fake)
    monkeypatch.setattr(tts.sys.stdout, "isatty", lambda: True)
    monkeypatch.setattr(tts, "_speaking", None)
    monkeypatch.delenv("SYMBIO_NO_TTS", raising=False)
    monkeypatch.setattr(constants, "CONFIG_FILE", tmp_path / "config.json")

    out: list[str] = []

    class Fake(CommandsMixin):
        def __init__(self):
            self.output_fn = out.append
            self.session_id = "current"
            self.retriever = SimpleNamespace(invalidate_cache=lambda: None)
            self.config = {"tts": {"enabled": True, "voice": "", "gender": "",
                                   "accent": "", "depth": 0.5,
                                   "rate_slider": 0.5, "rate": 0,
                                   "max_chars": 600}}

    shell = Fake()

    def run(text=""):
        out.clear()
        spoken.clear()
        shell._cmd_voice(text.split())
        return "\n".join(out), spoken

    return SimpleNamespace(run=run, shell=shell, spoken=spoken)


def test_bare_voice_lists_what_is_installed(voice):
    text, spoken = voice.run()
    # Flo is listed as "Flo (English (UK))" — the qualifier is there so `say -v`
    # gets the right accent, not to be read in a list already grouped by one.
    assert "British: Daniel (male), Flo (female)" in text
    assert "Australian: Karen (female)" in text
    assert "Bells" not in text                      # novelty voices stay hidden
    assert spoken == []                             # listing does not speak


def test_plain_words_change_and_speak(voice):
    text, spoken = voice.run("female british")
    assert voice.shell.config["tts"]["gender"] == "female"
    assert voice.shell.config["tts"]["accent"] == "british"
    # The full name, qualifier included: `say -v Flo` and
    # `say -v "Flo (English (UK))"` are different audio.
    assert spoken and spoken[0][:3] == ["say", "-v", "Flo (English (UK))"]
    assert "Flo" in text and "No female" not in text


def test_the_change_is_persisted_not_just_applied(voice):
    import json

    voice.run("male british deeper")
    saved = json.loads(constants.CONFIG_FILE.read_text())["tts"]
    assert saved["gender"] == "male" and saved["accent"] == "british"
    assert saved["depth"] == 0.75


def test_deeper_reaches_the_spoken_command(voice):
    _text, spoken = voice.run("deeper")
    assert spoken[0][-1].startswith("[[pbas "), spoken


def test_off_stops_it_speaking_and_says_so(voice):
    text, spoken = voice.run("off")
    assert voice.shell.config["tts"]["enabled"] is False
    assert "off" in text.lower()
    assert spoken == []                             # muting does not speak


def test_a_name_clears_the_gender_and_accent_it_contradicts(voice):
    voice.run("female british")
    voice.run("daniel")
    cfg = voice.shell.config["tts"]
    assert cfg["voice"] == "Daniel"                  # capitalised as installed
    assert cfg["gender"] == "" and cfg["accent"] == ""


def test_an_unknown_word_is_reported_not_swallowed(voice):
    text, _spoken = voice.run("female martian")
    assert "martian" in text
    assert "Australian" in text                      # offers what is possible
    assert voice.shell.config["tts"]["gender"] == "female"   # the rest applies


def test_nothing_understood_changes_nothing(voice):
    before = dict(voice.shell.config["tts"])
    text, spoken = voice.run("sexy")
    assert voice.shell.config["tts"] == before
    assert spoken == []
    assert "sexy" in text


def test_normal_puts_everything_back(voice):
    voice.run("daniel deeper faster")
    voice.run("normal")
    cfg = voice.shell.config["tts"]
    assert cfg["depth"] == 0.5 and cfg["rate_slider"] == 0.5
    assert cfg["voice"] == "" and cfg["gender"] == ""


def test_on_starts_speaking_and_proves_it_by_speaking(voice):
    """The confirmation is the sample itself: if turning it on did not
    actually work, there is nothing to hear and the line would be a lie."""
    voice.shell.config["tts"]["enabled"] = False
    text, spoken = voice.run("on")
    assert voice.shell.config["tts"]["enabled"] is True
    assert spoken and spoken[0][0] == "say"
    assert "off" not in text.lower()


def test_on_and_a_voice_in_one_breath(voice):
    _text, spoken = voice.run("on female irish faster")
    cfg = voice.shell.config["tts"]
    assert cfg["enabled"] is True and cfg["gender"] == "female"
    assert spoken[0][:3] == ["say", "-v", "Moira"]
    assert "-r" in spoken[0]
