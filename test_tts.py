"""Speaking the reply aloud.

Nothing here plays audio: `say` is replaced with a recorder, because a test
suite that makes noise is one people run with the volume off, and then it is
testing nothing.
"""
import pathlib
import subprocess

import pytest

from symbio.app import tts


@pytest.fixture
def spoken(monkeypatch):
    """Captures the command instead of speaking it."""
    calls = []

    class _Fake:
        def __init__(self, cmd, **_kw):
            calls.append(cmd)

        def poll(self):
            return None

        def terminate(self):
            calls.append(["terminated"])

    monkeypatch.setattr(tts.subprocess, "Popen", _Fake)
    # system_rate() shells out too, and stubbing Popen breaks subprocess.run
    # underneath it. Stubbed out regardless of that: a test that reads the
    # developer's own Accessibility preferences gives a different answer on
    # every machine.
    monkeypatch.setattr(tts, "system_rate", lambda: None)
    monkeypatch.setattr(tts.shutil, "which", lambda _n: "/usr/bin/say")
    monkeypatch.setattr(tts.sys.stdout, "isatty", lambda: True)
    monkeypatch.setattr(tts, "_speaking", None)
    monkeypatch.delenv("SYMBIO_NO_TTS", raising=False)
    return calls


@pytest.fixture(autouse=True)
def installed(monkeypatch):
    """A fixed set of installed voices, fed in as real `say -v \'?\'` output.

    Two reasons it is shaped like this. First, without it every voice test asks
    the developer\'s own Mac what it has, and this Mac has no female British
    voice at all — so the suite would assert whatever happened to be installed
    the day it was written. Second, the listing goes through the real parser
    and the real ordering: an earlier version of this fixture set the cache
    directly, which skipped the sort entirely and made the "prefer a modern
    voice" test pass no matter what the sort did.

    So the order below is adversarial on purpose — the 1990s MacinTalk voices
    come first and sort first alphabetically, novelty voices are mixed in, and
    there is a male British voice with no female counterpart.
    """
    listing = (
        "Fred                en_US    # I sure like being inside this fancy computer.\n"
        "Bells               en_US    # Time flies when you are having fun.\n"
        "Karen               en_AU    # Hello! My name is Karen.\n"
        "Trinoids            en_US    # We cannot communicate with these carbon units.\n"
        "Daniel              en_GB    # Hello! My name is Daniel.\n"
        "Ralph               en_US    # Hello! My name is Ralph.\n"
        "Flo                 fr_FR    # Bonjour!\n"
        "Samantha            en_US    # Hello! My name is Samantha.\n"
        "Moira               en_IE    # Hello! My name is Moira.\n"
        "Zarvox              en_US    # That looks like a peaceful planet.\n"
        "Anna                de_DE    # Hallo! Ich heiße Anna.\n"
    )

    class _Listing:
        stdout = listing
        returncode = 0

    monkeypatch.setattr(tts, "_voices_cache", None)
    monkeypatch.setattr(tts, "_all_voices_cache", None)
    monkeypatch.setattr(tts.shutil, "which", lambda _n: "/usr/bin/say")
    monkeypatch.setattr(tts.subprocess, "run", lambda *_a, **_k: _Listing())
    monkeypatch.setattr(tts, "_system_locale", lambda: "en_AU")
    return listing


ON = {"tts": {"enabled": True}}


# ---- when it speaks at all ----

def test_off_by_default():
    """A machine that starts talking because someone updated is a machine
    someone mutes permanently."""
    assert tts.enabled({"tts": {}}) is False


def test_silent_without_a_terminal(monkeypatch):
    """Piped, redirected, or driven by the transcript harness — all cases where
    audio is noise in someone's background rather than output."""
    monkeypatch.setattr(tts.shutil, "which", lambda _n: "/usr/bin/say")
    monkeypatch.setattr(tts.sys.stdout, "isatty", lambda: False)

    assert tts.enabled(ON) is False


def test_one_env_var_switches_it_off(monkeypatch, spoken):
    monkeypatch.setenv("SYMBIO_NO_TTS", "1")

    assert tts.say("hello", ON) is False
    assert spoken == []


def test_it_speaks_when_asked(spoken):
    assert tts.say("All done.", ON) is True
    assert spoken[-1][0] == "say"
    assert "All done." in spoken[-1]


# ---- what it speaks ----

def test_code_and_links_are_not_read_aloud():
    """A voice reading a forty-line code block character by character is worse
    than silence."""
    said = tts.speakable(
        "Fixed it in `web.py`:\n\n```python\nx = 1\n```\n\nSee https://x.com/docs for why.")

    assert "x = 1" not in said
    assert "https" not in said
    assert "code block omitted" in said
    assert "Fixed it in" in said


def test_status_tags_are_not_read_aloud():
    """output_fn carries the whole transcript; only the answer is speech."""
    assert "browser_open" not in tts.speakable("[Tool: browser_open]\nOpened the page.")


def test_no_dangling_punctuation_where_code_was():
    """"I fixed it in : code block omitted" is what a naive strip produces, and
    a voice reads that colon as a pause in the wrong place."""
    said = tts.speakable("I fixed it in `web.py`:\n\n```py\nx\n```")

    assert " :" not in said


def test_a_long_reply_stops_on_a_sentence():
    """A voice that trails off mid-word sounds like a crash."""
    said = tts.speakable("One. " * 200, limit=120)

    assert len(said) <= 121
    assert said.endswith(".") or said.endswith("…")


def test_nothing_to_say_says_nothing(spoken):
    assert tts.say("```python\nonly code\n```", {"tts": {"enabled": True}}) is True
    assert tts.say("", ON) is False
    assert tts.say("   ", ON) is False


# ---- Apple's slider ----

@pytest.mark.parametrize("slider, wpm", [(0.0, 100), (0.5, 175), (1.0, 400)])
def test_the_slider_midpoint_is_the_macos_default(slider, wpm):
    """0.5 lands on 175 so half speed sounds like macOS does out of the box —
    a single straight ramp would have put the default at 0.25."""
    assert tts.slider_to_wpm(slider) == wpm


def test_the_slider_clamps_rather_than_extrapolates():
    assert tts.slider_to_wpm(9.0) == tts.FASTEST_WPM
    assert tts.slider_to_wpm(-9.0) == tts.SLOWEST_WPM


def test_a_raw_rate_beats_the_slider():
    """Someone who typed 300 meant 300; reinterpreting it through a 0-1 scale
    would be helpfulness nobody asked for."""
    assert tts.resolved_rate({"tts": {"rate": 300, "rate_slider": 0.1}}) == 300


def test_the_slider_is_used_when_no_raw_rate_is_set():
    assert tts.resolved_rate({"tts": {"rate_slider": 0.0}}) == 100


def test_the_rate_reaches_the_command(spoken):
    tts.say("hello", {"tts": {"enabled": True, "rate_slider": 1.0}})

    assert "-r" in spoken[-1]
    assert "400" in spoken[-1]


def test_a_voice_name_reaches_the_command(spoken):
    tts.say("hello", {"tts": {"enabled": True, "voice": "Samantha"}})

    assert spoken[-1][1:3] == ["-v", "Samantha"]


def test_a_named_voice_is_honoured_even_when_its_gender_is_unknown(spoken):
    """The gender table is a curated guess that will always be missing
    somebody's new download. Overriding a name because of that would be
    refusing the user their own Mac."""
    tts.say("hello", {"tts": {"enabled": True, "voice": "Zarvox"}})

    assert spoken[-1][1:3] == ["-v", "Zarvox"]


def test_a_voice_that_is_gone_falls_back_and_says_why(spoken):
    """`say -v Bogus` fails, and the process is detached with stderr
    discarded — so an uninstalled name is silence with no reason given."""
    name, note = tts.choose_voice({"tts": {"voice": "Bogus"}})

    assert name and name != "Bogus"
    assert "Bogus" in note and "not installed" in note
    assert tts.say("hello", {"tts": {"enabled": True, "voice": "Bogus"}})
    assert spoken[-1][1] == "-v"


def test_a_new_reply_interrupts_the_one_still_speaking(spoken):
    """Two voices over each other is worse than either."""
    tts.say("first", ON)
    tts.say("second", ON)

    assert ["terminated"] in spoken


# ---- voice character: gender, accent, depth ----

def test_novelty_and_foreign_voices_are_not_offered():
    """Bells and Trinoids ship on every Mac, and a French Flo is not an English
    voice just because the name is in the gender table. Modern voices sort
    ahead of the 1990s MacinTalk ones, so every "first match" is a best match.
    """
    assert tts.installed_voices() == [
        ("Daniel", "en_GB"), ("Karen", "en_AU"), ("Moira", "en_IE"),
        ("Samantha", "en_US"),                    # modern, alphabetical
        ("Fred", "en_US"), ("Ralph", "en_US"),    # legacy, last
    ]


def test_gender_and_accent_together():
    assert tts.pick_voice({"tts": {"gender": "male", "accent": "british"}}) == "Daniel"
    assert tts.pick_voice({"tts": {"gender": "female", "accent": "australian"}}) == "Karen"
    assert tts.pick_voice({"tts": {"gender": "female", "accent": "us"}}) == "Samantha"


def test_a_modern_voice_is_preferred_over_a_1990s_one():
    """Samantha and Fred are both American; only one of them should ever be
    handed to someone who just asked for an American voice."""
    assert tts.pick_voice({"tts": {"accent": "american"}}) == "Samantha"


def test_gender_wins_when_the_accent_is_not_installed():
    """There is no female British voice here. Handing back the male British one
    gives the opposite of what was asked; the accent is the softer half."""
    name, note = tts.choose_voice({"tts": {"gender": "female", "accent": "british"}})
    assert tts.voice_gender(name) == "female"
    assert "British" in note and name in note


def test_an_unmeetable_accent_says_so_rather_than_substituting():
    name, note = tts.choose_voice({"tts": {"accent": "klingon"}})
    assert name == ""
    assert "klingon" in note and "Australian" in note


def test_an_explicit_voice_overrides_gender_and_accent():
    """Someone who typed a name meant that name."""
    assert tts.pick_voice({"tts": {"voice": "Fred", "gender": "female",
                                   "accent": "irish"}}) == "Fred"


def test_no_character_asked_for_means_no_voice_forced():
    assert tts.choose_voice({"tts": {}}) == ("", "")


def test_depth_midpoint_leaves_the_voice_alone():
    """0.5 is the voice as its designers shipped it, not a number near it."""
    assert tts.depth_command({"tts": {"depth": 0.5}}) == ""
    assert tts.depth_command({"tts": {}}) == ""


def test_depth_runs_deep_to_high():
    deep = tts.depth_command({"tts": {"depth": 1.0}})
    high = tts.depth_command({"tts": {"depth": 0.0}})
    assert deep == "[[pbas 34]]" and high == "[[pbas 66]]"


def test_depth_is_clamped_and_survives_nonsense():
    assert tts.depth_command({"tts": {"depth": 9}}) == "[[pbas 34]]"
    assert tts.depth_command({"tts": {"depth": -9}}) == "[[pbas 66]]"
    assert tts.depth_command({"tts": {"depth": "loud"}}) == ""


def test_depth_reaches_the_command_with_a_voice_to_apply_it_to(spoken):
    """Measured: pbas is ignored unless a voice is named, so asking for depth
    without a voice must still name one or the knob does nothing."""
    tts.say("Hello there.", {"tts": {"enabled": True, "depth": 0.9}})
    cmd = spoken[0]
    assert "-v" in cmd, cmd
    assert cmd[-1].startswith("[[pbas ")


def test_the_depth_only_voice_follows_the_user_locale():
    """An en_AU machine should not get an alphabetically-first Indian voice."""
    assert tts.default_pitch_voice() == "Karen"


def test_gender_and_depth_reach_the_command_together(spoken):
    tts.say("Hello there.", {"tts": {"enabled": True, "gender": "male",
                                     "accent": "british", "depth": 0.1}})
    assert spoken[0][:3] == ["say", "-v", "Daniel"]
    assert spoken[0][-1].startswith("[[pbas ")


def test_no_expression_control_is_offered():
    """macOS accepts [[pmod]] and ignores it — verified by rendering to a file
    and comparing bytes. A knob that reads correct and changes nothing is the
    bug this project keeps finding, so it is absent on purpose."""
    source = (tts.__file__ or "")
    assert "pmod" in pathlib.Path(source).read_text()      # documented
    assert not hasattr(tts, "expression_command")          # but not offered


def test_an_unknown_voice_name_is_detectable():
    assert tts.voice_installed("daniel") is True
    assert tts.voice_installed("Nonexistent") is False


def test_the_system_rate_is_read_once_not_per_reply(monkeypatch):
    """It was shelling out to `defaults read` on every spoken reply, for a
    value that changes only when someone opens System Settings."""
    monkeypatch.setattr(tts, "_system_rate_cache", None)
    reads = []
    monkeypatch.setattr(tts, "_read_system_rate", lambda: reads.append(1) or 200)
    assert tts.system_rate() == 200
    assert tts.system_rate() == 200
    assert len(reads) == 1


def test_a_two_word_accent_is_one_accent():
    """'south african' is one accent spelled with a space; parsing it as two
    words left someone with an unknown-word complaint and no change."""
    changes, unknown = tts.parse_voice_words(["female", "south", "african"])
    assert changes == {"gender": "female", "accent": "south african"}
    assert unknown == []


def test_plain_words_map_to_settings():
    assert tts.parse_voice_words(["off"])[0] == {"enabled": False}
    assert tts.parse_voice_words(["deeper"], {"depth": 0.5})[0] == {"depth": 0.75}
    assert tts.parse_voice_words(["faster"], {"rate_slider": 0.5})[0] == {
        "rate_slider": 0.65, "rate": 0}


def test_an_unknown_word_is_returned_not_dropped():
    """Silently ignoring half of what someone typed is how a setting appears
    not to work."""
    changes, unknown = tts.parse_voice_words(["male", "klingon"])
    assert changes == {"gender": "male"} and unknown == ["klingon"]


def test_enhanced_voice_names_are_read_correctly(monkeypatch):
    """The voices this code tells people to install are listed as
    `Ava (Premium)  en_US`, so the second word is not the locale. Taking it as
    one would have hidden every downloaded voice — the exact ones someone
    installs after being told their accent is missing."""
    listing = ("Ava (Premium)            en_US    # Hello! My name is Ava.\n"
               "Daniel                   en_GB    # Hello! My name is Daniel.\n"
               "Daniel (Enhanced)        en_GB    # Hello! My name is Daniel.\n"
               "Flo (Premium)            fr_FR    # Bonjour!\n")

    class _Listing:
        stdout = listing
        returncode = 0

    monkeypatch.setattr(tts, "_voices_cache", None)
    monkeypatch.setattr(tts, "_all_voices_cache", None)
    monkeypatch.setattr(tts.subprocess, "run", lambda *_a, **_k: _Listing())
    # Ava is female and American; Daniel appears twice and is one voice.
    assert tts.installed_voices() == [("Ava", "en_US"), ("Daniel", "en_GB")]
    assert tts.pick_voice({"tts": {"gender": "female", "accent": "us"}}) == "Ava"


def _listing(monkeypatch, rows: str) -> None:
    class _Listing:
        stdout = rows
        returncode = 0

    monkeypatch.setattr(tts, "_voices_cache", None)
    monkeypatch.setattr(tts, "_all_voices_cache", None)
    monkeypatch.setattr(tts.subprocess, "run", lambda *_a, **_k: _Listing())


def test_a_missing_accent_falls_back_to_a_near_one_not_an_alphabetical_one(monkeypatch):
    """Asked for a male Australian voice with none installed, British is the
    right substitute and Indian is not. Before the neighbour table this picked
    whichever name sorted first, which was Aman."""
    _listing(monkeypatch,
             "Aman                en_IN    # Hello!\n"
             "Daniel              en_GB    # Hello!\n"
             "Karen               en_AU    # Hello!\n")
    name, note = tts.choose_voice({"tts": {"gender": "male",
                                           "accent": "australian"}})
    assert name == "Daniel"
    assert "No male Australian voice is installed" in note
    assert "British" in note


def test_no_accent_asked_for_follows_the_machine_region(monkeypatch):
    """\"male\" on an Australian Mac should not mean an Indian voice just
    because the name sorts first — and with no male Australian installed, the
    nearest region wins, silently, because no accent was ever requested."""
    _listing(monkeypatch,
             "Aman                en_IN    # Hello!\n"
             "Daniel              en_GB    # Hello!\n"
             "Karen               en_AU    # Hello!\n")
    monkeypatch.setattr(tts, "_system_locale", lambda: "en_AU")
    assert tts.choose_voice({"tts": {"gender": "female"}}) == ("Karen", "")
    assert tts.choose_voice({"tts": {"gender": "male"}}) == ("Daniel", "")


def test_a_locale_qualifier_is_part_of_the_name(monkeypatch):
    """`say -v Eddy` and `say -v "Eddy (English (US))"` are different audio, so
    the qualifier must survive into the command while the listing stays
    readable."""
    _listing(monkeypatch,
             "Eddy (English (UK)) en_GB    # Hello!\n"
             "Eddy (English (US)) en_US    # Hello!\n"
             "Eddy (French (France)) fr_FR # Bonjour!\n")
    assert tts.pick_voice({"tts": {"accent": "us"}}) == "Eddy (English (US))"
    assert tts.pick_voice({"tts": {"accent": "gb"}}) == "Eddy (English (UK))"
    assert tts.display_name("Eddy (English (US))") == "Eddy"


def test_a_dedicated_voice_is_preferred_over_the_multi_locale_family(monkeypatch):
    """Samantha is a voice for en_US; Flo is one name shared across a dozen
    locales. Someone who said only "American female" gets the dedicated one."""
    _listing(monkeypatch,
             "Flo (English (US))  en_US    # Hello!\n"
             "Samantha            en_US    # Hello!\n")
    assert tts.pick_voice({"tts": {"gender": "female", "accent": "us"}}) == "Samantha"


def test_a_short_name_typed_at_the_prompt_resolves_to_a_real_one(monkeypatch):
    _listing(monkeypatch, "Eddy (English (UK)) en_GB    # Hello!\n")
    changes, unknown = tts.parse_voice_words(["eddy"])
    assert changes["voice"] == "Eddy (English (UK))" and unknown == []
    assert tts.voice_installed("eddy") and tts.voice_installed("Eddy (English (UK))")
