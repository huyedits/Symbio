"""Speaking the reply out loud, on the terminal only.

macOS ships `say`, so this needs nothing installed. That matters more than it
sounds: a voice feature behind a pip install is one that breaks the session
when the install rots, and `say` is in /usr/bin on every Mac.

THREE RULES, all of them learned from the rest of this project rather than
from anything about speech:

  It never blocks a turn. The reply is already printed by the time this runs;
  making someone wait for audio to finish before they can type is the friction
  a voice is supposed to remove. It is a detached process nobody waits on.

  It speaks the ANSWER, not the transcript. Status lines, tool tags, reasoning
  blocks, code fences and URLs are for reading. A voice reading
  "[Tool: browser_open]" aloud, or a forty-line code block character by
  character, is worse than silence — and that is what a naive hook does,
  because output_fn carries all of it.

  One voice at a time. A second reply starting while the first is still
  speaking produces two voices over each other; the new one interrupts the old,
  the way a person would.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from typing import Any

# Everything that is screen furniture rather than speech. Bracket tags are the
# project's own protocol ("[Tool: ...]", "[Observation] ...", "[Reasoning] ..."),
# and they are read, not heard.
_TAG_LINE = re.compile(r"^\s*\[[^\]]+\]\s*", re.MULTILINE)
_FENCE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE = re.compile(r"`[^`]+`")
_URL = re.compile(r"https?://\S+")
_MARKDOWN = re.compile(r"[*_#>|]+")
_WHITESPACE = re.compile(r"\s{2,}")
# Punctuation left hanging where code or a link used to be. "I fixed it in :
# code block omitted" is what a naive strip produces, and a voice reads that
# colon as a pause in the wrong place.
_ORPHAN_PUNCT = re.compile(r"\s+([:;,])(?=\s|$)")

# A process handle, module level because there is one pair of speakers.
_speaking: subprocess.Popen | None = None


# ----------------------------------------------------------- voice character
#
# What `say` can and cannot actually do, measured on this machine by rendering
# to a file and comparing bytes rather than by listening:
#
#   pitch   [[pbas N]]  WORKS, and changes the audio — but only when a voice is
#                       named. With no -v the default voice ignores it entirely
#                       (verified: byte-identical output at pbas 30 and 70).
#   expression
#           [[pmod N]]  DOES NOTHING. Byte-identical at pmod 0 and 90, on a
#                       legacy MacinTalk voice as well as a modern one. So it
#                       is not offered: a knob that reads correct and changes
#                       nothing is the exact bug this project keeps finding.
#   gender
#   accent              Not parameters at all — they are which voice you pick.
#
# `say -v '?'` gives names and locales but NOT gender, so gender is a curated
# table. It is keyed on name, and names repeat across locales — this Mac has a
# French "Flo" and a Finnish "Eddy" but no English one — so the locale filter
# below is what makes the table mean anything.
_VOICE_GENDER = {
    "albert": "male", "alex": "male", "aman": "male", "bruce": "male",
    "daniel": "male", "eddy": "male", "fred": "male", "gordon": "male",
    "grandpa": "male", "junior": "male", "lee": "male", "oliver": "male",
    "ralph": "male", "reed": "male", "rishi": "male", "rocko": "male",
    "tom": "male",
    "agnes": "female", "allison": "female", "ava": "female",
    "catherine": "female", "fiona": "female", "flo": "female",
    "grandma": "female", "karen": "female", "kate": "female",
    "kathy": "female", "martha": "female", "moira": "female",
    "nicky": "female", "samantha": "female", "sandy": "female",
    "serena": "female", "shelley": "female", "susan": "female",
    "tara": "female", "tessa": "female", "veena": "female",
    "victoria": "female", "vicki": "female", "zoe": "female",
}

# The MacinTalk voices from the 1990s. Real speech, so they stay eligible, but
# they sound synthetic next to the modern ones and should never be the voice
# picked on a user's behalf while a modern one fits the request.
_LEGACY = {"agnes", "albert", "bruce", "fred", "junior", "kathy", "ralph",
           "vicki", "victoria"}

# Accent as someone would ask for it, mapped to the locale `say` reports.
_ACCENTS = {
    "us": "en_US", "american": "en_US",
    "gb": "en_GB", "uk": "en_GB", "british": "en_GB", "english": "en_GB",
    "au": "en_AU", "australian": "en_AU", "aussie": "en_AU",
    "ie": "en_IE", "irish": "en_IE",
    "in": "en_IN", "indian": "en_IN",
    "za": "en_ZA", "south african": "en_ZA",
}
_ACCENT_NAMES = {
    "en_US": "American", "en_GB": "British", "en_AU": "Australian",
    "en_IE": "Irish", "en_IN": "Indian", "en_ZA": "South African",
}

# pbas around a voice's own pitch. Deliberately narrow: the legacy range is
# 0-127 and its far ends do not sound like a deeper person, they sound like a
# broken recording. The midpoint emits no pitch command at all, so "middle" is
# the voice as its designers shipped it rather than a number that lands near it.
_DEPTH_LOW = 34
_DEPTH_MID = 50
_DEPTH_HIGH = 66

_voices_cache: list[tuple[str, str]] | None = None


def installed_voices() -> list[tuple[str, str]]:
    """[(name, locale)] for the English speech voices actually present.

    Novelty voices (Bells, Boing, Trinoids, Zarvox …) are installed on every
    Mac and are excluded: nobody wants their assistant to sound like an organ.
    """
    global _voices_cache
    if _voices_cache is not None:
        return _voices_cache
    found: list[tuple[str, str]] = []
    if not available():
        _voices_cache = found
        return found
    try:
        out = subprocess.run(["say", "-v", "?"], capture_output=True,
                             text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        _voices_cache = found
        return found
    for line in out.stdout.splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        name, locale = parts[0], parts[1]
        if not locale.startswith("en_"):
            continue
        if name.lower() not in _VOICE_GENDER:
            continue
        found.append((name, locale))
    # Modern voices first, so every "first match" below is the best match.
    found.sort(key=lambda v: (v[0].lower() in _LEGACY, v[0]))
    _voices_cache = found
    return found


def voice_installed(name: str) -> bool:
    want = name.strip().lower()
    return any(n.lower() == want for n, _ in installed_voices())


def voice_gender(name: str) -> str:
    return _VOICE_GENDER.get(name.strip().lower(), "")


def accents_available() -> list[str]:
    """Accent words the installed voices can actually deliver."""
    locales = {loc for _, loc in installed_voices()}
    return sorted(_ACCENT_NAMES.get(loc, loc) for loc in locales)


def choose_voice(config: dict[str, Any] | None) -> tuple[str, str]:
    """(voice name, note). The note is non-empty only when the request could
    not be met exactly, and says what was traded away — so a compromise is
    something the user gets told about rather than something they wonder at."""
    cfg = (config or {}).get("tts", {})
    explicit = str(cfg.get("voice", "") or "").strip()
    if explicit:
        return explicit, ""
    want_gender = str(cfg.get("gender", "") or "").strip().lower()
    if want_gender not in ("male", "female"):
        want_gender = ""
    raw_accent = str(cfg.get("accent", "") or "").strip().lower()
    want_accent = _ACCENTS.get(raw_accent)
    if raw_accent and not want_accent:
        return "", (f"No accent called {raw_accent!r}; try one of: "
                    + ", ".join(accents_available()))
    if not (want_gender or want_accent):
        return "", ""

    voices = installed_voices()
    exact = [n for n, loc in voices
             if (not want_gender or voice_gender(n) == want_gender)
             and (not want_accent or loc == want_accent)]
    if exact:
        return exact[0], ""

    # Gender wins over accent. A female voice with the wrong accent is closer
    # to "a female British voice" than a male British one is, and picking the
    # male voice silently is how you hand someone the opposite of their ask.
    if want_gender:
        same_gender = [n for n, _ in voices if voice_gender(n) == want_gender]
        if same_gender:
            accent_name = _ACCENT_NAMES.get(want_accent or "", raw_accent)
            return same_gender[0], (
                f"No {want_gender} {accent_name} voice is installed — using "
                f"{same_gender[0]} ({_ACCENT_NAMES.get(dict(voices)[same_gender[0]], '')}). "
                "Install more in System Settings > Accessibility > Spoken Content.")
    if want_accent:
        same_accent = [n for n, loc in voices if loc == want_accent]
        if same_accent:
            return same_accent[0], (
                f"No {want_gender} voice with that accent is installed — using "
                f"{same_accent[0]} ({voice_gender(same_accent[0])}).")
    return "", "No matching voice is installed; using the system voice."


def pick_voice(config: dict[str, Any] | None) -> str:
    return choose_voice(config)[0]


def default_pitch_voice() -> str:
    """A voice to name when depth is asked for but no voice was chosen.

    Pitch is ignored without an explicit -v (measured: byte-identical output),
    so honouring depth means naming something — and what it names should follow
    the user's own locale. macOS exposes no "selected voice" to read (the
    com.apple.speech.voice.prefs domain does not exist until something writes
    it), so the region from AppleLocale is the closest real signal; picking the
    alphabetically first voice instead gave a UK user an Indian male voice.
    """
    voices = installed_voices()
    if not voices:
        return ""
    locale = _system_locale()
    for want in (locale, "en_US"):
        for name, voice_locale in voices:
            if want and voice_locale == want:
                return name
    return voices[0][0]


def _system_locale() -> str:
    """The user's region as a `say` locale (en_GB …), or "" if unreadable."""
    try:
        out = subprocess.run(["defaults", "read", "-g", "AppleLocale"],
                             capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return ""
    raw = out.stdout.strip().replace("-", "_")
    # en_US, or en_US@rg=gbzzzz when the region is overridden in Settings.
    region = ""
    if "@rg=" in raw:
        region = raw.split("@rg=", 1)[1][:2].upper()
    parts = raw.split("@", 1)[0].split("_")
    if not region and len(parts) >= 2:
        region = parts[1][:2].upper()
    return f"en_{region}" if region else ""


DEPTH_STEP = 0.25
RATE_STEP = 0.15


def parse_voice_words(words: list[str],
                      current: dict[str, Any] | None = None,
                      ) -> tuple[dict[str, Any], list[str]]:
    """Turn `/voice female british deeper` into config changes.

    Plain words rather than flags, because this is typed at a chat prompt where
    `--gender female --accent british` is friction for no gain. Returns
    (changes, unrecognised words) — unknown words are reported rather than
    ignored, since silently dropping half of what someone typed is how a
    setting appears not to work.
    """
    now = dict(current or {})
    changes: dict[str, Any] = {}
    unknown: list[str] = []

    def depth_now() -> float:
        try:
            return float(changes.get("depth", now.get("depth", 0.5)))
        except (TypeError, ValueError):
            return 0.5

    def rate_now() -> float:
        try:
            return float(changes.get("rate_slider", now.get("rate_slider", 0.5)))
        except (TypeError, ValueError):
            return 0.5

    index = 0
    while index < len(words):
        word = words[index]
        index += 1
        token = word.strip().lower().rstrip(",")
        if not token:
            continue
        # "south african" is one accent spelled with a space, so a pair is
        # tried before the single word that starts it.
        if index < len(words):
            pair = f"{token} {words[index].strip().lower().rstrip(',')}"
            if pair in _ACCENTS:
                changes["accent"] = pair
                index += 1
                continue
        if token in ("on", "enable", "enabled"):
            changes["enabled"] = True
        elif token in ("off", "disable", "disabled", "mute", "quiet"):
            changes["enabled"] = False
        elif token in ("female", "woman", "f"):
            changes["gender"] = "female"
        elif token in ("male", "man", "m"):
            changes["gender"] = "male"
        elif token in _ACCENTS:
            changes["accent"] = token
        elif token in ("deep", "deeper", "lower"):
            changes["depth"] = min(1.0, depth_now() + DEPTH_STEP)
        elif token in ("high", "higher", "lighter"):
            changes["depth"] = max(0.0, depth_now() - DEPTH_STEP)
        elif token in ("faster", "quicker"):
            changes["rate_slider"] = min(1.0, rate_now() + RATE_STEP)
            changes["rate"] = 0
        elif token in ("slower",):
            changes["rate_slider"] = max(0.0, rate_now() - RATE_STEP)
            changes["rate"] = 0
        elif token in ("normal", "default", "reset"):
            changes.update({"depth": 0.5, "rate_slider": 0.5, "rate": 0,
                            "voice": "", "gender": "", "accent": ""})
        elif voice_installed(token):
            # An explicit name overrides gender and accent, so clear them
            # rather than leave a contradiction in the config.
            changes.update({"voice": _installed_name(token),
                            "gender": "", "accent": ""})
        else:
            unknown.append(word)
    return changes, unknown


def _installed_name(lowered: str) -> str:
    """The installed voice's own capitalisation, given any casing."""
    for name, _ in installed_voices():
        if name.lower() == lowered:
            return name
    return lowered


def depth_command(config: dict[str, Any] | None) -> str:
    """An inline pitch command, or "" to leave the voice at its own pitch."""
    raw = (config or {}).get("tts", {}).get("depth")
    if raw in (None, ""):
        return ""
    try:
        depth = max(0.0, min(1.0, float(raw)))
    except (TypeError, ValueError):
        return ""
    if abs(depth - 0.5) < 0.02:
        return ""                         # the voice as shipped
    if depth < 0.5:
        pbas = round(_DEPTH_HIGH - (depth / 0.5) * (_DEPTH_HIGH - _DEPTH_MID))
    else:
        pbas = round(_DEPTH_MID - ((depth - 0.5) / 0.5) * (_DEPTH_MID - _DEPTH_LOW))
    return f"[[pbas {pbas}]]"


# Apple's Speaking Rate slider, in the terms `say -r` understands.
#
# macOS exposes one rate control (Accessibility > Spoken Content > Speaking
# Rate) and stores nothing until you move it, so there is no system value to
# read on a fresh machine — checked, the preference domains are empty. The
# slider is therefore reproduced rather than followed: 0.0 to 1.0, with the
# MIDPOINT on the system default of 175 wpm so a slider at half speed sounds
# like macOS does out of the box. Two segments rather than one straight line
# for exactly that reason; a single 100-to-400 ramp would put the default at
# 0.25 and make the useful half of the travel feel lopsided.
SLOWEST_WPM = 100
DEFAULT_WPM = 175
FASTEST_WPM = 400

# Where macOS keeps the rate once someone has moved the slider.
_SYSTEM_RATE_DOMAIN = "com.apple.speech.voice.prefs"
_SYSTEM_RATE_KEY = "SpeechRate"


_system_rate_cache: tuple[int | None] | None = None


def system_rate() -> int | None:
    """The rate from Apple's own slider, or None if it was never moved.

    Cached: this was spawning a `defaults read` for every single spoken reply,
    to read a value that only changes when someone opens System Settings.
    """
    global _system_rate_cache
    if _system_rate_cache is not None:
        return _system_rate_cache[0]
    _system_rate_cache = (_read_system_rate(),)
    return _system_rate_cache[0]


def _read_system_rate() -> int | None:
    try:
        out = subprocess.run(
            ["defaults", "read", _SYSTEM_RATE_DOMAIN, _SYSTEM_RATE_KEY],
            capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    try:
        value = int(float(out.stdout.strip()))
    except ValueError:
        return None
    return value if SLOWEST_WPM // 2 <= value <= FASTEST_WPM * 2 else None


def slider_to_wpm(slider: float) -> int:
    """A 0.0-1.0 slider position as words per minute."""
    slider = max(0.0, min(1.0, float(slider)))
    if slider <= 0.5:
        return round(SLOWEST_WPM + (slider / 0.5) * (DEFAULT_WPM - SLOWEST_WPM))
    return round(DEFAULT_WPM + ((slider - 0.5) / 0.5) * (FASTEST_WPM - DEFAULT_WPM))


def resolved_rate(config: dict[str, Any] | None) -> int | None:
    """Words per minute to use, or None to leave `say` on its own default.

    Precedence, most explicit first: a raw wpm in tts.rate, then the slider,
    then whatever Apple's own slider was set to. Raw wins because someone who
    typed 300 meant 300, and silently reinterpreting that through a 0-1 scale
    would be the kind of helpfulness nobody asked for.
    """
    cfg = (config or {}).get("tts", {})
    raw = cfg.get("rate")
    if raw:
        try:
            return int(raw)
        except (TypeError, ValueError):
            pass
    slider = cfg.get("rate_slider")
    if slider not in (None, ""):
        try:
            return slider_to_wpm(float(slider))
        except (TypeError, ValueError):
            pass
    return system_rate()


def available() -> bool:
    return shutil.which("say") is not None


def enabled(config: dict[str, Any] | None) -> bool:
    """On only when asked for, and only with someone there to hear it.

    stdout not being a terminal means piped, redirected, or driven by the
    transcript harness — all cases where audio is noise in someone's
    background rather than output. SYMBIO_NO_TTS switches it off for one run
    without editing config, the same escape hatch SYMBIO_NO_COLOR gives the
    terminal skin.
    """
    if os.environ.get("SYMBIO_NO_TTS"):
        return False
    if not (config or {}).get("tts", {}).get("enabled", False):
        return False
    if not sys.stdout.isatty():
        return False
    return available()


def why_silent(config: dict[str, Any] | None) -> str:
    """Why nothing would be spoken, or "" if it would be. Exists so a preview
    command can say "you are not on a terminal" instead of the useless
    "nothing was spoken"."""
    if not available():
        print_reason = "macOS `say` is not on this machine"
        return print_reason
    if os.environ.get("SYMBIO_NO_TTS"):
        return "SYMBIO_NO_TTS is set for this run"
    if not sys.stdout.isatty():
        return "output is piped or redirected, not a terminal"
    if not (config or {}).get("tts", {}).get("enabled", False):
        return "tts.enabled is false"
    return ""


def speakable(text: str, limit: int = 600) -> str:
    """The part of a reply worth hearing, or "".

    Order matters: fenced blocks go before inline code, or the fence markers
    survive as stray backticks in the middle of a sentence.
    """
    if not text or not isinstance(text, str):
        return ""
    out = _FENCE.sub(" code block omitted. ", text)
    out = _TAG_LINE.sub("", out)
    out = _INLINE_CODE.sub(" ", out)
    out = _URL.sub(" a link ", out)
    out = _MARKDOWN.sub(" ", out)
    out = _ORPHAN_PUNCT.sub("", out)
    out = _WHITESPACE.sub(" ", out).strip()
    if not out:
        return ""
    if len(out) > limit:
        # Cut on a sentence if there is one nearby, so it does not stop
        # mid-word. A voice that trails off sounds like a crash.
        window = out[:limit]
        cut = max(window.rfind(". "), window.rfind("? "), window.rfind("! "))
        out = window[:cut + 1] if cut > limit // 3 else window.rstrip() + "…"
    return out


def stop() -> None:
    """Silence whatever is speaking. Safe to call when nothing is."""
    global _speaking
    if _speaking is not None and _speaking.poll() is None:
        try:
            _speaking.terminate()
        except OSError:
            pass
    _speaking = None


def say(text: str, config: dict[str, Any] | None = None) -> bool:
    """Speak `text`, interrupting anything already speaking. Never blocks.

    Returns whether anything was started, which is what the tests assert on —
    a voice feature that silently does nothing is indistinguishable from one
    that is off.
    """
    global _speaking
    if not enabled(config):
        return False
    line = speakable(text, int((config or {}).get("tts", {}).get("max_chars", 600)))
    if not line:
        return False
    stop()
    cmd = ["say"]
    voice = pick_voice(config)
    depth = depth_command(config)
    if depth and not voice:
        # pbas is ignored without an explicit voice — measured, byte-identical
        # output. Asking for depth and getting none would be a silent no-op, so
        # name a voice that honours it.
        voice = default_pitch_voice()
    if voice:
        cmd += ["-v", voice]
    if depth:
        line = f"{depth} {line}"
    rate = resolved_rate(config)
    if rate:
        cmd += ["-r", str(rate)]
    try:
        # Detached and unwaited: the reply is already on screen, and nobody
        # should be held at the prompt until the audio finishes.
        _speaking = subprocess.Popen(cmd + [line],
                                     stdout=subprocess.DEVNULL,
                                     stderr=subprocess.DEVNULL)
    except OSError:
        _speaking = None
        return False
    return True
