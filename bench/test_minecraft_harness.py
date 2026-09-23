"""Tests for the Minecraft harness's unattended-run machinery.

None of these needs the game. They cover what stopped the first unattended
attempt on 2026-09-23 — a JVM crash nobody noticed and a locked screen that
turned every capture black — plus the F3 reader every later rung is graded
by, and the guard that keeps injected input out of other apps.
"""
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import minecraft_harness as mh  # noqa: E402


# ---- preflight: who has to fix what ----

def _probe(**overrides):
    base = dict(locked=lambda: False, screen_recording=lambda: True,
                capture_blank=lambda: False, accessibility=lambda: True,
                launcher=lambda: True, display_sleep=lambda: 0,
                caffeinated=lambda: False, game_pid=lambda: 4242,
                rivals=lambda: [])
    base.update(overrides)
    return base


def test_a_ready_machine_is_autonomous():
    checks = mh.preflight(_probe())

    assert all(c.ok for c in checks)
    assert mh.autonomous(checks)


def test_a_locked_screen_needs_a_person():
    """The state the real run ended in: locked, every capture black."""
    checks = mh.preflight(_probe(locked=lambda: True))

    assert not mh.autonomous(checks)
    failing = [c for c in checks if not c.ok]
    assert [c.name for c in failing] == ["session unlocked"]
    assert failing[0].who == "human"


def test_a_locked_screen_is_not_misreported_as_a_blank_capture():
    """Locked, the capture IS blank — but the fix is unlocking, not a
    permission. The blank-capture check only runs when unlocked."""
    names = [c.name for c in mh.preflight(_probe(
        locked=lambda: True, capture_blank=lambda: True))]

    assert "capture shows the screen" not in names


def test_a_blank_capture_while_unlocked_blocks():
    checks = mh.preflight(_probe(capture_blank=lambda: True))

    assert not mh.autonomous(checks)


def test_missing_permissions_need_a_person():
    for probe in (dict(screen_recording=lambda: False),
                  dict(accessibility=lambda: False)):
        assert not mh.autonomous(mh.preflight(_probe(**probe)))


def test_what_the_harness_can_fix_itself_does_not_block():
    checks = mh.preflight(_probe(display_sleep=lambda: 20, game_pid=lambda: None,
                                 rivals=lambda: ["RobloxPlayer"]))

    assert mh.autonomous(checks)
    assert {c.name for c in checks if not c.ok} == {
        "display stays awake", "game running", "no rival for the input"}


def test_a_display_held_awake_is_safe_whatever_the_sleep_setting():
    checks = mh.preflight(_probe(display_sleep=lambda: 20, caffeinated=lambda: True))

    assert next(c for c in checks if c.name == "display stays awake").ok


def test_display_sleep_is_read_from_pmset():
    out = " sleep                0 (sleep prevented by caffeinate)\n displaysleep         20\n"

    assert mh.display_sleep_minutes(out) == 20
    assert mh.display_sleep_minutes("nothing here") == 0


# ---- finding the game, not the launcher ----

_PS = [
    "32749 /Applications/Feather Launcher.app/Contents/MacOS/Feather Launcher",
    "32771 /Applications/Feather Launcher.app/Contents/Frameworks/Feather Launcher "
    "Helper (Renderer).app/Contents/MacOS/Feather Launcher Helper (Renderer) "
    "--minecraft-dir=/Users/x/Library/Application Support/minecraft",
    "39356 /Applications/Roblox.app/Contents/MacOS/RobloxPlayer",
]
_JAVA = ("34478 /Users/x/Library/Application Support/minecraft/jre/bin/java "
         "-Xmx4G -cp ... net.fabricmc.loader.impl.launch.knot.KnotClient "
         "--gameDir /Users/x/Library/Application Support/minecraft")


def test_the_launchers_helpers_are_not_the_game():
    assert mh.game_pid(_PS) is None


def test_the_game_jvm_is_found():
    assert mh.game_pid(_PS + [_JAVA]) == 34478


def test_rivals_are_named():
    assert mh.rivals(_PS) == ["RobloxPlayer"]


# ---- noticing a crash ----

def test_a_jit_crash_with_no_crash_report_is_detected(tmp_path):
    """The real 10:59 crash left only replay_pid34478.log, nothing in
    crash-reports/, and a latest.log that just stopped."""
    since = time.time() - 5
    (tmp_path / "replay_pid34478.log").write_text("version 2\n")

    why = mh.detect_crash(since, tmp_path)

    assert why and "JIT" in why and "replay_pid34478" in why


def test_an_old_crash_is_not_this_launches_crash(tmp_path):
    old = tmp_path / "hs_err_pid1.log"
    old.write_text("x")
    import os
    os.utime(old, (time.time() - 3600, time.time() - 3600))

    assert mh.detect_crash(time.time() - 5, tmp_path) is None


def test_a_game_crash_report_is_detected(tmp_path):
    (tmp_path / "crash-reports").mkdir()
    (tmp_path / "crash-reports" / "crash-2026-09-23_10.59.57-client.txt").write_text("x")

    assert "game crash report" in mh.detect_crash(time.time() - 5, tmp_path)


def _watch(pids, crashes, windows, timeout=10):
    pids, crashes, windows = iter(pids), iter(crashes), iter(windows)
    return mh.watch_launch(
        since=0, timeout=timeout, poll=1,
        pid_fn=lambda: next(pids, None),
        crash_fn=lambda since: next(crashes, None),
        window_fn=lambda pid: next(windows, False),
        sleep=lambda s: None)


def test_launch_ready_when_the_window_opens():
    assert _watch([None, 7, 7], [None] * 3, [False, True]) == ("ready", "7")


def test_launch_crash_is_reported_from_the_crash_file():
    state, why = _watch([None, 7, 7], [None, None, "JVM JIT compiler crash: replay_pid7.log"],
                        [False, False])

    assert state == "crashed" and "JIT" in why


def test_a_process_that_vanishes_is_a_crash_even_with_no_file():
    state, why = _watch([7, None], [None, None], [False])

    assert state == "crashed" and "pid 7 exited" in why


def test_a_launch_that_never_starts_times_out_and_says_so():
    state, why = _watch([], [], [], timeout=3)

    assert state == "timeout" and "never started" in why


# ---- reading F3 ----

def test_f3_lines_as_the_game_draws_them():
    got = mh.parse_f3([
        "Minecraft 26.1 (26.1/fabric)",
        "XYZ: 12.345 / 64.00000 / -8.500",
        "Block: 12 64 -9 [12 0 7]",
        "Facing: north (Towards negative Z) (-179.9 / 12.3)",
    ])

    assert got["xyz"] == (12.345, 64.0, -8.5)
    assert got["block"] == (12, 64, -9)
    assert got["facing"] == "north"
    assert (got["yaw"], got["pitch"]) == (-179.9, 12.3)


def test_f3_lines_as_ocr_misreads_them():
    """OCR reads the slash as | or I, and a colon as a semicolon."""
    got = mh.parse_f3([
        "XYZ; 12.345 | 64.00000 I -8.500",
        "Facing: east (Towards positive X) (-90.0 l 0.0)",
    ])

    assert got["xyz"] == (12.345, 64.0, -8.5)
    assert (got["yaw"], got["pitch"]) == (-90.0, 0.0)


def test_a_missing_overlay_reads_as_missing_not_zero():
    assert mh.parse_f3(["Singleplayer", "Multiplayer"]) == {}


def test_yaw_turn_crosses_the_seam():
    assert mh.turned(179.0, -179.0) == pytest.approx(2.0)
    assert mh.turned(-179.0, 179.0) == pytest.approx(-2.0)


@pytest.mark.skipif(sys.platform != "darwin", reason="Apple Vision OCR")
def test_f3_read_off_a_rendered_overlay(tmp_path):
    """Through the real OCR: grey text on a dark sky, the way F3 looks."""
    from PIL import Image, ImageDraw, ImageFont

    image = Image.new("RGB", (1400, 260), (40, 60, 90))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=30)
    for i, line in enumerate(["XYZ: 12.345 / 64.00000 / -8.500",
                              "Block: 12 64 -9",
                              "Facing: north (Towards negative Z) (-179.9 / 12.3)"]):
        draw.text((20, 20 + i * 70), line, fill=(224, 224, 224), font=font)
    path = tmp_path / "f3.png"
    image.save(path)

    got = mh.parse_f3(mh.read_lines(path))

    assert got.get("xyz") == (12.345, 64.0, -8.5), mh.read_lines(path)
    assert (got.get("yaw"), got.get("pitch")) == (-179.9, 12.3), mh.read_lines(path)


# ---- input goes only to the game ----

def test_input_is_refused_when_the_game_is_not_frontmost(monkeypatch):
    sent = []
    monkeypatch.setattr(mh, "post", sent.append)

    with pytest.raises(mh.NotFrontmost):
        mh.post_to_game("event", pid=7, front=lambda: 39356)
    assert sent == []

    mh.post_to_game("event", pid=7, front=lambda: 7)
    assert sent == ["event"]


def test_a_held_key_is_released_even_when_focus_is_lost(monkeypatch):
    posted = []
    monkeypatch.setattr(mh, "post", lambda e: posted.append(e))
    monkeypatch.setattr(mh, "key_event", lambda key, down: (key, down))
    monkeypatch.setattr(mh.time, "sleep", lambda s: None)

    mh.hold_key("w", 0.5, pid=7, front=lambda: 7)
    assert posted == [("w", True), ("w", False)]

    # Focus moves to another app during the hold: the key-up still goes out.
    fronts = iter([7, 99])
    posted.clear()
    mh.hold_key("w", 0.5, pid=7, front=lambda: next(fronts))
    assert posted == [("w", True), ("w", False)]


def test_a_refused_key_down_sends_nothing(monkeypatch):
    posted = []
    monkeypatch.setattr(mh, "post", lambda e: posted.append(e))
    monkeypatch.setattr(mh, "key_event", lambda key, down: (key, down))

    with pytest.raises(mh.NotFrontmost):
        mh.hold_key("w", 0.5, pid=7, front=lambda: 99)
    assert posted == []


@pytest.mark.skipif(sys.platform != "darwin", reason="Quartz")
def test_the_mouse_event_carries_the_delta_the_camera_reads():
    import Quartz

    event = mh.mouse_delta_event(40, -12, at=(500.0, 400.0))

    assert Quartz.CGEventGetIntegerValueField(event, Quartz.kCGMouseEventDeltaX) == 40
    assert Quartz.CGEventGetIntegerValueField(event, Quartz.kCGMouseEventDeltaY) == -12
    loc = Quartz.CGEventGetLocation(event)
    assert (loc.x, loc.y) == (500.0, 400.0), "delta only: the position must not move"
