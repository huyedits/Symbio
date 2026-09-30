"""Symbio's own screen: the desktop tools work on the desk and never on the user's.

Desk mode (symbio/desk.py) gives the harness a virtual display of its own.
The promise it makes the user is narrow and absolute: with it on, nothing the
desktop tools do reaches the user's screen, pointer or keyboard -- unless the
user has left the Mac alone, and then only for one action, handed straight
back. These tests hold the harness to that with a stubbed desk and a stubbed
accessibility tree: nothing here creates a display or touches the real screen.
"""
import json
import shutil
import subprocess
import sys

import pytest

from symbio import ax, computer, desk, safety
from symbio.app import chat_tools


# ---------- geometry ----------

def test_a_corner_is_not_an_edge():
    main = (0, 0, 1920, 1080)
    assert desk.shared_edge((-1440, 1080, 1440, 900), main) == 0
    assert desk.shared_edge((-1440, 0, 1440, 900), main) == 900
    assert desk.shared_edge((-1439, 1080, 1440, 900), main) == 1
    assert desk.shared_edge((1920, 500, 1440, 900), main) == 580


def test_the_desk_goes_diagonally_off_the_bottom_left_of_the_arrangement():
    assert desk.plan_corner((1440, 900), [(0, 0, 1920, 1080)]) == (-1440, 1080)
    # A second screen below-right does not move the corner off the main one.
    others = [(0, 0, 1920, 1080), (1920, 200, 2560, 1440)]
    assert desk.plan_corner((1440, 900), others) == (-1440, 1640)


def test_points_convert_between_a_desk_capture_and_the_window_server():
    d = desk.Desk(display=8, pid=1, serial=1, x=-1440, y=1080, width=1440, height=900)
    assert d.to_global(100, 50) == (-1340, 1130)
    assert d.to_global(200, 100, scale=2.0) == (-1340, 1130)
    assert d.to_local(-1340, 1130) == (100, 50)
    assert d.contains(-1440, 1080) and not d.contains(0, 0)


@pytest.mark.parametrize("combo, want", [
    ("cmd+s", (1, 1 << 20, "s", ["cmd"])),
    ("Cmd+Shift+S", (1, (1 << 20) | (1 << 17), "s", ["cmd", "shift"])),
    ("enter", (36, 0, "enter", [])),
    ("ArrowDown", (125, 0, "down", [])),
    ("ctrl-a", (0, 1 << 18, "a", ["ctrl"])),
    ("cmd+-", (27, 1 << 20, "-", ["cmd"])),
])
def test_chords_parse(combo, want):
    assert desk.parse_chord(combo) == want


def test_an_unknown_key_is_refused_rather_than_guessed():
    assert desk.parse_chord("hyper+x") is None
    assert desk.parse_chord("cmd+banana") is None
    assert desk.parse_chord("") is None


# ---------- which display is the desk ----------

@pytest.fixture
def state_file(tmp_path, monkeypatch):
    path = tmp_path / "desk.json"
    monkeypatch.setattr(desk.constants, "DESK_STATE_FILE", path)
    return path


def _desk_display(monkeypatch, online=(8,), desks=(8,), bounds=(-1440, 1080, 1440, 900)):
    monkeypatch.setattr(desk, "_cg", object())
    monkeypatch.setattr(desk, "_display_ids", lambda active=False: list(online))
    monkeypatch.setattr(desk, "_is_desk_display", lambda d: d in desks)
    monkeypatch.setattr(desk, "_bounds", lambda d: bounds)


def test_only_the_display_the_running_helper_named_is_the_desk(state_file, monkeypatch):
    _desk_display(monkeypatch)
    state_file.write_text(json.dumps({"ok": True, "pid": 4242, "display": 8, "serial": 3}))
    monkeypatch.setattr(desk, "_helper_alive", lambda pid: pid == 4242)
    found = desk.current()
    assert (found.display, found.pid, found.serial) == (8, 4242, 3)
    assert found.rect == (-1440, 1080, 1440, 900)


def test_a_desk_whose_helper_died_is_a_leftover_and_never_used(state_file, monkeypatch):
    """A stopped desk lingers while the Mac's panel sleeps, then vanishes on
    wake -- taking any window on it back to the user's screen."""
    _desk_display(monkeypatch)
    state_file.write_text(json.dumps({"ok": True, "pid": 4242, "display": 8}))
    monkeypatch.setattr(desk, "_helper_alive", lambda pid: False)
    assert desk.current() is None


def test_a_display_that_is_not_a_desk_is_never_taken_for_one(state_file, monkeypatch):
    _desk_display(monkeypatch, online=(1, 8), desks=())
    state_file.write_text(json.dumps({"ok": True, "pid": 4242, "display": 1}))
    monkeypatch.setattr(desk, "_helper_alive", lambda pid: True)
    assert desk.current() is None


def test_desk_mode_reads_the_config():
    assert not desk.enabled({})
    assert not desk.enabled({"desk": {"enabled": False}})
    assert desk.enabled({"desk": {"enabled": True}})


def test_the_model_cannot_switch_the_desk_off():
    assert safety.is_sensitive_config_key("desk.enabled")
    assert safety.is_sensitive_config_key("desk.borrow_input_after_idle_s")


def test_the_browser_stays_where_it_was_with_desk_mode_off():
    assert desk.browser_args({}) == []
    assert desk.browser_args({"desk": {"enabled": False}}) == []


def test_the_browser_window_goes_on_the_desk(monkeypatch):
    running = desk.Desk(display=8, pid=1, serial=1, x=-1440, y=1080, width=1440, height=900)
    monkeypatch.setattr(desk, "ensure", lambda config: running)
    args = desk.browser_args({"desk": {"enabled": True}})
    assert args == [f"--window-position=-1440,{1080 + desk.MENU_BAR}",
                    f"--window-size=1440,{900 - desk.MENU_BAR}"]


def test_a_browser_session_launches_as_before_without_a_desk():
    session = computer.BrowserSession()
    assert session._desk_args() == []
    session.desk_window = lambda: ["--window-position=1,2"]
    assert session._desk_args() == ["--window-position=1,2"]

    def broken():
        raise desk.DeskError("no desk")

    # The desk failing is no reason to have no browser.
    session.desk_window = broken
    assert session._desk_args() == []


# ---------- the harness, on the desk ----------

DESK = desk.Desk(display=8, pid=77, serial=1, x=-1440, y=1080, width=1440, height=900)
WINDOW = desk.Window(number=5001, pid=900, owner="Notes", title="New Note",
                     x=-1400, y=1130, width=800, height=600)


@pytest.fixture
def user_screen(monkeypatch):
    """Everything that would touch the user's screen, recorded instead of done."""
    touched = []
    for name in ("desktop_click", "desktop_click_in_image", "desktop_move",
                 "desktop_type", "desktop_press", "desktop_hotkey",
                 "desktop_scroll", "desktop_drag", "open_app",
                 "desktop_screenshot_path"):
        monkeypatch.setattr(computer, name,
                            lambda *a, _n=name, **k: touched.append(_n) or f"{_n} ran")
    return touched


@pytest.fixture
def on_desk(monkeypatch):
    """A running desk with one window on it, and a record of what reached it."""
    record = {"posted": [], "borrowed": [], "pressed": [], "inserted": [],
              "fingerprints": iter(["a", "b"]), "idle_ok": False,
              "hit": None, "focus": None, "values": {}}
    monkeypatch.setattr(desk, "ensure", lambda config: DESK)
    monkeypatch.setattr(desk, "session_locked", lambda: False)
    monkeypatch.setattr(desk, "front_window", lambda d: WINDOW)
    monkeypatch.setattr(desk, "window_at",
                        lambda d, x, y: WINDOW if WINDOW.contains(x, y) else None)
    monkeypatch.setattr(desk, "fingerprint",
                        lambda d: next(record["fingerprints"], "b"))
    for name in ("click", "move", "drag", "scroll", "key", "type_text"):
        monkeypatch.setattr(desk, name,
                            lambda pid, *a, _n=name, **k: record["posted"].append((_n, pid)))

    class Borrow:
        def __init__(self, pid, idle_after, window=0):
            if not record["idle_ok"]:
                raise desk.NotNow("The user touched the Mac 2s ago.")
            record["borrowed"].append(pid)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(desk, "borrowed", Borrow)
    record["focus_on_desk"] = True
    monkeypatch.setattr(desk, "focus_on_desk", lambda d, pid: record["focus_on_desk"])
    monkeypatch.setattr(ax, "available", lambda: True)
    monkeypatch.setattr(ax, "trusted", lambda: True)
    monkeypatch.setattr(ax, "element_at", lambda pid, x, y: record["hit"])
    monkeypatch.setattr(ax, "press",
                        lambda e: record["pressed"].append(e.get("label")) is None)
    monkeypatch.setattr(ax, "focused_element", lambda pid=None: record["focus"])
    monkeypatch.setattr(ax, "menu_item_for_chord", lambda pid, key, mods: None)

    def insert(element, text):
        record["inserted"].append(text)
        record["values"]["focus"] = text
        return True

    monkeypatch.setattr(ax, "insert_text", insert)
    monkeypatch.setattr(ax, "value_of", lambda e: record["values"].get("focus", ""))
    return record


@pytest.fixture
def session():
    class S(chat_tools.ToolsMixin):
        def __init__(self):
            self.config = {"desk": {"enabled": True, "borrow_input_after_idle_s": 30}}
            self.enabled_groups = None
            self.output_fn = lambda *a, **k: None
            self._last_ax = None
    return S()


def test_on_and_broken_refuses_rather_than_falling_back_to_the_users_screen(
        session, user_screen, monkeypatch):
    def cannot(config):
        raise desk.DeskError("macOS would not create a virtual display.")

    monkeypatch.setattr(desk, "ensure", cannot)
    out = session._desktop_action("desktop_click", {"x": 10, "y": 10})
    assert "would not create" in out and "not fall back" in out
    assert session._desktop_action("open_app", {"name": "Notes"}).startswith("macOS would not")
    assert user_screen == []


def test_a_click_by_point_presses_the_control_there_without_the_pointer(
        session, on_desk, user_screen):
    on_desk["hit"] = {"role": "AXButton", "label": "Done", "enabled": True, "_ref": 1}
    out = session._desktop_action("desktop_click", {"x": 100, "y": 100})
    assert on_desk["pressed"] == ["Done"]
    assert "through the accessibility API" in out
    assert on_desk["posted"] == [] and user_screen == []


def test_a_click_with_nothing_to_press_is_posted_to_the_app_first(
        session, on_desk, user_screen):
    out = session._desktop_action("desktop_click", {"x": 100, "y": 100})
    assert on_desk["posted"] == [("click", 900)]
    assert "desk changed" in out and "not used" in out
    assert on_desk["borrowed"] == [] and user_screen == []


def test_a_click_that_changed_nothing_waits_for_the_user_to_be_away(
        session, on_desk, user_screen):
    on_desk["fingerprints"] = iter(["same", "same", "same"])
    out = session._desktop_action("desktop_click", {"x": 100, "y": 100})
    assert "most likely did not land" in out and "2s ago" in out
    assert on_desk["posted"] == [("click", 900)]
    assert on_desk["borrowed"] == [] and user_screen == []


def test_with_the_user_away_the_real_pointer_is_borrowed_for_one_action(
        session, on_desk, user_screen):
    on_desk["fingerprints"] = iter(["same", "same", "changed"])
    on_desk["idle_ok"] = True
    out = session._desktop_action("desktop_click", {"x": 100, "y": 100})
    assert on_desk["posted"] == [("click", 900), ("click", None)]
    assert on_desk["borrowed"] == [900]
    assert "handed straight back" in out and "Nothing on the desk changed" not in out
    assert user_screen == []


def test_borrowed_keys_never_go_to_a_window_off_the_desk(session, on_desk, user_screen):
    """An app with one window on the desk and one on the user's screen may
    hold its focus in theirs: real keys then would type into their document."""
    on_desk["fingerprints"] = iter(["same", "same", "same"])
    on_desk["idle_ok"] = True
    on_desk["focus_on_desk"] = False
    on_desk["focus"] = None
    out = session._desktop_action("desktop_type", {"text": "hello"})
    assert "no keys were sent" in out
    assert on_desk["posted"] == [("type_text", 900)]
    assert user_screen == []


def test_a_point_off_the_desk_is_refused(session, on_desk, user_screen):
    out = session._desktop_action("desktop_click", {"x": 5000, "y": 10})
    assert "off your desk" in out
    assert on_desk["posted"] == [] and user_screen == []


def test_typing_goes_in_at_the_desk_apps_own_focus(session, on_desk, user_screen):
    on_desk["focus"] = {"role": "AXTextArea", "label": "Body", "takes_text": True,
                        "_ref": 1}
    out = session._desktop_action("desktop_type", {"text": "buy milk"})
    assert on_desk["inserted"] == ["buy milk"]
    assert "accessibility API" in out
    assert on_desk["posted"] == [] and user_screen == []


def test_typing_at_a_control_that_takes_no_text_is_refused(session, on_desk, user_screen):
    on_desk["focus"] = {"role": "AXButton", "label": "Delete", "takes_text": False,
                        "_ref": 1}
    out = session._desktop_action("desktop_type", {"text": "x"})
    assert out.startswith("Refused to type")
    assert on_desk["inserted"] == [] and on_desk["posted"] == [] and user_screen == []


def test_a_cmd_chord_is_the_menu_item_pressed(session, on_desk, user_screen, monkeypatch):
    monkeypatch.setattr(ax, "menu_item_for_chord",
                        lambda pid, key, mods: {"label": "Save", "_ref": 1}
                        if (key, mods) == ("s", ["cmd"]) else None)
    out = session._desktop_action("desktop_press", {"key": "cmd+s"})
    assert on_desk["pressed"] == ["Save"] and "menu item 'Save'" in out
    assert on_desk["posted"] == [] and user_screen == []


def test_open_app_opens_on_the_desk(session, on_desk, user_screen, monkeypatch):
    opened = []
    monkeypatch.setattr(desk, "open_app",
                        lambda name, d: opened.append((name, d.display)) or "Opened Notes on your desk.")
    assert session._desktop_action("open_app", {"name": "Notes"}) == "Opened Notes on your desk."
    assert opened == [("Notes", 8)] and user_screen == []


def test_a_locked_mac_is_said_out_loud(session, on_desk, user_screen, monkeypatch):
    monkeypatch.setattr(desk, "session_locked", lambda: True)
    assert session._desktop_action("desktop_click", {"x": 1, "y": 1}) == desk.LOCKED_NOTE
    assert session._see_screen({"target": "desktop"}) == desk.LOCKED_NOTE
    assert user_screen == []


def test_an_empty_desk_says_how_to_put_something_on_it(session, on_desk, monkeypatch):
    monkeypatch.setattr(desk, "front_window", lambda d: None)
    out = session._see_screen({"target": "desktop"})
    assert "is empty" in out and "open_app" in out


def test_a_look_reads_the_desks_window_not_the_front_one(session, on_desk, monkeypatch):
    seen = {}
    monkeypatch.setattr(ax, "window_elements", lambda pid: [
        {"_ref": "theirs", "number": 1, "frame": (0, 0, 10, 10), "title": ""},
        {"_ref": "desk", "number": WINDOW.number, "frame": WINDOW.rect, "title": ""}])

    def snapshot(limit=40, include_text=False, pid=None, window=None, origin=None):
        seen.update(pid=pid, window=window, origin=origin)
        return {"ok": True, "app": "Notes", "window": "New Note", "taken_at": 1e12,
                "origin": origin, "elements": [
                    {"index": 1, "role": "AXButton", "label": "Done", "x": -1300,
                     "y": 1200, "w": 60, "h": 24, "enabled": True, "focused": False,
                     "_ref": 1}]}

    monkeypatch.setattr(ax, "snapshot", snapshot)
    out = session._see_screen({"target": "desktop"})
    assert seen == {"pid": 900, "window": "desk", "origin": (-1440, 1080)}
    assert "YOUR desk" in out
    # Rendered on the desk's own coordinates, the ones a capture of it shows.
    assert "at (140,120)" in out


def test_a_number_from_the_users_screen_never_presses_their_control(
        session, on_desk, monkeypatch):
    """target='user' is for looking. A listing of their screen, cached, must
    not turn desktop_click element=1 into a press on one of their controls."""
    theirs = {"ok": True, "window": "Mail", "taken_at": 1e12, "elements": [
        {"index": 1, "role": "AXButton", "label": "Send", "x": 10, "y": 10,
         "w": 40, "h": 20, "enabled": True, "_ref": "theirs"}]}
    session._last_ax = theirs
    monkeypatch.setattr(session, "_ax_snapshot", lambda limit, d=None: {
        "ok": True, "window": "New Note", "taken_at": 1e12, "desk_window": 5001,
        "elements": [{"index": 1, "role": "AXButton", "label": "Done", "x": -1300,
                      "y": 1200, "w": 60, "h": 24, "enabled": True, "_ref": "desk"}]})
    monkeypatch.setattr(session, "_ax_state", lambda: "")
    session._desktop_action("desktop_click", {"element": 1})
    assert on_desk["pressed"] == ["Done"]


def test_desk_mode_off_leaves_the_old_desktop_alone(user_screen, monkeypatch):
    class S(chat_tools.ToolsMixin):
        def __init__(self):
            self.config = {}
            self.enabled_groups = None
            self.output_fn = lambda *a, **k: None
            self._last_ax = None

    monkeypatch.setattr(desk, "ensure", lambda config: pytest.fail("no desk when off"))
    out = S()._desktop_action("open_app", {"name": "Notes"})
    assert out == "open_app ran" and user_screen == ["open_app"]


def test_symb_desk_parses():
    from symbio.app import cli

    parser = cli._build_parser()
    assert parser.parse_args(["desk"]).desk_command == "status"
    assert parser.parse_args(["desk", "on"]).desk_command == "on"
    assert parser.parse_args(["desk", "peek", "--no-open"]).no_open


def test_selfcheck_skips_the_desk_when_it_is_off():
    from symbio.app import health

    result = health._check_desk({})
    assert result.severity == "info" and "off" in result.message


@pytest.mark.skipif(sys.platform != "darwin" or not shutil.which("clang"),
                    reason="the helper is Objective-C for macOS")
def test_the_native_helper_compiles(tmp_path):
    """Built, never run: running it adds a display to the Mac."""
    done = subprocess.run(
        ["clang", "-fobjc-arc", "-fsyntax-only", "-Wall", "-Wextra", "-Werror",
         str(desk.SOURCE)],
        capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
