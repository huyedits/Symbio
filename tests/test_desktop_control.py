"""Driving the machine itself: the tree says what is there, not a guess.

The desktop surface used to be a mouse that could click a coordinate and a
keyboard that could send one key. Everything else -- which control is where,
whether a field has focus, whether the click landed -- was inferred from a
screenshot by a vision model that cannot ground anything thinner than a 32px
patch, and that had to be swapped in over the headmaster to be asked at all.

macOS already publishes the answer. Every native control exposes its role, its
label and its exact frame through the accessibility API, which is what a
screen reader reads. These tests drive that path with a stubbed tree: nothing
here touches the real screen, and none of it needs the Accessibility grant.
"""
import pytest

from symbio import ax, computer
from symbio.app import chat_tools


def _element(index, role, label, x=100, y=200, w=80, h=30, **kw):
    element = {"index": index, "role": role, "label": label,
               "x": x, "y": y, "w": w, "h": h, "enabled": True,
               "focused": False, "_ref": object()}
    element.update(kw)
    return element


@pytest.fixture
def tree(monkeypatch):
    """A window with three controls, and a record of what was done to them."""
    state = {
        "elements": [
            _element(1, "AXButton", "Post", x=1200, y=320, w=64, h=32),
            _element(2, "AXTextArea", "What's happening?", x=900, y=200,
                     w=448, h=96),
            _element(3, "AXButton", "Close", x=80, y=40, w=24, h=24),
        ],
        "pressed": [], "typed": [], "focused": [], "window": "Home / X",
        "press_ok": True, "set_ok": True, "values": {}, "focus": None,
        # What the field does to what it is given: a length cap, a reformat.
        "clamp": lambda text: text,
    }

    def snapshot(limit=40, include_text=False):
        return {"ok": True, "app": "Safari", "window": state["window"],
                "elements": state["elements"], "truncated": False,
                "taken_at": 1e12}

    monkeypatch.setattr(ax, "available", lambda: True)
    monkeypatch.setattr(ax, "trusted", lambda: True)
    monkeypatch.setattr(ax, "snapshot", snapshot)
    monkeypatch.setattr(ax, "press",
                        lambda e: state["pressed"].append(e["index"]) is None
                        and state["press_ok"])
    def set_text(element, text):
        state["typed"].append((element["index"], text))
        state["values"][element["index"]] = state["clamp"](text)
        return state["set_ok"]

    monkeypatch.setattr(ax, "set_text", set_text)
    monkeypatch.setattr(ax, "focus",
                        lambda e: state["focused"].append(e["index"]) is None)
    monkeypatch.setattr(ax, "value_of",
                        lambda e: state["values"].get(e["index"], ""))
    monkeypatch.setattr(ax, "focused_element", lambda: state["focus"])
    return state


@pytest.fixture
def session():
    class S(chat_tools.ToolsMixin):
        def __init__(self):
            self.config = {}
            self.enabled_groups = None
            self.output_fn = lambda *a, **k: None
            self._last_ax = None
    return S()


@pytest.fixture
def mouse(monkeypatch):
    done = []
    monkeypatch.setattr(computer, "desktop_click",
                        lambda x, y, clicks=1, button="left":
                        done.append(("click", x, y, clicks, button)) or "ok")
    monkeypatch.setattr(computer, "desktop_move",
                        lambda x, y: done.append(("move", x, y)) or "ok")
    monkeypatch.setattr(computer, "desktop_drag",
                        lambda x1, y1, x2, y2, duration=0.4, button="left":
                        done.append(("drag", x1, y1, x2, y2)) or "ok")
    monkeypatch.setattr(computer, "desktop_type",
                        lambda text, interval=0.01:
                        done.append(("type", text)) or "ok")
    monkeypatch.setattr(computer, "desktop_scroll",
                        lambda direction="down", amount=5, x=None, y=None:
                        done.append(("scroll", direction, amount, x, y)) or "ok")
    return done


# --- looking ---------------------------------------------------------------


def test_a_look_at_the_desktop_lists_real_controls(session, tree):
    """No screenshot, no vision model, no deep sleep: the window is asked."""
    out = session._dispatch_tool("see_screen", {"target": "desktop"})

    assert "Post" in out and "What's happening?" in out
    assert "Safari" in out and "Home / X" in out
    # The numbers are the handle. A coordinate is a fallback, not the interface.
    assert "1 Button" in out.replace("  ", " ")
    assert "desktop_click" in out


def test_the_28px_composer_is_listed_like_anything_else(session, tree):
    """Vision could not ground a control thinner than one patch — x.com's
    composer is 28px — and reported it as absent. The tree has no floor."""
    tree["elements"].append(
        _element(4, "AXTextField", "Post your reply", x=500, y=700, w=300, h=28))

    out = session._dispatch_tool(
        "see_screen", {"target": "desktop", "question": "the reply box"})

    assert "Post your reply" in out
    assert "300x28" in out


def test_a_question_with_no_match_says_it_is_not_there(session, tree):
    """The whole value of an exact tree: absence is a fact, not a miss."""
    out = session._dispatch_tool(
        "see_screen", {"target": "desktop", "question": "the Publish button"})

    assert "Nothing in this window is labelled" in out
    assert "Publish" in out


def test_a_window_with_no_tree_falls_through_to_vision(session, tree,
                                                       monkeypatch):
    """A canvas, a game, a screen share: the tree is empty and vision is the
    only instrument left, so the look must not stop here."""
    monkeypatch.setattr(ax, "snapshot", lambda limit=40, include_text=False: {
        "ok": True, "app": "Game", "window": "", "elements": [],
        "truncated": False, "taken_at": 1e12})

    # Vision is off here, so the fall-through is visible as vision's own
    # answer rather than the listing -- and nothing in this file has to touch
    # the real screen to see it. Left on, this reaches a live screen capture
    # and reports whatever the Screen Recording grant allows, which is a fact
    # about the box running the suite, not about the fall-through.
    session.config = {"vision": {"enabled": False}}

    out = session._dispatch_tool("see_screen", {"target": "desktop"})

    assert "vision" in out.lower() or "disabled" in out.lower()


# --- acting ----------------------------------------------------------------


def test_clicking_by_number_presses_the_control_itself(session, tree, mouse):
    session._dispatch_tool("see_screen", {"target": "desktop"})

    out = session._dispatch_tool("desktop_click", {"element": 1})

    assert tree["pressed"] == [1]
    assert mouse == [], "a press that worked should not also move the mouse"
    assert "Post" in out


def test_a_control_that_will_not_press_is_clicked_at_its_centre(session, tree,
                                                               mouse):
    """AXPress is refused by canvas-backed views and some table cells. The
    frame is still exact, and it is already in the points the mouse uses —
    unlike a screenshot coordinate, which needs the Retina scale undone."""
    tree["press_ok"] = False

    session._dispatch_tool("desktop_click", {"element": 1})

    assert mouse == [("click", 1232, 336, 1, "left")]


def test_a_right_click_and_a_double_click_go_to_the_mouse(session, tree, mouse):
    session._dispatch_tool("desktop_click", {"element": 3, "button": "right"})
    session._dispatch_tool("desktop_click", {"element": 3, "clicks": 2})

    assert mouse == [("click", 92, 52, 1, "right"),
                     ("click", 92, 52, 2, "left")]
    assert tree["pressed"] == [], "neither is the control's default action"


def test_a_disabled_control_is_refused_with_the_reason(session, tree, mouse):
    tree["elements"][0]["enabled"] = False

    out = session._dispatch_tool("desktop_click", {"element": 1})

    assert "disabled" in out
    assert tree["pressed"] == [] and mouse == []


def test_a_number_that_is_not_on_screen_says_to_look_again(session, tree):
    out = session._dispatch_tool("desktop_click", {"element": 9})

    assert "no element 9" in out.lower()
    assert "see_screen" in out


def test_a_click_that_changed_nothing_says_so(session, tree, monkeypatch):
    """"Clicked" is not "did something". The window title and the focused
    control are what a person glances at, and they cost two reads."""
    monkeypatch.setattr(chat_tools.ToolsMixin, "_ax_state",
                        lambda self: "Home / X|AXButton|Post")

    out = session._dispatch_tool("desktop_click", {"element": 1})

    assert "Nothing about the window changed" in out


# --- typing ----------------------------------------------------------------


def test_typing_into_a_numbered_field_names_the_field(session, tree):
    out = session._dispatch_tool(
        "desktop_type", {"element": 2, "text": "hello there"})

    assert tree["typed"] == [(2, "hello there")]
    assert tree["focused"] == [2]
    assert "hello there" in out


def test_typing_blind_at_a_button_is_refused(session, tree):
    """Keys at a window with no text field focused are not discarded, they are
    shortcuts — which is how a message meant for a composer becomes a string
    of commands the app happened to bind."""
    tree["focus"] = {"role": "AXButton", "label": "Post", "takes_text": False}

    out = session._dispatch_tool("desktop_type", {"text": "hello"})

    assert "Refused to type" in out
    assert "element" in out


def test_typing_blind_into_a_focused_field_is_allowed(session, tree, mouse):
    tree["focus"] = {"role": "AXTextArea", "label": "composer",
                     "takes_text": True}

    session._dispatch_tool("desktop_type", {"text": "hello"})

    assert mouse == [("type", "hello")]


def test_a_field_that_rejects_the_value_is_reported_not_claimed(session, tree):
    """Some fields reformat or refuse what is set. Saying "typed" when the
    field holds something else is the fabricated-completion shape."""
    tree["clamp"] = lambda text: text[:2]

    out = session._dispatch_tool("desktop_type", {"element": 2, "text": "1234"})

    assert "not what was sent" in out


# --- the rest of a pointer -------------------------------------------------


def test_a_chord_is_sent_as_a_chord(monkeypatch):
    """desktop_press used to send one key, and a desktop is driven with
    chords. 'cmd' on its own does nothing and reports success."""
    sent = []

    class _Fake:
        FAILSAFE = True
        PAUSE = 0.05

        @staticmethod
        def hotkey(*keys):
            sent.append(keys)

        @staticmethod
        def press(key):
            sent.append((key,))

    monkeypatch.setattr(computer, "_init_pyautogui", lambda: _Fake)

    computer.desktop_hotkey("cmd+shift+4")
    computer.desktop_hotkey("Enter")

    assert sent == [("command", "shift", "4"), ("enter",)]


def test_scroll_and_drag_take_element_numbers(session, tree, mouse):
    session._dispatch_tool("desktop_scroll", {"element": 2, "direction": "down"})
    session._dispatch_tool("desktop_drag", {"from_element": 3, "to_element": 1})

    assert mouse == [("scroll", "down", 5, 1124, 248),
                     ("drag", 92, 52, 1232, 336)]


def test_a_drag_with_neither_end_says_what_to_give(session, tree, mouse):
    out = session._dispatch_tool("desktop_drag", {"to_element": 1})

    assert "from_element" in out and "from_x" in out
    assert mouse == []


def test_waiting_is_a_tool_so_the_turn_does_not_report_too_early(session, tree,
                                                                monkeypatch):
    slept = []
    monkeypatch.setattr(chat_tools.time, "sleep", lambda s: slept.append(s))

    out = session._dispatch_tool("desktop_wait", {"seconds": 30})

    assert slept == [10.0], "capped, not obeyed"
    assert "Look again" in out


# --- the names a model reaches for -----------------------------------------


GROUPS = {"desktop", "browser", "memory", "notes", "terminal", "code",
          "web_search", "config", "cron", "digest", "train", "delegate",
          "system"}


@pytest.mark.parametrize("call,expected", [
    ('{"name": "left_click", "arguments": {"element": 3}}',
     ("desktop_click", {"element": 3})),
    ('{"name": "key", "arguments": {"combo": "cmd+s"}}',
     ("desktop_press", {"key": "cmd+s"})),
    ('{"name": "mouse_move", "arguments": {"element": 2}}',
     ("desktop_move", {"element": 2})),
    ('{"name": "left_click_drag", "arguments": {"x1": 1, "y1": 2, "x2": 3, "y2": 4}}',
     ("desktop_drag", {"from_x": 1, "from_y": 2, "to_x": 3, "to_y": 4})),
    ('{"name": "open_application", "arguments": {"app": "Notes"}}',
     ("open_app", {"name": "Notes"})),
    ('{"name": "screenshot", "arguments": {"target": "desktop"}}',
     ("see_screen", {"target": "desktop"})),
    ('{"name": "wait", "arguments": {"seconds": 3}}',
     ("desktop_wait", {"seconds": 3})),
])
def test_the_computer_use_spellings_all_resolve(call, expected):
    """These are the names a model has seen on other computer-use tools. Every
    one of them was an unknown tool here, which reads back as "this assistant
    cannot drive a computer"."""
    from symbio.app import tooling

    assert tooling.parse_tools(f"<tool_call>{call}</tool_call>",
                               GROUPS) == [expected]


def test_the_other_loop_runs_the_same_desktop_code(tree):
    """symbio/tools.py is a second registry over the same catalog. The element
    numbers and the focus guard are not worth a second implementation."""
    from symbio.tools import build_tool_registry, tool_metadata

    agent = type("A", (), {"config": {}, "enabled_groups": None})()
    registry = build_tool_registry(agent)
    meta = tool_metadata("desktop_click", registry, agent)

    assert meta["run"]({"element": 1}).startswith("Pressed 'Post'")
    assert tree["pressed"] == [1]


# --- the shipped wording that never reached an existing install ------------


def test_a_tool_file_seeded_before_a_change_is_reported_not_overwritten(
        tmp_path, monkeypatch):
    """tools/*.md is authoritative, so a description improved in code is
    invisible on any install that already seeded the old one — the shape of a
    guard that is live in the repo and dead in the running config. Reporting
    it is the fix; overwriting the user's directory silently is not."""
    from symbio import constants
    from symbio.app import tool_docs

    monkeypatch.setattr(constants, "TOOLS_DIR", tmp_path)
    built_in = [{"name": "widget", "description": "The new wording.",
                 "parameters": {"type": "object", "properties": {}}}]

    # Seeded from an older version of the same tool, fingerprint and all.
    old = [{"name": "widget", "description": "The old wording.",
            "parameters": {"type": "object", "properties": {}}}]
    tool_docs.ensure_seeded(old, {"widget": "other"}, {"widget": "core"})

    rewritten, kept = tool_docs.refresh(built_in, {"widget": "other"},
                                        {"widget": "core"})

    assert rewritten == ["widget"] and kept == []
    assert "The new wording." in (tmp_path / "widget.md").read_text()


def test_an_edited_tool_file_is_kept(tmp_path, monkeypatch):
    from symbio import constants
    from symbio.app import tool_docs

    monkeypatch.setattr(constants, "TOOLS_DIR", tmp_path)
    schema = [{"name": "widget", "description": "Built in.",
               "parameters": {"type": "object", "properties": {}}}]
    tool_docs.ensure_seeded(schema, {"widget": "other"}, {"widget": "core"})
    path = tmp_path / "widget.md"
    path.write_text(path.read_text().replace("Built in.", "Mine, hands off."),
                    encoding="utf-8")

    rewritten, kept = tool_docs.refresh(schema, {"widget": "other"},
                                        {"widget": "core"})

    assert rewritten == [] and kept == ["widget"]
    assert "Mine, hands off." in path.read_text()
    # Naming it explicitly is how the user asks for it anyway.
    assert tool_docs.refresh(schema, {"widget": "other"}, {"widget": "core"},
                             names=["widget"])[0] == ["widget"]


# --- the grant that was there, reported as missing -------------------------


def test_the_frontmost_app_falls_back_to_the_window_server(monkeypatch):
    """AXFocusedApplication on the system-wide element returns
    kAXErrorCannotComplete (-25204) on Huy's Mac — granted, trusted, every
    per-application tree readable. Believing that one call made a working
    accessibility path report itself as a missing permission, which is the
    worst possible answer: the model is told it cannot see the screen when it
    can. The window server knows who is in front and needs no grant to say."""
    calls = []

    monkeypatch.setattr(ax, "_attr",
                        lambda element, name: calls.append(name) or None)
    monkeypatch.setattr(ax, "frontmost_window", lambda: (658, "Code", "main.py"))
    monkeypatch.setattr(ax._ax, "AXUIElementCreateSystemWide", lambda: 1)
    monkeypatch.setattr(ax._ax, "AXUIElementCreateApplication",
                        lambda pid: f"app-{pid}")

    assert ax._focused_app() == "app-658"
    assert "AXFocusedApplication" in calls, "the documented route is tried first"


def test_no_window_at_all_is_still_reported_as_the_permission(monkeypatch):
    """With nothing in front and nothing from the API there is no way to tell
    an empty screen from a missing grant, and the grant is the likelier one."""
    monkeypatch.setattr(ax, "_attr", lambda element, name: None)
    monkeypatch.setattr(ax, "frontmost_window", lambda: (0, "", ""))
    monkeypatch.setattr(ax._ax, "AXUIElementCreateSystemWide", lambda: 1)
    monkeypatch.setattr(ax, "trusted", lambda: True)

    snap = ax.snapshot()

    assert snap["ok"] is False
    assert "Accessibility" in snap["reason"]


# --- the dead ends that are training data ----------------------------------


@pytest.mark.parametrize("observation", [
    "Refused to type: the focused control is a Button ('Post'), not a text field.",
    "Nothing about the window changed — same title, same focus — so this may not have landed.",
    "There is no element 9 on screen — this window lists 3.",
    "The accessibility tree is empty because macOS has not granted this process "
    "Accessibility permission — it returns nothing rather than an error.",
])
def test_a_desktop_dead_end_counts_as_a_failure(observation):
    """Each of these is an action that produced no work while reading like an
    ordinary result. Counted as a success, the turn ends there — and the
    recovery that would have followed is the training example."""
    from symbio.app import learn

    assert learn.sounds_like_tool_error(observation), observation


@pytest.mark.parametrize("observation", [
    "Pressed 'Post' (Button, element 1).",
    "Typed into 'What's happening?' (element 2); it now holds 'hello'.",
    "Scrolled down by 5.",
    "Opened Notes. It is now frontmost.",
])
def test_a_desktop_action_that_worked_is_not_a_failure(observation):
    from symbio.app import learn

    assert not learn.sounds_like_tool_error(observation), observation
