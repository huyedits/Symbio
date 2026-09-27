"""Guardrails: kinds of action the user can read and set, one plain-English
question per action, and a click judged by what it lands on.

Live 2026-09-27, asked to `go to x.com and make a tweet “testing"`, the model
typed "Hi" into X's composer with browser_type and pressed Post with
browser_click. Neither tool was on any approval list, so "Hi" went out under
the user's name and nobody was asked. The same day the user called the gate
"too rigid", and then said plainly: no tweet-to-X command — the model is to
navigate the browser like a person, not call a hard-coded function. So there
is no posting tool; there is a question asked in the page, on any site, when a
click or a key would send what is in a box.
"""
import json

import pytest

from symbio import computer, guardrails, safety
from symbio.app import chat_tools
from symbio.app.chat_constants import _ALWAYS_CONFIRM_TOOLS, _LOCAL_TRUSTED_TOOLS


def _full_config(**overrides):
    from symbio.app import config as app_config

    config = app_config.load_config()
    for key, value in overrides.items():
        config.setdefault(key, {}).update(value)
    return config


def _session(config=None, answer=True, policy="risk", user_text=""):
    asked = []

    class S(chat_tools.ToolsMixin):
        def __init__(self):
            self.config = config if config is not None else _full_config()
            self.enabled_groups = None
            self.output_fn = lambda *a, **k: None
            self.confirm_fn = lambda p: asked.append(p) or answer
            self._confirm_policy = policy
            self._user_text_this_turn = user_text

    session = S()
    session.asked = asked
    return session


# ---- the kinds, and the defaults that are the old behaviour ----

@pytest.mark.parametrize("tool", sorted(_ALWAYS_CONFIRM_TOOLS))
def test_what_always_asked_by_name_is_a_kind_that_always_asks(tool):
    assert guardrails.mode_for(guardrails.kind_of(tool), {}) == "ask"


@pytest.mark.parametrize("tool", sorted(_LOCAL_TRUSTED_TOOLS))
def test_what_asked_only_from_a_phone_still_does(tool):
    kind = guardrails.kind_of(tool)
    assert guardrails.mode_for(kind, {}) == "risky"
    assert guardrails.mode_for(kind, {}, remote=True) == "ask"


def test_the_users_choice_wins_and_nonsense_is_ignored():
    config = {"guardrails": {"modes": {"publish": "allow", "commands": "sometimes"}}}
    modes = guardrails.modes(config)
    assert modes["publish"] == "allow"
    assert modes["commands"] == "risky"


def test_only_a_known_kind_and_mode_can_be_written(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"model_name": "m", "guardrails": {"translate": False}}))

    assert guardrails.set_mode(path, "publish", "block")["ok"] is True
    assert guardrails.set_mode(path, "model_name", "allow")["ok"] is False
    assert guardrails.set_mode(path, "publish", "yolo")["ok"] is False
    data = json.loads(path.read_text())
    assert data["guardrails"] == {"translate": False, "modes": {"publish": "block"}}
    assert data["model_name"] == "m"


def test_the_model_changing_its_guardrails_is_a_sensitive_setting():
    assert safety.is_sensitive_config_key("guardrails.modes.publish")


# ---- the words the user quoted ----

@pytest.mark.parametrize("text,quoted", [
    ('go to x.com and make a tweet “testing"', ["testing"]),     # the live one
    ("tweet 'gm'", ["gm"]),
    ("don't post it, it's not ready", []),
])
def test_quoted_words_are_found_even_in_mismatched_quotes(text, quoted):
    assert guardrails.quoted_texts(text) == quoted


def test_a_post_that_is_not_what_was_asked_for_is_flagged():
    ask = 'go to x.com and make a tweet “testing"'
    assert guardrails.mismatch_warning(ask, "Hi") == "You asked for “testing”, not this."
    assert guardrails.mismatch_warning(ask, "testing") == ""
    assert guardrails.mismatch_warning("tweet something funny", "Hi") == ""


def test_a_card_is_still_a_plain_string():
    """The terminal prints it and Telegram sends it, unchanged."""
    card = guardrails.Card("Run a shell command on this Mac.", "$ ls", kind="commands",
                           reason="Asking because it deletes things.")
    assert isinstance(card, str)
    assert "$ ls" in card and "Run a shell command" in card
    assert card.as_dict()["kind_label"] == "Run commands and code"


# ---- one question per action ----

class _Preview:
    """A browser whose box holds `text`, for submit_form's card."""

    def __init__(self, text):
        self.text = text

    def sending_preview(self):
        return {"text": self.text, "site": "news.example", "url": "https://news.example/submit"}


def test_a_form_that_sends_asks_once_with_its_words():
    session = _session(answer=False)
    session.browser = _Preview("shipping today")

    out = session._execute_tool("submit_form", {"target": "submit"})

    assert len(session.asked) == 1, "it used to ask twice, the first time with no text"
    card = session.asked[0]
    assert isinstance(card, guardrails.Card) and card.kind == "publish"
    assert card.details.startswith("shipping today")
    assert out == "Tool 'submit_form' was not approved."


def test_the_card_says_when_the_words_are_not_the_ones_asked_for():
    session = _session(answer=False, user_text='go to x.com and make a tweet “testing"')
    session.browser = _Preview("Hi")

    session._execute_tool("submit_form", {"target": "Post"})

    assert session.asked[0].warning == "You asked for “testing”, not this."


def test_there_is_no_posting_command():
    """The user's rule: no tweet-to-X function. A model that reaches for one
    is told what exists instead."""
    from symbio.app import tooling

    names = {t["name"] for t in tooling._BUILTIN_TOOLS}
    assert not any("tweet" in n or n.endswith("_x") for n in names), names
    groups = {"browser", "memory", "terminal", "code", "web_search"}
    call = '<tool_call>{"name": "tweet", "arguments": {"text": "hi"}}</tool_call>'
    assert all(name != "post_to_x" for name, _ in tooling.parse_tools(call, groups))


def test_never_refuses_without_asking_and_reads_as_the_users_decision():
    from symbio.app import learn

    session = _session(config=_full_config(guardrails={"modes": {"commands": "block"}}))

    out = session._execute_tool("run_command", {"cmd": "ls"})

    assert session.asked == []
    assert "Never" in out and "did not run" in out
    assert learn.is_user_refusal(out), "a retry must not re-ask what the user ruled out"


def test_always_allow_stops_asking_but_not_for_destruction(monkeypatch):
    from symbio.app import sandbox

    ran = []
    monkeypatch.setattr(sandbox, "run_sandboxed",
                        lambda cmd, config, **k: ran.append(cmd) or (True, "ok"))
    monkeypatch.setattr(sandbox, "run_shell",
                        lambda cmd, config, **k: ran.append(cmd) or (True, "ok"))
    risky = _session(config=_full_config(safety={"require_confirm_score": 2}), answer=False)
    risky._execute_tool("run_command", {"cmd": "curl https://example.com"})     # 2/3
    assert len(risky.asked) == 1, "ask-if-risky asks at the user's threshold"

    session = _session(config=_full_config(guardrails={"modes": {"commands": "allow"}},
                                           safety={"require_confirm_score": 2}),
                       answer=False)
    session._execute_tool("run_command", {"cmd": "curl https://example.com"})
    assert session.asked == [] and ran == ["curl https://example.com"]
    out = session._execute_tool("run_command", {"cmd": "rm -rf /"})
    assert len(session.asked) == 1 and "was not approved" in out
    assert "always allow, but" in session.asked[0].reason


def test_a_yes_on_the_card_is_not_asked_again_by_the_sandbox(tmp_path):
    """The card showed the exact command. "'chmod' is normally blocked. Allow
    once?" straight after it was the same question twice."""
    import os

    target = tmp_path / "x.txt"
    target.write_text("x")
    target.chmod(0o600)
    session = _session(answer=True)

    session._execute_tool("run_command", {"cmd": f"chmod 644 {target}"})

    assert len(session.asked) == 1, [str(a)[:60] for a in session.asked]
    assert oct(os.stat(target).st_mode & 0o777) == "0o644", "it ran, on the one yes"


def test_nobody_to_ask_keeps_the_old_headless_answers(monkeypatch):
    """A scheduled job has no one to ask: a post is refused, and a form the
    operator authorised to submit unattended still goes."""
    monkeypatch.setattr("sys.stdout.isatty", lambda: False)
    session = _session()
    session.confirm_fn = None
    session.browser = _Preview("hi")

    assert "was not approved" in session._execute_tool("submit_form", {"target": "Post"})
    risk = safety.assess_tool_risk("submit_form", {"target": "submit"},
                                   {"safety": {"unattended_submit": True}})
    assert risk["risk_score"] == 0


def test_always_allow_from_the_window_reaches_a_running_session(tmp_path):
    """The window writes config.json; the daemon holds its config in memory."""
    path = tmp_path / "config.json"
    path.write_text("{}")
    session = _session()
    session._guardrails_file = path

    assert session._asks_by_name("train_adapter") is True
    guardrails.set_mode(path, "training", "allow")
    assert session._asks_by_name("train_adapter") is False


# ---- the model says what it is about to do ----

class _Tokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False,
                            enable_thinking=True):
        return "\n".join(m["content"] for m in messages)


def _translating(reply, **config):
    calls = []
    session = _session(config=_full_config(**config), answer=False,
                       user_text='go to x.com and make a tweet “testing"')
    session.model, session.tokenizer = object(), _Tokenizer()

    def generate(model, tokenizer, prompt="", **kwargs):
        calls.append(prompt)
        if isinstance(reply, Exception):
            raise reply
        return reply

    session.generate_fn = generate
    return session, calls


def test_the_headline_is_the_models_translation_of_the_call():
    session, calls = _translating("I'll post “Hi” publicly on your X account, "
                                  "which is not the “testing” you asked for.")
    session.browser = _Preview("Hi")

    session._execute_tool("submit_form", {"target": "Post"})

    card = session.asked[0]
    assert card.said_by == "model"
    assert card.headline.startswith("I'll post “Hi”")
    assert card.details.startswith("Hi"), "the exact words are the harness's, whatever the model says"
    assert "go to x.com and make a tweet" in calls[0], "it is told what the user asked"


@pytest.mark.parametrize("reply", [
    '<tool_call>{"name": "post_to_x"}</tool_call>',
    "Sure! Here is a sentence.",
    "",
    RuntimeError("metal went away"),
])
def test_anything_but_a_sentence_about_the_action_falls_back(reply):
    session, _calls = _translating(reply)
    session.browser = _Preview("Hi")

    session._execute_tool("submit_form", {"target": "Post"})

    card = session.asked[0]
    assert card.said_by == "harness"
    assert card.headline.startswith("Submit the form on news.example")


def test_translation_can_be_switched_off():
    session, calls = _translating("I'll post it.", guardrails={"translate": False})
    session.browser = _Preview("Hi")

    session._execute_tool("submit_form", {"target": "Post"})

    assert calls == []
    assert session.asked[0].said_by == "harness"


# ---- a click is judged by what it lands on ----

_POST_FACTS = {"site": "x.com", "label": "Post", "text": "Hi",
               "disabled": False, "action": "send"}


class _Target:
    def __init__(self, facts):
        self.facts, self.clicked = facts, 0

    def evaluate(self, script, arg=None):
        if isinstance(self.facts, Exception):
            raise self.facts
        return self.facts

    def click(self, timeout=None):
        self.clicked += 1


class _Locator:
    def __init__(self, targets):
        self.targets = targets

    def count(self):
        return len(self.targets)

    def filter(self, visible=True):
        return self

    @property
    def first(self):
        return self.targets[0]


class _Page:
    url = "https://x.com/compose/post"

    def __init__(self, facts=None, button=None, link=None):
        self.facts = facts
        self.button, self.link = button, link
        self.pressed, self.mouse_clicks = [], []
        page = self

        class _Keyboard:
            def press(self, key):
                page.pressed.append(key)

        class _Mouse:
            def click(self, x, y):
                page.mouse_clicks.append((x, y))

        self.keyboard, self.mouse = _Keyboard(), _Mouse()
        self.viewport_size = {"width": 1280, "height": 800}

    def locator(self, selector):
        return _Locator([])

    def get_by_role(self, role, name="", exact=False):
        found = self.button if role == "button" else self.link if role == "link" else None
        return _Locator([found] if found else [])

    def get_by_text(self, text, exact=False):
        return _Locator([])

    def evaluate(self, script, arg=None):
        if "PENDING" in script or "activeElement" not in script and "elementFromPoint" not in script:
            return ""
        return self.facts

    def wait_for_timeout(self, ms):
        return None


def _browser(page, approve):
    session = computer.BrowserSession.__new__(computer.BrowserSession)
    session._page = page
    session._ensure_open = lambda: page
    session._pending_state = lambda _p: ("", "")
    session._submit_note = lambda _p, _s: ""
    session._editable_focused = lambda _p: True
    session._enter_submitted = lambda _p: True
    seen = []
    session.publish_gate = lambda facts: (seen.append(facts) or approve,
                                          "" if approve else "Not sent: the user declined.")
    session._MODAL_SELECTOR = "dialog"
    return session, seen


def test_pressing_post_by_its_label_is_asked_about_and_a_no_stops_it_dead():
    """And it does not fall through to the next candidate for "Post" — the
    sidebar link that opens a fresh composer."""
    button, link = _Target(dict(_POST_FACTS)), _Target(None)
    page = _Page(button=button, link=link)
    session, seen = _browser(page, approve=False)

    out = session.click(text="Post")

    assert seen and seen[0]["text"] == "Hi"
    assert out.startswith("Not sent")
    assert button.clicked == 0 and link.clicked == 0


def test_an_approved_post_is_clicked():
    button = _Target(dict(_POST_FACTS))
    session, _seen = _browser(_Page(button=button), approve=True)

    assert session.click(text="Post").startswith("Clicked")
    assert button.clicked == 1


def test_a_disabled_post_button_is_not_a_question():
    button = _Target({**_POST_FACTS, "disabled": True})
    session, seen = _browser(_Page(button=button), approve=False)

    session.click(text="Post")

    assert seen == []


def test_submit_form_was_already_approved_as_a_post():
    button = _Target(dict(_POST_FACTS))
    session, seen = _browser(_Page(button=button), approve=False)

    session._try_click(session._page, "", "Post", gate=False)

    assert seen == [] and button.clicked == 1


def test_a_coordinate_on_the_post_button_is_asked_about():
    page = _Page(facts=dict(_POST_FACTS))
    session, seen = _browser(page, approve=False)

    out = session.click_at(600, 300)

    assert seen and out.startswith("Not sent") and page.mouse_clicks == []


@pytest.mark.parametrize("key,asked", [("Meta+Enter", True), ("cmd+enter", True),
                                       ("Tab", False)])
def test_the_send_shortcut_is_asked_about(key, asked):
    page = _Page(facts=dict(_POST_FACTS))
    session, seen = _browser(page, approve=False)

    out = session.press(key)

    assert bool(seen) is asked
    assert (page.pressed == []) is asked
    if asked:
        assert out.startswith("Not sent")


def test_an_unreadable_page_is_not_a_question():
    button = _Target(RuntimeError("Execution context was destroyed"))
    page = _Page(button=button)
    page.url = "https://example.com/"
    session, seen = _browser(page, approve=False)

    assert session.click(text="Post").startswith("Clicked")
    assert seen == []


def test_the_app_gate_asks_with_the_words_in_the_box():
    session = _session(answer=False, user_text='go to x.com and make a tweet “testing"')

    ok, why = session._publish_gate(dict(_POST_FACTS))

    assert ok is False
    card = session.asked[0]
    assert card.kind == "publish" and card.details == "Hi"
    assert card.warning == "You asked for “testing”, not this."
    assert "“testing”" in why and "post_to_x" not in why, \
        "the refusal says how to do it right, in the page"


def test_the_app_gate_honours_never_and_always():
    never = _session(config=_full_config(guardrails={"modes": {"publish": "block"}}))
    assert never._publish_gate(dict(_POST_FACTS))[0] is False and never.asked == []
    always = _session(config=_full_config(guardrails={"modes": {"publish": "allow"}}))
    assert always._publish_gate(dict(_POST_FACTS))[0] is True and always.asked == []


# ---- the page's handles, handed over with the page ----

class _ControlsBrowser:
    def __init__(self, controls):
        self._controls = controls

    def controls(self, limit=25):
        return self._controls[:limit]


_X_CONTROLS = [
    {"kind": "field", "selector": '[data-testid="tweetTextarea_0"]', "label": "Post text",
     "value": "", "x": 600, "y": 90},
    {"kind": "button", "selector": '[data-testid="tweetButtonInline"]', "label": "Post",
     "disabled": True, "x": 900, "y": 140},
    {"kind": "button", "selector": 'a[aria-label="Home"]', "label": "Home", "x": 40, "y": 80},
]


def test_a_page_with_a_box_comes_with_its_box_and_its_send_button():
    """Live 2026-09-27 the model typed at nothing and clicked the sidebar's
    "Post" link: the page's real handles only ever arrived after a failure."""
    session = _session()
    session.browser = _ControlsBrowser(_X_CONTROLS)

    note = chat_tools._controls_note(session)

    assert 'selector=\'[data-testid="tweetTextarea_0"]\'' in note
    assert "tweetButtonInline" in note and "[disabled]" in note
    assert "Begin untrusted" in note, "labels come from the page: data, not instructions"


def test_a_page_with_nothing_to_fill_gets_no_list():
    session = _session()
    session.browser = _ControlsBrowser([_X_CONTROLS[2]])

    assert chat_tools._controls_note(session) == ""


def test_the_button_that_sends_comes_before_the_navigation():
    browser = computer.BrowserSession.__new__(computer.BrowserSession)
    found = [{"kind": "button", "label": f"Nav {i}", "selector": f"#n{i}"} for i in range(18)]
    found += [{"kind": "field", "label": "Post text", "selector": "#box"},
              {"kind": "button", "label": "Post", "selector": "#send"}]

    class _P:
        def evaluate(self, script, arg=None):
            return found

    browser._ensure_open = lambda: _P()
    controls, ok = browser.controls_read(limit=4)

    assert ok and [c["selector"] for c in controls[:2]] == ["#box", "#send"]


@pytest.mark.parametrize("typed,noted", [("Hi", True), ("testing", False),
                                         ("testing ", False)])
def test_words_typed_that_the_user_did_not_give_are_pointed_out(typed, noted):
    session = _session(user_text='go to x.com and make a tweet “testing"')

    note = chat_tools._typed_words_note(session, "browser_type", {"text": typed},
                                        f"Typed '{typed}'.")

    assert bool(note) is noted
    if noted:
        assert "“testing”" in note and "“Hi”" in note


def test_words_in_smart_quotes_are_something_the_turn_must_do():
    """`make a tweet “testing"` named no target: the page opened, the model
    said "I've opened X", and nothing noticed “testing” was never written."""
    from symbio.app.chat_constants import request_targets, unmet_targets

    ask = 'go to x.com and make a tweet “testing"'
    assert request_targets(ask) == ["testing"]
    assert unmet_targets(ask, ["Opened browser at https://x.com."]) == ["testing"]
    assert unmet_targets(ask, ["Typed 'testing'."]) == []
    assert request_targets("it’s fine, don’t worry about it") == []


def test_a_click_given_a_point_is_a_click_at_that_point():
    """The controls list shows coordinates; the model sent them to
    browser_click, got a schema error, and lost its task."""
    session = _session(config=_full_config(browser={"enabled": True}))
    clicked = []

    class _B:
        is_open = True

        def click_at(self, x, y):
            clicked.append((x, y))
            return f"Clicked at ({x}, {y})."

        def get_text(self):
            return "page"

        def controls(self, limit=25):
            return []

    session.browser = _B()
    session._last_browsed_url = "https://plants.example/t/42"
    session._status = lambda *a, **k: None
    out = session._execute_tool("browser_click", {"x": 290, "y": 222})

    assert clicked == [(290, 222)] and out.startswith("Clicked at")


def test_a_reply_that_chains_page_steps_runs_them_in_order(monkeypatch, tmp_path):
    """"type it, then click Post" in one reply used to run the type and drop
    the click, costing a whole round; "close, then open x.com" dropped the
    open and the model reported the task as stuck."""
    from tests.test_thinking_truncation import _reply, _session as _turn_session

    output, ran, generated = [], [], []
    replies = iter([
        '<tool_call>{"name": "browser_type", "arguments": {"selector": "#body", "text": "testing"}}</tool_call>'
        '<tool_call>{"name": "browser_click", "arguments": {"target": "Reply"}}</tool_call>',
        "Posted your reply. <end>",
    ])

    def fake_generate(messages, chunk_prefix="", timings=None, think=True, reasoning_budget=0):
        generated.append(1)
        return _reply(timings, next(replies, "Done. <end>"))

    session = _turn_session(monkeypatch, tmp_path, output)
    session.config["browser"]["enabled"] = True
    session.enabled_groups.add("browser")
    monkeypatch.setattr(session, "_generate_reply", fake_generate)
    monkeypatch.setattr(session, "_execute_tool", lambda name, params: ran.append(name) or (
        "Typed 'testing'." if name == "browser_type" else "Clicked button 'Reply'."))

    session._agent_turn("reply “testing” to the thread")

    assert ran == ["browser_type", "browser_click"]
    assert len(generated) == 2, "both steps in one round, then the answer"
    assert "were also requested in the same reply but" not in "\n".join(
        str(m.get("content")) for m in session.history)


def test_a_chain_stops_at_the_first_step_that_fails(monkeypatch, tmp_path):
    from tests.test_thinking_truncation import _reply, _session as _turn_session

    output, ran = [], []
    replies = iter([
        '<tool_call>{"name": "browser_type", "arguments": {"text": "testing"}}</tool_call>'
        '<tool_call>{"name": "browser_click", "arguments": {"target": "Reply"}}</tool_call>',
        "I couldn't type it. <end>",
    ])
    session = _turn_session(monkeypatch, tmp_path, output)
    session.config["browser"]["enabled"] = True
    session.enabled_groups.add("browser")
    monkeypatch.setattr(session, "_generate_reply", lambda *a, **k: _reply(
        k.get("timings"), next(replies, "Done. <end>")))
    monkeypatch.setattr(session, "_execute_tool", lambda name, params: ran.append(name) or (
        "Type failed: nothing editable is focused, so no keys were sent."))

    session._agent_turn("reply “testing” to the thread")

    assert ran[0] == "browser_type" and "browser_click" not in ran[:1]
    assert ran.count("browser_click") == 0 or ran.index("browser_click") > 0


def test_the_page_note_says_how_to_send_on_any_site():
    browser = computer.BrowserSession.__new__(computer.BrowserSession)

    class _P:
        url = "https://example.org/"

        def title(self):
            return "Example"

    browser._page, browser._last_url = _P(), "https://example.org/"
    note = browser.status()

    assert "click the button beside the box that sends it" in note
    assert "post_to_x" not in note


def test_a_turn_whose_answer_predates_its_last_step_says_how_it_ended(monkeypatch, tmp_path):
    """"I've opened X. Let's create your tweet." was the whole visible answer
    to a turn that went on to click Post three more times."""
    from tests.test_thinking_truncation import _reply, _session as _turn_session

    output = []
    replies = iter([
        "I've opened X. Let's create your tweet.\n<click>Post</click>",
        "<click>Post it</click>",
    ])

    def fake_generate(messages, chunk_prefix="", timings=None, think=True, reasoning_budget=0):
        return _reply(timings, next(replies, ""))

    session = _turn_session(monkeypatch, tmp_path, output)
    session.config["browser"]["enabled"] = True
    session.enabled_groups.add("browser")
    monkeypatch.setattr(session, "_generate_reply", fake_generate)
    steps = iter(["Clicked link 'Post'.",
                  "Click failed: no visible element with text 'Post'."])
    monkeypatch.setattr(session, "_execute_tool",
                        lambda name, params: next(steps, "Click failed: gone."))

    session._agent_turn("go to x.com and write something")

    log = "\n".join(output)
    assert "I didn't finish — my last step failed: Click failed" in log


# ---- the soul store reads the user, not the tools ----

def test_tool_output_is_not_mined_as_the_users_values():
    from symbio.app import soul

    history = [
        {"role": "user", "content": 'go to x.com and make a tweet “testing"'},
        {"role": "assistant", "content": "<press>Enter</press>"},
        {"role": "user", "content": "[System observation: Press failed: Enter did not "
                                    "submit the form. Click the submit button to submit.]"},
    ]

    prompt = soul.build_prompt(history, {"user_name": "Huy"})

    assert "make a tweet" in prompt
    assert "Click the submit button" not in prompt
