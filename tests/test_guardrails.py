"""Guardrails: kinds of action the user can read and set, one plain-English
question per action, and a click judged by what it lands on.

Live 2026-09-27, asked to `go to x.com and make a tweet “testing"`, the model
typed "Hi" into X's composer with browser_type and pressed Post with
browser_click. Neither tool was on any approval list — post_to_x was, and it
was never called — so "Hi" went out under the user's name and nobody was
asked. The same day the user called the gate "too rigid": a post that did go
through post_to_x stopped twice, once as "Allow tool 'post_to_x'?" with no
text at all, and every question spoke in risk scores and flags.
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

def test_a_post_asks_once_with_its_words():
    session = _session(answer=False)

    out = session._execute_tool("post_to_x", {"text": "shipping today"})

    assert len(session.asked) == 1, "it used to ask twice, the first time with no text"
    card = session.asked[0]
    assert isinstance(card, guardrails.Card) and card.kind == "publish"
    assert card.details == "shipping today"
    assert out == "Tool 'post_to_x' was not approved."


def test_the_card_says_when_the_words_are_not_the_ones_asked_for():
    session = _session(answer=False, user_text='go to x.com and make a tweet “testing"')

    session._execute_tool("post_to_x", {"text": "Hi"})

    assert session.asked[0].warning == "You asked for “testing”, not this."


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

    assert "was not approved" in session._execute_tool("post_to_x", {"text": "hi"})
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

    session._execute_tool("post_to_x", {"text": "Hi"})

    card = session.asked[0]
    assert card.said_by == "model"
    assert card.headline.startswith("I'll post “Hi”")
    assert card.details == "Hi", "the exact words are the harness's, whatever the model says"
    assert "go to x.com and make a tweet" in calls[0], "it is told what the user asked"


@pytest.mark.parametrize("reply", [
    '<tool_call>{"name": "post_to_x"}</tool_call>',
    "Sure! Here is a sentence.",
    "",
    RuntimeError("metal went away"),
])
def test_anything_but_a_sentence_about_the_action_falls_back(reply):
    session, _calls = _translating(reply)

    session._execute_tool("post_to_x", {"text": "Hi"})

    card = session.asked[0]
    assert card.said_by == "harness"
    assert card.headline == "Post this on x.com, publicly, as you."


def test_translation_can_be_switched_off():
    session, calls = _translating("I'll post it.", guardrails={"translate": False})

    session._execute_tool("post_to_x", {"text": "Hi"})

    assert calls == []
    assert session.asked[0].said_by == "harness"


# ---- a click is judged by what it lands on ----

_POST_FACTS = {"site": "x.com", "control": "tweetButton", "label": "Post",
               "text": "Hi", "disabled": False, "action": "post"}


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


def test_an_unreadable_page_on_x_is_treated_as_a_post():
    """Fails closed on X only: asking once too often costs a click."""
    button = _Target(RuntimeError("Execution context was destroyed"))
    session, seen = _browser(_Page(button=button), approve=False)

    assert session.click(text="Post").startswith("Not sent")
    assert seen and seen[0]["control"] == "unknown"


def test_an_unreadable_page_elsewhere_is_not():
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
    assert '"text": "testing"' in why, "the refusal names the call that would do it right"


def test_the_app_gate_honours_never_and_always():
    never = _session(config=_full_config(guardrails={"modes": {"publish": "block"}}))
    assert never._publish_gate(dict(_POST_FACTS))[0] is False and never.asked == []
    always = _session(config=_full_config(guardrails={"modes": {"publish": "allow"}}))
    assert always._publish_gate(dict(_POST_FACTS))[0] is True and always.asked == []


# ---- post_to_x: the box is emptied first, and must then hold exactly the post ----

class _Box:
    def __init__(self, page):
        self.page = page

    def click(self):
        self.page.focused = True

    def inner_text(self):
        return self.page.box


class _XPage:
    url = "https://x.com/home"

    def __init__(self, box="", mangle=None):
        self.box, self.mangle, self.timeline, self.selected = box, mangle, [], False
        page = self

        class _Keyboard:
            def type(self, text):
                page.box += page.mangle(text) if page.mangle else text

            def press(self, key):
                if key in ("Meta+A", "Control+A"):
                    page.selected = True
                elif key == "Backspace" and page.selected:
                    page.box, page.selected = "", False

        self.keyboard = _Keyboard()

    def query_selector(self, selector):
        if selector == computer.X_COMPOSER:
            return _Box(self)
        if selector in computer.X_POST_BUTTONS:
            page = self

            class _Button:
                def get_attribute(self, name):
                    return "false"

                def click(self):
                    page.timeline.append(page.box)
                    page.box = ""
            return _Button()
        return None

    def evaluate(self, script, arg=None):
        return any(arg in entry for entry in self.timeline) if arg else False

    def wait_for_timeout(self, _ms):
        return None


def _poster(page):
    session = computer.BrowserSession.__new__(computer.BrowserSession)
    session._ensure_open = lambda: page
    return session


def test_a_draft_left_in_the_box_is_cleared_before_the_post():
    """The live case: "Hi" was sitting in the composer."""
    page = _XPage(box="Hi")

    verdict = _poster(page).post_to_x("testing")

    assert verdict.startswith(computer.X_CONFIRMED)
    assert page.timeline == ["testing"]


def test_an_old_post_with_the_same_words_is_not_proof_of_a_new_one():
    """"testing" posted yesterday is still on the timeline. Finding it again
    after a send that did nothing confirmed a post that never went out."""
    page = _XPage()
    page.timeline.append("testing")
    page.query_selector = (lambda original: lambda selector: (
        type("Dud", (), {"get_attribute": lambda self, n: "false",
                         "click": lambda self: None})()
        if selector in computer.X_POST_BUTTONS else original(selector)))(page.query_selector)

    verdict = _poster(page).post_to_x("testing", timeout_ms=1000)

    assert not verdict.startswith(computer.X_CONFIRMED), verdict


def test_a_box_that_ends_up_holding_something_else_is_not_sent():
    page = _XPage(mangle=lambda text: text + " extra")

    verdict = _poster(page).post_to_x("testing")

    assert verdict.startswith(computer.X_NOT_CONFIRMED)
    assert page.timeline == [] and page.box == ""


def test_an_approved_post_does_not_then_ask_to_visit_x():
    """Live 2026-09-27: "I'll post “testing” on x.com" was approved, and the
    next card asked whether the browser could open x.com."""
    session = _session(config=_full_config(browser={"enabled": True}), answer=True)
    asked_by_browser = []

    class _Browser:
        is_open = False
        _confirmed: set = set()

        def open(self, url):
            if "x.com" not in self._confirmed:
                asked_by_browser.append(url)
                if not session.confirm_fn(f"Open {url}?"):
                    return "Browser open blocked: User denied access to 'x.com'."
            self.is_open = True
            return "Opened browser at https://x.com/home. Page title: Home / X."

        def post_to_x(self, text):
            return f"{computer.X_CONFIRMED}] The post is rendered on the timeline: {text!r}"

    session.browser = _Browser()
    out = session._execute_tool("post_to_x", {"text": "testing"})

    assert out.startswith(computer.X_CONFIRMED)
    assert len(session.asked) == 1 and asked_by_browser == []


# ---- the harness posts what the user spelled out ----

@pytest.mark.parametrize("text,words", [
    ('go to x.com and make a tweet “testing"', "testing"),
    ('Go to x.com and tweet "hello."', "hello."),
    ("post 'hello world' to x", "hello world"),
    ("tweet something about cats", None),
    ("look up the tweet 'hello world' on x.com", None),
    ("what does 'ratio' mean on twitter?", None),
    ('draft a tweet "hello"', None),
    ("make a post 'hi'", None),
])
def test_only_a_post_spelled_out_word_for_word_is_taken_from_the_model(text, words):
    from symbio.app.chat_text import x_post_request

    assert x_post_request(text) == words


def test_the_first_round_is_the_post_with_the_users_words(monkeypatch, tmp_path):
    from tests.test_thinking_truncation import _reply, _session as _turn_session

    output, generated, ran = [], [], []

    def fake_generate(messages, chunk_prefix="", timings=None, think=True, reasoning_budget=0):
        generated.append(messages[-1]["content"])
        return _reply(timings, "Posted it. <end>")

    session = _turn_session(monkeypatch, tmp_path, output)
    session.config["browser"]["enabled"] = True
    session.enabled_groups.add("browser")
    monkeypatch.setattr(session, "_generate_reply", fake_generate)
    monkeypatch.setattr(session, "_execute_tool", lambda name, params: ran.append(
        (name, params)) or f"{computer.X_CONFIRMED}] The post is rendered on the timeline: 'testing'")

    session._agent_turn('go to x.com and make a tweet “testing"')

    assert ran[0] == ("post_to_x", {"text": "testing"})
    assert len(generated) == 1, "the model speaks only after the verdict"
    assert "[Plan] Posting exactly “testing”" in "\n".join(output)


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
