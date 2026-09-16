"""Recovering a tool call whose JSON the model failed to escape.

Writing code through a JSON string is the one thing this model reliably gets
wrong. Asked to add a row to a sqlite database it emitted an unescaped double
quote inside the "code" value, the object did not parse, and the entire call
vanished — the turn did nothing and said nothing. Any code containing a quote
hits this, which is most code worth running.
"""
import pytest

from symbio.app.tooling import parse_tools

# Verbatim from a real run, quotes and newline exactly as the model wrote them.
REAL = '''<tool_call>{"name": "execute_code", "arguments": {"code": "import sqlite3
conn = sqlite3.connect('shop.db')
cur = conn.cursor()
cur.execute("INSERT INTO users (name, email) VALUES ('Dmitri', 'dmitri@example.com')")
conn.commit()"}}</tool_call>'''


def test_the_real_failure_now_parses():
    tools = parse_tools(REAL)
    assert len(tools) == 1
    name, params = tools[0]
    assert name == "execute_code"
    assert "INSERT INTO users" in params["code"]
    assert "sqlite3.connect" in params["code"]


def test_the_recovered_code_is_runnable_python():
    """A repair that returns mangled code is worse than no repair."""
    import ast
    _name, params = parse_tools(REAL)[0]
    ast.parse(params["code"])


def test_well_formed_json_is_untouched():
    reply = ('<tool_call>{"name": "run_command", "arguments": '
             '{"cmd": "echo hi"}}</tool_call>')
    assert parse_tools(reply) == [("run_command", {"cmd": "echo hi"})]


def test_correct_escaping_is_honoured_not_doubled():
    reply = ('<tool_call>{"name": "execute_code", "arguments": '
             '{"code": "print(\\"hi\\")"}}</tool_call>')
    _name, params = parse_tools(reply)[0]
    assert params["code"] == 'print("hi")'


def test_several_arguments_survive():
    reply = '''<tool_call>{"name": "write_file", "arguments": {"path": "/tmp/x.py", "content": "print("a")"}}</tool_call>'''
    name, params = parse_tools(reply)[0]
    assert name == "write_file"
    assert params["path"] == "/tmp/x.py"
    assert params["content"] == 'print("a")'


def test_numeric_arguments_are_not_strings():
    reply = '''<tool_call>{"name": "delete_cron_job", "arguments": {"job_id": 3, "note": "say "hi""}}</tool_call>'''
    name, params = parse_tools(reply)[0]
    assert name == "delete_cron_job"
    assert params["job_id"] == 3


def test_an_unrecognisable_blob_is_not_guessed_at():
    """A wrong repair is worse than no call at all."""
    assert parse_tools("<tool_call>{not json at all</tool_call>") == []
    assert parse_tools("<tool_call>{}</tool_call>") == []


def test_a_call_with_no_arguments_still_works():
    reply = '<tool_call>{"name": "list_cron_jobs", "arguments": {}}</tool_call>'
    assert parse_tools(reply) == [("list_cron_jobs", {})]


@pytest.mark.parametrize("code", [
    'print("hello")',
    "x = {'a': 1}\nprint(x)",
    'import re\nre.sub("a", "b", "aaa")',
])
def test_a_range_of_quoted_code_survives_the_round_trip(code):
    reply = f'<tool_call>{{"name": "execute_code", "arguments": {{"code": "{code}"}}}}</tool_call>'
    tools = parse_tools(reply)
    assert tools, f"lost the call for {code!r}"
    assert tools[0][1]["code"] == code


# ---- a tool tag that parses to nothing ----
#
# 22 of the declared tools have no <name>arg</name> form: the alias table is
# derived from _PRIMARY_ARG, which covers only single-argument tools, and the
# rest are reached through richer syntax (<note title=...>, <config set=...>,
# <digest />). When the model reaches for the plain form anyway — which it does,
# having seen the name in the <tools> catalog — the tag matches nothing,
# strip_tool_tags removes it from the display, and the turn ends having done
# nothing and said nothing. Live 2026-08-24: <fetch_html>URL</fetch_html> was
# printed as the whole visible reply and no tool ran.

def test_an_unparseable_tool_tag_is_detected():
    from symbio.app import tooling
    assert tooling.unparsed_tool_tags("<write_file>hello.txt</write_file>") == ["write_file"]
    assert tooling.unparsed_tool_tags(
        "I'll use <delegate_task>scrape</delegate_task>") == ["delegate_task"]


def test_tags_that_do_parse_are_not_flagged():
    from symbio.app import tooling
    for reply in ["<cmd>ls</cmd>", "<read_file>x.py</read_file>",
                  "<search>weather</search>", "<browse>https://x.com</browse>",
                  "<fetch_html>https://x.com</fetch_html>"]:
        assert tooling.unparsed_tool_tags(reply) == [], reply


def test_prose_and_malformed_json_are_not_flagged():
    from symbio.app import tooling
    # A malformed <tool_call> leaves a dangling tag, which the loop already
    # detects and resamples on — a different, non-silent failure.
    assert tooling.unparsed_tool_tags("no tags here at all") == []
    assert tooling.unparsed_tool_tags('<tool_call>{"name": "x"}</tool_call>') == []


def test_a_non_tool_tag_is_ignored():
    from symbio.app import tooling
    assert tooling.unparsed_tool_tags("<thinking>hmm</thinking>") == []
    assert tooling.unparsed_tool_tags("<b>bold</b>") == []


# ---- one closing brace short ----
#
# Live 2026-08-26, six rounds in a row against cloudflare.com/pricing:
#
#   <tool_call>{"name": "browser_open", "arguments": {"url": "https://..."}</tool_call>
#
# The model picked the right tool and the right URL every time. The repair
# extracted the argument body with a greedy \{(.*)\}, which ate the object's
# only closing brace outside the capture group, so the field regex — which
# required a `}` or a following key behind each value — matched nothing and the
# call was rebuilt as `arguments: {}`. The user saw "Browser open error: no URL
# provided" six times and the model finally invented a page-title observation
# to cover for it.

MISSING_ONE_BRACE = (
    '<tool_call>{"name": "browser_open", "arguments": '
    '{"url": "https://www.cloudflare.com/pricing/"}</tool_call>'
)


def test_a_call_missing_one_brace_keeps_its_argument():
    assert parse_tools(MISSING_ONE_BRACE) == [
        ("browser_open", {"url": "https://www.cloudflare.com/pricing/"})]


@pytest.mark.parametrize("name,arg,value", [
    ("read_page", "url", "https://www.cloudflare.com/pricing/"),
    ("web_search", "query", "cloudflared tunnel github"),
    ("run_command", "cmd", "ls -la /tmp"),
    ("browser_click", "target", "Start building for free"),
])
def test_every_single_argument_tool_survives_the_missing_brace(name, arg, value):
    reply = f'<tool_call>{{"name": "{name}", "arguments": {{"{arg}": "{value}"}}</tool_call>'
    assert parse_tools(reply) == [(name, {arg: value})]


def test_a_call_cut_off_before_any_brace_still_recovers():
    """Truncated by the token budget mid-object, no closing brace at all."""
    reply = ('<tool_call>{"name": "web_search", "arguments": '
             '{"query": "cloudflared tunnel github"</tool_call>')
    assert parse_tools(reply) == [
        ("web_search", {"query": "cloudflared tunnel github"})]


# ---- a call to a tool that does not exist ----
#
# A well-formed <tool_call> naming an unknown tool parses cleanly and is then
# removed by the group filter with no error anywhere. Live 2026-08-26 the model
# called `browser_read` three turns running and each time the prose beside it
# ("That page's text: ...") shipped as the answer over a page it never read.
# A `delegate` call — the correct decision, made under pressure after five
# failed rounds — vanished the same way, leaving "After that, I'll ask what's
# next." as the entire reply.

GROUPS = {"memory", "notes", "terminal", "code", "web_search", "digest",
          "train", "cron", "config", "delegate", "system", "browser"}


def test_browser_read_now_reaches_the_real_tool():
    reply = '<tool_call>{"name": "browser_read", "arguments": {}}</tool_call>'
    assert parse_tools(reply, GROUPS) == [("browser_get_text", {})]


def test_delegate_now_reaches_delegate_task():
    reply = ('<tool_call>{"name": "delegate", "arguments": '
             '{"role": "browser_driver", "task": "search for cloudflared"}}</tool_call>')
    assert parse_tools(reply, GROUPS) == [
        ("delegate_task", {"role": "browser_driver",
                           "task": "search for cloudflared"})]


def test_an_invented_tool_name_is_reported_not_swallowed():
    from symbio.app import tooling
    reply = '<tool_call>{"name": "browser_screenshot", "arguments": {}}</tool_call>'
    assert parse_tools(reply, GROUPS) == []
    assert tooling.dropped_tool_calls(reply, GROUPS) == [
        ("browser_screenshot", "unknown")]


def test_a_disabled_tool_is_reported_as_disabled():
    from symbio.app import tooling
    reply = '<tool_call>{"name": "brain_solve", "arguments": {"q": "x"}}</tool_call>'
    assert tooling.dropped_tool_calls(reply, GROUPS) == [("brain_solve", "disabled")]


def test_a_call_that_runs_is_not_reported_as_dropped():
    from symbio.app import tooling
    reply = '<tool_call>{"name": "web_search", "arguments": {"query": "x"}}</tool_call>'
    assert tooling.dropped_tool_calls(reply, GROUPS) == []


def test_the_advertised_names_are_what_the_model_is_told_to_use():
    """The retry message lists these; a name that isn't real would teach one."""
    from symbio.app import tooling
    names = tooling.enabled_tool_names(GROUPS)
    assert "browser_get_text" in names
    assert "delegate_task" in names
    for advertised in names:
        assert parse_tools(
            f'<tool_call>{{"name": "{advertised}", "arguments": {{}}}}</tool_call>',
            GROUPS), f"catalog advertises {advertised} but it parses to nothing"
