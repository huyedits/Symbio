"""The improvised function form, for tools the prompt declares but never shows.

Only 2 of the 27 tools in the <tools> block have a worked example anywhere in
the assembled system prompt. For the other 25 the model has seen a schema but
never a call, and it invents a shape — reliably a dotted or parenthesised
function with keyword attributes. Nothing matched those, so the tool silently
did not run and the raw text was printed to the user as the reply.

These tests pin the recovery, and — just as importantly — pin the two ways it
must NOT fire: on ordinary prose, and on an example inside a code fence.
"""
import pytest

from symbio.app.tooling import parse_tools, strip_tool_tags


# ---- the improvised forms now resolve ----

@pytest.mark.parametrize("reply", [
    '.schedule_job schedule="0 9 * * *" text="stretch"',
    'schedule_job(schedule="0 9 * * *", text="stretch")',
    '<schedule_job schedule="0 9 * * *" text="stretch" />',
])
def test_an_undemonstrated_tool_is_callable_in_the_shape_the_model_invents(reply):
    assert parse_tools(reply) == [
        ("schedule_job", {"schedule": "0 9 * * *", "text": "stretch"})]


def test_a_no_argument_call_is_recognised_when_dotted():
    assert parse_tools("Let me look. .list_cron_jobs") == [("list_cron_jobs", {})]


def test_a_no_argument_call_is_recognised_with_empty_parentheses():
    assert parse_tools("list_cron_jobs()") == [("list_cron_jobs", {})]


def test_arguments_survive_for_a_tool_that_takes_an_id():
    assert parse_tools('.delete_cron_job id="3"') == [("delete_cron_job", {"id": "3"})]


# ---- and must not fire otherwise ----

def test_a_tool_name_in_ordinary_prose_is_not_a_call():
    """A bare name is far too common in prose to treat as an invocation; the
    dot or the parentheses are what make it a call."""
    assert parse_tools("I could delegate_task to a worker if you want.") == []


def test_an_example_inside_a_code_fence_is_not_a_call():
    """Inside a fence the model is showing the syntax, not using it."""
    reply = 'Like this:\n```\n.schedule_job schedule="0 9 * * *" text="x"\n```\n'
    assert parse_tools(reply) == []


# ---- and must never double-count a call another parser already caught ----

def test_a_well_formed_json_call_is_counted_exactly_once():
    reply = ('<tool_call>{"name":"schedule_job","arguments":'
             '{"schedule":"0 9 * * *","text":"stretch"}}</tool_call>')
    assert parse_tools(reply) == [
        ("schedule_job", {"schedule": "0 9 * * *", "text": "stretch"})]


def test_an_attribute_bearing_xml_tag_is_counted_exactly_once():
    """<type enter="true"> also matches an attribute-form pattern, so this is
    the case that would double-fire if the recovery ran unconditionally."""
    assert parse_tools('<type enter="true">hello</type>') == [
        ("browser_type", {"text": "hello", "enter": True})]


def test_a_note_tag_is_counted_exactly_once():
    assert parse_tools('<note title="T">body here</note>') == [
        ("write_note", {"title": "T", "body": "body here"})]


def test_the_recovery_yields_to_any_other_syntax_in_the_same_reply():
    """When something else already parsed, the recovery stays out of the way
    entirely rather than adding a second interpretation of the same intent."""
    reply = ('<cmd>ls</cmd>\n.schedule_job schedule="0 9 * * *" text="stretch"')
    assert parse_tools(reply) == [("run_command", {"cmd": "ls"})]


def test_a_disabled_group_still_filters_an_improvised_call():
    assert parse_tools('.schedule_job schedule="0 9 * * *" text="x"',
                       enabled_groups={"terminal"}) == []


# ---- and the recovered markup must not reach the user ----

def test_a_recovered_call_is_stripped_from_the_visible_reply():
    """Observed live: the job WAS created and the user was still shown the raw
    tag. Recovering a call but printing its markup is worse than not
    recovering it at all."""
    reply = 'Scheduled. <schedule_job schedule="0 9 * * *" text="stretch"/>'
    assert parse_tools(reply) == [
        ("schedule_job", {"schedule": "0 9 * * *", "text": "stretch"})]
    assert strip_tool_tags(reply) == "Scheduled."


def test_a_dotted_call_leaves_nothing_behind():
    assert strip_tool_tags('.schedule_job schedule="0 9 * * *" text="x"') == ""


def test_stripping_leaves_ordinary_prose_alone():
    text = "I could delegate_task to a worker if you want."
    assert strip_tool_tags(text) == text


# ---- the few-shot examples are not history ----
#
# tool_few_shots() is injected into `messages` as plain user/assistant turns
# (chat.py:4068), so from inside the model it is indistinguishable from what
# actually happened — and the last browser observation in it says Wikipedia is
# open. Live 2026-08-26, in a session whose only tool call was fetch_html and
# which never opened the browser:
#
#   Huy  : NOW READ THE PAGE AGAIN AND TELL ME THE STAR COUNT
#   Caine: I don't see any GitHub repository open right now - the current page
#          is Wikipedia's homepage.
#
# The model read its context correctly; the context was wrong.

def _few_shots():
    # load_config, not config.json: that file is per-install and gitignored,
    # so reading it directly failed on every fresh clone and in CI.
    from symbio.app.config import load_config
    from symbio.tools import tool_few_shots
    return tool_few_shots(load_config())


def test_the_examples_are_closed_out_before_real_history():
    shots = _few_shots()
    disclaimer = next(
        (m for m in shots
         if m["role"] == "user" and "formatting examples" in m["content"]), None)
    assert disclaimer is not None, "nothing tells the model the examples are fiction"


def test_the_disclaimer_comes_after_every_browser_observation():
    """A reset before the observation it is resetting is no reset at all."""
    shots = _few_shots()
    last_browser = max(
        i for i, m in enumerate(shots) if "Opened browser at" in m["content"])
    disclaimer_at = next(
        i for i, m in enumerate(shots)
        if m["role"] == "user" and "formatting examples" in m["content"])
    assert disclaimer_at > last_browser


def test_the_disclaimer_names_the_state_that_leaked():
    shots = _few_shots()
    text = next(m["content"] for m in shots
                if m["role"] == "user" and "formatting examples" in m["content"])
    assert "Wikipedia is not loaded" in text
    assert "no page is open" in text


def test_the_block_still_ends_on_an_assistant_turn():
    """The template alternates; a trailing user turn would double up with the
    first real user message."""
    assert _few_shots()[-1]["role"] == "assistant"


def test_the_tool_examples_themselves_are_untouched():
    """The disclaimer must not cost the format teaching it sits behind."""
    shots = _few_shots()
    joined = " ".join(m["content"] for m in shots)
    for name in ("browser_open", "web_search", "terminal", "browser_click",
                 "browser_press", "browser_scroll"):
        assert name in joined, name


# ---- a tag quoted inside a well-formed call's arguments is text, not a call ----

def test_a_tag_quoted_in_a_note_body_does_not_dispatch_a_second_tool():
    """The legacy tag scanners used to run over the whole reply, the inside of
    a <tool_call> envelope included. So a note ABOUT the tag syntax executed
    the tag: this reply dispatched execute_code alongside the write_note, and
    the code ran. The JSON already said what tool to run and with what
    arguments; nothing inside its argument strings gets a second reading."""
    reply = ('<tool_call>{"name": "write_note", "arguments": {"title": "Snippets", '
             '"body": "Run <py>import os; os.remove(\'x\')</py> for that."}}'
             '</tool_call> Saved.')
    assert parse_tools(reply) == [
        ("write_note", {"title": "Snippets",
                        "body": "Run <py>import os; os.remove('x')</py> for that."})]


def test_a_search_tag_inside_a_query_argument_searches_once():
    reply = ('<tool_call>{"name": "web_search", "arguments": '
             '{"query": "<search>pricing</search>"}}</tool_call> Checking.')
    assert parse_tools(reply) == [
        ("web_search", {"query": "<search>pricing</search>"})]


def test_a_broken_envelope_still_falls_back_to_the_tag_it_spliced():
    """The exclusion is scoped to envelopes that actually parse. A splice —
    JSON opened, legacy tag closed — is not a call the JSON branch can read,
    so the legacy scanner stays its only chance of running what was meant."""
    reply = ('<tool_call>{"name": "web_search", "arguments": '
             '<search>cloudflare pricing</search>.</tool_call> Checking.')
    assert ("web_search", {"query": "cloudflare pricing"}) in parse_tools(reply)


# ---- `name: {json}`, the form that swallowed a live delegation ----

def test_the_name_colon_json_form_resolves():
    """Observed live 2026-09-02 on an explicit "delegate this" instruction:

        _delegate_task: {"role": "summarize", "task": "The cat sat on the mat."}

    The dotted recovery wants `key="value"` pairs and the bare-JSON scan wants
    a "name" key, so neither matched. The call vanished and the raw line was
    printed to the user as the reply."""
    reply = ('_delegate_task: {"role": "summarize", '
             '"task": "The cat sat on the mat."}')
    assert parse_tools(reply) == [
        ("delegate_task", {"role": "summarize",
                           "task": "The cat sat on the mat."})]
    assert strip_tool_tags(reply) == ""


def test_the_leading_underscore_is_optional():
    assert parse_tools('delegate_task: {"role": "summarize", "task": "x"}') == [
        ("delegate_task", {"role": "summarize", "task": "x"})]


def test_the_underscore_needs_a_lookbehind_not_a_word_boundary():
    """`_?\\b(name)` cannot match `_delegate_task` — there is no word boundary
    between the underscore and the name. This pins the spelling that motivated
    the whole recogniser."""
    assert parse_tools('_web_search: {"query": "weather"}') == [
        ("web_search", {"query": "weather"})]


def test_a_tool_name_before_a_colon_in_prose_is_not_a_call():
    assert parse_tools(
        "Sure, I can use delegate_task: it hands work to a worker.") == []


def test_the_json_form_inside_a_code_fence_is_not_a_call():
    assert parse_tools('```\ndelegate_task: {"role": "x", "task": "y"}\n```') == []


def test_a_partial_word_before_the_name_does_not_match():
    """The lookbehind must not let `my_delegate_task:` or `x.web_search:` in."""
    assert parse_tools('mydelegate_task: {"role": "x", "task": "y"}') == []


def test_the_json_form_is_counted_exactly_once():
    reply = ('<tool_call>{"name": "delegate_task", "arguments": '
             '{"role": "summarize", "task": "x"}}</tool_call>')
    assert parse_tools(reply) == [
        ("delegate_task", {"role": "summarize", "task": "x"})]


# ---- a <tool_response> the model wrote itself ----
#
# That tag is the runtime's channel: it is how a real tool's output is fed
# back in. A model writing one is imitating the transcript format and
# inventing an observation it never received.

_FABRICATED = (
    '<tool_call>{"name": "terminal", "arguments": {"cmd": "echo hi"}}</tool_call>\n'
    '<tool_response>{"name": "terminal", "content": "221\\n"}</tool_response>\n'
    'The answer is 221'
)


def test_a_fabricated_observation_does_not_become_a_second_call():
    """Live 2026-09-06, Falcon3-10B asked for 13*17: it wrote its own
    tool_response and answered from it. The JSON scanner then found
    {"name": "terminal"} inside that fabrication and parsed it as a SECOND
    call -- run_command with no arguments -- so an invented observation became
    an executable one."""
    assert parse_tools(_FABRICATED) == [("run_command", {"cmd": "echo hi"})]


def test_the_real_call_beside_it_still_runs():
    """The fix must not cost the genuine call sharing the reply."""
    tools = parse_tools(_FABRICATED)
    assert len(tools) == 1 and tools[0][1]["cmd"] == "echo hi"


def test_a_fabricated_observation_is_not_shown_to_the_user():
    display = strip_tool_tags(_FABRICATED)
    assert "tool_response" not in display
    assert display.strip() == "The answer is 221"


def test_a_lone_fabricated_response_dispatches_nothing():
    assert parse_tools(
        '<tool_response>{"name": "run_command", "arguments": {"cmd": "rm -rf /"}}'
        '</tool_response>') == []


def test_a_tag_quoted_inside_a_fabricated_response_is_inert():
    """Blanking has to happen before the legacy scanners too, not only the
    JSON ones -- otherwise <cmd> inside a fake observation still fires."""
    assert parse_tools(
        '<tool_response>I ran <cmd>rm -rf /</cmd> and it worked</tool_response>') == []


def test_a_real_tool_call_mentioning_the_tag_in_its_data_is_unharmed():
    """The word appearing inside a note body is data, not a channel."""
    tools = parse_tools(
        '<note title="Formats">The runtime replies with '
        '&lt;tool_response&gt; blocks.</note>')
    assert tools and tools[0][0] == "write_note"
