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

from symbio.app.tooling import parse_tools


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
