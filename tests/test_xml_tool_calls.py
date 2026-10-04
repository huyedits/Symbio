"""The call format Qwen3.5-family and Qwen3-Coder models are trained on.

    <tool_call>
    <function=browser_type>
    <parameter=text>
    testing
    </parameter>
    </function>
    </tool_call>

Nothing read it, so a model answering in its own native format had every call
dropped and was graded as if it never acted — the harness measuring itself.
"""
import pytest

from symbio.app import tooling

GROUPS = {"browser", "memory", "terminal", "code", "web_search"}


@pytest.mark.parametrize("reply,expected", [
    ("<tool_call>\n<function=browser_type>\n<parameter=selector>\n#body\n</parameter>\n"
     "<parameter=text>\ntesting\n</parameter>\n</function>\n</tool_call>",
     [("browser_type", {"selector": "#body", "text": "testing"})]),
    ("Opening it. <function=browser_open><parameter=url>https://x.com</parameter></function>",
     [("browser_open", {"url": "https://x.com"})]),
    ("<tool_call><function=browser_click_at><parameter=x>290</parameter>"
     "<parameter=y>222</parameter></function></tool_call>",
     [("browser_click_at", {"x": 290, "y": 222})]),
])
def test_native_xml_calls_are_read(reply, expected):
    assert tooling.parse_tools(reply, GROUPS) == expected


def test_the_json_form_is_unchanged():
    reply = '<tool_call>{"name": "browser_click", "arguments": {"target": "Post"}}</tool_call>'
    assert tooling.parse_tools(reply, GROUPS) == [("browser_click", {"target": "Post"})]


def test_prose_that_mentions_the_syntax_is_not_a_call():
    assert tooling.parse_tools("Use the function= syntax, like parameter= values.", GROUPS) == []
