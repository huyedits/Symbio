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
