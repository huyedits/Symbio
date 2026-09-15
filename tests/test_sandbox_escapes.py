"""Tests for the sandbox's AST guard (symbio.app.sandbox).

The import blocklist reads `import` *statements*, so for as long as it was the
only check, every dynamic route to the same module walked straight past it:
`__import__('shutil').rmtree('adapters')` parses to a Call, not an Import, and
was allowed on the line under a refused `import shutil`. A blocklist that only
reads one spelling of the thing it blocks is decoration, so the escapes are
tested here beside the honest spelling.

The false-positive cases matter as much: the sandbox exists to run code, and a
guard that refuses ordinary computation gets its teeth pulled within a day.
"""

import pytest

from symbio.app import config as app_config
from symbio.app import sandbox

BLOCKED = {"os", "sys", "subprocess", "shutil", "pathlib", "importlib", "builtins"}


@pytest.mark.parametrize("code", [
    "import os",
    "from os import remove",
    "import os.path",
    "from shutil import rmtree",
])
def test_the_honest_spelling_is_still_refused(code):
    safe, msg = sandbox._is_code_safe(code, BLOCKED)
    assert not safe and "not allowed" in msg


@pytest.mark.parametrize("code,culprit", [
    ("__import__('shutil').rmtree('adapters')", "__import__"),
    ("eval(compile(\"import shutil\", 'x', 'exec'))", "eval"),
    ("exec('import os')", "exec"),
    ("m = compile('import os', '<s>', 'exec')", "compile"),
    ("print(globals()['__builtins__'])", "globals"),
    ("vars()", "vars"),
])
def test_dynamic_routes_to_a_blocked_module_are_refused(code, culprit):
    """Each of these reaches a module the blocklist refuses, without ever
    writing an import statement."""
    safe, msg = sandbox._is_code_safe(code, BLOCKED)
    assert not safe, f"{code!r} walked past the blocklist"
    assert culprit in msg, "the refusal should name what it caught"


@pytest.mark.parametrize("code", [
    "getattr(__builtins__, '__import__')('os')",
    "().__class__.__bases__[0].__subclasses__()",
    "[].__class__.__mro__[1].__subclasses__()",
    "f = lambda: 0\nprint(f.__globals__)",
    "print(open.__self__.__loader__)",
])
def test_interpreter_back_doors_are_refused(code):
    """The classic routes from any object back to a module the blocklist
    refuses. Blocking `__import__` by name means nothing while these spell
    the same thing."""
    safe, msg = sandbox._is_code_safe(code, BLOCKED)
    assert not safe, f"{code!r} reached past the import blocklist"
    assert "reaches past" in msg


def test_getattr_itself_is_still_allowed():
    """Deliberately not blocked: honest computation uses it, and the escape
    needs one of the dunder targets as its argument, which is what gets
    caught instead."""
    safe, _ = sandbox._is_code_safe("data = [1, 2, 1]\nprint(getattr(data, 'count')(1))",
                                    BLOCKED)
    assert safe


@pytest.mark.parametrize("code", [
    "print(sum(range(10)))",
    "import math\nprint(math.sqrt(2))",
    "import json\nprint(json.dumps({'a': 1}))",
    "with open('out.txt', 'w') as f:\n    f.write('hi')",
    "class Point:\n    def __init__(self, x):\n        self.x = x\nprint(Point(3).x)",
    "print([w.upper() for w in 'a b c'.split()])",
    "def fib(n):\n    return n if n < 2 else fib(n - 1) + fib(n - 2)\nprint(fib(10))",
])
def test_ordinary_code_still_runs(code):
    safe, msg = sandbox._is_code_safe(code, BLOCKED)
    assert safe, f"the guard ate legitimate code: {msg}"


def test_relative_imports_stay_refused():
    safe, msg = sandbox._is_code_safe("from . import something", BLOCKED)
    assert not safe and "Relative imports" in msg


def test_the_escape_is_refused_before_a_process_is_spawned():
    """End to end through the runner: the guard has to reject the code, not
    catch it after the interpreter already had it."""
    config = app_config.load_config()

    ok, out = sandbox.run_python_code(
        "__import__('shutil').rmtree('adapters')", config)

    assert not ok
    assert "not allowed in the sandbox" in out


def test_the_runner_still_runs_honest_code():
    config = app_config.load_config()

    ok, out = sandbox.run_python_code("print(6 * 7)", config)

    assert ok and out.strip() == "42"


# ---- the denylist was a list of spellings, not of programs ----
#
# `args[0] in blocked` was a literal string match. On 2026-08-27, with "rm" on
# the denylist the whole time, both of these ran with no prompt:
#
#     /bin/rm /tmp/symbio_victim.txt   -> (True, '')   the file was deleted
#     env rm  /tmp/nothing_here        -> rm executed
#
# A denylist any absolute path defeats is decoration.

import json as _json
import pytest as _pytest


def _cfg():
    return {"sandbox": {"blocked_commands": [
        "rm", "sudo", "dd", "mkfs", "curl", "wget", "python", "python3",
        "bash", "sh", "zsh", "chmod", "chown", "ssh", "scp"]}}


@_pytest.mark.parametrize("command,expected", [
    # Absolute and relative paths resolve to the same program.
    ("/bin/rm -rf /tmp/x", "rm"),
    ("/usr/bin/rm /tmp/x", "rm"),
    ("../../bin/rm /tmp/x", "rm"),
    # Wrappers that run a program named in their own arguments.
    ("env rm /tmp/x", "rm"),
    ("/usr/bin/env python3 -c 'x'", "python3"),
    ("xargs rm", "rm"),
    ("nice -n 5 curl http://x", "curl"),     # the flag VALUE used to stop the walk
    ("timeout 5 bash -c ls", "bash"),
    ("nohup dd if=/dev/zero of=/tmp/x", "dd"),
    ("sudo rm -rf /", "sudo"),
])
def test_a_denylisted_program_is_found_however_it_is_spelled(command, expected):
    from symbio.app import sandbox
    import shlex
    hit = sandbox._blocked_binary(shlex.split(command),
                                  set(_cfg()["sandbox"]["blocked_commands"]))
    assert hit == expected, f"{command!r} evaded the denylist"


@_pytest.mark.parametrize("command", [
    "echo hello",
    "ls -la /tmp",
    "grep -rn foo .",
    "/bin/echo hi",
    "find . -name '*.py'",
    "wc -l setup.sh",
    "cat notes/x.md",
])
def test_ordinary_commands_are_not_caught(command):
    from symbio.app import sandbox
    import shlex
    assert sandbox._blocked_binary(
        shlex.split(command), set(_cfg()["sandbox"]["blocked_commands"])) is None


def test_a_blocked_program_still_reaches_the_confirmation_prompt():
    """Blocking is not refusal — the user can still approve a one-off run.
    Non-interactive callers (the cron thread) must never be prompted."""
    from symbio.app import sandbox
    ok, out = sandbox.run_sandboxed("/bin/rm /tmp/nothing", _cfg(), interactive=False)
    assert ok is False and "blocked in sandbox" in out
    assert "rm" in out, "the message should name the real program, not the path"


def test_a_blocked_network_tool_names_the_working_alternative():
    """A dead-end block is how the model gets stranded: it reaches for curl,
    is told only 'no', finds no way to make an HTTP call, and invents the
    response instead. curl/wget stay blocked, but the refusal must point at
    the in-sandbox tool that does the job (observed 2026-08-31: the agent
    hallucinated a whole API session rather than find fetch())."""
    from symbio.app import sandbox
    for prog in ("curl", "wget"):
        ok, out = sandbox.run_sandboxed(f"{prog} http://127.0.0.1:8000/x",
                                        _cfg(), interactive=False)
        assert ok is False and "blocked in sandbox" in out
        assert "fetch" in out and ("read_page" in out or "fetch_html" in out), (
            f"{prog} block should redirect to the working HTTP tool, got: {out!r}")


def test_a_plain_blocked_program_gets_no_network_redirect():
    """The redirect is for network tools only; rm must not sprout HTTP advice."""
    from symbio.app import sandbox
    ok, out = sandbox.run_sandboxed("/bin/rm /tmp/nothing", _cfg(), interactive=False)
    assert ok is False and "blocked in sandbox" in out
    assert "fetch" not in out and "read_page" not in out
