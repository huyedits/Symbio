"""Scripts Symbio writes for itself and runs again.

A saved script is execute_code that is kept: same sandbox, same refusals,
plus a name, a line saying what it is for, and its arguments as ARGS.
"""
import pytest

from symbio.app import chat_tools, scripts

CONFIG = {"sandbox": {"blocked_commands": [], "blocked_imports": ["os", "subprocess", "sys"]},
          "agent": {"code_timeout": 30, "max_output_len": 4000}}


@pytest.fixture(autouse=True)
def scratch(tmp_path, monkeypatch):
    monkeypatch.setattr(scripts.constants, "SANDBOX_DIR", tmp_path / "sandbox")
    return tmp_path


def test_a_script_is_saved_listed_and_run_with_its_arguments():
    saved = scripts.save_script("Word Count", "print(len(' '.join(ARGS).split()))",
                                "how many words came in", CONFIG)
    assert saved["name"] == "word_count" and not saved["replaced"]
    assert scripts.list_scripts() == [{"name": "word_count",
                                       "description": "how many words came in", "lines": 1}]
    assert scripts.run_script("word_count", ["one two", "three"], CONFIG) == (True, "3")
    assert scripts.run_script("word_count", "'one two' three four", CONFIG) == (True, "4")


def test_saving_a_name_again_replaces_it():
    scripts.save_script("hello", "print('a')", "", CONFIG)
    again = scripts.save_script("hello", "print('b')", "", CONFIG)
    assert again["replaced"] and scripts.run_script("hello", [], CONFIG) == (True, "b")


def test_what_the_sandbox_would_refuse_is_refused_when_saved():
    with pytest.raises(ValueError, match="'os' is not allowed"):
        scripts.save_script("peek", "import os\nprint(os.listdir('/'))", "", CONFIG)
    assert scripts.list_scripts() == []


@pytest.mark.parametrize("bad", ["", "9lives", "../../etc/passwd" * 5])
def test_names_are_plain(bad):
    with pytest.raises(ValueError):
        scripts.normalize_name(bad)


def test_a_missing_script_names_what_there_is():
    scripts.save_script("alpha", "print(1)", "", CONFIG)
    with pytest.raises(ValueError, match="Saved scripts: alpha"):
        scripts.run_script("beta", [], CONFIG)


def test_delete():
    scripts.save_script("gone", "print(1)", "", CONFIG)
    assert scripts.delete_script("gone") == {"name": "gone"}
    assert scripts.list_scripts() == []


class Agent:
    """Not a ChatSession: the AIAgent loop's stand-in for one."""
    def __init__(self):
        self.config = CONFIG
        self.enabled_groups = None
        self.owner = None


def test_both_loops_run_the_same_tools():
    agent = Agent()
    out = chat_tools.script_for(agent, "save_script",
                                {"name": "square", "code": "print(int(ARGS[0]) ** 2)",
                                 "description": "squares a number"})
    assert "Saved script 'square'" in out and "script:square" in out
    assert "exited ok.\nOutput:\n49" in chat_tools.script_for(
        agent, "run_script", {"name": "square", "args": ["7"]})
    assert "square — squares a number" in chat_tools.script_for(agent, "list_saved_scripts", {})
    assert "exited error" in chat_tools.script_for(
        agent, "run_script", {"name": "square", "args": []})


def test_a_silent_script_is_called_silent():
    scripts.save_script("quiet", "x = 1", "", CONFIG)
    out = chat_tools.script_for(Agent(), "run_script", {"name": "quiet"})
    assert "printed NOTHING" in out


def test_scripts_are_kinds_the_user_already_has_switches_for():
    from symbio import guardrails

    assert guardrails.kind_of("run_script") == "commands"
    assert guardrails.kind_of("save_script") == "files"
    assert guardrails.kind_of("delete_script") == "files"
