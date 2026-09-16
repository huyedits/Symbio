"""Every tool name the prompt teaches must be a tool that exists.

There are two registries in this project — the chat stack's (symbio/app/
tooling.py `_TOOLS`, plus the Hermes alias table) and the agent stack's
(symbio/tools.py `build_tool_registry`) — and ONE prompt, one set of worked
examples, shared between them. Before this test the shared prompt named six
tools the agent stack cannot execute and the few-shots named one the chat stack
cannot: a syntactically perfect call, dropped by the dispatcher, with the
model's own prose beside it claiming it had worked.

The cost is not the dropped call. It is that a worked example of a name that
does not resolve teaches the model that tool names are free text — that the
shape matters and the spelling is negotiable — and a model that believes that
invents a plausible name every time it wants something it has not been shown.

So: every `{"name": ...}` in the prompt and in the few-shots resolves in BOTH
registries, or this fails.
"""

import json
import re
from pathlib import Path

import pytest

from symbio.app import prompts, tooling
from symbio.tools import tool_few_shots

# A tool name inside a worked example, in either spelling the prompt uses:
# real JSON ({"name": "x"}) and the doubled braces of a .format template.
_NAME_IN_EXAMPLE = re.compile(r'\{?\{?\s*"name"\s*:\s*"([a-z_][a-z0-9_]*)"')


def _chat_stack_names() -> set[str]:
    """Names the chat stack resolves: the catalog plus every Hermes alias."""
    return ({t["name"] for t in tooling._TOOLS}
            | set(tooling._HERMES_NAME_MAP)
            | set(tooling._HERMES_NAME_MAP.values()))


def _agent_stack_names() -> set[str]:
    """Names the agent stack resolves.

    Read out of the source rather than by building a registry, because
    build_tool_registry needs a live AIAgent (a model, a session store, a
    retriever) and this is a question about a literal list.
    """
    src = Path("symbio/tools.py").read_text(encoding="utf-8")
    body = src[src.index("def build_tool_registry"):src.index("def openai_tool_schemas")]
    return set(re.findall(r'"name":\s*"([a-z_][a-z0-9_]*)"', body))


def _examples_in(text: str) -> set[str]:
    return set(_NAME_IN_EXAMPLE.findall(text))


PROMPT_SOURCES = {
    "DEFAULT_SYSTEM_PROMPT": prompts.DEFAULT_SYSTEM_PROMPT,
    "prompt.md": (Path("prompt.md").read_text(encoding="utf-8")
                  if Path("prompt.md").exists() else ""),
}


@pytest.mark.parametrize("label", sorted(PROMPT_SOURCES))
def test_every_prompt_example_resolves_in_both_stacks(label):
    names = _examples_in(PROMPT_SOURCES[label])
    # "tool_name" is the placeholder in the syntax line, not a claim that a
    # tool is called that.
    names.discard("tool_name")
    chat, agent = _chat_stack_names(), _agent_stack_names()
    unresolved = {
        n: [s for s, known in (("chat", chat), ("agent", agent)) if n not in known]
        for n in sorted(names)
    }
    unresolved = {n: where for n, where in unresolved.items() if where}
    assert not unresolved, (
        f"{label} teaches tool names that do not resolve: {unresolved}. "
        "Either rename the example to a tool that exists in both registries, "
        "add the alias, or drop the example.")


def test_every_few_shot_example_resolves_in_both_stacks():
    """The few-shots are shared verbatim by both stacks — see symbio/tools.py."""
    config = json.loads(Path("config.json").read_text(encoding="utf-8"))
    chat, agent = _chat_stack_names(), _agent_stack_names()
    # Every rotation, not just the default: a family only shown for file work
    # is exactly the one nobody would notice had gone stale.
    from symbio.tools import FEW_SHOT_FAMILIES

    for family in (None, *FEW_SHOT_FAMILIES):
        joined = " ".join(m["content"] for m in tool_few_shots(config, family=family))
        for name in sorted(_examples_in(joined)):
            assert name in chat, f"{family}: {name} is not a chat-stack tool"
            assert name in agent, f"{family}: {name} is not an agent-stack tool"


def test_the_index_catalog_names_only_real_tools():
    """The index replaces schemas with names, so the names carry all the
    weight: a name in it that resolves to nothing is a tool the model will
    call and never reach."""
    block = tooling.build_tools_block(None, mode="index")
    listed: set[str] = set()
    for line in block.splitlines():
        if re.fullmatch(r" {4}[a-z_]+(, [a-z_]+)*", line):
            listed |= {n.strip() for n in line.split(",")}
    assert listed, "the index listed no tools at all"
    known = _chat_stack_names()
    assert not (listed - known), f"index lists unknown tools: {sorted(listed - known)}"


def test_the_index_tells_the_model_how_to_get_the_rest():
    """An index without the lookup is just a shorter prompt with less in it."""
    block = tooling.build_tools_block(None, mode="index")
    assert "tool_docs" in block
    assert any(t["name"] == "tool_docs" for t in tooling._TOOLS)


# ---------------------------------------------------- tools/*.md as the source

def test_a_tool_file_dropped_in_is_advertised_without_a_code_change(tmp_path,
                                                                    monkeypatch):
    """The point of the per-tool files: adding a tool is a file, not a list in
    Python. It still needs a dispatcher to RUN — but describing it, grouping
    it and advertising it are the parts that used to need four edits."""
    from symbio import constants
    from symbio.app import tool_docs

    directory = tmp_path / "tools"
    directory.mkdir()
    (directory / "check_tide.md").write_text(
        "---\nname: check_tide\nfamily: web\n---\n\n"
        "Read the tide table for a harbour.\n\n"
        '```json\n{"type":"object","properties":{"harbour":{"type":"string"}}}\n```\n',
        encoding="utf-8")
    monkeypatch.setattr(constants, "TOOLS_DIR", directory)
    monkeypatch.setattr(tooling, "_disk_signature", None)
    try:
        tooling.sync_tool_files(seed=False)
        assert any(t["name"] == "check_tide" for t in tooling._TOOLS)
        assert tooling.tool_family("check_tide") == "web"
        assert "check_tide" in tooling.build_tools_block(None, mode="index")
        assert "tide table" in tool_docs.docs_for(
            tooling.tool_schemas(), tooling.tool_family, names="check_tide")
    finally:
        # The catalog is module state; leaving a test's tool in it would leak
        # into every later test in the same process.
        tooling._TOOLS[:] = [t for t in tooling._TOOLS if t["name"] != "check_tide"]
        tooling._TOOL_GROUPS.pop("check_tide", None)
        tooling._TOOL_FAMILIES.pop("check_tide", None)
        tooling._disk_signature = None


def test_a_broken_tool_file_costs_only_itself(tmp_path, monkeypatch):
    from symbio import constants

    directory = tmp_path / "tools"
    directory.mkdir()
    (directory / "good_one.md").write_text(
        "---\nname: good_one\nfamily: web\n---\n\nFine.\n\n"
        '```json\n{"type":"object"}\n```\n', encoding="utf-8")
    (directory / "broken.md").write_text(
        "---\nname: broken\n---\n\nBad.\n\n```json\n{not json at all\n```\n",
        encoding="utf-8")
    monkeypatch.setattr(constants, "TOOLS_DIR", directory)
    monkeypatch.setattr(tooling, "_disk_signature", None)
    try:
        tooling.sync_tool_files(seed=False)
        names = {t["name"] for t in tooling._TOOLS}
        assert "good_one" in names
        assert "broken" not in names
    finally:
        tooling._TOOLS[:] = [t for t in tooling._TOOLS
                             if t["name"] not in ("good_one", "broken")]
        for n in ("good_one", "broken"):
            tooling._TOOL_GROUPS.pop(n, None)
            tooling._TOOL_FAMILIES.pop(n, None)
        tooling._disk_signature = None


# ------------------------------------------- a file written earlier is not an edit

def _stale_file(tmp_path, monkeypatch, body: str, name: str = "browser_type"):
    """Point the catalog at a directory holding one out-of-date tool file."""
    from symbio import constants

    directory = tmp_path / "tools"
    directory.mkdir()
    (directory / f"{name}.md").write_text(body, encoding="utf-8")
    monkeypatch.setattr(constants, "TOOLS_DIR", directory)
    monkeypatch.setattr(tooling, "_disk_signature", None)


def _restore(name: str):
    """Put the built-in back: sync mutates the catalog entry in place."""
    import copy

    built_in = copy.deepcopy(tooling._BUILTIN_BY_NAME[name])
    for spec in tooling._TOOLS:
        if spec["name"] == name:
            spec.clear()
            spec.update(built_in)
    tooling._disk_signature = None


def test_a_file_written_before_an_argument_existed_still_advertises_it(
        tmp_path, monkeypatch):
    """2026-09-16, live: tools/desktop_click.md was seeded on the 14th with
    {x, y}, and the accessibility work landed on the 16th. The file replaced
    the schema wholesale, so `element` — the numbered control ax.py exists to
    provide — could not be named in a tool call at all. The capability was
    shipped, tested, and unreachable, and nothing said so."""
    _stale_file(tmp_path, monkeypatch,
                "---\nname: browser_type\nfamily: browser\ngroup: browser\n---\n\n"
                "Type text into the page.\n\n"
                '```json\n{"type":"object","properties":'
                '{"text":{"type":"string"}},"required":["text"]}\n```\n')
    try:
        tooling.sync_tool_files(seed=False)
        live = next(t for t in tooling._TOOLS if t["name"] == "browser_type")
        properties = live["parameters"]["properties"]

        assert "text" in properties
        assert "selector" in properties, "the file cannot un-declare an argument"
        assert "enter" in properties
    finally:
        _restore("browser_type")


def test_the_file_still_wins_on_wording(tmp_path, monkeypatch):
    """Restoring the arguments must not undo the edit the file was made for."""
    _stale_file(tmp_path, monkeypatch,
                "---\nname: browser_type\nfamily: browser\ngroup: browser\n---\n\n"
                "Type into the composer, never the page body.\n\n"
                '```json\n{"type":"object","properties":'
                '{"text":{"type":"string"}},"required":["text"]}\n```\n')
    try:
        tooling.sync_tool_files(seed=False)
        live = next(t for t in tooling._TOOLS if t["name"] == "browser_type")

        assert live["description"] == "Type into the composer, never the page body."
    finally:
        _restore("browser_type")


def test_a_file_may_relax_a_requirement_but_never_add_one(tmp_path, monkeypatch):
    """desktop_click's stale file required {x, y} from when coordinates were
    the only way to click. Restoring `element` beside a required x would have
    advertised a control number the model was not allowed to send alone."""
    _stale_file(tmp_path, monkeypatch,
                "---\nname: browser_type\nfamily: browser\ngroup: browser\n---\n\n"
                "Type text into the page.\n\n"
                '```json\n{"type":"object","properties":'
                '{"text":{"type":"string"},"selector":{"type":"string"}},'
                '"required":["text","selector"]}\n```\n')
    try:
        tooling.sync_tool_files(seed=False)
        live = next(t for t in tooling._TOOLS if t["name"] == "browser_type")

        assert live["parameters"]["required"] == ["text"]
    finally:
        _restore("browser_type")


def test_a_tool_file_with_no_built_in_is_left_exactly_as_written(tmp_path,
                                                                 monkeypatch):
    """The merge is about files that have fallen behind code. A tool that only
    exists as a file has nothing to fall behind."""
    _stale_file(tmp_path, monkeypatch,
                "---\nname: check_tide\nfamily: web\n---\n\n"
                "Read the tide table.\n\n"
                '```json\n{"type":"object","properties":'
                '{"harbour":{"type":"string"}},"required":["harbour"]}\n```\n',
                name="check_tide")
    try:
        tooling.sync_tool_files(seed=False)
        live = next(t for t in tooling._TOOLS if t["name"] == "check_tide")

        assert live["parameters"]["required"] == ["harbour"]
        assert set(live["parameters"]["properties"]) == {"harbour"}
    finally:
        tooling._TOOLS[:] = [t for t in tooling._TOOLS if t["name"] != "check_tide"]
        tooling._TOOL_GROUPS.pop("check_tide", None)
        tooling._TOOL_FAMILIES.pop("check_tide", None)
        tooling._disk_signature = None
