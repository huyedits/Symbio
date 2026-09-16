"""Tests for the MCP tool builder and registry integration."""

import json
from pathlib import Path

import pytest

from symbio.app import mcp_tools, tooling


@pytest.fixture
def fake_mcp_dir(tmp_path, monkeypatch):
    """Redirect MCP modules and sandbox dirs into a temp path."""
    monkeypatch.setattr(mcp_tools, "MCP_MODULE_DIR", tmp_path / "mcp_modules")
    monkeypatch.setattr(tooling, "_TOOLS", list(tooling._TOOLS))
    monkeypatch.setattr(tooling, "_TOOL_GROUPS", dict(tooling._TOOL_GROUPS))
    return tmp_path / "mcp_modules"


def test_list_mcp_tools_empty(fake_mcp_dir):
    assert mcp_tools.list_mcp_tools() == []
    assert mcp_tools.discover_mcp_tools() == []


def test_build_mcp_tool_stub_no_model(fake_mcp_dir):
    result = mcp_tools.build_mcp_tool(
        "test_greet",
        "Greet a user by name",
        model=None,
        tokenizer=None,
        generate_fn=None,
        config={"agent": {"code_timeout": 5}, "sandbox": {"blocked_imports": []}},
    )
    assert result["smoke_ok"] is True
    assert result["tool_name"] == "mcp_test_greet"

    module_dir = fake_mcp_dir / "test_greet"
    assert module_dir.exists()
    assert (module_dir / "tool.py").exists()
    manifest = json.loads((module_dir / "manifest.json").read_text())
    assert manifest["name"] == "test_greet"
    assert manifest["group"] == "mcp"
    assert manifest["schema"]["name"] == "mcp_test_greet"


def test_execute_mcp_tool(fake_mcp_dir):
    config = {"agent": {"code_timeout": 5, "max_output_len": 4000}, "sandbox": {"blocked_imports": []}}
    mcp_tools.build_mcp_tool(
        "double_it",
        "Return twice the input number",
        model=None,
        tokenizer=None,
        generate_fn=None,
        config=config,
    )

    ok, out = mcp_tools.execute_mcp_tool("double_it", {"input": "21"}, config)
    assert ok is True
    assert "42" in out or "21" in out


def test_refresh_mcp_tools_registers_tool(fake_mcp_dir):
    config = {"agent": {"code_timeout": 5}, "sandbox": {"blocked_imports": []}}
    mcp_tools.build_mcp_tool(
        "my_tool",
        "A sample tool",
        model=None,
        tokenizer=None,
        generate_fn=None,
        config=config,
    )

    before = {t["name"] for t in tooling.tool_schemas()}
    assert "mcp_my_tool" not in before

    added = tooling.refresh_mcp_tools()
    assert any(t["name"] == "mcp_my_tool" for t in added)

    after = {t["name"] for t in tooling.tool_schemas()}
    assert "mcp_my_tool" in after
    assert tooling.tool_group("mcp_my_tool") == "mcp"


def test_refresh_mcp_tools_is_idempotent(fake_mcp_dir):
    config = {"agent": {"code_timeout": 5}, "sandbox": {"blocked_imports": []}}
    mcp_tools.build_mcp_tool(
        "idempotent",
        "Idempotent test tool",
        model=None,
        tokenizer=None,
        generate_fn=None,
        config=config,
    )

    first = tooling.refresh_mcp_tools()
    second = tooling.refresh_mcp_tools()
    assert len(first) == 1
    assert len(second) == 0

    schemas = tooling.tool_schemas()
    assert sum(1 for t in schemas if t["name"] == "mcp_idempotent") == 1


def test_tool_group_mcp_prefix():
    assert tooling.tool_group("mcp_whatever") == "mcp"
