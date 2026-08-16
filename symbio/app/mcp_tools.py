"""Build, discover, and execute user-generated MCP tools.

MCP modules live under mcp_modules/<name>/:
  - manifest.json  (name, description, JSON schema)
  - tool.py        (the runnable Python implementation)

Discovered tools are injected into the main agent tool registry so the model
can call them with Hermes-style <tool_call> JSON. Each generated tool runs in a
sandboxed subprocess with restricted imports.
"""

import json
import os
import re
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

from symbio import constants
from symbio.app import tooling, training
from symbio.app.sandbox import _is_code_safe

MCP_MODULE_DIR = constants.PROJECT_DIR / "mcp_modules"
MCP_GROUP = "mcp"


def _module_dir(name: str) -> Path:
    safe = re.sub(r"[^a-z0-9_]+", "_", name.lower()).strip("_") or "mcp_tool"
    return MCP_MODULE_DIR / safe


def list_mcp_tools() -> list[dict[str, Any]]:
    """Return manifests for every generated MCP tool."""
    tools: list[dict[str, Any]] = []
    if not MCP_MODULE_DIR.exists():
        return tools
    for d in sorted(MCP_MODULE_DIR.iterdir()):
        if not d.is_dir():
            continue
        manifest = d / "manifest.json"
        if not manifest.exists():
            continue
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
            data["_module_dir"] = str(d)
            tools.append(data)
        except Exception:
            continue
    return tools


def discover_mcp_tools() -> list[dict[str, Any]]:
    """Return Hermes-style tool schemas for all discovered MCP modules."""
    schemas: list[dict[str, Any]] = []
    for data in list_mcp_tools():
        schema = data.get("schema")
        if not schema:
            continue
        schema.setdefault("name", f"mcp_{data['name']}")
        schemas.append(schema)
    return schemas


def _tool_module_path(tool_name: str) -> Path | None:
    d = _module_dir(tool_name)
    tool_file = d / "tool.py"
    return tool_file if tool_file.exists() else None


def execute_mcp_tool(tool_name: str, params: dict[str, Any], config: dict[str, Any]) -> tuple[bool, str]:
    """Run a generated MCP tool in a sandboxed subprocess.

    Returns (ok, output_or_error).
    """
    tool_file = _tool_module_path(tool_name)
    if tool_file is None:
        return False, f"MCP tool '{tool_name}' not found."

    try:
        source = tool_file.read_text(encoding="utf-8")
    except Exception as exc:
        return False, f"Could not read {tool_file.name}: {exc}"

    safe, msg = _is_code_safe(source, set(config["sandbox"].get("blocked_imports", [])))
    if not safe:
        return False, f"MCP tool '{tool_name}' failed safety check: {msg}"

    runner = f"""\
import json, sys, os
sys.path.insert(0, {str(tool_file.parent)!r})
from tool import run
with open({str(tool_file.parent / "params.json")!r}, "r") as f:
    params = json.load(f)
try:
    result = run(params)
except Exception as exc:
    print(json.dumps({{"_mcp_error": repr(exc)}}), file=sys.stderr)
    sys.exit(1)
print(json.dumps({{"_mcp_result": result}}, ensure_ascii=False))
"""

    work_dir = _module_dir(tool_name)
    work_dir.mkdir(parents=True, exist_ok=True)
    params_file = work_dir / "params.json"
    params_file.write_text(json.dumps(params, ensure_ascii=False), encoding="utf-8")

    fd, runner_path = tempfile.mkstemp(suffix=".py", dir=str(constants.SANDBOX_DIR), prefix="mcp_runner_")
    with os.fdopen(fd, "w") as f:
        f.write(runner)

    try:
        result = subprocess.run(
            [sys.executable, runner_path],
            capture_output=True,
            text=True,
            timeout=config["agent"].get("code_timeout", 300),
            cwd=str(constants.SANDBOX_DIR),
            env={"PATH": os.environ.get("PATH", "")},
        )
        if result.returncode != 0:
            err = result.stderr.strip() or result.stdout.strip() or "unknown error"
            return False, f"MCP tool '{tool_name}' failed: {err}"
        try:
            parsed = json.loads(result.stdout.strip())
            if isinstance(parsed, dict) and "_mcp_result" in parsed:
                out = json.dumps(parsed["_mcp_result"], ensure_ascii=False, indent=2)
            else:
                out = result.stdout.strip()
        except json.JSONDecodeError:
            out = result.stdout.strip()
        max_len = config["agent"].get("max_output_len", 4000)
        if len(out) > max_len:
            out = out[:max_len] + "\n... (truncated)"
        return True, out
    except subprocess.TimeoutExpired:
        return False, f"MCP tool '{tool_name}' timed out after {config['agent'].get('code_timeout', 300)}s."
    except Exception as exc:
        return False, f"MCP tool '{tool_name}' execution error: {exc}"
    finally:
        try:
            os.unlink(runner_path)
        except OSError:
            pass


_DEFAULT_TOOL_TEMPLATE = '''"""MCP tool: {name}

{description}
"""

from typing import Any


def run(params: dict[str, Any]) -> Any:
    """{description}"""
{body}
'''


def _default_body(schema: dict[str, Any]) -> str:
    """Fallback implementation when no model is available.

    Lines are returned unindented; build_mcp_tool wraps them inside run(params).
    """
    required = schema.get("parameters", {}).get("required", [])
    if required:
        lines = ["# Stub implementation for a generated MCP tool.", "return {"]
        for key in required:
            lines.append(f'    {key!r}: params.get({key!r}, ""),')
        lines.append("}")
        return "\n".join(lines)
    return "# Stub implementation for a generated MCP tool.\nreturn {}"


def _generate_schema_and_body(name: str, description: str,
                              model=None, tokenizer=None,
                              generate_fn=None, config: dict[str, Any] | None = None) -> tuple[dict[str, Any], str]:
    """Use the loaded model to generate a JSON schema and Python body.

    Falls back to a stub if no model/tokenizer/generate_fn is provided.
    """
    if model is None or tokenizer is None or generate_fn is None:
        schema = {
            "name": f"mcp_{name}",
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {
                    "input": {
                        "type": "string",
                        "description": "Primary input for the tool.",
                    }
                },
                "required": ["input"],
            },
        }
        return schema, _default_body(schema)

    system = (
        "You are a tool-generation assistant. Given a tool name and description, "
        "produce a JSON object with two keys: 'schema' and 'body'.\n\n"
        "The 'schema' must be a valid Hermes-style tool schema with name, description, "
        "and parameters (type: object, properties, required).\n\n"
        "The 'body' must be the body lines of a Python function `run(params: dict[str, Any]) -> Any`. "
        "Do not include the function signature or docstring. Lines should not be indented; "
        "they will be wrapped inside run(params) automatically. "
        "Use only standard library imports. No network, filesystem, or shell access. "
        "Return the Python body as a plain string (not a JSON string).\n\n"
        "Respond ONLY with the JSON object."
    )
    prompt = f"Name: {name}\nDescription: {description}\n\nGenerate schema and body."
    messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=training.THINKING_ENABLED,
        )
    else:
        text = f"{system}\n\nUser: {prompt}\nAssistant:"

    output = generate_fn(model, tokenizer, prompt=text, max_tokens=1024, verbose=False)
    try:
        # The model may reason before emitting the JSON; drop the thinking
        # block so json.loads sees only the object.
        parsed = json.loads(tooling.strip_reasoning_block(output).strip())
        schema = parsed.get("schema", {})
        body = parsed.get("body", _default_body(schema))
        # Validate schema basics.
        schema.setdefault("name", f"mcp_{name}")
        schema.setdefault("description", description)
        schema.setdefault("parameters", {"type": "object", "properties": {}, "required": []})
        return schema, body
    except Exception:
        # If generation didn't return valid JSON, build a safe stub.
        schema = {
            "name": f"mcp_{name}",
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {"input": {"type": "string", "description": "Primary input."}},
                "required": ["input"],
            },
        }
        return schema, _default_body(schema)


def build_mcp_tool(name: str,
                   description: str,
                   model=None,
                   tokenizer=None,
                   generate_fn=None,
                   config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Generate a new MCP tool module and manifest.

    Returns a result dict with name, module_dir, schema, and success status.
    """
    d = _module_dir(name)
    d.mkdir(parents=True, exist_ok=True)

    schema, body = _generate_schema_and_body(
        name, description, model=model, tokenizer=tokenizer, generate_fn=generate_fn, config=config
    )

    # Indent body to sit inside `run(params)`.
    indented_body = "\n".join(
        ("    " + line if line.strip() else line) for line in body.splitlines()
    )
    tool_source = _DEFAULT_TOOL_TEMPLATE.format(
        name=name,
        description=description,
        body=indented_body,
    )

    (d / "tool.py").write_text(tool_source, encoding="utf-8")

    manifest = {
        "name": name,
        "description": description,
        "schema": schema,
        "group": MCP_GROUP,
        "created": datetime.now().isoformat(),
    }
    (d / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # Basic smoke test: compile the generated code.
    try:
        compile(tool_source, str(d / "tool.py"), "exec")
        smoke_ok = True
        smoke_error = None
    except SyntaxError as exc:
        smoke_ok = False
        smoke_error = f"Generated tool has a syntax error: {exc}"

    try:
        display_path = d.relative_to(constants.PROJECT_DIR)
    except ValueError:
        display_path = d
    return {
        "name": name,
        "tool_name": schema.get("name", f"mcp_{name}"),
        "module_dir": str(d),
        "schema": schema,
        "smoke_ok": smoke_ok,
        "smoke_error": smoke_error,
        "message": f"MCP tool '{name}' generated at {display_path}."
    }
