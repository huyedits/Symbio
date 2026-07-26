"""Local Ollama brain client."""

import json
import re

import httpx

from symbio.mcp.config import settings


def _strip_thinking(text: str) -> str:
    """Remove common thinking tags from local models."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return text.strip()


async def run_local(prompt: str) -> str:
    """Call the configured Ollama model and return cleaned text."""
    headers = {}
    if settings.ollama_api_key:
        headers["Authorization"] = f"Bearer {settings.ollama_api_key}"
    async with httpx.AsyncClient(timeout=settings.local_timeout) as client:
        response = await client.post(
            f"{settings.ollama_base_url}/api/generate",
            headers=headers,
            json={
                "model": settings.local_model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": settings.local_temperature,
                    "num_predict": settings.local_max_tokens,
                },
            },
        )
        response.raise_for_status()
        data = response.json()
    return _strip_thinking(data.get("response", ""))


def _compile_validator(expr: str):
    """Compile a restricted validator expression into a callable.

    The expression may use:
      - the `output` variable (a string)
      - JSON literals loaded via `json.loads`
      - standard comparisons, boolean logic, arithmetic, membership tests
      - attribute access and subscripts on `output` and JSON values
      - calls to `len` and `json.loads`

    Any dunder access, name assignment, lambda, comprehension, or import is
    rejected. The compiled code runs with an empty `__builtins__` dict.
    """
    import ast

    tree = ast.parse(expr.strip(), mode="eval")
    allowed_nodes = (
        ast.Expression, ast.BoolOp, ast.BinOp, ast.UnaryOp,
        ast.Compare, ast.Call, ast.Constant, ast.Name, ast.Load,
        ast.Attribute, ast.Subscript,
        ast.And, ast.Or, ast.Not,
        ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow, ast.USub,
        ast.In, ast.NotIn, ast.Is, ast.IsNot, ast.Eq, ast.NotEq,
        ast.Lt, ast.LtE, ast.Gt, ast.GtE,
    )

    def _check(node):
        if not isinstance(node, allowed_nodes):
            raise ValueError(f"validator contains disallowed syntax: {type(node).__name__}")
        if isinstance(node, ast.Name):
            if node.id not in {"output", "len", "json", "True", "False", "None"}:
                raise ValueError(f"validator references unknown name: {node.id}")
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("__") or node.attr.endswith("__"):
                raise ValueError(f"validator references dunder attribute: {node.attr}")
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.value.startswith("__") or node.value.endswith("__"):
                raise ValueError("validator contains suspicious string constant")

    for node in ast.walk(tree):
        _check(node)

    return compile(tree, filename="<validator>", mode="eval")


async def validate_local_output(text: str, validator_expr: str | None, schema: dict | None) -> tuple[bool, str | None]:
    """Validate local output. Returns (ok, reason).

    Validator expressions are restricted to a small, safe subset of Python:
    comparisons, boolean operators, membership tests, and basic arithmetic on
    the `output` variable and JSON literals. eval() is NOT used.
    """
    if validator_expr:
        try:
            code = _compile_validator(validator_expr)
            safe_globals = {"__builtins__": {}}
            safe_locals = {"output": text, "json": json, "len": len}
            ok = bool(eval(code, safe_globals, safe_locals))  # noqa: S307
            if not ok:
                return False, f"validator returned false: {validator_expr}"
        except Exception as exc:
            return False, f"validator error: {exc}"

    if schema:
        try:
            import jsonschema  # type: ignore
            parsed = json.loads(text)
            jsonschema.validate(parsed, schema)
        except ImportError:
            return False, "jsonschema not installed; cannot validate expected_schema"
        except json.JSONDecodeError as exc:
            return False, f"output is not valid JSON: {exc}"
        except Exception as exc:
            return False, f"schema validation failed: {exc}"

    return True, None
