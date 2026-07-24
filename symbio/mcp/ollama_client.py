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


async def validate_local_output(text: str, validator_expr: str | None, schema: dict | None) -> tuple[bool, str | None]:
    """Validate local output. Returns (ok, reason)."""
    if validator_expr:
        try:
            safe_globals = {"__builtins__": {"json": json, "len": len, "str": str, "int": int, "float": float}}
            safe_locals = {"output": text}
            ok = bool(eval(validator_expr, safe_globals, safe_locals))  # noqa: S307
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
