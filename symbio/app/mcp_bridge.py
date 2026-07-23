"""Synchronous bridge from the chat agent to the MCP local/frontier brain."""

import asyncio
from typing import Any

from symbio.mcp.config import settings
from symbio.mcp.frontier_client import run_frontier
from symbio.mcp.ollama_client import run_local


def brain_solve(prompt: str, use_frontier: bool = False) -> dict[str, Any]:
    """Call the MCP brain from synchronous chat code.

    Tries the local Ollama model first unless use_frontier is True.
    Falls back to the configured frontier model when the local brain
    fails or is unavailable.
    """

    async def _run() -> dict[str, Any]:
        if use_frontier:
            try:
                if not settings.frontier_api_key:
                    return {
                        "source": "frontier",
                        "output": "",
                        "success": False,
                        "error": "FRONTIER_API_KEY is not set in .env",
                    }
                output = await run_frontier(prompt)
                return {"source": "frontier", "output": output, "success": True}
            except Exception as exc:
                return {"source": "frontier", "output": "", "success": False, "error": str(exc)}

        try:
            output = await run_local(prompt)
            return {"source": "ollama", "output": output, "success": True}
        except Exception as local_exc:
            try:
                if not settings.frontier_api_key:
                    return {
                        "source": "frontier",
                        "output": "",
                        "success": False,
                        "error": f"Local brain failed: {local_exc}; FRONTIER_API_KEY is not set for fallback.",
                    }
                output = await run_frontier(prompt)
                return {
                    "source": "frontier",
                    "output": output,
                    "success": True,
                    "fallback": True,
                }
            except Exception as frontier_exc:
                return {
                    "source": "frontier",
                    "output": "",
                    "success": False,
                    "error": f"Local brain failed: {local_exc}; frontier fallback failed: {frontier_exc}",
                }

    try:
        return asyncio.run(_run())
    except RuntimeError as exc:
        if "cannot be called from a running event loop" in str(exc):
            loop = asyncio.get_running_loop()
            return loop.run_until_complete(_run())
        raise
