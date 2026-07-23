"""FastMCP server exposing `brain_solve`."""

from fastmcp import FastMCP

import asyncio

from fastmcp import FastMCP

from symbio.mcp.config import settings
from symbio.mcp.frontier_client import run_frontier
from symbio.mcp.memory import MemoryStore
from symbio.mcp.models import MemoryEntry, SolveRequest, SolveResult
from symbio.mcp.ollama_client import run_local, validate_local_output

mcp = FastMCP("local_brain_mcp")
store = MemoryStore()


@mcp.tool()
async def brain_solve(request: SolveRequest) -> SolveResult:
    """
    Try the local Ollama brain first; fall back to a frontier model if it fails.
    Save the example and bump the miss counter for the skill tag.
    """
    local_output: str | None = None
    failure_reason: str | None = None
    frontier_output: str | None = None

    # 1. Local attempt
    if request.use_frontier is True:
        failure_reason = "frontier use forced by caller"
    else:
        try:
            local_output = await run_local(request.prompt)
            ok, failure_reason = await validate_local_output(
                local_output,
                request.validator,
                request.expected_schema,
            )
            if ok:
                return SolveResult(
                    source="local",
                    output=local_output,
                    success=True,
                    frontier_fallback_used=False,
                )
        except Exception as exc:
            failure_reason = f"local brain error: {exc}"

    # 2. Frontier fallback
    try:
        frontier_output = await run_frontier(request.prompt)
    except Exception as exc:
        return SolveResult(
            source="frontier",
            output=local_output or "",
            success=False,
            reason=f"{failure_reason}; frontier fallback also failed: {exc}",
            frontier_fallback_used=True,
        )

    # 3. Save the teaching example
    store.save(
        MemoryEntry(
            skill_tag=request.skill_tag,
            prompt=request.prompt,
            local_output=local_output,
            frontier_output=frontier_output,
            failure_reason=failure_reason,
            validator=request.validator,
            expected_schema=request.expected_schema,
        )
    )
    miss_count = store.count_misses(request.skill_tag)
    finetune_ready = miss_count >= settings.miss_threshold

    if finetune_ready and settings.auto_finetune:
        try:
            from symbio.mcp.learn import auto_finetune

            asyncio.create_task(auto_finetune(request.skill_tag, store.db_path))
        except Exception as exc:
            print(f"[MCP Server] Could not schedule auto_finetune: {exc}")

    return SolveResult(
        source="frontier",
        output=frontier_output,
        success=True,
        reason=failure_reason,
        frontier_fallback_used=True,
        saved_to_memory=True,
        miss_count=miss_count,
        finetune_ready=finetune_ready,
    )


@mcp.tool()
async def memory_stats(skill_tag: str | None = None) -> dict:
    """Return how many frontier examples are stored, optionally filtered by skill_tag."""
    if skill_tag:
        return {"skill_tag": skill_tag, "miss_count": store.count_misses(skill_tag)}
    return {"skill_tag": "all", "miss_count": store.count_misses()}


@mcp.tool()
async def export_finetune_data(skill_tag: str, output_path: str) -> dict:
    """Export stored examples for a skill_tag as OpenAI-style chat JSONL for fine-tuning."""
    from pathlib import Path

    count = store.export_jsonl(skill_tag, Path(output_path))
    return {"skill_tag": skill_tag, "examples": count, "path": output_path}


@mcp.tool()
async def trigger_finetune(skill_tag: str) -> dict:
    """Manually run the automated LoRA finetune loop for a skill tag."""
    from symbio.mcp.learn import auto_finetune

    result = await auto_finetune(skill_tag, store.db_path)
    return {"skill_tag": skill_tag, **result}


if __name__ == "__main__":
    mcp.run()
