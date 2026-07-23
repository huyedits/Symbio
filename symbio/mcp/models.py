"""Shared Pydantic models."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class SolveRequest(BaseModel):
    """Request to solve a task."""

    prompt: str = Field(..., description="Full prompt or task description.")
    skill_tag: str = Field(default="general", description="Category used to group training examples.")
    validator: str | None = Field(
        default=None,
        description="Optional Python expression to validate local output. "
                    "Use `output` to refer to the text. Example: 'json.loads(output)'",
    )
    expected_schema: dict[str, Any] | None = Field(
        default=None,
        description="Optional JSON schema the local output should satisfy.",
    )
    use_frontier: bool | None = Field(
        default=None,
        description="Force frontier use. If None, local brain is attempted first.",
    )


class SolveResult(BaseModel):
    """Result of a solve attempt."""

    source: str = Field(..., description="'local' or 'frontier'")
    output: str = Field(..., description="Generated solution / answer.")
    success: bool = Field(..., description="Whether the result passed validation.")
    reason: str | None = Field(default=None, description="Why local failed, if it failed.")
    frontier_fallback_used: bool = Field(default=False)
    saved_to_memory: bool = Field(default=False)
    finetune_ready: bool = Field(default=False)
    miss_count: int = Field(default=0)


class MemoryEntry(BaseModel):
    """One saved learning example."""

    id: int | None = None
    created_at: datetime | None = None
    skill_tag: str
    prompt: str
    local_output: str | None
    frontier_output: str
    failure_reason: str | None
    validator: str | None = None
    expected_schema: dict[str, Any] | None = None
