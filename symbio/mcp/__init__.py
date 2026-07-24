"""MCP server: local brain with frontier fallback and learning loop."""

__version__ = "0.1.0"

from symbio.mcp.memory import MemoryStore
from symbio.mcp.models import MemoryEntry, SolveRequest, SolveResult
from symbio.mcp.learn import auto_finetune
from symbio.mcp.server import brain_solve, export_finetune_data, memory_stats, mcp, trigger_finetune

__all__ = [
    "MemoryStore",
    "MemoryEntry",
    "SolveRequest",
    "SolveResult",
    "auto_finetune",
    "brain_solve",
    "export_finetune_data",
    "memory_stats",
    "mcp",
    "trigger_finetune",
]
