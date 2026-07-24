"""Frontier model fallback client (Anthropic-first, extensible)."""

import anthropic

from symbio.mcp.config import settings


def _client() -> anthropic.AsyncAnthropic:
    if not settings.frontier_api_key:
        raise RuntimeError("FRONTIER_API_KEY is not set")
    return anthropic.AsyncAnthropic(
        api_key=settings.frontier_api_key,
        timeout=settings.frontier_timeout,
    )


async def run_frontier(prompt: str) -> str:
    """Call the configured frontier model."""
    client = _client()
    response = await client.messages.create(
        model=settings.frontier_model,
        max_tokens=settings.frontier_max_tokens,
        temperature=settings.frontier_temperature,
        messages=[{"role": "user", "content": prompt}],
    )
    content_parts = [block.text for block in response.content if block.type == "text"]
    return "\n".join(content_parts)
