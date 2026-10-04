"""Tool results in the form a Qwen3.5-family chat template recognises.

Symbio hands every tool result back to the model as a user turn:

    [System observation: <output>]
    <tool_response>{"name": ..., "content": <output>}</tool_response>

Qwen3.5 / 3.6 / 3.8 templates (Bonsai 27B among them) find "the user's request"
by walking back from the end and skipping user turns that are ONLY a tool
response — content that starts with <tool_response> and ends with
</tool_response>. Symbio's start with "[System observation:", so each one was
read as a fresh user message with no request in it. Live 2026-10-04, ternary
Bonsai 27B asked to fix a bug in scripts.py: it read the file, then answered
"What would you like me to do with this?", and after the continue-nudge, "I
don't see any mention of a 'b'".

So for a template that does that walk, an observation is rendered as just its
<tool_response> part (which already carries the whole output). Only the
rendering changes: history, logs, training data on disk and every check on the
"[System observation:" prefix see what they always saw. Training samples are
rendered through the same tokenizer, so what a model is trained on and what it
is served stay identical.
"""

from __future__ import annotations

import re
from typing import Any

_OBSERVATION = re.compile(r"^\[System observation: .*\]\s*\n(<tool_response>.*</tool_response>)\s*$",
                          re.S)


def wants_native_tool_responses(tokenizer: Any) -> bool:
    template = getattr(tokenizer, "chat_template", None)
    return (isinstance(template, str) and "<tool_response>" in template
            and "multi_step_tool" in template)


def native_tool_messages(messages: Any) -> Any:
    """`messages` with each tool-result observation reduced to its <tool_response>."""
    if not isinstance(messages, list):
        return messages
    out = []
    for message in messages:
        content = message.get("content") if isinstance(message, dict) else None
        if (isinstance(content, str) and message.get("role") == "user"
                and content.startswith("[System observation:")):
            match = _OBSERVATION.match(content)
            if match:
                message = {**message, "content": match.group(1)}
        out.append(message)
    return out


def adapt(tokenizer: Any) -> bool:
    """Render tool results natively through this tokenizer. Idempotent.

    The HF tokenizer inside mlx_lm's TokenizerWrapper is the one patched: the
    wrapper forwards attribute writes to it and calls it from its own
    apply_chat_template, so patching the wrapper would call itself forever.
    """
    inner = getattr(tokenizer, "_tokenizer", tokenizer)
    if getattr(inner, "symbio_native_tools", False) or not wants_native_tool_responses(inner):
        return False
    shipped = inner.apply_chat_template

    def apply_chat_template(conversation, *args, **kwargs):
        return shipped(native_tool_messages(conversation), *args, **kwargs)

    inner.apply_chat_template = apply_chat_template
    inner.symbio_native_tools = True
    return True


_installed = False


def install() -> bool:
    """Adapt the tokenizer of every model mlx_lm.load returns. Idempotent."""
    global _installed
    if _installed:
        return False
    import mlx_lm
    from mlx_lm import utils

    original = utils.load

    def load(*args, **kwargs):
        out = original(*args, **kwargs)
        try:
            adapt(out[1])
        except Exception:
            pass  # rendering stays as shipped; never fail a load over this
        return out

    load.__wrapped__ = original
    utils.load = load
    mlx_lm.load = load
    _installed = True
    return True
