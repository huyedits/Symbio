"""Automated learning loop for the MCP local brain.

When a skill_tag accumulates enough frontier-labeled examples, this module:
1. Exports the examples as chat training samples.
2. Appends them to the main Symbio training corpus.
3. Backs up the current MLX LoRA adapter.
4. Runs LoRA fine-tuning.
5. Validates the new adapter by running the local MLX model on the saved prompts.
6. Keeps the new adapter if validation passes, otherwise rolls back.
"""

import asyncio
import json
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

from symbio import constants
from symbio.app.config import load_config
from symbio.app.training import (
    backup_adapter,
    build_chat_training_sample,
    restore_adapter,
    run_training,
)
from symbio.chat import build_system_prompt
from symbio.mcp.memory import MemoryStore
from symbio.mcp.models import MemoryEntry
from symbio.mcp.ollama_client import validate_local_output


def _append_training_samples(examples: list[MemoryEntry], system_prompt: str, tokenizer) -> int:
    """Render memory examples as chat samples and append to TRAIN_FILE."""
    count = 0
    for ex in examples:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": ex.prompt},
            {"role": "assistant", "content": ex.frontier_output},
        ]
        sample = build_chat_training_sample(messages, tokenizer)
        with open(constants.TRAIN_FILE, "a", encoding="utf-8") as f:
            json.dump({"text": sample}, f)
            f.write("\n")
        count += 1
    return count


async def _validate_adapter(
    model,
    tokenizer,
    examples: list[MemoryEntry],
    system_prompt: str,
) -> tuple[bool, int, int]:
    """Validate a freshly trained adapter against the frontier-labeled examples.

    Returns (passed, pass_count, total).
    """
    from mlx_lm import generate
    from mlx_lm.sample_utils import make_sampler

    passed = 0
    total = len(examples)
    for ex in examples:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": ex.prompt},
        ]
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        try:
            output = generate(
                model,
                tokenizer,
                prompt=input_text,
                sampler=make_sampler(temp=0.2, top_p=0.9),
                max_tokens=256,
                verbose=False,
            ).strip()
        except Exception as exc:
            print(f"  [MCP Learn] Validation generation failed: {exc}")
            continue

        ok, _ = await validate_local_output(output, ex.validator, ex.expected_schema)
        if ok:
            passed += 1
    return passed >= max(1, int(total * 0.8)), passed, total


async def auto_finetune(skill_tag: str, db_path: Path | None = None) -> dict[str, Any]:
    """Run the full automated learning loop for a skill_tag.

    This is intentionally async and long-running; callers should fire-and-forget
    via asyncio.create_task so the MCP server response is not blocked.
    """
    print(f"\n  [MCP Learn] Auto-finetune triggered for skill_tag='{skill_tag}'")
    config = load_config()
    store = MemoryStore(db_path)
    examples = store.get_examples(skill_tag, limit=10_000)
    if not examples:
        return {"ok": False, "error": "no examples found"}

    system_prompt = build_system_prompt(
        config.get("assistant_name") or "Assistant",
        config.get("user_name") or "User",
        [],
    )

    # 1. Load the base model + tokenizer (without adapter so we train from the
    # current best checkpoint, not an old adapter).
    print("  [MCP Learn] Loading base model...")
    try:
        from mlx_lm import load

        model, tokenizer = load(config["model_name"])
    except Exception as exc:
        return {"ok": False, "error": f"failed to load model: {exc}"}

    # 2. Append training samples.
    count = _append_training_samples(examples, system_prompt, tokenizer)
    print(f"  [MCP Learn] Appended {count} training samples.")

    # 3. Backup current adapter.
    print("  [MCP Learn] Backing up current adapter...")
    backup_dir = backup_adapter()
    if backup_dir:
        print(f"  [MCP Learn] Backup saved to {backup_dir}")

    # 4. Run LoRA training.
    print("  [MCP Learn] Running LoRA training...")
    try:
        # Run training in a thread because mlx_lm lora is synchronous.
        loop = asyncio.get_running_loop()
        ok = await loop.run_in_executor(None, lambda: run_training(config))
    except Exception as exc:
        ok = False
        print(f"  [MCP Learn] Training error: {exc}")

    if not ok:
        if backup_dir:
            print("  [MCP Learn] Training failed; restoring adapter backup.")
            restore_adapter(backup_dir)
        return {"ok": False, "error": "training failed or was skipped"}

    # 5. Validate the new adapter.
    print("  [MCP Learn] Validating new adapter...")
    try:
        from mlx_lm import load

        new_model, new_tokenizer = load(config["model_name"], adapter_path=str(constants.ADAPTER_DIR))
        valid, passed, total = await _validate_adapter(new_model, new_tokenizer, examples, system_prompt)
    except Exception as exc:
        valid = False
        print(f"  [MCP Learn] Validation error: {exc}")

    if not valid:
        print(f"  [MCP Learn] Validation failed ({passed}/{total}). Rolling back adapter.")
        if backup_dir:
            restore_adapter(backup_dir)
        return {"ok": False, "error": f"validation failed ({passed}/{total})"}

    print(f"  [MCP Learn] Validation passed ({passed}/{total}). New adapter active.")
    # 6. Clean up backup on success.
    if backup_dir:
        try:
            shutil.rmtree(backup_dir, ignore_errors=True)
        except Exception:
            pass

    return {"ok": True, "samples": count, "validated": f"{passed}/{total}"}
