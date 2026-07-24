"""Retrain the local model's LoRA adapter from scratch after a model switch."""

from typing import Any

from symbio import constants
from symbio.app import config as app_config
from symbio.app.training import (
    build_chat_training_sample,
    digest_notes_to_training,
    remove_adapter,
    run_training,
    seed_training_data,
)
from symbio.llm import load_model_with_adapter


def retrain_model(config: dict[str, Any], *, digest: bool = True, seed: bool = True) -> bool:
    """Rebuild the LoRA adapter for the current model from accumulated data.

    This is useful when the user switches `model_name` in config.json, because
    an adapter trained for a different base model is incompatible. The old
    adapter is removed, the digest manifest is reset so notes are re-encoded
    with the new tokenizer, and LoRA training runs from scratch.
    """
    model_name = config["model_name"]
    print(f"\n  [System] Retraining adapter for {model_name}\n")

    # 1. Remove incompatible adapter.
    print("  [System] Removing old adapter...")
    remove_adapter()

    # 2. Reset digest manifest so notes get re-digested with the new tokenizer.
    if constants.DIGEST_MANIFEST.exists():
        try:
            constants.DIGEST_MANIFEST.unlink()
            print("  [System] Cleared digest manifest.")
        except OSError as exc:
            print(f"  [System warning] Could not clear digest manifest: {exc}")

    # 3. Load base model + tokenizer for the new model.
    print("  [System] Loading model and tokenizer...")
    try:
        _, tokenizer, _ = load_model_with_adapter(config, adapter_path=False)
    except Exception as exc:
        print(f"  [System] Failed to load model: {exc}")
        return False

    system_prompt = config.get("system_prompt", "") or build_default_system_prompt(config)

    # 4. Seed baseline training data if requested and training file is empty.
    if seed:
        print("  [System] Seeding baseline training data...")
        seed_training_data(tokenizer, system_prompt, config)

    # 5. Digest notes / memory / profile with the new tokenizer.
    if digest:
        print("  [System] Digesting notes and memory...")
        try:
            added = digest_notes_to_training(tokenizer, system_prompt, config)
            print(f"  [System] Digested {added} samples.")
        except Exception as exc:
            print(f"  [System warning] Note digestion failed: {exc}")

    # 6. Run full LoRA training.
    print("  [System] Starting LoRA training...")
    ok = run_training(config)
    if ok:
        print("  [System] Retrain complete.")
    else:
        print("  [System] Retrain failed.")
    return ok


def build_default_system_prompt(config: dict[str, Any]) -> str:
    """Build a minimal system prompt for training when none is configured."""
    from symbio.chat import build_system_prompt

    tools: list[dict[str, Any]] = []
    return build_system_prompt(
        config.get("assistant_name") or "Assistant",
        config.get("user_name") or "User",
        tools,
    )
