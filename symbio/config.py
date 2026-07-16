"""Configuration loading, saving, and model preset helpers for Symbio."""

import json
import re
from pathlib import Path
from typing import Any

from symbio.constants import ADAPTER_DIR, CONFIG_FILE, DEFAULT_CONFIG, MODELS_FILE, NOTES_DIR


def load_config() -> dict[str, Any]:
    """Load config.json if present; merge with sensible defaults."""
    config = DEFAULT_CONFIG.copy()
    if CONFIG_FILE.exists():
        try:
            user_config = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
            config.update(user_config)
            for nested_key in ("lora", "agent", "model", "rag", "training_planner"):
                if nested_key in user_config and isinstance(user_config[nested_key], dict):
                    config[nested_key] = {**DEFAULT_CONFIG.get(nested_key, {}), **user_config[nested_key]}
        except Exception as e:
            print(f"[Config warning] Could not read {CONFIG_FILE}: {e}")
    return config


def save_config(config: dict[str, Any]):
    """Persist the merged config back to config.json."""
    try:
        CONFIG_FILE.write_text(json.dumps(config, indent=2), encoding="utf-8")
    except Exception as e:
        print(f"[Config warning] Could not write {CONFIG_FILE}: {e}")


def detect_model_type(model: Any) -> str:
    """Return 'moe' if the loaded MLX model uses Mixture-of-Experts, else 'dense'."""
    # Inspect config if available.
    config = getattr(model, "config", None)
    if isinstance(config, dict):
        arch = config.get("architectures", [])
        arch_str = " ".join(arch) if isinstance(arch, list) else str(arch)
        if "moe" in arch_str.lower():
            return "moe"
        moe_keys = (
            "num_experts", "num_local_experts", "num_shared_experts",
            "num_experts_per_tok", "moe_intermediate_size", "n_routed_experts",
        )
        if any(k in config for k in moe_keys):
            return "moe"
    # Walk the model graph looking for expert/router structures.
    moe_attrs = {"experts", "moe", "router", "gate", "routed_experts", "shared_experts"}
    visited: set[int] = set()
    queue = [model]
    while queue:
        obj = queue.pop()
        obj_id = id(obj)
        if obj_id in visited:
            continue
        visited.add(obj_id)
        if any(hasattr(obj, attr) for attr in moe_attrs):
            # Make sure it is not a false positive: verify the attribute is a container/list/module.
            for attr in moe_attrs:
                val = getattr(obj, attr, None)
                if val is not None and not isinstance(val, (int, float, bool, str, type(None))):
                    return "moe"
        # Queue child modules/arrays. Avoid expanding large tensors or leaf arrays.
        if hasattr(obj, "__dict__"):
            for child in obj.__dict__.values():
                if hasattr(child, "__dict__") or isinstance(child, (list, tuple)):
                    queue.append(child)
        if isinstance(obj, (list, tuple)):
            queue.extend(obj)
    return "dense"


def _adapter_matches_model(config: dict[str, Any]) -> bool:
    """Check whether the saved adapter was trained for the current model_name."""
    adapter_config = ADAPTER_DIR / "adapter_config.json"
    if not adapter_config.exists():
        return True  # no adapter present is fine
    try:
        adapter_cfg = json.loads(adapter_config.read_text(encoding="utf-8"))
        saved_model = adapter_cfg.get("model", "")
        return not saved_model or saved_model == config["model_name"]
    except Exception:
        return False


def can_run_lora(config: dict[str, Any], model_type: str) -> tuple[bool, str]:
    """Return (ok, reason) for running LoRA on the current model/policy."""
    model_cfg = config.get("model", {})
    if not model_cfg.get("allow_lora", True):
        return False, "LoRA is disabled in config (model.allow_lora=false)."
    if model_type == "moe" and not model_cfg.get("allow_moe_lora", False):
        mode = model_cfg.get("moe_fine_tuning_mode", "rag_only")
        return False, f"MoE model detected: LoRA disabled. Fine-tuning mode is '{mode}'."
    return True, ""


def _write_identity_notes(assistant_name: str, user_name: str):
    """Persist identity notes for the agent to reference."""
    for pattern in ("*_My_Identity.md", "*_User_Identity.md"):
        for old in NOTES_DIR.glob(pattern):
            old.unlink()
    (NOTES_DIR / "My_Identity.md").write_text(
        f"# My Identity\n\nI am {assistant_name}, a helpful personal AI assistant.\n",
        encoding="utf-8",
    )
    (NOTES_DIR / "User_Identity.md").write_text(
        f"# User Identity\n\nMy user's name is {user_name}.\n",
        encoding="utf-8",
    )


def _extract_name(text: str, patterns: list[str]) -> str | None:
    """Return the first captured group matching one of the regex patterns.

    Strips trailing conjunctions and noise phrases so combined sentences like
    "My name is Alice and your name is HAL" don't leak into the wrong name.
    Also rejects obvious negations and non-name placeholders.
    """
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            raw = match.group(1).strip()
            # Stop at conjunctions / phrase markers.
            name = re.split(
                r"\b(?:and|or|but|,|from\s+now\s+on)\b",
                raw,
                flags=re.IGNORECASE,
            )[0].strip()
            name = name.strip(".!?,'\"")
            # Reject obviously non-name spans and negations.
            if (
                name
                and len(name) <= 40
                and not name.isdigit()
                and not re.match(r"^(not|no|never|nothing|anything|whatever|nobody)\b", name, re.IGNORECASE)
            ):
                return name
    return None


# Patterns that reveal the user's name during ordinary conversation.
_USER_NAME_PATTERNS = [
    r"\bmy\s+name\s+(?:is|'?s)\s+(.+?)(?:\.|$|\?|\!)",
    r"\bcall\s+me\s+(.+?)(?:\.|$|\?|\!)",
    r"\byou\s+can\s+call\s+me\s+(.+?)(?:\.|$|\?|\!)",
    r"\bfrom\s+now\s+on\s+(?:call\s+me\s+|my\s+name\s+(?:is|'?s)\s+)(.+?)(?:\.|$|\?|\!)",
    r"\bchange\s+(?:my\s+)?name\s+(?:to|as)\s+(.+?)(?:\.|$|\?|\!)",
    r"\bset\s+(?:my\s+)?name\s+(?:to|as)\s+(.+?)(?:\.|$|\?|\!)",
    r"\bi\s+go\s+by\s+(.+?)(?:\.|$|\?|\!)",
]

# Patterns that reveal the assistant's name during ordinary conversation.
# NOTE: we intentionally do NOT match "Your name is X" because small models
# frequently confuse it with the user's name.
_ASSISTANT_NAME_PATTERNS = [
    r"\bcall\s+yourself\s+(.+?)(?:\.|$|\?|\!)",
    r"\bi\s+will\s+call\s+you\s+(.+?)(?:\.|$|\?|\!)",
    r"\bi(?:'m|\s+am)?\s+(?:going\s+to\s+)?call\s+you\s+(.+?)(?:\.|$|\?|\!)",
    r"\bchange\s+(?:your\s+)?name\s+(?:to|as)\s+(.+?)(?:\.|$|\?|\!)",
    r"\bset\s+(?:your\s+)?name\s+(?:to|as)\s+(.+?)(?:\.|$|\?|\!)",
]


def maybe_update_names_from_message(user_input: str, config: dict[str, Any]) -> bool:
    """Detect explicit name changes in user chat and update config/notes.

    Returns True if either name was changed.
    """
    current_user = config.get("user_name") or DEFAULT_CONFIG["user_name"] or "User"
    current_assistant = config.get("assistant_name") or DEFAULT_CONFIG["assistant_name"]

    new_user = _extract_name(user_input, _USER_NAME_PATTERNS)
    new_assistant = _extract_name(user_input, _ASSISTANT_NAME_PATTERNS)

    # Don't let a user-name pattern override an assistant-name reveal and vice versa.
    if new_user and new_user.lower() == current_assistant.lower():
        new_user = None
    if new_assistant and new_assistant.lower() == current_user.lower():
        new_assistant = None

    user_changed = bool(new_user and new_user != current_user)
    assistant_changed = bool(new_assistant and new_assistant != current_assistant)

    if not user_changed and not assistant_changed:
        return False

    if user_changed:
        config["user_name"] = new_user
    if assistant_changed:
        config["assistant_name"] = new_assistant

    config["first_run"] = False
    save_config(config)
    _write_identity_notes(config["assistant_name"], config["user_name"])
    return True


def setup_names(config: dict[str, Any], first_run: bool = True) -> bool:
    """Interactive first-run setup: ask for assistant and user names."""
    if first_run:
        print("\n  It looks like this is the first time we're chatting.")
        print("  Let's set things up so I know who we are.\n")
    else:
        print("\n  Let's update who we are.\n")

    try:
        user_name = input("  What is your name? ").strip()
    except (EOFError, KeyboardInterrupt):
        user_name = ""
    if not user_name:
        user_name = config.get("user_name") or DEFAULT_CONFIG["user_name"] or "User"
        print(f"  Using default user name: {user_name}")

    try:
        assistant_name = input("  What would you like to name me? ").strip()
    except (EOFError, KeyboardInterrupt):
        assistant_name = ""
    if not assistant_name:
        assistant_name = config.get("assistant_name") or DEFAULT_CONFIG["assistant_name"]
        print(f"  Using default assistant name: {assistant_name}")

    changed = (
        config.get("user_name") != user_name
        or config.get("assistant_name") != assistant_name
    )

    config["user_name"] = user_name
    config["assistant_name"] = assistant_name
    config["first_run"] = False
    save_config(config)
    _write_identity_notes(assistant_name, user_name)

    print(f"\n  Great — I'll call you {user_name}, and you can call me {assistant_name}.")
    return changed


def _load_model_presets() -> dict[str, dict]:
    """Load named model presets from models.json if present."""
    if not MODELS_FILE.exists():
        return {}
    try:
        return json.loads(MODELS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def switch_model_preset(config: dict[str, Any], preset_key: str) -> bool:
    """Update config.json to use a named model preset. Returns True on success."""
    presets = _load_model_presets()
    if not presets:
        print("  No models.json preset file found.")
        return False
    if preset_key not in presets:
        print(f"  Unknown preset: {preset_key}")
        print("  Available presets:")
        for key, info in presets.items():
            print(f"    - {key}: {info.get('model_name')} — {info.get('description', '')}")
        return False

    preset = presets[preset_key]
    old_model = config.get("model_name", "<unset>")
    config["model_name"] = preset["model_name"]
    save_config(config)

    print(f"  Switched model preset: {preset_key}")
    print(f"    {old_model} -> {preset['model_name']}")
    if preset.get("adapter_compatible"):
        print("    LoRA adapter is compatible.")
    else:
        print("    LoRA adapter will be disabled (base model only).")
    print(f"    Memory estimate: {preset.get('memory_note', '')}")
    print("  Restart Symbio to load the new model.")
    return True


def list_model_presets(config: dict[str, Any]):
    """Print available model presets and mark the active one."""
    presets = _load_model_presets()
    if not presets:
        print("  No models.json preset file found.")
        return
    current = config.get("model_name", "")
    print("  Available model presets:")
    for key, info in presets.items():
        marker = "*" if info.get("model_name") == current else " "
        adapter = "LoRA OK" if info.get("adapter_compatible") else "base only"
        print(f"    [{marker}] {key}: {info.get('model_name')}")
        print(f"        {info.get('description', '')}")
        print(f"        Adapter: {adapter} | {info.get('memory_note', '')}")
