#!/usr/bin/env python3
"""Switch the active model preset for Symbio."""
import argparse
import json
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).parent.resolve()
CONFIG_FILE = PROJECT_DIR / "config.json"
MODELS_FILE = PROJECT_DIR / "models.json"
ADAPTER_DIR = PROJECT_DIR / "adapters"


def load_models() -> dict:
    if not MODELS_FILE.exists():
        print(f"Model presets file not found: {MODELS_FILE}", file=sys.stderr)
        sys.exit(1)
    return json.loads(MODELS_FILE.read_text(encoding="utf-8"))


def load_config() -> dict:
    if not CONFIG_FILE.exists():
        print(f"Config file not found: {CONFIG_FILE}", file=sys.stderr)
        sys.exit(1)
    return json.loads(CONFIG_FILE.read_text(encoding="utf-8"))


def save_config(config: dict):
    CONFIG_FILE.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


def saved_adapter_model() -> str | None:
    """The model_name the saved LoRA adapter was trained for, or None if absent.

    Mirrors symbio.config._adapter_matches_model, but standalone so this
    script stays import-light. A killed/OOM'd run can leave adapter_config.json
    behind with no weights; that is still a real answer here, not an error.
    """
    adapter_config = ADAPTER_DIR / "adapter_config.json"
    if not adapter_config.exists():
        return None
    try:
        adapter_cfg = json.loads(adapter_config.read_text(encoding="utf-8"))
        return adapter_cfg.get("model") or None
    except Exception:
        return None


def adapter_matches_model(model_name: str) -> bool:
    """Whether the saved LoRA adapter was trained for the given model_name."""
    saved = saved_adapter_model()
    return saved is None or saved == model_name


def list_presets(models: dict, current_model: str):
    print("Available model presets:")
    saved = saved_adapter_model()
    for key, info in models.items():
        marker = "*" if info.get("model_name") == current_model else " "
        if saved is None:
            adapter = "no adapter"
        elif saved == info.get("model_name", ""):
            adapter = "LoRA OK"
        else:
            adapter = "base only"
        print(f"  [{marker}] {key}: {info.get('model_name')}")
        print(f"      {info.get('description', '')}")
        print(f"      Adapter: {adapter} | {info.get('memory_note', '')}")


def switch_preset(preset_key: str):
    models = load_models()
    config = load_config()

    if preset_key not in models:
        print(f"Unknown preset: {preset_key}", file=sys.stderr)
        print("Run with --list to see available presets.", file=sys.stderr)
        sys.exit(1)

    preset = models[preset_key]
    old_model = config.get("model_name", "<unset>")
    config["model_name"] = preset["model_name"]
    save_config(config)

    print(f"Switched model preset: {preset_key}")
    print(f"  {old_model} -> {preset['model_name']}")
    saved = saved_adapter_model()
    if saved is None:
        print("  No LoRA adapter present (base model only).")
    elif saved == preset["model_name"]:
        print("  The current LoRA adapter is compatible with this model.")
    else:
        print(f"  The current LoRA adapter was trained for {saved} and will be DISABLED (base model only).")
        print("  Re-train with /train if you want a LoRA for this model.")
    print(f"  Memory estimate: {preset.get('memory_note', '')}")
    print("\nRestart Symbio to load the new model.")


def main():
    parser = argparse.ArgumentParser(description="Switch Symbio's active model preset.")
    parser.add_argument("preset", nargs="?", help="Preset key to switch to")
    parser.add_argument("--list", "-l", action="store_true", help="List available presets")
    args = parser.parse_args()

    models = load_models()
    config = load_config()

    if args.list or not args.preset:
        list_presets(models, config.get("model_name", ""))
        return

    switch_preset(args.preset)


if __name__ == "__main__":
    main()
