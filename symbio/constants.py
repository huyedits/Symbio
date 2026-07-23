"""Shared constants and default configuration for Symbio."""

from pathlib import Path
from typing import Any

PROJECT_DIR = Path(__file__).parent.parent.resolve()
LOG_DIR = PROJECT_DIR / "logs"
DATA_DIR = PROJECT_DIR / "training_data"
TRAIN_FILE = DATA_DIR / "train.jsonl"
VALID_FILE = DATA_DIR / "valid.jsonl"
ADAPTER_DIR = PROJECT_DIR / "adapters"
# Worker-model adapters (see symbio/app/dispatch.py) live under the same
# adapters/ tree — one subdirectory per role — so they're covered by the
# existing "adapters/" .gitignore entry and by any tooling that already
# treats ADAPTER_DIR's contents as disposable/local-only.
WORKER_ADAPTERS_DIR = ADAPTER_DIR / "workers"
WORKER_MODELS_FILE = PROJECT_DIR / "symbio" / "app" / "worker_models.json"
NOTES_DIR = PROJECT_DIR / "notes"
MISTAKES_DIR = NOTES_DIR / "mistakes"
MISTAKES_ARCHIVE_DIR = MISTAKES_DIR / "archive"
# Decayed research notes go here instead of being deleted, mirroring how
# digested mistake notes are archived. Subdirectory, so excluded from the
# non-recursive *.md globs that feed RAG and digestion.
NOTES_ARCHIVE_DIR = NOTES_DIR / "archive"
SANDBOX_DIR = PROJECT_DIR / "sandbox"
SCREENSHOTS_DIR = PROJECT_DIR / "screenshots"
DIGEST_MANIFEST = DATA_DIR / "digest_manifest.json"
CONFIG_FILE = PROJECT_DIR / "config.json"
MODELS_FILE = PROJECT_DIR / "models.json"
GATEWAY_PID_FILE = PROJECT_DIR / "gateway.pid"
# Paths used by the tag-based agent in symbio.app.
PROMPT_FILE = PROJECT_DIR / "prompt.md"
CRON_FILE = PROJECT_DIR / "cron_jobs.json"
MEMORY_FILE = PROJECT_DIR / "agent_memory.md"
PROFILE_FILE = PROJECT_DIR / "user_profile.md"
SESSIONS_DIR = PROJECT_DIR / "sessions"
# Snapshot of the last shipped default prompt; used to auto-update prompt.md
# when the user has not customized it.
PROMPT_DEFAULT_FILE = PROJECT_DIR / "prompt.md.default"

for d in (
    LOG_DIR,
    DATA_DIR,
    ADAPTER_DIR,
    NOTES_DIR,
    MISTAKES_DIR,
    MISTAKES_ARCHIVE_DIR,
    NOTES_ARCHIVE_DIR,
    SANDBOX_DIR,
    SCREENSHOTS_DIR,
    SESSIONS_DIR,
):
    d.mkdir(parents=True, exist_ok=True)


def adapter_dir_for(role: str | None = None) -> Path:
    """Path to the adapter directory for a worker role, or the headmaster's
    own single adapter when role is None — unchanged from before per-role
    adapters existed, so every existing call site (chat.py, training.py)
    keeps working with zero changes as long as it doesn't pass a role."""
    if role is None:
        return ADAPTER_DIR
    return WORKER_ADAPTERS_DIR / role


def data_dir_for(role: str | None = None) -> Path:
    """Training-data directory for a worker role, or the headmaster's own
    shared corpus when role is None. A worker trains on its own narrow
    task data, not the headmaster's general conversation corpus."""
    if role is None:
        return DATA_DIR
    return DATA_DIR / "workers" / role

DEFAULT_CONFIG: dict[str, Any] = {
    "model_name": "mlx-community/Qwen2.5-3B-Instruct-4bit",
    "assistant_name": "Symbio",
    # Empty by default so the first run triggers interactive name setup and
    # every install seeds its own identity and training data.
    "user_name": "",
    "lora": {
        "rank": 8,
        "dropout": 0.1,
        "scale": 5.0,
        "num_layers": 8,
        "batch_size": 1,
        "learning_rate": 1e-4,
        "iters": 50,
        # Adaptive training: keep running chunks of `iters` while validation
        # loss improves by `min_improvement`, until `target_val_loss` or the
        # hard `max_iters` cap. Keep `iters` a multiple of `steps_per_eval`.
        "adaptive": True,
        "max_iters": 200,
        "target_val_loss": 0.05,
        "min_improvement": 0.02,
        # Every sample carries the full system prompt (~800 tokens); 1024
        # truncated long samples mid-reply, which trains truncated outputs.
        "max_seq_length": 2048,
        "steps_per_eval": 25,
        "save_every": 50,
    },
    "agent": {
        "max_turns": 5,
        "history_limit": 40,
        "sandbox_timeout": 30,
        "code_timeout": 300,
        "max_output_len": 4000,
        "temperature": 0.1,
        "top_p": 0.9,
        "repetition_penalty": 1.15,
    },
    "model": {
        "allow_lora": True,
        "allow_moe_lora": False,
        "moe_fine_tuning_mode": "rag_only",
    },
    "rag": {
        "enabled": True,
        "top_k": 5,
        "max_context_tokens": 1500,
        "sources": ["notes", "sessions"],
    },
    "training_planner": {
        "enabled": True,
        "min_turns": 200,
        "min_repetitions": 3,
        "neutrality_review": True,
        "auto_train": False,
    },
    "learn": {
        "enabled": True,
        "auto": True,
        "auto_train": True,
        "mistake_threshold": 5,
        "batch_train_iters": 25,
        "boost_factor": 3,
        "short_train_iters": 10,
        "correction_phrases": [
            "no,",
            "not quite",
            "that's wrong",
            "incorrect",
            "wrong",
            "you misunderstood",
            "try again",
            "actually",
            "i meant",
            "i said",
            "i asked",
            "not what",
            "that's not",
            "you're wrong",
            "fix it",
            "correction",
            "rephrase",
        ],
    },
    "tools": {
        "enabled_groups": [
            "memory", "notes", "terminal", "code", "web_search",
            "browser", "digest", "train", "cron", "config", "frontier",
        ],
    },
}

# Shell commands that small models sometimes emit as Hermes tool names.
_SHELL_COMMANDS: frozenset[str] = frozenset({
    "pwd", "ls", "date", "whoami", "uname", "df", "du", "find", "grep",
    "cat", "head", "tail", "echo", "wc", "sort", "ps", "top", "env",
    "printenv", "id", "hostname", "uptime", "which", "whereis", "mkdir",
    "touch", "cp", "mv", "basename", "dirname", "seq", "tr", "cut", "awk",
    "sed", "uniq", "xargs", "tee", "less", "more", "file", "stat", "realpath",
})
