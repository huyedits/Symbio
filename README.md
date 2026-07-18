# Symbio
Your machine, your way.

[![Live Demo on Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Live%20Demo-Hugging%20Face%20Spaces-blue)](https://huggingface.co/spaces/HuyEdits/symbio-demo)

Symbio develops as you tell it what to do in repeat.

**Try the interactive demo** — the agent's real tag parser, self-correction miner, research memory, and RAG retriever running in your browser: https://huggingface.co/spaces/HuyEdits/symbio-demo

## What it does

- Chat through a local CLI.
- Save facts and notes as markdown files in `notes/`.
- Read, write, search, and patch files inside the project directory.
- Run sandboxed shell commands and short Python snippets.
- Check email via IMAP/SMTP (when configured).
- Digest notes into training data and fine-tune a LoRA adapter on the fly.
- Persist every conversation turn to JSONL and an SQLite store.

## Quick start

```bash
# Create a virtual environment and install dependencies
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Start chatting
python main.py
```

On first run, Symbio asks for your name and its name. These are saved to `config.json`.

## Configuration

Edit `config.json` to change the model, LoRA settings, or agent behavior:

| Key | Default | Note |
|---|---|---|
| `model_name` | `mlx-community/Qwen2.5-3B-Instruct-4bit` | Base MLX model |
| `assistant_name` | `Symbio` | What the assistant calls itself |
| `user_name` | *(asked at first run)* | Your name |
| `agent.max_turns` | `5` | Max tool rounds per user turn |
| `agent.temperature` | `0.1` | Sampling temperature (low for deterministic tool use) |
| `lora.dropout` | `0.1` | LoRA dropout to reduce overfitting |
| `lora.scale` | `5.0` | LoRA adapter scale |
| `lora.iters` | `50` | LoRA iterations per training chunk |
| `lora.adaptive` | `true` | Keep training in chunks while validation loss improves |
| `lora.max_iters` | `200` | Hard cap on total adaptive iterations |
| `lora.target_val_loss` | `0.05` | Stop early once validation loss reaches this |
| `lora.min_improvement` | `0.02` | Stop when a chunk improves val loss less than this |
| `learn.boost_factor` | `3` | Copies of a correction sample written to training data |
| `learn.short_train_iters` | `10` | Iterations for the quick `/learn` fine-tune pass |

## Slash commands

| Command | Description |
|---|---|
| `/quit` | Exit the chat |
| `/save` | Save the current conversation to training data |
| `/train` | Run LoRA fine-tuning and reload the adapter |
| `/learn` | Manually learn from your last correction (auto-learn is on by default) |
| `/digest` | Convert notes into training samples |
| `/note [title]` | Create a markdown note |
| `/notes` | List saved notes |
| `/status` | Show model, adapter, notes, and session info |
| `/setup` | Change assistant/user names |
| `/model` | List model presets |
| `/model <preset>` | Switch to a named model preset (restart to load) |
| `/run <cmd>` | Run a sandboxed shell command |
| `/forget_last` | Remove the last exchange from history |
| `/prune` | Remove stale adapter checkpoints |

## Model presets

Model presets live in `models.json`. Each preset records the Hugging Face model id, whether the current LoRA adapter is compatible, and an approximate memory budget. Switch presets with the CLI helper or a slash command inside chat:

```bash
# List available presets
python switch_model.py --list

# Switch to a different model
python switch_model.py moe_a27b
```

Inside chat:
```
/model
/model moe_a27b
```

After switching, restart Symbio to load the new model. The current LoRA adapter is automatically disabled when the selected preset is not adapter-compatible.

## Learning from corrections

Symbio detects natural corrections automatically and turns them into training data without you typing `/learn`. Instead of training on every single correction, it saves each mistake as a markdown note in `notes/mistakes/` and only fine-tunes once enough notes have accumulated.

Typical flow:
```
You:      What is my name?
Symbio:   Your name is Bob.
You:      No, I'm Alice.
Symbio:   Your name is Alice.
          [System] Correction detected (correction phrase).
          Saved mistake note: 20260715_123456_What_is_my_name.md
          1/5 mistake note(s) collected. Training will run after 4 more correction(s).
```

Symbio will:
1. Detect correction phrases ("No, ...", "Actually ...", "That's wrong", etc.) or an exact repeat of your last question.
2. Extract the original question, the wrong answer, the user's correction, and the corrected answer.
3. Save them as a markdown note in `notes/mistakes/`.
4. When `learn.mistake_threshold` (default 5) notes have accumulated, digest them into `training_data/train.jsonl` and run a LoRA update (`learn.batch_train_iters`, default 25).
5. Archive the used mistake notes to `notes/mistakes/archive/` and reload the adapter.

The `/learn` command is still available to force a mistake note from the last correction, but it is no longer required.

Tune the behaviour in `config.json`:

| Key | Default | Note |
|---|---|---|
| `learn.enabled` | `true` | Enable correction learning |
| `learn.auto` | `true` | Detect corrections automatically |
| `learn.auto_train` | `true` | Run the fine-tune automatically when the threshold is reached |
| `learn.mistake_threshold` | `5` | Number of mistake notes before a batch fine-tune runs |
| `learn.batch_train_iters` | `25` | LoRA iterations for the threshold-triggered batch update |
| `learn.boost_factor` | `3` | Copies of each correction sample written per mistake note |
| `learn.correction_phrases` | `[...]` | Phrases that trigger correction detection |

## Dynamic names

### Supported user-name phrasings

- *"My name is Alice."*
- *"Call me Bob."*
- *"You can call me Charlie."*
- *"From now on call me Dana."*
- *"Change my name to Eve."*
- *"I go by Frank."*

### Supported assistant-name phrasings

- *"Call yourself Jarvis."*
- *"I will call you Friday."*
- *"I'm going to call you HAL."*
- *"Change your name to Jeeves."*
- *"Set your name as Alfred."*

> Note: *"Your name is X"* is intentionally **not** treated as an assistant rename because small models often confuse it with the user's name.

## Tool formats

Symbio understands two ways to call tools:

- **Legacy XML tags**:
  - `<note title="User Preference">The user likes coffee.</note>` — save a note
  - `<cmd>ls</cmd>` — run a sandboxed command (legacy, still supported)
  - `<digest />` / `<train />` — digest notes or train

- **Hermes JSON-in-XML** (preferred):
  ```xml
  <tool_call>{"name": "read_file", "arguments": {"path": "config.json"}}</tool_call>
  <tool_call>{"name": "terminal", "arguments": {"cmd": "ls -la"}}</tool_call>
  <tool_call>{"name": "note", "arguments": {"action": "add", "target": "note", "content": "The user likes coffee."}}</tool_call>
  ```

## Security notes

- `terminal` and `execute_code` are best-effort sandboxes. They run with the privileges of the user who started the program and are scoped to the project directory.
- `execute_code` requires the script to import from `symbio_tools` (or the backward-compatible `caine_tools` alias) and blocks known dangerous imports.
- Do not paste untrusted code into the agent without reviewing it first.

## Architecture

The project is organized as a `symbio/` Python package with a thin `main.py` wrapper:

```
.
├── main.py              # Thin CLI; re-exports public API for backward compatibility
├── symbio/
│   ├── __init__.py      # Public re-exports
│   ├── constants.py     # Paths, DEFAULT_CONFIG, _SHELL_COMMANDS
│   ├── utils.py         # Pure helpers: cleaning, parsing, note filenames
│   ├── config.py        # load_config, save_config, name setup, model presets
│   ├── store.py         # SQLite session store
│   ├── sandbox.py       # Sandboxed command/code execution, symbio_tools stub
│   ├── computer.py      # Browser/desktop automation helpers
│   ├── tools.py         # Tool registry and standalone tool runners
│   ├── llm.py           # MLX/LoRA training helpers
│   ├── learn.py         # Correction detection and batch learning
│   ├── chat.py          # System prompt, banner, chat loop
│   └── agent.py         # AIAgent class
├── rag.py               # Lightweight keyword-based RAG
├── planner.py           # SQLite-backed training planner
├── seed_training.py     # Seed training corpus generator
├── smoke_test.py        # Automated smoke tests
├── config.json          # User configuration
├── models.json          # Model presets
├── notes/               # Markdown notes / memory
├── training_data/       # train.jsonl and valid.jsonl
├── adapters/            # LoRA adapter weights
├── logs/                # Session logs and SQLite store
└── sandbox/             # Scratch space for code execution
```

Import conventions:
- `constants.py` and `utils.py` do not import from other `symbio` modules.
- `config.py` uses `constants`/`utils`.
- `store.py` and `computer.py` use `constants` only.
- `sandbox.py` uses `constants`/`utils`/`config`.
- `tools.py` uses `constants`/`utils`/`config`/`store`/`computer`/`sandbox`.
- `chat.py` uses `constants`/`utils`/`config`/`llm`/`learn`/`agent` (lazy import).
- `llm.py` uses `constants`/`utils`/`config`.
- `learn.py` uses `constants`/`utils`/`llm`.
- `agent.py` uses all of the above.

## Roadmap / high-priority contributions

We are actively looking for help on:

1. **CUDA port** — MLX is Apple Silicon only. A PyTorch or Transformers backend would let Symbio run on NVIDIA/AMD hardware.
2. **llama.cpp backend** — Support GGUF models through llama.cpp for broader model coverage and lower memory use.
3. **LoRA optimization** — Faster adapter swaps, gradient checkpointing, and memory-efficient training.
4. **Refactoring** — Cleaner separation between inference, tools, training, and storage; better test coverage.
5. **Sparse / quantized adapters** — Experiment with QLoRA, 8-bit/4-bit base models, and sparse LoRA updates.

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup, testing, and how to open issues/PRs.


 > future projection <
- [ ] **Add MCP (Model Context Protocol)**
- [ ] **Add More Tools**
- [x] **Self correction when hallucinating**
- [x] **Be able to learn new skills on the fly**
- [ ] **Add Other Messaging Platforms**
- [ ] **Prune Old Weights (Future Milestone)**

## Star History

<a href="https://www.star-history.com/?repos=huyedits%2FSymbio&type=date&legend=top-left">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/chart?repos=huyedits/Symbio&type=date&theme=dark&legend=top-left&sealed_token=BARd7crHixQjaz11nJTAZ7mVM0hzMRPkR0XsWnt0JCDpfGb7UODGYP_v1vWqVZ7oBnYNeBSjSPD41Jz3zptiRq5d4it22dMAG2hzDZp-hqN1WUU71TnCUQzen-QuIt_rS3gQGtX2rxkJBNKMo5q86C2O0Q4om5BuX_2rj91AZGictnTvSaGS7Yb0fayE" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/chart?repos=huyedits/Symbio&type=date&legend=top-left&sealed_token=BARd7crHixQjaz11nJTAZ7mVM0hzMRPkR0XsWnt0JCDpfGb7UODGYP_v1vWqVZ7oBnYNeBSjSPD41Jz3zptiRq5d4it22dMAG2hzDZp-hqN1WUU71TnCUQzen-QuIt_rS3gQGtX2rxkJBNKMo5q86C2O0Q4om5BuX_2rj91AZGictnTvSaGS7Yb0fayE" />
   <img alt="Star History Chart" src="https://api.star-history.com/chart?repos=huyedits/Symbio&type=date&legend=top-left&sealed_token=BARd7crHixQjaz11nJTAZ7mVM0hzMRPkR0XsWnt0JCDpfGb7UODGYP_v1vWqVZ7oBnYNeBSjSPD41Jz3zptiRq5d4it22dMAG2hzDZp-hqN1WUU71TnCUQzen-QuIt_rS3gQGtX2rxkJBNKMo5q86C2O0Q4om5BuX_2rj91AZGictnTvSaGS7Yb0fayE" />
 </picture>
</a>
