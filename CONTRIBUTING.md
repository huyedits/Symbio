# Contributing to Symbio

Thanks for helping make Symbio better. This guide covers how to set up a development environment, run tests, understand the architecture, and open issues or pull requests.

## How to contribute

1. **Open an issue first** for large changes (new backends, major refactors, or new tools) so we can align on direction.
2. **Fork the repo** and create a feature branch.
3. **Keep changes focused.** One logical change per PR makes review faster.
4. **Preserve behavior.** The smoke test and unit tests should pass after your change.
5. **Update docs.** If you change the CLI, config, or tool behavior, update `README.md`.

## Development setup

```bash
# Clone the repo
git clone <your-fork-url>
cd agi

# Create a virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

The project uses MLX for inference and LoRA fine-tuning, so the default path works best on Apple Silicon. See the roadmap in `README.md` for ports to CUDA or llama.cpp.

## Running tests

Fast unit tests (no model loading):

```bash
venv/bin/python test_learn.py
```

End-to-end tests load the configured MLX model and may run short LoRA updates. They require the model downloaded and enough memory:

```bash
venv/bin/python smoke_test.py
venv/bin/python test_dynamic_names.py
venv/bin/python test_learn_e2e.py
venv/bin/python test_auto_learn_e2e.py
venv/bin/python test_deferred_learn.py
venv/bin/python test_threshold_training.py
```

The smoke test is the canonical integration check. Run it after any refactor or rebrand change.

## Architecture overview

The project is split into a `symbio/` package and a thin `main.py` CLI wrapper.

- `symbio/constants.py` — project paths, `DEFAULT_CONFIG`, and `_SHELL_COMMANDS`.
- `symbio/utils.py` — small pure helpers: response cleaning, tool parsing, note filename helpers.
- `symbio/config.py` — loading/saving `config.json`, name detection, model presets.
- `symbio/store.py` — SQLite session store with FTS5 search.
- `symbio/computer.py` — Playwright browser and PyAutoGUI desktop automation.
- `symbio/sandbox.py` — sandboxed shell commands and `execute_code`; writes `sandbox/symbio_tools.py` plus a `caine_tools.py` alias.
- `symbio/tools.py` — Hermes-style tool registry and standalone tool runners.
- `symbio/llm.py` — MLX loading/generation helpers and LoRA training logic.
- `symbio/learn.py` — correction detection, mistake notes, and batch learning.
- `symbio/chat.py` — system prompt, banner, and interactive chat loop.
- `symbio/agent.py` — `AIAgent` class that ties everything together.
- `main.py` — backward-compatible CLI that re-exports the public API.

Dependency rules to avoid circular imports:
- `constants.py` and `utils.py` import nothing from `symbio`.
- `config.py` imports `constants`/`utils`.
- `store.py` and `computer.py` import `constants` only.
- `sandbox.py` imports `constants`/`utils`/`config`.
- `tools.py` imports `constants`/`utils`/`config`/`store`/`computer`/`sandbox`.
- `llm.py` imports `constants`/`utils`/`config`.
- `learn.py` imports `constants`/`utils`/`llm`.
- `chat.py` imports `constants`/`utils`/`config`/`llm`/`learn` and lazily imports `agent`.
- `agent.py` imports all of the above.

## Concrete asks

These are larger projects that would have high impact. If one interests you, open an issue to discuss the design before starting.

### CUDA port

Add a PyTorch or Hugging Face Transformers backend so Symbio can run on NVIDIA/AMD GPUs. The minimal change would be a backend abstraction around `load`, `generate`, and LoRA training, with MLX remaining the default.

### llama.cpp backend

Support GGUF models via llama.cpp. This would broaden model support and reduce memory use, especially for larger dense models.

### LoRA optimization

- Faster adapter swapping at runtime.
- Gradient checkpointing for larger batch sizes.
- Memory profiling and automatic rank selection.

### Refactoring

- Split `tools.py` further if it grows (e.g., file tools, email tools, browser tools).
- Add more unit tests that don't require model loading.
- Introduce a proper plugin interface for new tools.

### Sparse / quantized adapters

- QLoRA-style 4-bit/8-bit base model loading.
- Sparse LoRA updates (e.g., random or magnitude-based sparsity).
- Adapter pruning beyond checkpoint removal.

## Code style

- Follow PEP 8.
- Keep functions small and focused.
- Add docstrings for public functions and classes.
- Prefer type hints where they add clarity.
- Avoid circular imports; respect the module dependency rules above.

## Opening issues / PRs

- **Bug reports:** include the command you ran, the error message, and your environment (macOS version, Python version, MLX version).
- **Feature requests:** describe the use case and any alternatives you considered.
- **PRs:** keep the description concise, reference any related issues, and confirm which tests you ran.

End PR bodies with if you are using a harness or ai agent:

```
🤖 Generated with [Claude Code](https://claude.com/claude-code) or the harness.
```

Thank you for contributing.
