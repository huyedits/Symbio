# Symbio - that fine tuning agent.

> **A local-first AI agent that learns from your corrections.**
>
> Runs on your Mac. Remembers what matters. Learns new skills. Fine-tunes itself with LoRA. No cloud inference. No subscription.

[![Live Demo](https://img.shields.io/badge/%F0%9F%A4%97-Live%20Demo-yellow)](https://huggingface.co/spaces/HuyEdits/symbio-demo)
[![GitHub](https://img.shields.io/badge/GitHub-Symbio-black?logo=github)](https://github.com/huyedits/Symbio)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](#license)

**[Try the interactive demo](https://huggingface.co/spaces/HuyEdits/symbio-demo)** · **[Quick Start](#quick-start)** · **[How it learns](#how-it-learns)** · **[Roadmap](#roadmap)**

---

## What is Symbio?

Most AI agents have a problem:

**They forget.**

*like forget forget*
get it cause it was in short term memory? 

You correct an agent today, and tomorrow it makes the same mistake again. You can put instructions in a system prompt, but that makes the prompt larger, it becomes slower when processing, and doesn't really teach the model anything.

Symbio takes a different approach - obviously why else make this?

```text
You → Agent → Mistake → Correction
                    ↓
              Learning data
                    ↓
              LoRA training
                    ↓
             New adapter
                    ↓
              Agent improves
       it now can be shown in the layer
```

Corrections and successful tool recoveries are automatically collected as training examples. Once enough examples accumulate, Symbio performs a small LoRA fine-tune and reloads the resulting adapter.

The goal is simple:

> **The longer you use Symbio, the more you define what AI means to you**

Everything can stay on your machine, nothing phones "home"

---

## Le Features

*  **Learns from corrections** — automatically detects corrections and turns them into training data.
*  **Self-corrects tool mistakes** — successful recovery from a failed command can become a training example.
*  **Learnable skills** — create a skill as a Markdown procedure and train a dedicated worker adapter for it.
*  **LoRA fine-tuning** — only small adapter weights are trained; the base model stays frozen.
*  **Mixture of Agents** — a headmaster can delegate bounded tasks to smaller worker models.
*  **Local memory** — notes, sessions, training data, adapters and caches live locally.
*  **RAG retrieval** — relevant notes can be retrieved and supplied as context.
*  **Web research** — search the web and automatically save useful discoveries as notes.
*  **Browser automation** — open pages, click, type and scroll through a live browser.
*  **Shell & Python tools** — execute sandboxed commands and short Python programs.
*  **Telegram gateway** — use your local Symbio instance from your phone.
*  **Permission gates** — dangerous actions require explicit approval.
*  **Golden-set regression protection** — bad fine-tunes can automatically roll back.
*  **Skill evaluation** — compare base, prompted and adapter performance.
*  **Crash recovery** — interrupted training is recorded and can be resumed.
*  **Self-pruning** — junk notes and duplicate session turns can be archived.
*  **No API required for inference** — the default architecture is designed around local models.
* 
---

## 🎥 Demo

### Live browser demo

Try the real tag parser, correction miner, research memory and RAG retriever in your browser:

**[https://huggingface.co/spaces/HuyEdits/symbio-demo](https://huggingface.co/spaces/HuyEdits/symbio-demo)**

### Screenshots

<img width="1300" alt="Symbio CLI" src="https://github.com/user-attachments/assets/c4e02593-f527-44dc-9bcb-181f329360ad" />

<img width="272" alt="Symbio mobile interface" src="https://github.com/user-attachments/assets/e8e7475a-aac8-455b-b978-3996f1d4d3fd" />

### Browser automation

Symbio can use a live browser to perform tasks such as opening Chrome and interacting with pages.

[https://github.com/user-attachments/assets/9e910d11-d204-4fb1-b42f-e09dd6243d20](https://github.com/user-attachments/assets/9e910d11-d204-4fb1-b42f-e09dd6243d20)

---

# Quick Start

## Requirements

Symbio currently targets **Apple Silicon Macs** using Apple's MLX stack. (unfortunately until we can get support for CUDA, etc )

### Recommended

* macOS
* Apple Silicon M-series Mac
* **16 GB+ unified memory**
* Python 3.10+
* ~8 GB free disk space for the default setup
* Additional space for models, adapters and browser data

The default 8B-class (or others - check the wizard) configuration is much more comfortable with 16 GB+ RAM. Smaller models can be used on machines with less memory.

> **Hardware compatibility:** Symbio is intended for Apple Silicon Macs. If you test it on different M-series generations or RAM configurations, please open an issue and share the model, RAM and configuration so compatibility can be documented properly.

---

## Install

```bash
cd symbio
git clone https://github.com/huyedits/Symbio
./install.sh
```

The installer:

1. Checks the machine and available resources.
2. Creates an isolated virtual environment.
3. Installs dependencies.
4. Downloads the browser engine when enabled.
5. Optionally prefetches the model.
6. Drops you into an activated environment.

Exit the environment with:

```bash
exit or control + c
```

Your original shell is untouched.

Then start Symbio:

```bash
symbio
```

Or use the shorter command:

```bash
symb
```

On first launch, an interactive setup wizard asks for your name, Symbio's name, model preset and enabled features.

Re-run the setup wizard at any time:

```bash
symb setup
```

### Installer options

<details>
<summary>Show installer options</summary>

```bash
./install.sh --prefetch-model   # Download the model during installation
./install.sh --no-browser       # Skip the Chromium download
./install.sh --with-native      # Include experimental native extras
./install.sh --dev              # Install development/test dependencies
./install.sh --no-shell         # Install without entering the environment
./install.sh --venv PATH        # Use a custom virtualenv location
```

</details>

---

# How it learns

The core learning loop is intentionally simple:

```text
1. You use Symbio
        ↓
2. Symbio makes a mistake
        ↓
3. You correct it
        ↓
4. Symbio detects the correction
        ↓
5. The mistake becomes training data
        ↓
6. Enough mistakes accumulate
        ↓
7. LoRA fine-tuning runs
        ↓
8. The adapter is loaded
        ↓
9. Symbio has learned from the examples
```

For example:

```text
You:      What is my name?

Symbio:   Your name is Bob.

You:      No, I'm Alice.

Symbio:   Your name is Alice.

          [Correction detected]
          Saved mistake note
          1/5 examples collected
```

Once the configured threshold is reached, Symbio digests the examples and runs a short LoRA update.

The default threshold is **5 mistake notes** - but you can edit that.

```text
notes/mistakes/
        ↓
training_data/train.jsonl
        ↓
      LoRA
        ↓
adapters/
        ↓
     Symbio
```

The `/learn` command can still be used to manually trigger learning from the previous correction.

---

# Learning from tool mistakes

Symbio can also learn from its own successful recovery.

For example:

```text
You: Open Chrome.

Symbio: <cmd>chrome</cmd>

Tool:
Command not found: chrome

Symbio:
'chrome' isn't a command here — trying the native way.

<cmd>open -a 'Google Chrome'</cmd>

[Learn]
Tool mistake captured.
```

The failed → successful sequence can become training data.

This means Symbio can learn not only from:

> "No, that's wrong."

but also from:

> "That command failed, so here's what actually worked."

Only a confirmed successful recovery is captured.

---

# Skills

Skills let Symbio turn procedures into dedicated, trainable capabilities.

Create one:

```bash
symb skill new "Fix wifi"
```

Or from the chat:

```text
/new-skill Fix wifi
```

A skill starts as a readable Markdown procedure:

```text
notes/skills/fix_wifi.md
```

As the skill is used, mistakes and corrections are tracked separately:

```text
notes/skills/fix_wifi.md
notes/skills/fix_wifi.md.health.jsonl
```

The Markdown file remains clean while the hidden health log collects training examples.

After enough examples accumulate, Symbio trains a dedicated worker adapter:

```text
Skill
  ↓
Training examples
  ↓
LoRA
  ↓
adapters/workers/fix_wifi/
```

Each skill can therefore have its own adapter.

Adapters can be:

* loaded
* hot-swapped
* evaluated
* archived
* restored

Useful commands:

```bash
symb skill list
symb skill new "Fix wifi"
symb skill eval "Fix wifi"
symb skill rm fix_wifi

symb archive
symb archive --dry-run
symb archive --restore adapter fix_wifi
```

---

# Proving a skill is actually in the weights

There is an obvious objection to learned skills:

> "Couldn't you just put the procedure in the prompt?"

Symbio includes a three-way evaluation harness specifically to test this.

```text
                ┌─────────────┐
                │   Skill     │
                └──────┬──────┘
                       │
          ┌────────────┼────────────┐
          ↓            ↓            ↓
        Base        Prompted      Adapter
          │            │            │
       no steps     steps given   no steps
          │            │            │
          └────────────┴────────────┘
                       ↓
                    Compare
```

| Condition  | Procedure in prompt? | Measures                          |
| ---------- | -------------------: | --------------------------------- |
| `base`     |                   No | What the base model already knows |
| `prompted` |              **Yes** | The "just prompt it" baseline     |
| `adapter`  |                   No | What the LoRA adapter learned     |

Run:

```bash
symb skill eval "Fix wifi"
```

Or choose specific arms:

```bash
symb skill eval fix_wifi --threshold 0.7 --arms base,adapter
```

Example:

```text
Skill: Fix wifi
----------------------------------------------------------
condition   steps in prompt   score     coverage
----------------------------------------------------------
base        no                0/5         0%
prompted    YES               5/5        93%
adapter     no (in weights)   5/5        100%
----------------------------------------------------------
```

The adapter receives the worker's normal system prompt but **not the procedure itself**.

That makes the experiment much more interesting than simply checking whether the model can follow a prompt containing the answer.

### Evaluation methodology

By default, Symbio:

* generates multiple task phrasings
* deliberately avoids simply replaying training prompts
* compares step vocabulary
* strips enumerators such as `1.` and `2.` from the metric
* stores raw responses in the JSON report
* reports null results rather than inventing a score

The goal is to make the evaluation auditable rather than flattering.

> A high adapter score demonstrates recall of the trained procedure. It does not prove general intelligence or deep conceptual understanding.

---

# Six-skill evaluation

A larger evaluation using six generated skills produced:

| Skill                      | Base | Prompted | Adapter |
| -------------------------- | ---: | -------: | ------: |
| Quick Task Helper          |  0/5 |      1/5 | **5/5** |
| Coffee Making              |  1/5 |      5/5 | **5/5** |
| Bicycle Tuning             |  1/5 |      5/5 | **5/5** |
| Repotting a Houseplant     |  2/5 |      5/5 | **5/5** |
| Shipping a Parcel Overseas |  0/5 |      5/5 | **5/5** |
| Sharpening a Kitchen Knife |  1/5 |      4/5 | **5/5** |

WOWIE, that is a BIG BIG jump!!!!
Overall:

```text
Adapter: 30/30
Base:     5/30
```

These numbers should be treated as an experiment, not a benchmark claim. The evaluation metric measures reproduction of the skill's procedure, which is specifically what the experiment is designed to test.

Custom evaluation tasks can be added under:

```text
training_data/workers/<role>/eval_tasks.json
```

Example:

```json
[
  {
    "id": "no_wifi",
    "prompt": "wifi's dead again",
    "must_include": ["toggle"]
  },
  "the network dropped, sort it out"
]
```

---

# Mixture of Agents

Symbio can optionally use a **Mixture of Agents (MoA)** architecture.

Instead of asking one large model to perform every task, a **headmaster** model can delegate bounded tasks to smaller worker models.

```text
                         ┌──────────────┐
                         │  Headmaster  │
                         └──────┬───────┘
                                │
                   delegate bounded task
                                │
             ┌──────────────────┼──────────────────┐
             ↓                  ↓                  ↓
        Summarizer           Browser            Custom
          Worker             Worker             Worker
             │                  │                  │
             └──────────────────┴──────────────────┘
                                ↓
                         Result → Headmaster
```

Delegation is disabled by default:

```json
{
  "dispatch": {
    "enabled": false
  }
}
```

This is intentional because loading multiple models increases memory usage.

### Included workers

| Worker      | Purpose                                              |
| ----------- | ---------------------------------------------------- |
| `summarize` | Condense text supplied by the headmaster             |
| `browser`   | Choose bounded browser actions from the current page |

Workers are loaded lazily and can be unloaded when idle.

Each worker can also have its **own training corpus and LoRA adapter**:

```text
training_data/workers/<role>/
adapters/workers/<role>/
```

This means the browser worker can learn browser behavior without modifying the headmaster.

### Worker training

Delegated tasks automatically generate `(input, output)` training examples.

Worker training uses the same safety mechanisms as headmaster training:

* golden-set evaluation
* regression detection
* automatic rollback
* separate adapters
* memory preflight
* crash recovery

---

# Training safety

Self-training is useful, but blindly training on everything an agent produces is dangerous.

Symbio therefore has several safeguards.

## Golden-set regression testing

Before and after each LoRA update, Symbio runs a fixed golden set.

The set checks behaviors such as:

* identifying itself correctly
* distinguishing itself from the user
* producing expected tool formats
* avoiding repetitive output

If a new adapter causes previously passing cases to fail:

```text
[Golden] Regression: 2 case(s) newly failing.
[Golden] Rolled back to the previous adapter.
```

The previous adapter is restored automatically.

Run manually:

```bash
/golden
```

or:

```bash
symb eval-lora
```

---

## Retrieval hygiene

A self-learning agent has an unusual failure mode:

```text
bad output
   ↓
saved
   ↓
retrieved
   ↓
repeated
   ↓
trained
   ↓
bad output becomes stronger
```

Symbio tries to break this loop.

Retrieval excludes internal machinery such as:

* tool transcripts
* tool-call syntax
* system observation scaffolding
* other generated machinery

Retrieval also requires meaningful, relatively rare terms rather than simply returning the least-bad matches.

If nothing relevant matches, retrieval is allowed to return **nothing**.

---

## Self-pruning

Junk notes and duplicate session turns can be archived automatically.

```bash
symb archive
```

Or preview:

```bash
symb archive --dry-run
```

The `/tidy` command performs additional cleanup:

```text
/tidy
/tidy dry
```

Notes are archived rather than silently deleted.

---

## Crash-safe training

Long-running training jobs are recorded before they begin.

If the process crashes:

```text
[Resume] training for worker 'fix_wifi' was interrupted.
[Resume] 1 unfinished task(s) carried over.
```

Check pending work:

```bash
/resume
```

Run it:

```bash
/resume run
```

Discard it:

```bash
/resume clear
```

Training is **not automatically restarted after a crash**. This prevents a machine from repeatedly entering an out-of-memory cycle.

---

# 💻 Tools

Symbio can interact with the local machine through several tool groups.

### Files

Read, write, search and patch files within the project environment.

### Terminal

Run sandboxed shell commands:

```xml
<tool_call>
{"name":"terminal","arguments":{"cmd":"ls -la"}}
</tool_call>
```

### Python

Execute short Python programs through the controlled execution environment.

### Browser

Interact with a live browser:

* open
* click
* type
* scroll
* inspect page content

### Notes

Save information for future retrieval:

```xml
<tool_call>
{"name":"note","arguments":{
  "action":"add",
  "target":"note",
  "content":"The user likes coffee."
}}
</tool_call>
```

### Web research

Search the web and save useful discoveries as local `Learned:` notes.

### Telegram

Run the same agent through a Telegram gateway.

---

# 📱 Telegram

Start the gateway:

```bash
symb gateway start
```

Check readiness:

```bash
symb gateway status
```

Stop it:

```bash
symb gateway stop
```

Set the bot token through the setup wizard or environment:

```bash
export SYMBIO_TELEGRAM_TOKEN="..."
```

You must explicitly configure allowed chat IDs:

```bash
symb config set telegram.allowed_chat_ids '[123456789]'
```

Telegram dangerous actions use inline approval.

For example, actions involving:

* shell commands
* browser domains
* Python execution
* configuration changes
* scheduled jobs
* training

can require an explicit approval before execution.

> **Important:** saying `No` rejects the action for the entire turn. Symbio will not retry the same action through another tool.

### Telegram commands

| Command      | Description                            |
| ------------ | -------------------------------------- |
| `/start`     | Welcome message                        |
| `/help`      | Show available commands                |
| `/ping`      | Show latency breakdown                 |
| `/status`    | Show model, adapter and session status |
| `/golden`    | Run the golden set                     |
| `/train`     | Start LoRA training                    |
| `/selfcheck` | Check enabled features                 |
| `/setup`     | Configuration help                     |
| `/tools`     | Toggle tool groups                     |
| `/cancel`    | Clear the current session              |

---

#CLI
```bash
symbio # start chat
symbio config                # Show configuration
symbio config get <key>      # Read a config value
symbio config set <key> <value>
symbio train                 # Run LoRA training
symbio skill list            # List skills
symbio skill new <name>      # Create a skill
symbio skill rm <role>       # Delete a skill
symbio skill eval <name>     # Evaluate a skill
symbio eval-lora             # Evaluate headmaster adapter
symbio archive               # Archive idle data
symbio archive --dry-run     # Preview archive actions
symbio gateway status        # Check Telegram
symbio gateway start         # Start Telegram
symbio gateway stop          # Stop Telegram
```
---

# Slash commands

Once inside Symbio:

| Command                         | Description                          |
| ------------------------------- | ------------------------------------ |
| `/quit`                         | Exit                                 |
| `/save`                         | Save the current conversation        |
| `/train`                        | Run LoRA training                    |
| `/train_worker <role>`          | Train a worker                       |
| `/resume`                       | Show unfinished work                 |
| `/learn`                        | Learn from the last correction       |
| `/digest`                       | Convert notes into training data     |
| `/note [title]`                 | Create a note                        |
| `/notes`                        | List notes                           |
| `/new-skill <name>`             | Create a skill                       |
| `/skills`                       | List skills                          |
| `/skill-adapters`               | List skill adapters                  |
| `/archive`                      | Archive idle notes/adapters          |
| `/restore note\|adapter <name>` | Restore archived data                |
| `/status`                       | Show current state                   |
| `/selfcheck`                    | Run health checks                    |
| `/setup`                        | Re-run setup                         |
| `/compact`                      | Compress memory                      |
| `/model`                        | List model presets                   |
| `/model <preset>`               | Switch model                         |
| `/run <cmd>`                    | Run a sandboxed command              |
| `/forget_last`                  | Remove the last exchange             |
| `/prune`                        | Remove stale adapter checkpoints     |
| `/tidy`                         | Clean junk notes and duplicate turns |

---

# LoRA fine-tuning

Symbio uses **LoRA (Low-Rank Adaptation)** through Apple's MLX ecosystem.

The base model remains frozen.

Only small adapter matrices are trained:

```text
Base model
████████████████████████████
             +
       Small LoRA adapter
             ↓
       Personalized model
```

Adapters are stored separately:

```text
adapters/
adapters/workers/
```

This allows Symbio to:

* train incrementally
* keep the base model unchanged
* switch adapters
* archive unused adapters
* roll back failed updates
* maintain separate adapters for different skills

Run training manually:

```bash
symb train
```

### Main LoRA settings

| Setting               | Default |
| --------------------- | ------: |
| `lora.rank`           |     `8` |
| `lora.num_layers`     |     `8` |
| `lora.scale`          |  `20.0` |
| `lora.dropout`        |   `0.0` |
| `lora.learning_rate`  |  `1e-4` |
| `lora.iters`          |   `300` |
| `lora.max_seq_length` |   `512` |
| `lora.save_every`     |   `100` |

Training uses validation checks and can stop early when validation loss plateaus.

---

#Configuration

Configuration lives in:

```text
config.json
```

You can edit it directly or use the CLI:

```bash
symb config
symb config get agent.temperature
symb config set agent.temperature 0.7
```

Some important settings:

| Key                             |           Default | Purpose                          |
| ------------------------------- | ----------------: | -------------------------------- |
| `model_name`                    | `Qwen/Qwen3-0.6B` | Base model                       |
| `assistant_name`                |          `Symbio` | Assistant name                   |
| `agent.temperature`             |             `0.7` | Generation temperature           |
| `agent.max_tool_rounds`         |               `3` | Tool rounds per turn             |
| `agent.max_reply_tokens`        |             `128` | Maximum reply length             |
| `agent.prompt_cache_enabled`    |            `true` | Reuse prompt KV cache            |
| `lora.rank`                     |               `8` | LoRA rank                        |
| `lora.iters`                    |             `300` | Full training iterations         |
| `learn.enabled`                 |            `true` | Enable learning                  |
| `learn.auto`                    |            `true` | Detect corrections automatically |
| `learn.auto_train`              |            `true` | Automatically train at threshold |
| `learn.mistake_threshold`       |               `5` | Corrections before auto-training |
| `learn.batch_train_iters`       |              `25` | Auto-training iterations         |
| `learn.boost_factor`            |               `3` | Correction sample weighting      |
| `dispatch.enabled`              |           `false` | Enable worker delegation         |
| `dispatch.max_resident_workers` |               `1` | Workers kept in memory           |
| `telegram.allowed_chat_ids`     |              `[]` | Authorized Telegram chats        |

---

#Security

Symbio is designed to run locally, but **local does not mean automatically safe**.

Shell and Python execution run with the privileges of the user who launched Symbio.

The sandbox is intended to reduce accidental damage, not provide a perfect security boundary.

### Important rules

* ### Review untrusted code before executing it.
* ### Do not give Symbio access to files you would not give a local program access to.
* ### Pay attention to permission prompts.
* ### A denied action is not retried through another tool.
* ### Telegram actions can require explicit approval.
* ### Keep secrets such as Telegram tokens out of source control.

The environment variable:

```bash
SYMBIO_TELEGRAM_TOKEN
etc
```

takes precedence over the token stored in `config.json`.

---

# Architecture

The project is organized as a Python package with a thin compatibility wrapper:

```text
.
├── main.py
├── symbio/
│   ├── constants.py
│   ├── app/
│   │   ├── cli.py
│   │   ├── chat.py
│   │   ├── config.py
│   │   ├── training.py
│   │   ├── learn.py
│   │   ├── golden.py
│   │   ├── eval.py
│   │   ├── skill_eval.py
│   │   ├── prune.py
│   │   ├── pending.py
│   │   ├── dispatch.py
│   │   ├── memory.py
│   │   ├── sandbox.py
│   │   ├── computer.py
│   │   ├── cron.py
│   │   ├── telegram.py
│   │   ├── tooling.py
│   │   ├── prompts.py
│   │   └── skills.py
│   └── utils.py
├── rag.py
├── models.json
├── config.json
├── notes/
├── training_data/
├── adapters/
├── cache/
├── logs/
├── sessions/
├── screenshots/
└── sandbox/
```

### Major components

| Component       | Responsibility              |
| --------------- | --------------------------- |
| `chat.py`       | Agent loop and sessions     |
| `training.py`   | LoRA training               |
| `learn.py`      | Correction detection        |
| `golden.py`     | Regression protection       |
| `skill_eval.py` | Skill evaluation            |
| `dispatch.py`   | Mixture-of-Agents workers   |
| `memory.py`     | Notes and persistent memory |
| `sandbox.py`    | Shell/Python execution      |
| `computer.py`   | Browser automation          |
| `telegram.py`   | Telegram gateway            |
| `prune.py`      | Corpus cleanup              |
| `pending.py`    | Crash-safe unfinished work  |
| `tooling.py`    | Tool parsing and formatting |
| `skills.py`     | Skill management            |

---

# Tool formats

Symbio supports legacy XML tags as well as the preferred Hermes-style tool format.

### Preferred

```xml
<tool_call>
{"name":"read_file","arguments":{"path":"config.json"}}
</tool_call>
```

```xml
<tool_call>
{"name":"terminal","arguments":{"cmd":"ls -la"}}
</tool_call>
```

```xml
<tool_call>
{"name":"note","arguments":{
  "action":"add",
  "target":"note",
  "content":"The user likes coffee."
}}
</tool_call>
```

### Legacy

```xml
<note title="User Preference">
The user likes coffee.
</note>
```

```xml
<cmd>ls</cmd>
```

Legacy formats remain supported for compatibility.

---

# Dynamic names

Symbio can learn both the user's name and its own name.

### User

```text
"My name is Alice."
"Call me Bob."
"You can call me Charlie."
"From now on call me Dana."
"Change my name to Eve."
"I go by Frank."
```

### Assistant

```text
"Call yourself Jarvis."
"I will call you Friday."
"I'm going to call you HAL."
"Change your name to Jeeves."
"Set your name as Alfred."
```

The phrase:

```text
"Your name is X"
```

is intentionally not treated as an assistant rename because smaller models can confuse it with a statement about the user's identity.

---

# Alternative installation

For an isolated install:

```bash
pipx install .
```

Or:

```bash
pipx install /path/to/Symbio
```

For development:

```bash
pip install -e .
```

The `symbio` and `symb` commands will then be available.

Legacy commands remain supported:

```bash
python main.py --telegram
python main.py --train
```

---

# Roadmap

## High priority

* [ ] **CUDA backend** — Support NVIDIA/AMD hardware through PyTorch or Transformers.
* [ ] **llama.cpp backend** — Support GGUF models and broader hardware.
* [ ] **LoRA optimization** — Faster adapter swaps and more memory-efficient training.
* [ ] **Better architecture separation** — Further isolate inference, tools, training and storage.
* [ ] **Sparse / quantized adapters** — Explore QLoRA, 4-bit/8-bit models and sparse updates.
* [ ] **MCP support** — Model Context Protocol.
* [ ] **Adapter marketplace** — See [`docs/adapter-marketplace.md`](docs/adapter-marketplace.md).
* [ ] **Additional messaging platforms**.
* [ ] **Long-term weight pruning**.

## Completed

* [x] More tools
* [x] Live browser automation
* [x] Permission-gated sandbox
* [x] Automatic self-correction
* [x] Learning new skills
* [x] Web research memory
* [x] Telegram bot
* [x] Mixture of Agents
* [x] Independently trainable worker adapters
* [x] Golden-set regression protection
* [x] Crash-safe training recovery
* [x] Skill evaluation harness
* [x] Automatic corpus cleanup

---

# Contributing

Contributions are welcome.

Some especially useful areas:

1. **Non-Apple hardware support**
2. **Model/backend integrations**
3. **Training performance**
4. **Memory optimization**
5. **Evaluation methodology**
6. **Browser automation**
7. **New worker types**
8. **Testing and regression coverage**
9. **Security hardening**
10. **Documentation**

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for development setup, testing and pull-request guidelines.

If you find a bug, please include:

* Mac model
* Apple Silicon generation
* unified memory
* model preset
* relevant configuration
* error/log output
* steps to reproduce

---

# Current limitations

Symbio is still experimental.

The biggest current limitations are:

* Apple Silicon only
* Local model size is constrained by unified memory
* Self-training can still overfit
* Skill evaluation primarily measures procedural recall
* Tool sandboxing is best-effort rather than a security boundary
* Multiple resident models can consume substantial memory
* Some features are experimental and may change

If you have a different Apple Silicon configuration, please report whether it works. Hardware reports are particularly useful for building a real compatibility matrix rather than guessing based on the chip name.

---

# License

Apache 2.0

---

## Support the project :

If Symbio is useful or interesting to you, **a GitHub star helps other people discover it.** :P

If you build something with Symbio, open an issue or discussion and show me what it learned.
