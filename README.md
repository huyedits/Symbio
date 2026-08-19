# Symbio

A local-first agent that fine-tunes itself from your corrections — no cloud, no subscriptions, runs on your Mac.

Symbio takes notes, runs shell commands, searches the web, and turns your corrections into LoRA training data so it stops repeating your mistakes.

[![Live Demo on Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Live%20Demo-Hugging%20Face%20Spaces-blue)](https://huggingface.co/spaces/HuyEdits/symbio-demo) | [GitHub](https://github.com/huyedits/Symbio) | [Try it now](#quick-start)

<!-- TODO: drop a 15-20s GIF here showing the CLI status line (adapter "trained Xm ago", mistake counter ticking toward the threshold). 
     e.g. ffmpeg -i recording.mov -vf "fps=10,scale=960:-1" docs/demo.gif -->

> **Try the interactive demo** — the real tag parser, self-correction miner, research memory, and RAG retriever running in your browser: https://huggingface.co/spaces/HuyEdits/symbio-demo

## What it does

- Chat through a local CLI or a Telegram bot.
- Save facts and notes as markdown files in `notes/`.
- Read, write, search, and patch files inside the project directory.
- Run sandboxed shell commands and short Python snippets.
- Check email via IMAP/SMTP (when configured).
- Digest notes into training data and fine-tune a LoRA adapter on the fly.
- Persist every conversation turn to JSONL and an SQLite store.

## MOA feature
Symbio has a **MOA** (Mixture of Agents) mode. Instead of fine-tuning one big model for every task, the headmaster delegates bounded sub-tasks to smaller worker models via tool calls. The worker executes, and if it fails it returns to the headmaster for guidance. Once it works, a note is saved for both sides. If the same mistake repeats past the configured threshold, both the worker and the headmaster are fine-tuned: the worker learns how to execute the task, and the headmaster learns how to delegate it more efficiently.

## Skill feature
Symbio can learn **skills** on the fly. A skill starts as a simple markdown note with step-by-step instructions. As errors and corrections accumulate, they are logged in a hidden `.md.health.jsonl` sidecar so the note itself stays clean and readable. Once the mistake threshold is reached, the collected examples are fed into a LoRA fine-tune that creates a dedicated worker adapter for that skill — one adapter = one skill. Adapters are hot-swappable and can be archived if unused.

Use `/new-skill <name>` or `symb skill new <name>` to create one, `/skill-adapters` to list them, and `/archive` / `/restore` to manage idle notes and adapters.

### Proving the skill is actually in the weights

The obvious objection to "the model learned a skill" is *you could have just put the steps in the prompt*. `symb skill eval <name>` answers that with a number instead of an argument. It runs the same task battery three times:

| condition | steps in prompt? | what it measures |
|---|---|---|
| `base` | no | what the model already knew |
| `prompted` | **yes** | the "just prompt it" baseline |
| `adapter` | no — stripped out | what the LoRA weights hold |

The `adapter` arm gets the exact system prompt the worker was **trained** under, which deliberately names the skill but withholds the procedure. If it scores above `base`, the procedure came from the weights, because it was never in the context.

```bash
symb skill eval "Fix wifi"
symb skill eval fix_wifi --threshold 0.7 --arms base,adapter
```

```
Skill: Fix wifi   (5 tasks, generated)
----------------------------------------------------------
condition   steps in prompt   score     coverage
----------------------------------------------------------
base        no                0/5         0%
prompted    YES               5/5        93%
adapter     no (in weights)   5/5       100%
----------------------------------------------------------
```

Grading is deliberately dumb — the fraction of the skill's own step vocabulary the reply reproduces — so it cannot flatter the adapter, and every raw reply is written to the JSON report so the score can be audited by hand. A null result is reported as a null result.

Read the numbers honestly: `base 0/5` does **not** mean the base model is useless at the task. In the run above it answered with real `networksetup` commands, which is arguably better — it just isn't *your* saved procedure. And on a two-step skill this is memorisation, which is the claim being tested but the weakest form of it. Skills with a substantial procedure give a far more meaningful delta.

#### A six-skill run

To get past that objection, six skills with real procedures, where the **model wrote every step itself** and the training data was generated from those steps by `_seed_skill_training_data` — six samples each, none hand-written:

| skill | base | prompted | adapter |
|---|---|---|---|
| Quick Task Helper | 0/5 (5%) | 1/5 (22%) | **5/5** (93%) |
| Coffee Making | 1/5 (27%) | 5/5 (100%) | **5/5** (100%) |
| Bicycle Tuning | 1/5 (17%) | 5/5 (96%) | **5/5** (100%) |
| Repotting a Houseplant | 2/5 (44%) | 5/5 (90%) | **5/5** (100%) |
| Shipping a Parcel Overseas | 0/5 (24%) | 5/5 (80%) | **5/5** (99%) |
| Sharpening a Kitchen Knife | 1/5 (32%) | 4/5 (90%) | **5/5** (100%) |

**30/30 adapter, 5/30 base**, step order 100% on every one. Asked only `Sharpen a kitchen knife`, with nothing about the procedure in context, the adapter returns the saved steps verbatim while the base model gives a perfectly reasonable but completely different answer about whetstones.

One correction worth recording, because it moved the numbers. The first run of this table scored the baseline higher, and it was the metric's fault: `_keywords` counted the bare step numbers `1`–`5` as vocabulary, so any numbered reply collected them for free. The base model's whetstone procedure scored **68%** against a 60% threshold and *passed* a task it had answered differently. Enumerators are now stripped before keywords are extracted (`20-degree` survives; `3.` doesn't), which took that task from 68% to 32%. The adapter arm never moved — it reproduces the steps either way — so the leak had only ever been flattering the baseline the whole comparison rests on.

By default the harness generates five task phrasings, deliberately worded unlike the training seeds so a pass means recall rather than memorised strings. Drop your own in `training_data/workers/<role>/eval_tasks.json` to use a real battery:

```json
[
  {"id": "no_wifi", "prompt": "wifi's dead again", "must_include": ["toggle"]},
  "the network dropped, sort it out"
]
```

| Flag | Default | Note |
|---|---|---|
| `--output` | timestamped file | Where to write the JSON report |
| `--threshold` | `0.6` | Step-coverage fraction required to pass |
| `--max-tokens` | `400` | Max reply tokens per task |
| `--arms` | all three | Subset of `base,prompted,adapter` |

Because the seeds are rendered with the headmaster's chat template but a worker trains the model named in its own catalog entry, the two can drift apart after a model switch. Training now refuses to run on data tokenized for a different model rather than silently learning another model's turn markers.

## Hardware prerequisites 
Symbio (for now) runs on Apple Silicon using MLX and Metal performance shaders
- **Recommended Unified RAM requirements** 16gb (the program itself takes 8 but overhead and expansion so comfortably would be 16)
- **Minimum architecture** any m-series chip ideally. (if you can let me know if it works for m1,m2,m3,m5 and the pro and max variants)

## Quick start

```bash
./install.sh
```

That is the whole install. It checks you are on Apple silicon with enough RAM
and disk, creates its own virtualenv, installs everything, fetches the browser
engine, and then **drops you into a shell with the environment already
active** — so there is nothing to remember to activate. Type `exit` to leave;
your original shell is untouched.

Then:

```bash
symbio        # or the short alias: symb
```

First launch runs an interactive setup wizard (names, model preset, speed mode,
and toggles for browser, web search, MOA dispatch and Telegram), downloads the
model, and spends ~25 seconds warming its prompt cache. Later starts take about
a second, because that cache is saved to disk. Re-run the wizard any time with
`symb setup`.

<details>
<summary>Install options</summary>

```bash
./install.sh --prefetch-model   # download the 4.1 GB model now, not on first run
./install.sh --no-browser       # skip the ~150 MB Chromium download
./install.sh --with-native      # include the experimental symbio_native extras
./install.sh --dev              # include dev/test dependencies
./install.sh --no-shell         # install only; do not enter the environment
./install.sh --venv PATH        # put the virtualenv somewhere else
```

Requirements are checked before anything is downloaded: macOS on Apple silicon
(MLX has no CPU fallback), Python 3.10+, ~8 GB free disk, and 16 GB of RAM for
the default 8B model — less RAM works, but pick a smaller model in the wizard.

</details>
## How it works

1. **You talk to the AI** — Ask it anything
2. **It makes mistakes** — Sometimes gets it wrong
3. **You correct it** — "No, it's actually..."
4. **It learns** — Saves the correction
5. **After 5+ corrections, it fine-tunes itself**
6. **Next time: it gets it right** :)))

That's it. No manual training. No API calls. All local.

[See it in action](#example-screenshot)

If you prefer an isolated, non-editable install (e.g. with `pipx`):

```bash
pipx install .
# or, from any directory containing this repo:
pipx install /path/to/agi
```

Make sure `~/.local/bin` (or your pip/pipx bin directory) is on your `PATH`.

On first run, Symbio asks for your name and its name. These are saved to `config.json`.
## Why custom?
AI agents tend to forget and also not personalised to the work you want the agent to do, as well as the agent being in the cloud which brings on the costs and privacy risk. This repo helps you have access to a highly aggressive persoanlised model that does not leave your machine unless you ask it to.

## Configuration

Edit `config.json` to change the model, LoRA settings, or agent behavior. You can also use the CLI:

```bash
symb config                    # show full config (bot token redacted)
symb config get agent.temperature
symb config set agent.temperature 0.7
symb config set telegram.allowed_chat_ids '[123456789]'
```

| Key | Default | Note |
|---|---|---|
| `model_name` | `Qwen/Qwen3-0.6B` | Base MLX model (the setup wizard offers larger presets) |
| `assistant_name` | `Symbio` | What the assistant calls itself |
| `user_name` | *(asked at first run)* | Your name |
| `agent.max_tool_rounds` | `3` | Max tool rounds per user turn |
| `agent.temperature` | `0.7` | Sampling temperature |
| `agent.tool_use_temperature` | `0.2` | Lower temperature used when a tool call is expected |
| `agent.max_reply_tokens` | `128` | Max tokens generated per reply |
| `agent.prompt_cache_enabled` | `true` | Reuse the warmed system-prompt KV cache across turns |
| `agent.persist_prompt_cache` | `true` | Save that cache to `cache/` and reload it on start |
| `lora.rank` | `8` | LoRA rank (adapter width) |
| `lora.dropout` | `0.0` | LoRA dropout to reduce overfitting |
| `lora.scale` | `20.0` | LoRA adapter scale |
| `lora.num_layers` | `8` | Number of layers to attach adapters to |
| `lora.iters` | `300` | LoRA iterations per `/train` run |
| `lora.max_seq_length` | `512` | Maximum sequence length during training |
| `lora.learning_rate` | `1e-4` | LoRA learning rate |
| `lora.save_every` | `100` | Checkpoint frequency during training |
| `lora.steps_per_eval` | `100` | Iterations between validation passes |
| `lora.early_stop_enabled` | `true` | Stop once validation loss plateaus |
| `lora.early_stop_patience` | `2` | Validation passes without improvement before stopping |
| `lora.early_stop_min_delta` | `0.005` | Improvement below this counts as no improvement |
| `learn.boost_factor` | `3` | Copies of a correction sample written to training data |
| `learn.batch_train_iters` | `25` | Iterations for threshold-triggered correction training |
| `learn.mistake_threshold` | `5` | Mistake notes collected before auto-training runs |
| `telegram.bot_token` | *(prompted)* | Telegram bot token from @BotFather |
| `telegram.allowed_chat_ids` | `[]` | Chat IDs allowed to use the bot (required) |

## CLI

After installing (`pip install -e .`) the `symbio` and `symb` commands are available. During development you can use the `symb` wrapper script in the repo root.

```bash
symb                    # Start interactive chat
symb chat               # Same as above
symb config             # Open interactive config editor
symb config show        # Print config.json (token redacted)
symb config get <key>   # Print one value, e.g. agent.temperature
symb config set <key> <value>
symb train              # Run LoRA training
symb skill list         # List saved skills and their adapter status
symb skill new <name>   # Create a new skill (interactive steps)
symb skill rm <role>   # Delete a skill, adapter, and training data
symb skill eval <name>  # Score a skill: base vs prompt-only vs adapter
symb eval-lora          # Benchmark the headmaster adapter against the base model
symb archive            # Run auto-archive for idle notes/adapters
symb archive --dry-run  # Preview what would be archived
symb archive --restore note|adapter <name>
symb gateway status     # Check Telegram gateway readiness
symb gateway start      # Start the Telegram bot
symb gateway stop       # Stop a running gateway
```

Legacy `python main.py` flags still work:

```bash
python main.py --telegram
python main.py --train
```

## Telegram bot

Run Symbio as a Telegram bot so you can chat from your phone:

```bash
symb gateway start
# Legacy equivalent: python main.py --telegram
```

Check gateway readiness first:

```bash
symb gateway status
```

On first run you will be prompted for a bot token from [@BotFather](https://t.me/botfather). The token is saved to `config.json`. For better security, set the environment variable `SYMBIO_TELEGRAM_TOKEN` instead; it overrides the config file.

You must add your Telegram chat ID to `telegram.allowed_chat_ids`:

```bash
symb config set telegram.allowed_chat_ids '[123456789]'
```

Send any message to the bot, then copy the chat ID from the refusal message if you haven't set it yet.

Dangerous actions from Telegram — blocked shell commands, new browser domains, Python code, config changes, cron jobs, digest, and training — ask for approval via an inline keyboard before running.

### Telegram slash commands

| Command | Description |
|---|---|
| `/start` | Welcome message |
| `/help` | List available commands |
| `/ping` | Last turn latency breakdown |
| `/status` | Model, adapter, data, and last turn timings |
| `/golden` | Run golden-set regression check |
| `/train` | Start LoRA training |
| `/selfcheck` | Verify enabled features and auto-fix safe issues |
| `/setup` | How to change configuration |
| `/tools` | Toggle tool groups |
| `/cancel` | Clear the current session |

## Slash commands

| Command | Description |
|---|---|
| `/quit` | Exit the chat |
| `/save` | Save the current conversation to training data |
| `/train` | Run LoRA fine-tuning and reload the adapter |
| `/train_worker <role>` | Train one worker's adapter, golden-checked and rolled back on regression |
| `/resume` | List work a previous session didn't finish (`run` to do it, `clear` to drop it) |
| `/learn` | Manually learn from your last correction (auto-learn is on by default) |
| `/digest` | Convert notes into training samples |
| `/note [title]` | Create a markdown note |
| `/notes` | List saved notes |
| `/new-skill <name>` | Create a skill note and start training a worker adapter |
| `/skills` | List saved skill notes |
| `/skill-adapters` | List skill adapters and their training/idle status |
| `/archive` | Archive idle notes/adapters (or preview with `--dry-run`) |
| `/restore note|adapter <name>` | Restore an archived note or adapter |
| `/status` | Show model, adapter, notes, and session info |
| `/selfcheck` | Verify enabled features and auto-fix safe issues |
| `/setup` | Re-run the setup wizard (names, model, features) |
| `/compact` | Compress memory/profile store and archive the original |
| `/model` | List model presets |
| `/model <preset>` | Switch to a named model preset (restart to load) |
| `/run <cmd>` | Run a sandboxed shell command |
| `/forget_last` | Remove the last exchange from history |
| `/prune` | Remove stale adapter checkpoints |
| `/tidy` | Prune junk notes and duplicate session turns (`/tidy dry` to preview) |

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
4. When `learn.mistake_threshold` (default 5) notes have accumulated, digest them into `training_data/train.jsonl` and run a short LoRA update (`learn.batch_train_iters`, default 25).
5. Archive the used mistake notes to `notes/mistakes/archive/` and reload the adapter.

The `/learn` command is still available to force a mistake note from the last correction, but it is no longer required.

### Learning from its own tool mistakes

The same mistake-note pipeline also captures a second, fully automatic pattern that needs no user involvement at all: a tool call that fails, immediately followed by one that works. This is exactly the "wrong command, try the right one" pattern already hand-seeded into every install's base training data —

```
You:      Open Chrome.
Symbio:   <cmd>chrome</cmd>
          [Tool: run_command]
          [Observation] Command 'chrome' exited error.
                        Output:
                        Command not found: chrome
Symbio:   'chrome' isn't a command here — trying the native way. <cmd>open -a 'Google Chrome'</cmd>
          [Learn] Tool mistake captured: 20260721_213045_System_observation_Command_chrome.md
```

— except now it's learned from real usage, not just the seed examples. It feeds into the exact same `notes/mistakes/` → threshold → digest → guarded-training pipeline as conversational corrections above, so both count toward the same `learn.mistake_threshold`. Nothing is saved if the model keeps failing without ever finding a working alternative within the turn — only a confirmed fix gets captured.

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

## Fine-tuning details

Symbio uses **LoRA** (Low-Rank Adaptation) via Apple's **MLX-LM** framework. The base model weights stay frozen; only small adapter matrices are trained on curated conversation, notes, and corrections. Training is invoked through the official `mlx_lm lora` CLI:

```bash
symb train            # full pass using lora.iters
```

The resulting adapter is saved to `adapters/` and loaded automatically on the next start.

| Setting | Default | What it controls |
|---|---|---|
| `lora.rank` | `8` | Width of the low-rank matrices |
| `lora.num_layers` | `8` | How many transformer layers get adapters |
| `lora.scale` | `20.0` | Adapter output scaling |
| `lora.dropout` | `0.0` | Dropout for regularization |
| `lora.learning_rate` | `1e-4` | Training step size |
| `lora.iters` | `300` | Iterations for `/train` |
| `lora.max_seq_length` | `512` | Training context length |
| `lora.save_every` | `100` | Checkpoint frequency |

Training stops early once validation loss plateaus for `lora.early_stop_patience` passes, then promotes the best checkpoint to `adapters.safetensors`. It will not stop before a checkpoint exists — with `save_every` at 100, a run that plateaus at iteration 60 keeps going rather than ending with no weights at all.

### Golden set: catching a fine-tune that silently breaks things

Every LoRA update (`/train`, the `train_adapter` tool, the auto-training that follows enough corrections, or the end-of-session prompt) is checked against a small, fixed **golden set** — prompts that exercise behavior baked into every install's seed training data: stating its own name, not confusing itself with the user, emitting the right tool tag for code/notes/reminders/search, and not degenerating into repeated phrases. Each check is single-turn and side-effect-free (no tool is actually executed), so it's safe to run automatically.

Symbio grades the golden set before training (the baseline) and again after reloading the new adapter. If a case that passed before now fails, it's a **regression**, and the previous adapter is restored automatically:

```
  [Golden] Regression: 2 case(s) newly failing (run_code_for_math, web_search_unknown).
  [Golden] Rolled back to the previous adapter.
```

Run it manually anytime with `/golden` to see the current pass/fail breakdown without training.

| Key | Default | Note |
|---|---|---|
| `learn.golden_set_enabled` | `true` | Grade every LoRA update against the golden set |
| `learn.golden_rollback_on_regression` | `true` | Automatically restore the previous adapter on a regression |
| `learn.golden_regression_threshold` | `0` | Newly-failing cases allowed before it counts as a regression |
| `learn.golden_max_tokens` | `150` | Max tokens generated per golden-set case |

### Keeping the corpus clean

A self-training agent has a failure mode a normal chatbot doesn't: its own bad output becomes tomorrow's input. A junk note gets retrieved, echoed, logged, and retrieved again — and if it survives to a digest, it gets trained in. Two mechanisms push back.

**Self-pruning.** On boot (and on demand with `/tidy`) Symbio archives notes that carry no information — empty bodies, degenerate titles, questions with no subject — and collapses duplicate session turns down to `prune.session_max_copies`. Notes are moved to `notes/archive/`, never deleted, and `# Skill:` and identity notes are protected. Preview first with `/tidy dry`.

**Retrieval hygiene.** Anything that looks like machinery is excluded from retrieval rather than fed back as prose: tool transcripts, any tool-call syntax (including the legacy short tags), and the `[System observation: ...]` scaffold the agent uses internally to hand tool results back to the model.

**Retrieval can also return nothing.** Scoring is IDF-weighted, which ranks well but has no way to say *none of these*. Any note sharing a single common word scored above zero and made the cut, so a query whose topical words matched nothing came back with the three least-bad notes — as confidently as a real match. That is not merely a relevance problem here: a retrieved note is pasted into the model's context, skill notes are procedures, and the model performs them. Measured, before the fix: a request to save a bicycle-tuning skill retrieved the Browser Driver note and produced an eighteen-fold repetition of "Clicking the Steps. Scrolling down." and a real `browser_open` on google.com.

A note now has to share a term that is both **rare** (present in at most a third of notes) and **not a stopword** — the frequency test alone isn't enough, because in a corpus of numbered procedures words like "how" and "do" are genuinely rare and were qualifying unrelated notes on their own. `skill` and `what are the steps` now retrieve nothing, since they distinguish nothing; every skill still retrieves its own note first.

That last one matters more than it sounds. The scaffold appears in a sixth of the seed training corpus — always as a *user* turn, always followed by an assistant reply — so a model can learn to write the scaffold itself and then answer its own invented observation, on repeat:

```
Huy     : yo
Caine   : system observation: User says 'yo' — how can I help?
          system observation: User says 'yo' — how can I help?
          system observation: User says 'yo' — how can I help?
```

Symbio now detects a reply impersonating the scaffold (matching case- and bracket-insensitively, so near-misses like the one above are caught) or looping a single line, discards it, and regenerates once. The check runs *before* the reply is printed or written to the session store, so a discarded turn never reaches `sessions/` and can never be retrieved or digested into training data later.

The one exception is streaming: with `agent.stream_output` on, tokens have already been emitted to your terminal by the time the reply can be judged, so you may briefly see the start of a bad reply before the `[Echo]` notice replaces it. Nothing is persisted either way.

| Key | Default | Note |
|---|---|---|
| `prune.enabled` | `true` | Enable self-pruning |
| `prune.on_boot` | `true` | Run a prune pass at startup |
| `prune.notes` | `true` | Archive junk notes |
| `prune.sessions` | `true` | Collapse duplicate session turns |
| `prune.session_max_copies` | `2` | Copies of a repeated turn kept |

### Idle-adapter reminders

If a trained adapter exists on disk but the current session isn't using it (most commonly after switching `model_name` to something the adapter isn't compatible with), Symbio tracks how long it's sat unused. Past `learn.adapter_idle_days`, it asks once whether to remove it:

```
  A saved LoRA adapter hasn't been used in 45 day(s) (not loaded with the
  current model). Remove it to free up space? [y/N]:
```

Answering yes deletes it; declining or saying "keep" both just leave it alone and reset the grace period, so the reminder won't repeat until it's been idle that long again. Nothing is ever removed without an explicit yes. Check the current status anytime with `/status`.

| Key | Default | Note |
|---|---|---|
| `learn.adapter_idle_reminder_enabled` | `true` | Ask about removing an adapter that's gone unused |
| `learn.adapter_idle_days` | `30` | Days unused before the reminder fires |

### Surviving a crash mid-fine-tune

A fine-tune is minutes of GPU time that exists only in RAM until it writes an adapter, and on a unified-memory Mac the process can be killed outright — no unwind, no traceback, nothing written down. What that used to cost wasn't the compute, it was *the knowledge that the work was owed*: a skill saved with auto-train would be seeded, never trained, and nothing on the next start remembered it was supposed to be.

Expensive work is now recorded in `logs/pending_tasks.json` **before** it begins, written atomically so a crash mid-write can't truncate it. Whether the owning process is still alive is decided by pid *and* boot id together — after a reboot the kernel reissues low pids, so a pid alone would call a dead trainer healthy forever.

On the next start:

```
  [Resume] training for worker 'fix_wifi': interrupted mid-run; restored the
           adapter from the backup it left behind. That backup is kept at
           adapters/workers/fix_wifi.FIX_WIFI_ITER150.bak — delete it once
           you are happy.
  [Resume] 1 unfinished task(s) carried over. Run /resume to pick them up,
           /resume clear to drop them.
    - training for worker 'fix_wifi' — owning process died before it finished
```

Repair is automatic, because a truncated adapter sitting next to a complete backup has only one right answer. Re-running the training is **not** — that is minutes of GPU and a second full copy of the weights, and starting one unprompted at boot is a fair description of how the machine went down in the first place. `/resume` lists, `/resume run` runs them one at a time (headmaster last), `/resume clear` drops them. `/status` shows the same list.

Work that was *refused* is recorded the same way. When the memory preflight declines a run there is no adapter and no error — just a line in a log nobody is reading, which is exactly how a skill ends up permanently untrained. It stays on the list instead, with the reason:

```
  Unfinished tasks: 1 (/resume)
    - training for worker 'coffee_making' — training did not produce an
      adapter (see the log; most often not enough free memory at the time)
```

## Example screenshots
<img width="1300" height="89" alt="Screenshot 2026-07-23 at 11 23 21 am" src="https://github.com/user-attachments/assets/c4e02593-f527-44dc-9bcb-181f329360ad" />
<img width="272" height="475" alt="Screenshot 2026-07-23 at 11 22 52 am" src="https://github.com/user-attachments/assets/e8e7475a-aac8-455b-b978-3996f1d4d3fd" />

## Example video of Symbio opening up google chrome then clicking a button
https://github.com/user-attachments/assets/9e910d11-d204-4fb1-b42f-e09dd6243d20

## Mixture of agents: delegating to smaller worker models

One model doing everything — from picking a browser click to answering a factual question — means every micro-decision pays the cost of the headmaster's full system prompt and persona. Symbio can instead hand a bounded sub-task off to a smaller, faster **worker** model, and each worker can be fine-tuned independently on its own narrow task, with its own adapter, separate from the headmaster's.

This is off by default (`dispatch.enabled: false`) — it loads and runs additional models on your machine, a bigger resource commitment than anything else here, so it's opt-in.

### How it works

The headmaster requests delegation the same way it requests any other tool:

```
<delegate role='summarize'>the full text to condense</delegate>
```

or the Hermes form: `<tool_call>{"name": "delegate_task", "arguments": {"role": "summarize", "task": "..."}}</tool_call>`.

`symbio/app/worker_models.json` is the catalog of available workers — model, role, description, rough memory footprint. Ships with two roles:

| Role | What it does |
|---|---|
| `summarize` | Condenses page/document text handed off by the headmaster |
| `browser` | Picks the next click/type/scroll action from the current page text, in a bounded loop, using the same `BrowserSession` the headmaster's own browser tools drive |

Workers load lazily on first use and are evicted LRU-style once `dispatch.max_resident_workers` is exceeded, or after sitting idle past `dispatch.worker_idle_unload_minutes` — sequential by default (one resident worker) to fit alongside the headmaster on a typical machine, but this is a real, working setting: raise `max_resident_workers` if you have the RAM to keep several loaded at once.

### Fine-tuning a worker

Every delegated task's (input, output) pair is recorded as a training sample under that worker's own data directory (`training_data/workers/<role>/`) — real usage builds the corpus. Training a worker reuses the exact golden-set-guarded-rollback machinery the headmaster's own `/train` uses: a small, role-scoped golden set (e.g. "does the browser worker still reply with a known action verb") is checked before and after training, and a regression rolls the worker's adapter back automatically, the same way `_guarded_train` protects the headmaster's. Worker adapters live under `adapters/workers/<role>/`, fully separate from the headmaster's own `adapters/`.

| Key | Default | Note |
|---|---|---|
| `dispatch.enabled` | `false` | Turn on delegation |
| `dispatch.max_resident_workers` | `1` | How many worker models can be loaded at once |
| `dispatch.worker_idle_unload_minutes` | `10` | Unload a worker after this long unused |
| `dispatch.max_worker_rounds` | `4` | Round cap for a multi-step worker task (e.g. browser) |
| `dispatch.worker_golden_set_enabled` | `true` | Golden-check a worker's adapter around training |
| `dispatch.worker_golden_rollback_on_regression` | `true` | Auto-rollback a worker's adapter on regression |
| `dispatch.hot_swap_adapters` | `true` | Switch between workers sharing a base model by replacing only their LoRA tensors |
| `dispatch.headmaster_deep_sleep_while_workers` | `false` | Unload the headmaster while a worker runs, for machines that cannot hold both |
| `dispatch.allow_second_headmaster_copy` | `false` | Permit a worker on the headmaster's own model to load a second full copy |

### One model at a time

The failure mode on a unified-memory Mac is never one model being too big — it is two of them resident at once. Three rules keep that from happening:

- A worker running the **headmaster's own model** is refused rather than loaded, because that is a second copy of the largest allocation on the machine. Hot-swapping its LoRA tensors onto an already-resident model costs ~19 MB instead of gigabytes, and is tried first. The refusal names the reason and tells you which setting overrides it.
- With `headmaster_deep_sleep_while_workers` on, the headmaster unloads before a worker loads **and the worker unloads before the headmaster comes back**. Reloading it on top of a still-resident worker is the same double residency moved one turn later — which is exactly how this machine went down. The cost is a cold worker on the next delegation; that is the price of the setting.
- Every eviction hands the memory back immediately. Dropping the reference is not the same as freeing the weights: MLX keeps them charged to the process until the allocator reclaims, so an eviction without that leaves the old worker fully resident right through the load of its replacement.

Training gets its own preflight. A run is refused, with the arithmetic shown, when the machine cannot hold it — and refusing is not the same as forgetting, so the run goes on the `/resume` list instead of vanishing:

```
  [Train] Not enough free memory to train safely: the run needs about 4.1 GB
          (2.3 GB of weights plus optimiser state and retained activations)
          and only 3.1 GB is free. Skipping rather than risking an
          out-of-memory kill.
```

Enabling dispatch also needs `"delegate"` in `tools.enabled_groups` — it's included by default going forward, but an existing `config.json` written before this feature won't have picked it up automatically; add it with `/config set tools.enabled_groups '[...]'` if delegation seems to silently do nothing.

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
- And also do pay attention to the 'Do you wanna yes/no' questions those are there to keep you from having Symbio do random stuff without your consent because you might not want to do it.
- **A "no" ends the action for the whole turn, not just for the tool that asked.** Answering `N` to a browser domain prompt used to leave the model free to reach the same end another way — it ran `open -a 'Google Chrome'` through the shell on the next round and reported success. The sandbox is a *denylist*, so `open` was never on it, but the gate you answered was about the action, not about which tool happened to ask. A refusal is also not a retryable error: no second attempt turns a "no" into a "yes", and retrying one only puts the identical prompt in front of you again. The reply now says plainly that it did not happen, instead of describing the blocked action as done.

## Architecture

The project is organized as a `symbio/` Python package with a thin `main.py` wrapper:

```
.
├── main.py              # Delegates to the modern CLI in symbio/app/cli.py
├── symbio/
│   ├── constants.py     # Paths, DEFAULT_CONFIG
│   ├── app/
│   │   ├── cli.py         # symbio / symb command-line interface
│   │   ├── chat.py        # ChatSession, agent loop, slash commands
│   │   ├── config.py      # Defaults, loading, redaction, token prompt
│   │   ├── training.py    # Training data and LoRA fine-tuning via mlx_lm
│   │   ├── learn.py       # Correction detection and batch learning
│   │   ├── golden.py      # Golden set: regression checks around every LoRA update
│   │   ├── eval.py        # Held-out eval set: headmaster adapter vs base model
│   │   ├── skill_eval.py  # Three-way skill scoring: base vs prompted vs adapter
│   │   ├── prune.py       # Self-pruning of junk notes and duplicate session turns
│   │   ├── pending.py     # Durable journal of in-flight work; survives a kill (/resume)
│   │   ├── dispatch.py    # MoA: WorkerPool, delegated tasks, worker fine-tuning
│   │   ├── worker_models.json  # Catalog of available worker models/roles
│   │   ├── memory.py      # Notes, memory, profile management
│   │   ├── sandbox.py     # Sandboxed commands and Python execution
│   │   ├── computer.py    # Browser automation helpers
│   │   ├── cron.py        # Scheduled jobs and reminders
│   │   ├── telegram.py    # Telegram bot gateway
│   │   ├── tooling.py     # Tag parsing and tool stripping
|   |   ├── prompts.py     # just the prompt idk nothing special
|   |   └── skills.py      # saves the skills as adapters as well as manage the notes/
│   └── utils.py         # Shared helpers
├── rag.py               # Lightweight keyword-based RAG
├── README.md
├── docs/
│   └── adapter-marketplace.md  # Design doc, not yet implemented
├── config.json          # User configuration
├── models.json          # Model presets
├── notes/               # Markdown notes / memory
├── training_data/       # train.jsonl and valid.jsonl (workers/<role>/ for MoA workers)
├── adapters/            # LoRA adapter weights (workers/<role>/ for MoA workers)
├── cache/               # Persisted system-prompt KV cache

├── logs/                # Session logs, plus pending_tasks.json (unfinished work)
├── sessions/            # Session stores
├── screenshots/         # Browser screenshots
└── sandbox/             # Scratch space for code execution
```

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
- [x] **Add More Tools** — live browser (`<browse>`/`<click>`/`<type>`/`<scroll>`), `<skill>`, permission-gated sandbox
- [x] **Self correction when hallucinating**
- [x] **Be able to learn new skills on the fly**
- [x] **Remember new info found from web research** — auto-saved as `Learned:` notes, trained in on digest
- [x] **Add Telegram bot** — full tool loop with inline-keyboard approval for dangerous actions
- [x] **Mixture of agents** — headmaster delegates bounded sub-tasks to smaller, independently fine-tunable worker models (`dispatch.enabled`, off by default)
- [ ] **Adapter marketplace** — design doc: [docs/adapter-marketplace.md](docs/adapter-marketplace.md); not yet implemented
- [ ] **Add Other Messaging Platforms**
- [ ] **Prune Old Weights (Future Milestone)**
## Licence
Apache 2.0

---

⭐ If Symbio is useful to you, a star helps others find it.
