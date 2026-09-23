### CUDA SUPPORT NEEDED
# Symbio - that fine tuning agent.

> **Learns from your corrections, on your machine — and rolls back a fine-tune that made it worse.**
>
> Runs on your Mac. Remembers what matters. Learns new skills. Fine-tunes itself with LoRA. No cloud inference. No subscription.

[![Live Demo](https://img.shields.io/badge/%F0%9F%A4%97-Live%20Demo-yellow)](https://huggingface.co/spaces/HuyEdits/symbio-demo)
[![GitHub](https://img.shields.io/badge/GitHub-Symbio-black?logo=github)](https://github.com/huyedits/Symbio)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](#license)

**[Try the interactive demo](https://huggingface.co/spaces/HuyEdits/symbio-demo)** · **[Quick Start](#quick-start)** · **[How it learns](#how-it-learns)** · **[Roadmap](#roadmap)** 

---

## APGI — artificial personalised general intelligence

A term coined for this project, because the thing being built is not on the
road to AGI and is not trying to be.

AGI is one mind that is general across everyone. **APGI is one mind that is
general across everything YOU do** — the same breadth of capability, narrowed
to a single person, a single machine, and the particular way they work. It is
not a smaller AGI. It is a different target, and most of what makes it hard is
different too.

What it implies in practice, all of which this repository is an attempt at:

- **The weights change for you, not for a population.** A correction becomes a
  LoRA update on your machine, guarded by a battery that rolls it back if it
  made anything worse. Nobody else's install gets it.
- **It learns an environment by acting in it.** Given a world it has never seen
  it earns its own corpus: it acts, the world grades it by changing or not
  changing, and only what worked is kept. Measured here at 10/45 to 26/45, and
  six-step chains from 0/19 to 7/19, with no human writing a sample.
- **It is one installation, not a service.** No cloud inference, no shared
  fine-tune, no telemetry on by default. What it learns about you cannot leak
  into anyone else's copy, because there is no anyone else's copy.
- **Personalisation is the general capability.** The skills, the soul store,
  the constitution and the per-skill adapters exist to make it general across
  YOUR work rather than average across everyone's.

The honest limits are in this README too, measured rather than asserted: what
it cannot bootstrap, where the harness was the thing in its way, and which
numbers are one run rather than a result.

## What is Symbio?

Most AI agents have a problem:

**They forget.**

*like forget forget*,
get it? Cause it was in short term memory and you might shut down your computer? 

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
*  **Fine-tunes itself when it decides to** — besides training automatically once enough examples pile up, the model can call its own `train_adapter` tool mid-conversation. With default settings that runs without asking you; it is logged, and the golden set still rolls back a run that made things worse.
*  **Watches its own weights change** — `scripts/weight_delta.py` shows, module by module, how far each fine-tune moved the weights, and how much the latest run moved them compared with the one before.
*  **Looks inside the model (experimental)** — record which neurons fire on your own traffic, and grow new neurons that start at exactly zero change. See [Looking inside the weights](#looking-inside-the-weights).
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
---

## Demo

### Live browser demo

Try the real tag parser, correction miner, research memory and RAG retriever in your browser:

**[https://huggingface.co/spaces/HuyEdits/symbio-demo](https://huggingface.co/spaces/HuyEdits/symbio-demo)**
(no fine tuning just actions)
### Screenshots

<img width="1300" alt="Symbio CLI" src="https://github.com/user-attachments/assets/c4e02593-f527-44dc-9bcb-181f329360ad" />

<img width="272" alt="Symbio mobile interface" src="https://github.com/user-attachments/assets/e8e7475a-aac8-455b-b978-3996f1d4d3fd" />

### Browser automation

Symbio can use a live browser to perform tasks such as opening Chrome and interacting with pages. wowie

[https://github.com/user-attachments/assets/9e910d11-d204-4fb1-b42f-e09dd6243d20](https://github.com/user-attachments/assets/9e910d11-d204-4fb1-b42f-e09dd6243d20)

### Desktop control

Symbio drives the Mac itself, not only a browser page. It reads the frontmost
window through the macOS accessibility API — the same interface a screen
reader uses — so `see_screen` comes back with the window's real controls:

```
Notes — window "Shopping"
   1 Button        'New Note' at (48,96) 28x28
   2 TextArea      '(empty text field)' at (320,140) 600x420
   3 Button        'Share' at (980,96) 28x28
```

Every action then takes a number: `desktop_click {"element": 1}`,
`desktop_type {"element": 2, "text": "..."}`, plus `desktop_press` (chords
included: `cmd+shift+4`), `desktop_scroll`, `desktop_drag`, `desktop_move`,
`desktop_wait` and `open_app`. Nothing here loads a vision model, so the
listing costs no RAM beside the headmaster, it has no minimum control size,
and a control that is absent from the tree is genuinely not on screen.

Two things the tree gives that a screenshot cannot:

- **Typing goes into a named field.** Keys sent at a window with no text field
  focused are not discarded, they are shortcuts. `desktop_type` refuses to
  type blind when the focused control cannot take text, and names the field to
  use instead.
- **A click that changed nothing says so.** The window title and the focused
  control are read before and after, so "clicked" is not reported as "done".

Vision stays as the fallback for windows that draw their own interface — a
canvas, a game, a screen share — where there is no tree to read.

**Permission:** macOS returns an empty tree rather than an error until the
terminal running Symbio is granted Accessibility. `python3 -m symbio.ax` shows
the grant dialog and then prints the live control listing; `/selfcheck` in
chat reports the same thing.

### The window

```bash
symb daemon start          # loads the model once, in its own process
symbio-desktop             # opens the chat window in your browser
```

A chat-first interface: conversation in the middle, the agent's own tool calls
folded into one line you can open, approval prompts inline (the same gate the
terminal shows), and the adapter map, skills, corpus and health report behind a
drawer you open when you want them.

It is deliberately not an Electron app. The server is the Python standard
library, the model lives in `symb daemon` and is reached over a Unix socket,
and the page is plain HTML and CSS — **28 MB resident idle, 31 MB after serving
every endpoint**, against the 200-400 MB an Electron shell starts at.
`--window` opens a native WKWebView instead of a browser tab; that hosts WebKit
in this process and costs about 400 MB all told, which is why it is off by
default.

### Staying online

```bash
symb daemon start     # load the model once, in its own process
symb watch            # keep it loaded, and run what is due
symb watch --status   # the last heartbeat
```

Scheduled jobs used to run inside a chat session's background thread, so
"every morning at 8" meant "every morning at 8, if a window happens to be
open". `symb watch` is the process that watches instead: it restarts the
resident model when it dies (with a backoff, because each attempt maps several
GB of weights), ticks the cron table itself, and hands each fired job to the
model as an ordinary turn.

**Unattended turns are denied by default.** The model asks its client before a
risky tool, and here the client is a loop — nobody is reading the prompt. The
answer is no, and the refusal is recorded in the heartbeat.
`cron.unattended_approve` turns that off for someone who has read this
paragraph.

### Posting to X

```
post_to_x  {"text": "shipping the desktop window today"}
```

The browser must already be open at x.com and signed in — it does not navigate
there on its own, because posting is not something to do on a page nobody
asked for. It fills the composer **by selector** (the composer is 28px tall,
under the vision model's one-patch floor, and a coordinate for it missed by
~36px every time), sends with x's own Post button, then reads the timeline back
and returns a verdict the model cannot shape:

```
[Post CONFIRMED] The post is rendered on the timeline: 'shipping the desktop window today'
[Post NOT confirmed] The composer still holds the text, so it was not sent.
[Post NOT confirmed] The composer cleared but the post is not on the timeline yet …
```

A cleared composer is not confirmation — a discarded draft clears too. This
project has already posted something and reported that it had not, which is
why the proof is read from the DOM rather than from a toast.

---

## Quick Start

## Requirements

Symbio currently targets **Apple Silicon Macs** using Apple's MLX stack. (unfortunately until we can get support for CUDA, and other stuff)

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

Two ways, and you only need the repository for one of them.

### Just run it

```bash
pip install "symbio-cli[mlx]"
symb setup          # names, model, first-run checks
symb chat
```

Nothing is written into your Python environment. Everything this install
keeps — notes, adapters, training data, sessions, `config.json`, `prompt.md` —
lives in a workspace at **`~/.symbio`**, and `SYMBIO_HOME` points it somewhere
else if you would rather. Upgrading the package does not touch it; you can
delete the virtualenv and the agent still remembers everything.

The `[mlx]` extra is the inference and training engine. It is macOS/Apple
Silicon only and deliberately optional, because `mlx` publishes no sdist and a
hard dependency would make `pip install` fail during resolution on Linux
before anyone saw an explanation.

### Work on it

```bash
git clone https://github.com/huyedits/Symbio
cd Symbio
./install.sh
```

A checkout is its own workspace: run from a clone and the agent uses the clone,
which is what you want while editing it. You need the repository for this and
not for the above — the tests, the benchmark harness, the desktop app sources
and the installer script are all development things.

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

## How it learns

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

The threshold is not a constant. It scales with the corpus, because every
sample already in `train.jsonl` competes with the new ones: five boosted notes
are most of an epoch against 50 samples and a rounding error against 882. It
rises by log, not by multiple — a corpus ten times bigger needs a bit more
evidence, not ten times as much — and severe mistakes pull it back down.
Clamped to 2..20 either way; `learn.mistake_threshold` sets the base and
`learn.scale_threshold_with_corpus` turns the scaling off.

## Not every mistake asks for the same training

A correction is one of two things, and they want opposite recipes:

| | What it is | What the fix needs |
| --- | --- | --- |
| **Knowledge** | the model said something false — a name, a version, which file holds what | passes over the corpus; a fact repeated four times in one epoch is still one fact |
| **Reflex** | the model reached for the wrong SHAPE — a tool that does not exist, a guessed argument, a GNU flag on BSD, a click before a look | repetition of the exact form; an action learned once is not automatic |

Every mistake note now records its `**Kind:**`, classified at capture time,
and the batch decides the dials: a batch of wrong shapes repeats harder and
runs shorter, a batch of wrong facts trains at ordinary weight for the full
run, and a mixed batch lands between. One batch is still one LoRA pass — this
machine trains one model at a time.

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

## Mistakes in the environment, not just in the answer

The same loop runs over the machine itself. A desktop is full of failures that
come back looking like ordinary results, and each one used to end the turn:

| What came back | What it actually means |
| --- | --- |
| `No results found.` | the search produced no work |
| `Timed out after 30s.` | the command was killed |
| `Browser is not open.` | the action was aimed at nothing |
| `Refused to type: the focused control is a Button` | keys there are shortcuts, not text |
| `Nothing about the window changed` | the click may not have landed |
| `There is no element 9 on screen` | the listing is stale; look again |
| `The accessibility tree is empty` | a permission is missing, the screen is not |

Symbio treats all of these as failures rather than answers. Two things follow
from that: the turn keeps going instead of reporting a dead end as done, and
the recovery it finds becomes a training example the same way a corrected
answer does.

```text
You: post this to x.com

Symbio: <tool_call>{"name": "desktop_type", "arguments": {"text": "..."}}</tool_call>

Tool:
Refused to type: the focused control is a Button ('Post'), not a text field.
Look with see_screen target='desktop' and type into the field by its number.

Symbio: <tool_call>{"name": "see_screen", "arguments": {"target": "desktop"}}</tool_call>
        <tool_call>{"name": "desktop_type", "arguments": {"element": 2, "text": "..."}}</tool_call>

[Learn]
Tool mistake captured.
```

Some of these were learned the hard way and are in the corpus as fixed
behaviour: a `sed -i` written GNU-style silently does nothing on macOS and got
reported as done; a tweet posted while the model said it had not; keystrokes
sent at an unfocused window that fired shortcuts instead of typing. The
environment is where an agent's confident wrong answers actually cost
something, so it is the environment that gets checked after every action.

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

# How much does it take to learn something new?

Measured on 2026-09-15, Qwen3-14B on an M-series Mac, against a
deliberately **counter-familiar** cloud CLI built so that pretrained knowledge
actively misleads (`store`/`compute`/`access`, `--region` required on every
command, no URI scheme, `Key:Value` labels, `tag/Key:Value` filters). The
system prompt says nothing about the syntax. The base model scores **0/13**.

Nothing in the corpus is hand-written. The model attempts a task, the simulator
**runs** what it emits, and the pair is kept only if the world changed the way
the task asked. A model that never gets a command right earns exactly zero
samples.

## The curve: 51 checkpoints, held-out battery, single-shot

| iteration | 0 | 20 | 40 | 60 | 80 | **100** | 140 | 160–1000 |
| --------- | -: | -: | -: | -: | -: | ------: | --: | -------: |
| score     | 0/13 | 3/13 | 9/13 | 11/13 | 12/13 | **13/13** | 12/13 | 13/13 ×43 |

**Perfect score at iteration 100 of 1000.** The other 900 iterations — 90% of
the compute, 23 of the 25.7 minutes — changed nothing. Training cost 1.5s per
iteration at 8.4 GB peak.

## The same loop on a harder battery: 16/19, and it wobbles

A later run went against **19 held-out tasks** rather
than 13 — the six extra are chained, multi-command ones. Same model, same
self-earned loop (481 attempts, **82 kept**: a sample is earned only when the
simulator's world changed the way the task asked, so 17% of tries paid). 14
checkpoints, every 20 iterations:

| iter | 0 | 20 | 40 | 60 | 80 | 100 | 120 | 140 | 160 | 180 | **200** | 220 | 240 | 260 |
| ---- | -: | -: | -: | -: | -: | --: | --: | --: | --: | --: | ------: | --: | --: | --: |
| score | 0 | 6 | 6 | 7 | 13 | 12 | 11 | 15 | 8 | 13 | **16** | 13 | 13 | 16 |

**Peak 16/19 at iteration 200, and the curve is not monotone**: 15 at 140, then
**8 at 160**, then 16 at 200. A single checkpoint is not a measurement here.
The 13/13 above is real and reproducible on the 13-verb battery; it does not
survive being restated as "the model learned the CLI".

### What is left at the peak is not syntax

Reading what the best checkpoint actually emitted, rather than its score:

| task | emitted at iter 200 | exit |
| ---- | ------------------- | ---: |
| `filter_nodes` | `compute list-nodes --region eu-2 --where tag/Env:prodd` | **0** |
| `new_identity` | `access new-identity --region eu-2 --identity-name deployment-bot` | 1 |
| `chain_three_step` | three commands, last one `--to-container nightly --to-path run.log` | **0** |

Two of the three **succeed at the command line and still fail the task**. The
grammar is learned: `--where tag/<Key>:<Value>` is exactly right, and `prodd`
is a doubled character in the value. `chain_three_step` emits all three
commands cleanly and copies into the wrong container — the task names two
(`nightly`, `archive`) and the second one is dropped.

`new_identity` looks like the one honest option error, and it is not.
`--identity-name` is `--identity` with `-name` appended, and the environment
answers it with `Options for 'new-identity': --identity, --region` every one of
the eight times it appears. On the two checkpoints where the option is right
(80, 120) the run still fails, because the value comes out as
`name:deployment-bot` — the option's own name, glued to the front of its
argument. It passed exactly once, at iteration 140.

`filter_nodes` **never passed once in 14 checkpoints**, and what it emitted
says why. The value it needs is `prod`. Across the nine checkpoints from 100
on, it wrote `prodd` five times, `producers` three times, and once
`proddumps/db.dump` — every one of them `prod` followed by characters that
should not be there. It is not guessing the wrong value; it cannot stop
emitting the right one. Eight of those nine exit 0 with correct grammar; the
ninth is iteration 160, where the value ran on into an unrelated path and broke
the syntax as well — the same checkpoint where the whole score collapsed to
8/19.

Put beside each other, the residual failures are one failure. `prod` becomes
`prodd`. `--identity` becomes `--identity-name`. `deployment-bot` becomes
`name:deployment-bot`. In every case the correct token is emitted and then not
stopped — this is not a model that has learned the wrong thing, it is a model
that cannot end a string. Same shape as the emitter bug that lost a 15-round
crypto task to a variable whose name had a character dropped: the reasoning was
right and the emission was not.

That matters for what to do next, because it is the one failure more training
data cannot reach. The flat `13/13 ×43` tail above and this run's wobble are
both saying the same thing from opposite ends — the knowledge arrived early,
and the iterations after it are spent on something that is not knowledge.
Sampling (`repetition_penalty`, the stop conditions) and `lora.scale` are where
this lives, and none of it is measured yet.

## The ceiling is what it DISCOVERED, not how long it trained

Three runs, same model, same loop. Only the environment changed:

| errors the environment gives | tasks solved | verbs earned | peak score |
| ---------------------------- | -----------: | -----------: | ---------: |
| the current service's verbs  |        39/60 |           10 |     10/13  |
| every service's verbs        |        59/60 |           11 |        —   |
| + full task coverage         |    **70/72** |       **13** | **13/13**  |

It passed exactly the verbs it had earned samples for and generalised to
**none** it had not. Earned 10 → scored 10. Earned 13 → scored 13.

Why the error text mattered so much: asked to create a container, the model
tried `awsim create-container`, was told the three services, guessed `compute`,
and was then told compute's four verbs — none of which make containers. It
never tried `store`. Every message was locally accurate and kept it inside the
wrong service. Returning the **whole verb map** on any unknown service or verb
took discovery from 39/60 to 59/60 and cut the time by 40%.

Rigid means refusing wrong input, not rationing information. A model that is
lost gets the whole map.

## How many examples per capability

60 training examples (plus 10 held out) covered 13 verbs — roughly 2–6 each:

| verb | examples | result |
| ---- | -------: | ------ |
| `compute resume`  | 2 | passed |
| `compute halt`    | 3 | passed |
| `store duplicate` | 3 | passed |

Those numbers tempted an obvious rule — "about three examples per capability" —
and **the rule did not survive being tested forward.** Over-sampling two starved
shapes to feed them deliberately, 1 of 4 predictions held: two chained shapes
passed on a SINGLE example each, and one failed with thirteen. Reading what it
emitted showed the thirteen-example failure had learned the verb perfectly and
misread an ambiguous task (the target path began with another container's name).

What the rule really was: a description fitted after the fact, which held only
because the under-sampled shapes were also the hardest ones — sample count and
difficulty moved together across three runs and separated the moment they were
pulled apart.

The better hypothesis, and it is untested: **compositionality**. "Shut it down
and label it" is two verbs it already knows, sequenced — one example teaches the
sequencing. "Get rid of the container, it still holds a path" needs a
precondition nobody stated, and that is genuinely new. Treat any
per-capability sample figure here as unestablished.

## A warning worth more than the results

The first version of this benchmark **mimicked the real AWS CLI**, and the
model scored 13/13 before a single step of training. It was reciting `aws s3
mb` from pretraining. A benchmark a model already knows measures recall and
reports it as learning.

And when the curve first read 0/13 at every checkpoint while the training loss
said 0.055, that was the **evaluation**, not the model: `mlx_lm lora` has no
`--keys` flag and trains q/k/v/o plus all three MLP projections (112 tensors at
8 layers), so an eval that attaches only `q_proj`/`v_proj` loads 32 and
`load_weights(strict=False)` drops the other 80 in silence. A silent partial
load is indistinguishable from a model that learned nothing. The harness now
reads the training run's own `adapter_config.json` and refuses to score a
checkpoint whose tensors have nowhere to land.

The harness that produced all of this is **not shipped**. It was a research
rig — a fake cloud CLI, a self-teaching loop and a checkpoint scorer — that
nothing in Symbio imports and that no install can use. What it measured is
worth keeping and is written above; the rig itself was one machine's
scaffolding, and the repository is lighter without it.

# Teaching itself an environment it has never seen

Measured 2026-09-17/18, Qwen3-14B-3bit on a 16 GB Mac, against a reef-keeping
simulator: **50 verbs across 5 services**, `reef <service> <verb> --option
value`, deliberately counter-familiar — a running organism is a `graft`, you
do not create or delete anything, and every command needs `--basin`. **45
tasks, 19 of them six steps deep.** The six-step shape is seal, settle, lift,
bind, open, prime, and every step fails loudly if the one before it did not
happen, so it cannot be brute-forced by emitting six plausible lines.

Nothing in the corpus is written by hand. The model gets the task in English,
emits commands, the simulator RUNS them, and the pair is kept only if the basin
ends in the state the task described. The system prompt says what the world IS
and nothing about how to spell a command.

## What stops a fine-tune from making it worse

A loop that trains itself needs a way to notice that it has got worse, because
the loss will not tell it. Nine things do, and every one of them is a refusal
the run can lose to rather than a warning it prints.

**Before the run.** The corpus is checked, not trusted: samples with broken
tool calls are dropped (`drop_broken_tool_call_samples`), unparseable rows are
removed and counted, prompt-masking is verified against the model's own chat
template, and `lora.keys` is checked against the module tree — a key that
matches nothing produces a clean log and an adapter that learned nothing, so
it is caught before the trainer starts rather than after. A memory preflight
refuses a run the machine cannot hold, because an accepted one is not a failed
run, it is a Jetsam kill.

**The baseline.** `golden.run_golden_set` scores a fixed battery — identity,
tool-tag formatting, the contracts that must not break — against the CURRENT
model, before anything is trained. Without a before, "it still passes" is not
evidence.

**During the run.** The early-stop monitor reads each validation loss and
keeps the best: `val_loss=0.0540 best=0.0420 patience=1/2`. At patience it
kills the trainer and restores the best checkpoint — one run trained 120
iterations and kept 80. A validation loss that is implausible (nan, or a
number no real run produces) aborts instead of being recorded as an
improvement.

**After the run.** The battery runs again. A case that passed before and fails
now is a regression, and `golden_rollback_on_regression` restores the backup
`backup_adapter` took — the fine-tune is discarded, not shipped. Before that
it will try once more with extra iterations and remedy samples for the
specific failing cases (`append_golden_remedy_samples`), because a rollback
that gives up on a fixable regression just preserves it. A wildcard check then
asks about subjects absent from the corpus, which is where a narrow fine-tune
shows up as confident nonsense.

**What the loss cannot say.** `weight_delta_report` compares the new adapter
against the pre-train backup, per tensor and overall:

```
[Train] Weight delta vs the pre-train adapter: 8 tensor(s), 0.26M parameters,
        overall movement 4.73%.
[Train] Moved most: self_attn.lora_a 10.0%.
[Train] 7 of 8 tensor(s) did not move at all — a target the run never reached
        is the usual cause, and it is invisible in the loss.
```

A run whose LoRA targets matched nothing produces a respectable curve and an
adapter identical to the one before it. Movement is the only thing that
distinguishes the two, and "no previous adapter" and "the shape changed" are
reported as unknown rather than as zero — unknown and unchanged are the two
answers that must never be confused.

**Around all of it.** The adapter is sealed and verified at load time against a
Merkle root, so weights that changed since they were sealed are refused rather
than loaded quietly. In-flight runs are journalled with a pid and a boot id, so
an OOM kill leaves a resumable record instead of a half-written adapter that
loads as the real one. Worker adapters get the same treatment against their own
golden sets, one role at a time.

## No human writes the training data

The loop closes on itself. A task is put to the model in English; the model
emits commands; **the simulator runs them**; the pair is kept only if the world
ended in the state the task described; the kept commands are reduced to the
ones that carried it; the corpus is rendered, trained, and the same battery is
scored again with the new adapter attached. Nobody writes a sample, nobody
labels one, and nobody decides whether an answer was good — the environment
decides by being changed or not being changed.

That is what the four rounds below measure: **10/45 to 26/45, and six-step
tasks from 0/19 to 7/19, on a corpus the model earned by acting.** Five of the
seven six-step solves were tasks whose answers were never in its training data,
and one task it *was* trained on it still failed — which is what generalisation
looks like and what memorisation does not.

The honest boundary: a person still builds the environment and starts the runs,
and three fixes to the harness between rounds mattered more than any single
round of training. What no person does is supply an example of the work. Once
the environment exists, the data does not come from a human, and neither does
the grade.

## Four rounds: earn, train, earn again

| | trained on | solved | depth 6 |
| --- | --- | ---: | ---: |
| 1. base model | — | 10/45 | **0/19** |
| 2. first adapter | its own 10 samples (depths 1-4) | 16/45 | **0/19** |
| 3. more runway | same 10 samples, 12 rounds per task | 19/45 | **3/19** |
| 4. chains in the corpus | 19 samples including 3 six-step | 26/45 | 7/19 |
| 5. ten tasks held out | 21 samples, none from the held-out set | 21/45 | 5/19 |

Round 4 scores higher and is the weaker number: five of the tasks it was
scored on had their own answers in its training data. Round 5 is the honest
one — ten tasks were reserved before training, chosen by position so nothing
about how hard they turned out could decide which side of the line they fell.

**Scored against those same ten in every round:**

| round | all 45 | held-out 10 | held-out six-step |
| --- | ---: | ---: | ---: |
| 1. base | 10/45 | 2/10 | 0/4 |
| 2. first adapter | 16/45 | 3/10 | 0/4 |
| 3. runway + nudges | 19/45 | 2/10 | 0/4 |
| 4. chains (leaky) | 26/45 | 5/10 | 0/4 |
| 5. held-out clean | 21/45 | **4/10** | **1/4** |

On tasks that were never in any corpus, the base model solves 2 and the
self-taught one solves 4. Its first six-step chain on a reserved task —
`transplant_2`, seal, settle, lift, bind, open, prime — lands in round 5, on
an adapter that had never seen any of the four.

Train split 17/35 against held-out 4/10: 49% and 40%. A model that had
memorised its corpus would show that gap much wider, and this is the number
that says the six-step result is a learned shape rather than a recalled one.

Ten reserved tasks and four reserved chains is a small sample, and every cell
above is one run. Read the direction, not the decimals.

Round 1 said the long chain was not reachable from a corpus containing no
example of one. Rounds 3 and 4 are what it took to test that, and both halves
mattered: runway and two corrections got the first three six-step solves out
of the model, and training on those three took it to seven. **Nothing else
changed between rounds 3 and 4** — same tasks, same twelve rounds, same
prompts. The corpus is the variable.

Round 4's training run is also the clearest picture of the loop adjusting
itself: validation went 2.362 at iteration 1 down to 0.042 at 80, rose to
0.054, came back to 0.042, and the early-stop monitor — `patience=2/2` — kept
iteration **80** out of a planned 120 and threw the rest away.

## What twelve rounds bought, and what it did not

Runway alone did nothing. Watched live at twelve rounds, the model converged on
a set of commands the basin ACCEPTED and that did not accomplish the task, then
re-sent the identical eight lines four rounds running — `8 command(s), 6
accepted, world not there yet`, four times over. Nothing it sent was an error,
so no error text could reach it, and it never once read the basin back.

Two things fixed that, and neither is the answer to any task: that a repeat is
a repeat (the rule the assistant's own persistence ladder runs on), and that
the world can be READ — `graft show`, `host show`, `graft list`. It had those
verbs from the first round and used them zero times; it has used them 42 times
since. Depth 4 went 0/4 to 2/4 on that alone.

## What it records is not what it did

The first six-step chain it ever solved was recorded as **twenty-five
commands**: the working sequence, four redundant re-brews of the same culture,
four re-opens of the same channel, and the six inspection calls it made along
the way. Every one was accepted, so every one went into the sample — and a
corpus like that teaches a six-step task in twenty-five moves.

Samples are reduced now the way a minimal reproduction is: drop a command,
replay from a fresh seed, keep the drop if the task still passes. Twenty-five
became six.

```
reef channel seal  --graft g3
reef graft lift    --graft g3
reef graft bind    --graft g3 --host reef-s
reef channel open  --graft g3 --culture algae
reef channel flow  --graft g3 --rate 1
reef graft prime   --graft g3
```

Fifty-five redundant commands came out of the three corpora this way.

## Three things the harness got wrong, found by running it

**A task a do-nothing answer passed.** "Quarantine coral-a and then release it
again" ends in the state it starts in. The driver logged `world SOLVED` for two
commands the simulator had rejected, and wrote a training sample whose answer
was empty. The harness now validates both ways before it grades anything —
reference answers **45/45**, do-nothing answers **0/45** — and refuses any
sample where not one command was accepted.

**The harness rationed what the environment was handing over.** The simulator
answers an unknown verb with every verb it has, which is the finding the
previous environment paid for: discovery went 39/60 to 59/60 when the errors
stopped being helpful-but-narrow. The driver then clipped that error to 200
characters before feeding it back, cutting the list off exactly where `culture:
brew` would have been. The model hunted for "create" for four rounds while the
answer sat in a string the harness had already thrown away.

**Vocabulary is not grammar.** With the full verb list restored it still wrote
`reef culture plankton brew --potency 4` — noun where the verb goes — four
rounds running. The list told it which words exist and nothing about where they
go. One line added to the error text:

> The order is `reef <service> <verb> --option value` — service first, verb
> second, and every value passed as --option value. Example: `reef culture brew
> --culture plankton --potency 4 --basin tide-1`.

**1 solve in 27 became 6 in 8.** Same model, same tasks, same four rounds. Rigid
means refusing wrong input, not rationing information — and that applies to the
harness at least as much as to the environment it is testing.

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

## Looking inside the weights

### It decides when to train

Training starts two ways. The automatic way: corrections and recovered
mistakes collect as examples, and once enough accumulate a LoRA run starts on
its own. The other way: the model itself can call `train_adapter` (keep
training on what it has) or `retrain_adapter` (rebuild from scratch) whenever
it judges a fine-tune is due. At the default `safety.require_confirm_score`
of 3 these score 2/3, so they run without a prompt and land in the security
log. Raise the bar to 2 if you want to approve every self-started run.
Either way the golden set still grades the result and rolls it back if it
got worse.

### Seeing what a fine-tune changed

LoRA never rewrites the base model. Each module it touches becomes
`W + scale · (lora_a @ lora_b)`, so the change is readable straight from the
adapter file:

```bash
venv/bin/python scripts/weight_delta.py adapters/ --base mlx-community/Qwen3-14B-3bit
```

```text
module                              |dW|      rel
model.layers.38.self_attn.q_proj     11.616  10.181%  ########################
model.layers.38.self_attn.v_proj      6.225   9.342%  ############
model.layers.39.self_attn.q_proj     10.995   9.757%  ######################
model.layers.39.self_attn.v_proj      5.969   9.635%  ############
```

That is the live 14B adapter on one machine: four attention modules in the
last two layers, each moved by about a tenth of its own size. `--since OLD`
compares two adapters (two checkpoints of one run, or before and after a
retrain) and prints how much the latest training moved each module.

### Which neurons fire, and growing new ones (experimental)

`bench/activation_recorder.py` hooks every layer and MLP of the headmaster,
records which neurons fire on one set of prompts, then tests on a second set
whether that recording means anything, by pruning with it and with a random
choice. On Qwen3-14B-3bit, measured as how often the pruned model still picks
the full model's next token:

| cut | chosen by the recording | chosen at random |
|---|---|---|
| switch off 10% of neurons | 0.875 | 0.776 |
| switch off 50% of neurons | 0.584 | 0.200 |
| drop 8 of 40 layers | 0.478 | 0.484 |

Neuron firing is a real signal. Layer "influence" is not; it did no better
than random.

`bench/neuron_adapter.py` adds weights instead: new MLP neurons beside the
frozen base, whose output starts at exactly zero, so at step 0 the model is
bit-for-bit unchanged. On Qwen3-0.6B, four examples taught it a fact it could
not have known, and it answered a phrasing it never saw. Left switched on for
everything, the new neurons also bent unrelated answers (the capital of
Australia turned into Vancouver). Switched on only for their own skill, the
same way skill adapters are triggered, unrelated replies were identical to
the base model's. Neither of these is wired into the agent yet.

---

# 💻 Tools

Symbio can interact with the local machine through several tool groups.

Each tool is one markdown file in `tools/`, seeded on first run and yours to
edit afterwards:

```markdown
---
name: browser_click
family: browser
group: browser
---

Click an element in the open browser, identified by its visible text.

```json
{"type":"object","properties":{"target":{"type":"string"}},"required":["target"]}
```
```

The system prompt does not carry all of those schemas. It carries the *index* —
the families, the tool names in each, one line on what the family is for — and
the model asks for the arguments it needs with `tool_docs`. That is about 1,300
tokens instead of 5,100, on every turn, which on a 14B is room to think in.
`/tools` shows the same index you are showing the model; `/tools browser` prints
the schemas. `agent.tool_catalog: "full"` puts every schema back inline.

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

### The constitution

The longer you use it, the more it works out how *you* want to be worked with —
and writes it down where you can see it and change it:

```text
$ /constitution
Held — this is in every prompt:
  answers_vs_control: answers  Give the result first. Do not narrate the steps
                               or offer a menu of options unless they ask for
                               one. (4 for/0 against, since 2026-09-10)
    from: told me twice to just do it; asked for the number only
  act_vs_confirm: act          Take the ordinary reversible step without
                               asking. (yours, since 2026-09-14)
```

It holds **one stance per question**, not a pile of observations. That is the
whole design: `soul.md` collects what it saw each turn and only appends, so it
ends up holding "wants to approve everything" and "wants no confirmations" at
the same time, forever. An axis can only hold one, so new evidence either
reinforces the stance or argues against it, and an axis flips only once the
other side outweighs it — dated, so you can see when you changed.

The questions it holds a stance on (`/constitution axes`): answers vs control,
act vs confirm, brief vs complete, speed vs caution, do vs teach, blunt vs
cushioned, show vs summarize, code vs prose.

* `/constitution set <axis> <pole>` — your own word. Inference never
  overwrites it; it can only record that the evidence disagrees.
* `/constitution clear <axis>` — drop it.
* `/constitution revise` — fold in what has been observed since last time.
* It is `constitution.md`. Edit it by hand if you'd rather.

Everything *inferred* goes back to the model wrapped as untrusted data, like
every other store derived from conversation — so a web page cannot install a
preference by being read, written down, and read back as yours.

### Never giving up easily

When a turn keeps failing, the harness does not repeat "try something else". It
escalates, and what it attacks moves inward:

```text
1  the call        use the error you just got
2  the approach    that is the same attempt; here is what you have not tried
3  the assumptions name what this rests on and test the weakest one
4  the evidence    you may be wrong about what you SAW; go and re-read it
5  the method      solve it as if that tool didn't exist
6  the problem     you may be solving the wrong problem
```

A repeated call is refused without running — a repeat is not another attempt —
and the refusal comes back as the next rung. The ladder never runs out: what
ends a turn is the round budget, which counts work actually attempted, never
the harness running out of things to say. Every rung still leaves an honest way
out ("here is what blocked me"), because pressure with no acceptable answer but
success is pressure to fabricate one.

Tune with `agent.min_distinct_attempts` and `agent.max_persistence_challenges`.

### Your own slash commands

A command is a file in `commands/` whose body is a prompt:

```markdown
---
name: standup
description: What moved and what is blocked
---

Read my notes from the last two days$ARGUMENTS, then give me three lines:
what moved, what is blocked, what I should start with today.
```

Type `/standup` and that body becomes your next message, with `$ARGUMENTS`
replaced by whatever you typed after the name (`$1`, `$2`, … take the words).

* `/` on its own prints every command, yours first
* `/` then Tab completes against them
* `/commands new <name> | [description] | <prompt>` saves one without leaving the chat
* `/commands show <name>`, `/commands rm <name>`
* a mistyped command suggests the nearest real one

The assistant can write one too, with the `save_command` tool, when it notices
you asking for the same shape of thing repeatedly. Commands it wrote are marked
in the listing — and saving one never runs it; only you do, by typing the name.

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

# CLI
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

### Terminal interface

Interactive chat uses a compact, Hermes / Claude Code-inspired layout: a warm
welcome panel with the model and workspace, a clearly separated input line,
and grouped tool calls/results. It stays in normal terminal scrollback rather
than taking over the screen, so copying output and scrolling still work.

* **Enter** sends; **Tab** completes slash commands in both local and daemon chat.
* **`/`** opens the full command menu; **`/status`** shows detailed session state.
* **`/quit`** exits. Existing readline editing and history keys are unchanged.
* Narrow windows get a smaller layout; `NO_COLOR=1` or `TERM=dumb` retains the
  plain-text interface. Piped output and non-terminal front ends stay plain.

Preview the layout without loading a model, reading credentials, or running tools:

```bash
python symbio/app/chat_style.py --demo
python symbio/app/chat_style.py --demo --width 48 --no-color
```

Restart a running resident daemon with `symbio daemon stop` followed by
`symbio daemon start` to pick up the new welcome-panel metadata.

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
| `model_name`                    | `Qwen/Qwen3-14B` | Base model                       |
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

# Security

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

> **Note on the name:** the distribution is `symbio-cli`. Plain `symbio` on
> PyPI is an unrelated multi-agent framework that had the name first, so
> `pip install symbio` will fetch someone else's project — install from this
> repository as shown above. The commands you type are still `symbio` and
> `symb`.

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

## The BTW I am here badges:
 - <a href="https://nicklaunches.com/products/symbio/?utm_source=github.com&utm_medium=badge&utm_campaign=featured" target="_blank" rel="noopener"><img src="https://nicklaunches.com/badges/featured-dark.png" alt="Symbio on Nick Launches" width="244" height="56" /></a>
