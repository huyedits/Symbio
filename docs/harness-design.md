# Symbio Harness Design

Status: proposal. Written 2026-10-28. The P6 desktop items and the §3.5
control height and reply measure have since been adopted; the rest is unbuilt.

This document designs the two halves of Symbio's harness — the desktop window and
the agent loop — from evidence gathered by reading the harnesses already installed
on this Mac, and it records what it would take to run LFM2.5-24B-A2B at roughly
10 GB. Every claim below is tied to a file and line in this repository or in the
inspected application; anything inferred is marked UNVERIFIED.

The design goal is not to become another general agent platform. Symbio is one
person's agent on one Mac, it fine-tunes itself, and it asks before it acts. The
patterns below are chosen because they make that specific promise cheaper to keep.

---

## 1. Sources

| Source | Version examined | What was read |
|---|---|---|
| Claude Desktop | 2.37337.1 (commit ba51ab5, built 2026-10-13) | `app.asar` renderer CSS/JS, preload, IPC typed-RPC definitions, main-window security flags |
| OpenClaw | installed at `/opt/homebrew/lib/node_modules/openclaw` | gateway, agent loop (`pi` packages), approvals, compaction, session store |
| OpenCode | `/opt/homebrew/lib/node_modules/@opencode/cli/bin/opencode.exe` + `~/.local/share/opencode/opencode.db` | HTTP/SSE protocol, event vocabulary, SQLite schema, permission model |
| Symbio | working tree at `8bce9a5` | `symbio/app/chat_turn.py`, `chat_tools.py`, `daemon.py`, `tooling.py`, `symbio/tools.py`, `symbio_desktop/*` |
| Hugging Face | API queries 2026-10-28 | LFM2.5-24B-A2B configs, MLX and GGUF artifact sizes |

One thing to state plainly: Claude Desktop's chat UI is remote. The shell is local
Electron; `index.html` loads the conversation view from claude.ai. So its layout
numbers below are the shell's, and its conversation column and composer geometry
could not be read (UNVERIFIED). Its security model and IPC surface, however, are
fully local and are the most instructive part.

---

## 2. What the other harnesses actually do

### 2.1 Claude Desktop

**Window chrome.** Minimum window 600×400 with `titleBarStyle: "hidden"` and
`titleBarOverlay: false`; initial 1200×800. The main window's own titlebar height
is 0, other windows 28px. A 48px full-width drag strip sits above the content and
a 0.5px seam separates the sidebar. Sidebar is 288px and collapses when the window
or zoom reaches 700px or less; the real value is reported back from the page via
`WindowControl.reportFrameLayout`, so the shell never guesses.

**Tokens.** The design system is dense ("comfortable" density): radius 8, composer
14, panel 12, control height 32, icon 20, padding steps 6/8/12/16/24, gaps
8/12/16/28/40. Type is a variable sans with weights 400/500/580/600; body 14/20,
prose 16/24, caption 12/17, code 13/19, title 22/28. Light surfaces are
`#fcfcfb` / `#f9f9f7` / `#ffffff`, text `#0b0b0b` / `#52514e` / `#898781`,
accent text `#184f95` and fill `#2a78d6`, danger `#d03b3b`. Dark surfaces
`#151515` / `#0b0b0b` / `#1a1a19`, text `#f0efec` / `#c3c2b7`, accent `#6da7ec`,
danger `#ec7e7e`. Borders are the ink colour at 10% and 20%; the user bubble is
ink at 5%. Message bubbles are radius 12, padding 12×16, measure 85% of the
column, assistant text inset 4px.

**IPC.** A typed RPC layer: 14 namespaces, 56 method groups under `claude.web`
and 17 under `claude.settings`, each method with a schema and a return type, each
group with an explicit sender check. Channel names are generated, not free-form.
The renderer never sees raw Node: `sandbox: true`, `contextIsolation: true`,
`nodeIntegration: false` (also in subframes and workers), `webviewTag: false`,
`navigateOnDragDrop: false`, `backgroundThrottling: true`, with a frozen
preferences object. Remote content runs in a `WebContentsView`; the embedded
artifact pane is a separate iframe that performs a host-tools handshake and is
handed a `MessagePort`, and a frame token must match before any hybrid binding is
exposed.

**Events.** The shell uses event pairs, not callbacks: `uiHostAsked
{request_id, kind, text, mode}` and `uiHostAnswer`, plus `uiNotified`,
`uiSessionChanged`, `uiPanesChanged`. High-rate data (transcript, framebuffer)
goes over dedicated `MessagePort`s rather than the general channel. Approvals are
built in the main process as `{title, displayName, description, input}` and shown
as "Allow Claude to use {toolName}?" with "Allow once".

### 2.2 OpenClaw

One Node gateway process on port 18789 owns sessions, run dispatch and approvals.
TUI, control UI and channel plugins are thin clients. Methods are grouped
(`sessions.*`, `agent`, `agent.wait`, `chat.send`, `chat.abort`,
`exec.approval.*`); events are `connect.challenge`, `agent`, `chat`, `presence`,
`tick`, `health`, `heartbeat`, `cron`. Every agent event carries `runId` and
`seq`; a gap in `seq` synthesizes an error stream rather than silently continuing.

Approvals are out of band on their own Unix socket, deny by default, ask on a
miss, 120-second timeout, persisted in `~/.openclaw/exec-approvals.json`.

Compaction: reserve 16,384 tokens, keep the last 20,000 tokens, trigger when
`contextTokens > contextWindow - reserveTokens`, summary budget 0.8 × reserve,
with before/after hooks. Sessions are a tree (`id`, `parentId`, `branch()`), stored
as a sessions registry plus one JSONL per session with a lock file and explicit
v1→v2→v3 migrations.

Worth copying: sequenced events with gap detection; approvals on their own
channel with a deny default; one seam for retry, fallback and
compaction-failure handling.
Worth rejecting: a 68 MB distribution containing two hashed copies of nearly
every chunk, and a 900 KB gateway module mixing HTTP, WebSocket auth, device
pairing and every RPC method.

### 2.3 OpenCode

A single 188 MB Bun binary is both server and client. The HTTP API is
`/api/session/:id/{prompt,interrupt,compact,move}` plus an SSE stream; an
experimental durable log reads events after an aggregate sequence and continues
live when `follow=true`. Event names are a vocabulary, not ad-hoc strings:
`session.execution.{started,succeeded,failed,interrupted}`, `session.inbox.*`,
`session.text.*`, `session.reasoning.*`, `session.tool.{input.started,
input.ended,called,success,failed}`, `session.retry.scheduled`,
`session.compaction.*`.

Tools are `{name, description, inputSchema, outputSchema, cache, native}`;
permission replies are `once | always | reject`, emitted as
`permission.asked` / `permission.replied`, and "always" persists as a row in a
`permission(project_id, action, resource)` table. When no client is attached, a
permission request auto-rejects — it fails closed. Compaction is configured
(`auto: true`, `prune: false`, `tail_turns`), `/compact` enqueues a pending
compaction, and a unique partial index allows at most one pending compaction per
session. Agentic iterations are capped by `steps`; a `doom_loop` permission guards
continuing after repeated failures.

Persistence is one SQLite file: `session_v2` carries `parent_id`,
`fork_session_id`, `fork_boundary`, `cost`, `tokens_input/output/reasoning/
cache_read/cache_write`, `time_compacting`, `resume_attempts`;
`session_message(session_id, seq, data)` is unique per `(session_id, seq)`;
`session_inbox(delivery, enqueued_seq)` holds steering inputs; and
`event(aggregate_id, seq, type, data)` with `event_sequence` is an append-only log.

Worth copying: durable inbox and a sequenced event log that make reconnect
lossless; permission grants as rows keyed by project, action and resource;
cost and token counters stored per session so usage is free.
Worth rejecting: shipping an unauditable 188 MB binary for a tool that runs shell
commands, and keeping credentials in the same SQLite file as conversations.

### 2.4 Symbio today

The current structure is mapped in detail in the working notes; the parts that
matter for this design:

- Three tool surfaces. The advertised catalog holds 59 tools
  (`symbio/app/tooling.py:236`); the `symbio/tools.py` registry holds 39 with
  `run` lambdas (`symbio/tools.py:51`) and feeds a second complete turn loop,
  `AIAgent.run` (`symbio/app/agent.py:337`). Twenty-five overlap.
  `build_tools_block` (`tooling.py:1662`) folds `tools/*.md` over the catalog at
  every call, mutating module state.
- One enormous turn body. `_agent_turn_body` is 1,863 lines
  (`symbio/app/chat_turn.py:159-2022`) with 23 turn-local booleans, 17 injected
  `{"role":"user"}` observation messages and 26 `history.append` sites. Each
  correction ladder is another flag branch.
- One model, one conversation, one accept thread. `daemon_main` accepts serially
  (`symbio/app/daemon.py:617-640`), so desktop, ACP, MCP and the OpenAI bridge
  queue behind a single socket. Client input idles out at 90 seconds
  (`daemon.py:300`).
- The approver is whoever holds the socket. `confirm_fn` prompts the requesting
  client (`daemon.py:405`); answering "always" rewrites the guardrail mode to
  allow (`daemon.py:419-425`). A refusal persists for one turn only
  (`chat_turn.py:491,1665`). Headless callers cannot approve at all.
- Sessions live in three places: JSONL under `SESSIONS_DIR`
  (`symbio/app/sessions.py:12`), a second SQLite `SessionStore`
  (`symbio/store.py:9`), and `memory.db` for MCP benchmarks. `session_id` is a
  timestamp (`chat.py:400`); there are no lineage or parent ids.
- The prompt is rebuilt from disk every turn (`chat_turn.py:413`), RAG is
  keyword-only over non-recursive `notes/*.md` (`symbio/rag.py:191,551`), and the
  desktop loads `constants` and `guardrails` twice by path
  (`symbio_desktop/server.py:69,91`).
- Training safety is process-local: `TRAINER_LOCK` is a `threading.Lock()`
  (`training.py:45`), and guarded training exists twice (`chat.py:3018`,
  `dispatch.py:898`).

These are the gaps the two designs below close. They are also the reason the
proposal is sequenced: the event log and permission rows are useful even if the
turn-state refactor never happens.

---

## 3. Desktop harness design

### 3.1 Position

Keep the current architecture. `symbio_desktop/server.py` is standard library
only, speaks a hand-written WebSocket, and bridges to the daemon; measured idle
memory is 28 MB and the `tests/test_desktop_app.py` fence keeps `symbio` out of
the process. That is a better trade than Electron for a one-user app on a 16 GB
machine, and none of the patterns below require replacing it.

The desktop's job is narrower than Claude's: it is the window onto one local
daemon, the place approvals are answered, and the place the machine's state
(adapters, skills, corpus, health) is visible. It should not grow a remote chat
plane, extensions marketplace, or multi-account surface.

### 3.2 Process topology

```
window (WebKit or browser)  <-- WebSocket -->  symbio_desktop/server.py
                                                    |  Unix socket
                                                    v
                                              symb daemon (one model,
                                              one generation at a time)
                                                    |
                                            sessions / event log
```

The daemon is the only writer. The desktop is a subscriber and an approval
surface. MCP, ACP and the OpenAI bridge are peers of the desktop, not children of
it. This matches OpenClaw's gateway split and OpenCode's server/client split
without importing either's size.

### 3.3 Event protocol v2

The current daemon emits `status`, `output`, `stream`, `input_prompt`, `confirm`
and `done`; the bridge rewrites them to `token`, `system`, `confirm`, `awake`,
`done`, `quit`. That mapping loses ordering, cannot be replayed, and has no way
to detect a dropped frame.

Proposed frames (JSON lines, both directions):

```json
{"v":2,"seq":41,"run":"r-20261028-1","type":"model.delta","data":{"text":"..."}}
{"v":2,"seq":42,"run":"r-20261028-1","type":"tool.started","data":{"name":"read_file","args":{...}}}
{"v":2,"seq":43,"run":"r-20261028-1","type":"approval.asked","data":{"id":"a-7","kind":"write","card":{...}}}
{"v":2,"seq":44,"run":"r-20261028-1","type":"run.finished","data":{"reason":"completed","usage":{...}}}
```

Client methods: `chat.submit`, `chat.steer`, `run.cancel`, `approval.answer`,
`events.subscribe {since_seq}`, `session.list`, `session.open`, `settings.get`,
`settings.set`, `guardrails.get`, `guardrails.set`.

Event vocabulary mirrors OpenCode's because the names already describe what the
UI needs: `run.started`, `turn.started`, `model.delta`, `reasoning.delta`,
`tool.input`, `tool.started`, `tool.finished`, `approval.asked`,
`approval.resolved`, `inbox.accepted`, `compaction.started`,
`compaction.finished`, `run.finished`, `run.failed`, `cost`.

Rules:

1. `seq` is monotonic per daemon process and persisted with the event log. A
   client that sees a gap sends `events.subscribe {since_seq}` and replays; it
   never renders across a gap.
2. Every frame is idempotent to render: `model.delta` appends, `tool.started`
   keys on a tool-call id, `approval.asked` keys on an approval id.
3. `run.cancel` is honoured at a token boundary. There is no thread to abandon in
   a local MLX generate call; the honest contract is "stop emitting and drop the
   partial turn", not Hermes's abandoned HTTP request.

This is the single change with the widest payoff: the TUI, desktop, ACP, MCP and
OpenAI bridge all get the same ordered stream, reconnect becomes replay, and the
UI stops inferring `thinking` and `tool_progress` from one `output` blob.

### 3.4 Client components

Adopt from Claude Desktop and OpenCode, translated to Symbio's existing views:

- **Queued message strip.** When a run is active, a submitted message becomes an
  inbox row with `delivery: "steer" | "queue"`, shown above the composer the way
  Claude's `ChatComposerChin` and `QueuedMessages` do. Without this, the current
  90-second input prompt is the only way to talk to a busy agent.
- **Tool card.** One card per tool call, keyed by id, states
  `input → running → ok/failed`, with the arguments collapsible. Claude names no
  ToolCall component (its UI is remote) but OpenCode's
  `session.tool.input.started/input.ended/called` sequence is the state model to
  copy.
- **Reasoning disclosure.** A collapsed row that opens on click, fed by
  `reasoning.delta`, separate from the answer stream. Symbio already parses
  thinking blocks; this stops the reasoning from being rendered as the reply
  (the truncated-thinking failure recorded in the project notes).
- **Approval card.** Extend the existing card (`server.py:1002-1230`) with an
  approval id, a resource line, and three outcomes: Deny, Always allow (scoped),
  Allow once. The wording stays Symbio's ("what it is about to do", the exact
  payload in mono), not Claude's.
- **Artifact/preview pane.** Optional, and only after the event log exists. The
  Claude pattern is worth copying exactly: sandboxed iframe, a host-tools
  handshake, a `MessagePort` for high-rate traffic, and a frame token check
  before any privileged binding. Symbio's HTML previews become untrusted content
  the moment they are rendered.
- **Session rail.** Sessions with parent/fork markers once lineage ids exist
  (§4.5).

### 3.5 Visual contract

`symbio_desktop/DESIGN.md` is the contract and its identity (pond teal, tabby
ink, amber charm, SF Pro Rounded, 720px reading column) is not in question. The
measured applications suggest a small set of *structural* amendments, which are
proposed here as explicit changes rather than folded in silently:

| Item | DESIGN.md today | Proposal | Why |
|---|---|---|---|
| Control height | 32px (adopted 2026-10-28) | 32px | Claude's comfortable density; matches the existing 6px control radius |
| Sidebar | 260px | keep 260px | 288px is Claude's; Symbio's rail holds five views, not a chat history |
| Collapse breakpoint | none | sidebar collapses at ≤700px | Claude's `LUn` rule; prevents the current squeeze |
| Drag region | none | 48px strip, 0.5px sidebar seam | Claude's window chrome; cheap and native-feeling |
| Reply measure | 720px, 85% cap, 4px inset (adopted 2026-10-28) | 720px, 85% cap, 4px assistant inset | Claude's measure; keeps long lines readable at small windows |
| Composer radius | 22px | keep 22px | identity; Claude's 14px is theirs |
| Focus ring | pond teal | keep | identity |
| Type scale | 15.5px / 1.65 | keep | trained adapter and design both depend on it |

Everything else in Claude's palette stays out. The rule in DESIGN.md already
covers it: no cream-and-terracotta, no stock Tailwind colours, no decorative mono.

### 3.6 Security

Adopt Claude's frozen browser preferences wholesale:

```
sandbox: true
contextIsolation: true
nodeIntegration: false
nodeIntegrationInSubFrames: false
nodeIntegrationInWorker: false
webviewTag: false
navigateOnDragDrop: false
backgroundThrottling: true
```

For the WKWebView path (`window.py`), the equivalents are a content controller
that blocks navigation away from the local origin, and no JavaScript bridge beyond
the one used today. The desktop server already locks Origin and Host
(`server.py:1255`); keep that as the first line and treat the frame token as the
second.

The typed-RPC idea is worth copying in shape: instead of ad-hoc JSON keys, define
a small number of groups with declared arguments and a sender check —
`chat`, `sessions`, `approvals`, `workspace`, `settings`, `models`, `training`,
`events`. That is 8 groups, not Claude's 56, and it makes the bridge testable
without a browser.

---

## 4. Agent loop harness design

### 4.1 Turn as a state machine

Replace the 1,863-line body and its 23 booleans with an explicit state and a
transition table. The states are the ones the code already implements, just
named:

```
intake -> context -> model -> parse -> gate -> execute -> observe -> decide
                                                             |-> model (loop)
                                                             |-> settle
                                                  error -> recover -> decide
             cancel (any state) -> settle
```

`decide` is the only place that chooses to continue; every current correction
ladder (unparsed tag, scaffold echo, unknown tool, malformed arguments, dropped
persistence challenge, browser retry, unverified completion, auto web search,
provenance) becomes a named transition with a predicate and a test. This is not a
rewrite of behaviour; it is the same behaviour moved where it can be read and
counted.

Two rules make it safe:

1. Every injected `{"role":"user"}` observation gets a `kind`
   (`tool_result`, `correction`, `nudge`, `security`). The UI can then collapse
   them, and a trained adapter no longer sees an undifferentiated wall of user
   turns.
2. `max_tool_rounds` stays the hard cap (default 15, `config.py:143`), and every
   state transition increments a single `step` counter, the OpenCode `steps`
   idea. A loop that reaches the cap settles with a typed reason.

### 4.2 One tool registry

Collapse the three declarers into one `ToolSpec`:

```python
ToolSpec(
    name, group, description, schema,
    risk_kind,            # feeds guardrails.kind_of
    handler,              # the one dispatcher entry
    source,               # "core" | "tools/<name>.md" | "mcp"
)
```

- `symbio/tools.py`'s 39 handler lambdas and `tooling.py`'s 59 catalog entries are
  reconciled by hand once; the overlap of 25 is the merge set, the catalog-only 34
  either get handlers or are deleted from the prompt, and the registry-only 14
  either get advertised or are removed from `AIAgent`.
- `tools/*.md` becomes documentation attached to a `ToolSpec`, not a schema
  mutation. `sync_tool_files` no longer rewrites module globals at every prompt
  build.
- `AIAgent.run` (`agent.py:337`) is deleted or becomes a thin client of the same
  loop. A second turn loop is a second set of bugs, and it already has its own RAG
  and few-shot logic.

Kill criterion: the prompt-visible tool count, the dispatched tool count and the
schema-validated tool count must be the same number, asserted by a test that
fails on any new divergence.

### 4.3 Approvals as rows

The current model is a mode on a kind plus an ephemeral yes/no. Replace it with
the OpenCode shape, scoped to one user:

```sql
permission(
  id, project, action, resource,   -- e.g. ("~/Downloads/agi", "write", "notes/*.md")
  decision,                          -- 'allow' | 'deny'
  scope,                             -- 'once' | 'standing'
  created_at, expires_at, source     -- 'desktop' | 'cli' | 'cron'
)
```

- An approval answer writes a row. "Always allow" writes a `standing` row for that
  resource in that project; it does not rewrite `guardrails.set_mode` to `allow`
  for the whole kind.
- `guardrails` keeps the floors — the things the user said never — and the
  per-kind default. Permissions are grants *on top of* the floors and can never
  raise a floor.
- Headless callers fail closed, as OpenCode does. But instead of a bare refusal,
  they get `approval_pending` and the desktop shows a pending row the user can
  resolve; the action is retried only if the user explicitly asks.
- Refusals persist. A denied resource stays denied for the session and is
  remembered as a permission row, not mined as a preference. This is the direct
  fix for the recorded failure where a refusal was learned as a user value.
- The currently unreachable `safety.maybe_confirm` (`safety.py:1046`) is either
  wired into this path or deleted; a dead fallback gate is worse than none.

### 4.4 Durable inbox and steering

Adopt OpenCode's `session_inbox` and `delivery`:

```sql
inbox(id, session_id, seq, delivery, text, created_at)
-- delivery: 'steer' (apply at next step boundary) | 'queue' (run after this run)
```

This replaces the 90-second `input_prompt` wait (`daemon.py:300`) as the only way
to reach a busy agent. The desktop's queued-message strip renders these rows; the
TUI shows them as pending lines; MCP/ACP can enqueue without a socket held open.

### 4.5 Event log and sessions

One append-only log, written only by the daemon:

```sql
event(aggregate_id TEXT, seq INTEGER, type TEXT, data TEXT, ts TEXT,
      PRIMARY KEY(aggregate_id, seq));
event_sequence(aggregate_id TEXT PRIMARY KEY, last_seq INTEGER);
```

Sessions gain `parent_id`, `fork_boundary`, `cost`, `tokens_input`,
`tokens_output`, `tokens_reasoning`, `cache_read`, `cache_write`. The JSONL store
(`sessions.py:12`) becomes an export format, and the second SQLite store
(`store.py:9`) is deleted or folded in. One store, one writer, one migration
table — OpenCode's `migration(id, time_completed)` is the minimal shape.

Lineage matters for the self-training loop: a compaction that summarizes a
session should create a child with `parent_id` and `fork_boundary`, not silently
mutate one session id that is only a timestamp.

### 4.6 Context: two-stage compression with lineage

Symbio has a measured KV budget and a sticky trim boundary
(`chat.py:1081-1300`), which is the expensive part and should stay. Add the
policy Around it:

- preflight at 50% of the model's context window: flush memory/notes first,
  then summarize the oldest protected-excluded span;
- hard trigger at 85%;
- `protect_last_n = 20` turns, and never split a tool call from its result;
- every compaction emits `compaction.started`/`compaction.finished` and creates a
  child session with a new lineage id.

The KV measurement is MLX-specific and better than anything either reference
implementation does; the thresholds and the lineage id are the borrowed part.

### 4.7 Concurrency

The serial accept loop is the throughput wall. The minimal fix:

- one accept loop per connection, with a shared `generation_lock` (the existing
  `_generation_lock`, `chat_tools.py:186`);
- control methods (`status`, `events.subscribe`, `run.cancel`, `approval.answer`)
  handled outside the generation lock so a long turn cannot block health checks
  or a cancel;
- a FIFO queue with a fairness rule: a queued run may be preempted only by an
  approval answer or a cancel, never by another prompt.

MLX on one GPU is one generation at a time; the goal is not parallel inference,
it is that no client is starved and every client sees the same sequence.

### 4.8 Delegation

Keep the existing worker pool and its adapter-per-skill design, which is
validated. Apply the two caps from Hermes: a hard `max_iterations` per delegate
(50 is the reference) and a separate event aggregate per worker so a delegated
run's stream is attributable. Delegation must not widen approval scope: a worker
inherits the parent's permission rows and nothing else.

---

## 5. Model: LFM2.5-24B-A2B at ~10 GB

### 5.1 Verified model facts

From the published `config.json` and tokenizer (2026-10-28):

- 23,971,585,280 parameters, `model_type: lfm2_moe`.
- Hidden 2048; 40 layers, 30 short-convolution and 10 full-attention at layers
  2, 6, 10, …, 38; 2 dense layers.
- 64 experts, top-4 per token; `moe_intermediate_size` 1536; dense intermediate
  11,776; `use_expert_bias`.
- Vocabulary 128,000, `tie_word_embeddings: true`. This matters: the output-logit
  memory that gates LoRA on this 16 GB machine is below the 14B's 151,936, so this
  base is trainable in principle (as the 8B sibling already demonstrated).
- Context 131,072; RoPE theta 1e7; conv cache 3.
- KV is cheap: only the 10 attention layers keep a cache, ~20 KB per token, so
  16k context costs roughly 0.3 GB.
- Tokenizer class `TokenizersBackend`; the repo ships `chat_template.jinja`; EOS
  124900. The template renders `<|im_start|>user … <|im_end|>\n<|im_start|>assistant\n think`
  and renders tool definitions.

Runtime on this machine: `mlx_lm 0.31.3`, `mlx 0.32.0`, and `mlx_lm.models`
contains `lfm2_moe`. `quantized_matmul` works at 2/3/4/6/8 bits × group 32/64/128.
`gather_qmm` requires explicit `group_size` and `bits`; the repo's pager already
passes them (`symbio/app/paged_moe.py:452`, `bench/paged_moe/train_lfm.py:234`).

Metal's recommended working set is 12,713,115,648 bytes — 11.84 GiB. That number
is the real ceiling, not 16 GB.

### 5.2 Artifact landscape

| Artifact | Bits / group | Size | Loadable | Note |
|---|---|---|---|---|
| LiquidAI MLX-4bit | 4 / 64, 6-bit embed, fp32 gates | 13.585 GB | yes | does not fit resident |
| LiquidAI MLX-5/6/8bit | 5/6/8 | 16.55 / 19.51 / 25.50 GB | yes | irrelevant here |
| LiquidAI MLX-bf16 | 16 | 47.97 GB | yes | source for conversion only |
| mlx-community / nativ Preview-4bit | 4 / 64 | 13.51 GB | no | vision config + `language_model.` tensor prefix; upstream 401 |
| hermitdave DWQ-3bit | 3 / 64 | 10.509 GB | yes | older variant |
| **hermitdave DWQ-3bit-v2** | 3 / 64, 6-bit embed, bf16 gates | **10.612 GB** | yes | built from the final base |
| Custom 3-bit, group 128 | 3 / 128 | **~9.84 GB** | yes if built | needs bf16 source on a bigger box |
| GGUF UD-Q3_K_XL | Q3 | 10.947 GB | via llama.cpp | no MLX backend today |
| GGUF UD-IQ3_XXS | IQ3_XXS | 8.959 GB | via llama.cpp | quality hit, same gap |

"Q_3XXL" matches no MLX artifact. The nearest real names are Unsloth's
`UD-Q3_K_XL` (GGUF) and the DWQ-3bit repos (MLX). The 3-bit math is exact:
23.709B non-embedding parameters × 3.5 bits/parameter ÷ 8 ≈ 10.37 GB plus
0.213 GB of 6-bit embeddings. At group size 128 the scales cost 0.25 bit instead
of 0.5, which is where 9.84 GB comes from.

A local bf16 conversion is not possible: it needs a 47.97 GB download and
materializes the model in RAM. The machine has ~22-24 GiB free disk and 16 GB
memory.

### 5.3 Routes

| Route | Disk | Resident | Decode | Main risk |
|---|---|---|---|---|
| DWQ-3bit-v2, resident | 10.61 GB, ~13 GB free | ~10.6 GB + 0.3 KV | 45-60 tok/s (est., UNVERIFIED) | ~70% of 16 GB; no browser alongside; re-quant quality |
| DWQ-3bit-v2 + pager | 10.61 GB | ~2.4-4 GB | 15-30 tok/s (est., UNVERIFIED) | 3-bit + paging never driven for serving |
| Official 4-bit + pager | 13.59 GB, ~10 GB free | 2.4 GB res, 5.8 GB peak (measured) | 15-25 tok/s (UNVERIFIED) | the proven path; adapters were trained here |
| Custom 3-bit gs=128 | 9.84 GB | ~9.9 GB | 50-60 tok/s (est.) | build on a cloud box from bf16; extra quant loss |
| GGUF UD-Q3_K_XL + llama-server | 10.95 GB | ~8 GB hot | ~40 tok/s (est.) | no Symbio GGUF backend; no adapters, no pager |
| GGUF UD-IQ3_XXS | 8.96 GB | ~9 GB mmap | ~40 tok/s (est.) | same gap; larger quality hit |

### 5.4 Recommendation

Two checkpoints, two jobs:

1. **Serving: hermitdave `LFM2.5-24B-A2B-MLX-DWQ-3bit-v2` (10.61 GB), resident.**
   It is the only 3-bit MLX build made from the final base rather than the
   orphaned Preview, it loads in the installed `mlx_lm`, and 10.61 GB sits under
   the 11.84 GiB Metal ceiling. This is the "24B at ~10 GB, fast" answer. It is a
   re-quantization, so it must pass a sanity gate before it is trusted.
2. **Training: the official 4-bit checkpoint (13.59 GB) through the pager.**
   The existing adapters and the LoRA recipe were built against it, at 2.4 GB
   resident and 5.4-5.8 GB peak. Do not expect adapters trained on the 4-bit
   checkpoint to load onto a 3-bit base; the quantized weights differ.

If 9.9 GB is a hard number rather than a target, only two paths reach it: a
custom group-128 3-bit conversion built from the bf16 checkpoint on a machine with
≥64 GB RAM (~9.84 GB), or the 8.96 GB GGUF `UD-IQ3_XXS` served by `llama-server`
(installed, `lfm2moe` compiled in) behind a new backend seam. Both cost more than
the 0.8 GB they save.

### 5.5 What "fast" actually requires

The pager is a demonstration, not a speed path: it issues roughly 500
Python-level byte reads per token, and the 35B measured 2.8 tok/s. The 24B's
serving throughput has never been recorded (the governor is marked
"not yet seam-stable"). A resident 3-bit checkpoint sidesteps the read path
entirely, which is why it is the recommendation for the desktop's model chip.

Prerequisites before the model is usable in Symbio:

- LFM2.5 uses a third tool-call wire format,
  `<|tool_call_start|>[name(args)]<|tool_call_end|>`. `chat_tools` does not speak
  it yet, so a native tool-call benchmark would grade the parser, not the model.
- `models.json` needs an entry for the DWQ checkpoint, and `agent.draft_model`
  must stay empty: no draft model shares the 128k tokenizer yet.
- Wired memory: set `gpu.wired_limit_mb` so an 11 GB resident model does not page
  to SSD.

### 5.6 Validation gates

Cheap first, in this order, stopping at the first failure:

1. Load in `mlx_lm` without patching, and generate "17 × 23" plus one word
   problem. A community quant that cannot answer arithmetic is rejected here, as
   the earlier model hunt learned.
2. Emit one LFM-format tool call through the real `chat_tools` path.
3. Measure decode tokens/s and prefill tokens/s at 1k context, plus peak RSS.
4. Only then spend the time on `bfcl_simple` first-120 against the registry
   protocol; the board to beat is granite-4.0-h-tiny 97.50 at 0.8 s/task, falcon3-10b
   95.83, and the LFM2.5-8B-A1B sibling 91.67 at 3.3 s/task. A 24B that scores
   below 91.67 is not worth its 10.6 GB.
5. Golden battery only after an adapter is trained on that exact base.

### 5.7 Known unknowns

- Quality of the DWQ 3-bit re-quantization against the official 4-bit: unmeasured.
- Decode speed resident vs paged at 3 bits: unmeasured on this machine.
- Whether `lfm2_moe` at 3 bits preserves tool-form reliability: unmeasured.
- Whether the 10.61 GB checkpoint leaves enough headroom for the desktop's browser
  and the WebKit helpers: likely not; treat it as an "alone on the machine" tier.

---

## 6. Migration plan

Each phase is independently useful and has a kill criterion.

**P1 — model trial (no code).** Download the DWQ-3bit-v2 checkpoint, run gates
1-3 from §5.6, add a `models.json` entry. Kill: garbage text, wrong arithmetic,
or decode below 20 tok/s.

**P2 — event log (server).** Add `event` and `event_sequence` tables, emit the v2
frames alongside the current ones, teach the desktop bridge to subscribe with
`since_seq`, and add gap detection. Kill: any UI regression on the existing
screens; the old frame types stay until parity is proven.

**P3 — approvals as rows.** Add the `permission` table and route answers through
it; stop rewriting guardrail modes on "always". Kill: any path where an action
proceeds without a row or a floor check.

**P4 — one registry.** Merge `tooling.py` and `tools.py` into `ToolSpec`; delete
`AIAgent.run` or make it a client. Kill: the count assertion in §4.2 cannot be
made to pass without deleting a tool that has a test.

**P5 — turn state machine.** Extract the states and predicates from
`_agent_turn_body` one ladder at a time, each behind a test that reproduces the
current behaviour. Kill: more than one ladder failing its existing test.

**P6 — desktop adoption.** Queued-message strip, tool card, reasoning
disclosure, scoped approval outcomes, structural chrome amendments from §3.5.
Kill: DESIGN.md conflicts that the user has not approved.

**P7 — compression and lineage.** Only after the event log exists, because a
compaction is an event.

Sequencing note: P2 and P3 are worth doing even if P4 and P5 never happen.
They fix reconnect, steering, approvals and headless fail-closed behaviour without
touching the turn body.

---

## 7. Open questions

1. When several clients are attached (desktop, TUI, MCP), whose approval answer
   wins? Proposal: the first answer resolves the card, every client receives
   `approval.resolved`, and the card names which client answered.
2. Should the desktop become the default window again, given the 280 MB WebKit
   cost, or stay opt-in behind `--window`? The event log makes the browser path
   better, so the honest answer is "stay opt-in until measured".
3. Is 9.9 GB a hard requirement? If yes, plan the cloud conversion; if no, the
   10.61 GB checkpoint is the recommendation.
4. Does the user want the 24B as the headmaster, or as a worker under the 14B?
   The 14B currently wins on measured tool reliability; the 24B is a candidate
   that must clear gate 4 before it replaces anything.
