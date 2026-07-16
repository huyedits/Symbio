# Plan: Symbio — Hermes-Style Personal AI Agent

## Core Idea (preserved)
- A personal, autonomous, self-fine-tuning AI assistant (default name **Symbio**) customized to each user.
- XML-style tool tags let the assistant act on its own: save notes, run commands, digest notes, train.
- Persistent memory through markdown notes, digested into `training_data/*.jsonl`, and baked into a LoRA adapter via `mlx_lm`.
- Per-user configurable identity (assistant name + user name) via first-run setup, `/setup`, and natural-language triggers in chat.
- Small MLX base model: `mlx-community/Qwen2.5-3B-Instruct-4bit`.
- Sandboxed command execution and clean responses with no leaked reasoning/thinking tokens.

## Hermes Agent Features Adopted

### 1. `AIAgent` Class Architecture
- `AIAgent` owns the model, tokenizer, conversation history, tool registry, sampler, session log, and SQLite session store.
- Entry points:
  - `agent.run(user_input)` — full turn result (text + tool executions + observations + history).
  - `agent.chat(user_input)` — simple final text reply.
- The CLI `chat_loop()` is a thin wrapper over `AIAgent` that handles slash commands and end-of-session saving.

### 2. Tool Registry with JSON Schemas
- Tools are registered with `name`, `description`, `parameters` (JSON-schema style), and a `readonly` flag.
- The system prompt lists available tools by name and shows both legacy XML tags and the Hermes `<tool_call>` format.
- Full registry includes:
  - `memory` — save, update, or remove facts as markdown notes in `notes/`
  - `read_file` — read a file inside the project directory
  - `write_file` — write or replace a file inside the project directory
  - `patch` — targeted find-and-replace edit to a file
  - `search_files` — keyword search in project files
  - `terminal` — sandboxed shell command
  - `execute_code` — run a short Python snippet in the sandbox
  - `web_search` — web search stub (configure externally to enable real search)
  - `web_extract` — page-to-markdown extraction stub
  - `list_threads` — list unread email threads via IMAP
  - `get_thread` — read a specific email by id
  - `send_message` — send an email via SMTP
  - `reply_to_message` — reply to an email by id
  - `digest_notes` — convert notes to training samples
  - `train_adapter` — fine-tune the LoRA adapter
  - `session_search` — full-text search of past conversation turns

### 3. Dual-Format Tool Parsing
- The parser accepts:
  - Legacy compact tags: `<note title="T">body</note>`, `<cmd>command</cmd>`, `<digest />`, `<train />`.
  - Hermes JSON-in-XML: `<tool_call>{"name": "...", "arguments": {...}}</tool_call>`.
- Legacy tags are internally mapped to registry tools where applicable.
- The parser is robust to apostrophes inside single-quoted note titles.
- Both formats are stripped from the user-facing display.

### 4. Parallel Tool Execution
- Consecutive **read-only** tools (`read_file`, `search_files`, `web_search`, `web_extract`, `list_threads`, `get_thread`, `session_search`) execute concurrently via `ThreadPoolExecutor`.
- Mutating tools run sequentially to preserve deterministic side-effect order.
- Observations are collected in tool order and fed back to the model as a single system observation.

### 5. Anti-Loop Safeguards
- Tool calls are deduplicated by signature within a single user turn.
- Each **mutating** tool type can only execute once per user turn (e.g. one `write_file`), preventing runaway note/command loops while still allowing a `digest_notes` → `train_adapter` chain.

### 6. Session Persistence
- Every turn is appended to `logs/session_YYYY-MM-DD_HH-MM-SS.jsonl`.
- The same data is stored in `logs/sessions.db` (SQLite with FTS5 search fallback).
- `session_search` queries past conversation turns by keyword.
- The session log path is shown in `/status`.

### 7. Loop Control
- `agent.max_turns` (config key `agent.max_turns`) limits how many tool/action rounds the agent may perform per user turn, with a backward-compatible fallback to `max_tool_rounds`.
- History is soft-truncated to `history_limit + 8` turns to avoid unbounded context growth.

### 8. Code Execution
- `execute_code` writes a Python script to a temp file in `sandbox/` and runs it with the venv interpreter.
- The sandbox can import a whitelisted `caine_tools` stub to access safe file/web/terminal helpers.
- Subject to the same timeout and output-length limits as `terminal`.

### 9. Email Gateway
- `list_threads` / `get_thread` use `imaplib` to fetch unread mail.
- `send_message` / `reply_to_message` use `smtplib`.
- Configure via environment variables: `EMAIL_ADDRESS`, `EMAIL_PASSWORD`, `EMAIL_IMAP_HOST`, `EMAIL_SMTP_HOST`; optional `EMAIL_ALLOWED_USERS` allowlist.

### 10. Clean Final Replies
- `clean_response()` strips thinking/reasoning/analysis blocks before display and before writing to training data.
- `strip_generation_artifacts()` removes chat-template artifacts (`<|im_start|>`, `<|im_end|>`, `<tool_response>`, etc.) that the small model may hallucinate.
- Tool tags are removed from the display text.

### 11. Dynamic Identity Updates
- Natural-language name changes are detected before each turn and applied immediately:
  - **User name**: *"My name is Alice."*, *"Call me Bob."*, *"You can call me Charlie."*, *"From now on call me Dana."*, *"Change my name to Eve."*, *"I go by Frank."* → updates `config["user_name"]`.
  - **Assistant name**: *"Call yourself Jarvis."*, *"I will call you Friday."*, *"I'm going to call you HAL."*, *"Change your name to Jeeves."*, *"Set your name as Alfred."* → updates `config["assistant_name"]`.
- The updated names are saved to `config.json`, the identity notes (`notes/My_Identity.md`, `notes/User_Identity.md`) are rewritten, and `AIAgent.update_identity()` rebuilds the system prompt for the next turn.
- Negations and placeholder words ("not Alice", "anything", "whatever") are ignored.
- *"Your name is X"* is intentionally **not** treated as an assistant rename trigger because small models frequently confuse it with the user's name.

## Files
- `main.py` — `AIAgent`, tool registry, dual-format parser, parallel execution, session logging, SQLite store, anti-loop safeguards.
- `seed_training.py` — seed corpus with identity, greetings, legacy XML tags for memory/terminal, and Hermes `<tool_call>` examples for file/web/email/code tools.
- `config.json` — names and LoRA/agent settings; defaults: 3B model, temperature 0.1, max_turns 5, LoRA dropout 0.1, scale 5.0.
- `requirements.txt` — unchanged; `concurrent.futures` is stdlib.
- `smoke_test.py` — automated validation of identity, memory, file tools, terminal, code execution, email stubs, and search.
- `test_dynamic_names.py` — end-to-end validation of natural-language name changes with real multi-turn history.
- `PLAN.md` — this file.

## Training Run (2026-07-12)
- Regenerated `training_data/train.jsonl` and `training_data/valid.jsonl` from expanded seed examples (89 single-turn + 6 multi-turn samples), including dynamic identity update examples.
- Trained 50 LoRA iterations on `mlx-community/Qwen2.5-3B-Instruct-4bit` with rank 8, dropout 0.1, scale 5.0.
- Final validation loss: ~0.047 (down from 4.455 at iter 1).
- Pruned stale checkpoints, leaving `adapters/adapters.safetensors` (~26 MB).

## Training Run (2026-07-13)
- Removed ambiguous assistant-name trigger *"Your name is X"* from detection patterns and seed examples to prevent identity confusion.
- Added explicit confirmation instruction to the system prompt and more dynamic-identity update examples to `seed_training.py` and `main.py`.
- Regenerated `training_data/*.jsonl` (88 single-turn + 11 multi-turn samples) and retrained 50 LoRA iterations.
- Final validation loss: ~0.045 (down from 4.248 at iter 1).
- Pruned stale checkpoints; adapter footprint ~13 MB.

## Validation Checklist
- [x] Syntax check passes for `main.py` and `seed_training.py`.
- [x] Identity Q&A still works after refactor.
- [x] Legacy `<note>` tags still save notes.
- [x] Hermes `<tool_call>` JSON blocks parse and trigger file/web/email/code tools.
- [x] `<cmd>` / `terminal` run sandboxed commands when asked to check the system.
- [x] `execute_code` runs short Python snippets.
- [x] Email tools return the expected "not configured" message when env vars are absent.
- [x] `/train` still runs LoRA and bakes the adapter.
- [x] `/status` and `/notes` slash commands work.
- [x] Session log and SQLite store are written to `logs/`.
- [x] Natural-language name changes update `config.json`, identity notes, and the runtime system prompt.
