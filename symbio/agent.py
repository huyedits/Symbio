"""AIAgent class for Symbio."""

from __future__ import annotations

import hashlib
import json
import logging
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_logits_processors, make_sampler

from symbio.chat import build_system_prompt
from symbio.config import can_run_lora, detect_model_type
from symbio.constants import ADAPTER_DIR, DEFAULT_CONFIG, LOG_DIR, PROJECT_DIR
from symbio.learn import _is_system_observation
from symbio.llm import run_training, save_history_pairs
from symbio.store import SessionStore
from symbio.tools import (
    build_tool_registry,
    execute_tools,
    openai_tool_schemas,
    run_single_tool,
    tool_few_shots,
    tool_metadata,
)
from symbio.utils import (
    clean_response,
    has_dangling_tool_call,
    parse_tools,
    strip_dangling_tool_call,
    strip_generation_artifacts,
    strip_tool_tags,
)

from rag import Retriever
from planner import TrainingPlanner

# Browser / desktop automation helpers (lazy-imported inside runners if missing).
try:
    from symbio.computer import (
        BrowserSession,
        desktop_click,
        desktop_move,
        desktop_press,
        desktop_screenshot,
        desktop_type,
    )
except Exception:
    BrowserSession = None  # type: ignore
    desktop_click = desktop_move = desktop_press = desktop_screenshot = desktop_type = None  # type: ignore


logger = logging.getLogger("chat")


class _Spinner:
    """Terminal spinner shown while waiting for visible model output.

    Runs on a daemon thread and anchors itself with carriage returns; stop()
    erases the line so streamed text can take its place. No-op when stdout
    is not a TTY (tests, pipes).
    """

    _FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

    def __init__(self, label: str = "thinking…"):
        self.label = label
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self.active = sys.stdout.isatty()

    def start(self):
        if not self.active or self._thread is not None:
            return
        self._stop_event.clear()

        def _spin():
            i = 0
            while not self._stop_event.wait(0.08):
                frame = self._FRAMES[i % len(self._FRAMES)]
                sys.stdout.write(f"\r{frame} {self.label}")
                sys.stdout.flush()
                i += 1

        self._thread = threading.Thread(target=_spin, daemon=True)
        self._thread.start()

    def stop(self):
        if self._thread is None:
            return
        self._stop_event.set()
        self._thread.join()
        self._thread = None
        sys.stdout.write("\r\033[K")
        sys.stdout.flush()


class _StreamPrinter:
    """Incrementally print the visible (tool-markup-free) part of a streaming reply.

    Tool calls and other tags are suppressed live via strip_tool_tags; the
    name prefix is only shown once the first visible character arrives, so
    tool-only replies print nothing.
    """

    def __init__(self, prefix: str, spinner: _Spinner | None = None):
        self.prefix = prefix
        self.spinner = spinner
        self.printed = ""
        self.prefix_shown = False

    def update(self, full_text: str):
        visible = strip_tool_tags(full_text)
        # Only print monotonic extensions; cleanup can transiently shrink the
        # visible text while a tag is being generated.
        if not visible or not visible.startswith(self.printed):
            return
        delta = visible[len(self.printed):]
        if not delta:
            return
        if not self.prefix_shown:
            if self.spinner is not None:
                self.spinner.stop()
            sys.stdout.write(self.prefix)
            self.prefix_shown = True
        sys.stdout.write(delta)
        sys.stdout.flush()
        self.printed = visible

    def close(self, final_display: str) -> bool:
        """Reconcile streamed output with the final cleaned text.

        Returns True if the reply is now fully printed on screen.
        """
        if not self.prefix_shown:
            return False
        if final_display.startswith(self.printed):
            sys.stdout.write(final_display[len(self.printed):] + "\n")
        else:
            # A retry or final cleanup diverged from what was streamed;
            # reprint the canonical line so the transcript is correct.
            sys.stdout.write("\n" + self.prefix + final_display + "\n")
        sys.stdout.flush()
        self.printed = final_display
        return True


class AIAgent:
    """Hermes-style autonomous agent loop over an MLX model + LoRA adapter."""

    def __init__(
        self,
        config: dict[str, Any],
        model: Any,
        tokenizer: Any,
        adapter_loaded: bool,
    ):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.adapter_loaded = adapter_loaded
        self.tools = build_tool_registry(self)
        self.system_prompt = build_system_prompt(
            config["assistant_name"], config["user_name"], self.tools
        )
        self.history: list[dict[str, str]] = []
        self.sampler = make_sampler(
            temp=config["agent"]["temperature"],
            top_p=config["agent"]["top_p"],
        )
        # Near-greedy sampling on a small overfit model degenerates into
        # repetition loops on out-of-distribution input; penalize repeats.
        self.logits_processors = make_logits_processors(
            repetition_penalty=config["agent"].get("repetition_penalty", 1.15),
            repetition_context_size=64,
        )
        self.session_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.session_log = LOG_DIR / f"session_{self.session_id}.jsonl"
        self.store = SessionStore(PROJECT_DIR / "logs" / "sessions.db")
        self.store.new_session(self.session_id)
        self.retriever = Retriever(
            config, session_store=self.store, exclude_session_id=self.session_id
        )
        self.planner = TrainingPlanner(config)
        self._code_calls_this_turn = 0
        self._browser_session = BrowserSession() if BrowserSession else None

    def _openai_tool_schemas(self) -> list[dict[str, Any]]:
        return openai_tool_schemas(self.tools)

    def _tool_few_shots(self) -> list[dict[str, str]]:
        return tool_few_shots(self.config)

    def _tool_metadata(self, name: str) -> dict[str, Any]:
        return tool_metadata(name, self.tools, self)

    def _execute_tools(self, tools: list[tuple[str, dict[str, Any]]]) -> list[tuple[str, str]]:
        return execute_tools(self, tools)

    def _run_single_tool(self, name: str, params: dict[str, Any]) -> str:
        return run_single_tool(self, name, params)

    def _generate_stream(self, prompt: str, printer: _StreamPrinter | None) -> str:
        """Generate a reply, echoing visible text to stdout as tokens arrive.

        A spinner covers prompt processing and any stretch of non-visible
        output (e.g. while a tool call is being written); it disappears the
        moment the first visible character streams.
        """
        spinner = _Spinner()
        if printer is not None:
            printer.spinner = spinner
        spinner.start()
        parts: list[str] = []
        try:
            for response in stream_generate(
                self.model,
                self.tokenizer,
                prompt=prompt,
                sampler=self.sampler,
                logits_processors=self.logits_processors,
                max_tokens=self.config["agent"].get("max_output_len", 1024),
            ):
                parts.append(response.text)
                # Runaway-loop breaker: degenerate output (repeated chars,
                # cycling patterns) reuses the same 4-grams over and over,
                # while natural text keeps them mostly unique.
                if len(parts) % 16 == 0:
                    tail = "".join(parts)[-320:]
                    if len(tail) >= 320:
                        grams = {tail[i:i + 4] for i in range(len(tail) - 3)}
                        if len(grams) / (len(tail) - 3) < 0.35:
                            break
                if printer is not None:
                    printer.update("".join(parts))
        finally:
            spinner.stop()
        return "".join(parts)

    def update_identity(self, assistant_name: str, user_name: str):
        self.config["assistant_name"] = assistant_name
        self.config["user_name"] = user_name
        self.system_prompt = build_system_prompt(assistant_name, user_name, self.tools)

    def _persist_turn(self, role: str, content: str):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "role": role,
            "content": content,
        }
        with open(self.session_log, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
        self.store.append(self.session_id, role, content)

    def run(self, user_input: str) -> dict[str, Any]:
        self._code_calls_this_turn = 0
        self.history.append({"role": "user", "content": user_input})
        self._persist_turn("user", user_input)

        final_text = ""
        max_turns = self.config["agent"].get("max_turns") or self.config["agent"].get("max_tool_rounds", 10)
        history_limit = self.config["agent"]["history_limit"]

        executed_sigs: set[tuple[str, str]] = set()
        mutating_types_executed: set[str] = set()
        tools_used: list[str] = []

        for _round in range(max_turns):
            messages = [{"role": "system", "content": self.system_prompt}]

            # Retrieve relevant notes/sessions on the first round and inject
            # them as additional context before the conversation history.
            if _round == 0:
                context = self.retriever.build_context(user_input)
                if context:
                    messages.append({"role": "user", "content": context})

            if _round == 0:
                messages.extend(self._tool_few_shots())

            messages.extend(self.history[-history_limit:])

            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
                tools=self._openai_tool_schemas(),
            )

            reply = ""
            tools: list[tuple[str, dict[str, Any]]] = []
            mlx_error = False
            printer = _StreamPrinter(f"{self.config['assistant_name']:8}: ")
            for _attempt in range(2):
                try:
                    # Stream only the first attempt; a retry reconciles later
                    # via printer.close() so text is never shown twice.
                    raw_reply = self._generate_stream(
                        prompt, printer if _attempt == 0 else None
                    )
                except Exception as e:
                    print(f"[MLX Error: {e}]")
                    mlx_error = True
                    break
                reply = strip_generation_artifacts(clean_response(raw_reply.strip()))
                tools = parse_tools(reply)
                # A dangling <tool_call> means generation stopped mid-call;
                # resample once before giving up on the tool call.
                if not has_dangling_tool_call(reply):
                    break
            if mlx_error:
                break
            if not tools:
                # Drop truncated tool markup so it never reaches the user,
                # the history, or the session store (RAG would re-inject it).
                reply = strip_dangling_tool_call(reply)

            unique_tools: list[tuple[str, dict[str, Any]]] = []
            for name, params in tools:
                sig = (name, json.dumps(params, sort_keys=True, ensure_ascii=False))
                if sig in executed_sigs:
                    continue
                executed_sigs.add(sig)
                meta = self._tool_metadata(name)
                if not meta.get("readonly") and name in mutating_types_executed:
                    continue
                if not meta.get("readonly"):
                    mutating_types_executed.add(name)
                unique_tools.append((name, params))
            tools = unique_tools

            display = strip_tool_tags(reply)
            final_text = display

            if display.strip():
                if not printer.close(display):
                    print(f"{self.config['assistant_name']:8}: {display}")
                logger.info(f"{self.config['assistant_name']}: {display}")
            elif printer.prefix_shown:
                # Streamed text that cleanup later removed; end the line.
                print()

            self.history.append({"role": "assistant", "content": reply})
            self._persist_turn("assistant", reply)

            if not tools:
                if not display.strip():
                    final_text = "[Received an empty reply.]"
                    print(f"{self.config['assistant_name']:8}: {final_text}")
                break

            tool_results = self._execute_tools(tools)
            tools_used.extend(name for name, _ in tools)
            observations: list[str] = []
            for name, out in tool_results:
                indented = out.replace("\n", "\n  ")
                print(f"  [Observation {name}] {indented}")
                observations.append(f"{name}: {out}")
                tool_id = f"{name}_{hashlib.md5(out.encode()).hexdigest()[:8]}"
                self.history.append({"role": "tool", "content": out, "tool_call_id": tool_id})
                self._persist_turn("tool", f"[{tool_id}] {out}")

            obs_text = "\n".join(observations)
            observation_msg = (
                f"[System observation — do NOT repeat the same tool call; reply to the user]: {obs_text}"
            )
            self.history.append({"role": "user", "content": observation_msg})
            self._persist_turn("user", observation_msg)
        else:
            final_text = "[Reached the maximum number of turns.]"
            print(f"{self.config['assistant_name']:8}: {final_text}")

        while len(self.history) > history_limit + 8:
            self.history.pop(0)

        self.planner.record_turn(user_input, final_text, tools=list(dict.fromkeys(tools_used)))
        return {"text": final_text, "history": self.history}

    def chat(self, user_input: str) -> str:
        return self.run(user_input)["text"]

    def forget_last(self) -> int:
        removed = 0
        while self.history and self.history[-1]["role"] == "assistant":
            self.history.pop()
            removed += 1
        while (
            self.history
            and self.history[-1]["role"] == "user"
            and not _is_system_observation(self.history[-1]["content"])
        ):
            self.history.pop()
            removed += 1
        return removed

    def save_history_pairs(self) -> int:
        return save_history_pairs(self.history, self.tokenizer, self.system_prompt, planner=self.planner)

    def digest_notes(self) -> int:
        from symbio.llm import digest_notes_to_training
        return digest_notes_to_training(self.tokenizer, self.system_prompt, planner=self.planner)

    def reload_adapter(self):
        model_type = detect_model_type(self.model)
        ok, reason = can_run_lora(self.config, model_type)
        if not ok:
            print(f"  [System] Cannot reload adapter: {reason}")
            return
        self.model, self.tokenizer = load(
            self.config["model_name"], adapter_path=str(ADAPTER_DIR)
        )
        self.adapter_loaded = True
