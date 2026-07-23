"""Mixture-of-agents dispatch: the headmaster (the main chat model) can hand
a bounded sub-task off to a smaller, faster worker model instead of running
every micro-decision through its own multi-thousand-token system prompt.

Workers are loaded lazily and evicted by LRU + idle timeout (WorkerPool),
so this stays practical on modest hardware by default while still letting
someone with more RAM raise dispatch.max_resident_workers to keep several
loaded at once. Each worker can be fine-tuned independently, on its own
narrow task data, via the same LoRA + golden-set-guarded-rollback machinery
the headmaster uses — just pointed at the worker's own adapter/data
directory (constants.adapter_dir_for/data_dir_for(role)) instead of the
shared one.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler

from symbio import constants
from symbio.app import golden, training


def load_catalog() -> dict[str, dict[str, Any]]:
    if not constants.WORKER_MODELS_FILE.exists():
        return {}
    try:
        return json.loads(constants.WORKER_MODELS_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def catalog_entry_for_role(role: str) -> dict[str, Any] | None:
    for entry in load_catalog().values():
        if entry.get("role") == role:
            return entry
    return None


# Short, task-scoped system prompts — a worker doesn't carry the
# headmaster's persona, tool catalog, or memory; it only needs its one job.
ROLE_SYSTEM_PROMPTS: dict[str, str] = {
    "summarize": (
        "Summarize the text you are given in 2-4 sentences. Be factual and "
        "concise. Reply with only the summary, nothing else."
    ),
    "browser": (
        "You are given the visible text of a web page and a goal. Reply "
        "with exactly one action: 'click: <exact link/button text>', "
        "'type: <text>', 'scroll', or 'done' if the goal is already "
        "satisfied by the page text shown. No explanation, just the action."
    ),
}

# A tiny golden set per role — not the headmaster's identity/tool-tag
# battery (irrelevant to a worker), just "does it still follow its one
# action grammar." Reuses golden.GoldenCase / golden.sane_reply so the
# same overfit/degenerate-output guard applies to workers too.
WORKER_GOLDEN_CASES: dict[str, list[golden.GoldenCase]] = {
    "summarize": [
        golden.GoldenCase(
            "summarize_produces_output", "Produces a non-empty, non-degenerate summary",
            lambda cfg: (
                "The city council voted 5-2 Tuesday to approve the new bike lane "
                "network downtown, with construction beginning next spring and "
                "expected to finish by late summer 2027."
            ),
            lambda display, tools, cfg: bool(display.strip()) and golden.sane_reply(display),
        ),
    ],
    "browser": [
        golden.GoldenCase(
            "browser_emits_known_action", "Replies with one of the known action verbs",
            lambda cfg: "Page text: 'Sign in' link is visible top-right. Goal: log in.",
            lambda display, tools, cfg: golden.sane_reply(display) and any(
                display.strip().lower().startswith(verb)
                for verb in ("click:", "type:", "scroll", "done")
            ),
        ),
    ],
}


class WorkerPool:
    """Lazy-loads worker models on first delegated task, evicts by LRU once
    dispatch.max_resident_workers is exceeded, and unloads anything idle
    past dispatch.worker_idle_unload_minutes. Defaults to one resident
    worker (sequential swap) to fit alongside the headmaster's own model on
    a typical machine; raise max_resident_workers if you have the RAM to
    keep more loaded at once — it's genuinely respected, not just a stub."""

    def __init__(self, config: dict[str, Any], status_fn=None):
        self.config = config
        # role -> (model, tokenizer, last_used_ts)
        self._resident: dict[str, tuple[Any, Any, float]] = {}
        # Optional status callback: status_fn(message) is called with
        # user-facing progress lines so a chat front-end can show when workers
        # load and when tasks are delegated.
        self.status_fn = status_fn

    def _dispatch_cfg(self) -> dict[str, Any]:
        return self.config.get("dispatch", {})

    def _evict_idle(self):
        idle_minutes = float(self._dispatch_cfg().get("worker_idle_unload_minutes", 10))
        if idle_minutes <= 0:
            return
        cutoff = time.time() - idle_minutes * 60
        for role in [r for r, (_, _, ts) in self._resident.items() if ts < cutoff]:
            del self._resident[role]

    def _evict_lru_if_needed(self):
        max_resident = max(1, int(self._dispatch_cfg().get("max_resident_workers", 1)))
        while len(self._resident) >= max_resident:
            oldest_role = min(self._resident, key=lambda r: self._resident[r][2])
            del self._resident[oldest_role]

    def loaded_roles(self) -> list[str]:
        return list(self._resident)

    def _status(self, message: str):
        if self.status_fn is not None:
            self.status_fn(message)

    def get(self, role: str) -> tuple[Any, Any, dict[str, Any]] | None:
        """Return (model, tokenizer, catalog_entry) for `role`, loading it
        (with its own adapter, if trained) if not already resident. None if
        no catalog entry exists for that role."""
        self._evict_idle()
        if role in self._resident:
            model, tokenizer, _ = self._resident[role]
            self._resident[role] = (model, tokenizer, time.time())
            return model, tokenizer, (catalog_entry_for_role(role) or {})

        entry = catalog_entry_for_role(role)
        if entry is None:
            return None

        self._evict_lru_if_needed()
        adapter_dir = constants.adapter_dir_for(role)
        adapter_config = adapter_dir / "adapter_config.json"
        self._status(f"  [Dispatch] Loading worker '{role}' ({entry['model_name']})...")
        if adapter_config.exists():
            model, tokenizer = load(entry["model_name"], adapter_path=str(adapter_dir))
        else:
            model, tokenizer = load(entry["model_name"])
        self._resident[role] = (model, tokenizer, time.time())
        training.mark_adapter_used(role=role)
        self._status(f"  [Dispatch] Worker '{role}' ready.")
        return model, tokenizer, entry

    def run_delegated_task(self, role: str, task: str, max_tokens: int = 300,
                           browser: Any | None = None) -> str:
        """Execute a bounded task on the named worker and return an
        observation string — same contract as one of
        ChatSession._execute_tool's tool observations. The 'browser' role
        drives a multi-round click/type/scroll loop (see
        _run_browser_delegation) when a live BrowserSession is passed;
        every other role is a single-shot generation. Both record their
        (input, output) pairs as training samples for that worker, so real
        usage accumulates the corpus guarded_train_worker draws on."""
        if role == "browser" and browser is not None:
            max_rounds = int(self._dispatch_cfg().get("max_worker_rounds", 4))
            return self._run_browser_delegation(task, browser, max_rounds)

        loaded = self.get(role)
        if loaded is None:
            known = sorted({e.get("role") for e in load_catalog().values() if e.get("role")})
            return f"No worker configured for role '{role}'. Known roles: {', '.join(known) or 'none'}."
        model, tokenizer, entry = loaded
        self._status(f"  [Dispatch] Delegating to '{role}': {task[:80]}{'...' if len(task) > 80 else ''}")
        system_prompt = ROLE_SYSTEM_PROMPTS.get(
            role, "Complete the following task concisely and directly.")
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        try:
            reply = generate(
                model, tokenizer, prompt=prompt,
                sampler=make_sampler(temp=0.2, top_p=0.9),
                max_tokens=max_tokens, verbose=False,
            ).strip()
        except Exception as e:
            self._status(f"  [Dispatch] Worker '{role}' failed: {e}")
            return f"Worker '{role}' failed: {e}"

        self._status(f"  [Dispatch] Worker '{role}' returned {len(reply.split())} word(s).")
        if reply:
            training.append_chat_pair(task, reply, tokenizer, system_prompt, role=role)
        return reply or f"Worker '{role}' returned nothing."

    def _run_browser_delegation(self, task: str, browser: Any, max_rounds: int) -> str:
        """Drive a bounded click/type/scroll loop on the 'browser' worker
        to accomplish `task` on the currently open page. Each round: worker
        sees the page text and picks one action (click/type/scroll/done);
        we execute it via the same BrowserSession the headmaster's own
        browser_* tools use, then loop with the resulting page text."""
        loaded = self.get("browser")
        if loaded is None:
            return "No worker configured for role 'browser'."
        model, tokenizer, entry = loaded
        system_prompt = ROLE_SYSTEM_PROMPTS["browser"]

        try:
            page_text = browser.get_text()
        except Exception as e:
            return f"Could not read the page: {e}"

        last_action = "none"
        last_status = ""
        for _ in range(max_rounds):
            status_note = f"Result of your last action: {last_status}\n\n" if last_status else ""
            prompt_text = f"Goal: {task}\n\n{status_note}Page text:\n{page_text[:1500]}"
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt_text},
            ]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
            )
            try:
                action = generate(
                    model, tokenizer, prompt=prompt,
                    sampler=make_sampler(temp=0.2, top_p=0.9),
                    max_tokens=60, verbose=False,
                ).strip()
            except Exception as e:
                return f"Worker 'browser' failed: {e}"

            last_action = action
            training.append_chat_pair(prompt_text, action, tokenizer, system_prompt, role="browser")
            lowered = action.lower()

            if lowered.startswith("done"):
                return f"Worker finished: {action}"
            if lowered.startswith("click:"):
                last_status = browser.click(text=action.split(":", 1)[1].strip())
            elif lowered.startswith("type:"):
                last_status = browser.type_text(action.split(":", 1)[1].strip())
            elif lowered.startswith("scroll"):
                last_status = browser.scroll("down")
            else:
                return f"Worker gave an unrecognized action and stopped: {action}"

            try:
                page_text = browser.get_text()
            except Exception as e:
                return f"Worker took action '{action}' but could not read the resulting page: {e}"

        return (
            f"Worker did not finish within {max_rounds} round(s). "
            f"Last action: {last_action} ({last_status})"
        )


def guarded_train_worker(role: str, config: dict[str, Any], iters: int | None = None) -> tuple[bool, str]:
    """Train a worker's own adapter and golden-check it the same way
    ChatSession._guarded_train protects the headmaster's: baseline golden
    run, backup, train, reload, recheck, auto-rollback on regression.
    Returns (trained, status_message)."""
    entry = catalog_entry_for_role(role)
    if entry is None:
        return False, f"No worker configured for role '{role}'."

    dispatch_cfg = config.get("dispatch", {})
    golden_on = dispatch_cfg.get("worker_golden_set_enabled", True)
    cases = WORKER_GOLDEN_CASES.get(role)
    sampler = make_sampler(temp=0.2, top_p=0.9)
    system_prompt = ROLE_SYSTEM_PROMPTS.get(
        role, "Complete the following task concisely and directly.")

    def _run_golden(model, tokenizer):
        if not (golden_on and cases):
            return None
        return golden.run_golden_set(
            model, tokenizer, generate, sampler, system_prompt, config,
            enabled_groups=None, cases=cases,
        )

    baseline = None
    backup_dir = None
    adapter_dir = constants.adapter_dir_for(role)
    if adapter_dir.exists() and (adapter_dir / "adapter_config.json").exists():
        base_model, base_tok = load(entry["model_name"], adapter_path=str(adapter_dir))
        baseline = _run_golden(base_model, base_tok)
        backup_dir = training.backup_adapter(role=role)

    try:
        trained = training.run_training(
            config, iters=iters, role=role, model_name=entry["model_name"])
        if not trained:
            return False, "Training skipped (no new data or failed)."

        new_model, new_tok = load(entry["model_name"], adapter_path=str(adapter_dir))
        training.mark_adapter_used(role=role)

        if baseline is None:
            return True, f"Worker '{role}' trained."

        after = _run_golden(new_model, new_tok)
        regressions = sorted(baseline.passing - after.passing) if after else []
        threshold = int(dispatch_cfg.get("worker_golden_regression_threshold", 0))
        if len(regressions) > threshold:
            if backup_dir and dispatch_cfg.get("worker_golden_rollback_on_regression", True):
                training.restore_adapter(backup_dir, role=role)
                return True, (
                    f"Worker '{role}' trained but regressed on {len(regressions)} "
                    f"check(s) ({', '.join(regressions)}); rolled back.")
            return True, (
                f"Worker '{role}' trained but regressed on {len(regressions)} "
                f"check(s) ({', '.join(regressions)}); kept anyway.")
        return True, f"Worker '{role}' trained ({after.pass_count}/{after.total} checks passing)."
    finally:
        training.discard_adapter_backup(backup_dir)
