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
from symbio.app import golden, tooling, training


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
            ideal_reply=(
                "The city council approved a downtown bike lane network, with "
                "construction starting next spring and expected completion by late summer 2027."
            ),
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
            ideal_reply="click: Sign in",
        ),
    ],
}


def adapter_matches_model(adapter_dir: Path, model_name: str) -> bool:
    """True when a worker's adapter was trained for the model about to load it.

    mlx_lm attaches LoRA layers sized to the *new* model and then loads the
    adapter with strict=False, so weights from a different base are silently
    discarded rather than rejected — a 4B loaded with an 8B adapter (hidden
    2560 against 4096) comes up as an untrained model with no error anywhere.
    Nothing downstream can tell that apart from a genuinely weak fine-tune, so
    a golden baseline measured that way is measuring the base model while
    reporting the adapter's name.

    Missing or unreadable provenance is treated as a match: older adapters
    predate the stamp, and refusing them would be a regression.
    """
    config_file = adapter_dir / "adapter_config.json"
    if not config_file.exists():
        return False
    try:
        trained_for = json.loads(config_file.read_text(encoding="utf-8")).get("model")
    except (OSError, json.JSONDecodeError):
        return True
    if not trained_for:
        return True
    from symbio.app.skills import _model_stem

    return _model_stem(trained_for) == _model_stem(model_name)


def _adapter_fits_model(model: Any, adapter_file: Path) -> bool:
    """True when every tensor in the adapter has a same-shaped home in `model`.

    This check is the whole safety of hot-swapping. `load_weights(strict=False)`
    silently ignores keys it cannot place, so an adapter trained with a
    different rank, num_layers or target `keys` would load nothing at all and
    leave the previous worker's weights sitting in the model — which does not
    raise, does not warn, and answers confidently as the wrong specialist.
    Refusing the swap costs one full load; getting it wrong costs correctness.
    """
    try:
        import mlx.core as mx
        from mlx.utils import tree_flatten

        weights = mx.load(str(adapter_file))
        if not weights:
            return False
        params = dict(tree_flatten(model.parameters()))
        for key, value in weights.items():
            current = params.get(key)
            if current is None or tuple(current.shape) != tuple(value.shape):
                return False
        return True
    except Exception:
        # Any doubt at all falls back to the full load, which is always correct.
        return False


def _adapter_recipe(adapter_dir: Path) -> dict[str, Any]:
    """The LoRA knobs that decide which tensors an adapter contains.

    Two adapters over the same base weights are only interchangeable in a
    resident model when these agree: they determine which modules get LoRA
    layers attached and how big those layers are.
    """
    try:
        cfg = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    lora = cfg.get("lora_parameters") or {}
    return {
        "num_layers": cfg.get("num_layers"),
        "rank": lora.get("rank"),
        "keys": tuple(lora.get("keys") or ()) or ("<all projections>",),
    }


def describe_recipe_drift(worker_dir: Path, headmaster_dir: Path) -> str | None:
    """Human-readable diff of a worker's LoRA recipe against the headmaster's.

    Skill workers share the headmaster's base weights and are meant to cost
    only an adapter swap. Retraining the headmaster with different targets
    silently makes every older skill adapter unswappable, and the fallback is
    a second full copy of the headmaster-sized weights. Naming the drift is
    the difference between a one-line fix and hunting an OOM.
    """
    worker, headmaster = _adapter_recipe(worker_dir), _adapter_recipe(headmaster_dir)
    if not worker or not headmaster:
        return None
    drift = [f"{k}: worker={worker[k]!r} vs headmaster={headmaster[k]!r}"
             for k in worker if worker[k] != headmaster[k]]
    return "; ".join(drift) or None


def label_worker_reply(role: str, reply: str) -> str:
    """Frame a worker's reply so a tool call in it reads as proposed, not done.

    Nothing executes a worker's tool tags. Its reply comes back as the
    headmaster's observation, and an observation is where *results* live — so
    a worker that correctly answered "<search>Tallinn mayor current</search>"
    was read as a search that had already run. The headmaster then answered
    the question from memory and presented it as looked-up. A specialist whose
    whole job is choosing the right tool turns into a machine for dressing up
    recall as research, which is worse than not delegating at all.

    Saying plainly that the call has not run puts it back in the headmaster's
    hands, which is the same "suggest, don't route" stance the rest of
    dispatch takes.
    """
    if not reply:
        return f"Worker '{role}' returned nothing."
    if not tooling.parse_tools(reply):
        return reply
    return (
        f"Worker '{role}' recommends this tool call but it has NOT been run "
        f"and no result exists yet:\n{reply}\n"
        f"Issue the call yourself if you agree with it. Do not answer as "
        f"though it already returned.")


class SecondHeadmasterCopyRefused(RuntimeError):
    """Raised rather than putting a second copy of the headmaster's weights in RAM.

    A worker running the headmaster's own model is supposed to arrive by
    adapter swap. When the swap is refused the fallback is a full load, which
    is correct but doubles the largest allocation on the machine — on a 16 GB
    box, beside a training run, that is the difference between working and a
    kernel panic. Failing the delegation loudly beats taking the machine down.
    """


class WorkerPool:
    """Lazy-loads worker models on first delegated task, evicts by LRU once
    dispatch.max_resident_workers is exceeded, and unloads anything idle
    past dispatch.worker_idle_unload_minutes. Defaults to one resident
    worker (sequential swap) to fit alongside the headmaster's own model on
    a typical machine; raise max_resident_workers if you have the RAM to
    keep more loaded at once — it's genuinely respected, not just a stub."""

    def __init__(
        self,
        config: dict[str, Any],
        status_fn=None,
        before_worker_fn=None,
        after_worker_fn=None,
    ):
        self.config = config
        # role -> (model, tokenizer, last_used_ts)
        self._resident: dict[str, tuple[Any, Any, float]] = {}
        # Optional status callback: status_fn(message) is called with
        # user-facing progress lines so a chat front-end can show when workers
        # load and when tasks are delegated.
        self.status_fn = status_fn
        # Optional callbacks so the headmaster can unload itself before a
        # worker runs and reload itself afterwards — saves RAM on modest machines
        # by keeping only the active model resident.
        self.before_worker_fn = before_worker_fn
        self.after_worker_fn = after_worker_fn

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

        swapped = self._try_hot_swap(role, entry)
        if swapped is not None:
            return swapped

        refusal = self._second_headmaster_copy_refusal(role, entry)
        if refusal:
            raise SecondHeadmasterCopyRefused(refusal)

        self._evict_lru_if_needed()
        adapter_dir = constants.adapter_dir_for(role)
        use_adapter = (adapter_dir / "adapter_config.json").exists()
        if use_adapter and not adapter_matches_model(adapter_dir, entry["model_name"]):
            use_adapter = False
            self._status(
                f"  [Dispatch] Worker '{role}' has an adapter trained for a "
                f"different model; loading base weights only. Retrain the "
                f"worker to rebuild it for {entry['model_name']}.")
        self._status(f"  [Dispatch] Loading worker '{role}' ({entry['model_name']})...")
        if use_adapter:
            model, tokenizer = load(entry["model_name"], adapter_path=str(adapter_dir))
        else:
            model, tokenizer = load(entry["model_name"])
        self._resident[role] = (model, tokenizer, time.time())
        training.mark_adapter_used(role=role)
        self._status(f"  [Dispatch] Worker '{role}' ready.")
        return model, tokenizer, entry

    def _second_headmaster_copy_refusal(self, role: str, entry: dict[str, Any]) -> str | None:
        """Message explaining why `role` must not be loaded, or None to allow it.

        Only fires for workers on the headmaster's own model, which are exactly
        the ones hot-swapping exists to make free. Deep sleep unloads the
        headmaster first, so one copy is resident either way and there is
        nothing to refuse.
        """
        cfg = self._dispatch_cfg()
        if cfg.get("allow_second_headmaster_copy", False):
            return None
        if cfg.get("headmaster_deep_sleep_while_workers", False):
            return None
        headmaster_model = self.config.get("model_name")
        if not headmaster_model:
            return None
        from symbio.app.skills import _model_stem

        if _model_stem(entry.get("model_name") or "") != _model_stem(headmaster_model):
            return None
        drift = describe_recipe_drift(constants.adapter_dir_for(role), constants.ADAPTER_DIR)
        return (
            f"Worker '{role}' runs the headmaster's own model ({headmaster_model}) "
            f"and its adapter could not be hot-swapped"
            + (f" ({drift})" if drift else "")
            + ". Loading it would hold two full copies of those weights at once. "
            f"Retrain '{role}' with the headmaster's current LoRA recipe, or set "
            f"dispatch.headmaster_deep_sleep_while_workers to unload the "
            f"headmaster first, or dispatch.allow_second_headmaster_copy if you "
            f"have the RAM to spare."
        )

    def _try_hot_swap(self, role: str, entry: dict[str, Any]):
        """Reuse a resident model by replacing only its LoRA tensors.

        Skill workers all run the headmaster's own base weights and differ
        only by a ~19 MB adapter, so loading a second full copy to switch
        between them costs gigabytes and tens of seconds to change nothing but
        that adapter. Swapping the tensors in place is effectively free, which
        is what makes per-turn routing to a skill affordable at all.

        Returns the same triple as get(), or None when a swap is not safe and
        the caller should fall back to a full load.
        """
        if not self._dispatch_cfg().get("hot_swap_adapters", True):
            return None
        adapter_file = constants.adapter_dir_for(role) / "adapters.safetensors"
        if not adapter_file.exists():
            # Nothing to swap in. Reusing a model would silently leave the
            # previous worker's adapter attached and answer as the wrong
            # specialist, so this has to be a full load of the base weights.
            return None

        for donor, (model, tokenizer, _) in list(self._resident.items()):
            donor_entry = catalog_entry_for_role(donor)
            if not donor_entry:
                continue
            if donor_entry.get("model_name") != entry.get("model_name"):
                continue  # different base weights; nothing to reuse
            if not _adapter_fits_model(model, adapter_file):
                drift = describe_recipe_drift(
                    constants.adapter_dir_for(role), constants.ADAPTER_DIR)
                self._status(
                    f"  [Dispatch] Cannot swap '{donor}' -> '{role}': the adapter "
                    f"does not fit the resident model's LoRA layers"
                    + (f" ({drift})" if drift else "")
                    + f". Retrain worker '{role}' with the headmaster's current "
                    f"recipe; until then it needs a second full copy of "
                    f"{entry.get('model_name')}.")
                continue
            try:
                model.load_weights(str(adapter_file), strict=False)
            except Exception as exc:
                self._status(f"  [Dispatch] Adapter swap for '{role}' failed "
                             f"({exc}); loading the worker in full.")
                return None
            # The donor's model now carries someone else's adapter, so its
            # residency moves rather than being copied.
            del self._resident[donor]
            self._resident[role] = (model, tokenizer, time.time())
            training.mark_adapter_used(role=role)
            self._status(f"  [Dispatch] Swapped adapter '{donor}' -> '{role}' "
                         f"(no model reload).")
            return model, tokenizer, entry
        return None

    def _worker_system_prompt(self, role: str, entry: dict[str, Any]) -> str:
        """Return the system prompt for a worker role.

        Builtin roles use ROLE_SYSTEM_PROMPTS; skill/worker catalog entries
        may carry their own system_prompt field.
        """
        if role in ROLE_SYSTEM_PROMPTS:
            return ROLE_SYSTEM_PROMPTS[role]
        if entry and "system_prompt" in entry:
            return entry["system_prompt"]
        return "Complete the following task concisely and directly."

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

        deep_sleep = bool(self._dispatch_cfg().get("headmaster_deep_sleep_while_workers", False))
        if deep_sleep and self.before_worker_fn is not None:
            self._status("  [Dispatch] Putting headmaster to sleep before loading worker...")
            self.before_worker_fn()

        try:
            try:
                loaded = self.get(role)
            except SecondHeadmasterCopyRefused as exc:
                self._status(f"  [Dispatch] Refused to load '{role}': {exc}")
                return (f"Worker '{role}' was not run: {exc}")
            if loaded is None:
                known = sorted({e.get("role") for e in load_catalog().values() if e.get("role")})
                return f"No worker configured for role '{role}'. Known roles: {', '.join(known) or 'none'}."
            model, tokenizer, entry = loaded
            training.mark_adapter_used(role=role)
            self._status(f"  [Dispatch] Delegating to '{role}': {task[:80]}{'...' if len(task) > 80 else ''}")
            system_prompt = self._worker_system_prompt(role, entry)
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
            return label_worker_reply(role, reply)
        finally:
            if deep_sleep and self.after_worker_fn is not None:
                self._status("  [Dispatch] Waking headmaster back up...")
                self.after_worker_fn()

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
        training.mark_adapter_used(role="browser")
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

    # Refuse a run whose data was rendered by a different model's chat
    # template. Without this the run completes normally and produces an
    # adapter that has learned another model's turn markers.
    from symbio.app import skills as _skills

    mismatch = _skills.seed_model_mismatch(role, entry["model_name"])
    if mismatch:
        return False, mismatch

    dispatch_cfg = config.get("dispatch", {})
    golden_on = dispatch_cfg.get("worker_golden_set_enabled", True)
    sampler = make_sampler(temp=0.2, top_p=0.9)
    entry = catalog_entry_for_role(role)

    cases = WORKER_GOLDEN_CASES.get(role)
    skill_cases = False
    if cases is None:
        # Skills had no hand-written cases, so every skill retrained with no
        # regression check and no rollback at all — the workers that retrain
        # themselves unattended were the only ones unguarded. Their cases can
        # be derived, because unlike the headmaster a skill's correct answer
        # is known: it is its own steps.
        from symbio.app import skill_eval as _skill_eval

        try:
            cases = _skill_eval.golden_cases_for_role(role)
            skill_cases = cases is not None
        except Exception:
            cases = None

    if entry and "system_prompt" in entry:
        system_prompt = entry["system_prompt"]
    else:
        system_prompt = ROLE_SYSTEM_PROMPTS.get(
            role, "Complete the following task concisely and directly.")

    if skill_cases and entry:
        # A skill's *served* prompt contains its steps, as a safety net for a
        # weak adapter. Grading against that prompt would measure whether the
        # model can copy a procedure out of its own context, so a broken
        # adapter would pass and ship. Grade under the stripped prompt the
        # adapter was trained on instead, where only the weights can answer.
        from symbio.app import skills as _skills

        system_prompt = _skills.build_worker_system_prompt(
            entry.get("skill_name", role))

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
    # A baseline is only meaningful against weights this model can actually
    # load. An adapter from a different base is discarded silently, so the
    # "before" score would be the base model wearing the adapter's name — and
    # every later comparison would be against that fiction.
    if (adapter_dir.exists()
            and (adapter_dir / "adapter_config.json").exists()
            and adapter_matches_model(adapter_dir, entry["model_name"])):
        base_model, base_tok = load(entry["model_name"], adapter_path=str(adapter_dir))
        baseline = _run_golden(base_model, base_tok)
        backup_dir = training.backup_adapter(role=role)
        # The baseline scores are all we need from these weights, and the
        # trainer below loads its own full copy. Holding this one until the
        # function returns would mean two copies resident across the whole
        # run — on a unified-memory Mac that is the difference between a slow
        # fine-tune and an out-of-memory kill.
        base_model, base_tok = None, None
        training.release_model()

    try:
        trained = training.run_training(
            config, iters=iters, role=role, model_name=entry["model_name"])
        if not trained:
            return False, (
                f"Training for '{role}' produced no usable adapter. Check the "
                f"log above: either there was no new data, or the run ended "
                f"before any checkpoint was written."
            )

        new_model, new_tok = load(entry["model_name"], adapter_path=str(adapter_dir))
        training.mark_adapter_used(role=role)

        if baseline is None:
            return True, f"Worker '{role}' trained."

        after = _run_golden(new_model, new_tok)
        regressions = sorted(baseline.passing - after.passing) if after else []
        threshold = int(dispatch_cfg.get("worker_golden_regression_threshold", 0))

        if (len(regressions) > threshold
                and dispatch_cfg.get("worker_golden_retry_enabled", True)):
            print(
                f"  [Golden] Double-checking {len(regressions)} regression(s) "
                f"for worker '{role}'..."
            )
            recheck, consistent = golden.run_golden_set_retry(
                new_model, new_tok, generate, sampler, system_prompt, config,
                enabled_groups=None, cases=cases,
            )
            flaky = sorted(set(regressions) - consistent)
            if flaky:
                print(
                    f"  [Golden] {len(flaky)} regression(s) passed on recheck: "
                    f"{', '.join(flaky)}"
                )
            if not consistent:
                print("  [Golden] All regressions were flaky; using recheck result.")
                after = recheck
                regressions = sorted(baseline.passing - after.passing)
            else:
                print(
                    f"  [Golden] {len(consistent)} case(s) consistently failing "
                    f"for worker '{role}': {', '.join(sorted(consistent))}"
                )
                extra_iters = int(dispatch_cfg.get("worker_golden_retry_max_extra_iters", 50))
                copies = int(dispatch_cfg.get("worker_golden_retry_samples_per_case", 3))
                worker_cases = WORKER_GOLDEN_CASES.get(role) or []
                extra_ideals = {
                    case.id: case.ideal_reply
                    for case in worker_cases
                    if case.ideal_reply
                }
                added = golden.append_golden_remedy_samples(
                    sorted(consistent), new_tok, system_prompt, config,
                    role=role, copies=copies,
                    extra_ideal_replies=extra_ideals,
                    extra_cases=worker_cases,
                )
                if added:
                    print(
                        f"  [Train] Injected {added} remedy sample(s) for worker "
                        f"'{role}' consistent failures."
                    )
                    print(
                        f"  [Train] Running targeted remedy training ({extra_iters} "
                        f"iters) for worker '{role}'..."
                    )
                    # Same reason as the baseline model above: the remedy
                    # trainer loads its own copy of the weights, so this one
                    # has to go first. Everything still needed from it — the
                    # remedy samples — has already been written out.
                    new_model, new_tok = None, None
                    training.release_model()
                    trained2 = training.run_training(
                        config, iters=extra_iters, role=role, model_name=entry["model_name"])
                    if trained2:
                        new_model, new_tok = load(
                            entry["model_name"], adapter_path=str(adapter_dir))
                        print(
                            "  [Train] Remedy adapter reloaded. Re-checking worker "
                            f"'{role}' golden set..."
                        )
                        after = _run_golden(new_model, new_tok)
                        regressions = sorted(baseline.passing - after.passing) if after else []
                else:
                    print("  [Train] No remedy samples could be generated.")

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
