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
import re
import threading
import time
from pathlib import Path
from typing import Any

from mlx_lm import generate
from symbio.app.modelload import load
from mlx_lm.sample_utils import make_logits_processors, make_sampler

from symbio import constants
from symbio.app import golden, pending, tooling, training


# (path, mtime_ns, size) -> parsed catalog. The file is small, but it is read
# on a path that runs several times per delegated turn — once per resident
# donor inside _try_hot_swap, and again for every entry lookup — so re-parsing
# it each time is work done repeatedly to reach the same answer. Keyed on the
# stat rather than a TTL because the thing that invalidates it is a skill being
# saved or deleted, which rewrites the file; a stale read there would route a
# turn to a worker that no longer exists.
_CATALOG_CACHE: tuple[Any, ...] | None = None


def load_catalog() -> dict[str, dict[str, Any]]:
    global _CATALOG_CACHE
    path = constants.WORKER_MODELS_FILE
    try:
        st = path.stat()
    except OSError:
        _CATALOG_CACHE = None
        return {}
    key = (str(path), st.st_mtime_ns, st.st_size)
    if _CATALOG_CACHE is not None and _CATALOG_CACHE[0] == key:
        return _CATALOG_CACHE[1]
    try:
        catalog = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        _CATALOG_CACHE = None
        return {}
    _CATALOG_CACHE = (key, catalog)
    return catalog


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


# Demonstratives with no antecedent inside the task text. "the page", "this
# file" and friends are just as unresolvable as "that repo" — the worker has
# none of them — but only when nothing in the task supplies the referent, so a
# task carrying a URL or a quoted block is left alone.
_DEICTIC_RE = re.compile(
    r"\b(?:that|this|those|these|the)\s+"
    r"(?:repo|repository|page|site|article|file|document|text|code|"
    r"result|results|output|list|table|data|content|url|link|above|"
    r"conversation|thread|error|log)\b"
    r"|\b(?:it|them)\s*$",
    re.IGNORECASE)


def _unresolved_reference(task: str) -> str | None:
    """The phrase pointing outside the task, or None when the task stands alone.

    A task that carries its own material — a URL to fetch, or a pasted block —
    is self-contained even when it says "this page", so those are exempt.
    """
    if not task:
        return None
    if re.search(r"https?://|\n\s*\n|```", task):
        return None  # carries a URL or a pasted block: self-contained
    m = _DEICTIC_RE.search(task)
    return m.group(0).strip() if m else None


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


# Appended to every worker system prompt, builtin or skill-authored, so it
# applies to catalog entries already on disk without rewriting their stored
# prompts. label_worker_reply catches a question that gets through anyway, but
# catching it there has already cost a headmaster unload, a worker load, and a
# generation — it is much cheaper to tell the worker up front that there is
# nobody on the other end to ask.
_NO_SECOND_TURN = (
    "\n\nYou are answering a single delegated request and there is no second "
    "turn: your reply goes to another model, not to a person, so a question "
    "back can never be answered. Never ask a question or request more "
    "information from the user; if you cannot answer from what you were given, "
    "say so in one line and state what would be needed."
)
# Deliberately NOT here: an instruction to name a tool call. It lived here for
# a day and it poisoned every worker whose skill is a procedure rather than a
# command. Measured 2026-08-24 on the scrape worker, whose four steps are
# prose: with the clause it emitted "<cmd>fetch through the cache proxy on port
# 8817, never the live site twice</cmd>" — each step wrapped as if it were a
# shell command — and the headmaster dutifully ran one, producing "Command not
# found: selectolax". Without it, the same worker recites its procedure, which
# is what it was trained to do. A worker that needs to name a command should
# say so in its OWN steps, where the commands are actually known; a global
# instruction to produce commands makes every worker invent them.


# A worker runs once. Its reply becomes one of the headmaster's observations,
# and nothing routes a follow-up back to it — so a reply that asks for the
# information it was delegated to produce is not a partial result, it is a
# dead end that can never be filled in. Relaying it is worse than dropping it:
# an observation reads as a *result*, so the headmaster restates the worker's
# question to the user as its own and the turn ends waiting for the user to
# answer their own request. Seen live — "what is my keyboard layout?" went to
# a device worker with no tools and came back "Noted. Please provide the name
# of the keyboard layout you are using."
_WORKER_PUNT_RE = re.compile(
    r"\b(?:please|could you|can you|kindly)\s+"
    r"(?:provide|specify|tell me|share|clarify|confirm|let me know)\b"
    r"|\bwhat(?:'s| is)\s+(?:your|the name of)\b"
    r"|\b(?:i|we)\s+(?:need|require)\s+"
    r"(?:you\s+to\s+provide|more\s+(?:information|details|context))\b"
    r"|\b(?:i|we)\s+(?:don't|do not|cannot|can't)\s+have\s+access\b",
    re.IGNORECASE,
)


# A skill worker's steps are prose, and reciting them is what it was trained to
# do — see the note above _NO_SECOND_TURN, which chose recitation over an
# instruction to invent commands. But the recitation comes back through the
# observation channel, where the headmaster reads it as a *result*. Seen live
# 2026-08-24: "scrape the listing page" was answered by the worker restating
# its own four steps, and the headmaster replied "I'll handle the scraping and
# sorting" having scraped nothing, then opened Chrome.
#
# The label has to hold in both directions, because the identical recitation is
# the right answer to "walk me through it" and a non-answer to "do it". So it
# names what the text *is* — a procedure — and leaves the headmaster to decide
# which was asked, rather than declaring the turn a failure.
def _recites_own_steps(reply: str, entry: dict[str, Any] | None) -> bool:
    """True when a skill worker restated its procedure instead of performing it."""
    if not entry or not entry.get("skill_name"):
        return False
    try:
        from symbio.app import skill_eval as _skill_eval
        return _skill_eval.recites_steps(reply, _skill_eval.skill_steps(entry))
    except Exception:
        # Grading is a nicety; a worker reply must still reach the headmaster.
        return False


def label_worker_reply(role: str, reply: str, entry: dict[str, Any] | None = None) -> str:
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
        if _WORKER_PUNT_RE.search(reply):
            return (
                f"Worker '{role}' did not answer — it asked for information "
                f"instead:\n{reply}\n"
                f"It runs once, with no tools and no view of the user's "
                f"machine, and it will never receive a reply, so this is not "
                f"a result. Do NOT pass its question on to the user. Answer "
                f"the request yourself with your own tools, or say plainly "
                f"that you could not determine it."
            )
        if entry and entry.get("advisory"):
            # An advisory worker returns a JUDGEMENT, not a result, and the
            # observation channel does not distinguish the two: the headmaster
            # reads "[System observation: ...]" as something that HAPPENED and
            # defers to it. Measured 2026-08-23 — Qwen3-8B worked "100 days
            # from Wednesday" correctly to Friday, a phi-4 reviewer replied
            # "VERDICT: correct / ANSWER: Saturday" (incoherent, and wrong),
            # and the headmaster changed its correct answer to Saturday. A
            # second opinion that can overwrite a right answer with a wrong
            # one is worse than no second opinion, so the label has to say
            # what the reply is worth. Same shape as the tool-call label
            # below: name what has NOT been established.
            model = (entry.get("model_name") or "another model")
            return (
                f"Worker '{role}' ({model}) offers a SECOND OPINION on your "
                f"answer:\n{reply}\n"
                f"This is an opinion from a different model, not evidence, and "
                f"it is not more reliable than your own reasoning. Re-derive "
                f"the answer yourself. Change your answer only if you can "
                f"point to the specific step where yours went wrong; if the "
                f"two disagree and you cannot find that step, keep yours and "
                f"say the two models disagree."
            )
        if _recites_own_steps(reply, entry):
            return (
                f"Worker '{role}' returned the PROCEDURE for this skill, not a "
                f"report of work done. Nothing has been run and nothing has "
                f"changed on disk:\n{reply}\n"
                f"If the user asked how to do it, relay these steps. If they "
                f"asked you to DO it, carry them out yourself with your own "
                f"tools and describe only what actually happened — do not say "
                f"you have handled it, or are handling it, on the strength of "
                f"this text."
            )
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

    def _evict(self, roles: list[str]):
        """Drop `roles` and hand their memory back to the system now.

        Dropping the dict entry is not the same as freeing the weights. MLX
        holds them in unified memory charged to the process until the
        allocator reclaims it, so an eviction that is only a `del` leaves the
        old worker fully resident right up to the moment the *next* one is
        loaded — which is exactly when the machine can least afford it. This
        is the same reason _reload_model frees before it loads.
        """
        if not roles:
            return
        for role in roles:
            self._resident.pop(role, None)
        training.release_model()

    def _evict_idle(self):
        idle_minutes = float(self._dispatch_cfg().get("worker_idle_unload_minutes", 10))
        if idle_minutes <= 0:
            return
        cutoff = time.time() - idle_minutes * 60
        self._evict([r for r, (_, _, ts) in self._resident.items() if ts < cutoff])

    def _evict_lru_if_needed(self):
        max_resident = max(1, int(self._dispatch_cfg().get("max_resident_workers", 1)))
        while len(self._resident) >= max_resident:
            oldest_role = min(self._resident, key=lambda r: self._resident[r][2])
            self._evict([oldest_role])

    def unload_all(self, reason: str = "") -> list[str]:
        """Evict every resident worker. Returns the roles that were dropped."""
        roles = list(self._resident)
        self._evict(roles)
        if roles and reason:
            self._status(f"  [Dispatch] Unloading worker(s) "
                         f"{', '.join(roles)}: {reason}")
        return roles

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
            # Every path out of get() marks the adapter used, including this
            # one. It is the path a busy worker takes — the second and every
            # later delegation in a row — and leaving it out meant the more a
            # worker was used, the staler its last-used stamp looked, which is
            # what archive_idle_items reads to decide it can be archived.
            training.mark_adapter_used(role=role)
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

    def _run_worker(self, model, tokenizer, system_prompt: str, user_text: str,
                    max_tokens: int) -> str:
        """Render one worker turn and return its reply, reasoning stripped.

        Both delegation paths need exactly this, and having it written twice is
        how they drifted: the browser loop was generating without stripping,
        which matters more there than anywhere else, because its reply is
        matched with startswith() against 'click:'/'type:'/'scroll'/'done'. A
        model that opens with a think block does not fail a check — it fails
        *every* check, and the loop reports an unrecognized action and stops.

        enable_thinking is False for the same reason in both: a worker's reply
        is an observation for the headmaster and a training sample for its own
        adapter, and deliberation is noise in both. The headmaster is where
        THINKING_ENABLED belongs. Measured on Qwen3.5-4B: a one-line summary
        came back as 156 words of "Thinking Process:" with this True, and 13
        correct words with it False — and not strippable either, because that
        model reasons in prose rather than <think>. The strip below is for the
        models that do use the tag.
        """
        prompt = tokenizer.apply_chat_template(
            [{"role": "system", "content": system_prompt},
             {"role": "user", "content": user_text}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        # temp=0.2 is near-greedy, a skill worker is a 4B fine-tuned on one
        # short procedure, and a delegated request is frequently outside what
        # it was tuned on — the exact conditions agent.py's repetition penalty
        # was written for. Without one, this path produced
        # "![](https://777.777.777.7777777..." as a worker's entire reply.
        return tooling.strip_reasoning_block(generate(
            model, tokenizer, prompt=prompt,
            sampler=make_sampler(temp=0.2, top_p=0.9),
            logits_processors=make_logits_processors(
                repetition_penalty=1.15, repetition_context_size=64),
            max_tokens=max_tokens, verbose=False,
        )).strip()

    def _worker_system_prompt(self, role: str, entry: dict[str, Any]) -> str:
        """Return the system prompt for a worker role.

        Builtin roles use ROLE_SYSTEM_PROMPTS; skill/worker catalog entries
        may carry their own system_prompt field.
        """
        # The builtin prompts are strict output grammars — 'browser' is
        # matched with startswith() against click:/type:/scroll/done — so they
        # are left byte-identical. They also never punt: they are handed the
        # page or the text they need. The suffix goes to the open-ended
        # skill/default prompts, which are the ones that ask questions back.
        if role in ROLE_SYSTEM_PROMPTS:
            return ROLE_SYSTEM_PROMPTS[role]
        if entry and "system_prompt" in entry:
            return entry["system_prompt"] + _NO_SECOND_TURN
        return "Complete the following task concisely and directly." + _NO_SECOND_TURN

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
        # Resolve the role before anything is unloaded for it. The headmaster
        # used to be put to sleep first and the catalog consulted second, so a
        # task delegated to a role that does not exist — a typo, or a skill
        # deleted since the model last saw the list — unloaded the 8B, found
        # nothing to run, and reloaded it. A full model reload as the price of
        # a misspelling, for a turn that returns an error string either way.
        if catalog_entry_for_role(role) is None:
            known = sorted({e.get("role") for e in load_catalog().values() if e.get("role")})
            return f"No worker configured for role '{role}'. Known roles: {', '.join(known) or 'none'}."

        # A worker receives `task` and nothing else — no page, no history, no
        # retrieved context. So a task that points at something ("summarise
        # what THAT REPO does") points at nothing the worker can see, and a
        # small model asked to summarise a repo it was never shown does not say
        # so; it invents one. Live 2026-08-26, with github.com/huyedits/Symbio
        # in the headmaster's context and nothing in the worker's:
        #
        #   delegate_task(role="summarize", task="summarise what that repo does")
        #   -> "The repository provides a Python library for parsing and
        #      manipulating JSON data ... supports YAML and XML ..."
        #
        # Every word of that is fabricated, and it came back as a tool
        # observation, which reads as fact. Bounce it instead, and tell the
        # headmaster what to do about it: paste the material in.
        # The browser role is the exception: it drives a live BrowserSession
        # and reads the page each round, so "check the page" resolves for it.
        # Every other role sees the task text and nothing else.
        browser_role = (role == "browser" and browser is not None)
        unresolved = None if browser_role else _unresolved_reference(task)
        if unresolved:
            return (
                f"Not delegated: the task says {unresolved!r}, but a worker "
                f"receives only the text of the task — it cannot see the page, "
                f"the conversation, or anything you are looking at. It would "
                f"have to invent whatever {unresolved!r} refers to. Re-send the "
                f"task with the actual material pasted into it, or answer it "
                f"yourself.")

        deep_sleep = bool(self._dispatch_cfg().get("headmaster_deep_sleep_while_workers", False))
        if deep_sleep and self.before_worker_fn is not None:
            self._status("  [Dispatch] Putting headmaster to sleep before loading worker...")
            self.before_worker_fn()

        try:
            # Inside the bracket, not before it. The browser loop used to
            # return above all of this, so with deep sleep on it loaded its
            # worker while the headmaster was still resident — the two-models-
            # at-once state the whole setting exists to prevent, reached by the
            # one role that holds its worker for several rounds. Nothing
            # refused it either: _second_headmaster_copy_refusal stands down
            # when deep sleep is configured, on the assumption that the sleep
            # actually happened.
            if role == "browser" and browser is not None:
                max_rounds = int(self._dispatch_cfg().get("max_worker_rounds", 4))
                return self._run_browser_delegation(task, browser, max_rounds)

            try:
                loaded = self.get(role)
            except SecondHeadmasterCopyRefused as exc:
                self._status(f"  [Dispatch] Refused to load '{role}': {exc}")
                return (f"Worker '{role}' was not run: {exc}")
            if loaded is None:
                known = sorted({e.get("role") for e in load_catalog().values() if e.get("role")})
                return f"No worker configured for role '{role}'. Known roles: {', '.join(known) or 'none'}."
            model, tokenizer, entry = loaded
            # get() already marked it; a second call here does the same write
            # twice per delegation for no reader.
            self._status(f"  [Dispatch] Delegating to '{role}': {task[:80]}{'...' if len(task) > 80 else ''}")
            system_prompt = self._worker_system_prompt(role, entry)
            try:
                reply = self._run_worker(
                    model, tokenizer, system_prompt, task, max_tokens)
            except Exception as e:
                self._status(f"  [Dispatch] Worker '{role}' failed: {e}")
                return f"Worker '{role}' failed: {e}"

            self._status(f"  [Dispatch] Worker '{role}' returned {len(reply.split())} word(s).")
            # Opt-in, and off by default: this writes the worker's own output
            # back into the corpus it will next be trained on, with nothing
            # having checked the output first. Observed: a worker whose corpus
            # demonstrated vegetarian-only suggestions answered a held-out
            # prompt with lobster, and that answer was appended as a training
            # sample — so the next retrain would have learned the violation as
            # correct. A model graded only by itself drifts toward whatever it
            # already does, which is the one failure a corpus cannot recover
            # from on its own.
            if reply and self._dispatch_cfg().get("capture_worker_samples", False):
                training.append_chat_pair(task, reply, tokenizer, system_prompt, role=role)
            return label_worker_reply(role, reply, entry)
        finally:
            if deep_sleep and self.after_worker_fn is not None:
                # The worker is still resident at this point, so waking the
                # headmaster on top of it puts both models in RAM at once —
                # the exact state deep sleep exists to prevent, just moved
                # from before the worker ran to after it finished. Deep sleep
                # is the operator saying this machine holds one model at a
                # time, so the worker goes before the headmaster comes back.
                # It costs the worker's warm weights on the next delegation;
                # the alternative cost is the machine.
                self.unload_all("headmaster is waking up")
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
            try:
                action = self._run_worker(
                    model, tokenizer, system_prompt, prompt_text, max_tokens=60)
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


# One in-flight run per worker. Saving the same skill twice — which a retry
# does, and which a user re-running /new-skill does — starts a second daemon
# thread for the same role, and both then back up, overwrite and restore the
# same adapter directory. Observed at 20 skills: two concurrent runs for
# 'set_up_tent_in_wind', each with its own backup, interleaving on one
# directory. TRAINER_LOCK serialises the trainer *subprocess* and does nothing
# about that, because the damage is in the surrounding file operations.
_ROLE_LOCKS: dict[str, threading.Lock] = {}
_ROLE_LOCKS_GUARD = threading.Lock()


def _role_lock(role: str) -> threading.Lock:
    with _ROLE_LOCKS_GUARD:
        return _ROLE_LOCKS.setdefault(role, threading.Lock())


def guarded_train_worker(role: str, config: dict[str, Any], iters: int | None = None,
                         resume: bool = False) -> tuple[bool, str]:
    """Train a worker's own adapter and golden-check it the same way
    ChatSession._guarded_train protects the headmaster's: baseline golden
    run, backup, train, reload, recheck, auto-rollback on regression.
    Returns (trained, status_message).

    Refuses rather than queues when this role is already training: the caller
    is a background thread whose work is already being done, and waiting would
    just hold a second copy of everything until the first finished.
    """
    lock = _role_lock(role)
    if not lock.acquire(blocking=False):
        return False, (
            f"Worker '{role}' is already being trained by another run; "
            f"skipping this one rather than training the same adapter twice.")
    try:
        return _guarded_train_worker(role, config, iters=iters, resume=resume)
    finally:
        lock.release()


def _guarded_train_worker(role: str, config: dict[str, Any], iters: int | None = None,
                          resume: bool = False) -> tuple[bool, str]:
    entry = catalog_entry_for_role(role)
    if entry is None:
        # Nothing can ever run this, so a record asking for it is not owed
        # work — it is a line that reappears at every start and that /resume
        # run can never clear, because this return happens before the task is
        # even opened. Deleting a skill leaves exactly that behind.
        dropped = pending.clear(kind="train_worker", role=role)
        return False, (
            f"No worker configured for role '{role}'."
            + (f" Dropped {dropped} carried-over task(s) for it." if dropped else ""))

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

    # The second battery. `cases` above grades how much of the procedure a
    # reply recalls, which is the right question for a worker whose corpus
    # still teaches recitation and the wrong one for a worker taught to
    # perform — see skill_perform's module docstring, and the measurement in
    # skill_eval.corpus_teaches_recitation that this exists to answer. Minted
    # from the worker's own verified worked examples with the values swapped,
    # and graded by running the reply, so recall cannot pass it and a
    # fabricated completion cannot either.
    perform_cases: list = []
    perform_steps = ""
    if dispatch_cfg.get("worker_perform_set_enabled", True):
        try:
            from symbio.app import skill_perform as _skill_perform

            perform_cases = _skill_perform.load_cases(role)
            if perform_cases and entry:
                from symbio.app import skill_eval as _skill_eval

                perform_steps = _skill_eval.skill_steps(entry) or ""
        except Exception:
            perform_cases = []

    def _run_golden(model, tokenizer):
        if not (golden_on and cases):
            return None
        # Graded the way a worker is actually served — see the same flag in
        # run_delegated_task. Grading with thinking on measures a mode the
        # worker never runs in, and worker training is gated on this result.
        return golden.run_golden_set(
            model, tokenizer, generate, sampler, system_prompt, config,
            enabled_groups=None, cases=cases, enable_thinking=False,
        )

    def _run_perform(model, tokenizer):
        if not perform_cases:
            return None
        from symbio.app import skill_perform as _skill_perform

        return _skill_perform.run_perform_set(
            model, tokenizer, generate, sampler, system_prompt, config,
            perform_cases, steps=perform_steps,
        )

    baseline = None
    baseline_perform = None
    # True when the "before" reading came from the base model because there was
    # no usable adapter yet. Kept so the result can say what it compared to.
    baseline_is_base = False
    backup_dir = None
    adapter_dir = constants.adapter_dir_for(role)
    # Recorded before anything expensive starts. Auto-train runs this whole
    # function on a daemon thread, so an out-of-memory kill takes it with no
    # unwind: without a note on disk the next start has no way to know the
    # skill was seeded and never trained, or that the backup below is the only
    # intact copy of the adapter.
    task_id = pending.open_task(
        "train_worker", f"training for worker '{role}'", role=role)
    # A baseline is only meaningful against weights this model can actually
    # load. An adapter from a different base is discarded silently, so the
    # "before" score would be the base model wearing the adapter's name — and
    # every later comparison would be against that fiction.
    has_prior = (adapter_dir.exists()
                 and (adapter_dir / "adapter_config.json").exists()
                 and adapter_matches_model(adapter_dir, entry["model_name"]))
    # Nothing to measure means nothing to load. Previously this block ran on
    # the strength of an adapter existing, then handed the weights to checkers
    # that immediately returned None -- a full model load for no reading.
    needs_baseline = bool((golden_on and cases) or perform_cases)

    if needs_baseline:
        # This load is in-process and was unguarded: run_training's preflight
        # covers the trainer child, but the baseline copy loaded here lands
        # first and on top of whatever the session already holds. Auto-train
        # runs on a background thread while the headmaster is still resident,
        # so this is the load that goes over the edge.
        shortfall = training.load_memory_shortfall(
            config, entry["model_name"],
            purpose=f"golden-check worker '{role}' before training")
        if shortfall:
            # Refused, not forgotten: the run is owed and stays on the books
            # until it is resumed or dropped on purpose.
            pending.update(task_id, state=pending.DEFERRED, pid=None,
                           reason="not enough free memory to run safely")
            return False, (
                f"{shortfall} Free memory first (unload the headmaster, or "
                f"wait for the workers to idle out) and retrain '{role}' — "
                f"it is on the deferred list until then.")
        # With no usable adapter the baseline is the BASE model, and the run is
        # still checked. It used to be skipped entirely: no adapter meant no
        # baseline, which meant the early return below fired and a brand-new
        # skill's FIRST adapter shipped without either battery ever running.
        # That is the one adapter most likely to be bad -- it is trained on
        # nothing but seeds -- and the first training is exactly when a skill
        # is silently established as broken. Measured on the trial that built
        # this battery: two first trainings in a row reported only "Worker
        # trained." while the worker could not perform the skill at all.
        #
        # Comparing against the base model is also the right question to ask
        # of a first fine-tune: did the weights add anything, or did they make
        # a working base model worse.
        baseline_is_base = not has_prior
        base_model, base_tok = load(
            entry["model_name"],
            **({"adapter_path": str(adapter_dir)} if has_prior else {}))
        baseline = _run_golden(base_model, base_tok)
        # Same weights, same load. A perform baseline taken against anything
        # else would compare the new adapter to a different model.
        baseline_perform = _run_perform(base_model, base_tok)
        # The baseline scores are all we need from these weights, and the
        # trainer below loads its own full copy. Holding this one until the
        # function returns would mean two copies resident across the whole
        # run — on a unified-memory Mac that is the difference between a slow
        # fine-tune and an out-of-memory kill.
        base_model, base_tok = None, None
        training.release_model()

    if has_prior:
        backup_dir = training.backup_adapter(role=role)
        # From here until discard_adapter_backup, the previous adapter exists
        # only in that directory. If this process dies in between, recovery
        # needs to be told where to find it.
        pending.update(task_id, backup_dir=str(backup_dir) if backup_dir else None)

    # Whether the run got far enough that nothing is still owed. Only a run
    # that produced an adapter counts: a refusal reported into a log nobody is
    # reading — which is every auto-train, since it runs on a background
    # thread — leaves a skill seeded and permanently untrained, and clearing
    # its record is what makes that state invisible.
    settled = False
    try:
        trained = training.run_training(
            config, iters=iters, role=role, model_name=entry["model_name"],
            resume=resume)
        if not trained:
            return False, (
                f"Training for '{role}' produced no usable adapter. Check the "
                f"log above: either there was no new data, or the run ended "
                f"before any checkpoint was written."
            )
        settled = True

        # The trainer child has exited by now, but its memory only comes back
        # once the allocator reclaims it, and this is the load that would
        # otherwise race it. release_model() forces our own collection; the
        # settle waits out the kernel's, which is the half that panics — a
        # child exit unmaps every Metal buffer in one bulk teardown, and the
        # load below is a multi-gigabyte allocation landing straight into it.
        # Reading free memory *after* the wait also makes the shortfall check
        # below judge the machine as it will be, not as it is mid-reclaim.
        training.release_model()
        training.settle_after_trainer_exit(config)
        shortfall = training.load_memory_shortfall(
            config, entry["model_name"],
            purpose=f"reload worker '{role}' to check the new adapter")
        if shortfall:
            # The adapter is already on disk and cannot be verified. Reverting
            # to a backup loses one training run; shipping an unchecked
            # adapter is what the golden set exists to prevent.
            if backup_dir:
                training.restore_adapter(backup_dir, role=role)
                return False, (
                    f"{shortfall} Worker '{role}' trained but could not be "
                    f"golden-checked, so it was rolled back to the previous "
                    f"adapter. Retrain when there is more memory free.")
            return True, (
                f"{shortfall} Worker '{role}' trained and kept unverified "
                f"(no previous adapter to roll back to).")

        new_model, new_tok = load(entry["model_name"], adapter_path=str(adapter_dir))
        training.mark_adapter_used(role=role)

        if baseline is None and baseline_perform is None:
            return True, f"Worker '{role}' trained."

        after = _run_golden(new_model, new_tok) if baseline is not None else None
        regressions = sorted(baseline.passing - after.passing) if (
            baseline is not None and after) else []
        threshold = int(dispatch_cfg.get("worker_golden_regression_threshold", 0))

        if (len(regressions) > threshold
                and dispatch_cfg.get("worker_golden_retry_enabled", True)):
            print(
                f"  [Golden] Double-checking {len(regressions)} regression(s) "
                f"for worker '{role}'..."
            )
            recheck, consistent = golden.run_golden_set_retry(
                new_model, new_tok, generate, sampler, system_prompt, config,
                enabled_groups=None, cases=cases, enable_thinking=False,
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
                    # Second trainer child, second bulk teardown, and the same
                    # reload waiting on the other side of it. This is the pair
                    # that makes one guarded_train_worker call five model loads
                    # in a row — the densest load churn anywhere in the app,
                    # and it runs unattended on a background thread.
                    training.release_model()
                    training.settle_after_trainer_exit(config)
                    remedy_shortfall = training.load_memory_shortfall(
                        config, entry["model_name"],
                        purpose=f"reload worker '{role}' after remedy training")
                    if trained2 and remedy_shortfall:
                        # Leave `after` and `regressions` as the pre-remedy
                        # result: unchecked is treated as failed, so the
                        # rollback below still fires.
                        print(f"  [Train] {remedy_shortfall}")
                    elif trained2:
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

        # ---- The perform battery ------------------------------------------
        # Runs last, against whatever adapter is current by now -- including
        # one the recall remedy above just retrained and reloaded.
        #
        # The loop is: run the task with the adapter, and if it cannot do it,
        # train it more and ask again. It fires on FAILURE, not on regression.
        # Regression was the wrong trigger and made the retry unreachable in
        # the case that needs it most: on a first training the baseline is the
        # base model, which fails these checks too, so nothing "regresses" and
        # a worker that simply cannot do the job was never retried at all.
        # Measured -- two first trainings in a row scored 0/4 and went straight
        # to being kept.
        after_perform = (
            _run_perform(new_model, new_tok) if new_model is not None else None)
        perform_regressions: list[str] = []

        if after_perform is not None and after_perform.total:
            failing = sorted(cid for cid, ok in after_perform.results.items() if not ok)
            for case_id in failing:
                print(f"  [Perform] {case_id}: "
                      f"{after_perform.reasons.get(case_id, 'failed')}")

            max_rounds = int(dispatch_cfg.get("worker_perform_retry_max_rounds", 2))
            retry_on = dispatch_cfg.get("worker_perform_retry_enabled", True)
            rounds = 0
            while failing and retry_on and rounds < max_rounds:
                rounds += 1
                from symbio.app import skill_perform as _skill_perform

                added = _skill_perform.remedy_samples(
                    perform_cases, failing, new_tok, system_prompt, role,
                    copies=int(dispatch_cfg.get(
                        "worker_perform_retry_samples_per_case", 2)),
                    # The cases it still passes go back in as ballast. A remedy
                    # built only from failures makes them the bulk of the delta,
                    # and on a corpus this small whichever behaviour is repeated
                    # most just wins — which moves the failure instead of
                    # removing it.
                    passing=after_perform.passing,
                    passing_copies=int(dispatch_cfg.get(
                        "worker_perform_retry_passing_copies", 1)))
                if not added:
                    print("  [Perform] No remedy samples could be generated "
                          "(no case carried its source example).")
                    break

                extra = int(dispatch_cfg.get(
                    "worker_perform_retry_max_extra_iters", 50))
                print(f"  [Perform] Round {rounds}/{max_rounds}: "
                      f"{len(failing)} check(s) failing; injected {added} "
                      f"demonstration sample(s), retraining for {extra} iters.")
                # Same teardown discipline as the recall remedy above: the
                # trainer loads its own full copy of the weights, so ours goes
                # first. Everything still needed from it is on disk.
                new_model, new_tok = None, None
                training.release_model()
                retrained = training.run_training(
                    config, iters=extra, role=role,
                    model_name=entry["model_name"])
                training.release_model()
                training.settle_after_trainer_exit(config)
                shortfall = training.load_memory_shortfall(
                    config, entry["model_name"],
                    purpose=f"reload worker '{role}' after perform remedy")
                if not retrained or shortfall:
                    # Unchecked is treated as failed, so the decision below
                    # still fires on the pre-remedy result.
                    if shortfall:
                        print(f"  [Perform] {shortfall}")
                    break
                new_model, new_tok = load(
                    entry["model_name"], adapter_path=str(adapter_dir))
                after_perform = _run_perform(new_model, new_tok)
                still = sorted(
                    cid for cid, ok in after_perform.results.items() if not ok)
                if not still:
                    print(f"  [Perform] All {after_perform.total} check(s) pass "
                          f"after round {rounds}.")
                    failing = still
                    break
                if len(still) >= len(failing):
                    # No fewer failures than before the round. More of the same
                    # training is not going to find it, and each round costs a
                    # full fine-tune plus two model loads on a background
                    # thread. Stop and report rather than spend the budget
                    # proving it twice.
                    print(f"  [Perform] Round {rounds} did not reduce the "
                          f"failures ({len(still)}); stopping retries. The "
                          f"corpus, not the iteration count, is the limit.")
                    failing = still
                    break
                failing = still

            if baseline_perform is not None:
                perform_regressions = sorted(
                    baseline_perform.passing - after_perform.passing)

        # "Regressed" against a previous adapter; "is worse than the base
        # model" when this was a first training. They are different claims and
        # reporting the first for the second would be wrong.
        against = "the base model" if baseline_is_base else "the previous adapter"
        if perform_regressions and dispatch_cfg.get(
                "worker_perform_rollback_on_regression", True):
            # Blocks where a derived recall regression only reports. A derived
            # case cannot tell specialisation from damage -- that is why it is
            # allowed to be advisory. This one can: its pass condition is that
            # the reply ran, on values that appear nowhere in the corpus, so no
            # amount of learning to perform can cause it to fail.
            if backup_dir:
                training.restore_adapter(backup_dir, role=role)
                return True, (
                    f"Worker '{role}' trained but lost "
                    f"{len(perform_regressions)} performance check(s) "
                    f"({', '.join(perform_regressions)}); rolled back.")
            return True, (
                f"Worker '{role}' trained and kept — it performs worse than "
                f"{against} on {len(perform_regressions)} check(s) "
                f"({', '.join(perform_regressions)}), and there is no previous "
                f"adapter to roll back to. The corpus is what needs fixing.")

        if len(regressions) > threshold:
            # Derived cases grade a reply on how much of the steps text it
            # reproduces. That is the right target while the corpus is still
            # the seeded steps-text samples -- a worker answering "I don't
            # know." there is genuinely broken and must not ship. Once real
            # demonstrations replace the seeds the worker is meant to stop
            # reciting, and the same cases would revert the specialisation.
            #
            # A passing perform battery settles it outright. That battery
            # answers the question the derived cases were standing in for, on
            # evidence they do not have -- the worker just performed the skill
            # on values that appear nowhere in its corpus -- so a recall
            # regression alongside it is the specialisation being measured, not
            # damage, and reverting for it would throw away the retrain that
            # produced the better worker.
            derived_only = (
                skill_cases
                and not _skill_eval.has_custom_tasks(role)
                and (bool(after_perform and after_perform.pass_count)
                     or not _skill_eval.corpus_teaches_recitation(
                         role, _skill_eval.skill_steps(entry) if entry else "")))
            rollback_on = dispatch_cfg.get(
                "worker_golden_rollback_on_regression", True)
            if backup_dir and rollback_on and not derived_only:
                training.restore_adapter(backup_dir, role=role)
                return True, (
                    f"Worker '{role}' trained but regressed on {len(regressions)} "
                    f"check(s) ({', '.join(regressions)}); rolled back.")
            if derived_only and rollback_on:
                return True, (
                    f"Worker '{role}' trained; {len(regressions)} derived "
                    f"check(s) ({', '.join(regressions)}) no longer recite the "
                    f"steps text. Kept — derived checks report only. Add "
                    f"{_skill_eval.tasks_path_for(role).name} to make them block.")
            return True, (
                f"Worker '{role}' trained but regressed on {len(regressions)} "
                f"check(s) ({', '.join(regressions)}); kept anyway.")
        scores = []
        if after is not None:
            scores.append(f"{after.pass_count}/{after.total} recall")
        if after_perform is not None:
            scores.append(f"{after_perform.pass_count}/{after_perform.total} performance")
        if scores and baseline_is_base:
            # First training: there was no previous adapter, so the reading
            # above is the absolute score and the comparison was against the
            # base model. Both are worth saying -- an adapter that merely ties
            # the base model has not earned its place.
            scores[-1] += " (first adapter, measured against the base model)"
        return True, (
            f"Worker '{role}' trained ({', '.join(scores)} checks passing)."
            if scores else f"Worker '{role}' trained.")
    finally:
        # An adapter was produced, so the work is done however the checks then
        # graded it — a rollback is a decision about the result, not an
        # unfinished run. Anything else stays owed and visible at the next
        # start, which is the only place a background failure gets seen.
        if settled:
            pending.finish(task_id)
        else:
            pending.update(
                task_id, state=pending.DEFERRED, pid=None,
                reason="training did not produce an adapter (see the log; "
                       "most often not enough free memory at the time)")
        training.discard_adapter_backup(backup_dir)
        # Whatever this run loaded goes back to the system now rather than
        # whenever the collector next runs. Callers include a background
        # thread finishing beside a live chat session, where the difference is
        # a few gigabytes held for no reason.
        base_model, base_tok = None, None
        new_model, new_tok = None, None
        training.release_model()
