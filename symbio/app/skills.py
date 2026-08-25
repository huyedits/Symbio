"""Skill adapters: every saved skill gets its own worker LoRA adapter.

A skill is a markdown note under notes/ with a '# Skill: <name>' heading.
When a skill is saved we also create a worker role for it, store the role
in worker_models.json, and train a dedicated LoRA adapter under
adapters/workers/<slug>/ so the headmaster can later delegate to it.

Unused skills and adapters are archived after a configurable idle threshold.
"""

from __future__ import annotations

import json
import re
import shutil
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from symbio import constants
from symbio.app import dispatch, memory, pending, training


NOTES_USAGE_FILE = constants.NOTES_DIR / ".last_used.json"
ADAPTER_ARCHIVE_DIR = constants.ADAPTER_ARCHIVE_DIR
_SKILL_FLAG = {"is_skill": True}


def _skill_health_path(note_path: Path) -> Path:
    """Sidecar file that holds health errors and corrections for a skill note."""
    return note_path.with_suffix(note_path.suffix + ".health.jsonl")


def _is_skill_note(path: Path) -> bool:
    """True if the markdown file begins with '# Skill:'."""
    try:
        first = path.read_text(encoding="utf-8").splitlines()[0].strip().lower()
    except (OSError, IndexError):
        return False
    return first.startswith("# skill:")


def _append_health_entry(note_path: Path, entry_type: str, text: str):
    """Append a dated error/correction entry to a skill's sidecar file."""
    if not _is_skill_note(note_path):
        return
    sidecar = _skill_health_path(note_path)
    entry = {
        "t": datetime.now().isoformat(),
        "type": entry_type,
        "text": text,
    }
    with open(sidecar, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def record_skill_error(note_path: Path, error: str):
    """Record a health error against the skill note at note_path."""
    _append_health_entry(note_path, "error", error)


def record_skill_correction(note_path: Path, correction: str):
    """Record a user correction against the skill note at note_path."""
    _append_health_entry(note_path, "correction", correction)


def read_skill_health(note_path: Path) -> list[dict[str, Any]]:
    """Return all recorded health/correction entries for a skill note."""
    sidecar = _skill_health_path(note_path)
    if not sidecar.exists():
        return []
    entries = []
    with open(sidecar, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries


def _skill_slug(name: str) -> str:
    """Stable, filesystem-safe identifier from a skill title."""
    s = name.lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = s.strip("_")
    return s or "skill"


def _skill_prompt_opener(name: str) -> str:
    """The one sentence shared by both skill prompt forms.

    Training and evaluation must agree on this framing, so it lives in one
    place rather than being written out twice.
    """
    return f"You are the specialist worker for the skill '{name}'."


def build_worker_system_prompt(name: str) -> str:
    """The prompt a skill worker is TRAINED under: names the skill, withholds
    the procedure.

    The steps are deliberately absent. If they were present here, the adapter
    would only ever learn to copy a procedure out of its own context, and no
    later evaluation could tell weight-learning apart from prompt-following.
    Keeping them out is what makes 'the skill lives in the weights' a claim
    that can be tested — see symbio.app.skill_eval.
    """
    return (
        f"{_skill_prompt_opener(name)} "
        "Follow the steps for that skill exactly, produce only the requested "
        "output, and do not add extra commentary."
    )


def _build_skill_system_prompt(name: str, steps: str) -> str:
    """The prompt a skill worker is SERVED under: includes the steps.

    Production keeps its safety net — a weak adapter still behaves because
    the procedure is right there. Serving with more context than training is
    harmless; the reverse would not be.
    """
    return (
        f"{_skill_prompt_opener(name)} "
        "Follow the steps below exactly, produce only the requested output, "
        "and do not add extra commentary.\n\n"
        f"Steps:\n{steps}\n\n"
        "Reply with the result of applying these steps to the user's request."
    )


def skill_note_body(name: str, steps: str) -> str:
    """The steps plus a Triggers block, which is what makes the note findable.

    Retrieval is term-frequency over the note body, so a tight four-line
    procedure loses to any long note that happens to repeat a common word.
    Measured 2026-08-24: for the query "scrape a listing page", the
    Browser Control note (152 words, "page" many times over) outscored the
    Scrape A Listing Page skill itself (57 words, "page" once) — 1.153 to
    1.088 — so the agent opened a browser instead of running the runbook it
    had just been trained on. A skill that cannot be retrieved is a skill the
    agent does not have.

    The triggers are derived from the procedure rather than written by hand:
    its own distinctive vocabulary is exactly what should route to it, and
    deriving them means every skill gets them instead of only the ones
    somebody remembered to annotate.
    """
    from symbio.app import skill_eval

    keywords = skill_eval._keywords(skill_eval._step_body(steps))
    name_terms = [w.lower() for w in re.findall(r"[A-Za-z][A-Za-z0-9_-]+", name)]
    # Name terms first: they are what a user actually types.
    seen, terms = set(), []
    for t in name_terms + keywords:
        if t not in seen and len(t) > 2:
            seen.add(t)
            terms.append(t)

    examples = [task.prompt for task in skill_eval.default_tasks(name)]
    return (
        f"{steps}\n\n"
        f"## Triggers\n\n"
        f"Keywords: {', '.join(terms)}\n\n"
        f"Examples:\n\n"
        + "\n".join(f"- {e}" for e in examples)
        + "\n"
    )


def _load_worker_catalog() -> dict[str, Any]:
    if not constants.WORKER_MODELS_FILE.exists():
        return {}
    try:
        return json.loads(constants.WORKER_MODELS_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _save_worker_catalog(catalog: dict[str, Any]):
    constants.WORKER_MODELS_FILE.write_text(
        json.dumps(catalog, indent=2) + "\n", encoding="utf-8"
    )


def _ensure_skill_catalog_entry(
    name: str, config: dict[str, Any], system_prompt: str
) -> str:
    """Add or update a worker catalog entry for this skill. Returns the role slug."""
    role = _skill_slug(name)
    catalog = _load_worker_catalog()

    # Remove any existing entry with the same role to keep catalog clean.
    for key, entry in list(catalog.items()):
        if entry.get("role") == role:
            del catalog[key]

    catalog[f"skill_{role}"] = {
        "model_name": worker_model_name(config),
        "role": role,
        "description": f"Skill: {name}",
        "adapter_compatible": True,
        "memory_note": "worker-size RAM at runtime, alongside the headmaster",
        "system_prompt": system_prompt,
        "is_skill": True,
        "skill_name": name,
    }
    _save_worker_catalog(catalog)
    return role


def _seed_user_turns(name: str) -> list[str]:
    """Ways a user might ask for this skill, for the seed samples.

    Kept deliberately distinct from skill_eval.default_tasks: if the seeds
    and the eval prompts were the same strings, a passing score would only
    show memorisation of the training set.
    """
    lower = name.lower()
    return [
        f"Apply the skill '{name}'.",
        f"How do I perform '{name}'?",
        f"What are the steps for {lower}?",
        f"Can you take care of {lower}?",
        f"Use your {lower} skill on this.",
        f"Time for {lower}.",
    ]


# Two seed kinds were tried here and reverted, because they were measured and
# they lost. Adding per-step questions ("what comes after X?") plus contrast
# samples pairing off-topic asks with a decline did stop the adapter reciting
# the procedure for "how do I reset my bluetooth headphones?" — and cost far
# more than it bought. On the same held-out battery, recall coverage fell from
# ~98% to ~38%: legitimate requests ("keys expired, what do I run first?") were
# declined, single steps came back in place of the procedure, and the decline
# string itself became an attractor strong enough to degrade unrelated
# generation into "That that that".
#
# The cause is structural, not a bad word list. A specialist worker only ever
# sees "fix my X" shaped prompts, so "my printer is offline" and "my keys
# expired" are the same shape to it, and 20-odd synthetic samples at several
# epochs cannot draw that boundary — whichever behaviour is repeated most just
# wins. Over-triggering inside a worker is also the cheaper failure: the
# headmaster decides what reaches it, so a false decline loses the skill
# outright while a false recite is merely noise.
#
# Worth revisiting with real usage samples rather than synthetic contrast, or
# with fewer iterations over a tiny corpus. Not worth shipping as it stood.


def _seed_skill_training_data(
    role: str, name: str, steps: str, tokenizer: Any
) -> int:
    """Write synthetic training samples for a brand-new skill worker.

    The system turn names the skill but withholds the procedure; the steps
    appear only in the assistant turn. That is what pushes the procedure into
    the weights instead of teaching the adapter to copy it out of context.

    Known limits, measured rather than assumed: with these seeds the procedure
    is recalled reliably for held-out phrasings, but its steps transpose on
    unusually worded requests (~80% step order), and an unrelated
    troubleshooting question can draw the whole procedure. See the note above
    for what was tried against that and why it was reverted.

    Returns the number of samples written. Real usage samples accumulate
    automatically in dispatch.WorkerPool.run_delegated_task.
    """
    data_dir = constants.data_dir_for(role)
    data_dir.mkdir(parents=True, exist_ok=True)
    train_file = data_dir / "train.jsonl"

    # Clear any stale auto-seeded samples so re-saving a skill refreshes the seed.
    if train_file.exists():
        lines = [
            ln for ln in train_file.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not json.loads(ln).get("metadata", {}).get("skill_seed")
        ]
    else:
        lines = []

    system_prompt = build_worker_system_prompt(name)
    # Chat templates are model-specific: ChatML turn markers mean nothing to a
    # Mistral tokenizer and vice versa. Record which model's template rendered
    # these samples so training can refuse a mismatched pairing instead of
    # burning an hour learning noise. See seed_model_mismatch().
    tokenized_for = getattr(tokenizer, "name_or_path", None) or ""

    samples = [(turn, steps, "recall") for turn in _seed_user_turns(name)]

    written = 0
    for user_turn, answer, kind in samples:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_turn},
            {"role": "assistant", "content": answer},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False, enable_thinking=False
        )
        lines.append(json.dumps({
            "text": text,
            # Carried alongside the rendered text so mlx_lm selects ChatDataset
            # and can mask the prompt out of the loss. Without it these samples
            # depend on training.upgrade_corpus_to_messages recovering the
            # structure by re-parsing, which only works for templates it knows.
            "messages": messages,
            "metadata": {
                "skill_seed": True,
                "skill": name,
                "seed_kind": kind,
                "tokenized_for": tokenized_for,
            },
        }))
        written += 1

    train_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return written


def _model_stem(model_name: str) -> str:
    """Bare model id, ignoring the publishing org.

    'Qwen/Qwen3-8B-MLX-4bit' and 'mlx-community/Qwen3-8B-MLX-4bit' are the
    same weights republished; comparing full paths would flag them as a
    mismatch and block a training run that is actually fine.
    """
    return model_name.rsplit("/", 1)[-1].strip().lower()


def delegatable_role_for_note(note_path: Path, config: dict[str, Any]) -> str | None:
    """The worker role a retrieved skill note can be handed to, if any.

    Retrieval already does the matching work: it scores the user's message
    against every note, and skills *are* notes. When "my wifi is broken" pulls
    up the 'Skill: Fix wifi' note, that hit is a routing signal — it was simply
    never consulted for dispatch, so the model had to rediscover from the tool
    schema alone that a matching specialist existed.

    Returns None unless the skill has a trained adapter that belongs to the
    model that would load it: suggesting a worker whose weights are absent or
    were built for another model would send the turn somewhere worse than
    answering directly.
    """
    if not config.get("dispatch", {}).get("enabled", False):
        return None
    if not _is_skill_note(note_path):
        return None
    try:
        heading = note_path.read_text(encoding="utf-8").splitlines()[0]
    except (OSError, IndexError):
        return None
    name = heading.split(":", 1)[1].strip() if ":" in heading else ""
    if not name:
        return None
    role = _skill_slug(name)

    from symbio.app import dispatch

    entry = dispatch.catalog_entry_for_role(role)
    if entry is None:
        return None
    adapter_dir = constants.adapter_dir_for(role)
    if not (adapter_dir / "adapters.safetensors").exists():
        return None
    if not dispatch.adapter_matches_model(adapter_dir, entry["model_name"]):
        return None
    return role


def worker_model_name(config: dict[str, Any]) -> str:
    """Which model a skill worker should run.

    Defaults to `dispatch.worker_model_name` so a worker is not a second copy
    of the headmaster's weights — a skill answers one narrow question under a
    short prompt, and does not need the size the general agent does. Falls back
    to the headmaster's own model when unset.
    """
    configured = config.get("dispatch", {}).get("worker_model_name")
    return configured or config["model_name"]


def worker_tokenizer(model_name: str, fallback: Any) -> Any:
    """The tokenizer of the model this worker will actually be trained on.

    Seeds are stamped with the tokenizer that rendered them and
    seed_model_mismatch refuses to train when that disagrees with the worker's
    model. Once workers stopped sharing the headmaster's model, seeding with
    the headmaster's tokenizer would stamp every new skill with the wrong name
    and block its first training run. Falls back to the caller's tokenizer if
    the worker's cannot be loaded — a mismatch is then reported honestly rather
    than being silently papered over.
    """
    current = getattr(fallback, "name_or_path", "") or ""
    if current and _model_stem(current) == _model_stem(model_name):
        return fallback
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(model_name)
    except Exception:
        return fallback


def seed_model_mismatch(role: str, model_name: str) -> str | None:
    """Explain why a role's seed data cannot train `model_name`, or None.

    The seeds are rendered by the headmaster's tokenizer at skill-creation
    time, but training uses the model recorded in the worker catalog. When a
    skill outlives a model switch those two drift apart, and the mismatch is
    invisible: training runs happily to completion on samples whose turn
    markers the model has never seen.
    """
    train_file = constants.data_dir_for(role) / "train.jsonl"
    if not train_file.exists():
        return None
    for raw in train_file.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        try:
            meta = json.loads(raw).get("metadata", {})
        except json.JSONDecodeError:
            continue
        stamped = meta.get("tokenized_for")
        if not stamped:
            continue
        if _model_stem(stamped) != _model_stem(model_name):
            return (
                f"Training data for role '{role}' was tokenized for "
                f"'{stamped}' but the worker is configured to train "
                f"'{model_name}'. Their chat templates differ, so this run "
                f"would learn nothing useful. Re-save the skill to re-seed "
                f"it for the current model."
            )
    return None


def save_skill_adapter(
    name: str,
    steps: str,
    config: dict[str, Any],
    tokenizer: Any,
    auto_train: bool = True,
) -> dict[str, Any]:
    """Save a skill note and create a dedicated worker adapter for it.

    Returns a dict with note_path, role, adapter_dir, and training status.
    """
    note_path = memory.save_skill(name, steps)
    system_prompt = _build_skill_system_prompt(name, steps)
    role = _ensure_skill_catalog_entry(name, config, system_prompt)

    # Refresh dispatch's in-memory view of the catalog.
    # (load_catalog is lazy, so stale disk state is harmless on next call.)

    # Make the new role visible to delegate_task in this session rather than
    # at the next restart — a skill the model is not told about is one it
    # cannot route to.
    try:
        from symbio.app import tooling

        tooling.refresh_delegate_roles()
    except Exception:
        pass

    # Seed with the tokenizer of the model the worker will be trained on, not
    # the headmaster's: the samples are stamped with it, and training refuses
    # to run when that stamp disagrees with the worker's model.
    worker_model = worker_model_name(config)
    seed_tokenizer = tokenizer
    if worker_model != config.get("model_name"):
        seed_tokenizer = worker_tokenizer(worker_model, tokenizer)

    seeded = _seed_skill_training_data(role, name, steps, seed_tokenizer)
    adapter_dir = constants.adapter_dir_for(role)

    result = {
        "note_path": str(note_path),
        "role": role,
        "adapter_dir": str(adapter_dir),
        "seeded_samples": seeded,
        "trained": False,
        "message": f"Skill '{name}' saved as worker role '{role}' with {seeded} seed samples.",
    }

    # The seeds are on disk and the adapter is not: from this line until a
    # training run finishes, the skill is real but cannot answer from its own
    # weights. Written down before the thread starts and before the crash
    # window opens — guarded_train_worker supersedes this entry when it picks
    # the work up, and clears it when it finishes, so a skill can never end up
    # permanently seeded-but-untrained without something saying so.
    pending.defer("train_worker", f"first adapter for skill '{name}'",
                  role=role, reason="skill saved; adapter not trained yet")

    if auto_train:
        # Training the headmaster-sized model blocks for minutes; run in the
        # background so the chat front-end stays responsive.
        def _train():
            trained, msg = dispatch.guarded_train_worker(role, config, iters=None)
            result["trained"] = trained
            result["training_message"] = msg

        threading.Thread(target=_train, daemon=True, name=f"train-skill-{role}").start()
        result["message"] += " Adapter training started in the background."
    else:
        result["message"] += " Run /train_worker {} when ready to train.".format(role)

    return result


def list_skill_adapters() -> list[dict[str, Any]]:
    """Return metadata for every active skill adapter."""
    out = []
    catalog = _load_worker_catalog()
    for entry in catalog.values():
        if not entry.get("is_skill"):
            continue
        role = entry["role"]
        adapter_dir = constants.adapter_dir_for(role)
        exists = (adapter_dir / "adapter_config.json").exists()
        last_used = training.adapter_last_used(role=role)
        out.append({
            "role": role,
            "name": entry.get("skill_name", role),
            "description": entry.get("description", ""),
            "adapter_exists": exists,
            "adapter_dir": str(adapter_dir),
            "last_used": last_used.isoformat() if last_used else None,
        })
    return out


def delete_skill_adapter(role: str) -> dict[str, Any]:
    """Remove a skill's worker catalog entry, adapter weights, training data,
    and any health sidecar tied to its note."""
    catalog = _load_worker_catalog()
    removed_keys = [k for k, e in catalog.items() if e.get("role") == role and e.get("is_skill")]
    for k in removed_keys:
        del catalog[k]
    _save_worker_catalog(catalog)

    # Remove the skill note and its health sidecar if they exist.
    note_path = None
    for title, p in memory.list_skills():
        if _skill_slug(title[7:].strip()) == role:
            note_path = p
            break
    if note_path and note_path.exists():
        note_path.unlink()
        sidecar = _skill_health_path(note_path)
        if sidecar.exists():
            sidecar.unlink()

    adapter_dir = constants.adapter_dir_for(role)
    data_dir = constants.data_dir_for(role)
    if adapter_dir.exists():
        shutil.rmtree(adapter_dir)
    if data_dir.exists():
        shutil.rmtree(data_dir)
    return {"role": role, "removed_entries": removed_keys}


# ---- Usage tracking and archival ----


def record_note_usage(path: Path):
    """Update the last-accessed timestamp for a markdown note."""
    manifest = _load_note_usage_manifest()
    manifest[str(path.resolve())] = datetime.now().isoformat()
    _save_note_usage_manifest(manifest)


def _load_note_usage_manifest() -> dict[str, str]:
    if not NOTES_USAGE_FILE.exists():
        return {}
    try:
        return json.loads(NOTES_USAGE_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _save_note_usage_manifest(manifest: dict[str, str]):
    NOTES_USAGE_FILE.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def _note_mtime(path: Path) -> datetime:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime)
    except OSError:
        return datetime.min


def _is_protected_note(path: Path) -> bool:
    """Identity and preference notes should never be auto-archived."""
    try:
        first = path.read_text(encoding="utf-8").splitlines()[0].lower()
    except (OSError, IndexError):
        return False
    protected = {
        "# my identity",
        "# user identity",
        "# user preference",
        "# assistant identity",
    }
    return any(first.startswith(p) for p in protected)


def archive_idle_notes(config: dict[str, Any], dry_run: bool = False) -> list[str]:
    """Move markdown notes that haven't been used recently to notes/archive/.

    Returns the list of archived filenames. In dry-run mode the candidates are
    returned but nothing is moved.
    """
    days = int(config.get("archive", {}).get("note_idle_days", 90))
    if days <= 0:
        return []
    cutoff = datetime.now() - timedelta(days=days)
    manifest = _load_note_usage_manifest()
    archived: list[str] = []

    for f in sorted(constants.NOTES_DIR.glob("*.md")):
        if not f.is_file() or _is_protected_note(f):
            continue
        # Use explicit last-used if available, else file mtime.
        last_used_str = manifest.get(str(f.resolve()))
        if last_used_str:
            try:
                last_used = datetime.fromisoformat(last_used_str)
            except ValueError:
                last_used = _note_mtime(f)
        else:
            last_used = _note_mtime(f)
        if last_used > cutoff:
            continue
        archived.append(f.name)
        if dry_run:
            continue
        dest = constants.NOTES_ARCHIVE_DIR / f.name
        counter = 1
        while dest.exists():
            dest = constants.NOTES_ARCHIVE_DIR / f"{f.stem}_{counter}{f.suffix}"
            counter += 1
        f.rename(dest)
        # Move any health sidecar with the note.
        sidecar = _skill_health_path(f)
        if sidecar.exists():
            sidecar_dest = _skill_health_path(dest)
            sidecar.rename(sidecar_dest)
        # Drop from manifest so a restored note starts fresh.
        manifest.pop(str(f.resolve()), None)

    if archived and not dry_run:
        _save_note_usage_manifest(manifest)
    return archived


def archive_idle_adapters(config: dict[str, Any], dry_run: bool = False) -> list[str]:
    """Move worker/skill adapters that haven't been loaded recently to an archive dir.

    The headmaster's own adapter (role=None) is never archived. Returns the
    list of archived role names. In dry-run mode the candidates are returned
    but nothing is moved.
    """
    days = int(config.get("archive", {}).get("adapter_idle_days", 90))
    if days <= 0:
        return []
    cutoff = datetime.now() - timedelta(days=days)
    archived: list[str] = []

    catalog = _load_worker_catalog()
    active_roles = {e.get("role") for e in catalog.values() if e.get("role")}

    if not dry_run:
        ADAPTER_ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

    for role in active_roles:
        if not role:
            continue
        adapter_dir = constants.adapter_dir_for(role)
        if not adapter_dir.exists():
            continue
        last_used = training.adapter_last_used(role=role)
        if last_used is None:
            # Never loaded; use directory mtime as a proxy.
            last_used = datetime.fromtimestamp(adapter_dir.stat().st_mtime)
        if last_used > cutoff:
            continue
        archived.append(role)
        if dry_run:
            continue
        dest = constants.adapter_archive_dir_for(role).with_suffix(
            f".bak.{datetime.now():%Y%m%d_%H%M%S}"
        )
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(adapter_dir), str(dest))

    return archived


def archive_idle_items(config: dict[str, Any], dry_run: bool = False) -> dict[str, list[str]]:
    """Run both archival passes and return what would be archived."""
    return {
        "notes": archive_idle_notes(config, dry_run=dry_run),
        "adapters": archive_idle_adapters(config, dry_run=dry_run),
    }


# ---- Restore ----


def list_archived_notes() -> list[str]:
    """Return filenames of notes currently in notes/archive/."""
    if not constants.NOTES_ARCHIVE_DIR.exists():
        return []
    return sorted(f.name for f in constants.NOTES_ARCHIVE_DIR.glob("*.md"))


def list_archived_adapters() -> list[str]:
    """Return basenames of archived adapter directories."""
    if not ADAPTER_ARCHIVE_DIR.exists():
        return []
    return sorted(
        f.name for f in ADAPTER_ARCHIVE_DIR.rglob("*")
        if f.is_dir() and (f / "adapter_config.json").exists()
    )


def restore_archived_note(filename: str) -> Path | None:
    """Move a note from notes/archive/ back to notes/."""
    src = constants.NOTES_ARCHIVE_DIR / filename
    if not src.exists():
        return None
    dest = constants.NOTES_DIR / filename
    counter = 1
    while dest.exists():
        dest = constants.NOTES_DIR / f"{src.stem}_{counter}{src.suffix}"
        counter += 1
    shutil.move(str(src), str(dest))
    # Restore any health sidecar alongside the note.
    sidecar_src = _skill_health_path(src)
    if sidecar_src.exists():
        sidecar_src.rename(_skill_health_path(dest))
    record_note_usage(dest)
    return dest


def restore_archived_adapter(role: str) -> Path | None:
    """Restore the most recently archived adapter for a role to its live path.

    Returns the restored adapter directory path, or None if no archive exists.
    """
    adapter_dir = constants.adapter_dir_for(role)
    candidates = sorted(
        ADAPTER_ARCHIVE_DIR.rglob(f"{role}.bak.*"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        return None
    src = candidates[0]
    if adapter_dir.exists():
        # Back up the live adapter before overwriting, just in case.
        backup = ADAPTER_ARCHIVE_DIR / f"{role}.live.bak.{datetime.now():%Y%m%d_%H%M%S}"
        shutil.move(str(adapter_dir), str(backup))
    shutil.move(str(src), str(adapter_dir))
    training.mark_adapter_used(role=role)
    return adapter_dir
