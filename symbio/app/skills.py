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
from symbio.app import dispatch, memory, training


NOTES_USAGE_FILE = constants.NOTES_DIR / ".last_used.json"
ADAPTER_ARCHIVE_DIR = constants.ADAPTER_ARCHIVE_DIR
_SKILL_FLAG = {"is_skill": True}


def _skill_slug(name: str) -> str:
    """Stable, filesystem-safe identifier from a skill title."""
    s = name.lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = s.strip("_")
    return s or "skill"


def _build_skill_system_prompt(name: str, steps: str) -> str:
    """A concise worker system prompt derived from the skill's steps."""
    return (
        f"You are the specialist worker for the skill '{name}'. "
        "Follow the steps below exactly, produce only the requested output, "
        "and do not add extra commentary.\n\n"
        f"Steps:\n{steps}\n\n"
        "Reply with the result of applying these steps to the user's request."
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
        "model_name": config["model_name"],
        "role": role,
        "description": f"Skill: {name}",
        "adapter_compatible": True,
        "memory_note": "~1 GB on disk, headmaster-size RAM at runtime",
        "system_prompt": system_prompt,
        "is_skill": True,
        "skill_name": name,
    }
    _save_worker_catalog(catalog)
    return role


def _seed_skill_training_data(
    role: str, system_prompt: str, name: str, steps: str, tokenizer: Any
) -> int:
    """Write a few synthetic training samples for a brand-new skill worker.

    Returns the number of samples written. Real usage samples accumulate
    automatically in dispatch.WorkerPool.run_delegated_task.
    """
    data_dir = constants.data_dir_for(role)
    data_dir.mkdir(parents=True, exist_ok=True)
    train_file = constants.data_dir_for(role) / "train.jsonl"

    # Clear any stale auto-seeded samples so re-saving a skill refreshes the seed.
    if train_file.exists():
        lines = [
            ln for ln in train_file.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not json.loads(ln).get("metadata", {}).get("skill_seed")
        ]
    else:
        lines = []

    samples = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": f"Apply the skill '{name}'.",
        },
        {
            "role": "assistant",
            "content": f"Using skill '{name}':\n{steps}",
        },
    ]
    text = tokenizer.apply_chat_template(
        samples, tokenize=False, add_generation_prompt=False, enable_thinking=False
    )
    seed = {"text": text, "metadata": {"skill_seed": True, "skill": name}}
    lines.append(json.dumps(seed))

    # Also seed a generic "how do I do X?" sample pointing at the skill.
    samples2 = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": f"How do I perform '{name}'?",
        },
        {
            "role": "assistant",
            "content": steps,
        },
    ]
    text2 = tokenizer.apply_chat_template(
        samples2, tokenize=False, add_generation_prompt=False, enable_thinking=False
    )
    seed2 = {"text": text2, "metadata": {"skill_seed": True, "skill": name}}
    lines.append(json.dumps(seed2))

    train_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 2


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

    seeded = _seed_skill_training_data(role, system_prompt, name, steps, tokenizer)
    adapter_dir = constants.adapter_dir_for(role)

    result = {
        "note_path": str(note_path),
        "role": role,
        "adapter_dir": str(adapter_dir),
        "seeded_samples": seeded,
        "trained": False,
        "message": f"Skill '{name}' saved as worker role '{role}' with {seeded} seed samples.",
    }

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
    """Remove a skill's worker catalog entry, adapter weights, and training data."""
    catalog = _load_worker_catalog()
    removed_keys = [k for k, e in catalog.items() if e.get("role") == role and e.get("is_skill")]
    for k in removed_keys:
        del catalog[k]
    _save_worker_catalog(catalog)

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
