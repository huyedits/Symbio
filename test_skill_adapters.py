"""Tests for skill adapters and idle archival."""

import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from symbio import constants


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=False, enable_thinking=False):
        return " ".join(f"{m['role']}: {m['content']}" for m in messages)


@pytest.fixture
def isolated_skill_env(tmp_path, monkeypatch):
    """Point skill/adapter/note directories into a temporary tree."""
    notes_dir = tmp_path / "notes"
    adapter_dir = tmp_path / "adapters"
    adapter_archive_dir = tmp_path / "adapters_archive"
    data_dir = tmp_path / "training_data"
    worker_models = tmp_path / "worker_models.json"

    notes_dir.mkdir(parents=True, exist_ok=True)
    (notes_dir / "archive").mkdir(parents=True, exist_ok=True)
    adapter_dir.mkdir(parents=True, exist_ok=True)
    adapter_archive_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    # Patch constants before importing modules that read them at import time.
    monkeypatch.setattr(constants, "NOTES_DIR", notes_dir)
    monkeypatch.setattr(constants, "NOTES_ARCHIVE_DIR", notes_dir / "archive")
    monkeypatch.setattr(constants, "ADAPTER_DIR", adapter_dir)
    monkeypatch.setattr(constants, "WORKER_ADAPTERS_DIR", adapter_dir / "workers")
    monkeypatch.setattr(constants, "ADAPTER_ARCHIVE_DIR", adapter_archive_dir)
    monkeypatch.setattr(constants, "DATA_DIR", data_dir)
    monkeypatch.setattr(constants, "WORKER_MODELS_FILE", worker_models)

    # skills.py captures ADAPTER_ARCHIVE_DIR at import time; patch the module copy too.
    from symbio.app import skills as skills_mod
    monkeypatch.setattr(skills_mod, "ADAPTER_ARCHIVE_DIR", adapter_archive_dir)
    monkeypatch.setattr(skills_mod, "NOTES_USAGE_FILE", notes_dir / ".last_used.json")

    yield {
        "notes_dir": notes_dir,
        "adapter_dir": adapter_dir,
        "adapter_archive_dir": adapter_archive_dir,
        "data_dir": data_dir,
        "worker_models": worker_models,
    }


@pytest.fixture
def config():
    return {
        "model_name": "dummy-model",
        "assistant_name": "Symbio",
        "user_name": "Tester",
        "archive": {
            "note_idle_days": 7,
            "adapter_idle_days": 7,
        },
    }


def test_skill_slug_simple():
    from symbio.app.skills import _skill_slug
    assert _skill_slug("Write Python") == "write_python"
    assert _skill_slug("  Mixed-CASE!!!123  ") == "mixed_case_123"
    assert _skill_slug("") == "skill"


def test_save_skill_adapter_creates_note_and_catalog(isolated_skill_env, config):
    from symbio.app import skills

    result = skills.save_skill_adapter(
        "Summarize News", "1. Extract key facts.\n2. Draft a headline.",
        config, FakeTokenizer(), auto_train=False,
    )

    assert "role" in result
    assert result["role"] == "summarize_news"
    note_path = Path(result["note_path"])
    assert note_path.exists()
    assert "# Skill: Summarize News" in note_path.read_text(encoding="utf-8")

    catalog = json.loads(isolated_skill_env["worker_models"].read_text(encoding="utf-8"))
    key = f"skill_{result['role']}"
    assert key in catalog
    entry = catalog[key]
    assert entry.get("is_skill") is True
    assert entry.get("skill_name") == "Summarize News"
    assert "system_prompt" in entry
    assert "Summarize News" in entry["system_prompt"]


def test_save_skill_adapter_seeds_training_data(isolated_skill_env, config):
    from symbio.app import skills

    result = skills.save_skill_adapter(
        "Refactor Code", "1. Find smells.\n2. Simplify.",
        config, FakeTokenizer(), auto_train=False,
    )

    train_file = isolated_skill_env["data_dir"] / "workers" / result["role"] / "train.jsonl"
    assert train_file.exists()
    lines = [json.loads(ln) for ln in train_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) >= 2
    assert all(ln.get("metadata", {}).get("skill_seed") for ln in lines)


def test_list_skill_adapters_reports_entries(isolated_skill_env, config):
    from symbio.app import skills

    skills.save_skill_adapter(
        "List Maker", "1. Read items.\n2. Format a list.",
        config, FakeTokenizer(), auto_train=False,
    )
    adapters = skills.list_skill_adapters()
    roles = {a["role"] for a in adapters}
    assert "list_maker" in roles
    list_meta = next(a for a in adapters if a["role"] == "list_maker")
    assert list_meta["name"] == "List Maker"
    assert list_meta["adapter_exists"] is False


def test_delete_skill_adapter_removes_files_and_catalog(isolated_skill_env, config):
    from symbio.app import skills

    result = skills.save_skill_adapter(
        "Temp Skill", "steps here", config, FakeTokenizer(), auto_train=False,
    )
    role = result["role"]

    deleted = skills.delete_skill_adapter(role)
    assert deleted["role"] == role
    assert len(deleted["removed_entries"]) == 1

    catalog = json.loads(isolated_skill_env["worker_models"].read_text(encoding="utf-8"))
    assert not any(e.get("is_skill") for e in catalog.values())

    assert not (isolated_skill_env["data_dir"] / "workers" / role).exists()


def test_record_note_usage_updates_manifest(isolated_skill_env):
    from symbio.app import skills

    note = isolated_skill_env["notes_dir"] / "sample.md"
    note.write_text("# Sample\nbody", encoding="utf-8")
    skills.record_note_usage(note)

    manifest = json.loads((isolated_skill_env["notes_dir"] / ".last_used.json").read_text(encoding="utf-8"))
    assert str(note.resolve()) in manifest
    assert datetime.fromisoformat(manifest[str(note.resolve())])


def test_archive_idle_notes_moves_old_unused_notes(isolated_skill_env, config):
    from symbio.app import skills

    note = isolated_skill_env["notes_dir"] / "old_note.md"
    note.write_text("# Old\nstale", encoding="utf-8")
    old = (datetime.now() - timedelta(days=10)).timestamp()
    os.utime(note, (old, old))

    archived = skills.archive_idle_notes(config)
    assert "old_note.md" in archived
    assert not note.exists()
    assert (isolated_skill_env["notes_dir"] / "archive" / "old_note.md").exists()


def test_archive_idle_notes_respects_protected_notes(isolated_skill_env, config):
    from symbio.app import skills

    protected = isolated_skill_env["notes_dir"] / "identity.md"
    protected.write_text("# My Identity\nI am Symbio.", encoding="utf-8")
    old = (datetime.now() - timedelta(days=10)).timestamp()
    os.utime(protected, (old, old))

    archived = skills.archive_idle_notes(config)
    assert "identity.md" not in archived
    assert protected.exists()


def test_archive_idle_adapters_moves_old_unused_adapters(isolated_skill_env, config):
    from symbio.app import skills
    from symbio.app import training

    role = "old_skill"
    adapter_dir = isolated_skill_env["adapter_dir"] / "workers" / role
    adapter_dir.mkdir(parents=True, exist_ok=True)
    (adapter_dir / "adapter_config.json").write_text("{}", encoding="utf-8")

    # Pretend the adapter was last loaded long ago.
    old = datetime.now() - timedelta(days=10)
    (adapter_dir / "last_used.json").write_text(
        json.dumps({"last_used": old.isoformat()}), encoding="utf-8"
    )

    # Ensure the role is in the worker catalog.
    isolated_skill_env["worker_models"].write_text(
        json.dumps({
            f"skill_{role}": {
                "role": role,
                "is_skill": True,
                "skill_name": "Old Skill",
                "description": "Old",
            }
        }, indent=2) + "\n",
        encoding="utf-8",
    )

    archived = skills.archive_idle_adapters(config)
    assert role in archived
    assert not adapter_dir.exists()
    archived_files = list(isolated_skill_env["adapter_archive_dir"].rglob("*"))
    assert any(role in str(p) for p in archived_files)


def test_archive_idle_items_runs_both_passes(isolated_skill_env, config):
    from symbio.app import skills

    note = isolated_skill_env["notes_dir"] / "idle.md"
    note.write_text("# Idle", encoding="utf-8")
    old = (datetime.now() - timedelta(days=10)).timestamp()
    os.utime(note, (old, old))

    result = skills.archive_idle_items(config)
    assert "idle.md" in result["notes"]


def test_archive_idle_items_dry_run_does_not_move(isolated_skill_env, config):
    from symbio.app import skills

    note = isolated_skill_env["notes_dir"] / "dry_run.md"
    note.write_text("# Dry", encoding="utf-8")
    old = (datetime.now() - timedelta(days=10)).timestamp()
    os.utime(note, (old, old))

    result = skills.archive_idle_items(config, dry_run=True)
    assert "dry_run.md" in result["notes"]
    assert note.exists()


def test_restore_archived_note(isolated_skill_env, config):
    from symbio.app import skills

    note = isolated_skill_env["notes_dir"] / "to_restore.md"
    note.write_text("# Restore me", encoding="utf-8")
    old = (datetime.now() - timedelta(days=10)).timestamp()
    os.utime(note, (old, old))

    skills.archive_idle_notes(config)
    restored = skills.restore_archived_note("to_restore.md")
    assert restored is not None
    assert restored.exists()
    assert restored.name == "to_restore.md"
    assert not (isolated_skill_env["notes_dir"] / "archive" / "to_restore.md").exists()
    # Restored notes should be marked as recently used so they are not
    # immediately re-archived.
    manifest = json.loads((isolated_skill_env["notes_dir"] / ".last_used.json").read_text(encoding="utf-8"))
    last_used = datetime.fromisoformat(manifest[str(restored.resolve())])
    assert (datetime.now() - last_used).total_seconds() < 5


def test_restore_archived_note_missing(isolated_skill_env):
    from symbio.app import skills

    assert skills.restore_archived_note("missing.md") is None


def test_restore_archived_adapter(isolated_skill_env, config):
    from symbio.app import skills, training

    role = "restorable_skill"
    archived = isolated_skill_env["adapter_archive_dir"] / f"{role}.bak.20240101_120000"
    archived.mkdir(parents=True, exist_ok=True)
    (archived / "adapter_config.json").write_text("{}", encoding="utf-8")

    restored = skills.restore_archived_adapter(role)
    assert restored is not None
    assert restored == constants.adapter_dir_for(role)
    assert restored.exists()
    assert (restored / "adapter_config.json").exists()
    assert not archived.exists()
    # Restored adapters should have a fresh last_used timestamp.
    last_used = training.adapter_last_used(role=role)
    assert last_used is not None
    assert (datetime.now() - last_used).total_seconds() < 5


def test_list_archived_notes_and_adapters(isolated_skill_env):
    from symbio.app import skills

    note = isolated_skill_env["notes_dir"] / "archive" / "old.md"
    note.write_text("# Old", encoding="utf-8")
    archived = isolated_skill_env["adapter_archive_dir"] / "some_role.bak.20240101_120000"
    archived.mkdir(parents=True, exist_ok=True)
    (archived / "adapter_config.json").write_text("{}", encoding="utf-8")

    assert "old.md" in skills.list_archived_notes()
    assert "some_role.bak.20240101_120000" in skills.list_archived_adapters()


def test_skill_health_sidecar_records_error_and_correction(isolated_skill_env):
    from symbio.app import skills

    note = isolated_skill_env["notes_dir"] / "20260726_120000_Skill_Foo.md"
    note.write_text("# Skill: Foo\nsteps", encoding="utf-8")

    skills.record_skill_error(note, "model failed to load")
    skills.record_skill_correction(note, "do it this way instead")

    entries = skills.read_skill_health(note)
    assert len(entries) == 2
    assert entries[0]["type"] == "error"
    assert "failed to load" in entries[0]["text"]
    assert entries[1]["type"] == "correction"


def test_skill_health_sidecar_moves_with_archive_and_restore(isolated_skill_env, config):
    from symbio.app import skills

    note = isolated_skill_env["notes_dir"] / "skill.md"
    note.write_text("# Skill: Archivable\nsteps", encoding="utf-8")
    skills.record_skill_error(note, "error one")
    old = (datetime.now() - timedelta(days=10)).timestamp()
    os.utime(note, (old, old))

    skills.archive_idle_notes(config)
    archived_note = isolated_skill_env["notes_dir"] / "archive" / "skill.md"
    archived_sidecar = isolated_skill_env["notes_dir"] / "archive" / "skill.md.health.jsonl"
    assert archived_note.exists()
    assert archived_sidecar.exists()
    assert skills.read_skill_health(archived_note)

    restored = skills.restore_archived_note("skill.md")
    assert restored is not None
    assert skills.read_skill_health(restored)
    assert (isolated_skill_env["notes_dir"] / "archive" / "skill.md.health.jsonl").exists() is False


def test_delete_skill_adapter_removes_note_and_sidecar(isolated_skill_env, config):
    from symbio.app import skills

    result = skills.save_skill_adapter(
        "Disposable", "steps", config, FakeTokenizer(), auto_train=False,
    )
    note = Path(result["note_path"])
    skills.record_skill_error(note, "bad")
    sidecar = note.with_suffix(note.suffix + ".health.jsonl")

    skills.delete_skill_adapter(result["role"])

    assert not note.exists()
    assert not sidecar.exists()
