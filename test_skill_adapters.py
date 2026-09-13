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


def test_seed_samples_keep_steps_out_of_the_system_turn(isolated_skill_env, config):
    """The procedure must be the training target, not context to copy.

    If the steps appear in the system turn, the adapter learns to echo what
    it was handed and no evaluation can distinguish that from real learning.
    """
    from symbio.app import skills

    steps = "1. Find smells.\n2. Simplify."
    result = skills.save_skill_adapter(
        "Refactor Code", steps, config, FakeTokenizer(), auto_train=False,
    )
    train_file = isolated_skill_env["data_dir"] / "workers" / result["role"] / "train.jsonl"
    lines = [json.loads(ln) for ln in train_file.read_text(encoding="utf-8").splitlines() if ln.strip()]

    for line in lines:
        # FakeTokenizer renders turns as "<role>: <content>".
        system_turn = line["text"].split("user:")[0]
        assert "Find smells" not in system_turn
        assert "Steps:" not in system_turn
        assert "assistant:" in line["text"]

    # ...and the steps must still be present as the assistant target. Only for
    # the recall samples: contrast samples answer with a decline by design, and
    # step samples answer with one step rather than the whole procedure.
    recall = [ln for ln in lines if ln["metadata"].get("seed_kind") == "recall"]
    assert recall
    for line in recall:
        assert "Find smells" in line["text"].split("assistant:")[1]


def test_seed_user_turns_do_not_collide_with_eval_tasks():
    """Overlap would turn the eval into a memorisation check."""
    from symbio.app import skill_eval, skills

    name = "Refactor Code"
    seeds = {t.strip().lower() for t in skills._seed_user_turns(name)}
    evals = {t.prompt.strip().lower() for t in skill_eval.default_tasks(name)}
    assert not (seeds & evals)


def test_seeds_record_the_tokenizer_they_were_rendered_with(
        isolated_skill_env, config):
    from symbio.app import skills

    tok = FakeTokenizer()
    tok.name_or_path = "org/ModelA"
    result = skills.save_skill_adapter(
        "Refactor Code", "1. Find smells.", config, tok, auto_train=False)
    train_file = isolated_skill_env["data_dir"] / "workers" / result["role"] / "train.jsonl"
    lines = [json.loads(ln) for ln in train_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert all(ln["metadata"]["tokenized_for"] == "org/ModelA" for ln in lines)


def test_mismatched_tokenizer_is_detected(isolated_skill_env, config):
    """Training data from one chat template must not train another model."""
    from symbio.app import skills

    tok = FakeTokenizer()
    tok.name_or_path = "org/Qwen3-8B-MLX-4bit"
    result = skills.save_skill_adapter(
        "Refactor Code", "1. Find smells.", config, tok, auto_train=False)

    assert skills.seed_model_mismatch(result["role"], "org/Qwen3-8B-MLX-4bit") is None
    msg = skills.seed_model_mismatch(result["role"], "mlx/Mistral-Nemo-3bit")
    assert msg and "Mistral-Nemo-3bit" in msg and "Qwen3-8B-MLX-4bit" in msg


def test_republished_model_is_not_a_mismatch(isolated_skill_env, config):
    """Same weights under a different org must not block training."""
    from symbio.app import skills

    tok = FakeTokenizer()
    tok.name_or_path = "Qwen/Qwen3-8B-MLX-4bit"
    result = skills.save_skill_adapter(
        "Refactor Code", "1. Find smells.", config, tok, auto_train=False)
    assert skills.seed_model_mismatch(
        result["role"], "mlx-community/Qwen3-8B-MLX-4bit") is None


def test_guarded_train_refuses_mismatched_seed_data(isolated_skill_env, config, monkeypatch):
    from symbio.app import dispatch, skills

    tok = FakeTokenizer()
    tok.name_or_path = "org/Qwen3-8B-MLX-4bit"
    result = skills.save_skill_adapter(
        "Refactor Code", "1. Find smells.", config, tok, auto_train=False)
    role = result["role"]

    monkeypatch.setattr(dispatch, "catalog_entry_for_role",
                        lambda r: {"model_name": "mlx/Mistral-Nemo-3bit", "role": r})

    def _boom(*a, **k):
        raise AssertionError("training must not start on mismatched data")

    monkeypatch.setattr(dispatch.training, "run_training", _boom)
    trained, msg = dispatch.guarded_train_worker(role, config)
    assert trained is False
    assert "tokenized for" in msg


def test_worker_and_served_prompts_share_one_opener(isolated_skill_env, config):
    from symbio.app import skills

    name, steps = "Refactor Code", "1. Find smells."
    trained = skills.build_worker_system_prompt(name)
    served = skills._build_skill_system_prompt(name, steps)
    opener = skills._skill_prompt_opener(name)
    assert trained.startswith(opener) and served.startswith(opener)
    assert steps in served and steps not in trained


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


# ---- seed composition ----
#
# Seeds are recall-only by measurement: richer seed kinds were tried and lost
# badly (see the note in skills._seed_skill_training_data).

def test_seed_samples_carry_messages_for_prompt_masking(
        isolated_skill_env, config):
    """Without this the loss covers the system turn too, and these samples are
    almost entirely system turn."""
    from symbio.app import skills

    result = skills.save_skill_adapter(
        "Refactor Code", "1. Find smells. 2. Simplify.", config,
        FakeTokenizer(), auto_train=False)
    train_file = (isolated_skill_env["data_dir"] / "workers"
                  / result["role"] / "train.jsonl")
    lines = [json.loads(ln) for ln in
             train_file.read_text(encoding="utf-8").splitlines() if ln.strip()]

    assert lines
    for line in lines:
        roles = [m["role"] for m in line["messages"]]
        assert roles == ["system", "user", "assistant"]


# ---- worker model sizing ----
#
# A worker answers one narrow question under a short prompt. Running it at
# headmaster size meant a second full-size copy of the weights resident beside
# the one already loaded.

def test_worker_model_defaults_to_the_configured_worker(config):
    from symbio.app import skills

    cfg = {**config, "dispatch": {"worker_model_name": "org/Small-4B"}}
    assert skills.worker_model_name(cfg) == "org/Small-4B"


def test_worker_model_falls_back_to_the_headmaster(config):
    """Unset or null means "same model as the agent itself"."""
    from symbio.app import skills

    assert skills.worker_model_name(config) == config["model_name"]
    cfg = {**config, "dispatch": {"worker_model_name": None}}
    assert skills.worker_model_name(cfg) == config["model_name"]


def test_new_skills_are_catalogued_against_the_worker_model(
        isolated_skill_env, config, monkeypatch):
    from symbio.app import skills

    tok = FakeTokenizer()
    tok.name_or_path = "org/Small-4B"
    monkeypatch.setattr(skills, "worker_tokenizer", lambda name, fallback: tok)
    cfg = {**config, "dispatch": {"worker_model_name": "org/Small-4B"}}

    result = skills.save_skill_adapter(
        "Refactor Code", "1. Find smells.", cfg, FakeTokenizer(),
        auto_train=False)

    catalog = json.loads(
        isolated_skill_env["worker_models"].read_text(encoding="utf-8"))
    entry = catalog[f"skill_{result['role']}"]
    assert entry["model_name"] == "org/Small-4B"
    assert entry["model_name"] != cfg["model_name"]


def test_seeds_are_stamped_for_the_worker_not_the_headmaster(
        isolated_skill_env, config, monkeypatch):
    """The invariant that makes the first training run possible.

    Seeds carry the tokenizer that rendered them, and seed_model_mismatch
    refuses to train when that disagrees with the worker's model. Seeding a
    smaller worker with the headmaster's tokenizer would block every new
    skill's first fine-tune.
    """
    from symbio.app import skills

    worker_tok = FakeTokenizer()
    worker_tok.name_or_path = "org/Small-4B"
    monkeypatch.setattr(skills, "worker_tokenizer",
                        lambda name, fallback: worker_tok)
    headmaster_tok = FakeTokenizer()
    headmaster_tok.name_or_path = "org/Big-8B"
    cfg = {**config, "dispatch": {"worker_model_name": "org/Small-4B"}}

    result = skills.save_skill_adapter(
        "Refactor Code", "1. Find smells.", cfg, headmaster_tok,
        auto_train=False)

    catalog = json.loads(
        isolated_skill_env["worker_models"].read_text(encoding="utf-8"))
    entry = catalog[f"skill_{result['role']}"]
    assert skills.seed_model_mismatch(result["role"], entry["model_name"]) is None


def test_worker_tokenizer_reuses_a_matching_one_without_loading(config):
    """Republished weights share a stem, so no download is needed."""
    from symbio.app import skills

    tok = FakeTokenizer()
    tok.name_or_path = "Qwen/Qwen3-4B-4bit"
    assert skills.worker_tokenizer("mlx-community/Qwen3-4B-4bit", tok) is tok


# ---- routing signal from retrieval ----
#
# Retrieval already scores the user's message against every note, and skills
# are notes, so a skill-note hit is a routing signal. It was computed on every
# turn and never consulted for dispatch, leaving the model to infer from the
# tool schema alone that a matching specialist existed.

def _skill_note(tmp_path, name, body="1. Toggle wifi off. 2. Toggle it on."):
    p = tmp_path / f"{name.replace(' ', '_')}.md"
    p.write_text(f"# Skill: {name}\n\n{body}\n", encoding="utf-8")
    return p


def _catalogued(monkeypatch, isolated_skill_env, role, model_name, trained_for=None):
    isolated_skill_env["worker_models"].write_text(json.dumps({
        f"skill_{role}": {"model_name": model_name, "role": role,
                          "is_skill": True, "skill_name": "Fix wifi"},
    }), encoding="utf-8")
    if trained_for is not None:
        d = isolated_skill_env["adapter_dir"] / "workers" / role
        d.mkdir(parents=True, exist_ok=True)
        (d / "adapters.safetensors").write_bytes(b"w")
        (d / "adapter_config.json").write_text(
            json.dumps({"model": trained_for}), encoding="utf-8")


def test_a_matched_skill_note_offers_its_worker(isolated_skill_env, monkeypatch, tmp_path):
    from symbio.app import skills

    _catalogued(monkeypatch, isolated_skill_env, "fix_wifi", "m/4b", trained_for="m/4b")
    note = _skill_note(tmp_path, "Fix wifi")
    config = {"dispatch": {"enabled": True}}

    assert skills.delegatable_role_for_note(note, config) == "fix_wifi"


def test_no_offer_when_dispatch_is_off(isolated_skill_env, monkeypatch, tmp_path):
    from symbio.app import skills

    _catalogued(monkeypatch, isolated_skill_env, "fix_wifi", "m/4b", trained_for="m/4b")
    note = _skill_note(tmp_path, "Fix wifi")

    assert skills.delegatable_role_for_note(note, {"dispatch": {"enabled": False}}) is None


def test_no_offer_when_the_worker_has_no_trained_weights(
        isolated_skill_env, monkeypatch, tmp_path):
    """Handing the turn to an untrained worker is worse than answering."""
    from symbio.app import skills

    _catalogued(monkeypatch, isolated_skill_env, "fix_wifi", "m/4b")
    note = _skill_note(tmp_path, "Fix wifi")

    assert skills.delegatable_role_for_note(note, {"dispatch": {"enabled": True}}) is None


def test_no_offer_when_the_adapter_belongs_to_another_model(
        isolated_skill_env, monkeypatch, tmp_path):
    from symbio.app import skills

    _catalogued(monkeypatch, isolated_skill_env, "fix_wifi",
                "mlx-community/Qwen3-4B-4bit", trained_for="Qwen/Qwen3-8B-MLX-4bit")
    note = _skill_note(tmp_path, "Fix wifi")

    assert skills.delegatable_role_for_note(note, {"dispatch": {"enabled": True}}) is None


def test_a_plain_note_is_never_offered(isolated_skill_env, tmp_path):
    from symbio.app import skills

    plain = tmp_path / "plain.md"
    plain.write_text("# Groceries\n\nmilk\n", encoding="utf-8")

    assert skills.delegatable_role_for_note(plain, {"dispatch": {"enabled": True}}) is None
