"""Collecting adapters into one readable folder, losslessly.

The round trip is the thing that matters: an archive you cannot get back out
of is a way to lose 26 adapters, not a way to organise them.
"""
import json

import pytest

import archive_adapters as aa


@pytest.fixture
def project(tmp_path, monkeypatch):
    """A miniature project: one headmaster, two workers, training data."""
    (tmp_path / "adapters").mkdir()
    (tmp_path / "adapters/adapters.safetensors").write_bytes(b"HEADMASTER-WEIGHTS")
    (tmp_path / "adapters/adapter_config.json").write_text(
        json.dumps({"iters": 600, "fine_tune_type": "lora"}))
    (tmp_path / "adapters/training_progress.json").write_text(
        json.dumps({"total_iters": 600, "label": "HEADMASTER"}))
    (tmp_path / "training_data").mkdir()
    (tmp_path / "training_data/train.jsonl").write_text('{"text": "hm"}\n')
    (tmp_path / "training_data/valid.jsonl").write_text('{"text": "hmv"}\n')

    for name, iters in (("brew_tea", 150), ("fix_wifi", 12)):
        d = tmp_path / "adapters/workers" / name
        d.mkdir(parents=True)
        (d / "adapters.safetensors").write_bytes(f"{name}-WEIGHTS".encode())
        (d / "adapter_config.json").write_text(json.dumps({"iters": iters}))
        td = tmp_path / "training_data/workers" / name
        td.mkdir(parents=True)
        (td / "train.jsonl").write_text(f'{{"text": "{name}"}}\n')

    aa.set_root(tmp_path)
    return tmp_path


def test_it_finds_the_headmaster_and_every_worker(project, capsys):
    assert aa.build() == 0
    out = project / "Adapter_skills"
    names = sorted(p.name for p in out.iterdir())
    assert names == ["HEADMASTER_600_HEADMASTER",
                     "brew_tea_150_WORKER",
                     "fix_wifi_12_WORKER"]


def test_the_folder_name_carries_skill_iterations_and_kind(project):
    aa.build()
    assert (project / "Adapter_skills/fix_wifi_12_WORKER").is_dir()


def test_the_weights_are_renamed_and_byte_identical(project):
    aa.build()
    src = (project / "adapters/workers/brew_tea/adapters.safetensors").read_bytes()
    dst = (project / "Adapter_skills/brew_tea_150_WORKER"
                     "/brew_tea_150.safetensors").read_bytes()
    assert src == dst


def test_metadata_and_training_data_come_along(project):
    aa.build()
    f = project / "Adapter_skills/HEADMASTER_600_HEADMASTER"
    assert (f / "adapter_config.json").exists()
    assert (f / "training_progress.json").exists()
    assert (f / "training_data/train.jsonl").exists()
    assert (f / "training_data/valid.jsonl").exists()


def test_the_originals_are_left_alone(project):
    """It copies. adapters/ is what the app loads at boot and must not move."""
    before = (project / "adapters/adapters.safetensors").read_bytes()
    aa.build()
    assert (project / "adapters/adapters.safetensors").read_bytes() == before
    assert (project / "adapters/workers/brew_tea/adapters.safetensors").exists()


def test_iterations_fall_back_to_training_progress(project):
    """A worker whose adapter_config has no iters still gets a real number."""
    d = project / "adapters/workers/no_iters"
    d.mkdir(parents=True)
    (d / "adapters.safetensors").write_bytes(b"x")
    (d / "adapter_config.json").write_text(json.dumps({"fine_tune_type": "lora"}))
    (d / "training_progress.json").write_text(json.dumps({"total_iters": 42}))
    aa.build()
    assert (project / "Adapter_skills/no_iters_42_WORKER").is_dir()


def test_an_unknown_iteration_count_is_zero_not_a_crash(project):
    d = project / "adapters/workers/mystery"
    d.mkdir(parents=True)
    (d / "adapters.safetensors").write_bytes(b"x")
    aa.build()
    assert (project / "Adapter_skills/mystery_0_WORKER").is_dir()


def test_a_directory_without_weights_is_skipped(project):
    (project / "adapters/workers/empty").mkdir(parents=True)
    aa.build()
    assert not (project / "Adapter_skills/empty_0_WORKER").exists()


def test_dry_run_writes_nothing(project):
    assert aa.build(dry_run=True) == 0
    assert not (project / "Adapter_skills").exists()


def test_running_twice_is_safe(project):
    aa.build()
    assert aa.build() == 0
    assert (project / "Adapter_skills/brew_tea_150_WORKER"
                      "/brew_tea_150.safetensors").exists()


# ---- the round trip ----

def test_restore_rebuilds_the_filename_mlx_lm_demands(project):
    """mlx_lm hardcodes adapter_path/'adapters.safetensors'. An archive that
    cannot produce that name back is a one-way trip."""
    aa.build()
    assert aa.restore("brew_tea_150_WORKER") == 0
    dest = project / "restored_brew_tea_150_WORKER"
    assert (dest / "adapters.safetensors").exists()
    assert (dest / "adapter_config.json").exists()


def test_the_restored_weights_are_the_original_bytes(project):
    aa.build()
    aa.restore("brew_tea_150_WORKER")
    original = (project / "adapters/workers/brew_tea/adapters.safetensors").read_bytes()
    restored = (project / "restored_brew_tea_150_WORKER"
                          "/adapters.safetensors").read_bytes()
    assert restored == original


def test_restore_does_not_overwrite_the_live_adapter(project):
    """Putting a restored adapter into service is the operator's decision."""
    live = (project / "adapters/adapters.safetensors").read_bytes()
    aa.build()
    aa.restore("brew_tea_150_WORKER")
    assert (project / "adapters/adapters.safetensors").read_bytes() == live


def test_restoring_something_that_is_not_there_fails_cleanly(project):
    aa.build()
    assert aa.restore("no_such_skill") == 1


def test_the_manifest_records_where_it_came_from(project):
    aa.build()
    man = json.loads((project / "Adapter_skills/fix_wifi_12_WORKER"
                                "/manifest.json").read_text())
    assert man["skill"] == "fix_wifi"
    assert man["kind"] == "WORKER"
    assert man["iters"] == 12
    assert man["weights"] == "fix_wifi_12.safetensors"
    assert "adapters/workers/fix_wifi" in man["source_adapter_dir"]


def test_nothing_to_archive_is_reported_not_crashed(tmp_path):
    aa.set_root(tmp_path)
    assert aa.build() == 1
