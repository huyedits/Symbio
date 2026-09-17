#!/usr/bin/env python3
"""Where an install keeps the things it writes.

PROJECT_DIR was `Path(__file__).parent.parent` — the repository root when you
run from a clone, and **site-packages** when you do not. So `pip install
symbio-cli` produced an agent that would create notes/, adapters/,
training_data/, logs/, sessions/, config.json and prompt.md inside
site-packages: scattered across virtualenvs, destroyed by an upgrade, and
unwritable outright on a system Python. The package installed cleanly and the
user's data had nowhere to live.
"""
import os
from pathlib import Path

import pytest

from symbio import constants


def _resolve(monkeypatch, package_file: Path, home: Path, env=None):
    """_default_project_dir as it would answer from a given install layout."""
    monkeypatch.setattr(constants, "__file__", str(package_file))
    monkeypatch.setattr(Path, "home", staticmethod(lambda: home))
    if env is None:
        monkeypatch.delenv("SYMBIO_HOME", raising=False)
    else:
        monkeypatch.setenv("SYMBIO_HOME", env)
    return constants._default_project_dir()


def _checkout(tmp_path) -> Path:
    """A source tree: pyproject.toml beside the package directory."""
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
    (tmp_path / "symbio").mkdir()
    return tmp_path / "symbio" / "constants.py"


def _site_packages(tmp_path) -> Path:
    """An installed wheel: a package directory and nothing else around it."""
    site = tmp_path / "lib" / "python3.12" / "site-packages"
    (site / "symbio").mkdir(parents=True)
    return site / "symbio" / "constants.py"


def test_an_installed_package_writes_to_the_home_workspace(tmp_path, monkeypatch):
    """The case that was broken: nothing may be created in site-packages."""
    home = tmp_path / "home"

    resolved = _resolve(monkeypatch, _site_packages(tmp_path), home)

    assert resolved == home / ".symbio"
    assert "site-packages" not in str(resolved)


def test_a_checkout_keeps_every_path_it_already_had(tmp_path, monkeypatch):
    """`pip install -e .` and ./install.sh both produce this, and an existing
    clone's notes and adapters must not move because of a packaging fix."""
    package_file = _checkout(tmp_path)

    assert _resolve(monkeypatch, package_file, tmp_path / "home") == tmp_path


def test_symbio_home_wins_over_both(tmp_path, monkeypatch):
    elsewhere = tmp_path / "elsewhere"

    resolved = _resolve(monkeypatch, _checkout(tmp_path), tmp_path / "home",
                        env=str(elsewhere))

    assert resolved == elsewhere.resolve()


def test_a_blank_symbio_home_is_not_a_setting(tmp_path, monkeypatch):
    """An exported-but-empty variable is the shell's, not the user's."""
    resolved = _resolve(monkeypatch, _checkout(tmp_path), tmp_path / "home",
                        env="   ")

    assert resolved == tmp_path


def test_symbio_home_expands_a_tilde(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    resolved = _resolve(monkeypatch, _site_packages(tmp_path), tmp_path,
                        env="~/somewhere")

    assert resolved == (tmp_path / "somewhere").resolve()


# ---- the roster ----

def test_the_worker_roster_is_not_written_inside_the_package(tmp_path):
    """It lived inside the package while it was committed. A wheel's package
    directory is the wrong place to write a file and often not writable.

    In a SUBPROCESS, because conftest reassigns WORKER_MODELS_FILE for the
    whole suite — a value the fixtures patch is a value no in-process test can
    check, which is exactly how symbio/rag.py read a directory that had never
    existed while every retrieval test passed.
    """
    import subprocess
    import sys

    probe = (
        "from symbio import constants; "
        "print(constants.WORKER_MODELS_FILE); "
        "print(constants.PROJECT_DIR)"
    )
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True,
                         text=True, cwd=str(tmp_path))

    assert out.returncode == 0, out.stderr
    roster, project = (Path(line) for line in out.stdout.split())
    assert "site-packages" not in str(roster)
    assert roster.is_relative_to(project)


def test_importing_survives_a_workspace_it_cannot_create(tmp_path, monkeypatch):
    """`symb --help` must not raise because SYMBIO_HOME points somewhere
    unwritable — the commands that need a directory report that themselves."""
    import subprocess
    import sys

    unwritable = tmp_path / "ro"
    unwritable.mkdir()
    unwritable.chmod(0o500)
    env = {**os.environ, "SYMBIO_HOME": str(unwritable / "workspace")}
    try:
        out = subprocess.run(
            [sys.executable, "-c", "from symbio import constants; print('ok')"],
            capture_output=True, text=True, env=env,
            cwd=str(tmp_path))
        assert out.returncode == 0, out.stderr
        assert "ok" in out.stdout
    finally:
        unwritable.chmod(0o700)
