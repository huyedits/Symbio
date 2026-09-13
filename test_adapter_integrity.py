"""Checking the adapter before the model is given it.

adapter_seal.py could always prove an adapter was the bytes it was sealed as,
and nothing ever asked it — no code path in symbio/ referenced it. These tests
are about the asking: what happens at load, and how loudly.
"""
import json

import pytest

from symbio import constants
from symbio.app import modelload


@pytest.fixture
def project(tmp_path, monkeypatch):
    """A project root with a live adapter directory in it."""
    monkeypatch.setattr(constants, "PROJECT_DIR", tmp_path)
    monkeypatch.setattr(constants, "CONFIG_FILE", tmp_path / "config.json")
    real = __import__("pathlib").Path("adapter_seal.py").resolve()
    (tmp_path / "adapter_seal.py").write_bytes(real.read_bytes())
    live = tmp_path / "adapters"
    live.mkdir()
    (live / "adapters.safetensors").write_bytes(b"WEIGHTS" * 16)
    (live / "adapter_config.json").write_text('{"model": "some/model"}')
    return tmp_path


@pytest.fixture(autouse=True)
def _fresh_module_cache():
    """The loaded adapter_seal and the on-disk policy are cached per process."""
    from symbio import adapter_integrity as ai
    ai._seal_module = None
    ai._seal_tried = False
    ai._disk_policy = None
    yield
    ai._seal_module = None
    ai._seal_tried = False
    ai._disk_policy = None


def _seal(project, integrity_only=True):
    from symbio import adapter_integrity as ai
    module = ai._seal()
    module.seal(project / "adapters", quiet=True, integrity_only=integrity_only)
    module.build_root(project, quiet=True)
    return module


def _tamper(project):
    (project / "adapters" / "adapters.safetensors").write_bytes(b"EVIL" * 28)


# ---- what it says, and when ----

def test_an_unsealed_adapter_says_nothing(project):
    from symbio import adapter_integrity as ai
    said = []

    ai.enforce(project / "adapters", {"agent": {}}, output_fn=said.append)

    assert said == []


def test_an_intact_adapter_says_nothing(project):
    """A tripwire that talks when nothing happened gets unplugged."""
    from symbio import adapter_integrity as ai
    _seal(project)
    said = []

    ai.enforce(project / "adapters", {"agent": {}}, output_fn=said.append)

    assert said == []


def test_tampered_weights_are_reported_and_name_what_moved(project):
    from symbio import adapter_integrity as ai
    _seal(project)
    _tamper(project)
    said = []

    ai.enforce(project / "adapters", {"agent": {}}, output_fn=said.append)

    assert len(said) == 1
    assert "SEAL BROKEN" in said[0]
    assert "weights file changed: adapters.safetensors" in said[0]
    assert "adapter_seal.py seal" in said[0]      # tells you how to fix it


def test_warn_is_the_default_and_still_loads(project):
    """The common cause of a mismatch is an adapter retrained since it was
    sealed. Refusing that by default would end every training run in a broken
    session."""
    from symbio import adapter_integrity as ai
    _seal(project)
    _tamper(project)

    ai.enforce(project / "adapters", None, output_fn=lambda _t: None)  # no raise


def test_refuse_does_not_hand_the_model_tampered_weights(project):
    from symbio import adapter_integrity as ai
    _seal(project)
    _tamper(project)

    with pytest.raises(RuntimeError, match="failed its seal"):
        ai.enforce(project / "adapters", {"agent": {"verify_adapters": "refuse"}},
                   output_fn=lambda _t: None)


def test_off_does_not_even_look(project):
    from symbio import adapter_integrity as ai
    _seal(project)
    _tamper(project)
    said = []

    ai.enforce(project / "adapters", {"agent": {"verify_adapters": "off"}},
               output_fn=said.append)

    assert said == []


def test_the_policy_is_read_from_config_json_when_none_is_passed(project):
    """Most callers of load() never had a config to pass, so a policy read only
    from the argument would leave "refuse" unreachable for nearly every path."""
    from symbio import adapter_integrity as ai
    (project / "config.json").write_text(
        json.dumps({"agent": {"verify_adapters": "refuse"}}))
    _seal(project)
    _tamper(project)

    with pytest.raises(RuntimeError):
        ai.enforce(project / "adapters", None, output_fn=lambda _t: None)


def test_a_missing_checker_is_a_missing_feature_not_an_error(project):
    """An installed copy of the package that ships no root scripts."""
    from symbio import adapter_integrity as ai
    _seal(project)
    _tamper(project)
    (project / "adapter_seal.py").unlink()
    ai._seal_module, ai._seal_tried = None, False

    ai.enforce(project / "adapters", {"agent": {"verify_adapters": "refuse"}},
               output_fn=lambda _t: None)     # no raise


# ---- it is actually wired to the load ----

def test_the_load_path_checks_before_handing_over_the_weights(project, monkeypatch):
    """Before, not after: every caller that loads an adapter comes through
    modelload.load, and a check after the fact is a check of nothing."""
    _seal(project)
    _tamper(project)
    order = []
    monkeypatch.setattr(modelload, "load", modelload.load)

    def _fake_mlx_load(*_a, **_kw):
        order.append("loaded")
        return object(), object()

    import mlx_lm
    monkeypatch.setattr(mlx_lm, "load", _fake_mlx_load)

    with pytest.raises(RuntimeError, match="failed its seal"):
        modelload.load("some/model", adapter_path=str(project / "adapters"),
                       config={"agent": {"verify_adapters": "refuse"}})

    assert order == []           # the model was never given the weights


def test_a_load_with_no_adapter_is_untouched(project, monkeypatch):
    called = []

    def _fake_mlx_load(*_a, **_kw):
        called.append(True)
        return object(), object()

    import mlx_lm
    monkeypatch.setattr(mlx_lm, "load", _fake_mlx_load)
    monkeypatch.setattr(modelload, "trust_tokenizer_eos", lambda _t: None)

    modelload.load("some/model")

    assert called == [True]


# ---- training re-seals what it just wrote ----

def test_a_finished_training_run_reseals_and_rebuilds_the_root(project):
    """Without this the seal is a photograph that goes out of date the moment
    you train, every later load warns about weights that are fine, and
    everyone learns to ignore the warning — the only way a tripwire really
    fails."""
    from symbio.app import training
    from symbio import adapter_integrity as ai

    assert training.reseal_adapter(project / "adapters") is True
    assert ai.check_adapter(project / "adapters")["state"] == "intact"

    # ...and again after the next run writes new weights.
    (project / "adapters" / "adapters.safetensors").write_bytes(b"NEWER" * 20)
    said = []
    ai.enforce(project / "adapters", {"agent": {}}, output_fn=said.append)
    assert said, "stale seal should complain before the re-seal"

    assert training.reseal_adapter(project / "adapters") is True
    said.clear()
    ai.enforce(project / "adapters", {"agent": {}}, output_fn=said.append)
    assert said == []


def test_resealing_an_empty_directory_is_not_reported_as_success(project):
    from symbio.app import training

    empty = project / "adapters" / "workers" / "nothing"
    empty.mkdir(parents=True)

    assert training.reseal_adapter(empty) is False
