"""The remaining findings: the CUDA path, the corpus backup nothing read, and
a skin that was editing the model's answers.
"""
import pytest

from symbio import backend
from symbio.app import chat_style, training


# ---- the CUDA trainer's own command line ----

def _lora(**over):
    base = {"batch_size": 1, "num_layers": 8, "learning_rate": 1e-4,
            "steps_per_eval": 50, "max_seq_length": 2048, "save_every": 50,
            "rank": 8, "dropout": 0.0, "scale": 20.0, "keys": ["q_proj"],
            "grad_checkpoint": True}
    base.update(over)
    return base


def _cuda_cmd(**over):
    return backend.trainer_command(
        "m/x", "/data", "/adapters", _lora(**over), 100, "/tmp/train.yaml",
        config={"backend": "cuda"})


def test_the_cuda_trainer_gets_its_grad_checkpoint_value_once():
    """cuda_lora declares --grad-checkpoint value-taking (mlx_lm's is
    store_true), so a bare flag appended on top made argparse exit 2 —
    "expected one argument" — and every CUDA run died at launch."""
    cmd = _cuda_cmd()

    assert cmd.count("--grad-checkpoint") == 1
    assert cmd[cmd.index("--grad-checkpoint") + 1] in ("true", "false")


def test_the_cuda_trainer_is_given_the_training_yaml():
    """The MLX branch passes --config and the CUDA branch dropped it, while
    cuda_lora declares the flag and run_training still builds and unlinks the
    file. Anything the YAML carries and the flags do not was lost silently."""
    cmd = _cuda_cmd()

    assert "--config" in cmd
    assert cmd[cmd.index("--config") + 1] == "/tmp/train.yaml"


def test_the_cuda_trainer_accepts_the_command_it_is_given():
    """Parsed by cuda_lora's own parser, which is where the mismatch was."""
    from symbio import cuda_lora

    args = cuda_lora.parse_args(_cuda_cmd()[3:])   # drop python -m module

    assert args.grad_checkpoint in ("true", "false")
    assert args.config == "/tmp/train.yaml"


def test_resume_is_read_rather_than_only_accepted():
    """The flag was parsed and never used, so the caller printed "Resuming
    from the existing adapter" while the trainer built a fresh one — the exact
    silent-nothing failure resume_source exists to catch."""
    import inspect

    from symbio import cuda_lora

    source = inspect.getsource(cuda_lora.main)
    assert "resume_adapter_file" in source
    assert "from_pretrained" in source


# ---- the corpus backup that nothing read ----

def test_an_interrupted_weighted_run_is_undone_on_the_next_one(tmp_path, monkeypatch):
    """The finally block does not run on SIGKILL — the OOM kill this project
    takes regularly — so train.jsonl was left holding the duplicated lines and
    .preweight was left holding the original, unconsulted. The next run then
    weighted the already weighted corpus."""
    from symbio import constants

    train = tmp_path / "train.jsonl"
    train.write_text('{"a": 1}\n{"a": 1}\n{"a": 1}\n{"b": 2}\n')      # expanded
    train.with_suffix(".jsonl.preweight").write_text('{"a": 1}\n{"b": 2}\n')
    monkeypatch.setattr(constants, "TRAIN_FILE", train)

    with training.weighted_corpus(train, [3.0, 1.0]) as path:
        assert len(path.read_text().splitlines()) == 4                # 3x + 1x

    assert train.read_text() == '{"a": 1}\n{"b": 2}\n'                # restored
    assert not train.with_suffix(".jsonl.preweight").exists()


def test_a_clean_run_still_restores_and_cleans_up(tmp_path, monkeypatch):
    from symbio import constants

    train = tmp_path / "train.jsonl"
    train.write_text('{"a": 1}\n{"b": 2}\n')
    monkeypatch.setattr(constants, "TRAIN_FILE", train)

    with training.weighted_corpus(train, [2.0, 1.0]):
        pass

    assert train.read_text() == '{"a": 1}\n{"b": 2}\n'
    assert not train.with_suffix(".jsonl.preweight").exists()


# ---- the skin must stay off the model's own words ----

def test_style_line_rewrites_tagged_lines():
    """Established behaviour, and exactly why it must not see a reply: an
    answer containing "[Note] ..." — prose, or inside a fenced code block —
    came out lowercased, bulleted and re-indented."""
    styled = chat_style.style_line("[Note] remember to seal it", color=True)

    assert "· note" in styled
    assert "[Note]" not in styled
