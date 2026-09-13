"""The backend seam, tested to the edge of what a machine without CUDA can see.

Everything here runs on the Mac this was written on. What it can prove:
selection, the trainer's argv, the dataset shaping and prompt masking the CUDA
trainer does before it touches a GPU, and that every CUDA entry point fails
with a message that says what to install. What it cannot prove is that a
training step or a generation on an NVIDIA card produces anything — no such
device was present, and no test here pretends otherwise.

The most important tests in this file are the ones asserting the MLX path is
UNCHANGED. A backend abstraction that quietly alters the working setup would
be a bad trade for a path nobody has run.
"""

import sys

import pytest

from symbio import backend, cuda_lora


# ---- selection ----

def test_apple_silicon_detects_mlx():
    assert backend.detect() == backend.MLX


def test_config_overrides_detection():
    assert backend.selected({"backend": "cuda"}) == backend.CUDA
    assert backend.selected({"backend": "mlx"}) == backend.MLX


def test_the_environment_overrides_detection(monkeypatch):
    monkeypatch.setenv("SYMBIO_BACKEND", "cuda")
    assert backend.selected() == backend.CUDA


def test_config_beats_the_environment(monkeypatch):
    """An explicit setting in the file the user edited wins over an
    environment variable they may have exported months ago."""
    monkeypatch.setenv("SYMBIO_BACKEND", "cuda")
    assert backend.selected({"backend": "mlx"}) == backend.MLX


def test_an_unknown_backend_falls_back_and_says_so(capsys):
    assert backend.selected({"backend": "tpu"}) == backend.detect()
    assert "tpu" in capsys.readouterr().out


def test_an_empty_setting_is_not_an_error(capsys):
    assert backend.selected({"backend": ""}) == backend.detect()
    assert capsys.readouterr().out == ""


def test_describe_names_the_machine():
    assert "mlx" in backend.describe({})


# ---- the MLX path must be exactly what it was ----

def test_the_mlx_trainer_argv_is_unchanged():
    """This is the command that has trained every adapter in this project.
    The seam exists to add a second one, not to alter this."""
    lora = {"batch_size": 1, "num_layers": 2, "learning_rate": 1e-4,
            "steps_per_eval": 30, "val_batches": 8, "max_seq_length": 2048,
            "save_every": 15}
    cmd = backend.trainer_command("some/model", "/data", "/adapters", lora,
                                  150, "/cfg.yaml", {"backend": "mlx"})
    assert cmd[:4] == [sys.executable, "-m", "mlx_lm", "lora"]
    assert cmd[4:8] == ["--model", "some/model", "--train", "--data"]
    for flag, value in [("--batch-size", "1"), ("--num-layers", "2"),
                        ("--iters", "150"), ("--val-batches", "8"),
                        ("--max-seq-length", "2048"),
                        ("--adapter-path", "/adapters"),
                        ("--save-every", "15"), ("--config", "/cfg.yaml")]:
        assert cmd[cmd.index(flag) + 1] == value, flag


def test_the_mlx_path_does_not_import_torch(monkeypatch):
    """Loading on Apple Silicon must not drag the CUDA stack in — the lazy
    import discipline here is what keeps `symb chat` starting quickly."""
    def _boom(*a, **k):
        raise AssertionError("the mlx path must not require torch")

    monkeypatch.setattr(backend, "_require_torch", _boom)
    assert backend.selected({"backend": "mlx"}) == backend.MLX


# ---- the CUDA trainer's command line ----

def test_the_cuda_trainer_takes_the_same_flags():
    """run_training parses this process's stdout and manages the adapter
    directory around it. Matching mlx_lm's interface is what keeps all of
    that free of a second code path."""
    lora = {"batch_size": 2, "num_layers": 4, "learning_rate": 2e-4,
            "steps_per_eval": 10, "val_batches": 4, "max_seq_length": 1024,
            "save_every": 20, "rank": 16, "dropout": 0.05, "scale": 20.0,
            "keys": ["self_attn.q_proj", "self_attn.v_proj"]}
    cmd = backend.trainer_command("m", "/d", "/a", lora, 99, "/c.yaml",
                                  {"backend": "cuda"})
    assert cmd[:3] == [sys.executable, "-m", "symbio.cuda_lora"]
    for flag in ("--model", "--data", "--adapter-path", "--batch-size",
                 "--num-layers", "--iters", "--learning-rate",
                 "--steps-per-eval", "--val-batches", "--max-seq-length",
                 "--save-every"):
        assert flag in cmd, flag
    assert cmd[cmd.index("--iters") + 1] == "99"
    assert cmd[cmd.index("--rank") + 1] == "16"


def test_the_cuda_trainer_parses_what_the_backend_emits():
    """The two halves have to agree, and nothing else checks that they do."""
    lora = {"batch_size": 2, "num_layers": 4, "learning_rate": 2e-4,
            "steps_per_eval": 10, "val_batches": 4, "max_seq_length": 1024,
            "save_every": 20, "rank": 16, "dropout": 0.05, "scale": 20.0,
            "keys": ["self_attn.q_proj"]}
    cmd = backend.trainer_command("m", "/d", "/a", lora, 99, "/c.yaml",
                                  {"backend": "cuda"})
    args = cuda_lora.parse_args(cmd[3:])
    assert args.model == "m" and args.iters == 99 and args.rank == 16
    assert args.adapter_path == "/a" and args.max_seq_length == 1024


# ---- the trainer's pure parts ----

def test_lora_keys_map_onto_peft_module_names():
    """config.lora.keys holds MLX paths; peft matches bare module names."""
    assert cuda_lora.lora_target_modules(
        "self_attn.q_proj,self_attn.v_proj") == ["q_proj", "v_proj"]


def test_duplicate_projections_collapse():
    assert cuda_lora.lora_target_modules(
        "layers.0.self_attn.q_proj,layers.1.self_attn.q_proj") == ["q_proj"]


def test_no_keys_falls_back_to_attention():
    assert cuda_lora.lora_target_modules("") == ["q_proj", "v_proj"]


def test_the_prompt_boundary_is_the_last_assistant_turn():
    """Masking the wrong half is how 99% of the loss became the system prompt
    on the MLX side; the boundary is the model's own final turn."""
    text = ("<|im_start|>system\nrules<|im_end|>\n"
            "<|im_start|>user\nhi<|im_end|>\n"
            "<|im_start|>assistant\nhello")
    cut = cuda_lora.split_index(text)
    assert text[:cut].endswith("<|im_start|>assistant")
    assert "hello" in text[cut:]


def test_text_with_no_marker_trains_on_everything():
    """0 means "mask nothing", which is what mlx_lm does unmasked. Guessing a
    boundary would be worse than not masking."""
    assert cuda_lora.split_index("just a plain string") == 0


def test_labels_mask_the_prompt_and_keep_the_reply():
    class _Tok:
        def __call__(self, text, truncation=False, max_length=None):
            return {"input_ids": list(range(len(text.split())))}

    text = "a b c <|im_start|>assistant d e"
    row = cuda_lora.encode(text, _Tok(), max_len=64)
    assert row["labels"][0] == -100
    assert row["labels"][-1] != -100
    assert len(row["labels"]) == len(row["input_ids"])
    assert len(row["attention_mask"]) == len(row["input_ids"])


def test_a_row_is_never_masked_end_to_end():
    """A sample whose prompt fills the window would train on nothing and
    contribute a NaN loss."""
    class _Tok:
        def __call__(self, text, truncation=False, max_length=None):
            return {"input_ids": [1, 2, 3]}

    row = cuda_lora.encode("x <|im_start|>assistant y", _Tok(), max_len=3)
    assert any(v != -100 for v in row["labels"])


def test_masking_can_be_turned_off():
    class _Tok:
        def __call__(self, text, truncation=False, max_length=None):
            return {"input_ids": list(range(len(text.split())))}

    row = cuda_lora.encode("a <|im_start|>assistant b", _Tok(), max_len=64,
                           mask_prompt=False)
    assert all(v != -100 for v in row["labels"])


def test_reading_a_corpus_skips_blank_and_malformed_rows(tmp_path):
    p = tmp_path / "train.jsonl"
    p.write_text('{"text": "one"}\n\nnot json\n{"text": ""}\n{"text": "two"}\n')
    assert cuda_lora.read_jsonl(p) == ["one", "two"]


def test_a_missing_corpus_is_empty_rather_than_an_error(tmp_path):
    assert cuda_lora.read_jsonl(tmp_path / "nope.jsonl") == []


# ---- failure messages have to say what to install ----

def test_torch_missing_names_what_to_install(monkeypatch):
    real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) \
        else __builtins__.__import__

    def _no_torch(name, *a, **k):
        if name == "torch":
            raise ImportError("No module named 'torch'")
        return real_import(name, *a, **k)

    monkeypatch.setattr("builtins.__import__", _no_torch)
    with pytest.raises(backend.BackendUnavailable) as excinfo:
        backend._require_torch()
    assert "pytorch.org" in str(excinfo.value)


def test_the_cuda_trainer_exits_cleanly_without_its_dependencies(capsys):
    """Run as a subprocess it must not traceback: run_training reads its exit
    code and its output."""
    rc = cuda_lora.main([
        "--model", "m", "--data", "/nonexistent", "--adapter-path", "/tmp/a",
        "--train",
    ])
    assert rc == 2
    assert "cuda" in capsys.readouterr().err.lower()


def test_the_last_n_layers_are_selected():
    class _Cfg:
        num_hidden_layers = 32

    class _Model:
        config = _Cfg()

    assert cuda_lora._last_n_layers(_Model(), 2) == [30, 31]
    assert cuda_lora._last_n_layers(_Model(), 0) is None
    assert cuda_lora._last_n_layers(_Model(), 99) is None


def test_progress_is_printed_in_the_shape_the_caller_parses(capsys):
    """run_training reads 'Iter N: train loss X' to drive its early stop."""
    class _State:
        global_step = 7

    cb = cuda_lora._ProgressInMlxShape()
    cb.on_log(None, _State(), None, logs={"loss": 1.25, "learning_rate": 1e-4})
    cb.on_log(None, _State(), None, logs={"eval_loss": 0.5})
    out = capsys.readouterr().out
    assert "Iter 7: train loss 1.250" in out
    assert "Val loss 0.500" in out


def test_unknown_callbacks_are_absorbed():
    """transformers calls a wide callback surface; missing one must not crash
    a training run."""
    cb = cuda_lora._ProgressInMlxShape()
    cb.on_train_begin(None, None, None)
    cb.on_step_end(None, None, None)


# ---- the two stacks must not meet ----

def test_generation_goes_through_the_seam_by_default(monkeypatch, tmp_path):
    """The seam was wired one way: only load() and trainer_command() went
    through it, while generation called mlx_lm directly — so selecting cuda
    produced a transformers model that mlx_lm was then handed."""
    from symbio import backend as backend_mod
    from symbio.app.chat import ChatSession

    import inspect

    # Read the defaults off the constructor rather than building a session:
    # what matters is which function a caller that passes nothing ends up with.
    source = inspect.getsource(ChatSession.__init__)

    assert "else backend.generate" in source
    assert "else backend.stream_generate" in source
    assert backend_mod.generate is not None


def test_the_mlx_only_features_are_off_on_cuda():
    """KV quantisation, the draft model, the boot prefill and logits
    processors are all MLX machinery — make_prompt_cache, generate_step,
    mx.array. On CUDA they must be absent, not broken."""
    from symbio import backend as backend_mod
    from symbio.app.chat import ChatSession

    session = ChatSession.__new__(ChatSession)
    session.stream_fn = backend_mod.stream_generate
    session.config = {"backend": "cuda", "agent": {}}

    assert session._mlx_generation() is False


def test_they_stay_on_for_mlx():
    from symbio import backend as backend_mod
    from symbio.app.chat import ChatSession

    session = ChatSession.__new__(ChatSession)
    session.stream_fn = backend_mod.stream_generate
    session.config = {"backend": "mlx", "agent": {}}

    assert session._mlx_generation() is True


def test_an_injected_fake_is_not_mistaken_for_a_real_generator():
    """Front-ends and tests inject their own callables, and every path behind
    this gate passes mlx-specific arguments those fakes never took."""
    from symbio.app.chat import ChatSession

    session = ChatSession.__new__(ChatSession)
    session.stream_fn = lambda *a, **k: iter(())
    session.config = {"backend": "mlx", "agent": {}}

    assert session._mlx_generation() is False
