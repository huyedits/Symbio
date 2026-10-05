"""The ROCm port: what holds without a device, and what the AMD box answers.

The CUDA/ROCm path in backend.py and cuda_lora.py has never executed on real
hardware — written on a Mac, kept importable by construction. These tests
hold the parts a port must not silently change: detection prefers truth,
quantized loading asks before it assumes, and the trainer's command line
stays mlx_lm-shaped so run_training's parser never grows a branch.

ROCm specifics pinned here:
- torch on ROCm exposes HIP through torch.cuda; backend detection needs no
  change, and describe() must say which stack it found.
- bitsandbytes on ROCm is build-dependent: quantization_available() is the
  probe that keeps a missing NF4 capability from dying inside transformers
  with an error that names CUDA.
- Nothing here may request flash_attention_2; SDPA is the portable kernel.
"""

import importlib.machinery
import sys

import pytest

from symbio import backend


# ---- detection: truth first, config beats env beats detection ----

def test_rocm_torch_reports_through_the_cuda_namespace(monkeypatch):
    """A ROCm torch build answers torch.cuda.is_available() True; selection
    must not look for an NVIDIA name and refuse it."""
    import types
    fake = types.ModuleType("torch")
    fake.cuda = types.SimpleNamespace(
        is_available=lambda: True,
        is_bf16_supported=lambda: True,
        get_device_name=lambda i: "AMD Radeon AI PRO R9700",
    )
    fake.__version__ = "2.9.0+rocm7.0"
    fake.__spec__ = importlib.machinery.ModuleSpec("torch", None)
    monkeypatch.setitem(sys.modules, "torch", fake)
    assert backend._rocm() is True
    assert backend.detect() in (backend.CUDA, backend.MLX)


def test_describe_names_the_stack_a_rocm_build_is_running(monkeypatch):
    import types
    fake = types.ModuleType("torch")
    fake.cuda = types.SimpleNamespace(
        is_available=lambda: True,
        get_device_name=lambda i: "AMD Instinct MI355X",
        get_device_properties=lambda i: types.SimpleNamespace(
            total_memory=192e9),
    )
    fake.__version__ = "2.9.0+rocm7.0"
    monkeypatch.setitem(sys.modules, "torch", fake)
    line = backend.describe({"backend": "cuda"})
    assert "MI355X" in line and "192 GB" in line


# ---- quantization: ask before assuming ----

def test_missing_bitsandbytes_on_rocm_names_the_real_problem(monkeypatch):
    import types
    fake = types.ModuleType("torch")
    fake.cuda = types.SimpleNamespace(is_available=lambda: True,
                                      is_bf16_supported=lambda: True)
    fake.__version__ = "2.9.0+rocm6.2"
    monkeypatch.setitem(sys.modules, "torch", fake)
    monkeypatch.setitem(sys.modules, "bitsandbytes", None)  # import fails
    assert backend.quantization_available({"cuda": {"load_in_bits": 4}}) is False
    assert backend.quantization_available({"cuda": {"load_in_bits": 16}}) is True


def test_a_load_refusal_says_what_to_set_not_what_broke(monkeypatch):
    """The one sentence a person on an AMD box needs: which key unblocks."""
    import importlib.machinery
    import types
    fake = types.ModuleType("torch")
    fake.cuda = types.SimpleNamespace(is_available=lambda: True,
                                      is_bf16_supported=lambda: True)
    fake.__version__ = "2.9.0+rocm6.2"
    fake.__spec__ = importlib.machinery.ModuleSpec("torch", None)
    monkeypatch.setitem(sys.modules, "torch", fake)
    monkeypatch.setitem(sys.modules, "bitsandbytes", None)
    with pytest.raises(backend.BackendUnavailable) as e:
        backend.load("m", config={"backend": "cuda", "cuda": {"load_in_bits": 4}})
    assert "cuda.load_in_bits" in str(e.value)


# ---- the port's only hard trap, kept open ----

def test_nothing_requests_flash_attention_2():
    """flash_attention_2 is CUDA-only; on ROCm it is the port's one real
    crash. The transformer-loading code must never PASS it as an
    attn_implementation value — SDPA is the portable kernel (MLX's own flash
    attention is a different, already-guarded thing). Strings inside
    comments and docstrings survive; only executable kwarg values fail."""
    import ast as astmod
    import pathlib

    root = pathlib.Path(__file__).parent.parent
    for mod in ("backend.py", "cuda_lora.py"):
        tree = astmod.parse((root / "symbio" / mod).read_text())
        for node in astmod.walk(tree):
            if isinstance(node, astmod.keyword) and node.arg == "attn_implementation":
                raise AssertionError(
                    f"{mod} passes attn_implementation — change to SDPA default")
            if (isinstance(node, astmod.Call)
                    and any(getattr(k, "attr", "") == "from_pretrained"
                            for k in [node.func])):
                for kw in node.keywords:
                    if kw.arg == "attn_implementation":
                        raise AssertionError(
                            f"{mod}::from_pretrained sets attn_implementation")


# ---- the trainer command stays mlx_lm-shaped on every backend ----

def test_rocm_trainer_argv_is_the_cuda_argv():
    """run_training's stdout parser, early stop and adapter handling read the
    trainer's flags; a ROCm run must emit exactly what the CUDA path emits,
    or the caller grows a backend branch for nothing."""
    lora = {"batch_size": 1, "num_layers": 2, "learning_rate": 1e-4,
            "steps_per_eval": 30, "val_batches": 8, "max_seq_length": 2048,
            "save_every": 15}
    cmd = backend.trainer_command("q", "/data", "/adapters", lora, 150,
                                  "/cfg.yaml", {"backend": "cuda"})
    assert cmd[1:3] == ["-m", "symbio.cuda_lora"]
    assert "--iters" in cmd and "150" in cmd
    assert "--config" in cmd