"""The main model's own eyes, when it has them.

Qwen3.5 checkpoints — 1-bit Bonsai 27B among them — are vision-language
models: the language model plus a ~0.9 GB vision tower. mlx_lm loads them
text-only and drops the tower, which is right for chat. For a look, Symbio
swaps a separate VLM in (symbio/vision.py): sleep the main model, load the
VLM, look, free it, reload the main model. On a 16 GB Mac that swap is the
whole cost of seeing, and with a vision-language main model it buys nothing —
the eyes were in the main model's own checkpoint all along.

So a look builds mlx-vlm's model AROUND the resident model instead of loading
a second one. mlx_vlm.utils.load_model(lazy=True) constructs the full model
without reading a byte; then every projection in its language model is
replaced by the resident model's own module object (the same arrays, and any
LoRA adapter the main model wears), the remaining language tensors are
assigned from the resident model, and only then is anything evaluated — the
vision tower, the one part not already in memory. When the look is over the
tower is dropped and the main model is exactly as it was.

This works only where mlx-vlm's language model has mlx_lm's parameter names,
and that is checked rather than assumed: after the graft, every parameter of
mlx-vlm's language model must BE one of the resident model's arrays. A name
that matched nothing would otherwise leave a randomly initialised tensor in a
model that runs and describes the screen wrong.
"""

from __future__ import annotations

import contextlib
import json
import struct
import threading
import weakref
from pathlib import Path
from typing import Any

# mlx-vlm model types whose language model was checked against mlx_lm's.
SUPPORTED = ("qwen3_5",)
# Ternary-Bonsai-2 (symbio/app/prism_pack.py): a qwen3_5 in a rotated basis,
# whose mlx-vlm model prism_pack builds itself.
PRISM_PACK = "prism_hadamard_qwen35"

_lock = threading.Lock()
# The name each model was loaded under -> (weakref to it, its local directory).
_resident: dict[str, tuple[Any, str]] = {}
_installed = False


def local_path(name: str) -> Path | None:
    """The local directory of a model name, without touching the network."""
    path = Path(name).expanduser()
    if path.is_dir():
        return path
    try:
        from huggingface_hub import try_to_load_from_cache

        hit = try_to_load_from_cache(name, "config.json")
    except Exception:
        return None
    return Path(hit).parent if isinstance(hit, str) else None


def _tensor_names(path: Path) -> list[str]:
    index = path / "model.safetensors.index.json"
    if index.is_file():
        try:
            return list(json.loads(index.read_text()).get("weight_map", {}))
        except (OSError, ValueError):
            pass
    names: list[str] = []
    for file in sorted(path.glob("*.safetensors")):
        try:
            with open(file, "rb") as f:
                (size,) = struct.unpack("<Q", f.read(8))
                header = json.loads(f.read(size))
        except (OSError, ValueError, struct.error):
            continue
        names.extend(k for k in header if k != "__metadata__")
    return names


def has_vision_tower(path: str | Path) -> bool:
    """A supported vision-language checkpoint whose files hold the tower.

    The config alone is not enough: text-only conversions keep vision_config
    and drop the tensors.
    """
    path = Path(path)
    try:
        config = json.loads((path / "config.json").read_text())
    except (OSError, ValueError):
        return False
    model_type = config.get("model_type")
    if model_type == PRISM_PACK:
        model_type = config.get("base_model_type")
    if model_type not in SUPPORTED or not config.get("vision_config"):
        return False
    return any(n.startswith(("vision_tower.", "model.visual.")) for n in _tensor_names(path))


def remember(model: Any, name: str) -> bool:
    """Note a model mlx_lm just loaded, if its checkpoint carries eyes."""
    path = local_path(str(name))
    if path is None or not has_vision_tower(path):
        return False
    with _lock:
        _resident[str(name)] = (weakref.ref(model), str(path))
    return True


def resident(name: str | None) -> tuple[Any, str] | None:
    """(model, local directory) for the model loaded under `name`, while
    something still holds it and its checkpoint has a vision tower."""
    if not name:
        return None
    with _lock:
        for key, (ref, _path) in list(_resident.items()):
            if ref() is None:
                del _resident[key]
        entry = _resident.get(str(name))
    if entry is None:
        return None
    model = entry[0]()
    return (model, entry[1]) if model is not None else None


def _is_projection(module: Any) -> bool:
    import mlx.nn as nn

    if isinstance(module, (nn.Linear, nn.QuantizedLinear, nn.Embedding, nn.QuantizedEmbedding)):
        return True
    # LoRA wrappers (mlx_lm's LoRALinear and friends) carry their adapter in
    # lora_a/lora_b around the base layer; other layer kinds that stand in for
    # a projection (a Prism pack's Packed) say so.
    return (hasattr(module, "lora_a") and hasattr(module, "lora_b")) or bool(
        getattr(type(module), "is_projection", False))


def _module_at(root: Any, path: str):
    parent = root
    parts = path.split(".")
    for part in parts[:-1]:
        parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
    return parent, parts[-1]


def graft(host_lm: Any, vl_lm: Any) -> None:
    """Make vl_lm compute with host_lm's own modules and arrays.

    Projections move across as the same module objects (OneBitLinear, a LoRA
    wrapper — whatever the host runs); everything else (norms, convolutions,
    decay rates) is assigned array by array, so mlx-vlm keeps its own layer
    classes around them.
    """
    from mlx.utils import tree_flatten

    moved: list[str] = []
    for name, module in host_lm.named_modules():
        if not name or any(name.startswith(m + ".") for m in moved):
            continue
        if _is_projection(module):
            parent, leaf = _module_at(vl_lm, name)
            setattr(parent, leaf, module)
            module.shared_with_eyes = True  # see _guard_fused_decode
            moved.append(name)
    host_params = dict(tree_flatten(host_lm.parameters()))
    rest = [(k, v) for k, v in host_params.items()
            if not any(k.startswith(m + ".") for m in moved)]
    vl_lm.load_weights(rest, strict=False)
    shared = {id(v) for v in host_params.values()}
    strays = [k for k, v in tree_flatten(vl_lm.parameters()) if id(v) not in shared]
    if strays:
        raise ValueError(
            f"mlx-vlm's language model has {len(strays)} parameter(s) the resident "
            f"model does not, e.g. {strays[:3]}; refusing to look with weights that "
            f"were never loaded.")


def _last_position_head(head: Any):
    """The output head, applied to the last position only.

    mlx-vlm's language model projects EVERY prompt position onto the
    vocabulary and generation keeps one row: for a 1,000-token screenshot that
    is a [1000, 248320] logit array, most of a gigabyte, built to be thrown
    away. Generation never reads another row, so the look's copy of the head
    projects only the last one. The main model's own head is untouched.
    """
    import mlx.nn as nn

    class LastPosition(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def __call__(self, x):
            if x.ndim == 3 and x.shape[1] > 1:
                x = x[:, -1:, :]
            return self.inner(x)

    return LastPosition(head)


# Screens are shown to the main model at most this many pixels. A 27B reads a
# 1280x800 screenshot as ~1,000 image tokens, 20 s of prefill per question on
# the M4; a Retina capture would be four times that, in time and in memory.
# Grounding stays in the image's own 0-1000 frame, so a smaller copy changes
# nothing downstream — only how small a control can still be found.
MAX_PIXELS = 1280 * 1024


def _guard_fused_decode() -> None:
    """Keep mlx-vlm's fused decode path off the shared projections.

    mlx-vlm 0.6.3 decodes a Qwen3.5 linear-attention layer by concatenating
    its four input projections into one quantized matmul. Two things go wrong
    when those projections are the main model's own: the concatenated copy is
    cached as an attribute ON the first projection, so it outlives the look
    as a second copy of those weights; and it calls mx.quantized_matmul with
    the layer's bit width directly, which for 1-bit Bonsai's layers is a
    kernel MLX does not have ("Unable to load kernel
    affine_qmv_fast_float16_t_gs_128_b_1"). Shared layers take the plain path,
    one projection at a time, through their own __call__.
    """
    import mlx.nn as nn
    from mlx_vlm.models.qwen3_5 import language

    original = language._decode_quantized_linears_fused
    if getattr(original, "symbio_guarded", False):
        return

    def guarded(linears, x):
        if any(type(linear) is not nn.QuantizedLinear
               or getattr(linear, "shared_with_eyes", False) for linear in linears):
            return None
        return original(linears, x)

    guarded.symbio_guarded = True
    language._decode_quantized_linears_fused = guarded


def open_eyes(name: str):
    """(mlx-vlm model, processor) over the resident model loaded as `name`."""
    found = resident(name)
    if found is None:
        raise LookupError(f"No resident model loaded as {name!r} has a vision tower.")
    host, path = found
    from symbio.mlx_gate import attr

    mx = attr("mlx.core")
    utils = attr("mlx_vlm.utils")
    _guard_fused_decode()
    if getattr(host, "prism_pack_path", None):
        from symbio.app import prism_pack

        vl, processor = prism_pack.load_vl_pack(path, host=host)
    else:
        # lazy=True: the whole model is built and nothing is read. The
        # language half is about to be replaced by the resident one, so it
        # never will be.
        vl = utils.load_model(Path(path), lazy=True)
        graft(host.language_model, vl.language_model)
        mx.eval(vl.vision_tower.parameters())
        vl.eval()
        processor = utils.load_processor(
            Path(path), True, eos_token_ids=getattr(vl.config, "eos_token_id", None))
        image_processor = utils.load_image_processor(Path(path))
        if image_processor is not None:
            processor.image_processor = image_processor
    if "lm_head" in vl.language_model:
        vl.language_model.lm_head = _last_position_head(vl.language_model.lm_head)
    _cap_pixels(processor, MAX_PIXELS)
    return vl, processor


# How far past its own weights a look may take MLX's allocator. Measured with
# 1-bit Bonsai 27B on the 16 GB M4, a 1280x800 screen (1,021 tokens): without
# a ceiling the prefill peaked 3.6 GB over the main model, because finished
# layers' scratch (1-bit weights respread to 2-bit for the matmul) is only
# released when the GPU retires the command buffer that used it; under this
# ceiling the allocator waits for that instead, and the peak was 2.5 GB over,
# in the same 18 s. Unlimited, the run that drove free memory to 14% ended in a
# Metal "GPU Timeout Error" instead of an answer.
LOOK_HEADROOM_GB = 1.0


@contextlib.contextmanager
def memory_ceiling(vl_model: Any):
    """Hold MLX's allocator to the look's weights plus LOOK_HEADROOM_GB."""
    from mlx.utils import tree_flatten

    from symbio.mlx_gate import attr

    mx = attr("mlx.core")
    weights = sum(v.nbytes for _, v in tree_flatten(vl_model.parameters()))
    old = mx.set_memory_limit(int(weights + LOOK_HEADROOM_GB * 2**30))
    try:
        yield
    finally:
        mx.set_memory_limit(old)


def _cap_pixels(processor: Any, max_pixels: int) -> None:
    """Lower the image processor's pixel ceiling (Qwen-VL: size.longest_edge)."""
    image_processor = getattr(processor, "image_processor", None)
    size = getattr(image_processor, "size", None)
    if isinstance(size, dict) and "longest_edge" in size:
        size["longest_edge"] = min(int(size["longest_edge"]), max_pixels)
    if hasattr(image_processor, "max_pixels"):
        image_processor.max_pixels = min(int(image_processor.max_pixels or max_pixels),
                                         max_pixels)


def install() -> bool:
    """Make mlx_lm.load note which loaded models have eyes. Idempotent."""
    global _installed
    if _installed:
        return False
    import mlx_lm
    from mlx_lm import utils

    original = utils.load

    def load(path_or_hf_repo, *args, **kwargs):
        out = original(path_or_hf_repo, *args, **kwargs)
        try:
            remember(out[0], path_or_hf_repo)
        except Exception:
            pass  # never let bookkeeping fail a load
        return out

    load.__wrapped__ = original
    utils.load = load
    mlx_lm.load = load
    _installed = True
    return True
