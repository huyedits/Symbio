"""Prism's Hadamard packs (Ternary-Bonsai-2), loaded like any other model.

prism-ml/Ternary-Bonsai-2-27B-mlx-2bit is Qwen3.8-27B in ternary weights:
98.2% of the FP16 model's score across Prism's 14 thinking-mode benchmarks, in
7.67 GB of language model — the only way that model fits a 16 GB Mac (the
plain MLX 4-bit is ~15 GB). It declares `model_type:
prism_hadamard_qwen35`, so stock mlx_lm refuses it ("Model type ... not
supported"), and it would be wrong to make mlx_lm load it as a plain Qwen3.5
2-bit checkpoint even if it could. The weights are stored in a rotated basis:
every projection expects its input multiplied by a sign vector and run through
a 1,024-wide Walsh-Hadamard transform first, and the embedding table hands back
rows that need the inverse. Skip that and the model loads fine and writes
garbage.

The pack ships its own loader (runtime/*.py) for exactly this reason. Symbio
does not execute it: this is the same arithmetic, written here, on mlx_lm's own
qwen3_5 model, so a pack comes back from mlx_lm.load() as an ordinary model —
the chat loop, the prompt cache, speculative decoding and the trainer all see a
Qwen3.5. The differences from the bundled loader are deliberate:

  * Only the language model is built. Schema-2 packs carry a 0.9 GB vision
    tower too; it is read only for a look (symbio/app/eyes.py), and dropped
    again after it.
  * The packed projections take a LoRA adapter (`to_lora`), which mlx_lm's
    tuner picks up by that name — the bundled runtime has no way to fine-tune.

The modules the pack lists in config.json are the only ones replaced, each
checked against the shapes the model expects before it is installed.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn

MODEL_TYPE = "prism_hadamard_qwen35"
_SCHEMAS = (1, 2)
_BLOCKS = (512, 1024, 2048, 4096)
_BITS = 2
_GROUP = 128

_installed = False


def is_pack(path: str | Path) -> bool:
    """True for a directory holding a Prism Hadamard pack."""
    try:
        config = json.loads((Path(path) / "config.json").read_text())
    except (OSError, ValueError):
        return False
    return config.get("model_type") == MODEL_TYPE


class Packed(nn.Module):
    """A 2-bit affine projection (or embedding) stored in a rotated basis.

    Linear: y = W' (H (s * x)), where W' is the folded weight, s the sign
    vector and H the normalised Hadamard transform over `block`-wide pieces.
    Embedding: row = s * (H w'), the inverse of the same rotation.
    """

    bits = _BITS
    group_size = _GROUP
    mode = "affine"
    # eyes.graft moves layers with this mark across as they are (see there).
    is_projection = True

    def __init__(self, weight, scales, biases, block: int = 0, signs=None,
                 embedding: bool = False, dtype=mx.float16):
        super().__init__()
        self.weight = weight
        self.scales = scales
        self.biases = biases
        if block:
            self.signs = signs
        self.block = block
        self.embedding = embedding
        self.dtype = dtype

    @property
    def input_dims(self) -> int:
        return self.weight.shape[1] * 32 // self.bits

    @property
    def output_dims(self) -> int:
        return self.weight.shape[0]

    def rotate(self, x, inverse: bool = False):
        """The activation transform the folded weights were made for."""
        shape, dtype = x.shape, x.dtype
        x = x.astype(mx.float32)
        if not inverse:
            x = x * self.signs
        x = mx.hadamard_transform(x.reshape(-1, self.block),
                                  scale=1 / math.sqrt(self.block)).reshape(shape)
        if inverse:
            x = x * self.signs
        return x.astype(dtype)

    def _matmul(self, x, weight, scales, biases):
        if self.block:
            x = self.rotate(x)
        return mx.quantized_matmul(x, weight, scales, biases, transpose=True,
                                   group_size=self.group_size, bits=self.bits)

    def __call__(self, x):
        if not self.embedding:
            return self._matmul(x, self.weight, self.scales, self.biases)
        shape = x.shape
        rows = x.reshape(-1)
        out = mx.dequantize(self.weight[rows], self.scales[rows], self.biases[rows],
                            group_size=self.group_size, bits=self.bits)
        out = out.reshape(*shape, -1).astype(self.dtype)
        return self.rotate(out, inverse=True) if self.block else out

    def as_linear(self, x):
        """Tied-head logits. x . (s * H w) == H (s * x) . w, the linear path."""
        return self._matmul(x, self.weight, self.scales, self.biases)

    def subset(self, rows):
        """hidden -> logits over `rows` of this head only (symbio.trim_lora)."""
        weight, scales, biases = self.weight[rows], self.scales[rows], self.biases[rows]
        return lambda x: self._matmul(x, weight, scales, biases)

    def to_lora(self, r: int = 8, scale: float = 20.0, dropout: float = 0.0):
        """mlx_lm's tuner calls this for any module that has it."""
        if self.embedding:
            raise ValueError("A packed embedding takes no LoRA adapter; leave "
                             "embed_tokens out of lora.keys.")
        return PackedLoRA(self, r=r, scale=scale, dropout=dropout)


class PackedLoRA(nn.Module):
    """mlx_lm's LoRALinear around a Packed projection, same keys, same maths."""

    def __init__(self, base: Packed, r: int = 8, scale: float = 20.0,
                 dropout: float = 0.0):
        super().__init__()
        self.linear = base
        self.dropout = nn.Dropout(p=dropout)
        self.scale = scale
        bound = 1 / math.sqrt(base.input_dims)
        self.lora_a = mx.random.uniform(low=-bound, high=bound,
                                        shape=(base.input_dims, r))
        self.lora_b = mx.zeros(shape=(r, base.output_dims))

    def __call__(self, x):
        y = self.linear(x)
        z = (self.dropout(x) @ self.lora_a) @ self.lora_b
        return y + (self.scale * z).astype(x.dtype)


def _check_pack(config: dict[str, Any]) -> None:
    if config.get("schema_version") not in _SCHEMAS:
        raise ValueError(f"Prism pack schema {config.get('schema_version')!r} is not "
                         f"one Symbio knows ({', '.join(map(str, _SCHEMAS))}).")
    base = config.get("base_model_type", "qwen3_5")
    if base != "qwen3_5":
        raise ValueError(f"Prism pack for {base!r}: only qwen3_5 packs are supported.")
    quant = config.get("quantization") or {}
    if quant.get("bits") != _BITS or quant.get("group_size") != _GROUP \
            or quant.get("mode", "affine") != "affine":
        raise ValueError(f"Prism pack quantization {quant} is not affine 2-bit/128.")
    layout = config.get("gdn_activation_layout", "grouped")
    if layout != "grouped":
        # The bundled runtime regroups ungrouped linear-attention heads while
        # converting from GGUF; a saved pack is grouped already. Anything else
        # would load and compute the wrong heads.
        raise ValueError(f"Prism pack GDN layout {layout!r} is not 'grouped'.")


def _module_at(root, path: str):
    parent = root
    parts = path.split(".")
    for part in parts[:-1]:
        parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
    return parent, parts[-1]


def _read_weights(model_path: Path) -> dict[str, Any]:
    """Every tensor in the pack, lazily: nothing is read until it is used.

    Schema 1 saved the bare text model; schema 2 the mlx-vlm namespace, text
    under language_model.* beside vision_tower.*. Both come back in the second
    form, so one set of names serves the text and the vision builds.
    """
    weights: dict[str, Any] = {}
    files = sorted(model_path.glob("model*.safetensors"))
    if not files:
        raise FileNotFoundError(f"No safetensors found in {model_path}")
    for file in files:
        weights.update(mx.load(str(file)))
    if not any(k.startswith("language_model.") for k in weights):
        weights = {"language_model." + k: v for k, v in weights.items()}
    return weights


def load_pack(model_path: str | Path, lazy: bool = False, strict: bool = True,
              model_config: dict[str, Any] | None = None):
    """(model, config) for a pack, the pair mlx_lm.utils.load_model returns."""
    from mlx_lm.models import qwen3_5
    from mlx_lm.utils import load_config

    model_path = Path(model_path)
    config = load_config(model_path)
    _check_pack(config)
    text = dict(config["text_config"])
    if model_config:
        text.update(model_config)
    model = qwen3_5.Model(qwen3_5.ModelArgs(model_type="qwen3_5", text_config=text))
    # The vision tower is left out here, unread: a text turn never needs it,
    # and symbio/app/eyes.py reads it on its own when a look does.
    weights = {k: v for k, v in _read_weights(model_path).items()
               if k.startswith("language_model.")}
    _install_packed(model.language_model, config, weights)
    model.eval()
    model.load_weights(list(weights.items()), strict=strict)
    if not lazy:
        mx.eval(model.parameters())
        _check_signs(model)
    # Where the pack lives, for eyes.open_eyes to build its vision half from.
    model.prism_pack_path = str(model_path)
    return model, config


def _check_signs(model) -> None:
    for name, module in model.named_modules():
        signs = getattr(module, "signs", None) if isinstance(module, Packed) else None
        if signs is not None and not bool(mx.all(mx.abs(signs) == 1).item()):
            raise ValueError(f"Prism pack module {name}: sign vector holds values "
                             f"other than +1/-1.")


def _install_packed(lm, config: dict[str, Any], weights: dict[str, Any]) -> None:
    """Replace each module the pack lists, under `lm`, with its Packed form."""
    seen: set[str] = set()
    for record in config.get("modules") or ():
        path = record["path"]
        if path in seen:
            raise ValueError(f"Prism pack lists {path} twice.")
        seen.add(path)
        if record.get("dtype", "float16") != "float16":
            raise ValueError(f"Prism pack module {path}: activation dtype "
                             f"{record.get('dtype')!r} is not float16.")
        block = int(record.get("block") or 0)
        if block and block not in _BLOCKS:
            raise ValueError(f"Prism pack module {path}: block {block} is not one of {_BLOCKS}.")
        key = "language_model." + path
        try:
            arrays = [weights[f"{key}.{suffix}"] for suffix in ("weight", "scales", "biases")]
        except KeyError as e:
            raise ValueError(f"Prism pack module {path} has no tensor {e}.") from None
        signs = weights.get(f"{key}.signs")
        parent, name = _module_at(lm, path)
        original = getattr(parent, name)
        embedding = bool(record.get("embedding"))
        if embedding != isinstance(original, nn.Embedding) or not isinstance(
                original, (nn.Linear, nn.Embedding)):
            raise ValueError(f"Prism pack module {path} is not the kind of layer "
                             f"the model has there ({type(original).__name__}).")
        rows, width = original.weight.shape
        expected = [(rows, width * _BITS // 32), (rows, width // _GROUP), (rows, width // _GROUP)]
        if [tuple(a.shape) for a in arrays] != expected or arrays[0].dtype != mx.uint32:
            raise ValueError(f"Prism pack module {path}: packed shapes "
                             f"{[tuple(a.shape) for a in arrays]} != {expected}.")
        if block:
            if signs is None or tuple(signs.shape) != (width,) or width % block:
                raise ValueError(f"Prism pack module {path}: sign vector missing or "
                                 f"not {width} wide.")
        elif signs is not None:
            raise ValueError(f"Prism pack module {path}: sign vector without a block.")
        setattr(parent, name, Packed(*arrays, block=block, signs=signs,
                                     embedding=embedding, dtype=mx.float16))


# ------------------------------------------------------------------- eyes
#
# A schema-2 pack is a vision-language model: the 27B plus Qwen3.8's own 0.9 GB
# fp16 vision tower, stored unrotated. symbio/app/eyes.py opens it around the
# resident text model; mlx_vlm.utils.load_model cannot build a pack (it does
# not know the model type), so the mlx-vlm model is put together here.


def has_vision(path: str | Path) -> bool:
    """A pack whose config declares the vision tower (eyes checks the tensors)."""
    try:
        config = json.loads((Path(path) / "config.json").read_text())
    except (OSError, ValueError):
        return False
    return (config.get("model_type") == MODEL_TYPE
            and bool((config.get("components") or {}).get("vision")))


def generation_eos_ids(model_path: str | Path) -> list[int]:
    """Every end-of-generation id generation_config.json lists ([] if none)."""
    try:
        eos = json.loads((Path(model_path) / "generation_config.json").read_text()).get(
            "eos_token_id")
    except (OSError, ValueError):
        return []
    if eos is None:
        return []
    return [int(e) for e in eos] if isinstance(eos, list) else [int(eos)]


def load_vl_pack(model_path: str | Path, host: Any = None, lazy: bool = False):
    """(mlx-vlm model, processor) for a pack.

    With `host` — the text model load_pack built from this pack — the language
    model is the host's own (eyes.graft) and only the vision tower is read.
    Without one, the whole pack is loaded.
    """
    from mlx_vlm.models.qwen3_5 import Model, ModelConfig

    from symbio.app import eyes

    model_path = Path(model_path)
    config = json.loads((model_path / "config.json").read_text())
    _check_pack(config)
    if not (config.get("components") or {}).get("vision"):
        raise ValueError(f"{model_path.name} carries no vision tower.")
    weights = _read_weights(model_path)
    model = Model(ModelConfig.from_dict(config))
    # mlx-vlm's prompt helpers key off model_type to place the image tokens;
    # a pack's is deliberately one they do not know.
    model.config.model_type = config.get("base_model_type", "qwen3_5")
    if host is not None:
        if Path(getattr(host, "prism_pack_path", "")) != model_path:
            raise ValueError("The resident model was not loaded from this pack.")
        eyes.graft(host.language_model, model.language_model)
    else:
        _install_packed(model.language_model, config, weights)
        model.language_model.load_weights(
            [(k[len("language_model."):], v) for k, v in weights.items()
             if k.startswith("language_model.")], strict=True)
    model.vision_tower.load_weights(
        [(k[len("vision_tower."):], v) for k, v in weights.items()
         if k.startswith("vision_tower.")], strict=True)
    model.eval()
    if not lazy:
        mx.eval(model.vision_tower.parameters() if host is not None else model.parameters())
    eos = generation_eos_ids(model_path)
    if eos:
        # mlx-vlm's generate() stops on model.config.eos_token_id unless told
        # otherwise; a pack ends turns on <|im_end|> AND <|endoftext|>.
        model.config.eos_token_id = eos
    return model, build_processor(model_path)


def build_processor(model_path: str | Path):
    """The Qwen3-VL processor, built by hand: AutoProcessor resolves classes by
    model_type, which for a pack is a type transformers has never heard of."""
    from mlx_vlm.models.qwen3_5 import Qwen3VLProcessor
    from mlx_vlm.tokenizer_utils import load_tokenizer
    from mlx_vlm.utils import StoppingCriteria
    from transformers import AutoTokenizer
    from transformers.models.qwen2_vl.image_processing_pil_qwen2_vl import (
        Qwen2VLImageProcessorPil,
    )

    model_path = Path(model_path)
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    processor = Qwen3VLProcessor(
        image_processor=Qwen2VLImageProcessorPil.from_pretrained(str(model_path)),
        tokenizer=tokenizer,
        video_processor=None,
        chat_template=(model_path / "chat_template.jinja").read_text(),
    )
    processor.detokenizer = load_tokenizer(model_path, return_tokenizer=False)(tokenizer)
    eos = generation_eos_ids(model_path) or [tokenizer.eos_token_id]
    criteria = StoppingCriteria(eos, tokenizer)
    processor.tokenizer.stopping_criteria = criteria
    processor.stopping_criteria = criteria
    return processor


def install() -> bool:
    """Teach mlx_lm.utils.load_model to load packs. Idempotent."""
    global _installed
    if _installed:
        return False
    from mlx_lm import utils

    original = utils.load_model

    def load_model(model_path, lazy=False, strict=True, model_config=None, *args, **kwargs):
        if is_pack(model_path):
            return load_pack(model_path, lazy=lazy, strict=strict, model_config=model_config)
        return original(model_path, lazy, strict, model_config, *args, **kwargs)

    load_model.__wrapped__ = original
    utils.load_model = load_model
    _installed = True
    return True
