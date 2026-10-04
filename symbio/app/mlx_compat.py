"""Compatibility patches for the installed mlx-lm, applied on first engine use.

Imported first from symbio.app.__init__, so every path that reaches a model —
chat, dispatch, telegram, health, the eval harnesses — gets the patches before
it calls mlx_lm.load(). Each patch states the version it was written against;
delete one once the upstream release makes it redundant.

Nothing here imports the engine at module scope. The patches are applied by
_apply() the first time a model is actually loaded, either because
symbio.mlx_gate (which every engine symbol resolves through) runs its prepare
hook, or because a direct mlx_lm path called ensure() first. That keeps
`import symbio.*` working on a host with no engine at all.
"""
from __future__ import annotations

import importlib as _importlib
from threading import Lock

from symbio.mlx_gate import on_engine_prepare

_done = False
_lock = Lock()


def _apply() -> None:
    """Import the engine and apply every patch exactly once."""
    global _done
    if _done:
        return
    with _lock:
        if _done:
            return

        from mlx_lm import utils as _utils
        from mlx_lm.models import gemma4 as _gemma4

        # --- gemma4_unified (mlx-lm 0.31.3) ---------------------------------
        #
        # Every MLX conversion of Gemma 4 declares `model_type: gemma4_unified`
        # (Gemma4UnifiedForConditionalGeneration — text + vision + audio in one
        # checkpoint). mlx-lm ships gemma4.py and gemma4_text.py but knows
        # nothing about that name, so load() raises "Model type gemma4_unified
        # not supported" before it ever looks at the weights.
        #
        # The existing gemma4 wrapper is already the right target: it builds
        # only the text tower from config["text_config"], and the checkpoint's
        # 1324 text tensors are named `language_model.model.*` — exactly the
        # module path that wrapper produces. So the model loads correctly
        # under the older name.
        _utils.MODEL_REMAPPING["gemma4_unified"] = "gemma4"

        _orig_gemma4_sanitize = _gemma4.Model.sanitize

        def _gemma4_sanitize(self, weights):
            """Drop the multimodal tensors gemma4.py's own skip list misses.

            It discards vision_tower / multi_modal_projector / audio_tower /
            embed_audio / embed_vision, but predates `vision_embedder` — 11
            tensors in a gemma4_unified checkpoint. Left in, they are
            unexpected keys and load_weights() fails on a model that would
            otherwise work.
            """
            weights = {k: v for k, v in weights.items()
                       if not k.startswith("vision_embedder")}
            return _orig_gemma4_sanitize(self, weights)

        _gemma4.Model.sanitize = _gemma4_sanitize

        # --- speculative decoding on a non-trimmable cache (mlx-lm 0.31.3) --
        #
        # mlx-lm refuses speculative decoding outright when the target's prompt
        # cache cannot be trimmed:
        #
        #     ValueError: Speculative decoding requires a trimmable prompt
        #                 cache (got {'ArraysCache'}).
        #
        # That rules out every hybrid linear-attention model — Qwen3.5, and so
        # the Ternary-Bonsai-27B this project runs. Their linear layers keep a
        # running summary of the sequence instead of a per-token record, so a
        # rejected draft cannot be trimmed back out of them the way a KV cache
        # can.
        #
        # It can still be *restored*: snapshot the layer state before drafting
        # and put it back when a draft is rejected. The naive form of that then
        # has to re-run the accepted tokens through the target to catch the
        # cache up, which costs a second pass — and on a bandwidth-bound model
        # a pass is a pass, so it hands most of the speedup straight back
        # (measured 1.08x on Bonsai).
        #
        # So the re-processing is not done as its own pass. The owed tokens
        # are carried forward and ride along on the front of the *next* round's
        # verification pass, which was going to happen anyway. Reading the
        # weights dominates the cost, so a slightly longer input is nearly free
        # and the rewind stops costing anything.
        #
        # Measured, greedy, output verified identical to non-speculative
        # generation:
        #   Qwen3-8B-4bit  + Qwen3-0.6B-4bit    21.1 -> 39.1 tok/s  (1.85x, k=2)
        #   Bonsai-27B     + Qwen3.5-0.8B-4bit  10.7 -> 13.4 tok/s  (1.25x, k=1)
        #
        # Bonsai gains far less, and gets *worse* as the draft deepens (k=4 is
        # 0.92x — slower than not speculating). Speculative decoding assumes
        # verifying k+1 tokens costs about what verifying one costs; that holds
        # for attention, which parallelises over the sequence, but linear
        # attention is a sequential scan, so a longer verify block genuinely
        # costs more. Keep k small on hybrid models.
        #
        # Only engages where upstream would raise. Trimmable caches keep the
        # upstream path untouched; delete this whole block once mlx-lm
        # supports the case.
        import mlx.core as mx

        # `mlx_lm.generate` is shadowed by the function of the same name that
        # mlx_lm/__init__ re-exports, so the submodule has to be fetched
        # directly.
        _gen = _importlib.import_module("mlx_lm.generate")
        from mlx_lm.models import cache as _mlx_cache

        _EMPTY_TOKENS = mx.array([], mx.uint32)

        def _snapshot_cache(caches):
            """Everything needed to put these caches back where they are now."""
            snap = []
            for c in caches:
                if isinstance(c, _mlx_cache.ArraysCache):
                    # The list is copied because the layer replaces entries;
                    # the arrays themselves are never mutated in place.
                    snap.append(("arrays", list(c.cache)))
                else:
                    snap.append(("offset", c.offset))
            return snap

        def _restore_cache(caches, snap):
            for c, (kind, val) in zip(caches, snap):
                if kind == "arrays":
                    c.cache = list(val)
                else:
                    extra = c.offset - val
                    if extra > 0:
                        c.trim(extra)

        def _hybrid_speculative_generate_step(
            prompt, model, draft_model, *, num_draft_tokens=1, max_tokens=256,
            sampler=None, prompt_cache=None, prefill_step_size=2048, **_,
        ):
            y = prompt.astype(mx.uint32)

            if prompt_cache is None:
                model_cache = _mlx_cache.make_prompt_cache(model)
                draft_cache = _mlx_cache.make_prompt_cache(draft_model)
            else:
                model_cache = prompt_cache[: len(model.layers)]
                draft_cache = prompt_cache[len(model.layers):]

            sampler = sampler or (lambda x: mx.argmax(x, axis=-1))

            def _step(m, c, y_in, n_predict=1):
                with mx.stream(_gen.generation_stream):
                    logits = m(y_in[None], cache=c)
                    logits = logits[:, -n_predict:, :].squeeze(0)
                    logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
                    return sampler(logprobs), logprobs

            def _prefill(m, c, y_in):
                # Leaves exactly one token behind: the loop below needs an
                # input token, and everything before it belongs in the cache.
                while y_in.size > 1:
                    n = min(prefill_step_size, y_in.size - 1)
                    m(y_in[:n][None], cache=c)
                    mx.eval([q.state for q in c])
                    y_in = y_in[n:]
                    mx.clear_cache()
                return y_in

            with mx.stream(_gen.generation_stream):
                draft_y = _prefill(draft_model, draft_cache, y)
                y = _prefill(model, model_cache, y)

            # Committed tokens the caches do not hold, because a rejected round
            # was rolled back. They ride along on the front of the next pass.
            pend_model = _EMPTY_TOKENS
            pend_draft = _EMPTY_TOKENS

            ntoks = 0
            while ntoks < max_tokens:
                num_draft = min(max_tokens - ntoks, num_draft_tokens)
                if num_draft <= 0:
                    break

                snap_model = _snapshot_cache(model_cache)
                snap_draft = _snapshot_cache(draft_cache)
                draft_y_start = draft_y

                yd = (mx.concatenate([pend_draft, draft_y])
                      if pend_draft.size else draft_y)
                drafted = []
                for _ in range(num_draft):
                    yd, _ = _step(draft_model, draft_cache, yd)
                    mx.async_eval(yd)
                    drafted.append(yd)
                draft_tokens = mx.concatenate(drafted)

                y_block = mx.concatenate([y, draft_tokens])
                y_in = (mx.concatenate([pend_model, y_block])
                        if pend_model.size else y_block)
                tokens, logprobs = _step(model, model_cache, y_in, num_draft + 1)
                mx.eval(tokens, draft_tokens)
                draft_list = draft_tokens.tolist()
                tok_list = tokens.tolist()

                n = 0
                while n < num_draft:
                    if tok_list[n] != draft_list[n]:
                        break
                    n += 1
                    ntoks += 1
                    yield tok_list[n - 1], logprobs[n - 1], True
                    if ntoks == max_tokens:
                        break
                if ntoks < max_tokens:
                    ntoks += 1
                    yield tok_list[n], logprobs[n], False
                if ntoks >= max_tokens:
                    break

                y = mx.array([tok_list[n]], mx.uint32)
                draft_y = y

                if n == num_draft:
                    # Every draft held. Both caches now contain exactly the
                    # committed sequence, anything owed included, so nothing is
                    # carried forward.
                    pend_model = _EMPTY_TOKENS
                    pend_draft = _EMPTY_TOKENS
                    draft_y = mx.concatenate(
                        [mx.array(draft_list[-1:], mx.uint32), draft_y])
                else:
                    # A draft was wrong. Put both caches back to where the
                    # round started and note what they now owe; the next pass
                    # pays it.
                    _restore_cache(model_cache, snap_model)
                    _restore_cache(draft_cache, snap_draft)
                    pend_model = mx.concatenate([pend_model, y_block[: 1 + n]])
                    pend_draft = mx.concatenate(
                        [pend_draft, draft_y_start, draft_tokens[:n]])

        _orig_speculative_generate_step = _gen.speculative_generate_step

        def _speculative_generate_step(prompt, model, draft_model, **kwargs):
            """Upstream's implementation, except where it would refuse."""
            prompt_cache = kwargs.get("prompt_cache")
            probe = (prompt_cache[: len(model.layers)] if prompt_cache
                     else _mlx_cache.make_prompt_cache(model))
            if _mlx_cache.can_trim_prompt_cache(probe):
                return _orig_speculative_generate_step(prompt, model,
                                                       draft_model, **kwargs)
            if kwargs.get("logits_processors"):
                # agent.py sends a repetition penalty along with the draft
                # model. The hybrid path does not implement the per-position
                # `prev_tokens` bookkeeping that upstream uses to apply
                # processors across a verified block, and guessing at it would
                # quietly change what the sampler does. Correct output without
                # the speedup beats a fast wrong distribution, so this falls
                # back to ordinary decoding rather than raising — the caller
                # asked for a repetition penalty and still gets exactly that.
                kwargs.pop("num_draft_tokens", None)
                kwargs.pop("draft_model", None)
                # The cache handed in is the two models' concatenated; ordinary
                # decoding must see only the target's half, or the model is
                # given more cache entries than it has layers.
                cache_arg = kwargs.get("prompt_cache")
                if cache_arg is not None and len(cache_arg) > len(model.layers):
                    kwargs["prompt_cache"] = cache_arg[: len(model.layers)]
                # generate_step yields (token, logprobs); the speculative
                # contract this function stands in for yields
                # (token, logprobs, from_draft).
                return ((tok, lp, False)
                        for tok, lp in _gen.generate_step(prompt, model, **kwargs))
            return _hybrid_speculative_generate_step(prompt, model,
                                                     draft_model, **kwargs)

        _gen.speculative_generate_step = _speculative_generate_step

        # --- Prism Hadamard packs (mlx-lm 0.31.3) ---------------------------
        #
        # Ternary-Bonsai-2 (Qwen3.8-27B at 98% of its FP16 score, 7.7 GB)
        # declares `model_type: prism_hadamard_qwen35` and stores its weights
        # in a rotated basis, so mlx_lm refuses it, and loading it as a plain
        # Qwen3.5 would only write garbage. load_model now builds such a pack
        # itself (symbio/app/prism_pack.py); every other model takes the
        # upstream path untouched.
        from symbio.app import prism_pack as _prism

        _prism.install()

        # --- Ternel-packed Bonsai 27B (mlx-lm 0.31.3) -------------------------
        #
        # inductiveML/Ternary-Bonsai-27B-mlx-lossless-1.75bpw brings its own
        # model code (config.json model_file), which mlx_lm imports as is. Its
        # packed layers run inference-only Metal kernels; once such a model
        # loads, symbio/app/ternel_pack.py gives them LoRA and a backward so
        # it can be fine-tuned like any other.
        from symbio.app import ternel_pack as _ternel

        _ternel.install()

        # --- a model's own prefill chunk (mlx-lm 0.31.3) --------------------
        #
        # mlx_lm prefills 2,048 tokens at a time. The 27B Qwen3.5 packs (head
        # dim 256: no fused attention kernel serves it) build a whole
        # [heads, chunk, context] fp32 score matrix per chunk at that size: a
        # 4,000-token prompt cost Ternary-Bonsai-2 +6.29 GB and 124 s, against
        # +2.21 GB and 75 s at 256 (prism_pack.PREFILL_STEP). A loaded model
        # that sets `symbio_prefill_step` gets that chunk; every other model,
        # and any caller that names one, keeps what it had.
        _gen.generate_step = model_prefill_step(_gen.generate_step)

        # --- the main model's own eyes (mlx-lm 0.31.3 + mlx-vlm 0.6.3) -----
        #
        # A Qwen3.5 checkpoint (1-bit Bonsai 27B included) is a vision-language
        # model that mlx_lm loads text-only. mlx_lm.load now notes which
        # loaded models still have a vision tower on disk, so a look can run
        # on the resident model through mlx-vlm instead of swapping in a
        # second one (symbio/app/eyes.py). Loading itself is unchanged.
        from symbio.app import eyes as _eyes

        _eyes.install()

        # --- tool results a Qwen3.5-family template recognises -------------
        #
        # Those templates skip user turns that are only a <tool_response>
        # when they look for the user's request; Symbio's observation turns
        # begin "[System observation:", so each read as a new, empty request
        # and Bonsai 27B lost the task after its first tool call. Rendering
        # only (symbio/app/template_compat.py).
        from symbio.app import template_compat as _template_compat

        _template_compat.install()

        # --- flash attention over the 8-bit KV cache (mlx-lm 0.31.3) --------
        #
        # agent.kv_bits sent every prefill through mlx_lm's unfused quantized
        # attention, which builds the full score matrix. Measured on the
        # 14B-3bit headmaster, 6,000-token prompt, kv_bits=8 (2026-09-26):
        #     as shipped   prefill 60.3s, peak +2,595 MB over the weights
        #     fused        prefill 53.3s, peak +1,132 MB — same 24 tokens out
        # See symbio/app/flash_attention.py. SYMBIO_NO_FLASH_ATTENTION=1 opts out.
        import os as _os

        if _os.environ.get("SYMBIO_NO_FLASH_ATTENTION") != "1":
            from symbio.app import flash_attention as _flash

            _flash.install()

        _done = True


def model_prefill_step(original):
    """`original` (mlx_lm's generate_step), prefilling in the chunk a loaded
    model asks for through `symbio_prefill_step` — see _apply."""

    def generate_step(prompt, model, *args, **kwargs):
        step = getattr(model, "symbio_prefill_step", None)
        if "prefill_step_size" not in kwargs and isinstance(step, int) and step > 0:
            kwargs["prefill_step_size"] = step
        return original(prompt, model, *args, **kwargs)

    generate_step.__wrapped__ = original
    return generate_step


def ensure() -> None:
    """Apply the model patches now.

    Called by any path about to reach the engine directly rather than through
    symbio.mlx_gate. Idempotent; safe to call once at the top of a model
    routine, or rely on the gate's prepare hook and call nothing.
    """
    _apply()


on_engine_prepare(_apply)
