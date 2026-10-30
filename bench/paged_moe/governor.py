#!/usr/bin/env python3
"""The trained avatar's own voice: in-process generation for lfm2_moe.

The governor is the learner. The same Bank, the same LoRA, the same spawned
delta-expert slots that grew this afternoon — reading telemetry and dictating
plans directly, in-process, under the default Metal working set (streaming
weights, nothing fully wired).

Generation design:
  * PREFILL: the prompt runs through the frozen stack + tail exactly as the
    trainer does, but layer functions are asked to keep caches: K/V per
    attention layer (after rope), conv window states (last L-1 rows of Bx),
    and the boundary stash after layer 37.
  * DECODE: one token at a time. Frozen layers use their caches
    incrementally; the grad region runs with lora + spawn machinery live —
    which is the whole point: the voice speaks WITH the grown expert.
  * greedy decoding, stop at EOS or the <|im_end|> token string.

stdlib + mlx only. Not part of the harness import chain.
"""
import json
import re
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from safetensors import safe_open

from train_paged import Bank
from train_lfm import load_cfg, shard_map, pin_dense, rope_tables, rms, \
    embed_rows

EOT_STR = "<|im_end|>"


class Governor:
    def __init__(self, model_dir: Path, adapters: Path | None = None,
                 hot_budget_mb: int = 2200):
        self.cfg = load_cfg(model_dir)
        self.cfg.spawn = 4
        self.cfg.eos = None
        try:
            c = json.loads((model_dir / "config.json").read_text())
            eos = c.get("eos_token_id")
            self.eos_id = eos[0] if isinstance(eos, list) else eos
        except Exception:
            self.eos_id = 124900
        self.bank = Bank(shard_map(model_dir), budget_mb=hot_budget_mb)
        attn_idx = [i for i, t in enumerate(self.cfg.layer_types)
                    if t == "full_attention"]
        self.grad_layers = [attn_idx[-1]]
        self.conv_tail = list(range(self.grad_layers[0] + 1, self.cfg.nlayers))
        pin_dense(self.bank, self.cfg)
        for p in ("model.embed_tokens.weight", "model.embed_tokens.scales",
                  "model.embed_tokens.biases"):
            if p in self.bank.names:
                self.bank.get(p, resident=True)
        self.bank.get("model.embedding_norm.weight", resident=True)

        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))

        self.P = None
        self.scale = 1.0
        if adapters and Path(adapters).exists():
            self.P = {}
            with safe_open(str(adapters), framework="mlx") as s:
                for k in s.keys():
                    self.P[k] = s.get_tensor(k)
            cfgp = Path(adapters).parent / "adapter_config.json"
            if cfgp.exists():
                try:
                    self.scale = float(json.loads(
                        cfgp.read_text()).get("scale", 8.0))
                    self.P["_scale"] = self.scale
                except Exception:
                    self.P["_scale"] = 8.0

    # ---------- quant linear on the fly (shared helpers reimplemented light)
    def qlin(self, x, base, gs=None, bits=None):
        o = (self.cfg.per_tensor.get(base, {}) if hasattr(self.cfg, "per_tensor")
             else {})
        gs = gs or o.get("group_size", self.cfg.gs)
        bits = bits or o.get("bits", self.cfg.bits)
        w = self.bank.get(base + ".weight")
        s = self.bank.get(base + ".scales") if base + ".scales" in self.bank.names else None
        b = self.bank.get(base + ".biases") if base + ".biases" in self.bank.names else None
        if s is None:
            return x @ w.T
        return mx.quantized_matmul(x, w, s, b, group_size=gs, bits=bits,
                                   transpose=True)

    # ---------- caches -------------------------------------------------
    def _new_caches(self):
        return {"kv": {}, "conv": {}}

    def _attn_prefill(self, x, i, caches):
        """Full-sequence prefill for layer i; LoRA added on q/v projections
        (pre-attention), caches stored AFTER rope, exactly as training."""
        Bc, T, H = x.shape
        cfg, h, hd = self.cfg, self.cfg.heads, self.cfg.head_dim
        base = f"model.layers.{i}.self_attn"

        def proj(name, key=None):
            y = self.qlin(x, f"{base}.{name}")
            if key is not None and self.P is not None:
                A = self.P.get(key + ".lora_a")
                if A is not None:
                    Bb = self.P[key + ".lora_b"]
                    y = y + ((x @ A) @ Bb) * self.P["_scale"]
            return y

        q = proj("q_proj", f"{base}.q_proj").reshape(Bc, T, h, hd)
        k = proj("k_proj").reshape(Bc, T, cfg.kv_heads, hd)
        v = proj("v_proj", f"{base}.v_proj").reshape(Bc, T, cfg.kv_heads, hd)
        q = rms(q, self.bank.get(f"{base}.q_layernorm.weight", resident=True),
                cfg.eps)
        k = rms(k, self.bank.get(f"{base}.k_layernorm.weight", resident=True),
                cfg.eps)
        q, k = q.transpose(0, 2, 1, 3), k.transpose(0, 2, 1, 3)
        d2 = hd // 2
        inv = 1.0 / (cfg.rope_theta ** (mx.arange(0, hd, 2,
                                                  dtype=mx.float32) / hd))
        t0 = mx.arange(T, dtype=mx.float32)
        cosf, sinf = mx.cos(mx.outer(t0, inv)), mx.sin(mx.outer(t0, inv))
        c, s = cosf[None, None, :, :], sinf[None, None, :, :]
        q = mx.concatenate([q[..., :d2] * c - q[..., d2:] * s,
                            q[..., :d2] * s + q[..., d2:] * c], axis=-1)
        k = mx.concatenate([k[..., :d2] * c - k[..., d2:] * s,
                            k[..., :d2] * s + k[..., d2:] * c], axis=-1)
        caches["kv"][i] = (k, v.transpose(0, 2, 1, 3))
        groups = h // cfg.kv_heads
        kk = mx.repeat(k, groups, axis=1)
        vv = mx.repeat(v.transpose(0, 2, 1, 3), groups, axis=1)
        scores = (q @ kk.transpose(0, 1, 3, 2)) * (hd ** -0.5)
        mask = mx.triu(mx.full((T, T), -mx.inf), k=1).astype(scores.dtype)
        ctx = mx.softmax(scores + mask, axis=-1).astype(x.dtype) @ vv
        ctx = ctx.transpose(0, 2, 1, 3).reshape(Bc, T, h * hd)
        return self.qlin(ctx, f"{base}.out_proj")

    def _attn_step(self, tok_x, i, caches):
        """One-token incremental attention: LoRA added on q/v projections
        (pre-attention), rope with offset = cache length."""
        cfg, h, hd = self.cfg, self.cfg.heads, self.cfg.head_dim
        base = f"model.layers.{i}.self_attn"
        offset = caches["kv"][i][0].shape[2]

        def proj(name, key=None):
            y = self.qlin(tok_x, f"{base}.{name}")
            if key is not None and self.P is not None:
                A = self.P.get(key + ".lora_a")
                if A is not None:
                    Bb = self.P[key + ".lora_b"]
                    y = y + ((tok_x @ A) @ Bb) * self.P["_scale"]
            return y

        q = proj("q_proj", f"{base}.q_proj").reshape(1, 1, h, hd)
        k = proj("k_proj").reshape(1, 1, cfg.kv_heads, hd)
        v = proj("v_proj", f"{base}.v_proj").reshape(1, 1, cfg.kv_heads, hd)
        q = rms(q, self.bank.get(f"{base}.q_layernorm.weight", resident=True),
                cfg.eps)
        k = rms(k, self.bank.get(f"{base}.k_layernorm.weight", resident=True),
                cfg.eps)
        d2 = hd // 2
        inv = 1.0 / (cfg.rope_theta ** (mx.arange(0, hd, 2,
                                                  dtype=mx.float32) / hd))
        c1, s1 = mx.cos(offset * inv), mx.sin(offset * inv)
        q = mx.concatenate([q[..., :d2] * c1 - q[..., d2:] * s1,
                            q[..., :d2] * s1 + q[..., d2:] * c1], axis=-1)
        k = mx.concatenate([k[..., :d2] * c1 - k[..., d2:] * s1,
                            k[..., :d2] * s1 + k[..., d2:] * c1], axis=-1)
        k_full, v_full = caches["kv"][i]
        caches["kv"][i] = (mx.concatenate([k_full, k.transpose(0, 2, 1, 3)],
                                          axis=2),
                           mx.concatenate([v_full, v.transpose(0, 2, 1, 3)],
                                          axis=2))
        k_all, v_all = caches["kv"][i]
        groups = h // cfg.kv_heads
        kk = mx.repeat(k_all, groups, axis=1)
        vv = mx.repeat(v_all, groups, axis=1)
        scores = (q @ kk.transpose(0, 1, 3, 2)) * (hd ** -0.5)
        ctx = mx.softmax(scores, axis=-1).astype(tok_x.dtype) @ vv
        print(f"    [step {i}] q{q.shape} kk{kk.shape} vv{vv.shape} "
              f"scores{scores.shape} ctx{ctx.shape}", flush=True)
        ctx = ctx.transpose(0, 2, 1, 3).reshape(1, 1, h * hd)
        return self.qlin(ctx, f"{base}.out_proj")

    def _conv_prefill(self, x, i, caches):
        cfg = self.cfg
        L = cfg.conv_L
        base = f"model.layers.{i}.conv"
        BCx = self.qlin(x, f"{base}.in_proj").reshape(x.shape[0], x.shape[1],
                                                      3, x.shape[2])
        B, C, xg = BCx[..., 0, :], BCx[..., 1, :], BCx[..., 2, :]
        bx = B * xg
        w = self.bank.get(f"{base}.conv.weight", resident=True)
        wt = w.reshape(w.shape[0], -1)
        if wt.shape[1] != L:
            wt = wt.T
        Bx_pad = mx.pad(bx, [(0, 0), (L - 1, 0), (0, 0)])
        acc = mx.zeros_like(bx)
        for kk in range(L):
            acc = acc + Bx_pad[:, kk:kk + x.shape[1], :] * wt[:, kk][None, None, :]
        caches["conv"][i] = bx[:, -(L - 1):, :]
        return self.qlin(C * acc, f"{base}.out_proj")

    def _conv_step(self, tok_x, i, caches):
        cfg = self.cfg
        L = cfg.conv_L
        base = f"model.layers.{i}.conv"
        BCx = self.qlin(tok_x, f"{base}.in_proj").reshape(1, 1, 3, tok_x.shape[-1])
        B, C, xg = BCx[..., 0, :], BCx[..., 1, :], BCx[..., 2, :]
        bx = B * xg
        state = caches["conv"][i]
        window = mx.concatenate([state, bx], axis=1)          # [1, L, H]
        w = self.bank.get(f"{base}.conv.weight", resident=True)
        wt = w.reshape(w.shape[0], -1)
        if wt.shape[1] != L:
            wt = wt.T
        acc = sum(window[:, jj, :] * wt[:, jj][None, :] for jj in range(L))
        caches["conv"][i] = window[:, -(L - 1):, :]
        return self.qlin(C * acc, f"{base}.out_proj")

    # ---------- moe with spawn, single or batch -------------------------
    def _moe(self, x, i):
        cfg = self.cfg
        base = f"model.layers.{i}.feed_forward"
        H = x.shape[-1]
        flat = x.reshape(-1, H)
        g = self.bank.get(f"{base}.gate.weight", resident=True)
        logits = (flat @ g.T).astype(mx.float32)
        if self.P is not None and cfg.spawn and \
                f"model.layers.{i}.spawn_router" in self.P:
            r = self.P[f"model.layers.{i}.spawn_router"]
            logits = mx.concatenate(
                [logits, (flat @ r.T).astype(mx.float32)], axis=-1)
        k = cfg.per_tok
        gates = mx.softmax(logits, axis=-1)
        inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
        scores = mx.take_along_axis(gates, inds, axis=-1)
        if cfg.norm_topk_prob:
            scores = scores / (mx.sum(scores, axis=-1, keepdims=True) + 1e-20)
        scores = scores.astype(x.dtype)
        xe = flat[:, None, None, :]
        wname = f"{base}.switch_mlp"

        def slot(part, src):
            w = self.bank.get(f"{wname}.{part}.weight")
            s = self.bank.get(f"{wname}.{part}.scales")
            b = self.bank.get(f"{wname}.{part}.biases") \
                if f"{wname}.{part}.biases" in self.bank.names else None
            y = mx.gather_qmm(src, w, s, b,
                              rhs_indices=mx.minimum(inds, cfg.experts - 1),
                              transpose=True, group_size=cfg.gs, bits=cfg.bits)
            return y[..., 0, :]

        gi, ui = slot("gate_proj", xe), slot("up_proj", xe)
        dn = slot("down_proj", (silu(gi) * ui)[:, :, None, :])
        keep = (inds < cfg.experts).astype(x.dtype)
        y = ((dn * scores[..., None]) * keep[..., None]).sum(-2)
        for s in range(cfg.spawn if self.P is not None else 0):
            A = self.P.get(f"model.layers.{i}.spawn.{s}.lora_a")
            if A is None:
                continue
            Bl = self.P[f"model.layers.{i}.spawn.{s}.lora_b"]
            w_s = ((inds == (cfg.experts + s)) * scores).sum(-1)
            y = y + (((flat @ A) @ Bl) * w_s[:, None])
        return y.reshape(x.shape)

    # ---------- layer dispatch -----------------------------------------
    def _frozen_layer(self, x, i, caches, cos_all=None):
        cfg = self.cfg
        nx = rms(x, self.bank.get(f"model.layers.{i}.operator_norm.weight",
                                  resident=True), cfg.eps)
        if cfg.layer_types[i] == "full_attention":
            r = self._attn_prefill(nx, i, caches)
            h = x + r
        else:
            r = self._conv_prefill(nx, i, caches)
            h = x + r
        h = rms(h, self.bank.get(f"model.layers.{i}.ffn_norm.weight",
                                 resident=True), cfg.eps)
        if i < cfg.num_dense:
            base = f"model.layers.{i}.feed_forward"
            flat = h.reshape(-1, cfg.hidden)
            g = self.qlin(flat, f"{base}.gate_proj")
            u = self.qlin(flat, f"{base}.up_proj")
            d = self.qlin(silu(g) * u, f"{base}.down_proj")
            return h + d.reshape(h.shape)
        return h + self._moe(h, i)

    def _layer_prefill(self, x, i, caches):
        """Full-stack prefill with cache capture, matching layer_fwd_exact
        math; adapter/spawn keys self-gate per layer (only layer 38 has them)."""
        cfg = self.cfg
        nx = rms(x, self.bank.get(f"model.layers.{i}.operator_norm.weight",
                                  resident=True), cfg.eps)
        if cfg.layer_types[i] == "full_attention":
            r = self._attn_prefill(nx, i, caches)
        else:
            r = self._conv_prefill(nx, i, caches)
        h = x + r
        h = rms(h, self.bank.get(f"model.layers.{i}.ffn_norm.weight",
                                 resident=True), cfg.eps)
        if i < cfg.num_dense:
            base = f"model.layers.{i}.feed_forward"
            flat = h.reshape(-1, cfg.hidden)
            g = self.qlin(flat, f"{base}.gate_proj")
            u = self.qlin(flat, f"{base}.up_proj")
            d = self.qlin(silu(g) * u, f"{base}.down_proj")
            return h + d.reshape(h.shape)
        return h + self._moe(h, i)

    def _layer_step(self, tok_x, i, caches):
        """One-token step through layer i (any layer; caches must exist)."""
        cfg = self.cfg
        nx = rms(tok_x, self.bank.get(f"model.layers.{i}.operator_norm.weight",
                                      resident=True), cfg.eps)
        if cfg.layer_types[i] == "full_attention":
            h = tok_x + self._attn_step(nx, i, caches)
        else:
            h = tok_x + self._conv_step(nx, i, caches)
        h = rms(h, self.bank.get(f"model.layers.{i}.ffn_norm.weight",
                                 resident=True), cfg.eps)
        if i < cfg.num_dense:
            base = f"model.layers.{i}.feed_forward"
            flat = h.reshape(1, cfg.hidden)
            g = self.qlin(flat, f"{base}.gate_proj")
            u = self.qlin(flat, f"{base}.up_proj")
            d = self.qlin(silu(g) * u, f"{base}.down_proj")
            return h + d.reshape(h.shape)
        return h + self._moe(h, i)

    def _head(self, h_last):
        x = rms(h_last, self.bank.get("model.embedding_norm.weight",
                                      resident=True), self.cfg.eps)
        return self.qlin(x, "model.embed_tokens")     # tied head

    def generate(self, ids, max_new=700, temperature=0.2):
        """Prefill prompt ids, then greedy-ish decode. Returns list[int]."""
        cfg = self.cfg
        caches = self._new_caches()
        x = embed_rows(self.bank, ids, cfg).astype(mx.bfloat16)[None, :, :]
        for i in range(cfg.nlayers):
            x = self._layer_prefill(x, i, caches)
            mx.eval(x)
        h_last = x[0, -1:, :]
        if self.eos_id is None:
            self.eos_id = 124900
        out = []
        cur_h = h_last
        generated = 0
        while generated < max_new:
            logits = self._head(cur_h)[0, 0]        # [V]
            if temperature and temperature < 1.0:
                logits = logits / max(temperature, 1e-3)
            tok = int(mx.argmax(logits))
            if tok == self.eos_id:
                break
            out.append(tok)
            generated += 1
            ex = embed_rows(self.bank, [tok], cfg).astype(mx.bfloat16)
            h = ex[:, None, :]
            for i in range(cfg.nlayers):
                h = self._layer_step(h, i, caches)
                if i in (10, 30):                   # occasional checkpoint
                    mx.eval(h)
            cur_h = h[:, -1:, :]
        return out

    def decide(self, state: dict, goal: str = "") -> dict:
        """THE TRAINED AVATAR DICTATES. Its own grown expert is live while
        the plan is written."""
        prompt = (
            "You are the learning governor of an on-device AI harness "
            "(Symbio). You decide how and when it trains itself — you are "
            "the same brain the lessons live in. Current self-state:\n\n"
            + json.dumps(state, indent=1, default=str)
            + "\n\nGoal: "
            + (goal or "learn how ARC-AGI-2 works; improve held-out "
                       "performance at sane cost.")
            + "\n\nDecide the next move. Reply with ONLY a JSON object:\n"
            + '{"action": "train" | "eval" | "wait", '
              '"tasks": <int>, "spawn": <int>, "iters": <int>, '
              '"reasoning": "<one paragraph, first person>"}'
            + "\n\nIf training is already running, wait. If a paired eval "
              "exists, read it before deciding. Be honest.")
        msgs = [{"role": "user", "content": prompt}]
        rendered = self.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)
        ids = [int(t) for t in self.tokenizer(rendered)["input_ids"]]
        if len(ids) > 1200:
            ids = ids[:1200]
        reply_ids = self.generate(ids, max_new=500, temperature=0.2)
        text = self.tokenizer.decode(reply_ids, skip_special_tokens=True)
        m = re.search(r'\{[^{}]*"action"[^{}]*\}', text, re.DOTALL)
        if not m:
            raise ValueError(f"no plan JSON in own-voice reply: {text[:200]}")
        plan = json.loads(m.group(0))
        plan["_decider"] = "LFM2.5 trained avatar (in-process, grown expert live)"
        return plan


def silu(x):
    return x * mx.sigmoid(x)