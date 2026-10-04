"""Build the decision model's text encoder as a Core ML model for the ANE.

Apple's own sentence embedding (NLContextualEmbedding) turned out to run on
the CPU: measured with macmon, embedding 744 sentences drew 9.4 W of CPU and
0.000 W of Neural Engine. So the encoder is our own: all-MiniLM-L6-v2
(Apache-2.0, 22.7M parameters, 384 dimensions), traced with a fixed input
shape, mean-pooled and normalised inside the graph, and converted to an fp16
ML Program that Core ML can place on the Neural Engine.

It needs coremltools, which does not install cleanly beside the project's
Python 3.14 venv, so it runs in its own:

    uv venv --python 3.12 /tmp/cml && uv pip install --python /tmp/cml/bin/python \\
        coremltools "torch==2.7.0" "transformers<5"
    /tmp/cml/bin/python symbio_ane/build_text_encoder.py --out <SYMBIO_HOME>/cache/text-encoder

The output directory holds encoder.mlpackage, tokenizer.json and meta.json.
symbio/app/ane.py finds it there; without it the decision model falls back
to Apple's CPU embedding.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SEQ_LEN = 64


def main() -> None:
    import coremltools as ct
    import numpy as np
    import torch
    from transformers import AutoModel, AutoTokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    bert = AutoModel.from_pretrained(MODEL, torchscript=True, attn_implementation="eager").eval()

    config = bert.config
    heads, width = config.num_attention_heads, config.hidden_size
    head_dim = width // heads

    class Encoder(torch.nn.Module):
        """Token ids in, one L2-normalised sentence vector out.

        BERT's forward written out with every shape a constant. Traced through
        transformers' own modules, the size()-to-int casts they make on the
        way (position ids, token types, mask helpers) do not convert:
        coremltools stops at "only 0-dimensional arrays can be converted to
        Python scalars". Fixed shapes are also what the Neural Engine wants.
        """

        def __init__(self, model):
            super().__init__()
            self.model = model
            self.register_buffer("positions", torch.arange(SEQ_LEN)[None, :])

        def forward(self, input_ids, attention_mask):
            emb = self.model.embeddings
            x = (emb.word_embeddings(input_ids) + emb.position_embeddings(self.positions)
                 + emb.token_type_embeddings.weight[0])
            x = emb.LayerNorm(x)
            mask = attention_mask.to(torch.float32)
            additive = (1.0 - mask[:, None, None, :]) * -1e4
            for layer in self.model.encoder.layer:
                attn = layer.attention.self
                q = attn.query(x).reshape(1, SEQ_LEN, heads, head_dim).transpose(1, 2)
                k = attn.key(x).reshape(1, SEQ_LEN, heads, head_dim).transpose(1, 2)
                v = attn.value(x).reshape(1, SEQ_LEN, heads, head_dim).transpose(1, 2)
                scores = torch.matmul(q, k.transpose(-1, -2)) * (head_dim ** -0.5) + additive
                context = torch.matmul(scores.softmax(dim=-1), v).transpose(1, 2).reshape(1, SEQ_LEN, width)
                x = layer.attention.output.LayerNorm(layer.attention.output.dense(context) + x)
                inner = torch.nn.functional.gelu(layer.intermediate.dense(x))
                x = layer.output.LayerNorm(layer.output.dense(inner) + x)
            weights = mask.unsqueeze(-1)
            pooled = (x * weights).sum(1) / weights.sum(1).clamp(min=1e-6)
            return pooled / pooled.norm(dim=-1, keepdim=True).clamp(min=1e-6)

    encoder = Encoder(bert).eval()
    sample = tokenizer(["an example message"], padding="max_length", truncation=True,
                       max_length=SEQ_LEN, return_tensors="pt")
    with torch.no_grad():
        traced = torch.jit.trace(encoder, (sample["input_ids"].int(), sample["attention_mask"].int()))
        # The reference is transformers' own forward, not the rewrite: the
        # written-out graph has to say what the library says.
        hidden = bert(input_ids=sample["input_ids"], attention_mask=sample["attention_mask"])[0]
        weights = sample["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * weights).sum(1) / weights.sum(1)
        reference = (pooled / pooled.norm(dim=-1, keepdim=True)).numpy()

    mlmodel = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=[ct.TensorType(name="input_ids", shape=(1, SEQ_LEN), dtype=np.int32),
                ct.TensorType(name="attention_mask", shape=(1, SEQ_LEN), dtype=np.int32)],
        outputs=[ct.TensorType(name="embedding")],
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.macOS14,
    )
    mlmodel.short_description = f"{MODEL}, mean-pooled and L2-normalised, fixed {SEQ_LEN} tokens"
    mlmodel.save(str(out / "encoder.mlpackage"))
    tokenizer.backend_tokenizer.save(str(out / "tokenizer.json"))

    # The converted graph must say what the original says.
    got = mlmodel.predict({"input_ids": sample["input_ids"].numpy().astype(np.int32),
                           "attention_mask": sample["attention_mask"].numpy().astype(np.int32)})["embedding"]
    cosine = float((got * reference).sum() / (np.linalg.norm(got) * np.linalg.norm(reference)))
    (out / "meta.json").write_text(json.dumps({"model": MODEL, "seq_len": SEQ_LEN, "dims": int(got.shape[-1]),
                                               "pad_id": tokenizer.pad_token_id,
                                               "cosine_to_torch": round(cosine, 5)}, indent=2))
    print(f"wrote {out}: dims {got.shape[-1]}, cosine to torch {cosine:.5f}")


if __name__ == "__main__":
    main()
