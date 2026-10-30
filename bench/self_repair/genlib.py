"""The model behind the self-repair scripts: one `raw(prompt, temp)` call
whichever backend serves it, so drive.py (repairs) and propose.py (writes
the bugs) talk to the same model the same way.

The model is reached through GEN:
    mlx     (default) mlx-lm on Apple Silicon; ADAPTER=dir loads a LoRA
    hf      symbio.backend on torch (CUDA or ROCm); ADAPTER=dir loads a peft LoRA
    openai  any OpenAI-compatible server (vLLM, llama.cpp); OPENAI_BASE_URL, MODEL is the served name

Other knobs: MAXTOK (900), THINK, LOAD_BITS (hf).
"""
import json, os

GEN = os.environ.get("GEN", "mlx")
THINK = bool(os.environ.get("THINK"))
MAXTOK = int(os.environ.get("MAXTOK", 900))
ADAPTER = os.environ.get("ADAPTER")


def load(model_name):
    """Load once, return raw(prompt, temp) -> the model's reply text."""
    if GEN == "openai":
        import urllib.request

        base = os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1").rstrip("/")

        def raw(prompt, temp=0.0):
            body = {"model": model_name, "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": MAXTOK, "temperature": temp}
            if not THINK:
                body["chat_template_kwargs"] = {"enable_thinking": False}
            req = urllib.request.Request(f"{base}/chat/completions", json.dumps(body).encode(),
                                         {"Content-Type": "application/json",
                                          "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', 'none')}"})
            with urllib.request.urlopen(req, timeout=900) as r:
                return json.loads(r.read())["choices"][0]["message"]["content"] or ""
        return raw

    if GEN == "hf":
        from symbio import backend
        cfg = {"backend": "cuda", "cuda": {"load_in_bits": int(os.environ.get("LOAD_BITS", 16))}}
        model, tok = backend.load(model_name, adapter_path=ADAPTER, config=cfg)
    else:
        from symbio.app import mlx_compat  # noqa: F401  (patches before any load)
        from symbio.app.modelload import load as mlx_load
        model, tok = mlx_load(model_name)
        if ADAPTER:
            from mlx_lm.tuner.utils import load_adapters
            model = load_adapters(model, ADAPTER)
            model.eval()
    if ADAPTER:
        print("  [bench] adapter", ADAPTER, flush=True)

    def raw(prompt, temp=0.0):
        text = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=THINK)
        if GEN == "hf":
            # backend reads agent.temperature, defaulting to 0.7: pin it
            cfg["agent"] = {"temperature": temp}
            return backend.generate(model, tok, text, max_tokens=MAXTOK, config=cfg)
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler
        return generate(model, tok, prompt=text, sampler=make_sampler(temp=temp),
                        max_tokens=MAXTOK, verbose=False)
    return raw


def run_tests(test, timeout=300):
    """pytest one test file from cwd; returns its stdout."""
    import subprocess, sys
    return subprocess.run([sys.executable, "-m", "pytest", test, "-q", "--no-header", "-p",
                           "no:cacheprovider", "--tb=short"], capture_output=True, text=True,
                          timeout=timeout).stdout
