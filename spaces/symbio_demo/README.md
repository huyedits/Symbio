# Symbio Hugging Face Golden Demo

This folder contains a **mocked golden-set demo tab** for the existing Hugging
Face Space `HuyEdits/symbio-demo`. It does not require MLX or a model download —
it runs Symbio's `GOLDEN_CASES` against canned replies and reports PASS/FAIL for
every regression check.

## Files

- `golden_demo.py` — Gradio tab + standalone app.
- `requirements.txt` — dependencies to add to the Space.

## How to add to the existing Space

1. Copy these files into the existing Space repo:

   ```bash
   git clone https://huggingface.co/spaces/HuyEdits/symbio-demo demo-space
   cp spaces/symbio_demo/golden_demo.py demo-space/
   cp spaces/symbio_demo/requirements.txt demo-space/requirements-golden.txt
   ```

2. Merge `requirements-golden.txt` into the Space's `requirements.txt`:

   ```text
   gradio>=5.0
   # ... existing Space deps ...
   # Symbio local package
   -e git+https://github.com/HuyEdits/symbio.git@main#egg=symbio
   ```

   The demo only needs `symbio.app.golden` and `symbio.app.tooling`, which do
   not use MLX, PyTorch, or Playwright at runtime. However, the top-level
   `symbio` package imports MLX unconditionally, so `golden_demo.py` loads the
   needed modules directly from the installed source files instead of using
   `import symbio`. This keeps the Space lightweight even though `symbio` is
   listed as a dependency.

3. In the Space's `app.py`, import and mount the tab:

   ```python
   import gradio as gr
   from golden_demo import build_golden_tab

   with gr.Blocks() as demo:
       # ... existing tabs ...
       build_golden_tab()

   demo.launch()
   ```

4. Commit and push back to Hugging Face:

   ```bash
   cd demo-space
   git add .
   git commit -m "Add golden-set regression demo tab"
   git push
   ```

## Standalone usage (local testing)

```bash
cd spaces/symbio_demo
python golden_demo.py
```

The browser will open at `http://127.0.0.1:7860` with a single **Golden Checks**
tab showing all case results.

## What it demonstrates

Each row corresponds to a case in `symbio/app/golden.py:GOLDEN_CASES`:

- Identity replies include the assistant/user names correctly.
- Tool tags (`<note>`, `<cron>`, `<py>`, `<search>`, `<cmd>`, `<browse>`, `<press>`)
  are parsed into the right internal tool names.
- The controllable browser (`<browse>`) is chosen over native `open -a` for
  interactive page tasks.
- Native app openers are only used for launching apps with no follow-up control.

A failing case means the tag parser or the golden check drifted apart from the
expected output format — exactly what the real agent's `/golden` command catches
before/after LoRA training.
