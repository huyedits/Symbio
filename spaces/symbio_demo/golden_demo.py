"""Mocked golden-set demo for the Hugging Face Symbio Space.

No MLX model is loaded. Instead, each GOLDEN_CASES prompt is paired with a
canned reply that exercises the correct tag / tool format. The demo parses those
replies with the same tooling.parse_tools and tooling.strip_tool_tags used by
the real agent, then runs each golden check to report PASS/FAIL.

Drop this file into an existing Gradio Space and mount it as an additional tab
(see README.md for integration).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import gradio as gr


def _load_symbio_modules():
    """Load only the Symbio modules this demo needs, bypassing the top-level
    package __init__ that imports MLX.

    Works both when this file lives inside the Symbio repo (local testing) and
    when it is copied to a Hugging Face Space with `symbio` installed via pip.
    """

    def _find_symbio_dir() -> Path:
        # Try the repo layout first: spaces/symbio_demo/golden_demo.py
        repo_candidate = Path(__file__).resolve().parent.parent.parent / "symbio"
        if (repo_candidate / "constants.py").exists():
            return repo_candidate
        # Otherwise search sys.path for the installed package.
        for p in sys.path:
            candidate = Path(p) / "symbio"
            if (candidate / "constants.py").exists():
                return candidate
        raise FileNotFoundError(
            "Could not locate the Symbio source files. "
            "Install `symbio` or run this file from inside the Symbio repo."
        )

    symbio_dir = _find_symbio_dir()
    app_dir = symbio_dir / "app"

    def _load(name: str, path: Path, parent: ModuleType | None = None):
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        if parent is not None:
            setattr(parent, name.rsplit(".", 1)[-1], mod)
        spec.loader.exec_module(mod)
        return mod

    # symbio package
    if "symbio" not in sys.modules:
        symbio_pkg = ModuleType("symbio")
        symbio_pkg.__path__ = [str(symbio_dir)]
        sys.modules["symbio"] = symbio_pkg

    # symbio.constants (needed by prompts)
    _load("symbio.constants", symbio_dir / "constants.py", sys.modules["symbio"])

    # symbio.app package
    app_pkg = ModuleType("symbio.app")
    app_pkg.__path__ = [str(app_dir)]
    sys.modules["symbio.app"] = app_pkg
    sys.modules["symbio"].app = app_pkg

    # tooling first (prompts depends on it), then prompts and golden
    _load("symbio.app.tooling", app_dir / "tooling.py", app_pkg)
    _load("symbio.app.prompts", app_dir / "prompts.py", app_pkg)
    return _load("symbio.app.golden", app_dir / "golden.py", app_pkg)


golden_mod = _load_symbio_modules()
from symbio.app import tooling  # noqa: E402


# Mock replies for each golden case. These are the *correct* formats the golden
# checks expect, so the demo acts as living documentation of Symbio's tag
# grammar and tool contracts.
_MOCK_REPLIES: dict[str, str] = {
    "greeting": "Hey! How can I help you today?",
    "identity_self": "I am Caine, your personal AI assistant.",
    "identity_not_user": "No — I'm Caine, your assistant. You're Huy.",
    "save_note": "Got it. <note title='User Preference'>The user prefers concise replies.</note>",
    "schedule_reminder": "Will do, Huy. <cron expr='0 9 * * *'>stretch</cron>",
    "run_code_for_math": "<py>import math\nprint(math.factorial(7))</py> Running that now.",
    "web_search_unknown": "I don't have that memorized — checking. <search>latest news</search>",
    "open_app_command": "<cmd>open -a 'Google Chrome'</cmd> Opening Chrome for you, Huy.",
    "browse_to_interact": "<browse>https://www.cloudflare.com</browse> Opening Cloudflare in the controllable browser — I'll click the first button once it loads.",
    "browser_press_key": "<press>down</press> Pressing the down arrow key.",
    # Added apple case
    "browse_apple": "<browse>https://www.apple.com</browse> Opening apple.com to read its contents.",
}


def _run_mock_golden_set(
    assistant_name: str = "Caine",
    user_name: str = "Huy",
    enabled_groups: set[str] | None = None,
) -> dict[str, Any]:
    """Run every golden case against the mocked replies and return details."""
    config = {"assistant_name": assistant_name, "user_name": user_name}
    cases = golden_mod.GOLDEN_CASES
    results = []
    passing = 0

    for case in cases:
        reply = _MOCK_REPLIES.get(case.id, "")
        display = tooling.strip_tool_tags(reply)
        tools = tooling.parse_tools(reply, enabled_groups=enabled_groups)
        ok = case.check(display, tools, config)
        if ok:
            passing += 1
        results.append(
            {
                "id": case.id,
                "description": case.description,
                "reply": reply,
                "display": display,
                "tools": [n for n, _ in tools],
                "result": "PASS" if ok else "FAIL",
            }
        )

    return {
        "passing": passing,
        "total": len(cases),
        "rate": passing / len(cases) if cases else 0,
        "results": results,
    }


def _format_markdown_report(data: dict[str, Any]) -> str:
    lines = [
        f"## Golden set: {data['passing']}/{data['total']} passing ({data['rate']:.0%})",
        "",
        "| Case | Result | Expected tool(s) | Reply snippet |",
        "|------|--------|------------------|---------------|",
    ]
    for r in data["results"]:
        badge = "✅ PASS" if r["result"] == "PASS" else "❌ FAIL"
        tools = ", ".join(r["tools"]) or "(none)"
        snippet = r["reply"].replace("|", "\\|").replace("\n", " ")[:80]
        lines.append(
            f"| **{r['id']}** — {r['description']} | {badge} | {tools} | {snippet}… |"
        )
    return "\n".join(lines)


def _run(assistant_name: str, user_name: str) -> tuple[str, str]:
    data = _run_mock_golden_set(assistant_name=assistant_name, user_name=user_name)
    summary = f"{data['passing']}/{data['total']} passing ({data['rate']:.0%})"
    report = _format_markdown_report(data)
    return summary, report


def build_golden_tab() -> gr.Tab:
    """Return a Gradio Tab ready to be added to an existing Blocks app."""
    with gr.Tab("Golden Checks") as tab:
        gr.Markdown(
            "### Symbio golden-set regression checks\n\n"
            "This tab exercises the same `GOLDEN_CASES` used by the real agent's "
            "`/golden` slash command and pre/post-train regression guard. "
            "Replies are mocked so the demo needs no MLX model — it only runs "
            "the tag parser and the check functions."
        )
        with gr.Row():
            assistant_input = gr.Textbox(
                label="Assistant name", value="Caine", interactive=True
            )
            user_input = gr.Textbox(label="User name", value="Huy", interactive=True)
        run_btn = gr.Button("Run golden set", variant="primary")
        summary = gr.Textbox(label="Summary", interactive=False)
        report = gr.Markdown()

        run_btn.click(
            fn=_run,
            inputs=[assistant_input, user_input],
            outputs=[summary, report],
        )
        # Run once on tab load so users see results immediately.
        tab.select(
            fn=_run,
            inputs=[assistant_input, user_input],
            outputs=[summary, report],
        )
    return tab


def standalone_app() -> gr.Blocks:
    """A minimal standalone Gradio Blocks app with only the golden tab."""
    with gr.Blocks(title="Symbio Golden Demo") as app:
        gr.Markdown("# Symbio — Golden-set regression demo")
        build_golden_tab()
    return app


if __name__ == "__main__":
    standalone_app().launch()
