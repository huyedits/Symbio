"""Training and serving must agree on enable_thinking.

Every sample in the corpus is rendered by training.build_chat_training_sample
with the same enable_thinking value the live agent loop prompts with
(training.THINKING_ENABLED). A mismatch between the two was the original
failure mode: the corpus was trained with thinking off while the loop asked
for it, and reasoning surfaced in place of the answer:

    Huy   : hi
    Caine : The assistant already greeted the user.

The golden set and eval both grade with the same value, so nothing in the
regression net could see the mismatch. These tests pin the alignment.
"""

import builtins

import test_main_loop as tml
from symbio.app import chat as chat_mod
from symbio.app import training


class RecordingTokenizer(tml.FakeTokenizer):
    """Records the enable_thinking used on every template call."""

    def __init__(self):
        self.calls: list[dict] = []

    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=False, enable_thinking=False):
        self.calls.append({
            "add_generation_prompt": add_generation_prompt,
            "enable_thinking": enable_thinking,
        })
        return super().apply_chat_template(
            messages, tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking,
        )


def test_training_renders_with_the_configured_thinking_mode():
    tok = RecordingTokenizer()
    training.build_chat_training_sample(
        [{"role": "user", "content": "hi"},
         {"role": "assistant", "content": "Hey."}], tok)
    assert tok.calls[0]["enable_thinking"] is training.THINKING_ENABLED


def test_agent_loop_generation_matches_training(monkeypatch):
    """The live generation prompt must use the training value."""
    tok = RecordingTokenizer()
    session = tml.ScriptedSession(
        user_inputs=["hi"],
        model_replies=["Hey Huy — what's up?"],
    )

    real_input = builtins.input
    real_make_cache = chat_mod.make_prompt_cache
    real_can_trim = chat_mod.can_trim_prompt_cache
    real_trim = chat_mod.trim_prompt_cache
    builtins.input = session.fake_input
    chat_mod.make_prompt_cache = lambda model: []
    chat_mod.can_trim_prompt_cache = lambda cache: True
    chat_mod.trim_prompt_cache = lambda cache, n: cache
    try:
        from symbio.app import config as app_config
        from test_utils import preserve_training_state

        cfg = dict(app_config.load_config())
        rag_cfg = dict(cfg.get("rag", {}))
        rag_cfg["tag_index_enabled"] = False
        rag_cfg["auto_index_enabled"] = False
        cfg["rag"] = rag_cfg
        with preserve_training_state(adapters=True):
            chat_mod.chat_loop(
                cfg, model=object(), tokenizer=tok, adapter_loaded=False,
                generate_fn=session.fake_generate,
                stream_fn=session.fake_stream_generate,
                stream_chunk_fn=lambda s: None,
                output_fn=lambda t: None,
                input_fn=session.fake_input,
            )
    finally:
        builtins.input = real_input
        chat_mod.make_prompt_cache = real_make_cache
        chat_mod.can_trim_prompt_cache = real_can_trim
        chat_mod.trim_prompt_cache = real_trim

    generation_calls = [c for c in tok.calls if c["add_generation_prompt"]]
    assert generation_calls, "the loop never built a generation prompt"
    for call in generation_calls:
        assert call["enable_thinking"] is training.THINKING_ENABLED, (
            f"generation used enable_thinking={call['enable_thinking']}, "
            f"but the corpus is trained with {training.THINKING_ENABLED}")


def test_corpus_thinking_matches_training_mode():
    """The corpus's thinking blocks must match the training mode: when
    THINKING_ENABLED is True the samples carry real reasoning, and when it is
    False they carry none. A mismatch means the corpus drifted from the
    training/serving value."""
    import json
    import re

    from symbio import constants

    train_file = constants.TRAIN_FILE
    if not train_file.exists():
        return
    for raw in train_file.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        text = json.loads(raw).get("text", "")
        for match in re.finditer(r" thinking(.*?) response", text, re.DOTALL):
            has_reasoning = bool(match.group(1).strip())
            assert has_reasoning is training.THINKING_ENABLED, (
                "corpus thinking blocks disagree with THINKING_ENABLED="
                f"{training.THINKING_ENABLED}")
