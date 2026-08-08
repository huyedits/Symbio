"""Training samples must fit inside the training window.

mlx_lm truncates an oversized sample to lora.max_seq_length and trains on
what is left, printing only a per-batch WARNING that is invisible among the
progress bars. When the assistant turn sits past the cutoff, the sample
contributes nothing to learn from.

That was the state of this corpus: 114 of 114 samples over a 768-token
window, assistant turns beginning around token 4,382, because the ~2,200
token <tools> catalog was embedded in every one. No corpus design could have
mattered while the answers were never reaching the model.
"""

import pytest

from symbio.app import prompts, training


class CountingTokenizer:
    """Whitespace tokenizer — enough to exercise the length arithmetic."""

    def encode(self, text, add_special_tokens=True):
        return text.split()

    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=False, enable_thinking=False):
        return "\n".join(f"{m['role']}: {m['content']}" for m in messages)


def test_training_prompt_is_a_strict_prefix_of_the_served_one():
    """Training may see less context than inference, never different."""
    full = prompts.build_system_prompt("Caine", "Huy")
    train = prompts.build_training_system_prompt("Caine", "Huy")
    assert full.startswith(train.rstrip())
    assert len(train) < len(full)


def test_training_prompt_drops_the_catalog_but_keeps_the_instructions():
    train = prompts.build_training_system_prompt("Caine", "Huy")
    # The JSON block is gone. Its closing tag only ever appears with the block,
    # whereas the opening tag also occurs in the prose that describes it —
    # which is instruction text and must stay.
    assert "</tools>" not in train
    assert '"parameters"' not in train
    # The behaviour rules must survive; they are what is being taught.
    assert "<tool_call>" in train
    assert "<note" in train
    assert "<search>" in train


def test_prose_mention_of_tools_does_not_truncate_the_prompt():
    """rfind, not find: the prompt names <tools> in its prose well before the
    real block, and cutting at the first match discarded almost everything."""
    train = prompts.build_training_system_prompt("Caine", "Huy")
    full = prompts.build_system_prompt("Caine", "Huy")
    # The prose mention sits early; a find()-based cut would land near it.
    first = full.find("<tools>")
    assert len(train) > first, (
        "training prompt was cut at the prose mention, not the catalog")


def test_strip_tool_catalog_only_touches_the_system_turn():
    messages = [
        {"role": "system", "content": "rules here\n\n<tools>[{...}]</tools>\n"},
        {"role": "user", "content": "keep <tools> in a user turn"},
        {"role": "assistant", "content": "and here too <tools>"},
    ]
    out = training.strip_tool_catalog(messages)
    assert "<tools>" not in out[0]["content"]
    assert "rules here" in out[0]["content"]
    assert out[1] == messages[1]
    assert out[2] == messages[2]


def test_strip_tool_catalog_is_a_noop_without_a_catalog():
    messages = [{"role": "system", "content": "just rules"}]
    assert training.strip_tool_catalog(messages) == messages


def test_built_sample_excludes_the_catalog():
    tok = CountingTokenizer()
    text = training.build_chat_training_sample([
        {"role": "system", "content": "rules\n\n<tools>[{\"a\":1}]</tools>"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ], tok)
    assert "<tools>" not in text
    assert "hello" in text


def test_seed_samples_fit_the_configured_window():
    """The corpus the project ships with must fit its own default."""
    from transformers import AutoTokenizer

    from symbio.app.config import DEFAULT_CONFIG

    limit = DEFAULT_CONFIG["lora"]["max_seq_length"]
    try:
        tok = AutoTokenizer.from_pretrained(DEFAULT_CONFIG["model_name"])
    except Exception:  # pragma: no cover - offline
        pytest.skip("tokenizer unavailable")
    system = prompts.build_system_prompt("Caine", "Huy")
    longest = 0
    for user_msg, assistant_msg in training.build_seed_pairs("Caine", "Huy"):
        text = training.build_chat_training_sample([
            {"role": "system", "content": system},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_msg},
        ], tok)
        longest = max(longest, len(tok.encode(text)))
    assert longest <= limit, (
        f"longest seed sample is {longest} tokens against a {limit}-token "
        f"window — the assistant turn would be truncated away")


def test_length_check_counts_lost_answers(tmp_path, monkeypatch):
    import json

    from symbio import constants

    train_file = tmp_path / "train.jsonl"
    # One short sample, one whose answer sits far past a tiny window.
    short = "system: hi\nuser: q\n<|im_start|>assistant\na"
    long_prefix = " ".join(["filler"] * 50)
    lost = f"system: {long_prefix}\n<|im_start|>assistant\nanswer"
    train_file.write_text(
        json.dumps({"text": short}) + "\n" + json.dumps({"text": lost}) + "\n",
        encoding="utf-8")
    monkeypatch.setattr(constants, "TRAIN_FILE", train_file)

    stats = training.check_sample_lengths(
        CountingTokenizer(), {"lora": {"max_seq_length": 10}})
    assert stats["total"] == 2
    assert stats["over_limit"] == 1
    assert stats["truncated_answers"] == 1
    warning = training.format_length_warning(stats)
    assert "lose their assistant turn" in warning


def test_no_warning_when_everything_fits(tmp_path, monkeypatch):
    import json

    from symbio import constants

    train_file = tmp_path / "train.jsonl"
    train_file.write_text(json.dumps({"text": "system: a\nassistant: b"}) + "\n",
                          encoding="utf-8")
    monkeypatch.setattr(constants, "TRAIN_FILE", train_file)
    stats = training.check_sample_lengths(
        CountingTokenizer(), {"lora": {"max_seq_length": 100}})
    assert stats["over_limit"] == 0
    assert training.format_length_warning(stats) is None


def test_compact_existing_samples_shrinks_and_preserves(tmp_path, monkeypatch):
    """Migration must shrink old samples without losing their content."""
    import json

    from symbio import constants

    train_file = tmp_path / "train.jsonl"
    valid_file = tmp_path / "valid.jsonl"
    catalog = '<tools>[{"name":"terminal","parameters":{"x":1}}]</tools>'
    sample = (f"system: rules here\n\n{catalog}\n"
              "user: how much does X cost\nassistant: <search>X price</search>")
    train_file.write_text(json.dumps({"text": sample}) + "\n", encoding="utf-8")
    valid_file.write_text(json.dumps({"text": sample}) + "\n", encoding="utf-8")
    monkeypatch.setattr(constants, "TRAIN_FILE", train_file)
    monkeypatch.setattr(constants, "VALID_FILE", valid_file)

    counts = training.compact_existing_samples()
    assert counts["train.jsonl"] == 1
    assert counts["valid.jsonl"] == 1

    out = json.loads(train_file.read_text().strip())["text"]
    assert "<tools>" not in out
    # Everything that carries signal survives.
    assert "rules here" in out
    assert "how much does X cost" in out
    assert "<search>X price</search>" in out
    # And the original is kept, not overwritten in place.
    assert list(tmp_path.glob("train.jsonl.precompact.*"))


def test_compact_is_idempotent(tmp_path, monkeypatch):
    import json

    from symbio import constants

    train_file = tmp_path / "train.jsonl"
    train_file.write_text(
        json.dumps({"text": "system: clean\nassistant: hi"}) + "\n", encoding="utf-8")
    monkeypatch.setattr(constants, "TRAIN_FILE", train_file)
    monkeypatch.setattr(constants, "VALID_FILE", tmp_path / "valid.jsonl")
    assert training.compact_existing_samples()["train.jsonl"] == 0
    # Nothing to do means no backup churn.
    assert not list(tmp_path.glob("train.jsonl.precompact.*"))


def test_migration_keeps_instructions_that_follow_the_prose_mention(
        tmp_path, monkeypatch):
    """The regex must anchor on the catalog, not the prose that names it.

    The prompt says "the <tools> catalog at the bottom of this message" long
    before the real block. A non-greedy pattern starting at that mention runs
    to the closing tag at the very end and deletes every instruction between
    them — which is exactly what happened to all 114 samples.
    """
    import json

    from symbio import constants

    sample = (
        "system: Trust rules first.\n"
        "The <tools> catalog at the bottom lists every tool.\n"
        "Legacy short tags still work: <note title='T'>body</note>\n"
        "NEVER include internal reasoning.\n\n"
        '<tools>[{"name":"terminal","parameters":{"x":1}}]</tools>\n'
        "user: hi\nassistant: hello"
    )
    train_file = tmp_path / "train.jsonl"
    train_file.write_text(json.dumps({"text": sample}) + "\n", encoding="utf-8")
    monkeypatch.setattr(constants, "TRAIN_FILE", train_file)
    monkeypatch.setattr(constants, "VALID_FILE", tmp_path / "valid.jsonl")

    training.compact_existing_samples()
    out = json.loads(train_file.read_text().strip())["text"]

    # The catalog is gone...
    assert "</tools>" not in out
    assert '"parameters"' not in out
    # ...and everything that sat between the mention and the block survives.
    assert "Legacy short tags still work" in out
    assert "NEVER include internal reasoning" in out
    assert "<note title='T'>body</note>" in out
    assert "Trust rules first." in out
    assert "hello" in out
