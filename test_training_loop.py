"""Guards for the two ways a training run went wrong on this machine.

One took the machine down: the skill-adapter path spawned `mlx_lm lora` while
the chat process still held the 8B model and a golden-baseline copy of it, and
the process peaked at 16.2 GB against 15.7 GB of RAM. macOS killed it and the
Mac rebooted.

The other made every run since a waste: samples were stored as rendered text,
which mlx_lm trains on unmasked, so ~99% of the loss came from reproducing the
system prompt carried in every sample rather than the answer it was supposed
to teach.
"""
import json
import threading
from pathlib import Path

import pytest

from symbio import constants
from symbio.app import config as app_config
from symbio.app import training


class ChatMLTokenizer:
    """A stand-in that renders the way Qwen's template does.

    The reasoning block matters: the template emits it around the assistant
    turn, so it appears in rendered text but not in the content that produced
    it. Getting that wrong is what made the first version of the corpus
    migration silently upgrade nothing.
    """

    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=False, enable_thinking=False,
                            tools=None, return_dict=False):
        out = []
        for m in messages:
            body = m["content"]
            if m["role"] == "assistant":
                body = "<think>\n\n</think>\n\n" + body
            out.append(f"<|im_start|>{m['role']}\n{body}<|im_end|>\n")
        if add_generation_prompt:
            out.append("<|im_start|>assistant\n")
        return "".join(out)

    def encode(self, text, add_special_tokens=True):
        return text.split(" ")


def _render(tok, system, user, assistant):
    return tok.apply_chat_template([
        {"role": "system", "content": system},
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ])


# A sample carrying another model's turn markers. Real: these were found in
# this project's own valid.jsonl, left behind by an earlier model.
FOREIGN = ("<|user|>How much disk space do I have?<|end|>"
           "<|assistant|>Checking disk space.<|end|><|endoftext|>")


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    """Point the training data files at a scratch directory."""
    monkeypatch.setattr(constants, "TRAIN_FILE", tmp_path / "train.jsonl")
    monkeypatch.setattr(constants, "VALID_FILE", tmp_path / "valid.jsonl")
    return tmp_path


def _write(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n",
                    encoding="utf-8")


def _records(path):
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()
            if l.strip()]


# ---- prompt masking ----

def test_sample_carries_both_rendered_text_and_messages(corpus):
    """mlx_lm reads `messages`; everything in this module reads `text`."""
    training.append_chat_pair("Who are you?", "I am Caine.", ChatMLTokenizer(),
                              "SYS")
    record = _records(constants.TRAIN_FILE)[0]
    assert "I am Caine." in record["text"]
    assert [m["role"] for m in record["messages"]] == [
        "system", "user", "assistant"]
    assert record["messages"][-1]["content"] == "I am Caine."


def test_stored_messages_render_back_to_the_stored_text(corpus):
    """The mask offset is computed from `messages`, so if they render to
    something other than `text` the mask covers the wrong tokens."""
    tok = ChatMLTokenizer()
    training.append_chat_pair("Q?", "A.", tok, "SYS")
    record = _records(constants.TRAIN_FILE)[0]
    assert training.render_messages(record["messages"], tok) == record["text"]


def test_legacy_text_only_samples_are_upgraded(corpus):
    tok = ChatMLTokenizer()
    _write(constants.TRAIN_FILE, [{"text": _render(tok, "SYS", "Q?", "A.")}])
    assert training._supports_prompt_masking() is False

    counts = training.upgrade_corpus_to_messages(tok)

    assert counts == {"upgraded": 1, "left": 0}
    assert training._supports_prompt_masking() is True
    messages = _records(constants.TRAIN_FILE)[0]["messages"]
    # The template's own reasoning block must not be carried into the content,
    # or re-rendering stacks a second one on top of it.
    assert messages[-1]["content"] == "A."


def test_masking_stays_off_when_a_sample_cannot_be_parsed(corpus):
    """Better unmasked than masked in the wrong place."""
    tok = ChatMLTokenizer()
    _write(constants.TRAIN_FILE, [
        {"text": _render(tok, "SYS", "Q?", "A.")},
        {"text": "not a chat template at all"},
    ])

    counts = training.upgrade_corpus_to_messages(tok)

    assert counts == {"upgraded": 1, "left": 1}
    assert training._supports_prompt_masking() is False


def test_upgrade_declines_without_a_tokenizer(corpus):
    """No template means no way to verify the round-trip."""
    _write(constants.TRAIN_FILE,
           [{"text": _render(ChatMLTokenizer(), "SYS", "Q?", "A.")}])

    assert training.upgrade_corpus_to_messages(None) == {"upgraded": 0, "left": 1}
    assert training._supports_prompt_masking() is False


# ---- foreign chat templates ----

def test_foreign_template_samples_are_dropped(corpus):
    tok = ChatMLTokenizer()
    native = _render(tok, "SYS", "Q?", "A.")
    _write(constants.TRAIN_FILE, [{"text": native}, {"text": FOREIGN}])

    removed = training.drop_foreign_template_samples(tok)

    assert removed == {str(constants.TRAIN_FILE): [2]}
    assert [r["text"] for r in _records(constants.TRAIN_FILE)] == [native]


def test_a_wholly_foreign_validation_file_is_cleared(corpus):
    """It is derived from the training data, so it can be rebuilt."""
    tok = ChatMLTokenizer()
    _write(constants.TRAIN_FILE, [{"text": _render(tok, "SYS", "Q?", "A.")}])
    _write(constants.VALID_FILE, [{"text": FOREIGN}])

    training.drop_foreign_template_samples(tok)

    assert constants.VALID_FILE.read_text(encoding="utf-8") == ""
    training.ensure_validation_split()
    assert len(_records(constants.VALID_FILE)) == 1


def test_a_wholly_foreign_training_file_is_left_alone(corpus):
    """Rejecting every sample means the rule is broken, not the corpus.

    Clearing the training file here would destroy the only copy of the user's
    corpus on the strength of a parser bug.
    """
    tok = ChatMLTokenizer()
    _write(constants.TRAIN_FILE, [{"text": FOREIGN}, {"text": FOREIGN}])

    removed = training.drop_foreign_template_samples(tok)

    assert removed == {}
    assert len(_records(constants.TRAIN_FILE)) == 2


def test_already_structured_samples_are_never_dropped(corpus):
    """They carry their messages, so there is nothing to re-derive."""
    tok = ChatMLTokenizer()
    training.append_chat_pair("Q?", "A.", tok, "SYS")
    before = _records(constants.TRAIN_FILE)

    assert training.drop_foreign_template_samples(tok) == {}
    assert _records(constants.TRAIN_FILE) == before


# ---- iterations scaled to the corpus ----

def test_iters_cover_the_corpus_a_whole_number_of_times():
    lora = {"epochs": 2, "batch_size": 1, "iters": 10, "max_iters": 2000}
    assert training.iters_for_corpus(lora, 219) == 438


def test_iters_respect_batch_size():
    lora = {"epochs": 2, "batch_size": 4, "iters": 10, "max_iters": 2000}
    assert training.iters_for_corpus(lora, 200) == 100


def test_configured_iters_are_a_floor_for_small_corpora():
    """Two passes over six samples is not enough steps to converge."""
    lora = {"epochs": 2, "batch_size": 1, "iters": 150, "max_iters": 2000}
    assert training.iters_for_corpus(lora, 6) == 150


def test_max_iters_caps_a_large_corpus():
    lora = {"epochs": 2, "batch_size": 1, "iters": 150, "max_iters": 500}
    assert training.iters_for_corpus(lora, 100_000) == 500


def test_empty_corpus_falls_back_to_the_configured_count():
    lora = {"epochs": 2, "batch_size": 1, "iters": 150, "max_iters": 2000}
    assert training.iters_for_corpus(lora, 0) == 150


def test_max_iters_outranks_the_floor():
    """Otherwise `max_iters` cannot do the one thing it exists for."""
    lora = {"epochs": 2, "batch_size": 1, "iters": 300, "max_iters": 200}
    assert training.iters_for_corpus(lora, 6) == 200
    assert training.iters_for_corpus(lora, 0) == 200


# ---- LoRA target keys ----

def test_block_relative_keys_are_accepted():
    assert training.validate_lora_keys(
        ["self_attn.q_proj", "self_attn.v_proj"], 36) == []


def test_full_paths_are_accepted_within_range():
    assert training.validate_lora_keys(
        ["model.layers.0.self_attn.q_proj", "model.layers.35.mlp.up_proj"], 36) == []


def test_nested_multimodal_paths_are_accepted():
    """Qwen3.5-9B keeps its text stack under `language_model.model.layers.N`.

    A validator that only knew the flat `model.layers.N` form would reject the
    one naming that actually works for these models.
    """
    assert training.validate_lora_keys(
        ["language_model.model.layers.31.self_attn.q_proj",
         "language_model.model.layers.23.linear_attn.in_proj_qkv"], 32) == []


def test_block_count_reads_a_nested_text_config(tmp_path, monkeypatch):
    """Multimodal configs describe the wrapper at the top level."""
    repo = tmp_path / "model"
    repo.mkdir()
    (repo / "config.json").write_text(
        json.dumps({"model_type": "qwen3_5",
                    "text_config": {"num_hidden_layers": 32}}), encoding="utf-8")
    monkeypatch.setattr(training, "_model_repo_dir", lambda name: repo)

    assert training.model_block_count("anything") == 32


def test_a_block_index_past_the_end_is_reported():
    """Out of range matches nothing, and mlx_lm would not say so."""
    problems = training.validate_lora_keys(["model.layers.40.mlp.up_proj"], 36)
    assert len(problems) == 1
    assert "block 40" in problems[0] and "36 blocks" in problems[0]


def test_mixing_relative_and_full_paths_is_reported():
    """The relative key would also apply across every selected block."""
    problems = training.validate_lora_keys(
        ["model.layers.5.self_attn.q_proj", "self_attn.v_proj"], 36)
    assert any("mixes" in p for p in problems)


def test_a_near_miss_full_path_is_reported():
    """'layers.5...' without the 'model.' prefix matches neither namespace."""
    problems = training.validate_lora_keys(["layers.5.self_attn.q_proj"], 36)
    assert any("match nothing" in p for p in problems)


def test_a_bare_module_name_is_reported():
    problems = training.validate_lora_keys(["q_proj"], 36)
    assert any("match nothing" in p for p in problems)


def test_no_keys_means_no_complaints():
    assert training.validate_lora_keys(None, 36) == []
    assert training.validate_lora_keys([], 36) == []


def test_range_is_not_checked_without_a_known_block_count():
    """An uncached model is not evidence the key is wrong."""
    assert training.validate_lora_keys(["model.layers.999.mlp.up_proj"], None) == []


# ---- memory guards ----

def _config(checkpoint=False, **gpu):
    return {"model_name": "some/model", "gpu": gpu,
            "lora": {"grad_checkpoint": checkpoint}}


def test_training_is_refused_when_memory_cannot_hold_it(monkeypatch):
    monkeypatch.setattr(training, "_model_weight_bytes", lambda name: 4_000_000_000)
    monkeypatch.setattr(training, "free_ram_bytes", lambda: 2_000_000_000)

    message = training._memory_shortfall(_config())

    assert message is not None
    assert "14.4 GB" in message and "2.0 GB" in message


def test_training_proceeds_with_room_to_spare(monkeypatch):
    monkeypatch.setattr(training, "_model_weight_bytes", lambda name: 4_000_000_000)
    monkeypatch.setattr(training, "free_ram_bytes", lambda: 16_000_000_000)

    assert training._memory_shortfall(_config()) is None


def test_checkpointing_lowers_what_the_run_is_expected_to_need(monkeypatch):
    """Measured: 15.58 GB without checkpointing, 8.22 GB with it."""
    monkeypatch.setattr(training, "_model_weight_bytes", lambda name: 4_000_000_000)
    monkeypatch.setattr(training, "free_ram_bytes", lambda: 9_000_000_000)

    assert training._memory_shortfall(_config()) is not None
    assert training._memory_shortfall(_config(checkpoint=True)) is None


def test_the_refusal_suggests_checkpointing_when_it_is_off(monkeypatch):
    monkeypatch.setattr(training, "_model_weight_bytes", lambda name: 4_000_000_000)
    monkeypatch.setattr(training, "free_ram_bytes", lambda: 2_000_000_000)

    assert "grad_checkpoint" in training._memory_shortfall(_config())
    assert "grad_checkpoint" not in training._memory_shortfall(_config(checkpoint=True))


def test_unknown_memory_is_not_treated_as_a_shortfall(monkeypatch):
    """A figure we cannot read is not evidence of a problem."""
    monkeypatch.setattr(training, "_model_weight_bytes", lambda name: None)
    monkeypatch.setattr(training, "free_ram_bytes", lambda: 1)
    assert training._memory_shortfall(_config()) is None

    monkeypatch.setattr(training, "_model_weight_bytes", lambda name: 4_000_000_000)
    monkeypatch.setattr(training, "free_ram_bytes", lambda: None)
    assert training._memory_shortfall(_config()) is None


def test_the_preflight_can_be_switched_off(monkeypatch):
    monkeypatch.setattr(training, "_model_weight_bytes", lambda name: 4_000_000_000)
    monkeypatch.setattr(training, "free_ram_bytes", lambda: 1)

    assert training._memory_shortfall(_config(memory_preflight=False)) is None


def test_a_plain_load_is_judged_by_what_a_load_costs(monkeypatch):
    """The golden-check copies loaded around a training run carry no optimiser
    state and no retained activations, so charging them the trainer's 3.6x
    would refuse loads that fit several times over."""
    monkeypatch.setattr(training, "_model_weight_bytes", lambda name: 4_000_000_000)
    monkeypatch.setattr(training, "free_ram_bytes", lambda: 6_000_000_000)

    assert training._memory_shortfall(_config()) is not None
    assert training.load_memory_shortfall(_config()) is None


def test_a_load_with_no_room_is_refused(monkeypatch):
    monkeypatch.setattr(training, "_model_weight_bytes", lambda name: 4_000_000_000)
    monkeypatch.setattr(training, "free_ram_bytes", lambda: 3_000_000_000)

    message = training.load_memory_shortfall(_config(), purpose="check a worker")

    assert message is not None
    assert "check a worker" in message
    assert "5.0 GB" in message and "3.0 GB" in message


def test_the_load_preflight_honours_the_same_switches(monkeypatch):
    monkeypatch.setattr(training, "free_ram_bytes", lambda: 1)
    monkeypatch.setattr(training, "_model_weight_bytes", lambda name: 4_000_000_000)
    assert training.load_memory_shortfall(_config(memory_preflight=False)) is None

    monkeypatch.setattr(training, "_model_weight_bytes", lambda name: None)
    assert training.load_memory_shortfall(_config()) is None


def test_free_ram_counts_reclaimable_pages(monkeypatch):
    """Inactive pages are populated but available, so ignoring them would
    refuse to train on a machine that is actually idle."""
    vm_stat = (
        "Mach Virtual Memory Statistics: (page size of 16384 bytes)\n"
        'Pages free:                    1000.\n'
        'Pages active:                 50000.\n'
        'Pages inactive:                2000.\n'
        'Pages speculative:              500.\n'
    )

    class Result:
        stdout = vm_stat

    monkeypatch.setattr(training.subprocess, "run", lambda *a, **k: Result())

    assert training.free_ram_bytes() == 3500 * 16384


def test_block_count_is_read_from_the_model_config(tmp_path, monkeypatch):
    repo = tmp_path / "model"
    repo.mkdir()
    (repo / "config.json").write_text(json.dumps({"num_hidden_layers": 36}),
                                      encoding="utf-8")
    monkeypatch.setattr(training, "_model_repo_dir", lambda name: repo)

    assert training.model_block_count("anything") == 36


def test_block_count_is_none_for_an_uncached_model(monkeypatch):
    monkeypatch.setattr(training, "_model_repo_dir", lambda name: None)
    assert training.model_block_count("nope/not-here") is None


def _run_and_capture_trainer_args(corpus, monkeypatch, lora_overrides,
                                  resume=False):
    """Run run_training against a stub trainer; return (argv, parsed yaml)."""
    training.append_chat_pair("Q?", "A.", ChatMLTokenizer(), "SYS")
    monkeypatch.setattr(training, "_memory_shortfall", lambda *a, **k: None)
    monkeypatch.setattr(training, "drop_foreign_template_samples", lambda *a, **k: {})
    monkeypatch.setattr(training, "upgrade_corpus_to_messages",
                        lambda *a, **k: {"upgraded": 0, "left": 0})
    monkeypatch.setattr(training, "_supports_prompt_masking", lambda **k: False)
    monkeypatch.setattr(training, "drop_degenerate_samples", lambda *a, **k: {})
    monkeypatch.setattr(training, "model_block_count", lambda name: 36)

    captured = {}

    def fake_run(cmd, check=False):
        import yaml
        captured["cmd"] = list(cmd)
        config_path = cmd[cmd.index("--config") + 1]
        with open(config_path, encoding="utf-8") as fh:
            captured["yaml"] = yaml.safe_load(fh)

        class Done:
            returncode = 0
        return Done()

    monkeypatch.setattr(training.subprocess, "run", fake_run)
    lora = {"rank": 8, "dropout": 0.0, "scale": 20.0, "num_layers": 8,
            "batch_size": 1, "learning_rate": 1e-4, "iters": 2, "epochs": 1,
            "max_iters": 10, "max_seq_length": 512, "steps_per_eval": 10,
            "save_every": 10, "early_stop_enabled": False}
    lora.update(lora_overrides)
    training.run_training({"model_name": "m", "gpu": {}, "lora": lora},
                          resume=resume)
    return captured["cmd"], captured["yaml"]


def test_lora_keys_reach_the_trainer_config(corpus, monkeypatch):
    keys = ["self_attn.q_proj", "self_attn.v_proj"]
    _, written = _run_and_capture_trainer_args(corpus, monkeypatch, {"keys": keys})

    assert written["lora_parameters"]["keys"] == keys


def test_no_keys_leaves_mlx_lm_defaulting_to_every_projection(corpus, monkeypatch):
    _, written = _run_and_capture_trainer_args(corpus, monkeypatch, {"keys": []})

    assert "keys" not in written["lora_parameters"]


def test_grad_checkpoint_is_passed_only_when_enabled(corpus, monkeypatch):
    cmd, _ = _run_and_capture_trainer_args(
        corpus, monkeypatch, {"grad_checkpoint": True})
    assert "--grad-checkpoint" in cmd

    cmd, _ = _run_and_capture_trainer_args(
        corpus, monkeypatch, {"grad_checkpoint": False})
    assert "--grad-checkpoint" not in cmd


def test_only_one_trainer_runs_at_a_time(corpus, monkeypatch):
    """Two concurrent trainers is two full copies of the weights.

    That is the shape of the crash this guards: a background skill-training
    thread starting a trainer while another run already had one.
    """
    training.append_chat_pair("Q?", "A.", ChatMLTokenizer(), "SYS")
    monkeypatch.setattr(training, "_memory_shortfall", lambda *a, **k: None)
    monkeypatch.setattr(training, "drop_foreign_template_samples",
                        lambda *a, **k: {})
    monkeypatch.setattr(training, "upgrade_corpus_to_messages",
                        lambda *a, **k: {"upgraded": 0, "left": 0})
    monkeypatch.setattr(training, "_supports_prompt_masking", lambda **k: False)
    monkeypatch.setattr(training, "drop_degenerate_samples", lambda *a, **k: {})

    overlap = []
    running = threading.Event()

    def fake_run(cmd, check=False):
        overlap.append(running.is_set())
        running.set()
        threading.Event().wait(0.05)
        running.clear()

        class Done:
            returncode = 0
        return Done()

    monkeypatch.setattr(training.subprocess, "run", fake_run)
    config = {"model_name": "m", "gpu": {},
              "lora": {"rank": 8, "dropout": 0.0, "scale": 20.0, "num_layers": 8,
                       "batch_size": 1, "learning_rate": 1e-4, "iters": 2,
                       "epochs": 1, "max_iters": 10, "max_seq_length": 512,
                       "steps_per_eval": 10, "save_every": 10,
                       "early_stop_enabled": False}}

    threads = [threading.Thread(target=training.run_training, args=(config,))
               for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert overlap == [False, False, False], (
        "a trainer started while another was still running")


def test_narrowing_the_lora_targets_lowers_the_estimate(monkeypatch):
    """Measured: 8.216 GB on every projection, 7.535 GB on attention only.

    An estimate blind to `keys` refuses runs that the very lever it should be
    recommending would have made fit.
    """
    monkeypatch.setattr(training, "_model_weight_bytes", lambda name: 4_000_000_000)
    monkeypatch.setattr(training, "free_ram_bytes", lambda: 7_900_000_000)

    everything = {"grad_checkpoint": True, "keys": []}
    attention = {"grad_checkpoint": True,
                 "keys": ["self_attn.q_proj", "self_attn.v_proj"]}

    assert training._memory_shortfall(
        {"model_name": "m", "gpu": {}, "lora": everything}) is not None
    assert training._memory_shortfall(
        {"model_name": "m", "gpu": {}, "lora": attention}) is None


def test_keys_including_mlp_are_not_treated_as_narrowed():
    """The MLP projections are what the saving comes from."""
    assert training._attention_only({"keys": ["self_attn.q_proj"]}) is True
    assert training._attention_only(
        {"keys": ["self_attn.q_proj", "mlp.up_proj"]}) is False
    assert training._attention_only({"keys": []}) is False


def test_every_overhead_combination_is_defined():
    for checkpoint in (True, False):
        for keys in ([], ["self_attn.q_proj"]):
            cfg = {"lora": {"grad_checkpoint": checkpoint, "keys": keys}}
            assert training._training_overhead(cfg) > 1.0


def test_validation_is_capped_so_it_cannot_dominate_the_run(corpus, monkeypatch):
    """mlx_lm defaults to 25 batches and rescores the whole split each time.

    At ~2,160-token samples that is ~11s a batch on an 8B, so a 438-iteration
    run spent about an hour inside evaluation. The number is a plateau
    detector, not a benchmark.
    """
    cmd, _ = _run_and_capture_trainer_args(corpus, monkeypatch, {})

    assert "--val-batches" in cmd
    assert cmd[cmd.index("--val-batches") + 1] == "8"


def test_validation_batch_count_is_configurable(corpus, monkeypatch):
    cmd, _ = _run_and_capture_trainer_args(corpus, monkeypatch, {"val_batches": 3})

    assert cmd[cmd.index("--val-batches") + 1] == "3"


# ---- resuming a run that came out weak ----
#
# Without --resume-adapter-file every run starts from random init and
# overwrites adapter_dir, so "train it a bit more" silently threw away the
# previous run instead of extending it. Resuming is only safe when the
# existing adapter has the shape this run will attach: mlx_lm loads the
# resume file into a freshly built LoRA, and a mismatch trains from scratch
# while still reporting a normal-looking loss.

_RESUME_LORA = {"rank": 8, "keys": ["self_attn.q_proj", "self_attn.v_proj"]}


def _adapter_at(tmp_path, name, *, model="Qwen/Qwen3-8B-MLX-4bit",
                num_layers=8, lora=None, weights=True, provenance=True):
    d = tmp_path / name
    d.mkdir(parents=True, exist_ok=True)
    if weights:
        (d / "adapters.safetensors").write_bytes(b"weights")
    if provenance:
        (d / "adapter_config.json").write_text(json.dumps({
            "model": model,
            "num_layers": num_layers,
            "lora_parameters": _RESUME_LORA if lora is None else lora,
        }), encoding="utf-8")
    return d


def _resume(adapter_dir):
    return training.resume_source(
        adapter_dir, "Qwen/Qwen3-8B-MLX-4bit", 8, _RESUME_LORA)


def test_resume_accepts_an_adapter_whose_recipe_matches(tmp_path):
    source, why = _resume(_adapter_at(tmp_path, "match"))

    assert source is not None
    assert source.name == "adapters.safetensors"
    assert why == "recipe matches"


def test_resume_refuses_an_adapter_from_a_different_model(tmp_path):
    source, why = _resume(
        _adapter_at(tmp_path, "other", model="mlx-community/Qwen3-4B-4bit"))

    assert source is None
    assert "Qwen3-4B-4bit" in why


def test_resume_refuses_an_adapter_with_different_lora_keys(tmp_path):
    """The drift that caused this whole investigation: narrowing the targets
    makes every older adapter a different shape."""
    source, why = _resume(_adapter_at(tmp_path, "keys", lora={"rank": 8}))

    assert source is None
    assert "keys" in why


def test_resume_refuses_an_adapter_with_a_different_rank(tmp_path):
    source, why = _resume(_adapter_at(
        tmp_path, "rank", lora={"rank": 16, "keys": _RESUME_LORA["keys"]}))

    assert source is None
    assert "rank" in why


def test_resume_refuses_an_adapter_with_a_different_layer_count(tmp_path):
    source, why = _resume(_adapter_at(tmp_path, "layers", num_layers=16))

    assert source is None
    assert "num_layers" in why


def test_resume_refuses_an_adapter_it_cannot_verify(tmp_path):
    """Missing provenance is not evidence the shapes match."""
    source, why = _resume(_adapter_at(tmp_path, "bare", provenance=False))

    assert source is None
    assert "cannot be verified" in why


def test_resume_is_a_no_op_when_there_is_nothing_to_resume_from(tmp_path):
    source, why = _resume(_adapter_at(tmp_path, "empty", weights=False))

    assert source is None
    assert "no existing adapter" in why


def test_training_does_not_resume_unless_asked(corpus, monkeypatch):
    cmd, _ = _run_and_capture_trainer_args(corpus, monkeypatch, {})

    assert "--resume-adapter-file" not in cmd


# ---- which checkpoint a stopped run hands back ----
#
# steps_per_eval and save_every rarely coincide, so the best-scoring step
# usually has no file and a fallback decides the run's output. Reaching for
# the newest checkpoint returns the most overfit adapter of the run, because
# early stopping fires precisely when validation loss has started climbing.

def _ckpts(*steps):
    return [Path(f"{s:07d}_adapters.safetensors") for s in steps]


def test_fallback_prefers_the_checkpoint_before_the_best_step():
    """Real case: best val at iter 60, checkpoints at 25/50/75/100.

    Iter 100 is 40 steps past the divergence with validation loss rising;
    iter 50 is the closest thing to the model that actually scored best.
    """
    chosen = training.checkpoint_at_or_before(_ckpts(25, 50, 75, 100), 60)

    assert training.checkpoint_step(chosen) == 50


def test_fallback_is_exact_when_the_best_step_was_saved():
    chosen = training.checkpoint_at_or_before(_ckpts(25, 50, 75, 100), 100)

    assert training.checkpoint_step(chosen) == 100


def test_fallback_takes_the_oldest_when_nothing_precedes_the_best_step():
    """Every candidate is past it, so the oldest is the least overfit."""
    chosen = training.checkpoint_at_or_before(_ckpts(25, 50), 10)

    assert training.checkpoint_step(chosen) == 25


def test_fallback_without_a_best_step_takes_the_oldest():
    chosen = training.checkpoint_at_or_before(_ckpts(25, 50, 75), None)

    assert training.checkpoint_step(chosen) == 25


def test_checkpoint_interval_divides_the_evaluation_interval():
    """Otherwise no evaluated step ever has a checkpoint of its own, and the
    fallback above decides every run's output instead of the best score."""
    lora = app_config.DEFAULT_CONFIG["lora"]

    assert lora["steps_per_eval"] % lora["save_every"] == 0


# ---- adapter backup/restore must not swallow the worker adapters ----
#
# The workers live inside the headmaster's own directory
# (adapters/workers/<role>), which made them collateral in every headmaster
# rollback: the snapshot copied them in, and the restore put that frozen copy
# back over everything they had learned since. Startup crash-recovery calls
# restore_adapter unattended, so this ran without anyone watching.

@pytest.fixture
def adapter_tree(monkeypatch, tmp_path):
    """A headmaster adapter with worker adapters nested underneath it."""
    root = tmp_path / "adapters"
    workers = root / "workers"
    (workers / "researcher").mkdir(parents=True)
    (root / "adapters.safetensors").write_text("headmaster v1")
    (root / "adapter_config.json").write_text("{}")
    (workers / "researcher" / "adapters.safetensors").write_text("researcher v1")
    monkeypatch.setattr(constants, "ADAPTER_DIR", root)
    monkeypatch.setattr(constants, "WORKER_ADAPTERS_DIR", workers)
    monkeypatch.setattr(training, "adapter_label", lambda role=None: "HEADMASTER")
    monkeypatch.setattr(training, "adapter_total_iters", lambda role=None: 125)
    return root, workers


def test_the_headmaster_backup_leaves_the_workers_out(adapter_tree):
    root, _ = adapter_tree

    backup = training.backup_adapter()

    assert (backup / "adapters.safetensors").read_text() == "headmaster v1"
    assert not (backup / "workers").exists(), (
        "a headmaster snapshot must not carry copies of the worker adapters")


def test_restoring_the_headmaster_does_not_touch_the_workers(adapter_tree):
    root, workers = adapter_tree
    backup = training.backup_adapter()
    # Time passes: the headmaster retrains, and so does a worker.
    (root / "adapters.safetensors").write_text("headmaster v2")
    (workers / "researcher" / "adapters.safetensors").write_text("researcher v2")
    (workers / "browser").mkdir()
    (workers / "browser" / "adapters.safetensors").write_text("browser v1")

    training.restore_adapter(backup)

    assert (root / "adapters.safetensors").read_text() == "headmaster v1"
    assert (workers / "researcher" / "adapters.safetensors").read_text() == (
        "researcher v2"), "the worker's later training must survive"
    assert (workers / "browser").exists(), (
        "a worker trained after the snapshot must not be deleted by a "
        "headmaster rollback")


def test_an_older_backup_carrying_workers_does_not_restore_them(adapter_tree):
    """Backups taken before this was fixed still have the stale tree inside —
    including the one sitting in the project root right now."""
    root, workers = adapter_tree
    backup = root.parent / "adapters.OLD.bak"
    (backup / "workers" / "researcher").mkdir(parents=True)
    (backup / "adapters.safetensors").write_text("headmaster v0")
    (backup / "workers" / "researcher" / "adapters.safetensors").write_text("stale")

    training.restore_adapter(backup)

    assert (root / "adapters.safetensors").read_text() == "headmaster v0"
    assert (workers / "researcher" / "adapters.safetensors").read_text() == (
        "researcher v1"), "the stale copy inside the backup must be ignored"


def test_a_worker_restore_is_unaffected(adapter_tree):
    """A worker's own directory has no nested tree to protect; its restore
    must still replace everything it holds."""
    root, workers = adapter_tree
    monkeypatch_role = workers / "researcher"
    backup = root.parent / "researcher.bak"
    backup.mkdir()
    (backup / "adapters.safetensors").write_text("researcher v0")
    (monkeypatch_role / "stray.json").write_text("left over")

    training.restore_adapter(backup, role="researcher")

    assert (monkeypatch_role / "adapters.safetensors").read_text() == "researcher v0"
    assert not (monkeypatch_role / "stray.json").exists()
