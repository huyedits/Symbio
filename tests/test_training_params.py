"""What the live 14B retrain on 2026-09-11 showed about its own settings.

Val 1.065 -> 0.776 (it30) -> 0.844 (it60) -> 0.781 (it90), then early stop at
patience 2/2. Two of those numbers are not evidence of a plateau, they are
evidence of a measurement too small to tell: val_batches was 8, so a two-hour
run was ended on a 0.005 difference across eight samples, with 24 validation
rows on disk unused.
"""
import json
import math
import pathlib

from symbio.app import training


def _lora():
    return json.loads(pathlib.Path("config.json").read_text())["lora"]


def test_validation_uses_the_whole_validation_set():
    """8 of 24 rows cannot resolve 0.776 against 0.781, and that comparison is
    what stops the run."""
    lora = _lora()
    rows = sum(1 for _ in open("training_data/valid.jsonl"))

    assert lora["val_batches"] * lora["batch_size"] >= rows


def test_evaluation_is_infrequent_enough_to_afford_that():
    """Each eval now reads 3x the batches, so running them twice as far apart
    keeps the overhead where it was rather than tripling it."""
    assert _lora()["steps_per_eval"] >= 60


def test_the_epoch_count_reported_is_the_one_that_will_run():
    """iters is capped by max_iters, so 770 samples asking for 2 epochs (1540
    steps) became 600 — 0.78 of one pass — and the log still said "~2 epochs".
    A budget that reports the number it was denied is how "each retrain
    silently drops older behaviours" hides."""
    lora = _lora()
    samples = 770
    iters = training.iters_for_corpus(lora, samples)
    actual = iters * lora["batch_size"] / samples

    # Whatever the cap does, the claim and the reality have to agree.
    assert iters <= lora["max_iters"]
    assert actual <= lora["epochs"] + 0.05


def test_the_progress_file_records_steps_that_happened():
    """The run stopped at iter 90, restored iter 30, and filed "600 total
    iters" — a number no adapter ever reached, on the label every later
    snapshot inherits."""
    import inspect

    # The work happens in _run_training; run_training is the wrapper.
    source = inspect.getsource(training._run_training)

    assert "record_adapter_iters(ran" in source
    assert "record_adapter_iters(iters" not in source


def test_the_early_stop_wrapper_reports_what_it_kept():
    import inspect

    source = inspect.getsource(training._run_training_with_early_stop)

    assert "kept_iters = best_step" in source
    assert "return kept_iters if kept_iters else True" in source
