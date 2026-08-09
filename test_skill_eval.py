"""Tests for the three-way skill adapter evaluation.

The point of skill_eval is to be a number a skeptic can trust, so these
tests mostly guard against ways the harness could flatter the adapter:
leaking the steps into the adapter arm's prompt, grading with a metric
that always passes, or reporting a win that the arms don't support.
"""

import json

import pytest

from symbio import constants
from symbio.app import skill_eval


STEPS = "1. Toggle wifi off. 2. Wait ten seconds. 3. Toggle wifi back on."
SKILL_NAME = "Fix wifi"
ROLE = "fix_wifi"


@pytest.fixture
def skill_entry(tmp_path, monkeypatch):
    """A catalog entry for one skill, isolated from the real catalog."""
    from symbio.app import skills

    catalog_file = tmp_path / "worker_models.json"
    monkeypatch.setattr(constants, "WORKER_MODELS_FILE", catalog_file)
    system_prompt = skills._build_skill_system_prompt(SKILL_NAME, STEPS)
    catalog = {
        f"skill_{ROLE}": {
            "model_name": "fake/model",
            "role": ROLE,
            "description": f"Skill: {SKILL_NAME}",
            "adapter_compatible": True,
            "system_prompt": system_prompt,
            "is_skill": True,
            "skill_name": SKILL_NAME,
        }
    }
    catalog_file.write_text(json.dumps(catalog), encoding="utf-8")
    return catalog[f"skill_{ROLE}"]


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=True, enable_thinking=False):
        return "\n".join(f"<|{m['role']}|>{m['content']}" for m in messages)


class FakeModel:
    """Stands in for a loaded MLX model; carries which arm loaded it."""

    def __init__(self, with_adapter: bool):
        self.with_adapter = with_adapter


def make_loader(record: list):
    def load_fn(model_name, adapter_path=None):
        record.append({"model_name": model_name, "adapter_path": adapter_path})
        return FakeModel(adapter_path is not None), FakeTokenizer()
    return load_fn


def make_generator(replies: dict[bool, str], seen: list | None = None):
    """Generate a canned reply keyed on whether the adapter was loaded."""
    def generate_fn(model, tokenizer, prompt="", sampler=None,
                    max_tokens=0, verbose=False):
        if seen is not None:
            seen.append(prompt)
        return replies[model.with_adapter]
    return generate_fn


# --- the metric itself ------------------------------------------------


def test_keywords_drop_stopwords_and_keep_procedure_words():
    kws = skill_eval._keywords(STEPS)
    assert "toggle" in kws
    assert "wifi" in kws
    # "wait" is procedural and must survive.
    assert "wait" in kws
    # State words are the operative content of a toggle procedure.
    assert "off" in kws and "on" in kws
    # Filler carries no procedural signal.
    assert "the" not in kws
    assert "it" not in kws


def test_keywords_ignore_trailing_punctuation():
    """A reply must not need matching full stops to get credit."""
    kws = skill_eval._keywords("Toggle wifi off. Toggle it on.")
    assert "off" in kws and "on." not in kws and "off." not in kws
    # Phrased without the periods, the reply still covers the vocabulary.
    assert skill_eval.coverage("toggle wifi off and then on again", kws) == 1.0


def test_keywords_keep_interior_punctuation():
    kws = skill_eval._keywords("Check interface en0.1 is well-formed.")
    assert "en0.1" in kws
    assert "well-formed" in kws


def test_keywords_are_deduped_preserving_order():
    kws = skill_eval._keywords("alpha beta alpha gamma beta")
    assert kws == ["alpha", "beta", "gamma"]


def test_coverage_is_zero_for_unrelated_reply():
    kws = skill_eval._keywords(STEPS)
    assert skill_eval.coverage("I have no idea what you mean.", kws) == 0.0


def test_coverage_is_one_when_reply_repeats_the_steps():
    kws = skill_eval._keywords(STEPS)
    assert skill_eval.coverage(STEPS, kws) == 1.0


def test_coverage_of_empty_keywords_is_zero_not_one():
    """A skill with no extractable steps must not score a free pass."""
    assert skill_eval.coverage("anything at all", []) == 0.0


# --- the control that makes the result meaningful ---------------------


def test_adapter_arm_prompt_never_contains_the_steps():
    """The whole experiment collapses if the adapter is told the procedure."""
    stripped = skill_eval._stripped_skill_system_prompt(SKILL_NAME)
    assert SKILL_NAME in stripped
    assert "Toggle wifi off" not in stripped
    assert "Steps:" not in stripped
    for token in ("ten seconds", "1.", "2.", "3."):
        assert token not in stripped


def test_stripped_prompt_keeps_the_trained_framing():
    """Adapter must be evaluated under the sentence it was trained beneath."""
    from symbio.app import skills

    trained = skills._build_skill_system_prompt(SKILL_NAME, STEPS)
    stripped = skill_eval._stripped_skill_system_prompt(SKILL_NAME)
    opener = f"You are the specialist worker for the skill '{SKILL_NAME}'."
    assert trained.startswith(opener)
    assert stripped.startswith(opener)


def test_prompted_arm_does_contain_the_steps(skill_entry):
    """The skeptic's baseline must genuinely get the steps handed to it."""
    assert "Toggle wifi off" in skill_entry["system_prompt"]


# --- task loading -----------------------------------------------------


def test_default_tasks_do_not_reuse_the_seed_wording():
    """Generated tasks must differ from skills.py's two training samples."""
    tasks = skill_eval.default_tasks(SKILL_NAME)
    prompts = [t.prompt for t in tasks]
    assert f"Apply the skill '{SKILL_NAME}'." not in prompts
    assert f"How do I perform '{SKILL_NAME}'?" not in prompts
    assert len(tasks) >= 3


def test_custom_tasks_are_loaded_when_present(tmp_path, monkeypatch):
    monkeypatch.setattr(constants, "DATA_DIR", tmp_path)
    path = skill_eval.tasks_path_for(ROLE)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([
        {"id": "a", "prompt": "my wifi died", "must_include": ["toggle"]},
        "plain string task",
    ]), encoding="utf-8")
    tasks, custom = skill_eval.load_tasks(ROLE, SKILL_NAME)
    assert custom is True
    assert tasks[0].id == "a"
    assert tasks[0].must_include == ["toggle"]
    assert tasks[1].prompt == "plain string task"


def test_malformed_task_file_falls_back_to_generated(tmp_path, monkeypatch):
    monkeypatch.setattr(constants, "DATA_DIR", tmp_path)
    path = skill_eval.tasks_path_for(ROLE)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json", encoding="utf-8")
    tasks, custom = skill_eval.load_tasks(ROLE, SKILL_NAME)
    assert custom is False
    assert tasks


# --- resolution -------------------------------------------------------


def test_resolve_skill_by_role_name_and_key(skill_entry):
    assert skill_eval.resolve_skill(ROLE)["skill_name"] == SKILL_NAME
    assert skill_eval.resolve_skill(SKILL_NAME)["role"] == ROLE
    assert skill_eval.resolve_skill("fix WIFI")["role"] == ROLE
    assert skill_eval.resolve_skill(f"skill_{ROLE}")["role"] == ROLE


def test_resolve_skill_returns_none_for_unknown(skill_entry):
    assert skill_eval.resolve_skill("no such skill") is None


def test_steps_recovered_from_system_prompt(skill_entry, tmp_path, monkeypatch):
    monkeypatch.setattr(constants, "NOTES_DIR", tmp_path / "empty_notes")
    (tmp_path / "empty_notes").mkdir()
    assert skill_eval.skill_steps(skill_entry).strip() == STEPS


# --- end to end -------------------------------------------------------


def _run(monkeypatch, tmp_path, base_reply, adapter_reply, adapter_exists=True):
    monkeypatch.setattr(constants, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(constants, "NOTES_DIR", tmp_path / "notes")
    (tmp_path / "notes").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(constants, "WORKER_ADAPTERS_DIR", tmp_path / "adapters")
    if adapter_exists:
        adir = tmp_path / "adapters" / ROLE
        adir.mkdir(parents=True, exist_ok=True)
        (adir / "adapter_config.json").write_text("{}", encoding="utf-8")
        # Weights too — a config alone is a half-written adapter.
        (adir / "adapters.safetensors").write_bytes(b"\x00")

    loads: list = []
    out = tmp_path / "report.json"
    skill_eval.run_skill_eval(
        ROLE,
        config={"model_name": "fake/model", "assistant_name": "Symbio",
                "user_name": "Huy",
                "agent": {"temperature": 0.7, "top_p": 0.9}},
        output_path=out,
        generate_fn=make_generator({False: base_reply, True: adapter_reply}),
        load_fn=make_loader(loads),
    )
    return json.loads(out.read_text(encoding="utf-8")), loads


def test_end_to_end_reports_a_win_when_adapter_recalls_steps(
        skill_entry, tmp_path, monkeypatch):
    report, loads = _run(
        monkeypatch, tmp_path,
        base_reply="I'm not sure what that involves.",
        adapter_reply=STEPS,
    )
    assert report["arms"]["base"]["score"] == 0
    assert report["arms"]["adapter"]["score"] == report["arms"]["adapter"]["total"]
    assert report["learned_delta"] > 0
    assert "recovered the procedure" in report["verdict"]


def test_end_to_end_reports_no_learning_when_adapter_is_blank(
        skill_entry, tmp_path, monkeypatch):
    """A useless adapter must produce an honest negative, not a hedge."""
    report, _ = _run(
        monkeypatch, tmp_path,
        base_reply="No idea.",
        adapter_reply="No idea.",
    )
    assert report["learned_delta"] == 0
    assert "No evidence of learning" in report["verdict"]


def test_adapter_arm_loads_the_adapter_and_base_arms_do_not(
        skill_entry, tmp_path, monkeypatch):
    _, loads = _run(monkeypatch, tmp_path, "nope", STEPS)
    # Two loads total: one shared by base+prompted, one for the adapter.
    assert len(loads) == 2
    assert loads[0]["adapter_path"] is None
    assert loads[1]["adapter_path"] is not None
    assert ROLE in loads[1]["adapter_path"]


def test_missing_adapter_still_runs_but_is_flagged(
        skill_entry, tmp_path, monkeypatch, capsys):
    report, loads = _run(monkeypatch, tmp_path, "nope", "nope",
                         adapter_exists=False)
    assert report["adapter_present"] is False
    assert all(ld["adapter_path"] is None for ld in loads)
    assert "WARNING" in capsys.readouterr().out


def test_report_records_both_deltas(skill_entry, tmp_path, monkeypatch):
    report, _ = _run(monkeypatch, tmp_path, "nope", STEPS)
    assert "learned_delta" in report
    assert "vs_prompted_delta" in report
    assert "vs_prompted_coverage_delta" in report


def test_report_saves_raw_outputs_for_auditing(skill_entry, tmp_path, monkeypatch):
    report, _ = _run(monkeypatch, tmp_path, "nope", STEPS)
    for arm in ("base", "prompted", "adapter"):
        for task in report["arms"][arm]["tasks"]:
            assert "output" in task
            assert "prompt" in task


def test_arms_can_be_subset(skill_entry, tmp_path, monkeypatch):
    monkeypatch.setattr(constants, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(constants, "NOTES_DIR", tmp_path / "notes")
    (tmp_path / "notes").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(constants, "WORKER_ADAPTERS_DIR", tmp_path / "adapters")
    loads: list = []
    out = tmp_path / "r.json"
    skill_eval.run_skill_eval(
        ROLE,
        config={"model_name": "fake/model", "agent": {"temperature": 0.7, "top_p": 0.9}},
        output_path=out,
        generate_fn=make_generator({False: "x", True: "y"}),
        load_fn=make_loader(loads),
        arms=(skill_eval.ARM_ADAPTER,),
    )
    report = json.loads(out.read_text(encoding="utf-8"))
    assert set(report["arms"]) == {"adapter"}
    assert len(loads) == 1
    assert "Incomplete" in report["verdict"]


def test_unknown_skill_raises(skill_entry, tmp_path):
    with pytest.raises(ValueError, match="No skill adapter named"):
        skill_eval.run_skill_eval("nonexistent", config={})


# --- guards against false positives -----------------------------------


def test_config_only_adapter_dir_is_not_usable(tmp_path):
    """A killed training run leaves a config with no weights beside it."""
    adir = tmp_path / "half"
    adir.mkdir()
    (adir / "adapter_config.json").write_text("{}", encoding="utf-8")
    assert skill_eval.adapter_is_usable(adir) is False


def test_adapter_dir_with_weights_is_usable(tmp_path):
    adir = tmp_path / "full"
    adir.mkdir()
    (adir / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adir / "adapters.safetensors").write_bytes(b"\x00")
    assert skill_eval.adapter_is_usable(adir) is True


def test_half_written_adapter_is_reported_absent_not_loaded(
        skill_entry, tmp_path, monkeypatch, capsys):
    """The report must not claim an adapter it never actually loaded."""
    monkeypatch.setattr(constants, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(constants, "NOTES_DIR", tmp_path / "notes")
    (tmp_path / "notes").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(constants, "WORKER_ADAPTERS_DIR", tmp_path / "adapters")
    adir = tmp_path / "adapters" / ROLE
    adir.mkdir(parents=True)
    (adir / "adapter_config.json").write_text("{}", encoding="utf-8")

    loads: list = []
    out = tmp_path / "r.json"
    skill_eval.run_skill_eval(
        ROLE,
        config={"model_name": "fake/model", "agent": {"temperature": 0.7, "top_p": 0.9}},
        output_path=out,
        generate_fn=make_generator({False: "nope", True: "nope"}),
        load_fn=make_loader(loads),
    )
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["adapter_present"] is False
    assert all(ld["adapter_path"] is None for ld in loads)
    assert "WARNING" in capsys.readouterr().out


# ---- step ordering ----
#
# coverage is set membership, so a procedure recited out of order scores a
# perfect 100%. For a runbook that is the difference between working and not.

_PROC = ("1. Freeze the scope with `symb-keyctl freeze --scope edge`. "
         "2. Wait until the drain counter reads zero. "
         "3. Mint the replacement with `symb-keyctl mint --ttl 36h`. "
         "4. Publish it to the vault path secret/edge/rotating. "
         "5. Release the scope with `symb-keyctl thaw`.")


def test_split_steps_finds_every_numbered_step():
    assert len(skill_eval.split_steps(_PROC)) == 5


def test_each_step_gets_a_distinguishing_anchor():
    assert skill_eval.step_anchors(_PROC) == [
        "freeze", "wait", "mint", "publish", "release"]


def test_a_correctly_ordered_recital_scores_full_marks():
    assert skill_eval.order_score(_PROC, _PROC) == 1.0


def test_swapped_steps_are_caught_even_at_full_coverage():
    """The exact defect the coverage metric cannot see."""
    swapped = ("1. Freeze the scope. 2. Wait until the drain counter reads zero. "
               "3. Mint the replacement with ttl 36h. 4. Release the scope with thaw. "
               "5. Publish it to the vault path secret/edge/rotating.")
    keywords = skill_eval._keywords(_PROC)
    assert skill_eval.coverage(swapped, keywords) > 0.9, "coverage stays high"
    assert skill_eval.order_score(swapped, _PROC) < 1.0, "order must drop"


def test_a_fully_reversed_recital_scores_low():
    reversed_proc = " ".join(reversed(skill_eval.split_steps(_PROC)))
    assert skill_eval.order_score(reversed_proc, _PROC) <= 0.4


def test_order_is_unjudgeable_when_too_little_is_mentioned():
    assert skill_eval.order_score("Freeze the scope.", _PROC) is None
    assert skill_eval.order_score("nothing relevant here", _PROC) is None


def test_a_single_step_procedure_has_no_order_to_judge():
    assert skill_eval.step_anchors("1. Just do the thing.") == []
    assert skill_eval.order_score("anything", "1. Just do the thing.") is None


# ---- derived golden cases for skills ----
#
# Skills retrain themselves off accumulated usage samples with nobody watching,
# and until now had no golden cases at all: guarded_train_worker looked the role
# up in WORKER_GOLDEN_CASES, got None, and took the early return with no check
# and no rollback. Unlike the headmaster's, a skill's cases can be derived,
# because its correct answer is its own steps.

_WIFI = "1. Toggle wifi off. 2. Toggle it on."


def test_a_skill_gets_one_case_per_held_out_paraphrase():
    cases = skill_eval.skill_golden_cases("Fix wifi", _WIFI)
    assert len(cases) == len(skill_eval.default_tasks("Fix wifi"))
    assert all(c.id.startswith("skill_") for c in cases)


def test_derived_cases_carry_the_steps_as_the_ideal_reply():
    """Lets golden's remedy path inject real training samples on regression."""
    for case in skill_eval.skill_golden_cases("Fix wifi", _WIFI):
        assert case.ideal_reply == _WIFI


def test_a_correct_recital_passes():
    case = skill_eval.skill_golden_cases("Fix wifi", _WIFI)[0]
    assert case.check("1. Toggle wifi off. 2. Toggle it on.", [], {}) is True


def test_a_reversed_procedure_fails_even_though_every_word_is_present():
    """The regression this gate exists for: right words, unrunnable order."""
    case = skill_eval.skill_golden_cases("Fix wifi", _WIFI)[0]
    assert case.check("2. Toggle it on. 1. Toggle wifi off.", [], {}) is False


def test_an_unrelated_answer_fails():
    case = skill_eval.skill_golden_cases("Fix wifi", _WIFI)[0]
    assert case.check("Have you tried restarting your router?", [], {}) is False


def test_a_degenerate_reply_fails():
    case = skill_eval.skill_golden_cases("Fix wifi", _WIFI)[0]
    assert case.check("wifi wifi wifi wifi wifi wifi wifi wifi", [], {}) is False


def test_a_skill_with_no_recoverable_steps_yields_no_cases():
    assert skill_eval.skill_golden_cases("Empty", "") == []
    assert skill_eval.skill_golden_cases("Empty", "   ") == []


def test_order_anchors_match_whole_tokens_only():
    """'on' must not count inside 'configuration'; this gates rollbacks."""
    steps = "1. Unplug the unit. 2. On."
    assert skill_eval.step_anchors(steps) == ["unplug", "on"]

    # Step 2's anchor appears only as a substring of an unrelated word, so
    # only one step is really mentioned and order is unjudgeable.
    assert skill_eval.order_score(
        "1. Unplug the unit. 2. Check the configuration.", steps) is None
    # Spelled as its own token, it counts.
    assert skill_eval.order_score("Unplug the unit, then switch On.", steps) == 1.0
