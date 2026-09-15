"""The perform battery: minting held-out cases and grading by execution.

The recall battery is covered by test_golden / test_skill_eval. What is tested
here is only what this file adds: that a case is minted from a value the ask
and the artifact share, that a memorised answer fails it, and that grading
runs the code rather than reading it.
"""

import json

import pytest

from symbio import constants
from symbio.app import skill_perform


CONFIG = {
    "sandbox": {"blocked_imports": ["os", "pathlib", "subprocess", "socket"]},
    "agent": {"code_timeout": 15, "max_output_len": 4000},
}


def _example(filename="catalog.html", rows=4):
    """A worked example shaped like the ones seeding produces."""
    request = f"Scrape {filename} and report how many rows it has."
    script = (
        f"with open('{filename}') as f:\n"
        f"    body = f.read()\n"
        f"print(body.count('<tr>'))\n"
    )
    output = f"```python\n{script}```"
    return request, output, True


@pytest.fixture
def sandbox_file(tmp_path, monkeypatch):
    """Redirect the sandbox so minting never touches the real one."""
    monkeypatch.setattr(constants, "SANDBOX_DIR", tmp_path)
    (tmp_path / "catalog.html").write_text("<tr>a</tr><tr>b</tr>", encoding="utf-8")
    return tmp_path


@pytest.fixture
def worker_dir(tmp_path, monkeypatch):
    d = tmp_path / "workers" / "scrape"
    d.mkdir(parents=True)
    monkeypatch.setattr(constants, "data_dir_for", lambda role=None: d)
    return d


# --------------------------------------------------------------------------
# Minting
# --------------------------------------------------------------------------

def test_it_swaps_a_value_the_request_and_the_script_share(sandbox_file):
    cases = skill_perform.mint_cases("scrape", [_example()], CONFIG)
    assert len(cases) == 1
    case = cases[0]
    assert "catalog.html" in case.forbidden
    assert case.required == ["ledger.html"] or case.required[0].endswith(".html")
    # The perturbed name is what the worker is asked for, in both halves.
    assert "catalog.html" not in case.prompt
    assert case.required[0] in case.prompt
    assert case.required[0] in case.expected_script


def test_a_value_only_in_the_script_is_not_swapped(sandbox_file):
    """Changing an implementation detail asks for a different procedure."""
    request = "Scrape the catalog and report the row count."
    output = "```python\nwith open('catalog.html') as f:\n    print(f.read().count('<tr>'))\n```"
    assert skill_perform.mint_cases("scrape", [(request, output, True)], CONFIG) == []


def test_an_unverified_example_is_never_minted(sandbox_file):
    """A blocked import at seed time means the perturbed twin cannot run either."""
    request, output, _ = _example()
    assert skill_perform.mint_cases("scrape", [(request, output, False)], CONFIG) == []


def test_the_input_file_is_staged_under_its_new_name(sandbox_file):
    case = skill_perform.mint_cases("scrape", [_example()], CONFIG)[0]
    new_name = case.required[0]
    assert (sandbox_file / new_name).exists(), "perturbed ask must have real input"
    assert (sandbox_file / new_name).read_text() == (sandbox_file / "catalog.html").read_text()
    assert case.fixtures == {"catalog.html": new_name}


def test_a_case_whose_substituted_script_breaks_is_dropped(sandbox_file, capsys):
    """The battery's claim is that a passing answer exists. Prove it or drop it."""
    request = "Read catalog.html and print the count."
    # References a name that will not exist after substitution renames nothing
    # it can reach -- the script raises whatever the values are.
    output = "```python\nopen('catalog.html')\nraise SystemExit(3)\n```"
    assert skill_perform.mint_cases("scrape", [(request, output, True)], CONFIG) == []
    assert "dropped" in capsys.readouterr().out


def test_minting_is_deterministic(sandbox_file):
    a = skill_perform.mint_cases("scrape", [_example()], CONFIG)[0]
    b = skill_perform.mint_cases("scrape", [_example()], CONFIG)[0]
    assert a.substitutions == b.substitutions


def test_it_is_capped(sandbox_file):
    examples = [_example(f"catalog{i}.html") for i in range(9)]
    cases = skill_perform.mint_cases("scrape", examples, CONFIG)
    assert len(cases) <= skill_perform.MAX_CASES


def test_round_trip_through_disk(sandbox_file, worker_dir):
    n = skill_perform.mint_and_save("scrape", [_example()], CONFIG)
    assert n == 1
    assert skill_perform.has_perform_cases("scrape")
    loaded = skill_perform.load_cases("scrape")
    assert loaded[0].substitutions == {"catalog.html": loaded[0].required[0]}
    assert json.loads((worker_dir / skill_perform.PERFORM_FILE).read_text())


def test_a_missing_or_corrupt_file_is_not_fatal(worker_dir):
    assert skill_perform.load_cases("scrape") == []
    (worker_dir / skill_perform.PERFORM_FILE).write_text("{not json", encoding="utf-8")
    assert skill_perform.load_cases("scrape") == []
    assert skill_perform.has_perform_cases("scrape") is False


# --------------------------------------------------------------------------
# Grading
# --------------------------------------------------------------------------

def _case(sandbox):
    return skill_perform.mint_cases("scrape", [_example()], CONFIG)[0]


def test_the_memorised_answer_fails(sandbox_file):
    """The whole point: a reply that emits the trained value, not the asked one."""
    case = _case(sandbox_file)
    memorised = "```python\nwith open('catalog.html') as f:\n    print(f.read().count('<tr>'))\n```"
    ok, why = skill_perform.grade(memorised, case, CONFIG)
    assert ok is False
    assert "catalog.html" in why


def test_the_correct_answer_passes_by_running(sandbox_file):
    case = _case(sandbox_file)
    ok, why = skill_perform.grade(f"```python\n{case.expected_script}```", case, CONFIG)
    assert ok is True, why
    assert "ran clean" in why


def test_code_that_uses_the_right_name_but_crashes_fails(sandbox_file):
    """Naming the file is not performing the skill -- this is why it executes."""
    case = _case(sandbox_file)
    broken = f"```python\nopen('{case.required[0]}')\nundefined_name()\n```"
    ok, why = skill_perform.grade(broken, case, CONFIG)
    assert ok is False
    assert "failed when run" in why


def test_a_fluent_claim_with_no_code_fails(sandbox_file):
    """The fabricated-completion shape: reports the work, never did it."""
    case = _case(sandbox_file)
    claim = f"I've scraped {case.required[0]} and found 24 rows; all clean."
    ok, why = skill_perform.grade(claim, case, CONFIG)
    assert ok is False
    assert "no runnable code" in why


def test_reciting_the_procedure_fails(sandbox_file):
    case = _case(sandbox_file)
    steps = "1. Open the file. 2. Count the rows. 3. Print the total."
    ok, why = skill_perform.grade(
        "1. Open the file. 2. Count the rows. 3. Print the total.",
        case, CONFIG, steps=steps)
    assert ok is False
    assert "recited" in why


def test_a_non_executable_skill_stops_at_the_value_check(sandbox_file):
    """A procedure a human carries out has no script to run."""
    case = skill_perform.PerformCase(
        id="perform_0", prompt="Steep the ledger tea for 4 minutes.",
        substitutions={"roster": "ledger"}, executable=False)
    ok, _ = skill_perform.grade("Steeping the ledger tea now.", case, CONFIG)
    assert ok is True
    ok, why = skill_perform.grade("Steeping the roster tea now.", case, CONFIG)
    assert ok is False and "memorised" in why


def test_fixtures_are_restaged_after_the_sandbox_is_cleaned(sandbox_file):
    case = _case(sandbox_file)
    (sandbox_file / case.required[0]).unlink()
    skill_perform.restage_fixtures([case])
    assert (sandbox_file / case.required[0]).exists()


# --------------------------------------------------------------------------
# Remedy
# --------------------------------------------------------------------------

def test_the_remedy_target_is_the_script_not_the_steps(sandbox_file, monkeypatch):
    """Teaching the runbook back would re-create the failure being fixed."""
    case = _case(sandbox_file)
    captured = []
    monkeypatch.setattr(
        skill_perform, "__name__", skill_perform.__name__)  # keep import path stable
    from symbio.app import training

    monkeypatch.setattr(
        training, "append_chat_pair",
        lambda user, answer, tok, sysp, role=None: captured.append((user, answer)))
    added = skill_perform.remedy_samples(
        [case], [case.id], object(), "sys", "scrape", copies=2)
    assert added == 2
    user, answer = captured[0]
    # The demonstration, not the test: see
    # test_the_remedy_trains_on_the_demonstration_not_the_test.
    assert user == case.source_request
    assert case.source_script.strip() in answer
    assert "```python" in answer, "a script, never the steps text"


# --------------------------------------------------------------------------
# The wiring: seeding a skill mints its battery
# --------------------------------------------------------------------------

class _Tok:
    name_or_path = "org/Worker-4B"

    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=False, enable_thinking=False):
        return " ".join(f"{m['role']}: {m['content']}" for m in messages)


STEPS = "1. Open catalog.html. 2. Count the <tr> rows. 3. Print the total."


def _teacher(request_name="catalog.html"):
    """Stands in for the headmaster generating a worked example."""
    def generate(prompt, max_tokens=700):
        return (
            f"REQUEST: Scrape {request_name} and report how many rows it has.\n"
            "OUTPUT:\n"
            "```python\n"
            f"with open('{request_name}') as f:\n"
            "    body = f.read()\n"
            "print(body.count('<tr>'))\n"
            "```"
        )
    return generate


def test_seeding_a_skill_mints_its_perform_battery(
        tmp_path, monkeypatch, sandbox_file, worker_dir):
    """The end-to-end wiring: worked examples in, golden_perform.json out."""
    from symbio.app import skills

    written = skills._seed_skill_training_data(
        "scrape", "Scrape Rows", STEPS, _Tok(),
        example_generator=_teacher(), config=CONFIG)

    assert written > 0
    corpus = [json.loads(l) for l in
              (worker_dir / "train.jsonl").read_text().splitlines() if l.strip()]
    kinds = {row["metadata"]["seed_kind"] for row in corpus}
    assert kinds == {"recall", "worked"}, "both seed kinds must survive"

    cases = skill_perform.load_cases("scrape")
    assert cases, "seeding a code skill must leave a perform battery behind"
    case = cases[0]
    # The battery asks for a file the corpus never mentions.
    assert "catalog.html" not in case.prompt
    corpus_text = (worker_dir / "train.jsonl").read_text()
    assert case.required[0] not in corpus_text, (
        "the perturbed value must appear nowhere in the training data, or the "
        "check is answerable from memory")


def test_a_dead_teacher_leaves_no_battery_and_no_crash(
        tmp_path, monkeypatch, sandbox_file, worker_dir):
    """A failed teacher costs the guard rail, never the skill."""
    from symbio.app import skills

    def boom(prompt, max_tokens=700):
        raise RuntimeError("teacher died")

    written = skills._seed_skill_training_data(
        "scrape", "Scrape Rows", STEPS, _Tok(),
        example_generator=boom, config=CONFIG)
    assert written == len(skills._seed_user_turns("Scrape Rows"))
    assert skill_perform.load_cases("scrape") == []


def test_a_blocked_import_fails_rather_than_being_excused(sandbox_file):
    """Seeding excuses an unrunnable example; this battery must not.

    A case exists only because its own script ran clean in this sandbox, so an
    unrunnable reply is the model reaching for a blocked import, not a dead
    environment. Excusing it would pass `import requests` on the strength of
    the filename alone.
    """
    case = _case(sandbox_file)
    blocked = f"```python\nimport socket\nopen('{case.required[0]}')\n```"
    ok, why = skill_perform.grade(blocked, case, CONFIG)
    assert ok is False
    assert "could not be run" in why


def test_a_hostonly_url_is_never_perturbed(sandbox_file):
    """Swapping the host produces a broken target, not a harder task.

    http://localhost:8817 is what the one real worked corpus in this checkout
    uses. Splitting it on the last slash hands back the host, and the case
    would grade a worker on failing to reach a machine that never existed.
    """
    from symbio.app.skill_perform import _candidate_values

    assert _candidate_values(
        "Fetch from http://localhost:8817",
        "requests.get('http://localhost:8817')") == {}


def test_a_url_path_segment_is_perturbed(sandbox_file):
    from symbio.app.skill_perform import _candidate_values

    subs = _candidate_values(
        "Fetch http://localhost:8817/listings and parse it",
        "requests.get('http://localhost:8817/listings')")
    assert list(subs) == ["http://localhost:8817/listings"]
    new = subs["http://localhost:8817/listings"]
    assert new.startswith("http://localhost:8817/")
    assert not new.endswith("/listings")


def test_the_remedy_mixes_failures_with_successes(sandbox_file, monkeypatch):
    """Both outcomes go in. A remedy built only from failures makes them the
    bulk of the delta, and on a corpus this small whichever behaviour is
    repeated most just wins -- which moves the failure, not removes it."""
    from symbio.app import training

    # Both inputs must really exist, or the second case is (correctly) dropped
    # when its substituted script cannot open the file.
    (sandbox_file / "gazette.html").write_text("<tr>x</tr>", encoding="utf-8")
    a = skill_perform.mint_cases("scrape", [_example("catalog.html")], CONFIG)[0]
    b = skill_perform.mint_cases("scrape", [_example("gazette.html")], CONFIG)[0]
    b.id = "perform_1"
    captured = []
    monkeypatch.setattr(
        training, "append_chat_pair",
        lambda user, answer, tok, sysp, role=None: captured.append(user))

    added = skill_perform.remedy_samples(
        [a, b], [a.id], object(), "sys", "scrape",
        copies=2, passing={b.id}, passing_copies=1)

    assert added == 3
    assert captured.count(a.source_request) == 2, "the failure is weighted higher"
    assert captured.count(b.source_request) == 1, "the success anchors it"


def test_a_case_is_never_written_twice_for_being_in_both_sets(sandbox_file, monkeypatch):
    from symbio.app import training

    case = _case(sandbox_file)
    captured = []
    monkeypatch.setattr(
        training, "append_chat_pair",
        lambda user, answer, tok, sysp, role=None: captured.append(user))
    added = skill_perform.remedy_samples(
        [case], [case.id], object(), "sys", "scrape",
        copies=2, passing={case.id}, passing_copies=1)
    assert added == 2 and len(captured) == 2


def test_two_filenames_do_not_collide_or_cascade(sandbox_file):
    """The bug the first real skill produced.

    The headmaster wrote "Process orders.json and generate tally.json", and
    "orders" hashes to the pool slot holding "tally" -- the other file in the
    same example. Sequential replacement then turned
    {orders.json -> tally.json, tally.json -> almanac.json} into a script that
    read and wrote one file, and the case died on KeyError: 'orders'.
    """
    from symbio.app.skill_perform import _apply, _candidate_values

    request = "Process orders.json and generate tally.json"
    script = "d = open('orders.json')\nopen('tally.json', 'w')\n"
    subs = _candidate_values(request, script)

    assert set(subs) == {"orders.json", "tally.json"}
    # No swap may land on a name the example already uses...
    assert not (set(subs.values()) & {"orders.json", "tally.json"})
    # ...and no two swaps may land on each other.
    assert len(set(subs.values())) == 2
    # One pass, so an earlier substitution is never re-substituted.
    out = _apply(script, subs)
    assert out.count(subs["orders.json"]) == 1
    assert out.count(subs["tally.json"]) == 1


def test_a_longer_name_containing_a_shorter_one_is_matched_whole(sandbox_file):
    from symbio.app.skill_perform import _apply

    subs = {"orders.json": "docket.json", "old_orders.json": "gazette.json"}
    assert _apply("old_orders.json orders.json", subs) == "gazette.json docket.json"


# --------------------------------------------------------------------------
# Fault reporting (shared with the seeding retry loop)
# --------------------------------------------------------------------------

def test_a_syntax_error_reports_the_error_not_the_caret(sandbox_file):
    """The sandbox pre-compiles and puts its summary FIRST, then a source
    excerpt ending in a lone caret. Reading from the end returned "^", which
    is what both this battery's reason and the teacher's retry feedback said.
    """
    from symbio.app import skills

    code = "import json\nprint(counts generated from docket with orders)\n"
    state, fault = skills._execution_feedback(code, CONFIG)
    assert state == "fault"
    assert "Syntax error" in fault and fault.strip() != "^"


def test_a_traceback_still_reports_its_last_line(sandbox_file):
    """A real traceback names the defect at the end; that must not regress."""
    from symbio.app import skills

    state, fault = skills._execution_feedback("undefined_name()\n", CONFIG)
    assert state == "fault"
    assert "NameError" in fault


def test_the_remedy_trains_on_the_demonstration_not_the_test(sandbox_file, monkeypatch):
    """Training on the case's own answer is training on the test.

    If the perturbed prompt and its expected script enter the corpus, the next
    run of the battery grades a question the worker has just been taught, and
    a pass stops meaning anything. The battery's whole claim is that its values
    appear nowhere in the training data.
    """
    from symbio.app import training

    case = _case(sandbox_file)
    assert case.source_request and case.prompt != case.source_request
    captured = []
    monkeypatch.setattr(
        training, "append_chat_pair",
        lambda user, answer, tok, sysp, role=None: captured.append((user, answer)))

    skill_perform.remedy_samples([case], [case.id], object(), "sys", "scrape", copies=1)

    user, answer = captured[0]
    assert user == case.source_request, "must reinforce the original demonstration"
    assert case.prompt not in user
    for perturbed in case.required:
        assert perturbed not in user and perturbed not in answer, (
            f"{perturbed} is a battery value and must never enter the corpus")


def test_a_case_without_its_source_writes_nothing(sandbox_file, monkeypatch):
    """Older battery files predate source_request; they must not fall back to
    the perturbed pair."""
    from symbio.app import training

    case = _case(sandbox_file)
    case.source_request, case.source_script = "", ""
    captured = []
    monkeypatch.setattr(
        training, "append_chat_pair",
        lambda *a, **k: captured.append(a))
    assert skill_perform.remedy_samples(
        [case], [case.id], object(), "sys", "scrape", copies=2) == 0
    assert captured == []


# --------------------------------------------------------------------------
# A skill with no procedure must be refused, not seeded and trained
# --------------------------------------------------------------------------

def test_a_skill_without_steps_is_refused(tmp_path, monkeypatch):
    """`/new-skill <name>` with no "| <steps>" used to save the placeholder
    "(no steps provided yet)" as the procedure and start a fine-tune on it.

    Worse than wasteful: skill_note_body derives the note's Triggers from its
    body, so the placeholder produced a note keyed on "provided, yet" sitting
    in a term-frequency retrieval index.
    """
    import pytest as _pytest

    from symbio.app import memory

    called = []
    monkeypatch.setattr(memory, "save_note", lambda *a, **k: called.append(a))

    for blank in ("", "   ", "(no steps provided yet)", "TBD", "todo", "None"):
        with _pytest.raises(ValueError, match="no steps"):
            memory.save_skill("Rotate Keys", blank)
    assert called == [], "nothing may be written for a skill with no procedure"


def test_a_real_procedure_is_still_accepted(tmp_path, monkeypatch):
    from symbio.app import memory

    saved = []
    monkeypatch.setattr(memory, "save_note",
                        lambda title, body: saved.append(title) or tmp_path / "n.md")
    memory.save_skill("Rotate Keys", "1. Read the key. 2. Issue a new one.")
    assert saved == ["Skill: Rotate Keys"]


# --------------------------------------------------------------------------
# Harvesting real work instead of inventing it
# --------------------------------------------------------------------------

def _turn(role, content):
    return {"role": role, "content": content}


def test_it_harvests_the_code_that_actually_ran(sandbox_file):
    from symbio.app import skills

    history = [
        _turn("user", "Fetch the invoices and flag anything over 30 days late"),
        _turn("assistant", "<py>import symbio_tools\nprint('INV-202')</py> Checking."),
        _turn("user", '[System observation: Python script exited ok.]\n'
                      '<tool_response>{"name":"execute_code","content":"ok"}</tool_response>'),
    ]
    got = skills.harvest_worked_examples(history)
    assert len(got) == 1
    request, output, verified = got[0]
    assert request == "Fetch the invoices and flag anything over 30 days late"
    assert "import symbio_tools" in output and output.startswith("```python")
    assert verified is True, "it ran; that is what verified means"


def test_code_that_did_not_run_is_not_a_demonstration(sandbox_file):
    from symbio.app import skills

    history = [
        _turn("user", "Do the thing"),
        _turn("assistant", "<py>undefined_name()</py>"),
        _turn("user", '[System observation: NameError: name undefined_name is not defined]'),
    ]
    assert skills.harvest_worked_examples(history) == []


def test_the_request_is_the_users_ask_not_the_observation(sandbox_file):
    """Observations are appended as user turns, so the most recent user message
    is usually the tool result rather than what was asked."""
    from symbio.app import skills

    history = [
        _turn("user", "Count the shipped orders"),
        _turn("assistant", "<py>print(1)</py>"),
        _turn("user", "[System observation: Python script exited ok.]"),
        _turn("assistant", "<py>print(2)</py>"),
        _turn("user", "[System observation: Python script exited ok.]"),
    ]
    got = skills.harvest_worked_examples(history)
    assert got, "both ran"
    for request, _out, _v in got:
        assert request == "Count the shipped orders"
        assert "System observation" not in request


def test_duplicate_code_is_harvested_once(sandbox_file):
    from symbio.app import skills

    ran = "[System observation: Python script exited ok.]"
    history = []
    for _ in range(3):
        history += [_turn("user", "Do it"), _turn("assistant", "<py>print(1)</py>"),
                    _turn("user", ran)]
    assert len(skills.harvest_worked_examples(history)) == 1


def test_no_history_is_not_an_error(sandbox_file):
    from symbio.app import skills

    assert skills.harvest_worked_examples(None) == []
    assert skills.harvest_worked_examples([]) == []
