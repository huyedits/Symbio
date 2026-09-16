"""Tests for the lightweight RAG retriever."""

import json
import time

import pytest

from symbio.rag import Retriever


@pytest.fixture
def base_config(tmp_path, monkeypatch):
    """Isolated RAG paths."""
    from symbio.rag import DATA_DIR, NOTES_DIR, PROJECT_DIR, TRAIN_FILE
    monkeypatch.setattr("symbio.rag.PROJECT_DIR", tmp_path)
    monkeypatch.setattr("symbio.rag.NOTES_DIR", tmp_path / "notes")
    monkeypatch.setattr("symbio.rag.DATA_DIR", tmp_path / "training_data")
    monkeypatch.setattr("symbio.rag.TRAIN_FILE", tmp_path / "training_data" / "train.jsonl")
    (tmp_path / "notes").mkdir(parents=True, exist_ok=True)
    (tmp_path / "training_data").mkdir(parents=True, exist_ok=True)
    return {
        "rag": {
            "enabled": True,
            "top_k": 5,
            "max_context_tokens": 500,
            "sources": ["notes"],
            "context_cache_ttl_seconds": 0.5,
            "context_cache_max_entries": 4,
        }
    }


def test_build_context_caches_result(base_config, tmp_path):
    note = tmp_path / "notes" / "hobbies.md"
    note.write_text("# Hobbies\nThe user likes hiking and coffee.")

    retriever = Retriever(base_config)
    ctx1 = retriever.build_context("hobbies")
    assert "hiking" in ctx1

    # Delete the note file; a cached result should still return the same text.
    note.unlink()
    ctx2 = retriever.build_context("hobbies")
    assert ctx2 == ctx1


def test_invalidate_cache_clears_context_cache(base_config, tmp_path):
    note = tmp_path / "notes" / "hobbies.md"
    note.write_text("# Hobbies\nThe user likes hiking.")

    retriever = Retriever(base_config)
    ctx1 = retriever.build_context("hobbies")
    retriever.invalidate_cache()
    note.write_text("# Hobbies\nThe user likes swimming.")
    ctx2 = retriever.build_context("hobbies")
    assert ctx1 != ctx2
    assert "swimming" in ctx2


def test_context_cache_expires_after_ttl(base_config, tmp_path):
    note = tmp_path / "notes" / "hobbies.md"
    note.write_text("# Hobbies\nThe user likes hiking.")

    retriever = Retriever(base_config)
    ctx1 = retriever.build_context("hobbies")
    note.write_text("# Hobbies\nThe user likes swimming.")
    ctx2 = retriever.build_context("hobbies")
    # Within TTL, the cached value is returned.
    assert ctx2 == ctx1

    # After TTL expires, a fresh retrieval (with a fresh note cache) sees the
    # updated note. We manually drop the note cache here because in normal
    # operation note writes call invalidate_cache(), which clears both caches.
    time.sleep(0.6)
    retriever._note_cache = None
    ctx3 = retriever.build_context("hobbies")
    assert "swimming" in ctx3


def test_search_training_data_scans_recent_bytes_only(base_config, tmp_path):
    base_config["rag"]["sources"] = ["training_data"]
    train_file = tmp_path / "training_data" / "train.jsonl"

    # Fill with many lines so scanning the whole file would be slow.
    filler = "\n".join(
        json.dumps({"text": f"old sample number {i} with keyword ancient"})
        for i in range(2000)
    )
    train_file.write_text(filler + "\n", encoding="utf-8")

    # Append a recent line containing the query term.
    recent = json.dumps({"text": "recent sample mentions hiking trail"})
    with open(train_file, "a", encoding="utf-8") as f:
        f.write(recent + "\n")

    retriever = Retriever(base_config)
    results = retriever.search_training_data("hiking", top_k=5)
    texts = [r["text"] for r in results]
    assert any("recent sample" in t for t in texts)
    assert not any("ancient" in t for t in texts)


# ---- a wrong note is worse than no note ----
#
# Retrieved notes are pasted into the model's context, and skill notes are
# procedures, so the model performs them. Measured live: a request to save a
# bicycle skill retrieved the Browser Driver note and the agent produced a
# repetition loop of "Clicking the Steps. Scrolling down." and a real
# browser_open on google.com. IDF ranked correctly the whole time — it just
# has no way to say "none of these", so when the topical words matched no note
# the ranking fell to leftovers like "skill", "for" and "how".

@pytest.fixture
def skill_notes(base_config, tmp_path):
    """A corpus shaped like the real one: procedures sharing boilerplate."""
    notes = tmp_path / "notes"
    bodies = {
        "Skill__Browser_Driver.md": "# Skill: Browser Driver\n1. Click the element. 2. Scroll down. 3. Read the page.",
        "Skill__Fix_wifi.md": "# Skill: Fix wifi\n1. Toggle the wifi adapter. 2. Forget the network. 3. Rejoin.",
        "Skill__Coffee_Making.md": "# Skill: Coffee Making\n1. Grind beans. 2. Add filter. 3. Brew.",
        "Skill__Device_Awareness.md": "# Skill: Device Awareness\n1. Save a fact when the user gives one. 2. Do arithmetic.",
        "Skill__Researcher.md": "# Skill: Researcher\n1. Search online. 2. Summarise findings. 3. Cite sources.",
        "Skill__Repotting.md": "# Skill: Repotting\n1. Check roots. 2. Choose a pot. 3. Water it.",
    }
    for name, text in bodies.items():
        (notes / name).write_text(text, encoding="utf-8")
    return Retriever(base_config)


def test_a_query_matching_no_note_returns_nothing(skill_notes):
    """Nothing here is about bicycles, so three confident wrong answers is the
    one result that must not come back."""
    assert skill_notes.search_notes("tuning a bicycle") == []
    # The query that actually derailed the trial.
    assert skill_notes.search_notes(
        "Emit one <skill> tag for bicycle tuning. Write the steps yourself.") == []


def test_a_real_shared_word_is_still_a_match(skill_notes):
    """The floor drops notes with nothing distinguishing in common — not
    notes that genuinely share a content word. "save" appears in the Device
    Awareness note, so matching it is correct behaviour, not a regression."""
    hits = skill_notes.search_notes("save a fact for me")
    assert hits and "Device_Awareness" in hits[0]["title"]


def test_the_word_skill_alone_retrieves_no_skill_note(skill_notes):
    """It appears in every one of them, so it distinguishes nothing."""
    assert skill_notes.search_notes("skill") == []
    assert skill_notes.search_notes("what are the steps") == []


def test_an_on_topic_query_still_finds_its_note(skill_notes):
    hits = skill_notes.search_notes("how do I fix my wifi")
    assert hits, "the floor must not suppress a real match"
    assert "Fix_wifi" in hits[0]["title"]


def test_every_skill_finds_its_own_note(skill_notes):
    for query, expected in [("how do I make coffee", "Coffee_Making"),
                            ("repotting a plant", "Repotting"),
                            ("click the button and scroll", "Browser_Driver")]:
        hits = skill_notes.search_notes(query)
        assert hits and expected in hits[0]["title"], (query, hits)


def test_stopwords_never_qualify_a_note(skill_notes):
    """On a small technical corpus "how" and "do" are genuinely rare — most
    notes are numbered steps and never use them — so a frequency test alone
    called them discriminative and let them admit an unrelated note."""
    terms = skill_notes._discriminative_terms(
        ["how", "do", "the", "steps", "coffee"],
        [["coffee"], ["a"], ["b"], ["c"], ["d"], ["e"]])
    assert terms == {"coffee"}


def test_a_small_corpus_keeps_the_old_behaviour(base_config, tmp_path):
    """With four notes one is 25% and two is 50%, so the share cutoff would be
    deciding on noise."""
    notes = tmp_path / "notes"
    for i in range(3):
        (notes / f"n{i}.md").write_text(f"note {i} about widgets", encoding="utf-8")
    r = Retriever(base_config)
    assert r._discriminative_terms(["widgets"], [["a"], ["b"], ["c"]]) is None
    assert r.search_notes("widgets"), "still finds things in a tiny corpus"


# ---- routing must not turn on incidental words ----
#
# Live 2026-08-24: "weigh UP ... whether a small model can understand anything"
# retrieved the Browser Driver note, whose triggers read "pull up / bring up /
# throw up / chuck up". "up" was the only query term the note contained, so it
# both admitted the note and supplied 52% of its score; the headmaster then
# delegated to the browser worker and tried to open google.com. Retrieved skill
# notes are procedures and the model performs them, so a coincidental match is
# not a cosmetic ranking problem.

_CORPUS = {
    "browser.md": (
        "# Skill: Browser Driver\n"
        "1. Use the controllable browser to act on a page.\n"
        "## Triggers\nKeywords: pull up, bring up, throw up, chuck up, "
        "navigate, click through, website, url\n"
        "Examples:\n- pull up example.com and click through to the docs\n"
    ),
    "device.md": "# Skill: Device Awareness\nKeyboard layout, timezone, shell, hostname.\n",
    "scrape.md": "# Skill: Scrape\nFetch through the cache proxy, parse with selectolax.\n",
    "tent.md": "# Skill: Tent\nPitch the tent with the narrow end into the wind.\n",
    "tyre.md": "# Skill: Tyre\nLoosen the lug nuts, raise the jack, fit the spare.\n",
    "coffee.md": "# Skill: Coffee\nGrind the beans, heat the water, brew.\n",
    "knife.md": "# Skill: Knife\nHold the whetstone flat and keep the bevel angle.\n",
    "parcel.md": "# Skill: Parcel\nComplete the customs declaration and weigh the postage.\n",
}


@pytest.fixture
def corpus(base_config, tmp_path):
    for name, text in _CORPUS.items():
        (tmp_path / "notes" / name).write_text(text, encoding="utf-8")
    return Retriever(base_config)


def test_up_is_not_a_topical_word():
    # Pins the specific regression: the stoplist had the prepositions but not
    # the particles, so "up" ranked as content.
    for particle in ("up", "down", "out", "off"):
        assert particle in Retriever._NEVER_DISCRIMINATIVE


def test_a_particle_alone_does_not_retrieve_a_procedure(corpus):
    # "pitch"+"tent" anchor the query to the tent note so the gate has a real
    # answer to give; the browser note's only tie to this query is "up".
    titles = [h["title"] for h in corpus.search_notes("weigh up the tradeoffs before I pitch my tent")]
    assert "browser.md" not in titles
    assert "tent.md" in titles


def test_a_real_browser_request_still_routes_to_the_browser_note(corpus):
    hits = corpus.search_notes("pull up example.com and click through to the docs")
    assert hits and hits[0]["title"] == "browser.md"


def test_one_shared_word_is_not_enough_when_the_query_offers_more(corpus):
    # Two discriminative terms, each landing in a different note, one apiece.
    # Coincidence in both directions, so the honest answer is nothing.
    assert corpus.search_notes("pitch a whetstone") == []


def test_two_shared_words_are_enough(corpus):
    hits = corpus.search_notes("pitch my tent in the wind")
    assert hits and hits[0]["title"] == "tent.md"


def test_a_single_word_query_can_still_match_on_its_one_word(corpus):
    # The floor is min(2, len(discriminative)) — it uses what the query gives
    # it rather than demanding words the user never typed.
    hits = corpus.search_notes("whetstone")
    assert hits and hits[0]["title"] == "knife.md"


def test_stopwords_do_not_decide_the_ranking(corpus):
    # Scoring used every query term while the gate used a filtered set, so a
    # note admitted on a content word could still be ranked by "from"/"can".
    assert corpus._score(["can", "from", "the", "up"], _CORPUS["browser.md"]) == 0.0


# ---- a retrieved note's refusal being executed ----
#
# Found live 2026-08-26 on the 14B. With the repo's HTML already in context —
# the answer, 201, sitting in it — "how many commits does that repo have?"
# retrieved Skill: Device Awareness and the model obeyed its step 4:
#
#   Caine: not a device_awareness request. I can't access real-time repo data
#          directly. Would you like me to help with anything related to your
#          machine settings or simple arithmetic instead?
#
# Its reasoning block named the cause outright: "The Device Awareness skill
# mentions that if the request is about anything else ... I should respond that
# it's not a device_awareness request and stop." The only discriminative term
# was "many" — present in exactly one note, because that note's derived
# Triggers block lists "how many seconds, how many minutes, how many hours,
# how many days". Retrieval injected a refusal and the agent carried it out.


@pytest.fixture
def triggered_notes(base_config, tmp_path):
    """A skill note carrying the derived `## Triggers` block, as skills.py writes it.

    The triggers are what made the note findable (2026-08-24) and also what
    makes it match on quantifier vocabulary, so a corpus without them cannot
    reproduce either behaviour.
    """
    notes = tmp_path / "notes"
    (notes / "Skill__Device_Awareness.md").write_text(
        "# Skill: Device Awareness\n"
        "1. Save a fact about the user with <note>.\n"
        "2. For arithmetic too large to do in your head, use <py>.\n"
        "4. This skill covers exactly seven machine settings: keyboard layout, "
        "timezone, shell, hostname, OS version, CPU, RAM. If the request is "
        "about anything else - scraping, databases, files, networks, code - "
        "reply \"not a device_awareness request\" in one line and stop.\n"
        "\n## Triggers\n\n"
        "Keywords: power, squared, multiply, times, how many seconds, "
        "how many minutes, how many hours, how many days, arithmetic, "
        "calculate, note, save, what is my, whats my, which layout, "
        "what timezone, currently set, switch my keyboard\n\n"
        "Examples:\n\n"
        "- multiply 33333 by 77777 exactly\n"
        "- how many hours in 400 days\n"
        "- what is my keyboard layout?\n"
        "- what timezone am i in\n"
        "- switch my keyboard to colemak\n",
        encoding="utf-8")
    (notes / "Skill__Coffee_Making.md").write_text(
        "# Skill: Coffee Making\n1. Grind beans. 2. Add filter. 3. Brew.\n"
        "\n## Triggers\n\nKeywords: coffee, grind, beans, filter, brew\n",
        encoding="utf-8")
    (notes / "Skill__Repotting.md").write_text(
        "# Skill: Repotting\n1. Check roots. 2. Choose a pot. 3. Water it.\n"
        "\n## Triggers\n\nKeywords: repot, roots, pot, water, plant\n",
        encoding="utf-8")
    return Retriever(base_config)


@pytest.mark.parametrize("query", [
    "how many commits does that repo have?",
    "how many stars does it have",
])
def test_a_long_query_is_not_matched_on_one_incidental_word(triggered_notes, query):
    assert triggered_notes.search_notes(query) == [], query


def test_the_reply_that_shipped_cannot_be_produced_again(triggered_notes):
    """The refusal came from the note; if the note is not retrieved, it cannot."""
    hits = triggered_notes.search_notes("how many commits does that repo have?")
    assert not any("device_awareness request" in h["text"] for h in hits)


@pytest.mark.parametrize("query", [
    "what is my keyboard layout?",
    "what timezone am i in",        # five words, one content word — must match
    "how many hours in 400 days",   # "many" is stoplisted; hours/days carry it
    "multiply 33333 by 77777 exactly",
    "switch my keyboard to colemak",
])
def test_the_skill_is_still_reachable_by_its_real_triggers(triggered_notes, query):
    hits = triggered_notes.search_notes(query)
    assert hits, f"{query!r} lost its skill"
    assert "Device_Awareness" in hits[0]["title"], query


@pytest.mark.parametrize("query,expected", [
    ("grind coffee beans", "Coffee_Making"),
    ("repot a plant", "Repotting"),
])
def test_the_other_skills_still_route_to_themselves(triggered_notes, query, expected):
    hits = triggered_notes.search_notes(query)
    assert hits and expected in hits[0]["title"], query


# ---- retrieved notes must be applied, not truncated ----
#
# A retrieved skill note is a procedure the model performs, so the top result
# must arrive whole: a long procedure cut off before its final step is a
# procedure the model cannot finish. max_context_tokens is 500 (~2000 chars);
# a 3000-char note would previously have been truncated mid-steps.


def test_a_long_top_note_is_not_truncated(base_config, tmp_path):
    steps = "\n".join(f"{i}. perform step {i} of the procedure" for i in range(1, 120))
    body = f"# Skill: Long Procedure\n{steps}\nFINAL_STEP_MARKER"
    (tmp_path / "notes" / "Skill__Long_Procedure.md").write_text(body, encoding="utf-8")

    ctx = Retriever(base_config).build_context("long procedure")

    assert "FINAL_STEP_MARKER" in ctx


def test_the_context_header_instructs_application(base_config, tmp_path):
    (tmp_path / "notes" / "hobbies.md").write_text(
        "# Hobbies\nThe user likes hiking and coffee.", encoding="utf-8")

    ctx = Retriever(base_config).build_context("hobbies")

    assert "answer from this first" in ctx
    assert "use its content directly" in ctx
