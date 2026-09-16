"""The seed corpus must teach rules, not memorisable constants.

A fine-tune generalises only when the *tool choice* is the invariant across
a set of samples and the *argument* varies. The inverse arrangement — many
differently-worded requests all answering with one hardcoded value — makes
that value the only learnable invariant, and the model then emits it for
anything resembling the request.

That is not hypothetical. It shipped: `<browse>` had 14 uses across just 4
distinct URLs (epa.gov/privacy x6, cloudflare.com x4, apple.com x3), and a
question about subscription pricing made the agent open Chrome at
cloudflare.com. These tests pin the shape so a future tool cannot quietly
reintroduce it.
"""

import collections
import re

import pytest

from symbio.app import training


# Tools whose argument is genuinely a small closed set — a keyboard has only
# so many keys — so low distinct-value counts are correct, not a defect.
_CLOSED_VOCABULARY = {"press", "scroll", "browser_close"}

# Below this many uses there is no pattern to memorise either way.
_MIN_USES_TO_JUDGE = 6

# Fraction of uses that must carry a distinct argument.
_MIN_DISTINCT_RATIO = 0.6


def _seed_pairs():
    """The (user, assistant) pairs seed_training_data would write."""
    return training.build_seed_pairs("Caine", "Huy")


def _tool_arguments(pairs) -> dict[str, list[str]]:
    args: dict[str, list[str]] = collections.defaultdict(list)
    for _user, assistant in pairs:
        for match in re.finditer(r"<([a-z_]+)>(.*?)</\1>", assistant, re.DOTALL):
            args[match.group(1)].append(match.group(2).strip())
    return args


def test_expand_intent_varies_the_argument():
    pairs = training.expand_intent(
        ["Open {slot}.", "Go to {slot}."],
        ["a.com", "b.com", "c.com"],
        lambda s: f"<browse>https://{s}</browse>",
    )
    assert len(pairs) == 3
    args = [re.search(r"<browse>(.*?)</browse>", a).group(1) for _, a in pairs]
    assert len(set(args)) == 3, "every sample must carry a different argument"
    # Phrasings cycle, so wording varies too.
    assert pairs[0][0] != pairs[1][0]


def test_expand_intent_is_empty_without_slots():
    assert training.expand_intent(["Open {slot}."], [], lambda s: s) == []
    assert training.expand_intent([], ["a.com"], lambda s: s) == []


@pytest.mark.parametrize("tool", ["browse", "search"])
def test_high_volume_tools_vary_their_argument(tool):
    """The exact defect that shipped: many uses, few distinct values."""
    args = _tool_arguments(_seed_pairs()).get(tool, [])
    if len(args) < _MIN_USES_TO_JUDGE:
        pytest.fail(
            f"<{tool}> has only {len(args)} seed sample(s) — too few to teach "
            f"a rule. Add more via training.expand_intent.")
    distinct = len(set(args))
    ratio = distinct / len(args)
    assert ratio >= _MIN_DISTINCT_RATIO, (
        f"<{tool}>: {len(args)} uses but only {distinct} distinct arguments "
        f"({ratio:.0%}). Samples that repeat one argument teach that constant "
        f"instead of the tool-selection rule. Build them with expand_intent.")


def test_no_single_argument_dominates_a_tool():
    """No one value may account for most of a tool's samples."""
    for tool, args in _tool_arguments(_seed_pairs()).items():
        if tool in _CLOSED_VOCABULARY or len(args) < _MIN_USES_TO_JUDGE:
            continue
        value, count = collections.Counter(args).most_common(1)[0]
        assert count / len(args) <= 0.5, (
            f"<{tool}>: {count}/{len(args)} samples all use {value!r}. "
            f"That value becomes the memorised answer for the whole intent.")


def test_search_is_not_drowned_out_by_browse():
    """Tool balance decides which instinct wins on an ambiguous request."""
    args = _tool_arguments(_seed_pairs())
    browse = len(args.get("browse", []))
    search = len(args.get("search", []))
    assert search >= browse / 3, (
        f"<search> has {search} samples against <browse>'s {browse}. A factual "
        f"question will reach for the browser because that is what the corpus "
        f"mostly demonstrates.")


# --- rationales: the rule must be stated, not merely implied ----------


def _leading_prose(assistant: str) -> str:
    """Text before the first tool tag — where the stated reason lives."""
    match = re.search(r"<[a-z_]+[ />]", assistant)
    return (assistant[:match.start()] if match else assistant).strip()


def test_expand_intent_prefixes_the_rationale():
    pairs = training.expand_intent(
        ["Open {slot}."], ["a.com", "b.com"],
        lambda s: f"<browse>https://{s}</browse>",
        rationale="Because reasons.",
    )
    assert all(a.startswith("Because reasons. ") for _, a in pairs)
    # The reason repeats while the argument varies — that is what makes the
    # reason the learnable invariant rather than any one URL.
    reasons = {_leading_prose(a) for _, a in pairs}
    assert len(reasons) == 1


def test_expand_intent_without_rationale_is_unchanged():
    pairs = training.expand_intent(
        ["Open {slot}."], ["a.com"], lambda s: f"<browse>https://{s}</browse>")
    assert pairs[0][1] == "<browse>https://a.com</browse>"


@pytest.mark.parametrize("tool", ["browse", "search"])
def test_high_volume_tools_state_their_reason(tool):
    """A tool the corpus leans on must explain itself, or it teaches a reflex.

    Without a stated reason the model has to infer the rule from the samples,
    which is what made a pricing question open a browser: the corpus showed
    *what* to do and never *why*, so the most-rehearsed action won.
    """
    explained = 0
    total = 0
    for _user, assistant in _seed_pairs():
        if f"<{tool}>" not in assistant:
            continue
        total += 1
        if len(_leading_prose(assistant)) >= 15:
            explained += 1
    if total < _MIN_USES_TO_JUDGE:
        pytest.skip(f"<{tool}> has too few samples to judge")
    ratio = explained / total
    assert ratio >= 0.5, (
        f"<{tool}>: only {explained}/{total} samples state why the tool was "
        f"chosen. Pass rationale=... to training.expand_intent so the rule is "
        f"taught outright instead of left to be inferred.")
