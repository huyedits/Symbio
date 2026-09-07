"""Programmatic checkers for IFEval's 25 instruction types.

No judge model anywhere: every one of these is a mechanical property of the
text. That is the whole point of IFEval as a signal — nothing here can be
flattered by a lenient grader, which is how the old 7-task benchmark ended up
scoring every model 100/100.

Approximations against the official google-research implementation are noted
per checker; sentence and word segmentation use regexes rather than nltk, so
counts near a boundary can differ by one. `harness.py --selfcheck` measures
how much that matters by scoring degenerate answers, which must score ~0.
"""
import json
import re
import unicodedata

_SENT = re.compile(r"[^.!?]+[.!?]+(?:\s|$)|[^.!?]+$")
_WORD = re.compile(r"\b\w+\b")


def _words(t):
    return _WORD.findall(t)


def _sentences(t):
    return [s.strip() for s in _SENT.findall(t.strip()) if s.strip()]


def _paragraphs(t):
    return [p for p in re.split(r"\n\s*\n|\*\*\*", t.strip()) if p.strip()]


def _cmp(n, relation, target):
    if relation == "less than":
        return n < target
    if relation == "at least":
        return n >= target
    raise ValueError(f"unknown relation {relation!r}")


def _script_of(text):
    """Dominant unicode script, standing in for a language detector.

    Enough to separate the non-Latin targets IFEval uses (Kannada, Hindi,
    Arabic, Thai, Korean, Japanese, Russian...) from an English refusal to
    comply, which is the failure that matters. Latin-script targets (fr, de,
    it, es, vi) cannot be told apart this way and are reported as unsupported
    rather than silently guessed.
    """
    counts = {}
    for ch in text:
        if not ch.isalpha():
            continue
        try:
            name = unicodedata.name(ch).split()[0]
        except ValueError:
            continue
        counts[name] = counts.get(name, 0) + 1
    return max(counts, key=counts.get) if counts else ""


_LANG_SCRIPT = {
    "kn": "KANNADA", "hi": "DEVANAGARI", "mr": "DEVANAGARI", "ne": "DEVANAGARI",
    "bn": "BENGALI", "ta": "TAMIL", "te": "TELUGU", "gu": "GUJARATI",
    "pa": "GURMUKHI", "ml": "MALAYALAM", "ur": "ARABIC", "fa": "ARABIC",
    "ar": "ARABIC", "he": "HEBREW", "th": "THAI", "ko": "HANGUL",
    "ja": "CJK", "zh": "CJK", "ru": "CYRILLIC", "uk": "CYRILLIC",
    "bg": "CYRILLIC", "el": "GREEK", "hy": "ARMENIAN", "ka": "GEORGIAN",
}

UNSUPPORTED = object()


def check(instruction_id, kwargs, response, prompt=""):
    """True / False, or UNSUPPORTED when this checker cannot judge honestly."""
    k = kwargs or {}
    r = response

    if instruction_id == "punctuation:no_comma":
        return "," not in r
    if instruction_id == "change_case:english_lowercase":
        return r == r.lower()
    if instruction_id == "change_case:english_capital":
        return r == r.upper()
    if instruction_id == "change_case:capital_word_frequency":
        n = sum(1 for w in _words(r) if w.isupper() and len(w) > 1)
        return _cmp(n, k["capital_relation"], k["capital_frequency"])
    if instruction_id == "length_constraints:number_words":
        return _cmp(len(_words(r)), k["relation"], k["num_words"])
    if instruction_id == "length_constraints:number_sentences":
        return _cmp(len(_sentences(r)), k["relation"], k["num_sentences"])
    if instruction_id == "length_constraints:number_paragraphs":
        return len(_paragraphs(r)) == k["num_paragraphs"]
    if instruction_id == "length_constraints:nth_paragraph_first_word":
        paras = _paragraphs(r)
        if len(paras) != k["num_paragraphs"]:
            return False
        nth = paras[k["nth_paragraph"] - 1] if 0 < k["nth_paragraph"] <= len(paras) else ""
        first = (_words(nth) or [""])[0]
        return first.lower() == str(k["first_word"]).lower()
    if instruction_id == "keywords:existence":
        low = r.lower()
        return all(str(w).lower() in low for w in k["keywords"])
    if instruction_id == "keywords:forbidden_words":
        low = r.lower()
        return not any(re.search(rf"\b{re.escape(str(w).lower())}\b", low)
                       for w in k["forbidden_words"])
    if instruction_id == "keywords:frequency":
        n = len(re.findall(rf"\b{re.escape(str(k['keyword']).lower())}\b", r.lower()))
        return _cmp(n, k["relation"], k["frequency"])
    if instruction_id == "keywords:letter_frequency":
        n = r.lower().count(str(k["letter"]).lower())
        return _cmp(n, k["let_relation"], k["let_frequency"])
    if instruction_id == "detectable_format:number_highlighted_sections":
        n = len(re.findall(r"\*[^\*\n]+\*", r))
        return n >= k["num_highlights"]
    if instruction_id == "detectable_format:number_bullet_lists":
        n = len(re.findall(r"^\s*[\*\-]\s+", r, re.MULTILINE))
        return n == k["num_bullets"]
    if instruction_id == "detectable_format:title":
        return bool(re.search(r"<<[^\n]+>>", r))
    if instruction_id == "detectable_format:json_format":
        t = re.sub(r"^\s*```(?:json)?|```\s*$", "", r.strip(), flags=re.MULTILINE).strip()
        try:
            json.loads(t)
            return True
        except ValueError:
            return False
    if instruction_id == "detectable_format:multiple_sections":
        sp = str(k["section_spliter"])
        n = len(re.findall(rf"{re.escape(sp)}\s*\d*", r))
        return n >= k["num_sections"]
    if instruction_id == "detectable_format:constrained_response":
        return any(o in r for o in
                   ("My answer is yes.", "My answer is no.", "My answer is maybe."))
    if instruction_id == "detectable_content:number_placeholders":
        return len(re.findall(r"\[[^\]\n]*\]", r)) >= k["num_placeholders"]
    if instruction_id == "detectable_content:postscript":
        marker = str(k["postscript_marker"])
        return bool(re.search(rf"(?m)^\s*{re.escape(marker)}", r)) or marker in r
    if instruction_id == "startend:quotation":
        t = r.strip()
        return len(t) >= 2 and t.startswith('"') and t.endswith('"')
    if instruction_id == "startend:end_checker":
        return r.strip().rstrip('"').strip().endswith(str(k["end_phrase"]).strip())
    if instruction_id == "combination:repeat_prompt":
        want = str(k["prompt_to_repeat"]).strip()
        return r.strip().startswith(want)
    if instruction_id == "combination:two_responses":
        return len(re.findall(r"\*\*\*+", r)) >= 1
    if instruction_id == "language:response_language":
        want = _LANG_SCRIPT.get(str(k["language"]))
        if want is None:
            return UNSUPPORTED
        got = _script_of(r)
        if want == "CJK":
            return got.startswith("CJK") or got in ("HIRAGANA", "KATAKANA")
        return got == want
    return UNSUPPORTED
