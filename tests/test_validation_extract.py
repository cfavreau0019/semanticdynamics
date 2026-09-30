import pytest

from validation import (
    FreeTextMatcher, LabeledEntryExtractor, LabeledValuesCheck, ForeignValuesCheck, Validator, VocabularyMatcher,
    default_aliases,
)
from validation.extract import normalize_line

COLORS = VocabularyMatcher({"Red": ["crimson"], "Blue": [], "Light Blue": ["sky blue"]})


def extractor(labels=("Primary", "Secondary", "Hopes and Fears"), matcher=COLORS, **kw):
    return LabeledEntryExtractor(list(labels), matcher, **kw)


@pytest.mark.parametrize("line, expected", [
    ("1. **Present — Three of Pentacles:** text", "Present — Three of Pentacles: text"),
    ("### Present", "Present"),
    ("- *Outcome*: x", "Outcome: x"),
    ("(3) Goal: y", "Goal: y"),
    ("> quoted: z", "quoted: z"),
])
def test_normalize_line(line, expected):
    assert normalize_line(line) == expected


def test_default_aliases_split_slash():
    assert default_aliases("Root Cause/Foundation") == ["Root Cause/Foundation", "Root Cause", "Foundation"]
    assert default_aliases("Outcome") == ["Outcome"]


def test_vocabulary_matcher_longest_alias_and_word_bounds():
    assert COLORS.find("a light blue sky") == "Light Blue"
    assert COLORS.find("sky blue") == "Light Blue"
    assert COLORS.find("CRIMSON dawn") == "Red"
    assert COLORS.find("reddish") is None           # word-bounded
    assert [v for v, _ in COLORS.find_all("red then blue then red")] == ["Red", "Blue", "Red"]


@pytest.mark.parametrize("text", [
    "Primary: Red - bold choice",
    "1. **Primary — Red:** bold choice",
    "- Primary (Red) is bold",
    "Primary = crimson",
    "The Primary position: Red",
    "Position 1 - Primary: Red",
    "### Primary\n\n**Red**\nexplanation",               # value on the following line
])
def test_line_formats(text):
    entries = extractor().extract(text)
    assert [(e.label, e.value, e.mode) for e in entries][:1] == [("Primary", "Red", "line")]


def test_value_segment_stops_at_description():
    # "Blue" appears in the description, not the value segment
    e = extractor().extract("Primary: Red - unlike Blue, it is warm")[0]
    assert e.value == "Red" and e.segment == "Red"


def test_label_mid_sentence_is_not_a_line_entry():
    assert extractor(inline_window=0).extract("My primary concern: Red") == []


def test_label_separator_required():
    assert extractor().extract("Primary colours are nice") == []


def test_label_spelling_variants():
    ex = extractor()
    for text in ("Hopes & Fears: Red", "hopes and fears: Red", "Hopes/Fears: Red", "HOPES  AND  FEARS — Red"):
        assert [e.label for e in ex.extract(text)] == ["Hopes and Fears"], text


def test_inline_fallback_window():
    ex = extractor(inline_window=40)
    found = ex.extract_inline("In the primary slot, Red dominates. The secondary one is discussed at considerable "
                              "length, much later on, alongside Blue. Tertiary.", ["Primary", "Secondary"])
    assert found["Primary"].value == "Red" and found["Primary"].mode == "inline"
    assert "Secondary" not in found  # "Blue" is beyond the 40-char window
    assert "Secondary" not in ex.extract_inline("The secondary one. Blue", ["Secondary"])  # sentence boundary
    assert ex.extract_inline("Primary: Red", ["Primary"])["Primary"].value == "Red"
    assert extractor(inline_window=0).extract_inline("Primary Red", ["Primary"]) == {}


def test_free_text_matcher():
    ex = LabeledEntryExtractor(["Name", "City"], FreeTextMatcher())
    entries = {e.label: e.value for e in ex.extract("Name: Ada Lovelace\nCity: \"London\"\n")}
    assert entries == {"Name": "Ada Lovelace", "City": "London"}


# ---- LabeledValuesCheck on a generic (non-tarot) task ---------------------------------
def run_check(text, expected, **kw):
    check = LabeledValuesCheck(extractor(labels=("Primary", "Secondary")), **kw)
    return Validator([check]).validate(text, expected=expected)


def test_all_present_and_matching():
    res = run_check("Primary: Red\nSecondary: Blue", {"Primary": "Red", "Secondary": "Blue"})
    assert res.passed
    assert res.checks["labeled_values"].details["extracted"] == {"Primary": "Red", "Secondary": "Blue"}


def test_missing_label_missing_value_and_mismatch():
    res = run_check("Primary: Green\nSecondary: Blue", {"Primary": "Red", "Secondary": "Red"})
    codes = {(i.label, i.code) for i in res.errors}
    assert codes == {("Primary", "missing_value"), ("Secondary", "value_mismatch")}
    mismatch = next(i for i in res.errors if i.code == "value_mismatch")
    assert mismatch.data == {"expected": "Red", "got": "Blue"}
    assert {(i.label, i.code) for i in run_check("Primary: Red", {"Primary": "Red", "Secondary": "Blue"}).errors} \
        == {("Secondary", "missing_label")}


def test_listed_without_value_message():
    res = run_check("Primary\nSecondary\n", {"Primary": "Red", "Secondary": "Blue"})
    assert all(i.code == "missing_value" and "listed without a value" in i.message for i in res.errors)


def test_conflicting_entries_warn_and_first_wins():
    res = run_check("Primary: Red\nSecondary: Blue\nPrimary: Blue", {"Primary": "Red", "Secondary": "Blue"})
    assert res.passed
    assert [(i.label, i.code) for i in res.warnings] == [("Primary", "conflicting_entries")]


def test_later_entry_with_value_beats_earlier_empty_heading():
    res = run_check("Primary\n\nsome intro text here\n\nPrimary: Red\nSecondary: Blue",
                    {"Primary": "Red", "Secondary": "Blue"})
    assert res.passed


def test_order_check():
    res = run_check("Secondary: Blue\nPrimary: Red", {"Primary": "Red", "Secondary": "Blue"}, check_order=True)
    assert res.passed and [i.code for i in res.warnings] == ["out_of_order"]


def test_check_values_off_only_requires_presence():
    assert run_check("Primary: Blue\nSecondary: Red", {"Primary": "Red", "Secondary": "Blue"},
                     check_values=False).passed


def test_bad_expected_and_unknown_label():
    assert run_check("x", ["not", "a", "mapping"]).errors[0].code == "no_expected"
    res = run_check("Primary: Red", {"Primary": "Red", "Tertiary": "Blue"})
    assert [(i.label, i.code) for i in res.errors] == [("Tertiary", "unknown_label")]


def test_foreign_values():
    v = Validator([ForeignValuesCheck(COLORS)])
    res = v.validate("Red and sky blue", expected={"Primary": "Red"})
    assert res.passed  # warnings only by default
    assert res.checks["foreign_values"].details["foreign"] == ["Light Blue"]
