"""Tarot validation against the formats real models produced (Hermes on Featherless, OpenAI)."""
import pytest

from validation.tarot import (
    CELTIC_CROSS_POSITIONS, TAROT_CARDS, TarotCardMatcher, card_aliases, celtic_cross_validator, compare_cards,
    parse_state, spread_validator,
)

INSTANT = {
    "Present": "Three of Pentacles",
    "Challenge": "Death reversed",
    "Conscious Goal": "Ace of Swords",
    "Root Cause/Foundation": "Eight of Wands reversed",
    "Recent Past": "Page of Cups reversed",
    "Near Future": "The Hanged Man",
    "Your Attitude": "The World reversed",
    "External Influences": "Queen of Wands",
    "Hopes and Fears": "Seven of Pentacles",
    "Outcome": "Seven of Pentacles reversed",
}
FILLER = "This card speaks to a meaningful theme in your life that is worth reflecting on carefully."


def hermes_style(instant=INSTANT):
    body = "\n\n".join(f"{i}. {pos}: {card} - {FILLER}" for i, (pos, card) in enumerate(instant.items(), 1))
    return f"Let's go through the cards you've drawn:\n\n{body}\n\nOverall, this reading suggests growth."


def openai_style(instant=INSTANT):
    short = {"Root Cause/Foundation": "Foundation", "Conscious Goal": "Conscious goal"}
    body = "\n\n".join(f"{i}. **{short.get(pos, pos)} — {card}:** {FILLER}"
                       for i, (pos, card) in enumerate(instant.items(), 1))
    return (f"Your spread feels like a reading about **work in progress**.\n\n### The Celtic Cross\n\n{body}\n\n"
            "### Overall\n\nThe contrast between **wanting a breakthrough** (Ace of Swords) and **needing a change "
            "of perspective first** (The Hanged Man); the two Sevens of Pentacles make time impossible to ignore.")


def prose_style(instant=INSTANT):
    return " ".join(f"In the {pos}, {card} points to an important theme for you. {FILLER}"
                    for pos, card in instant.items())


V = celtic_cross_validator()


# ---- vocabulary / parsing ------------------------------------------------------------
def test_deck_matches_population_generator_config():
    import json
    from pathlib import Path
    config = json.loads((Path(__file__).parents[1] / "sandbox" / "vector_space_config.json").read_text())
    assert sorted(TAROT_CARDS) == sorted(config["aliases"])
    assert list(CELTIC_CROSS_POSITIONS) == config["step_aliases"]


def test_parse_state():
    assert parse_state("Death reversed") == ("Death", True)
    assert parse_state("The Wheel of Fortune") == ("The Wheel of Fortune", False)


@pytest.mark.parametrize("segment, expected", [
    ("Death reversed", "Death reversed"),
    ("Death (Reversed)", "Death reversed"),
    ("Reversed Death", "Death reversed"),
    ("Death, inverted", "Death reversed"),
    ("Death (R)", "Death reversed"),
    ("Death (upright)", "Death"),
    ("Death, not reversed", "Death"),
    ("Hanged Man", "The Hanged Man"),
    ("Wheel of Fortune reversed", "The Wheel of Fortune reversed"),
    ("Judgment", "Judgement"),
    ("7 of Cups", "Seven of Cups"),
    ("the queen of wands", "Queen of Wands"),
    ("no card here", None),
])
def test_card_matcher(segment, expected):
    assert TarotCardMatcher().find(segment) == expected


def test_strict_aliases_skip_common_single_words():
    strict = card_aliases(loose=False)
    assert strict["The Sun"] == [] and strict["The Hanged Man"] == ["Hanged Man"]
    assert "Sun" in card_aliases()["The Sun"]


def test_compare_cards():
    assert compare_cards("Death reversed", "Death reversed") is None
    assert compare_cards("Death reversed", "Death")[0] == "orientation_mismatch"
    assert compare_cards("Death", "The Tower")[0] == "wrong_card"


# ---- full readings -----------------------------------------------------------------
@pytest.mark.parametrize("make", [hermes_style, openai_style])
def test_well_formed_readings_pass(make):
    res = V.validate(make(), expected=INSTANT, finish_reason="stop")
    assert res.passed, res.summary()
    details = res.checks["positions"].details
    assert details["n_found"] == 10 and set(details["modes"].values()) == {"line"}
    assert details["extracted"] == INSTANT
    assert not res.warnings, res.summary()  # "Sevens of Pentacles" (plural) is not a foreign card


def test_prose_reading_passes_via_inline_fallback():
    res = V.validate(prose_style(), expected=INSTANT, finish_reason="stop")
    assert res.passed, res.summary()
    assert set(res.checks["positions"].details["modes"].values()) == {"inline"}


def test_model_asking_user_to_draw_cards_fails():
    text = ("Certainly, I'll help you with your Celtic Cross Tarot reading.\n\nPlease draw your ten cards:\n" +
            "\n".join(f"{i}. {p}" for i, p in enumerate(INSTANT, 1)) + "\n\nOnce you have drawn all the cards, "
            "I will proceed with the reading and provide a thoughtful overview for you.")
    res = V.validate(text, expected=INSTANT, finish_reason="stop")
    assert not res.passed
    assert {i.code for i in res.errors} == {"missing_value"} and len(res.errors) == 10


def test_dropped_orientation_and_wrong_card():
    got = dict(INSTANT, Challenge="Death", Outcome="The Tower")
    res = V.validate(hermes_style(got), expected=INSTANT, finish_reason="stop")
    assert not res.passed
    assert {(i.label, i.code) for i in res.errors} == {("Challenge", "orientation_mismatch"),
                                                       ("Outcome", "wrong_card")}
    assert [i.data["value"] for i in res.warnings if i.code == "foreign_value"] == ["The Tower"]


def test_missing_position_and_truncation():
    partial = {k: v for k, v in INSTANT.items() if k != "Outcome"}
    res = V.validate(hermes_style(partial), expected=INSTANT, finish_reason="length")
    assert {(i.label, i.code) for i in res.errors} == {("Outcome", "missing_label"), (None, "truncated")}


def test_hallucinated_card_in_overview_is_a_warning():
    text = hermes_style() + " The energy of The Star also hovers over this spread."
    res = V.validate(text, expected=INSTANT, finish_reason="stop")
    assert res.passed
    assert [i.data["value"] for i in res.warnings] == ["The Star"]


def test_out_of_order_positions_warn():
    reordered = dict(reversed(list(INSTANT.items())))
    res = V.validate(hermes_style(reordered), expected=INSTANT, finish_reason="stop")
    assert res.passed and [i.code for i in res.warnings] == ["out_of_order"]


def test_refusal_fails():
    res = V.validate("I'm sorry, but I can't help with tarot readings. " + FILLER * 3, expected=INSTANT)
    assert "forbidden_pattern" in {i.code for i in res.errors}


def test_other_spreads_generalise():
    three_card = {"Past": "The Fool", "Present": "Two of Cups reversed", "Future": "The Sun"}
    v = spread_validator(["Past", "Present", "Future"], min_chars=10)
    text = "Past: The Fool - beginnings.\nPresent: Two of Cups reversed - tension.\nFuture: The Sun - joy."
    assert v.validate(text, expected=three_card).passed
    assert not v.validate(text.replace("The Sun", "The Moon"), expected=three_card).passed


# ---- readings written as connected prose (celtic_cross_v3 and later) --------------------
def narrative(instant=INSTANT, name_positions=True, card_first=True):
    """One paragraph per card; position and card share a sentence, in either order, or the position is left out."""
    parts = []
    for pos, card in instant.items():
        where = pos.split("/")[-1]
        if not name_positions:
            parts.append(f"The **{card}** brings an important theme forward. {FILLER}")
        elif card_first:
            parts.append(f"The **{card}** in the {where} position brings an important theme forward. {FILLER}")
        else:
            parts.append(f"As your {where}, the **{card}** brings an important theme forward. {FILLER}")
    return "### The reading\n\n" + "\n\n".join(parts) + "\n\n### Overview\n\nA spread about steady growth."


@pytest.mark.parametrize("card_first", [True, False])
def test_position_and_card_in_one_sentence_pass_in_either_order(card_first):
    res = V.validate(narrative(card_first=card_first), expected=INSTANT, finish_reason="stop")
    assert res.passed and not res.warnings, res.summary()
    details = res.checks["positions"].details
    assert details["extracted"] == INSTANT and set(details["modes"].values()) == {"inline"}


def test_cards_discussed_without_naming_positions_pass_with_warnings():
    text = narrative(name_positions=False)
    res = V.validate(text, expected=INSTANT, finish_reason="stop")
    assert res.passed and {i.code for i in res.warnings} == {"unlabeled_value"} and len(res.warnings) == 10
    assert res.checks["positions"].details["unlabeled"] == list(INSTANT)
    strict = celtic_cross_validator(require_positions=True).validate(text, expected=INSTANT, finish_reason="stop")
    assert not strict.passed and {i.code for i in strict.errors} == {"missing_label"}


def test_narrative_errors_are_still_errors():
    absent = narrative({k: v for k, v in INSTANT.items() if k != "Outcome"})          # a drawn card never appears
    assert [(i.label, i.code) for i in V.validate(absent, expected=INSTANT, finish_reason="stop").errors] == \
        [("Outcome", "missing_label")]
    reversed_position = next(k for k, v in INSTANT.items() if v.endswith(" reversed"))
    upright = narrative(dict(INSTANT, **{reversed_position: INSTANT[reversed_position][:-len(" reversed")]}))
    assert [(i.label, i.code) for i in V.validate(upright, expected=INSTANT, finish_reason="stop").errors] == \
        [(reversed_position, "orientation_mismatch")]                                  # never called reversed
    swapped = narrative(dict(INSTANT, Outcome="The Tower"))
    assert [(i.label, i.code) for i in V.validate(swapped, expected=INSTANT, finish_reason="stop").errors] == \
        [("Outcome", "wrong_card")]


def test_shortened_later_mentions_and_repeated_cards_are_not_contradictions():
    draw = dict(INSTANT, Present="Nine of Pentacles reversed", Outcome="Nine of Pentacles")
    text = (narrative({k: v for k, v in draw.items() if k not in ("Present", "Outcome")}) +
            "\n\nThe **Nine of Pentacles reversed** puts security at the centre. Later the **Nine of Pentacles upright** "
            "closes the spread.\n\nThe Nine of Pentacles in the Present and again as Outcome is the spine of this reading.")
    res = V.validate(text, expected=draw, finish_reason="stop")
    assert res.passed, res.summary()
    assert {i.label for i in res.warnings if i.code == "unlabeled_value"} == {"Present"}


@pytest.mark.parametrize("text, expected", [
    ("The Moon reversed in External Influences and the Sun in the Recent Past.", ["The Moon reversed", "The Sun"]),
    ("the reversed Queen of Wands, then Death (reversed); Judgement, reversed.",
     ["Queen of Wands reversed", "Death reversed", "Judgement reversed"]),
    ("Your inner strength and the wider world matter here.", []),                       # ordinary words, not cards
    ("Strength meets The World.", ["Strength", "The World"]),
    ("The Sun, and then the Moon reversed.", ["The Sun", "The Moon reversed"]),         # 'reversed' belongs to the Moon
])
def test_card_mentions_read_orientation_next_to_each_card(text, expected):
    from validation.tarot import TarotCardMatcher
    assert TarotCardMatcher().mentions(text) == expected
