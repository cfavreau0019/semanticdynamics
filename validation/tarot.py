"""
Tarot configuration of the generic checks: the 78-card vocabulary, Celtic Cross position
aliases, and orientation handling. States follow the population generator's convention:
"<card>" (upright) or "<card> reversed", e.g. "Death reversed".

    from validation.tarot import celtic_cross_validator
    v = celtic_cross_validator()
    result = v.validate(reading_text, expected=instant, finish_reason="stop")
    result.passed, result.summary()
"""
import re
from typing import Mapping, Optional, Sequence

from validation.checks import ForeignValuesCheck, LabeledValuesCheck, NonEmpty, NoRefusal, NotTruncated
from validation.core import Validator
from validation.extract import LabeledEntryExtractor, VocabularyMatcher, default_aliases

MAJOR_ARCANA = [
    "The Fool", "The Magician", "The High Priestess", "The Empress", "The Emperor", "The Hierophant",
    "The Lovers", "The Chariot", "Strength", "The Hermit", "The Wheel of Fortune", "Justice",
    "The Hanged Man", "Death", "Temperance", "The Devil", "The Tower", "The Star", "The Moon", "The Sun",
    "Judgement", "The World",
]
RANKS = ["Ace", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten",
         "Page", "Knight", "Queen", "King"]
SUITS = ["Wands", "Cups", "Swords", "Pentacles"]
MINOR_ARCANA = [f"{r} of {s}" for s in SUITS for r in RANKS]
TAROT_CARDS = MAJOR_ARCANA + MINOR_ARCANA  # names match sandbox/vector_space_config.json

REVERSED_SUFFIX = " reversed"
_REVERSED = re.compile(r"\b(?:reversed|reversal|inverted|upside[- ]down|rx)\b|\(\s*r\s*\)", re.IGNORECASE)
_NEGATED_REVERSED = re.compile(r"\bnot\s+(?:reversed|inverted)\b", re.IGNORECASE)


def card_aliases(cards: Sequence[str] = TAROT_CARDS, loose: bool = True) -> dict[str, list[str]]:
    """
    {card: [aliases]}. loose=True adds forms models commonly write: majors without "The",
    "Judgment", "Wheel of Fortune", numerals ("2 of Cups"). loose=False keeps only multi-word
    variants, for scanning free prose where single common words ("Sun", "World") would misfire.
    """
    numerals = {r: str(i) for i, r in enumerate(RANKS[1:10], start=2)}
    out: dict[str, list[str]] = {}
    for card in cards:
        aliases = []
        if card.startswith("The "):
            bare = card[4:]
            if loose or " " in bare:
                aliases.append(bare)
        if card == "Judgement":
            aliases.append("Judgment")
        rank, _, suit = card.partition(" of ")
        if loose and suit and rank in numerals:
            aliases.append(f"{numerals[rank]} of {suit}")
        out[card] = aliases
    return out


def parse_state(state: str) -> tuple[str, bool]:
    """'Death reversed' -> ('Death', True); 'Death' -> ('Death', False)."""
    if state.endswith(REVERSED_SUFFIX):
        return state[:-len(REVERSED_SUFFIX)], True
    return state, False


class TarotCardMatcher(VocabularyMatcher):
    """Finds a card in a segment and reads its orientation from the same segment."""

    def __init__(self, cards: Sequence[str] = TAROT_CARDS, loose: bool = True, case_sensitive: bool = False):
        super().__init__(card_aliases(cards, loose), case_sensitive=case_sensitive)

    def find(self, segment: str) -> Optional[str]:
        card = super().find(segment)
        if card is None:
            return None
        reversed_ = bool(_REVERSED.search(segment)) and not _NEGATED_REVERSED.search(segment)
        return card + REVERSED_SUFFIX if reversed_ else card

    def identity(self, value: str) -> str:
        return parse_state(value)[0]


def compare_cards(expected: str, got: str) -> Optional[tuple[str, str]]:
    """Separates a wrong card from a right card with the wrong orientation."""
    (e_card, e_rev), (g_card, g_rev) = parse_state(expected), parse_state(got)
    if e_card.casefold() != g_card.casefold():
        return "wrong_card", f"expected {expected!r}, got {got!r}"
    if e_rev != g_rev:
        return "orientation_mismatch", f"expected {'reversed' if e_rev else 'upright'} {e_card}, " \
                                       f"got {'reversed' if g_rev else 'upright'}"
    return None


CELTIC_CROSS_POSITIONS: dict[str, list[str]] = {
    "Present": ["Present", "Present Situation", "Current Situation"],
    "Challenge": ["Challenge", "Challenges", "Obstacle", "Crossing Card"],
    "Conscious Goal": ["Conscious Goal", "Conscious", "Goal"],
    "Root Cause/Foundation": [*default_aliases("Root Cause/Foundation"), "Subconscious"],
    "Recent Past": ["Recent Past", "Past"],
    "Near Future": ["Near Future", "Immediate Future", "Future"],
    "Your Attitude": ["Your Attitude", "Attitude", "Self", "Yourself"],
    "External Influences": ["External Influences", "External Influence", "Environment", "Outside Influences"],
    "Hopes and Fears": ["Hopes and Fears", "Hopes & Fears"],
    "Outcome": ["Outcome", "Final Outcome", "Likely Outcome"],
}


def spread_validator(
    positions: Mapping[str, Sequence[str]] | Sequence[str],
    cards: Sequence[str] = TAROT_CARDS,
    name: str = "tarot_spread",
    check_order: bool = True,
    inline_window: int = 40,
    min_chars: int = 200,
) -> Validator:
    """
    Validator for a reading of any spread. `positions` is a list of position names or
    {position: [aliases]}. Checks:
      non_empty, not_truncated, no_refusal
      positions      — every position present, with the drawn card and orientation
      foreign_cards  — (warning) cards mentioned that weren't drawn
    """
    extractor = LabeledEntryExtractor(positions, TarotCardMatcher(cards), inline_window=inline_window)
    return Validator([
        NonEmpty(min_chars),
        NotTruncated(),
        NoRefusal(),
        LabeledValuesCheck(extractor, name="positions", compare=compare_cards, check_order=check_order),
        ForeignValuesCheck(TarotCardMatcher(cards, loose=False, case_sensitive=True), name="foreign_cards"),
    ], name=name)


def celtic_cross_validator(**kwargs) -> Validator:
    """spread_validator preconfigured for the 10-card Celtic Cross."""
    kwargs.setdefault("name", "celtic_cross")
    return spread_validator(CELTIC_CROSS_POSITIONS, **kwargs)
