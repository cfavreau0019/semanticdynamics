"""
Tarot readings as a data_generation Application: sampling card draws from the deck config,
prompt templates, and the regex validator from validation.tarot.

The deck config (sandbox/vector_space_config.json, or $VECTOR_SPACE_CONFIG) supplies the
positions (step_aliases), the cards (aliases) and whether cards can be reversed
(negative_polarity). A draw is {position: state}, state = "<card>" or "<card> reversed",
the same shape as coordinate_systems.state_instantiation.Instantiation produces.
"""
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Mapping, Optional

from data_generation.pipeline import Application
from data_generation.store import REPO_ROOT
from validation.extract import default_aliases
from validation.tarot import CELTIC_CROSS_POSITIONS, REVERSED_SUFFIX, parse_state, spread_validator

DEFAULT_DECK_CONFIG = REPO_ROOT / "sandbox" / "vector_space_config.json"
DRAW_MODES = ("replacement", "unique")


def resolve_deck_config(path: Optional[str | Path] = None) -> Path:
    """Explicit path, else $VECTOR_SPACE_CONFIG (absolute, repo-relative or sandbox-relative), else the default."""
    candidates = []
    for p in (path, os.getenv("VECTOR_SPACE_CONFIG")):
        if p:
            p = Path(p)
            candidates += [p] if p.is_absolute() else [Path.cwd() / p, REPO_ROOT / p, REPO_ROOT / "sandbox" / p]
            if path is not None:  # an explicit path must exist
                break
    candidates.append(DEFAULT_DECK_CONFIG)
    for c in candidates:
        if c.exists():
            return c.resolve()
    raise FileNotFoundError(f"Deck config not found; tried {[str(c) for c in candidates]}")


def celtic_cross_v1(draw: dict) -> tuple[Optional[str], str]:
    """The prompt used in sandbox/population_generator.ipynb."""
    cards = "\n".join(f"{pos}: {state}" for pos, state in draw.items())
    return None, ("You're a tarot reader. Can you read my Celtic cross tarot? Give an overview of the reading "
                  f"after, any thoughts you have. I'll draw the cards:\n\n{cards}")


def spread_v1(draw: dict) -> tuple[Optional[str], str]:
    """Spread-agnostic wording, for decks/configs other than the Celtic Cross."""
    cards = "\n".join(f"{pos}: {state}" for pos, state in draw.items())
    return None, ("You're a tarot reader. Please read this tarot spread, interpreting each position and its card, "
                  f"then give an overview of the reading and any thoughts you have.\n\n{cards}")


class TarotReadings(Application):
    input_type = "tarot_draw"
    templates = {"celtic_cross_v1": celtic_cross_v1, "spread_v1": spread_v1}

    def __init__(self, deck_config: Optional[str | Path] = None, draw: str = "replacement",
                 pins: Optional[Mapping[str, str]] = None):
        if draw not in DRAW_MODES:
            raise ValueError(f"draw must be one of {DRAW_MODES}, got {draw!r}")
        self.deck_config = resolve_deck_config(deck_config)
        raw = self.deck_config.read_bytes()
        config = json.loads(raw)
        self.deck_sha256 = hashlib.sha256(raw).hexdigest()
        self.positions: list[str] = list(config["step_aliases"])
        self.cards: list[str] = list(config["aliases"])
        self.reversible = bool(config.get("negative_polarity", False))
        self.states = [c + s for c in self.cards for s in ([ "", REVERSED_SUFFIX] if self.reversible else [""])]
        self.draw = draw
        self.pins = dict(pins or {})
        for pos, state in self.pins.items():
            if pos not in self.positions:
                raise ValueError(f"Pinned position {pos!r} not in {self.positions}")
            if state not in self.states:
                raise ValueError(f"Pinned card {state!r} is not a valid state (e.g. 'Death' or 'Death reversed')")
        if draw == "unique":
            pinned_cards = [parse_state(s)[0] for s in self.pins.values()]
            if len(set(pinned_cards)) != len(pinned_cards):
                raise ValueError("draw='unique' but the same card is pinned twice")
            if len(self.positions) > len(self.cards):
                raise ValueError("More positions than cards; unique draws are impossible")

        self.is_celtic_cross = self.positions == list(CELTIC_CROSS_POSITIONS)
        self.name = "tarot_celtic_cross" if self.is_celtic_cross else "tarot_spread"
        self.default_template = "celtic_cross_v1" if self.is_celtic_cross else "spread_v1"

    def options(self) -> dict:
        return {"deck_config": str(self.deck_config), "deck_sha256": self.deck_sha256, "draw": self.draw,
                "pins": self.pins, "positions": self.positions}

    def sample_inputs(self, n: int, rng: random.Random) -> list[dict]:
        return [self._draw_one(rng) for _ in range(n)]

    def _draw_one(self, rng: random.Random) -> dict:
        draw = {}
        if self.draw == "replacement":  # same semantics as Instantiation: any state, repeats allowed
            for pos in self.positions:
                draw[pos] = self.pins.get(pos) or rng.choice(self.states)
            return draw
        used = {parse_state(s)[0] for s in self.pins.values()}
        free_cards = [c for c in self.cards if c not in used]
        picks = iter(rng.sample(free_cards, len(self.positions) - len(self.pins)))
        for pos in self.positions:
            if pos in self.pins:
                draw[pos] = self.pins[pos]
            else:
                card = next(picks)
                draw[pos] = card + REVERSED_SUFFIX if self.reversible and rng.random() < 0.5 else card
        return draw

    def input_items(self, payload: dict) -> list[dict]:
        items = []
        for pos, state in payload.items():
            card, reversed_ = parse_state(state)
            items.append({"label": pos, "value": state, "entity": card,
                          "qualifier": "reversed" if reversed_ else "upright"})
        return items

    def validator(self):
        aliases = {p: CELTIC_CROSS_POSITIONS.get(p, default_aliases(p)) for p in self.positions}
        return spread_validator(aliases, cards=self.cards, name=self.name)
