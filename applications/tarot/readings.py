"""
Tarot readings: the task that asks for a Celtic Cross reading of a sampled card draw. It
samples the draws from the deck config, hands them to the prompt templates
(prompts/library/tarot/) and checks each reading with the validator in validation.py.

The deck config (sandbox/vector_space_config.json, or $VECTOR_SPACE_CONFIG) supplies the
positions (step_aliases), the cards (aliases) and whether cards can be reversed
(negative_polarity). A draw is {position: state}, state = "<card>" or "<card> reversed",
the same shape as coordinate_systems.state_instantiation.Instantiation produces.
"""
import argparse
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Mapping, Optional

from applications.tarot.validation import CELTIC_CROSS_POSITIONS, REVERSED_SUFFIX, celtic_cross_validator, parse_state
from data_generation.pipeline import list_runs
from data_generation.store import REPO_ROOT
from data_generation.task import InferenceTask

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


def _pin_arg(text: str) -> tuple[str, str]:
    if "=" not in text:
        raise argparse.ArgumentTypeError("use POSITION=CARD, e.g. 'Outcome=Death reversed'")
    pos, card = text.split("=", 1)
    return pos.strip(), card.strip()


class TarotReadings(InferenceTask):
    """
    Celtic Cross readings of sampled card draws.

    Prompt templates live in prompts/library/tarot/ and receive these variables:
        cards     {position: card state}, i.e. the draw
        positions the spread's position names, in order
        persona   a persona record, or None
    The default templates are pinned to exact versions, so adding a template version does not
    change what a run produces: celtic_cross_v4 without personas, celtic_cross_v2 (which
    accepts `persona`) with them.
    """
    name = "tarot_celtic_cross"
    input_type = "tarot_draw"
    default_template = "celtic_cross_v4"
    persona_template = "celtic_cross_v2"

    def __init__(self, deck_config: Optional[str | Path] = None, draw: str = "replacement",
                 pins: Optional[Mapping[str, str]] = None):
        if draw not in DRAW_MODES:
            raise ValueError(f"draw must be one of {DRAW_MODES}, got {draw!r}")
        self.deck_config = resolve_deck_config(deck_config)
        raw = self.deck_config.read_bytes()
        config = json.loads(raw)
        self.deck_sha256 = hashlib.sha256(raw).hexdigest()
        self.positions: list[str] = list(config["step_aliases"])
        if self.positions != list(CELTIC_CROSS_POSITIONS):
            raise ValueError(f"{self.deck_config} is not a Celtic Cross deck: its positions are {self.positions}, "
                             f"not {list(CELTIC_CROSS_POSITIONS)}")
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

    def template_variables(self, payload: dict, persona: Optional[dict]) -> dict:
        return {"cards": payload, "positions": list(payload), "persona": persona}

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
        return celtic_cross_validator(cards=self.cards, name=self.name)

    # ---- command line: python -m data_generation.generate run tarot_celtic_cross ... --------
    @classmethod
    def add_arguments(cls, parser) -> None:
        parser.add_argument("--draw", choices=DRAW_MODES, default="replacement",
                            help="replacement: any card per position, repeats allowed (as Instantiation); "
                                 "unique: no card twice (default: %(default)s)")
        parser.add_argument("--pin", type=_pin_arg, action="append", default=[], metavar="POSITION=CARD",
                            help="fix a position's card in every draw (repeatable)")
        parser.add_argument("--deck-config", help="deck config JSON (default: $VECTOR_SPACE_CONFIG or sandbox's)")

    @classmethod
    def from_args(cls, args) -> "TarotReadings":
        return cls(args.deck_config, draw=args.draw, pins=dict(args.pin))

    @classmethod
    def from_options(cls, options: dict) -> "TarotReadings":
        return cls(options["deck_config"], draw=options["draw"], pins=options["pins"])

    def run_warnings(self, config, root) -> list[str]:
        """Same seed + same draw settings in one dataset would add the same card draws again."""
        if config.dataset is None or config.seed is None:
            return []
        mine = self.options()
        same_draws = lambda o: all(o.get(k) == mine[k] for k in ("deck_sha256", "draw", "pins"))
        clashes = [r["run_id"] for r in list_runs(root, config.dataset)
                   if r["config"].get("seed") == config.seed and same_draws(r["config"].get("application_options") or {})]
        if not clashes:
            return []
        return [f"dataset {config.dataset!r} already has run(s) with --seed {config.seed} and the same draw settings "
                f"({', '.join(clashes)}); this run will repeat their card draws. Omit --seed (a fresh one is chosen "
                f"and recorded) or use a different one."]
