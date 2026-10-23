"""instantiation: sampling a run's instants from an InstantiationSpec, and the original Instantiation."""
import json
import random
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from applications.tarot.code.validation import REVERSED_SUFFIX
from inference.pipeline import list_runs
from inference.store import DEFAULT_ROOT, RunStore
from inference.task_config import load_task_file
from instantiation import (
    Generate, Instantiation, InstantiationError, InstantiationSpec, Literal, OneOf, Personas, Steps, draw_steps,
    fixed_size, probes, read_instantiation,
)
from personas import PersonaSet
from spec_files import file_sha256

ROOT = Path(__file__).parents[1]
TAROT = ROOT / "applications" / "tarot"
PINS = {"Present": "Death", "Outcome": "The Sun reversed"}


@dataclass
class Draws:
    """How TarotReadings drew cards before the task file: from the deck, with a draw mode and pins."""
    draw: str = "replacement"                  # or "unique"
    pins: dict = field(default_factory=dict)

    def __post_init__(self):
        deck = json.loads((TAROT / "vector_space_config.json").read_text(encoding="utf-8"))
        self.positions, self.cards = deck["step_aliases"], deck["aliases"]
        self.reversible = bool(deck.get("negative_polarity", False))
        self.states = [c + s for c in self.cards for s in (["", REVERSED_SUFFIX] if self.reversible else [""])]


def drawn_before_task_files(readings: Draws, n: int, rng: random.Random) -> list[dict]:
    """The card draws TarotReadings made before the task file (its _draw_one, kept here as the reference)."""
    def one() -> dict:
        draw = {}
        if readings.draw == "replacement":
            for pos in readings.positions:
                draw[pos] = readings.pins.get(pos) or rng.choice(readings.states)
            return draw
        used = {s.removesuffix(REVERSED_SUFFIX) for s in readings.pins.values()}
        picks = iter(rng.sample([c for c in readings.cards if c not in used], len(readings.positions) - len(readings.pins)))
        for pos in readings.positions:
            if pos in readings.pins:
                draw[pos] = readings.pins[pos]
            else:
                card = next(picks)
                draw[pos] = card + REVERSED_SUFFIX if readings.reversible and rng.random() < 0.5 else card
        return draw
    return [one() for _ in range(n)]


def cards_of(readings: Draws) -> Steps:
    """The steps entry that describes those draws."""
    return Steps(tuple(readings.positions), tuple(readings.cards), readings.reversible,
                 REVERSED_SUFFIX if readings.reversible else None, readings.draw == "replacement", readings.pins)


def column(spec: InstantiationSpec, name: str, n: int, seed: int) -> list:
    return [i.instant[name] for i in Instantiation.sample(spec, n, seed=seed)]


# ---- the tarot draws, as they were made before the task file ------------------------------
@pytest.mark.parametrize("draw", ["replacement", "unique"])
@pytest.mark.parametrize("pins", [{}, PINS])
@pytest.mark.parametrize("seed", [0, 1, 42])
def test_tarot_draws_are_those_of_tarot_readings(draw, pins, seed):
    readings = Draws(draw=draw, pins=pins)
    spec = InstantiationSpec({"cards": cards_of(readings)})
    assert column(spec, "cards", 8, seed) == drawn_before_task_files(readings, 8, random.Random(seed))


def test_the_task_file_draws_as_tarot_readings_does():
    spec = load_task_file(TAROT / "tasks.yaml")["tarot_celtic_cross"].instantiation
    assert column(spec, "cards", 8, 7) == drawn_before_task_files(Draws(), 8, random.Random(7))


@pytest.mark.parametrize("mode", ["random", "cycle"])
def test_personas_are_drawn_after_the_cards_as_before(mode):
    readings, people = Draws(), PersonaSet.load("tarot_personas")
    spec = InstantiationSpec({"cards": cards_of(readings), "persona": Personas("tarot_personas", mode)})
    rng = random.Random(3)
    cards = drawn_before_task_files(readings, 6, rng)
    personas = people.sample(rng, 6, mode)                 # the pipeline's order: inputs, then personas
    run = Instantiation.sample(spec, 6, seed=3)
    assert [i.instant["cards"] for i in run] == cards and [i.instant["persona"] for i in run] == personas
    assert [i.index for i in run] == list(range(6)) and all(i.spec is spec for i in run)


def reproduce(root: Path) -> list[str]:
    """Every tarot reading run under root, sampled again from its seed: the ids of those that differ."""
    differ = []
    runs = [r for r in list_runs(root) if r["task"] == "tarot_celtic_cross"]
    for run in runs:
        config, store = run["config"], RunStore.open(run["run_id"], root)
        options = config["task_options"]
        assert options["deck_sha256"] == file_sha256(TAROT / "vector_space_config.json")   # the deck it drew from
        state = {"cards": cards_of(Draws(options["draw"], options["pins"]))}
        if config.get("persona_set"):
            state["persona"] = Personas(config["persona_set"], config.get("persona_sampling", "random"),
                                        tuple(config.get("persona_ids") or ()))
        inputs = sorted(store.read("inputs"), key=lambda row: row["input_index"])
        again = Instantiation.sample(InstantiationSpec(state), config["n"], seed=config["seed"])
        stored = [(row["payload"], row.get("persona_id")) for row in inputs]      # no persona_id before schema 2
        if [(i.instant["cards"], i.instant.get("persona", {}).get("id")) for i in again] != stored:
            differ.append(run["run_id"])
    assert runs, f"no tarot reading runs under {root}"
    return differ


@pytest.mark.skipif(not any(DEFAULT_ROOT.glob("*/runs.jsonl")), reason="no recorded runs in data/generations")
def test_the_recorded_runs_are_sampled_again_from_their_seed():
    assert reproduce(DEFAULT_ROOT) == []


# ---- the kinds of entry --------------------------------------------------------------------------
def test_a_literal_is_copied_into_every_instant():
    spec = InstantiationSpec({"settings": Literal({"tags": ["a"]})})
    first, second = Instantiation.sample(spec, 2, seed=0)
    first.instant["settings"]["tags"].append("b")
    assert second.instant["settings"] == {"tags": ["a"]} == spec.state["settings"].value


def test_one_of_at_random_and_in_cycle():
    spec = InstantiationSpec({"colour": OneOf(("red", "green", "blue")),
                              "size": OneOf(("S", "M"), "cycle")})
    run = Instantiation.sample(spec, 30, seed=5)
    assert {i.instant["colour"] for i in run} == {"red", "green", "blue"}
    assert [i.instant["size"] for i in run[:4]] == ["S", "M", "S", "M"]
    assert [i.instant for i in run] == [i.instant for i in Instantiation.sample(spec, 30, seed=5)]
    assert [i.instant for i in run] != [i.instant for i in Instantiation.sample(spec, 30, seed=6)]


def test_the_default_method_applies_to_entries_that_choose_only():
    readings = Draws()
    spec = InstantiationSpec({"cards": cards_of(readings), "size": OneOf(("S", "M"))}, sample="cycle")
    assert (spec.method("cards"), spec.method("size")) == (None, "cycle")
    assert column(spec, "cards", 4, 1) == drawn_before_task_files(readings, 4, random.Random(1))
    assert column(spec, "size", 4, 1) == ["S", "M", "S", "M"]


def test_exhaustive_takes_every_combination_once_and_refuses_another_n():
    spec = InstantiationSpec({"a": OneOf((1, 2), "exhaustive"), "b": OneOf(("x", "y", "z"), "exhaustive"),
                              "c": OneOf(("p", "q"))})
    run = Instantiation.sample(spec, 6, seed=0)
    assert [(i.instant["a"], i.instant["b"]) for i in run] == \
        [(1, "x"), (1, "y"), (1, "z"), (2, "x"), (2, "y"), (2, "z")]
    with pytest.raises(InstantiationError, match=r"a, b gives 6 combinations \(2 x 3\): n must be 6, not 5"):
        Instantiation.sample(spec, 5, seed=0)
    with pytest.raises(InstantiationError, match="at least one instant"):
        Instantiation.sample(spec, 0)


def pick(rng, k=1, items=()):
    return sorted(rng.sample(list(items), k))


def test_generate_is_called_with_the_run_generator_and_its_options():
    spec = InstantiationSpec({"chosen": Generate(f"{__name__}:pick", {"k": 2, "items": "abcdef"})})
    rng = random.Random(9)
    assert column(spec, "chosen", 3, 9) == [pick(rng, 2, "abcdef") for _ in range(3)]


def test_an_entry_added_later_leaves_the_earlier_ones_alone():
    cards = cards_of(Draws())
    before = column(InstantiationSpec({"cards": cards}), "cards", 5, 11)
    after = InstantiationSpec({"cards": cards, "persona": Personas("tarot_personas"), "size": OneOf(("S", "M"))})
    assert column(after, "cards", 5, 11) == before


def test_an_instantiation_file(tmp_path):
    shutil.copy(ROOT / "applications" / "template" / "instantiation.yaml", tmp_path / "instantiation.yaml")
    one = Instantiation.from_file(tmp_path / "instantiation.yaml", seed=4)
    assert one.instant == Instantiation.from_file(tmp_path / "instantiation.yaml", seed=4).instant
    assert list(one.instant) == ["tone", "themes", "character", "setting", "arc", "narrator", "constraints"]
    assert one.instant["arc"]["end"] == "discovery" and len(set(one.instant["arc"].values())) == 3

    (tmp_path / "broken.yaml").write_text("version: 1\nstate: {}\n", encoding="utf-8")
    with pytest.raises(InstantiationError, match=r"instantiation \(broken.yaml\)"):
        read_instantiation(tmp_path / "broken.yaml")


def test_probes_try_every_candidate_and_persona():
    spec = InstantiationSpec({"cards": cards_of(Draws()), "size": OneOf(("S", "M", "L")),
                              "persona": Personas("tarot_personas", select=(1, 2))})
    tried = list(probes(spec))
    assert [label for label, _ in tried] == [
        "a sample instant", "size = candidate 2", "size = candidate 3", "persona = persona 2 (Denise Whitaker)"]
    base = tried[0][1]
    assert (base["size"], base["persona"]["persona_number"]) == ("S", 1)
    assert all(instant["cards"] == base["cards"] for _, instant in tried)


# ---- values already chosen ---------------------------------------------------------------------
STORY = InstantiationSpec({
    "tone": Literal("whimsical"),
    "character": OneOf(("fox", "owl")),
    "arc": Steps(("start", "end"), ("hope", "loss"), True, " averted", False, {"end": "loss"}),
    "narrator": Personas("tarot_personas", select=(1, 2)),
    "twist": Generate(f"{__name__}:pick"),
})


def test_values_already_chosen_are_checked_and_filled_in():
    given = {"character": "owl", "arc": {"end": "loss", "start": "hope averted"}, "narrator": "Maya Delgado",
             "twist": ["anything"]}
    one = Instantiation.of(given, STORY, index=3)
    assert list(one.instant) == ["tone", "character", "arc", "narrator", "twist"]   # the state's order
    assert one.instant["tone"] == "whimsical"                                         # a literal, filled in
    assert list(one.instant["arc"]) == ["start", "end"]                               # the steps' order
    assert one.instant["narrator"] == PersonaSet.load("tarot_personas").get(1)        # by name, to its record
    assert (one.spec, one.index) == (STORY, 3) and STORY.problems(given) == []
    assert Instantiation.of(one.instant, STORY).instant == one.instant                 # a record is fine as it is
    assert Instantiation.of({"anything": [1]}).instant == {"anything": [1]}           # no spec: nothing to check


@pytest.mark.parametrize("given, problem", [
    ({"tone": "grim"}, "instant.tone: must be 'whimsical', the entry's only value"),
    ({"character": "cat"}, "instant.character: 'cat' is not one of the candidates"),
    ({"arc": {"start": "joy", "end": "loss"}}, "instant.arc.start: 'joy' is not one of the states"),
    ({"arc": {"start": "hope", "end": "hope"}}, "instant.arc.end: the step is fixed to 'loss', not 'hope'"),
    ({"arc": {"start": "loss averted", "end": "loss"}}, "replacement is false, but ['loss'] appear more than once"),
    ({"arc": {"start": "hope"}}, "instant.arc: no state for ['end']"),
    ({"arc": {"start": "hope", "end": "loss", "middle": "hope"}}, "['middle'] are not steps (start, end)"),
    ({"narrator": 3}, "instant.narrator: persona 3 (Javier Ruiz) is not among those selected ([1, 2])"),
    ({"narrator": {**PersonaSet.load("tarot_personas").get(1), "description": "edited"}},
     "persona 1 (Maya Delgado) differs from its record in tarot_personas"),
    ({"narrator": True}, "a persona is given by id, number or full name, or as its record"),
    ({"villain": "wolf"}, "instant.villain: not an entry of the state"),
])
def test_values_that_could_not_have_come_from_the_spec(given, problem):
    instant = {"character": "fox", "arc": {"start": "hope", "end": "loss"}, "narrator": 1, "twist": None, **given}
    assert any(problem in p for p in STORY.problems(instant)), STORY.problems(instant)
    with pytest.raises(InstantiationError, match=re.escape(problem)):
        Instantiation.of(instant, STORY)
    assert "instant.character: missing" in STORY.problems({})


def test_a_run_of_given_instants():
    given = ({"character": "fox", "arc": {"start": "hope", "end": "loss"}, "narrator": 2, "twist": 1},
             {"character": "owl", "arc": {"start": "hope averted", "end": "loss"}, "narrator": 1, "twist": 2})
    spec = InstantiationSpec(STORY.state, instants=given)
    assert fixed_size(spec) == 2
    run = Instantiation.sample(spec)                                     # n is theirs
    assert [i.instant["twist"] for i in run] == [1, 2] and [i.index for i in run] == [0, 1]
    assert [i.instant["narrator"]["persona_number"] for i in run] == [2, 1]
    assert Instantiation.sample(spec, 2, seed=1)[1].instant == run[1].instant           # nothing is drawn
    with pytest.raises(InstantiationError, match="the instantiation gives 2 instants: n must be 2, not 3"):
        Instantiation.sample(spec, 3)
    with pytest.raises(InstantiationError, match=re.escape("instants[1].character: 'cat'")):
        Instantiation.sample(InstantiationSpec(STORY.state, instants=(given[0], {**given[1], "character": "cat"})))
    assert [label for label, _ in probes(spec)] == ["instants[0]", "instants[1]"]

    bare = InstantiationSpec({}, instants=({"a": 1, "b": 2}, {"a": 3}))             # no state: taken as given
    assert [i.instant for i in Instantiation.sample(bare)] == [{"a": 1, "b": 2}, {"a": 3}] and bare.entries == ("a",)


def test_n_can_be_left_out_only_when_the_spec_fixes_it():
    exhaustive = InstantiationSpec({"a": OneOf((1, 2), "exhaustive"), "b": OneOf((3, 4, 5), "exhaustive")})
    assert fixed_size(exhaustive) == 6 and len(Instantiation.sample(exhaustive)) == 6
    assert fixed_size(STORY) is None
    with pytest.raises(InstantiationError, match="n is needed"):
        Instantiation.sample(STORY)


# ---- the original use --------------------------------------------------------------------------
def test_the_original_instantiation_still_works():
    from coordinate_systems.state_instantiation import Instantiation as FromCoordinateSystems
    assert FromCoordinateSystems is Instantiation

    states, steps = ["a", "b", "c", "d"], ["one", "two", "three"]
    drawn = Instantiation(states, steps, {"two": "c"}, rng=random.Random(2))
    assert drawn.instant == draw_steps(random.Random(2), steps, states, {"two": "c"})
    assert list(drawn.instant) == steps and drawn.instant["two"] == "c" and drawn.spec is None
    unique = Instantiation(states, steps, rng=random.Random(2), replacement=False).instant
    assert len(set(unique.values())) == 3
    with pytest.raises(ValueError, match="unknown steps"):
        Instantiation(states, steps, {"four": "a"})
    with pytest.raises(ValueError, match="unknown steps"):
        Instantiation(states, steps, {"one": "z"})
