"""instantiation: sampling a run's instants from an InstantiationSpec, and the original Instantiation."""
import random
import shutil
from pathlib import Path

import pytest

from applications.tarot.code.readings import DRAW_MODES, REVERSED_SUFFIX, TarotReadings
from inference.pipeline import list_runs
from inference.store import DEFAULT_ROOT, RunStore
from inference.task_file import load_task_file
from instantiation import (
    Generate, Instantiation, InstantiationError, InstantiationSpec, Literal, OneOf, Personas, Steps, draw_steps,
    probes, read_instantiation,
)
from personas import PersonaSet

ROOT = Path(__file__).parents[1]
TAROT = ROOT / "applications" / "tarot"
PINS = {"Present": "Death", "Outcome": "The Sun reversed"}


def cards_of(readings: TarotReadings) -> Steps:
    """The steps entry that describes a TarotReadings' draws."""
    return Steps(tuple(readings.positions), tuple(readings.cards), readings.reversible,
                 REVERSED_SUFFIX if readings.reversible else None, readings.draw == "replacement", readings.pins)


def column(spec: InstantiationSpec, name: str, n: int, seed: int) -> list:
    return [i.instant[name] for i in Instantiation.sample(spec, n, seed=seed)]


# ---- the tarot draws, as TarotReadings made them -------------------------------------------------
@pytest.mark.parametrize("draw", DRAW_MODES)
@pytest.mark.parametrize("pins", [{}, PINS])
@pytest.mark.parametrize("seed", [0, 1, 42])
def test_tarot_draws_are_those_of_tarot_readings(draw, pins, seed):
    readings = TarotReadings(draw=draw, pins=pins)
    spec = InstantiationSpec({"cards": cards_of(readings)})
    assert column(spec, "cards", 8, seed) == readings.sample_inputs(8, random.Random(seed))


def test_the_task_file_draws_as_tarot_readings_does():
    spec = load_task_file(TAROT / "tasks.yaml")["tarot_celtic_cross"].instantiation
    assert column(spec, "cards", 8, 7) == TarotReadings().sample_inputs(8, random.Random(7))


@pytest.mark.parametrize("mode", ["random", "cycle"])
def test_personas_are_drawn_after_the_cards_as_before(mode):
    readings, people = TarotReadings(), PersonaSet.load("tarot_personas")
    spec = InstantiationSpec({"cards": cards_of(readings), "persona": Personas("tarot_personas", mode)})
    rng = random.Random(3)
    cards = readings.sample_inputs(6, rng)
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
        state = {"cards": cards_of(TarotReadings.from_options(config["task_options"]))}
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
    readings = TarotReadings()
    spec = InstantiationSpec({"cards": cards_of(readings), "size": OneOf(("S", "M"))}, sample="cycle")
    assert (spec.method("cards"), spec.method("size")) == (None, "cycle")
    assert column(spec, "cards", 4, 1) == readings.sample_inputs(4, random.Random(1))
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
    cards = cards_of(TarotReadings())
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
    spec = InstantiationSpec({"cards": cards_of(TarotReadings()), "size": OneOf(("S", "M", "L")),
                              "persona": Personas("tarot_personas", select=(1, 2))})
    tried = list(probes(spec))
    assert [label for label, _ in tried] == [
        "a sample instant", "size = candidate 2", "size = candidate 3", "persona = persona 2 (Denise Whitaker)"]
    base = tried[0][1]
    assert (base["size"], base["persona"]["persona_number"]) == ("S", 1)
    assert all(instant["cards"] == base["cards"] for _, instant in tried)


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
