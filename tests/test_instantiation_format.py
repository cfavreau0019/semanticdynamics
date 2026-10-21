"""The instantiation format (instantiation/instantiation.schema.yaml)."""
from pathlib import Path

import pytest
import yaml
from jsonschema import Draft202012Validator

ROOT = Path(__file__).parents[1]
SCHEMA = yaml.safe_load((ROOT / "instantiation" / "instantiation.schema.yaml").read_text(encoding="utf-8"))
VALIDATOR = Draft202012Validator(SCHEMA)


def load(path: str):
    return yaml.safe_load((ROOT / path).read_text(encoding="utf-8"))


def valid(instantiation) -> bool:
    return VALIDATOR.is_valid(instantiation)


def test_the_schema_is_a_json_schema():
    Draft202012Validator.check_schema(SCHEMA)


def test_the_reference_the_template_example_and_the_tarot_task_are_valid():
    [reference] = load("applications/template/inference_task_reference.yaml").values()
    VALIDATOR.validate(reference["instantiation"])                       # left out: null
    VALIDATOR.validate(load("applications/template/instantiation.yaml"))  # every kind of entry
    [tarot] = load("applications/tarot/tasks.yaml").values()
    VALIDATOR.validate(tarot["instantiation"])
    assert tarot["validation"]["expected"] in tarot["instantiation"]["state"]


@pytest.mark.parametrize("instantiation", [
    None,                                                                 # levels 1 and 2: no state
    "instantiation.yaml",                                                 # a file of its own
    {"state": {"tone": "whimsical", "themes": ["friendship", "curiosity"], "length": 300, "rhymes": False}},
    {"state": {"character": {"one_of": ["fox", "owl"]}}},
    {"state": {"character": {"one_of": "characters.yaml", "sample": "cycle"}}, "sample": "random", "version": 1},
    {"state": {"cards": {"steps": {"from": "vector_space_config.json", "negative_suffix": " reversed",
                                   "replacement": False, "fixed": {"Outcome": "Death reversed"}}}}},
    {"state": {"arc": {"steps": {"names": ["start", "end"], "states": ["hope", "loss"]}}}},
    {"state": {"narrator": {"personas": "tarot_personas", "sample": "exhaustive", "select": [1, "Maya Delgado"]}}},
    {"state": {"twist": {"generate": {"using": "applications.tarot.code.readings:sample_spread", "options": {"k": 3}}}}},
    {"state": {"settings": {"value": {"strict": True}}}},
    {"instants": [{"cards": {"Present": "Death"}}, {"cards": {"Present": "The Sun"}}]},   # given, not sampled
    {"instants": "draws.jsonl", "state": {"cards": {"steps": {"from": "vector_space_config.json"}}}},
])
def test_valid_instantiations(instantiation):
    assert valid(instantiation), list(VALIDATOR.iter_errors(instantiation))


@pytest.mark.parametrize("instantiation, why", [
    ({"state": {"character": {"oneof": ["fox", "owl"]}}}, "a mistyped key is an error, not a literal mapping"),
    ({"state": {"settings": {"strict": True}}}, "a literal mapping must be wrapped in value"),
    ({"state": {"character": {"one_of": ["fox"], "sampel": "cycle"}}}, "unknown option of an entry"),
    ({"state": {"character": {"one_of": ["fox"], "sample": "shuffle"}}}, "unknown sampling method"),
    ({"state": {"character": {"one_of": []}}}, "no candidates"),
    ({"state": {"character": {"one_of": ["fox"], "steps": {"from": "deck.json"}}}}, "two kinds in one entry"),
    ({"state": {"cards": {"steps": {"replacement": False}}}}, "steps without their steps and states"),
    ({"state": {"cards": {"steps": {"from": "deck.json", "names": ["a"], "states": ["b"]}}}}, "both from a file and listed"),
    ({"state": {"cards": {"steps": {"from": "deck.json", "fixed": {"Outcome": 3}}}}}, "a fixed state is a name"),
    ({"state": {"twist": {"generate": {"using": "no_function_named"}}}}, "using is module:function"),
    ({"state": {}}, "an instantiation with no entries"),
    ({"version": 1}, "no state"),
    ({"instants": []}, "no given instants"),
    ({"instants": [{}]}, "an empty given instant"),
    ({"instants": [["Death"]]}, "a given instant maps entry names to values"),
    ({"instants": "draws.jsonl", "sample": "cycle"}, "given instants are not sampled"),
    ({"state": {"tone": "calm"}, "sampling": {"method": "random"}}, "unknown top-level key"),
    ({"state": {"tone": "calm"}, "version": 2}, "unknown format version"),
])
def test_invalid_instantiations(instantiation, why):
    assert not valid(instantiation), why
