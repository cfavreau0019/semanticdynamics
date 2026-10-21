"""inference.task_file: task files read into checked TaskSpecs, and the problems they report."""
import io
import json
import shutil
import sys
import textwrap
from pathlib import Path

import pytest
import yaml

import applications
from applications.tarot.code.readings import TarotReadings
from inference.task_file import TaskFileError, load_task_file, main
from instantiation import Literal, OneOf, Personas, Steps
from instantiation.files import read_yaml
from prompts import get_template

ROOT = Path(__file__).parents[1]
TAROT = ROOT / "applications" / "tarot"
DECK = TAROT / "vector_space_config.json"
REFERENCE = ROOT / "applications" / "template" / "inference_task_reference.yaml"

BASE = """
t:
  description: A test task.
  engine: inference
  prompt:
    templates:
      - inline: "Say hello."
  model:
    provider: openai
"""


def write(tmp_path: Path, text: str, name: str = "tasks.yaml") -> Path:
    path = tmp_path / name
    path.write_text(textwrap.dedent(text), encoding="utf-8")
    return path


def problems(tmp_path: Path, text: str) -> list[str]:
    with pytest.raises(TaskFileError) as caught:
        load_task_file(write(tmp_path, text))
    return caught.value.problems


# ---- the tarot task -------------------------------------------------------------------------
def test_the_tarot_task_reads_as_written():
    [spec] = load_task_file(TAROT / "tasks.yaml").values()
    assert (spec.name, spec.engine, spec.version, spec.created_ts) == ("tarot_celtic_cross", "inference", 1, "2026-10-19")
    assert spec.implemented_by == "applications.tarot.code.readings:TarotReadings"

    [template] = spec.prompt.templates
    assert template.id == "celtic_cross_v4" and template.template.sha256 == get_template("celtic_cross_v4").sha256
    assert spec.prompt.retired_templates == ("celtic_cross_v1", "celtic_cross_v3") and spec.prompt.sample is None

    cards = spec.instantiation.state["cards"]
    readings = TarotReadings()
    assert isinstance(cards, Steps) and cards.file == readings.deck_config
    assert cards.names == tuple(readings.positions) and cards.states == tuple(readings.cards)
    assert (cards.negative_polarity, cards.negative_suffix, cards.replacement, dict(cards.fixed)) == \
        (True, " reversed", True, {})

    assert (spec.model.provider, spec.model.name, dict(spec.model.parameters)) == \
        ("openai", "gpt-6-luna", {"max_tokens": 3600})
    assert (spec.execution.mode, spec.execution.workers, spec.execution.n) == ("live", 2, 1)
    assert spec.execution.seed is None and spec.execution.retry_mode is None          # not given: None
    assert spec.validation.custom_validator == "applications.tarot.code.validation:celtic_cross_validator"
    assert (spec.validation.expected, list(spec.validation.checks), spec.validation.retries) == \
        ("cards", ["non_empty", "not_truncated", "no_refusal"], 2)
    assert spec.references["templates"] == {"celtic_cross_v4": template.template.sha256}
    assert set(spec.references["files"]) == {"vector_space_config.json"} and len(spec.sha256) == 64


def test_task_files_and_task_classes_agree():
    specs = applications.task_specs()
    assert set(specs) == {"tarot_celtic_cross"} and applications.task_files() == [TAROT / "tasks.yaml"]
    for name, spec in specs.items():
        module, cls = spec.implemented_by.split(":")
        assert applications.get_task(name).__module__ == module and applications.get_task(name).__name__ == cls


def test_the_reference_has_a_valid_structure():
    [spec] = load_task_file(REFERENCE, check=False).values()
    assert spec.name == "task_name" and spec.prompt.templates[0].id == "example_v1"
    with pytest.raises(TaskFileError, match="example_v1"):           # its names are placeholders
        load_task_file(REFERENCE)


# ---- the three levels of instantiation ---------------------------------------------------------
def test_a_direct_prompt_needs_nothing_else(tmp_path):
    [spec] = load_task_file(write(tmp_path, BASE)).values()
    assert spec.instantiation is None and spec.prompt.templates[0].id == "t_inline_1_v1"
    assert spec.prompt.templates[0].template.required == () and spec.execution.n is None


def test_a_template_with_fixed_variables(tmp_path):
    draw = {"Present": "Death", "Outcome": "The Sun reversed"}
    text = BASE.replace('      - inline: "Say hello."',
                        f"      - template: celtic_cross_v1\n        variables: {{cards: {draw}}}")
    [spec] = load_task_file(write(tmp_path, text)).values()
    assert spec.prompt.templates[0].variables == {"cards": draw}


def test_an_instantiation_file_is_read_checked_and_hashed(tmp_path):
    shutil.copy(ROOT / "applications" / "template" / "instantiation.yaml", tmp_path / "instantiation.yaml")
    text = BASE.replace('      - inline: "Say hello."',
                        '      - inline: "A {{ tone }} story about a {{ character }} in the {{ setting }}."\n'
                        '        name: story') + "  instantiation: instantiation.yaml\n"
    [spec] = load_task_file(write(tmp_path, text)).values()
    state = spec.instantiation.state
    assert isinstance(state["tone"], Literal) and state["themes"] == Literal(["friendship", "curiosity"])
    assert state["setting"] == OneOf(("forest", "village", "mountain"), "cycle")
    assert isinstance(state["narrator"], Personas) and isinstance(state["arc"], Steps)
    assert dict(state["arc"].fixed) == {"end": "discovery"} and state["arc"].replacement is False
    assert spec.prompt.templates[0].id == "story_v1" and set(spec.content["instantiation"]["state"]) == set(state)
    assert "tarot_personas" in spec.references["persona_sets"]

    before = spec.sha256
    (tmp_path / "instantiation.yaml").write_text(
        (tmp_path / "instantiation.yaml").read_text(encoding="utf-8").replace("whimsical", "solemn"), encoding="utf-8")
    assert load_task_file(tmp_path / "tasks.yaml")["t"].sha256 != before       # the file is part of the task


# ---- what the hash covers -------------------------------------------------------------------------
def test_the_hash_ignores_comments_order_and_place_but_not_content(tmp_path):
    shutil.copy(DECK, tmp_path / DECK.name)
    original = (TAROT / "tasks.yaml").read_text(encoding="utf-8")
    [spec] = load_task_file(TAROT / "tasks.yaml").values()

    data = yaml.safe_load(original)
    data["tarot_celtic_cross"] = dict(reversed(list(data["tarot_celtic_cross"].items())))   # keys reordered
    data["tarot_celtic_cross"]["created_ts"] = "2026-10-19"
    elsewhere = tmp_path / "tasks.yaml"
    elsewhere.write_text("# another comment\n" + yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    assert load_task_file(elsewhere)["tarot_celtic_cross"].sha256 == spec.sha256

    data["tarot_celtic_cross"]["execution"]["n"] = 20
    elsewhere.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    assert load_task_file(elsewhere)["tarot_celtic_cross"].sha256 != spec.sha256


# ---- problems ---------------------------------------------------------------------------------------
def tarot_task(**changes) -> str:
    """The tarot task with some keys replaced: {dotted.path: value}; value None deletes the key."""
    task = read_yaml(TAROT / "tasks.yaml")["tarot_celtic_cross"]
    task["instantiation"]["state"]["cards"]["steps"]["from"] = str(DECK)         # the copy lives elsewhere
    for path, value in changes.items():
        *parents, last = path.split(".")
        node = task
        for key in parents:
            node = node[key]
        if value is None:
            node.pop(last)
        else:
            node[last] = value
    return yaml.safe_dump({"tarot_celtic_cross": task}, sort_keys=False)


@pytest.mark.parametrize("changes, expected", [
    ({"model.max_token": 10}, "model: Additional properties are not allowed ('max_token' was unexpected)"),
    ({"prompt.templates": [{"template": "celtic_cross"}]}, "'celtic_cross' is not a prompt template's id"),
    ({"prompt.templates": [{"template": "celtic_cross_v9"}]}, "No prompt template 'celtic_cross_v9'"),
    ({"prompt.templates": [{"template": "celtic_cross_v2", "personas": "tarot_personas"}]}, "('personas' was unexpected)"),
    ({"prompt.templates": [{"template": "celtic_cross_v4", "bind": {"cards": "spread"}}]}, "no state entry 'spread'"),
    ({"prompt.templates": [{"template": "celtic_cross_v4", "variables": {"mood": "calm"}}]}, "takes no variable 'mood'"),
    ({"prompt.templates": [{"inline": "Read {{ cards | kv_lines }"}]}, "syntax error"),
    ({"instantiation": None}, "celtic_cross_v4 needs 'cards', which neither the instantiation nor `variables`"),
    ({"validation.expected": "draw"}, "validation.expected: no state entry 'draw'"),
    ({"validation.custom_validator": "applications.tarot.code.validation:no_such_validator"}, "cannot import"),
    ({"implemented_by": "applications.tarot.code.celtic_cross_evaluation:CelticCrossEvaluation"},
     "is the task 'celtic_cross_evaluation', not 'tarot_celtic_cross'"),
    ({"model.provider": "anthropic"}, "'anthropic' is not a registered provider"),
    ({"created_ts": "10-19-2026"}, "'10-19-2026' is not ISO 8601"),
    ({"instantiation.state.cards.steps.fixed": {"Elsewhere": "Death"}}, "'Elsewhere' is not one of the steps"),
    ({"instantiation.state.cards.steps.fixed": {"Outcome": "Deth"}}, "'Deth' is not one of the states"),
    ({"instantiation.state.cards.steps.negative_suffix": None}, "negative_suffix must say how one is written"),
    ({"instantiation.state.cards.one_of": ["a"]}, "instantiation.state.cards: Additional properties are not allowed "
                                                  "('steps' was unexpected)"),
    ({"instantiation.state.cards.steps.names": ["a"]}, "cards.steps: 'from' cannot be combined with 'names'"),
    ({"prompt.templates": [{"template": "celtic_cross_v4", "inline": "x"}]}, "'template' cannot be combined with 'inline'"),
    ({"execution.mode": "nightly"}, "'nightly' is not one of ['live', 'batch', 'dry_run']"),
    ({"output": {"format": "text"}}, "Additional properties are not allowed ('output' was unexpected)"),
    ({"instantiation": None, "prompt.templates": [{"template": "celtic_cross_v4", "variables": {"cards": None}}]},
     "celtic_cross_v4 does not render with its fixed variables: missing required variables ['cards']"),
])
def test_problems_are_found_and_located(tmp_path, changes, expected):
    found = problems(tmp_path, tarot_task(**changes))
    assert any(expected in p for p in found), found
    assert all(p.startswith("tarot_celtic_cross") for p in found)


# ---- every template renders, with every persona ------------------------------------------------------
def with_personas(people: str, template: str = "celtic_cross_v2") -> str:
    return tarot_task(**{"prompt.templates": [{"template": template}],
                         "instantiation.state.persona": {"personas": people}})


def test_every_persona_of_a_set_is_rendered(tmp_path):
    [spec] = load_task_file(write(tmp_path, with_personas("tarot_personas"))).values()
    assert isinstance(spec.instantiation.state["persona"], Personas)

    people = json.loads((ROOT / "personas" / "tarot_personas.json").read_text(encoding="utf-8"))[:3]
    del people[1]["tarot_profile"]                                       # a field celtic_cross_v2 reads
    (tmp_path / "people.json").write_text(json.dumps(people), encoding="utf-8")
    [found] = problems(tmp_path, with_personas(str(tmp_path / "people.json")))
    assert found.startswith("tarot_celtic_cross.prompt.templates[0]: celtic_cross_v2 does not render in 1 of 3 "
                            "sample instants, e.g. with persona = persona 2 (")
    assert "tarot_profile" in found

    # a template that does not read personas is not affected by what they hold
    load_task_file(write(tmp_path, with_personas(str(tmp_path / "people.json"), "celtic_cross_v4")))


def test_every_problem_is_reported_at_once(tmp_path):
    found = problems(tmp_path, tarot_task(**{"model.provider": "anthropic", "validation.expected": "draw",
                                             "created_ts": "yesterday"}))
    assert len(found) == 1 and "ISO 8601" in found[0]                 # the structure is checked first
    found = problems(tmp_path, tarot_task(**{"model.provider": "anthropic", "validation.expected": "draw"}))
    assert len(found) == 2


def test_task_names_and_duplicate_keys(tmp_path):
    assert "lowercase letters" in problems(tmp_path, BASE.replace("\nt:", "\nTarot:"))[0]
    duplicated = BASE.replace("  engine: inference\n", "  engine: inference\n  engine: inference\n")
    assert "duplicate key 'engine'" in problems(tmp_path, duplicated)[0]


def test_dates_stay_as_written(tmp_path):
    assert read_yaml(write(tmp_path, "when: 2026-10-19\nat: 2026-10-19T22:00:00Z\n", "d.yaml")) == \
        {"when": "2026-10-19", "at": "2026-10-19T22:00:00Z"}


# ---- command line -----------------------------------------------------------------------------------
def test_the_check_command(tmp_path, capsys):
    assert main(["check"]) == 0
    out = capsys.readouterr().out
    assert "tarot_celtic_cross  OK" in out and "celtic_cross_v4  prompts/library/tarot/celtic_cross_v4.toml" in out
    assert main(["check", "--structure-only", str(REFERENCE)]) == 0
    bad = write(tmp_path, tarot_task(**{"model.provider": "anthropic"}))
    assert main(["check", str(bad)]) == 1
    assert "PROBLEM  tarot_celtic_cross.model.provider" in capsys.readouterr().out


def test_prompts_show_survives_a_legacy_console(monkeypatch):
    from prompts.__main__ import main as prompts
    console = io.TextIOWrapper(io.BytesIO(), encoding="cp1252")        # a Windows console that has no "↔"
    monkeypatch.setattr(sys, "stdout", console)
    assert prompts(["show", "celtic_cross_v4"]) == 0
