"""inference.task_config: InferenceTaskConfig, from task files, entries, settings and runs; the problems reported."""
import io
import json
import shutil
import sys
import textwrap
from pathlib import Path

import pytest
import yaml

import applications
from inference.pipeline import GenerationConfig
from inference.task_config import InferenceTaskConfig, TaskConfigError, edited, load_task_file, main, parse_edit
from instantiation import Literal, OneOf, Personas, Steps
from spec_files import read_yaml
from prompts import get_template

ROOT = Path(__file__).parents[1]
TAROT = ROOT / "applications" / "tarot"
DECK = TAROT / "vector_space_config.json"
POSITIONS = json.loads(DECK.read_text(encoding="utf-8"))["step_aliases"]
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
    with pytest.raises(TaskConfigError) as caught:
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

    cards, deck = spec.instantiation.state["cards"], json.loads(DECK.read_text(encoding="utf-8"))
    assert isinstance(cards, Steps) and cards.file == DECK
    assert cards.names == tuple(deck["step_aliases"]) and cards.states == tuple(deck["aliases"])
    assert (cards.negative_polarity, cards.negative_suffix, cards.replacement, dict(cards.fixed)) == \
        (True, " reversed", True, {})

    assert (spec.model.provider, spec.model.name, dict(spec.model.parameters)) == \
        ("openai", "gpt-6-luna", {"max_tokens": 3600})
    assert (spec.execution.mode, spec.execution.workers, spec.execution.n) == ("live", 2, 1)
    assert spec.execution.seed is None and spec.execution.retry_mode == "live"         # not given: the default
    assert spec.prompt.system is None
    assert spec.validation.custom_validator == "applications.tarot.code.validation:celtic_cross_validator"
    assert (spec.validation.expected, list(spec.validation.checks), spec.validation.retries) == \
        ("cards", ["non_empty", "not_truncated", "no_refusal"], 2)
    assert spec.references["templates"] == {"celtic_cross_v4": template.template.sha256}
    assert set(spec.references["files"]) == {"vector_space_config.json"} and len(spec.sha256) == 64


def test_task_files_and_task_classes_agree():
    specs = applications.task_configs()
    assert set(specs) == {"tarot_celtic_cross"} and applications.task_files() == [TAROT / "tasks.yaml"]
    for name, spec in specs.items():
        module, cls = spec.implemented_by.split(":")
        assert applications.get_task(name).__module__ == module and applications.get_task(name).__name__ == cls


def test_the_reference_has_a_valid_structure():
    [spec] = load_task_file(REFERENCE, check=False).values()
    assert spec.name == "task_name" and spec.prompt.templates[0].id == "example_v1"
    with pytest.raises(TaskConfigError, match="example_v1"):           # its names are placeholders
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
    ({"validation.checks": {"json_fields": {"schema": "reading"}}},
     "Additional properties are not allowed ('json_fields' was unexpected)"),
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
    [found] = problems(tmp_path, with_personas("people.json"))          # relative to the task file
    assert found.startswith("tarot_celtic_cross.prompt.templates[0]: celtic_cross_v2 does not render in 1 of 3 "
                            "sample instants, e.g. with persona = persona 2 (")
    assert "tarot_profile" in found

    # a template that does not read personas is not affected by what they hold
    load_task_file(write(tmp_path, with_personas("people.json", "celtic_cross_v4")))


def test_a_persona_set_path_is_relative_to_the_task_file(tmp_path, monkeypatch):
    shutil.copy(ROOT / "personas" / "tarot_personas.json", tmp_path / "people.json")
    (tmp_path / "elsewhere").mkdir()
    monkeypatch.chdir(tmp_path / "elsewhere")                            # not where the task file is
    [spec] = load_task_file(write(tmp_path, with_personas("people.json"))).values()
    assert spec.instantiation.state["persona"].file == tmp_path / "people.json"
    assert spec.references["persona_sets"] == {"people.json": spec.references["persona_sets"]["people.json"]}
    assert "persona.personas: no file 'missing.json'" in problems(tmp_path, with_personas("missing.json"))[0]


# ---- given instants -------------------------------------------------------------------------------------
def draw(**changes) -> dict:
    """A Celtic Cross draw: Death everywhere, but for `changes` ({position: card}, spaces as underscores)."""
    return {**{p: "Death" for p in POSITIONS}, **{k.replace("_", " "): v for k, v in changes.items()}}


def with_instants(instants, state: bool = True) -> str:
    task = yaml.safe_load(with_personas("tarot_personas"))
    if not state:
        del task["tarot_celtic_cross"]["instantiation"]["state"]
    if instants is not None:
        task["tarot_celtic_cross"]["instantiation"]["instants"] = instants
    return yaml.safe_dump(task, sort_keys=False)


def test_given_instants_are_checked_and_filled_in(tmp_path):
    given = [{"cards": draw(), "persona": 2}, {"cards": draw(Present="The Sun reversed"), "persona": "Maya Delgado"}]
    [spec] = load_task_file(write(tmp_path, with_instants(given))).values()
    assert spec.instantiation.instants == tuple(given)
    out = io.StringIO()
    sys.stdout, saved = out, sys.stdout
    try:
        main(["check", str(tmp_path / "tasks.yaml")])
    finally:
        sys.stdout = saved
    assert "instants        2 given (inline), checked against the state" in out.getvalue()

    found = problems(tmp_path, with_instants([{"cards": draw(Present="Deth"), "persona": 900},
                                              {"cards": draw(), "mood": "calm"}]))
    assert found == [
        "tarot_celtic_cross.instantiation.instants[0].cards.Present: 'Deth' is not one of the states",
        "tarot_celtic_cross.instantiation.instants[0].persona: No persona 900 in set 'tarot_personas'",
        "tarot_celtic_cross.instantiation.instants[1].mood: not an entry of the state (cards, persona)",
        "tarot_celtic_cross.instantiation.instants[1].persona: missing"]


def test_given_instants_from_a_jsonl_file(tmp_path):
    lines = [json.dumps({"cards": draw(Outcome=card), "persona": 1}) for card in ("The Sun", "The Moon", "Death")]
    (tmp_path / "draws.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    [spec] = load_task_file(write(tmp_path, with_instants("draws.jsonl"))).values()
    assert [i["cards"]["Outcome"] for i in spec.instantiation.instants] == ["The Sun", "The Moon", "Death"]
    assert "draws.jsonl" in spec.references["files"]
    before = spec.sha256
    (tmp_path / "draws.jsonl").write_text("\n".join(lines[:2]), encoding="utf-8")
    assert load_task_file(tmp_path / "tasks.yaml")["tarot_celtic_cross"].sha256 != before   # the file is part of it

    wrong = [json.dumps({"cards": draw(Outcome="Deth"), "persona": 1})] * 12
    (tmp_path / "draws.jsonl").write_text("\n".join(wrong), encoding="utf-8")
    found = problems(tmp_path, with_instants("draws.jsonl"))
    assert found[0] == ("tarot_celtic_cross.instantiation.instants (draws.jsonl)[0].cards.Outcome: "
                        "'Deth' is not one of the states")
    assert len(found) == 11 and found[-1].endswith("and 2 more instants that are not possible states")


def test_given_instants_without_a_state(tmp_path):
    [spec] = load_task_file(write(tmp_path, with_instants([{"cards": draw()}], state=False))).values()
    assert spec.instantiation.state == {} and spec.instantiation.entries == ("cards",)   # expected: cards, still
    assert "instantiation: needs 'state' or 'instants'" in problems(tmp_path, with_instants(None, state=False))[0]


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


# ---- one task, edited ---------------------------------------------------------------------------------------
def test_one_task_with_edits(tmp_path):
    spec = InferenceTaskConfig.from_file(TAROT / "tasks.yaml", "tarot_celtic_cross", {"execution.n": 20, "model.parameters.temperature": 0.5,
                                                                   "instantiation.state.cards.steps.fixed": {"Outcome": "Death"}})
    assert spec.execution.n == 20 and dict(spec.model.parameters) == {"max_tokens": 3600, "temperature": 0.5}
    assert dict(spec.instantiation.state["cards"].fixed) == {"Outcome": "Death"}
    assert spec.edits["execution.n"] == 20 and spec.content["execution"]["n"] == 20      # recorded, as edited
    plain = InferenceTaskConfig.from_file(TAROT / "tasks.yaml")                                             # the file's only task
    assert plain.sha256 == load_task_file(TAROT / "tasks.yaml")["tarot_celtic_cross"].sha256 != spec.sha256
    assert plain.edits == {} and plain.instantiation_sha256 != spec.instantiation_sha256
    assert InferenceTaskConfig.from_file(TAROT / "tasks.yaml", None, {"execution.n": 3}).instantiation_sha256 == plain.instantiation_sha256
    with pytest.raises(TaskConfigError, match="no task 'tarot' in the file; it has tarot_celtic_cross"):
        InferenceTaskConfig.from_file(TAROT / "tasks.yaml", "tarot")
    with pytest.raises(TaskConfigError, match="no such file"):
        InferenceTaskConfig.from_file(tmp_path / "nope.yaml")


def test_edits_reach_into_an_instantiation_file(tmp_path):
    shutil.copy(ROOT / "applications" / "template" / "instantiation.yaml", tmp_path / "instantiation.yaml")
    task = {"instantiation": "instantiation.yaml", "prompt": {"templates": [{"inline": "x"}]}}
    out = edited(task, {"instantiation.state.tone": "grim", "prompt.sample": "cycle"}, tmp_path)
    assert out["instantiation"]["state"]["tone"] == "grim" and "character" in out["instantiation"]["state"]
    assert out["prompt"] == {"templates": [{"inline": "x"}], "sample": "cycle"} and task["instantiation"] == "instantiation.yaml"
    assert edited({}, {"a.b.c": 1}, tmp_path) == {"a": {"b": {"c": 1}}}
    with pytest.raises(ValueError, match="cannot set prompt.templates.x: prompt.templates is"):
        edited(task, {"prompt.templates.x": 1}, tmp_path)


@pytest.mark.parametrize("text, edit", [
    ("execution.n=20", ("execution.n", 20)),
    ("model.name=null", ("model.name", None)),
    ("instantiation.state.cards.steps.replacement=false", ("instantiation.state.cards.steps.replacement", False)),
    ("instantiation.state.cards.steps.fixed={Hopes and Fears: The Sun reversed}",
     ("instantiation.state.cards.steps.fixed", {"Hopes and Fears": "The Sun reversed"})),
    ("prompt.templates=[{template: celtic_cross_v2}]", ("prompt.templates", [{"template": "celtic_cross_v2"}])),
    ("execution.name=a=b", ("execution.name", "a=b")),
])
def test_command_line_edits(text, edit):
    assert parse_edit(text) == edit


def test_a_bad_command_line_edit():
    with pytest.raises(ValueError, match="dotted.path=value"):
        parse_edit("execution.n")


def test_a_run_s_record_gives_back_its_config(tmp_path):
    spec = InferenceTaskConfig.from_file(TAROT / "tasks.yaml", None, {"execution.n": 2})
    record = spec.record()
    assert record["source"] == "applications/tarot/tasks.yaml" and record["edits"] == {"execution.n": 2}
    again = InferenceTaskConfig.from_record(json.loads(json.dumps(record)))    # as it is stored, in JSON
    assert (again.name, again.sha256, again.instantiation_sha256) == (spec.name, spec.sha256, spec.instantiation_sha256)
    assert (again.validation, again.execution, again.model) == (spec.validation, spec.execution, spec.model)
    assert again.implemented_by == spec.implemented_by and again.edits == spec.edits
    assert again.instantiation is None and again.content == spec.content    # its files are not read again
    assert InferenceTaskConfig.from_run({"task": spec.name, "config": {"task_config": record, "seed": 7}}).sha256 == spec.sha256


def test_a_config_written_in_code(tmp_path):
    entry = {"description": "A test task.", "engine": "inference", "model": {"provider": "openai"},
             "prompt": {"templates": [{"inline": "Say {{ word }}."}], "system": "Be brief."},
             "instantiation": {"state": {"word": {"one_of": ["hello", "goodbye"]}}}, "validation": None}
    config = InferenceTaskConfig.from_entry("t", entry, edits={"execution.n": 3})
    assert (config.prompt.system, config.validation, config.execution.n, config.source) == ("Be brief.", None, 3, None)
    assert config.prompt.templates[0].template.required == ("word",) and config.edits == {"execution.n": 3}
    with pytest.raises(TaskConfigError, match=r"t \(in code\):\n  t.model: 'provider' is a required property"):
        InferenceTaskConfig.from_entry("t", {**entry, "model": {}})


@pytest.mark.parametrize("version", [1, 4, 5])
def test_settings_of_runs_before_task_configs(version):
    """Runs before task configs stored flat settings, in the shape of GenerationConfig (version 1: one template)."""
    settings = {**vars(GenerationConfig(n=50, provider="featherless", model="m", templates=["celtic_cross_v4"],
                                        params={"max_tokens": 2400}, seed=42, max_workers=2, validation_retries=1,
                                        dataset="tarot-v4", persona_set="tarot_personas", persona_ids=["7"])),
                "task_options": {"draw": "replacement"}, "persona_set_sha256": "ab"}
    if version == 1:
        settings = {k: v for k, v in settings.items() if k not in ("templates", "persona_set", "persona_ids")}
        settings["template"] = "celtic_cross_v1"
    config = InferenceTaskConfig.from_run({"task": "tarot_celtic_cross", "config": settings})
    assert (config.model.provider, config.model.name, dict(config.model.parameters)) == ("featherless", "m", {"max_tokens": 2400})
    x = config.execution
    assert (x.n, x.seed, x.workers, x.dataset, x.mode, config.validation.retries) == (50, 42, 2, "tarot-v4", "live", 1)
    assert [t.id for t in config.prompt.templates] == (["celtic_cross_v1"] if version == 1 else ["celtic_cross_v4"])
    persona = config.instantiation.state["persona"] if version > 1 else None
    assert version == 1 and config.instantiation is None or (persona.set, persona.select) == ("tarot_personas", ("7",))
    assert InferenceTaskConfig.from_run({"task": "t", "config": {**settings, "validate": False}}).validation is None


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
