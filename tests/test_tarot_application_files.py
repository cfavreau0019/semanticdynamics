"""
applications/tarot/*.yaml: the application's files must say what the code does.

Until the task registry reads these files, the code reads the old places (prompts/library,
evaluation/rubrics, the task classes). These tests keep the two from drifting apart: the
copied prompts and rubric have the same content hashes, every reference between the files
resolves, the schemas accept real data, and every default and setting the files state is the
one the code uses.
"""
import hashlib
import importlib
import json
import re
from pathlib import Path

import pytest
import yaml
from jsonschema import Draft202012Validator

import applications
from applications.tarot.code import validation as tarot_validation
from applications.tarot.code.celtic_cross_evaluation import CelticCrossEvaluation
from applications.tarot.code.persona_expectations import TarotPersonaExpectations
from applications.tarot.code.readings import TarotReadings
from evaluation.expectations import EXPECTATION_FIELDS
from evaluation.rubric import get_rubric
from evaluation.subjects import load_subjects
from inference.task_config import InferenceTaskConfig, load_task_file
from instantiation import Instantiation
from personas import PersonaSet
from prompts import PromptTemplate, get_template
from test_evaluation import readings, registry  # noqa: F401

APP = Path(applications.__file__).parent / "tarot"


def load(name: str):
    return yaml.safe_load((APP / name).read_text(encoding="utf-8"))


DECK = json.loads((APP / "vector_space_config.json").read_text(encoding="utf-8"))
SCHEMAS, TASKS, WORKFLOWS, PERSONAS, APPLICATION = (
    load("schemas.yaml"), load("tasks_draft.yaml")["tasks"], load("workflows.yaml")["workflows"], load("personas.yaml"),
    load("application.yaml"))
PROMPTS = {p.stem: yaml.safe_load(p.read_text(encoding="utf-8")) for p in sorted((APP / "prompts").glob("*.yaml"))}
RUBRICS = {p.stem: p for p in sorted((APP / "rubrics").glob("*.yaml"))}


def schema(name: str) -> Draft202012Validator:
    root = {**SCHEMAS, "$schema": "https://json-schema.org/draft/2020-12/schema", "$ref": f"#/schemas/{name}"}
    Draft202012Validator.check_schema(root)
    return Draft202012Validator(root)


# ---- the copied prompts and rubric are the ones the code uses ---------------------------------------
@pytest.mark.parametrize("template_id", sorted(PROMPTS))
def test_prompt_files_match_the_library(template_id):
    d = PROMPTS[template_id]
    variables = d["variables"]
    t = PromptTemplate.from_dict({
        "name": d["name"], "version": d["version"], "description": d["description"], "changes": d["changes"],
        "based_on": d.get("based_on"), "tags": d["tags"], "template": d["template"],
        "required": [v for v, spec in variables.items() if spec["required"]],
        "optional": [v for v, spec in variables.items() if not spec["required"]]})
    live = get_template(template_id)
    assert t.id == template_id and t.sha256 == live.sha256          # what stored prompts reference
    assert (t.required, t.optional, t.description, t.changes, t.based_on, t.tags) == \
        (live.required, live.optional, live.description, live.changes, live.based_on, live.tags)
    assert all(spec.get("schema") in (None, *SCHEMAS["schemas"]) for spec in variables.values())


def test_rubric_file_has_the_stored_hash():
    [(rubric_id, path)] = RUBRICS.items()
    content = yaml.safe_load(path.read_text(encoding="utf-8"))
    sha = hashlib.sha256(json.dumps(content, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    assert sha == get_rubric(rubric_id).sha256                       # what evaluations store as rubric_sha256


# ---- the files refer to each other correctly ------------------------------------------------------------
def test_every_reference_between_the_files_resolves():
    assert APPLICATION["name"] == APP.name
    for name, task in TASKS.items():
        prompt = task["prompt"]
        templates = [t for t in [prompt["template"], prompt.get("with_personas"), *prompt.get("alternatives", [])] if t]
        assert set(templates) <= set(PROMPTS), (name, templates)
        needed = set().union(*(PROMPTS[t]["variables"] for t in templates))
        assert needed <= set(prompt["variables"]), name                 # the task offers what its templates take
        assert {task["input"]["schema"], task["output"]["schema"]} <= set(SCHEMAS["schemas"]), name
        assert task.get("started_by") in (None, *TASKS)
        if "rubric" in task:
            assert task["rubric"]["id"] in RUBRICS
    for name, workflow in WORKFLOWS.items():
        for step in workflow["steps"]:
            assert step.get("task") in (None, *TASKS) and step.get("workflow") in (None, *WORKFLOWS), (name, step)
            assert set(step.get("needs", [])) <= {s["id"] for s in workflow["steps"]}, (name, step)
    assert set(PERSONAS["disclosure"]) <= set(PROMPTS)
    persona_fields = SCHEMAS["schemas"]["persona"]["properties"]
    for path in {p for fields in PERSONAS["disclosure"].values() for p in fields} | set(PERSONAS["sensitive"]):
        node = {"properties": persona_fields}
        for key in path.split("."):
            node = node["properties"][key]                              # KeyError: a field the schema lacks


# ---- the schemas accept what the code produces -----------------------------------------------------------
def test_draws_and_personas_fit_their_schemas():
    draws = schema("celtic_cross_draw")
    for replacement in (True, False):
        spec = InferenceTaskConfig.from_file(APP / "tasks.yaml", "tarot_celtic_cross",
                                             {"instantiation.state.cards.steps.replacement": replacement})
        for one in Instantiation.sample(spec.instantiation, 100, seed=1):
            draw = one.instant["cards"]
            draws.validate(draw)
    assert not draws.is_valid({**draw, "Elsewhere": "Death"}) and not draws.is_valid({"Present": "Death"})
    people = schema("persona")
    for person in PersonaSet.load(PERSONAS["set"]["name"]):
        people.validate(person)
    schema("expectation_context").validate(TarotPersonaExpectations().context)
    answer = {"situation_summary": "Busy.", "what_they_hoped_for": "Insight", "framing_preference": "reflective",
              "must_haves": [], "sensitivities": ["grief"]}
    schema("persona_expectations").validate(answer)


def test_evaluation_inputs_fit_their_schemas(readings):  # noqa: F811
    root, store = readings
    task = CelticCrossEvaluation(subjects_root=root, subject_runs=[store.run_id])
    for subject in load_subjects(root, runs=[store.run_id]):
        payload = task._payload(subject)
        schema("evaluation_subject").validate(payload)
        schema("evaluation_context").validate(task.context(payload))


# ---- what the files say the code does, it does --------------------------------------------------------------
def test_tasks_are_the_registered_ones_with_their_stated_settings():
    assert set(TASKS) == set(applications.tasks())
    for name, spec in TASKS.items():
        module, cls_name = spec["implemented_by"].split(":")
        cls = getattr(importlib.import_module(module), cls_name)
        assert cls is applications.get_task(name)
        if name == "tarot_celtic_cross":            # tasks.yaml says how it runs now: test_task_file_agrees_with_the_code
            continue
        assert cls.default_template == spec["prompt"]["template"]
        assert cls.persona_template == spec["prompt"].get("with_personas", spec["prompt"]["template"])
        assert cls.input_type == spec["input"]["input_type"]
    assert TarotReadings.from_options({}).validator().name == TASKS["tarot_celtic_cross"]["output"]["validation"]["validator"]

    expectations = TASKS["persona_expectations"]
    assert TarotPersonaExpectations.default_context == expectations["input"]["source"]["value"] == \
        expectations["options"]["context"]["default"]
    assert TarotPersonaExpectations().validator().name == expectations["output"]["validation"]["validator"]

    evaluation, task = TASKS["celtic_cross_evaluation"], CelticCrossEvaluation()
    assert set(task.options()) == set(evaluation["options"])
    assert CelticCrossEvaluation.rubric_ref == evaluation["rubric"]["id"] == evaluation["options"]["rubric"]["default"]
    assert CelticCrossEvaluation.default_personas == evaluation["personas"]["set"] == PERSONAS["set"]["name"]
    assert list(CelticCrossEvaluation.default_subject_templates) == evaluation["options"]["subject_templates"]["default"]
    assert CelticCrossEvaluation.expectations_task.name == evaluation["personas"]["expectations"].split()[2] == \
        "persona_expectations"
    assert task.validator().name == evaluation["output"]["validation"]["validator"]
    assert set(evaluation["rubric"]["na_rules"]) == {"1.2", "7.4"} == {i.id for i in task.rubric.items if i.allow_na}


def test_vocabulary_is_the_one_the_code_uses():
    vocabulary = SCHEMAS["vocabulary"]
    positions = {p["name"]: [p["name"], *p["also"]] for p in vocabulary["positions"]}
    assert positions == tarot_validation.CELTIC_CROSS_POSITIONS and list(positions) == DECK["step_aliases"]
    cards = vocabulary["cards"]
    minor = [cards["minor_arcana"]["name"].format(rank=r, suit=s)
             for s in cards["minor_arcana"]["suits"] for r in cards["minor_arcana"]["ranks"]]
    assert cards["major_arcana"] + minor == tarot_validation.TAROT_CARDS
    assert cards["order"] == "alphabetical" and DECK["aliases"] == sorted(tarot_validation.TAROT_CARDS)
    in_text = vocabulary["orientations"]["in_text"]
    assert in_text["reversed"] == tarot_validation._REVERSED.pattern
    assert in_text["not_reversed"] == tarot_validation._NEGATED_REVERSED.pattern
    assert in_text["reversed_before_name"] in Path(tarot_validation.__file__).read_text(encoding="utf-8")
    assert re.search(in_text["reversed"], "Death (R)", re.IGNORECASE)


def test_stated_answer_fields_and_persona_set_are_the_code_s():
    fields = SCHEMAS["schemas"]["persona_expectations"]["properties"]
    stated = {name: (f["enum"] if "enum" in f else "list" if f.get("type") == "array" else "text")
              for name, f in fields.items()}
    assert stated == EXPECTATION_FIELDS
    people = PersonaSet.load(PERSONAS["set"]["name"])
    assert (people.sha256, len(people)) == (PERSONAS["set"]["sha256"], PERSONAS["set"]["records"])


def test_task_file_agrees_with_the_code():
    """tasks.yaml (the new format) is checked by inference.task_config; here, that what it says matches the
    deck, what runs before the task file did (any card, repeats allowed, none pinned), and the draft's template."""
    from applications.tarot.code.validation import celtic_cross_validator

    [spec] = load_task_file(APP / "tasks.yaml").values()
    assert [t.id for t in spec.prompt.templates] == [TASKS["tarot_celtic_cross"]["prompt"]["template"]]
    assert applications.get_task(spec.name) is TarotReadings and spec.implemented_by.endswith(":TarotReadings")
    templates = {t.id for t in spec.prompt.templates} | set(spec.prompt.retired_templates)
    assert templates <= set(PROMPTS)                                                # each has its copy here
    cards = spec.instantiation.state["cards"]
    assert cards.file == APP / "vector_space_config.json"
    assert (cards.names, cards.states) == (tuple(DECK["step_aliases"]), tuple(DECK["aliases"]))
    assert cards.replacement and dict(cards.fixed) == {} and DECK["negative_polarity"]
    assert cards.negative_suffix == tarot_validation.REVERSED_SUFFIX                 # how a reversed card is written
    module, function = spec.validation.custom_validator.split(":")
    assert getattr(importlib.import_module(module), function) is celtic_cross_validator
