"""evaluation: rubric scoring, answer checks, subjects, the two-stage pipeline, and the CLI."""
import json
import re
import threading

import pytest

import llm.providers as providers
from data_generation import schema
from data_generation.pipeline import GenerationConfig, load_config, prepare_run, rebuild_requests, run_live
from data_generation.store import RunStore
from data_generation.tarot import TarotReadings
from evaluation.celtic_cross import EXPECTATION_CONTEXT, CelticCrossEvaluation
from evaluation.checks import JsonFieldsCheck, RubricAnswerCheck, extract_json, quote_in_text
from evaluation.evaluate_tarot import main
from evaluation.expectations import EXPECTATION_FIELDS, generate_expectations, load_expectations
from evaluation.rubric import RubricError, get_rubric, load_rubric
from evaluation.subjects import load_subjects
from llm import ChatResponse, Usage
from llm.providers import LLMProvider, register_provider
from personas import PersonaSet
from test_data_generation import TarotScripted, check_primary_keys
from validation import Validator

POSITIONS = ["Present", "Challenge", "Conscious Goal", "Root Cause/Foundation", "Recent Past", "Near Future",
             "Your Attitude", "External Influences", "Hopes and Fears", "Outcome"]
LABELS = {"positions": POSITIONS}
RUBRIC = get_rubric("celtic_cross_reading_v1")

# The worked example in the Celtic Cross Reading Evaluation Form (persona #100): 70.87, a C.
WORKED_EXAMPLE = {
    "1.1": 5, "1.2": 5, "1.3": 3, "1.4": 4, "1.5": 5,
    "2a": [5, 4, 5, 4, 5, 4, 4, 4, 5, 4], "2b": [4, 4, 4, 4, 4, 3, 3, 3, 4, 3],
    "3.1": 2, "3.2": 1, "3.3": 3, "3.4": 4, "3.5": 3, "3.6": 3,
    "4.1": 2, "4.2": 1, "4.3": 2, "4.4": 2,
    "5.1": 4, "5.2": 5, "5.3": 4, "5.4": 3,
    "6.1": 5, "6.2": 5, "6.3": 5, "6.4": 4,
    "7.1": 5, "7.2": 4, "7.3": 4, "7.4": 2,
}
OUTPUTS = {"top_strengths": [{"item": "6.2", "note": "hard cards handled honestly"}],
           "top_improvements": [{"item": "4.2", "fix": "ask for a question"}],
           "missed_patterns": ["three Swords form the spine"],
           "persona_reaction": "It's kind and well written. I wouldn't screenshot it.",
           "would_use_again": 6, "felt_seen": 2, "emotional_after_state": "calmer"}


def answer(scores=None, flags=(), outputs=None, justify=True, **overrides):
    item_scores = {**(scores or WORKED_EXAMPLE), **overrides}
    justifications = {}
    if justify:
        for key, value in item_scores.items():
            values = value if isinstance(value, list) else [value]
            for n, v in enumerate(values, 1):
                if v in (1, 2, 5):
                    justifications[f"{key}.{n}" if isinstance(value, list) else key] = "because"
    return {"item_scores": item_scores, "justifications": justifications, "red_flags": list(flags),
            "outputs": dict(outputs or OUTPUTS)}


def read(data, **kw):
    return RUBRIC.read(data, LABELS, **kw)


def codes(problems, severity=None):
    return sorted({p.code for p in problems if severity is None or p.severity == severity})


# ---- rubric: structure and scoring ---------------------------------------------------------
def test_rubric_matches_the_form():
    assert RUBRIC.id == "celtic_cross_reading_v1" and len(RUBRIC.sections) == 7
    assert [s.weight for s in RUBRIC.sections] == [15, 15, 20, 15, 10, 15, 10]
    assert len(RUBRIC.items) == 29 and [f.cap for f in RUBRIC.red_flags] == [40, 40, 40, 40, 60, 60, 60, 60]
    types = {i.id: i.type for i in RUBRIC.items}
    assert (types["1.1"], types["2a"], types["2b"], types["3.5"], types["4.1"], types["7.1"]) == \
        ("O", "O", "S", "S", "S", "O")
    assert [i.id for i in RUBRIC.items if i.allow_na] == ["1.2", "7.4"]
    assert RUBRIC.item("2a").per == "positions" and RUBRIC.section_of("2a").key == "position_fidelity"


def test_rubric_content_is_pinned():
    """Scores are only comparable under an identical rubric: change the form -> make a new version file."""
    assert RUBRIC.sha256 == get_rubric("celtic_cross_reading").sha256
    assert RUBRIC.sha256.startswith("5972c90206"), \
        "celtic_cross_reading_v1.toml changed; restore it and put the change in celtic_cross_reading_v2.toml"


def test_worked_example_from_the_form():
    parsed, problems = read(answer())
    assert codes(problems, "error") == []
    result = RUBRIC.score(parsed.scores)
    assert {k: round(v, 2) for k, v in result.section_scores.items()} == {
        "card_accuracy": 4.4, "position_fidelity": 4.0, "synthesis": 2.67, "personalization": 1.75,
        "usefulness": 4.0, "tone_safety": 4.75, "communication_fit": 3.75}
    assert (result.score_before_caps, result.total_score, result.grade, result.cap_applied) == (70.87, 70.9, "C", None)
    assert round(sum(result.section_points.values()), 2) == 70.87


def test_red_flag_caps_lowest_wins_and_only_when_binding():
    scores = read(answer())[0].scores
    assert RUBRIC.score(scores, ["RF5"]).total_score == 60.0 and RUBRIC.score(scores, ["RF5"]).grade == "D"
    both = RUBRIC.score(scores, ["RF5", "RF2"])
    assert (both.total_score, both.cap_applied, both.grade, both.score_before_caps) == (40.0, 40, "F", 70.87)
    low = read(answer(scores={k: ([1] * 10 if isinstance(v, list) else 1) for k, v in WORKED_EXAMPLE.items()}))[0]
    result = RUBRIC.score(low.scores, ["RF5"])
    assert (result.total_score, result.cap_applied, result.grade) == (20.0, None, "F")     # cap not binding


@pytest.mark.parametrize("total, grade", [(100, "A"), (90, "A"), (89.9, "B"), (80, "B"), (70, "C"), (60, "D"),
                                          (59.9, "F"), (0, "F")])
def test_grade_bands(total, grade):
    assert next(g.grade for g in sorted(RUBRIC.grades, key=lambda g: -g.min) if total >= g.min) == grade


def test_na_drops_out_of_the_section_average():
    with_na = read(answer(**{"1.2": "N/A"}))[0]
    assert with_na.scores["1.2"] is None
    assert round(RUBRIC.score(with_na.scores).section_scores["card_accuracy"], 2) == 4.25   # (5+3+4+5)/4


def test_perfect_and_floor_scores():
    top = {k: ([5] * 10 if isinstance(v, list) else 5) for k, v in WORKED_EXAMPLE.items()}
    assert RUBRIC.score(read(answer(scores=top))[0].scores).total_score == 100.0
    bottom = {k: ([1] * 10 if isinstance(v, list) else 1) for k, v in WORKED_EXAMPLE.items()}
    assert RUBRIC.score(read(answer(scores=bottom))[0].scores).total_score == 20.0   # 1/5 on every item


# ---- rubric: reading answers ---------------------------------------------------------------
def test_scores_are_normalised():
    parsed, problems = read(answer(**{"1.1": "4", "1.3": 3.0, "7.4": None, "1.2": "n/a"}))
    assert codes(problems, "error") == []
    assert (parsed.scores["1.1"], parsed.scores["1.3"], parsed.scores["7.4"], parsed.scores["1.2"]) == (4, 3, None, None)
    assert [parsed.scores[f"2a.{n}"] for n in range(1, 11)] == WORKED_EXAMPLE["2a"]


@pytest.mark.parametrize("change, code", [
    ({"3.1": 6}, "out_of_range"), ({"3.1": 0}, "out_of_range"), ({"3.1": 3.5}, "not_a_whole_number"),
    ({"3.1": "good"}, "not_a_whole_number"), ({"3.1": True}, "not_a_whole_number"),
    ({"3.1": "N/A"}, "na_not_allowed"), ({"2a": [5] * 9}, "wrong_length"), ({"2a": 4}, "wrong_length"),
    ({"2b": [4] * 9 + [9]}, "out_of_range"),
])
def test_bad_scores_are_errors(change, code):
    assert code in codes(read(answer(**change))[1], "error")


def test_missing_and_unknown_items():
    data = answer()
    del data["item_scores"]["4.4"]
    data["item_scores"]["9.9"] = 3
    problems = read(data)[1]
    assert "missing_item" in codes(problems, "error") and "unknown_item" in codes(problems, "warning")
    assert read({"nope": 1}) == (None, read({"nope": 1})[1]) and codes(read("text")[1]) == ["bad_structure"]


def test_justifications_required_for_1_2_and_5_as_warnings():
    problems = read(answer(justify=False))[1]
    assert codes(problems) == ["missing_justification"] and codes(problems, "error") == []
    flagged = {p.item for p in problems}
    assert {"3.2", "1.1", "2a.1"} <= flagged and "1.3" not in flagged       # 1, 5 and 5 need one; a 3 doesn't
    data = answer(justify=False)
    data["justifications"] = {"2a": "all positions read well"}               # item-level covers its positions
    assert not any(p.item.startswith("2a.") for p in read(data)[1])


def test_na_decided_by_facts():
    scored, na = answer(), answer(**{"1.2": "N/A"})
    assert "na_not_applicable" in codes(read(na, na_required={"1.2": False})[1], "error")    # reversals exist
    assert codes(read(na, na_required={"1.2": True})[1], "error") == []
    assert "na_expected" in codes(read(scored, na_required={"1.2": True})[1], "warning")     # scored anyway


def test_red_flags_and_outputs():
    ok, problems = read(answer(flags=[{"id": "RF5", "quote": "this will happen"}, {"id": "RF5", "quote": "dup"}]))
    assert ok.red_flags == [{"id": "RF5", "quote": "this will happen"}] and codes(problems, "error") == []
    assert "unknown_red_flag" in codes(read(answer(flags=[{"id": "RF99", "quote": "x"}]))[1], "error")
    assert "red_flag_without_quote" in codes(read(answer(flags=[{"id": "RF1", "quote": " "}]))[1], "error")

    for bad in ({"felt_seen": 6}, {"would_use_again": 11}, {"emotional_after_state": "elated"},
                {"persona_reaction": ""}, {"top_strengths": [{"item": "6.2"}]}, {"missed_patterns": "none"}):
        assert "bad_output" in codes(read(answer(outputs={**OUTPUTS, **bad}))[1], "error"), bad
    lenient = read(answer(outputs={**OUTPUTS, "emotional_after_state": "More Anxious", "felt_seen": "3",
                                   "top_strengths": [{"item": "1.1", "note": "n"}] * 5}))[0]
    assert lenient.outputs["emotional_after_state"] == "more_anxious" and lenient.outputs["felt_seen"] == 3
    assert len(lenient.outputs["top_strengths"]) == 3                         # capped at max_items
    no_outputs = answer()
    del no_outputs["outputs"]
    assert "missing_outputs" in codes(read(no_outputs)[1], "error")


def test_strengths_and_improvements_must_name_real_items():
    for ref in ("3.2", "2a", "2a.3", "2a (positions 1-2)", "6.2: hard cards"):
        assert RUBRIC.is_item_reference(ref), ref
    for ref in ("9.7", "2.8", "3.25", "section 3", ""):
        assert not RUBRIC.is_item_reference(ref), ref
    outputs = {**OUTPUTS, "top_strengths": [{"item": "9.7", "note": "n"}, {"item": "6.2", "note": "n"}],
               "top_improvements": [{"item": "2.8", "fix": "f"}]}
    parsed, problems = read(answer(outputs=outputs))
    bad = [p for p in problems if p.code == "unknown_item_reference"]
    assert len(bad) == 2 and {p.severity for p in bad} == {"warning"} and codes(problems, "error") == []
    assert len(parsed.outputs["top_strengths"]) == 2          # kept, but flagged


def test_rendered_form_and_response_format():
    form = RUBRIC.render_form(LABELS)
    for expected in ("5 = Excellent", "## Section 3. Synthesis (weight 20)", "- 2a [O]", "- 4.1 [S] Swap test",
                     "(N/A only if no reversed cards were drawn)", "Positions, in order: 1. Present; 2. Challenge",
                     "10. Outcome", "- RF1 (critical)", "justification for any score of 1, 2 or 5"):
        assert expected in form, expected
    fmt = RUBRIC.render_response_format(LABELS)
    assert '"1.2": <1-5 or "N/A">' in fmt and '"2a": [10 whole numbers 1-5, one per position, in order]' in fmt
    assert '"emotional_after_state": "calmer | same | more_anxious | inspired | confused"' in fmt
    assert "Do not add section scores, a total or a grade" in fmt and '"2a.3"' in fmt
    with pytest.raises(RubricError, match="needs labels"):
        RUBRIC.render_response_format({}) if False else RUBRIC.score_keys(RUBRIC.item("2a"), {})


def test_rubric_loading_errors(tmp_path):
    with pytest.raises(KeyError, match="celtic_cross_reading_v1"):
        get_rubric("nope")
    bad = tmp_path / "x_v1.toml"
    bad.write_text('name = "x"\nversion = 1\n', encoding="utf-8")
    with pytest.raises(RubricError, match="missing key"):
        load_rubric(bad)


# ---- checks --------------------------------------------------------------------------------
@pytest.mark.parametrize("text", ['{"a": 1}', '```json\n{"a": 1}\n```', 'Here you go:\n{"a": 1}\nHope that helps.',
                                  '  {"a": 1}  '])
def test_extract_json(text):
    assert extract_json(text) == {"a": 1}


def test_extract_json_failures():
    assert extract_json("no json here") is None and extract_json("") is None and extract_json(None) is None
    assert extract_json('{"a": 1') is None


def test_quote_in_text():
    text = "The **Tower** says: “this will happen” no matter\n  what you do.  It’s decided."
    assert quote_in_text("this will happen", text)
    assert quote_in_text('"This will happen" no matter what you do.', text)        # case, quotes, whitespace
    assert quote_in_text("The Tower says ... It's decided", text)                   # elision, markdown, apostrophe
    assert not quote_in_text("you are doomed", text) and not quote_in_text("...", text)
    assert not quote_in_text("It's decided ... The Tower says", text)               # parts out of order


def test_rubric_answer_check():
    reading = "The Tower shows upheaval. This will happen no matter what. Take a breath."
    check = RubricAnswerCheck(RUBRIC, labels=lambda e: LABELS, quoted_text=lambda e: reading,
                              na_required=lambda e: {"1.2": e["no_reversals"]})
    v = Validator([check])
    good = v.validate(json.dumps(answer(flags=[{"id": "RF5", "quote": "This will happen no matter what."}])),
                      expected={"no_reversals": False})
    assert good.passed and good.checks["rubric_answer"].details["red_flags"] == ["RF5"]
    invented = v.validate(json.dumps(answer(flags=[{"id": "RF1", "quote": "you will fall ill"}])),
                          expected={"no_reversals": False})
    assert [i.code for i in invented.errors] == ["quote_not_in_text"]
    assert [i.code for i in v.validate("I can't do that.", expected={"no_reversals": False}).errors] == ["invalid_json"]
    na = v.validate(json.dumps(answer(**{"1.2": "N/A"})), expected={"no_reversals": False})
    assert "na_not_applicable" in {i.code for i in na.errors}


def test_json_fields_check():
    check = JsonFieldsCheck(EXPECTATION_FIELDS)
    good = {"situation_summary": "Busy.", "what_they_hoped_for": "Insight", "framing_preference": "Reflective",
            "must_haves": ["Plain language", " "], "sensitivities": None}
    fields, problems = check.read(json.dumps(good))
    assert problems == [] and fields["framing_preference"] == "reflective"
    assert fields["must_haves"] == ["Plain language"] and fields["sensitivities"] == []
    fields, problems = check.read(json.dumps({**good, "framing_preference": "mystical", "situation_summary": ""}))
    assert fields is None and {p[1] for p in problems} == {"framing_preference", "situation_summary"}
    assert not Validator([check]).validate("not json").passed


# ---- pipeline fixtures -----------------------------------------------------------------------
def answer_for(prompt: str, behaviour: str) -> str:
    """A form answer consistent with the prompt's facts (reversals, must-haves), or deliberately broken."""
    cards = prompt[prompt.index("- Cards drawn, by position:"):prompt.index("# The reading\n")]
    reading = prompt[prompt.index("<reading>") + 9:prompt.index("</reading>")].strip()
    any_reversed = bool(re.search(r" reversed$", cards, flags=re.M))
    has_must_haves = "- Your must-haves:" in prompt and "- Your must-haves: none" not in prompt
    scores = {k: ([4] * 10 if isinstance(v, list) else 4) for k, v in WORKED_EXAMPLE.items()}
    scores["1.2"] = 4 if any_reversed else "N/A"
    scores["7.4"] = 4 if has_must_haves else "N/A"
    flags = []
    if behaviour == "missing_item":
        del scores["3.1"]
    elif behaviour == "invented_quote":
        flags = [{"id": "RF1", "quote": "you will certainly be hospitalised next spring"}]
    elif behaviour == "real_flag":
        flags = [{"id": "RF5", "quote": reading[:40]}]
    text = json.dumps(answer(scores=scores, flags=flags))
    return f"Here is the completed form:\n```json\n{text}\n```" if behaviour == "fenced" else text


class EvalScripted(LLMProvider):
    """Plays both stages: persona expectations, then form answers. plan[input_id] = behaviours per call."""
    name = "eval-scripted"

    def __init__(self):
        super().__init__(default_model="evaluator-model")
        self.plan, self.calls, self.lock = {}, {}, threading.Lock()

    def chat(self, request):
        key = request.metadata["input_id"]
        with self.lock:
            n = self.calls.get(key, 0)
            self.calls[key] = n + 1
        if "Answer as that person" in request.messages[0]["content"]:          # stage 1
            wants_plain = int(request.metadata["persona_id"][-1], 16) % 2 == 0
            text = json.dumps({"situation_summary": "Work is busy.", "what_they_hoped_for": "Some perspective.",
                               "framing_preference": "reflective",
                               "must_haves": ["Plain language"] if wants_plain else [], "sensitivities": []})
        else:
            steps = self.plan.get(key, ["ok"])
            text = answer_for(request.prompt, steps[min(n, len(steps) - 1)])
        return ChatResponse(request.id, text, model="evaluator-model", finish_reason="stop",
                            usage=Usage(100, 50, 150), metadata=request.metadata)


@pytest.fixture
def registry(monkeypatch):
    monkeypatch.setattr(providers, "_CACHE", {})
    monkeypatch.setattr(providers, "_FACTORIES", dict(providers._FACTORIES))
    reader, evaluator = TarotScripted(), EvalScripted()
    register_provider("tarot-scripted", lambda **kw: reader)
    register_provider("eval-scripted", lambda **kw: evaluator)
    return evaluator


@pytest.fixture
def readings(tmp_path, registry):
    """A generation run of 4 readings (dataset 'demo'); returns (root, store)."""
    root = tmp_path / "generations"
    app = TarotReadings()
    store, requests = prepare_run(app, GenerationConfig(n=4, provider="tarot-scripted", seed=3, dataset="demo"),
                                  root=root)
    run_live(store, app, requests, progress=False)
    return root, store


def evaluation_app(readings, eval_root, **kw):
    root, store = readings
    kw.setdefault("subject_runs", [store.run_id])
    return CelticCrossEvaluation(subjects_root=root, expectations_root=eval_root, **kw)


def eval_config(**kw):
    kw.setdefault("n", 4)
    kw.setdefault("provider", "eval-scripted")
    kw.setdefault("seed", 11)
    kw.setdefault("persona_set", "tarot_personas")
    return GenerationConfig(**kw)


def check_foreign_keys(store, extra=()):
    for t in schema.TABLES.values():
        for c in t.columns:
            if not c.references:
                continue
            ref_table, ref_col = re.match(r"(\w+)\((\w+)\)", c.references).groups()
            valid = {r[ref_col] for s in (store, *extra) for r in s.read(ref_table)}
            missing = {r[c.name] for r in store.read(t.name) if r[c.name] is not None} - valid
            assert not missing, f"{t.name}.{c.name} -> {c.references}: {missing}"


# ---- subjects ----------------------------------------------------------------------------------
def test_load_subjects(readings):
    root, store = readings
    subjects = load_subjects(root, runs=[store.run_id])
    assert len(subjects) == 4 and all(s.text and list(s.input) == POSITIONS for s in subjects)
    assert {s.template for s in subjects} == {"celtic_cross_v1"} and all(s.valid for s in subjects)
    assert [s.response_id for s in subjects] == sorted(s.response_id for s in subjects)       # stable order
    assert len(load_subjects(root, dataset="demo")) == 4
    assert load_subjects(root, dataset="demo", templates=["celtic_cross_v2"]) == []
    with pytest.raises(ValueError, match="at least one run"):
        load_subjects(root)


def test_subjects_skip_invalid_and_failed_readings(tmp_path, registry):
    reader = providers.get_provider("tarot-scripted")
    root, app = tmp_path / "g", TarotReadings()
    store, requests = prepare_run(app, GenerationConfig(n=3, provider="tarot-scripted", validation_retries=0), root=root)
    reader.plan = {requests[0].metadata["input_id"]: ["invalid"], requests[1].metadata["input_id"]: ["error"]}
    run_live(store, app, requests, progress=False)
    assert len(load_subjects(root, runs=[store.run_id])) == 1
    assert len(load_subjects(root, runs=[store.run_id], valid_only=False)) == 2          # API failures never count


# ---- stage 1: expectations ---------------------------------------------------------------------
def test_expectations_run(tmp_path, registry):
    store = generate_expectations("tarot_personas", provider="eval-scripted", context=EXPECTATION_CONTEXT,
                                  persona_ids=["1", "2", "3"], root=tmp_path, progress=False)
    run = store.run()
    assert run["application"] == "persona_expectations" and run["status"] == "completed"
    assert store.run_id.startswith("run_persona_expectations_2")
    assert run["config"]["templates"] == ["reading_expectations_v1"]
    rows = store.read("persona_expectations")
    assert len(rows) == 3 and all(r["framing_preference"] == "reflective" for r in rows)
    assert all(r["context"] == EXPECTATION_CONTEXT for r in rows)
    prompt = store.read("prompts")[0]
    assert "You are about to ask for a Celtic Cross tarot reading." in prompt["user_text"]
    assert "You are not asking a specific question." in prompt["user_text"] and prompt["system"]
    expectations = load_expectations(store)
    assert set(expectations) == {p["persona_id"] for p in store.read("personas")}
    assert {tuple(e["must_haves"]) for e in expectations.values()} <= {(), ("Plain language",)}
    check_foreign_keys(store)
    check_primary_keys(store)


# ---- stage 2: evaluation -----------------------------------------------------------------------
def test_two_stage_evaluation_end_to_end(tmp_path, registry, readings):
    eval_root = tmp_path / "evaluations"
    stage1 = generate_expectations("tarot_personas", provider="eval-scripted", context=EXPECTATION_CONTEXT,
                                   root=eval_root, progress=False)
    app = evaluation_app(readings, eval_root, personas_per_subject=2, expectations_run=stage1.run_id)
    store, requests = prepare_run(app, eval_config(), root=eval_root)
    assert len(requests) == 8 and store.run()["application"] == "celtic_cross_evaluation"
    assert store.run_id.startswith("run_celtic_cross_evaluation_2")
    assert [r["rubric_id"] for r in store.read("rubrics")] == ["celtic_cross_reading_v1"]

    inputs = store.read("inputs")
    by_subject = {}
    for i in inputs:
        by_subject.setdefault(i["payload"]["subject_response_id"], []).append(i["persona_id"])
    assert len(by_subject) == 4 and all(len(set(p)) == 2 for p in by_subject.values())     # distinct evaluators
    prompt = store.read("prompts")[0]
    assert "# What you expected" in prompt["user_text"] and "<reading>" in prompt["user_text"]
    assert "## Section 3. Synthesis (weight 20)" in prompt["user_text"] and "Positions, in order: 1. Present" in prompt["user_text"]

    registry.plan = {requests[0].metadata["input_id"]: ["missing_item", "fenced"],          # retried, then fine
                     requests[1].metadata["input_id"]: ["real_flag"],
                     requests[2].metadata["input_id"]: ["invented_quote", "invented_quote", "invented_quote"]}
    summary = run_live(store, app, requests, progress=False)
    assert (summary["n_finished"], summary["n_valid"], summary["n_invalid"], summary["n_regenerated"]) == (8, 7, 1, 2)

    evaluations = {e["request_id"]: e for e in store.read("evaluations")}
    assert len(evaluations) == 7 and requests[2].id not in evaluations     # the invented quote never became a score
    plain = evaluations[requests[3].id]
    assert (plain["total_score"], plain["grade"], plain["cap_applied"], plain["n_red_flags"]) == (80.0, "B", None, 0)
    assert plain["evaluator_type"] == "llm" and plain["subject_type"] == "tarot_reading"
    assert plain["rubric_sha256"] == RUBRIC.sha256 and plain["outputs"]["felt_seen"] == 2
    assert plain["expectations"]["framing_preference"] == "reflective" and plain["subject_length_words"] > 50
    flagged = evaluations[requests[1].id]
    assert (flagged["score_before_caps"], flagged["cap_applied"], flagged["total_score"], flagged["grade"]) == \
        (80.0, 60, 60.0, "D")
    [flag] = store.read("evaluation_red_flags")
    assert (flag["evaluation_id"], flag["flag_id"], flag["severity"], flag["cap"]) == \
        (flagged["evaluation_id"], "RF5", "major", 60)
    assert evaluations[requests[0].id]["response_id"].endswith("-a2")       # scored from the retried answer

    items = [r for r in store.read("evaluation_item_scores") if r["evaluation_id"] == plain["evaluation_id"]]
    assert len(items) == 47                                                  # 27 single items + 2 x 10 positions
    position_rows = [r for r in items if r["item_id"] == "2b"]
    assert [r["label"] for r in position_rows] == POSITIONS and {r["item_type"] for r in position_rows} == {"S"}
    assert [r["item_key"] for r in position_rows][:2] == ["2b.1", "2b.2"]

    # every subject id points at a real reading; everything else is consistent within the run
    reading_ids = {r["response_id"] for r in readings[1].read("responses")}
    assert {e["subject_response_id"] for e in evaluations.values()} <= reading_ids
    check_foreign_keys(store)
    check_primary_keys(store)
    assert [r.to_dict() for r in rebuild_requests(store)] == [r.to_dict() for r in requests]


def test_na_follows_the_facts(tmp_path, registry, readings):
    """7.4 is N/A exactly when the evaluator has no must-haves; 1.2 exactly when no card is reversed."""
    eval_root = tmp_path / "e"
    stage1 = generate_expectations("tarot_personas", provider="eval-scripted", context=EXPECTATION_CONTEXT,
                                   root=eval_root, progress=False)
    expectations = load_expectations(stage1)
    app = evaluation_app(readings, eval_root, personas_per_subject=3, expectations_run=stage1.run_id)
    store, requests = prepare_run(app, eval_config(), root=eval_root)
    run_live(store, app, requests, progress=False)
    evaluations = store.read("evaluations")
    assert len(evaluations) == 12
    scores = {(r["evaluation_id"], r["item_key"]): r for r in store.read("evaluation_item_scores")}
    drawn = {r["response_id"]: i["payload"] for i in readings[1].read("inputs")
             for p in readings[1].read("prompts") if p["input_id"] == i["input_id"]
             for q in readings[1].read("requests") if q["prompt_id"] == p["prompt_id"]
             for r in readings[1].read("responses") if r["request_id"] == q["request_id"]}
    for e in evaluations:
        assert scores[(e["evaluation_id"], "7.4")]["is_na"] == (not expectations[e["persona_id"]]["must_haves"])
        no_reversals = not any(c.endswith(" reversed") for c in drawn[e["subject_response_id"]].values())
        assert scores[(e["evaluation_id"], "1.2")]["is_na"] == no_reversals


def test_evaluation_without_personas(tmp_path, registry, readings):
    eval_root = tmp_path / "e"
    app = evaluation_app(readings, eval_root)
    store, requests = prepare_run(app, eval_config(persona_set=None, n=2), root=eval_root)
    assert len(requests) == 2 and store.read("personas") == []
    prompt = store.read("prompts")[0]
    assert "# Who you are" not in prompt["user_text"] and "No evaluator persona is given" in prompt["system"]
    run_live(store, app, requests, progress=False)
    evaluations = store.read("evaluations")
    assert len(evaluations) == 2 and all(e["persona_id"] is None and e["expectations"] is None for e in evaluations)


def test_sampling_and_pairing(tmp_path, registry, readings):
    eval_root = tmp_path / "e"
    a, _ = prepare_run(evaluation_app(readings, eval_root), eval_config(n=2, mode="dry_run"), root=eval_root)
    b, _ = prepare_run(evaluation_app(readings, eval_root), eval_config(n=2, mode="dry_run"), root=eval_root)
    pick = lambda s: [(i["payload"]["subject_response_id"], i["persona_id"]) for i in s.read("inputs")]
    assert pick(a) == pick(b) and len(pick(a)) == 2                         # same seed -> same readings and personas

    cycled, _ = prepare_run(evaluation_app(readings, eval_root, personas_per_subject=2),
                            eval_config(mode="dry_run", persona_sampling="cycle"), root=eval_root)
    numbers = {p["persona_id"]: p["persona_number"] for p in cycled.read("personas")}
    assert [numbers[i["persona_id"]] for i in cycled.read("inputs")] == [1, 2, 3, 4, 5, 6, 7, 8]

    with pytest.raises(ValueError, match="generated without a persona"):
        prepare_run(evaluation_app(readings, eval_root, pairing="subject"), eval_config(mode="dry_run"), root=eval_root)
    with pytest.raises(KeyError, match="generate them first"):
        stage1 = generate_expectations("tarot_personas", provider="eval-scripted", persona_ids=["1"],
                                       context=EXPECTATION_CONTEXT, root=eval_root, progress=False)
        prepare_run(evaluation_app(readings, eval_root, expectations_run=stage1.run_id),
                    eval_config(mode="dry_run", persona_ids=["2"]), root=eval_root)


def test_the_reading_persona_can_evaluate_its_own_reading(tmp_path, registry):
    root, eval_root, gen = tmp_path / "g", tmp_path / "e", TarotReadings()
    store, requests = prepare_run(gen, GenerationConfig(n=3, provider="tarot-scripted", persona_set="tarot_personas",
                                                        seed=2), root=root)
    run_live(store, gen, requests, progress=False)
    app = CelticCrossEvaluation(subjects_root=root, subject_runs=[store.run_id], pairing="subject",
                                subject_templates=["celtic_cross_v2"], expectations_root=eval_root)
    evaluation, _ = prepare_run(app, eval_config(mode="dry_run"), root=eval_root)
    written_for = {r["response_id"]: i["persona_id"] for i in store.read("inputs")
                   for p in store.read("prompts") if p["input_id"] == i["input_id"]
                   for q in store.read("requests") if q["prompt_id"] == p["prompt_id"]
                   for r in store.read("responses") if r["request_id"] == q["request_id"]}
    assert all(i["persona_id"] == written_for[i["payload"]["subject_response_id"]] for i in evaluation.read("inputs"))


def test_resume_scores_only_what_is_missing(tmp_path, registry, readings):
    eval_root = tmp_path / "e"
    app = evaluation_app(readings, eval_root)
    store, requests = prepare_run(app, eval_config(persona_set=None), root=eval_root)
    run_live(store, app, requests[:2], progress=False)
    assert len(store.read("evaluations")) == 2
    fresh = CelticCrossEvaluation(**{k: v for k, v in store.run()["config"]["application_options"].items()
                                     if k != "rubric_sha256"})
    assert run_live(store, fresh, progress=False)["n_finished"] == 4
    assert len(store.read("evaluations")) == 4 and registry.calls[requests[0].metadata["input_id"]] == 1
    check_primary_keys(store)


# ---- CLI ---------------------------------------------------------------------------------------
def test_cli_evaluate_runs_both_stages(tmp_path, registry, readings, capsys):
    root, store = readings
    out = str(tmp_path / "evaluations")
    assert main(["--out", out, "evaluate", "--readings-root", str(root), "--readings-dataset", "demo", "--n", "2",
                 "--provider", "eval-scripted", "--persona", "1", "--persona", "2", "--personas-per-reading", "2",
                 "--seed", "5", "--dataset", "eval-demo"]) == 0
    printed = capsys.readouterr().out
    assert "Stage 1: recording persona expectations" in printed and "Stage 2: prepared 4 evaluations of 2 readings" in printed
    assert "evaluations: 4   mean total 80.0" in printed and "grades: B: 4" in printed and "red flags: none" in printed
    runs = {r["application"]: r for r in map(lambda p: RunStore(p).run(), (tmp_path / "evaluations").glob("run_*"))}
    assert set(runs) == {"persona_expectations", "celtic_cross_evaluation"}
    evaluation = runs["celtic_cross_evaluation"]
    assert evaluation["config"]["application_options"]["expectations_run"] == runs["persona_expectations"]["run_id"]
    assert evaluation["dataset"] == "eval-demo" and evaluation["config"]["params"] == {"max_tokens": 4000}

    assert main(["--out", out, "status", evaluation["run_id"]]) == 0
    assert main(["--out", out, "resume", evaluation["run_id"]]) == 0
    assert main(["--out", out, "list"]) == 0
    listing = capsys.readouterr().out
    assert "4 evaluations" in listing and "persona_expectations" in listing


def test_cli_expectations_then_reuse_and_options(tmp_path, registry, readings, capsys):
    root, store = readings
    out = str(tmp_path / "evaluations")
    assert main(["--out", out, "expectations", "--provider", "eval-scripted", "--persona", "4"]) == 0
    [stage1] = list((tmp_path / "evaluations").glob("run_*"))
    assert main(["--out", out, "evaluate", "--readings-root", str(root), "--readings-run", store.run_id, "--n", "1",
                 "--provider", "eval-scripted", "--persona", "4", "--expectations", stage1.name, "--json-mode",
                 "--temperature", "0"]) == 0
    assert "Stage 1" not in capsys.readouterr().out                           # reused, not regenerated
    evaluation = next(RunStore(p) for p in (tmp_path / "evaluations").glob("run_*") if p != stage1)
    assert evaluation.run()["config"]["params"] == {"max_tokens": 4000, "temperature": 0.0,
                                                    "response_format": {"type": "json_object"}}
    assert len(evaluation.read("evaluations")) == 1

    assert main(["--out", out, "evaluate", "--n", "1"]) == 2                  # no readings selected
    assert main(["--out", out, "evaluate", "--readings-root", str(root), "--readings-dataset", "demo",
                 "--no-personas", "--dry-run", "--provider", "eval-scripted"]) == 0
    assert "prepared 4 evaluations of 4 readings" in capsys.readouterr().out   # default: every reading


def test_cli_batch_evaluation_submit_then_collect(tmp_path, registry, readings, capsys):
    """Stage 1 runs live; stage 2 goes to the provider's batch API and is collected later."""
    from conftest import FakeOpenAIClient, batch_body
    from llm.providers.openai_compatible import OpenAICompatibleProvider

    def outputs(uploaded):
        return [{"custom_id": line["custom_id"], "error": None, "response": {"status_code": 200, "body": batch_body(
            answer_for(line["body"]["messages"][-1]["content"], "ok"))}} for line in uploaded], []

    client = FakeOpenAIClient(statuses=("validating", "in_progress", "completed"), batch_outputs=outputs)
    batch_llm = OpenAICompatibleProvider(name="fake-batch", client=client, default_model="fake-model",
                                         supports_batch=True, batch_dir=tmp_path / "unused")
    register_provider("fake-batch", lambda **kw: batch_llm)

    root, store = readings
    out = str(tmp_path / "evaluations")
    assert main(["--out", out, "expectations", "--provider", "eval-scripted", "--persona", "1", "--persona", "2"]) == 0
    [stage1] = list((tmp_path / "evaluations").glob("run_persona_expectations_*"))
    assert main(["--out", out, "evaluate", "--readings-root", str(root), "--readings-run", store.run_id, "--n", "3",
                 "--provider", "fake-batch", "--mode", "batch", "--persona", "1", "--persona", "2",
                 "--expectations", stage1.name]) == 0
    assert "Submitted batch batch_1" in capsys.readouterr().out
    [run_dir] = list((tmp_path / "evaluations").glob("run_celtic_cross_evaluation_*"))
    evaluation = RunStore(run_dir)
    assert evaluation.run()["status"] == "submitted" and evaluation.read("evaluations") == []
    assert len(client.uploaded) == 3 and client.uploaded[0]["url"] == "/v1/chat/completions"
    assert "<reading>" in client.uploaded[0]["body"]["messages"][-1]["content"]

    assert main(["--out", out, "collect", run_dir.name]) == 1             # still running
    assert main(["--out", out, "collect", run_dir.name]) == 0
    evaluations = evaluation.read("evaluations")
    assert len(evaluations) == 3 and {e["total_score"] for e in evaluations} == {80.0}
    assert evaluation.run()["status"] == "completed"
    assert {r["batch_id"] for r in evaluation.read("responses")} == {"batch_1"}
    assert all(b["collected"] for b in evaluation.read("batches"))
    check_foreign_keys(evaluation)


def test_cli_resume_can_lower_workers(tmp_path, registry, readings):
    """A run refused for concurrency is resumed with fewer workers; the new setting is kept."""
    eval_root = tmp_path / "e"
    app = evaluation_app(readings, eval_root)
    store, requests = prepare_run(app, eval_config(persona_set=None), root=eval_root)
    run_live(store, app, requests[:1], progress=False)
    assert load_config(store).max_workers == 4
    assert main(["--out", str(eval_root), "resume", store.run_id, "--workers", "1"]) == 0
    assert load_config(store).max_workers == 1 and len(store.read("evaluations")) == 4
    assert main(["--out", str(eval_root), "resume", store.run_id]) == 0
    assert load_config(store).max_workers == 1


def test_readings_can_be_selected_by_generating_model_and_template(tmp_path, registry, readings, capsys):
    root, store = readings
    assert len(load_subjects(root, runs=[store.run_id], models=["SCRIPTED-model"])) == 4      # case is ignored
    assert load_subjects(root, runs=[store.run_id], models=["other-model"]) == []
    out = str(tmp_path / "evaluations")
    common = ["--out", out, "evaluate", "--readings-root", str(root), "--readings-run", store.run_id,
              "--provider", "eval-scripted", "--no-personas", "--dry-run"]
    assert main([*common, "--readings-model", "scripted-model", "--readings-template", "celtic_cross_v1"]) == 0
    assert "prepared 4 evaluations of 4 readings" in capsys.readouterr().out
    [run] = [RunStore(p).run() for p in (tmp_path / "evaluations").glob("run_*")]
    assert run["config"]["application_options"]["subject_models"] == ["scripted-model"]

    assert main([*common, "--readings-model", "other-model"]) == 1                             # says what there is
    assert "The selected runs hold: 4 of celtic_cross_v1 by scripted-model" in capsys.readouterr().err
    assert main([*common, "--readings-template", "celtic_cross_v4"]) == 1
    assert "no readings match --readings-template celtic_cross_v4." in capsys.readouterr().err
