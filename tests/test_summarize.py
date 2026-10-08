"""evaluation.summarize: selecting evaluations, rendering their results as prompt variables, and the CLI."""
import json

import pytest

from evaluation.dashboard import build_data
from evaluation.summarize import DEFAULT_MODEL, main, select, template_variables
from llm import ChatResponse, Usage
from llm.providers import LLMProvider, register_provider
from prompts import get_template
from test_dashboard import evaluated
from test_evaluation import readings, registry  # noqa: F401

TASK = "tarot_celtic_cross"


class Summarizer(LLMProvider):
    name = "summary-scripted"

    def __init__(self):
        super().__init__(default_model="default-model")
        self.requests = []

    def chat(self, request):
        self.requests.append(request)
        return ChatResponse(request.id, "## Verdict\nSolid but generic.", model=request.model, finish_reason="stop",
                            usage=Usage(900, 100, 1000), metadata=request.metadata)


@pytest.fixture
def summarizer(registry):
    provider = Summarizer()
    register_provider("summary-scripted", lambda **kw: provider)
    return provider


def test_select_by_every_dimension(tmp_path, registry, readings):
    eval_root, store, _ = evaluated(tmp_path, registry, readings)
    data = build_data(eval_root)
    assert len(select(data, TASK)) == 8 and select(data, "other_task") == []
    assert len(select(data, TASK, {"template": ["celtic_cross_v1"], "generating_model": ["scripted-model"],
                                   "evaluating_model": ["evaluator-model"], "run": [store.run_id]})) == 8
    assert len(select(data, TASK, {"readings_dataset": ["demo"]})) == 8
    for dimension in ("template", "generating_model", "evaluating_model", "run", "readings_dataset"):
        assert select(data, TASK, {dimension: ["nope"]}) == []
    one = data["evaluations"][0]
    persona = data["personas"][one["pid"]]
    by_id = select(data, TASK, {"persona": [one["pid"]]})
    assert by_id and {e["pid"] for e in by_id} == {one["pid"]}
    assert select(data, TASK, {"persona": [str(persona["number"])]}) == by_id
    assert select(data, TASK, {"persona": [persona["name"].upper()]}) == by_id
    with pytest.raises(ValueError, match="Unknown dimensions"):
        select(data, TASK, {"colour": ["red"]})


def test_template_variables_hold_scores_and_comments(tmp_path, registry, readings):
    eval_root, _, _ = evaluated(tmp_path, registry, readings)
    data = build_data(eval_root)
    v = template_variables(data, select(data, TASK), TASK, by=["evaluating_model", "generating_model"], focus="tone")
    assert "8 evaluations of 4 texts, scored with the rubric celtic_cross_reading_v1" in v["scope"]
    assert "- Evaluating model: evaluator-model (8)" in v["scope"]
    assert "Total score (0-100): mean 77.5" in v["quantitative_results"]                 # 7 x 80 and one capped at 60
    assert "Grades: A 0, B 7, C 0, D 1, F 0" in v["quantitative_results"]
    assert "| 3. Synthesis | 20 | 4.00 | 0.00 |" in v["quantitative_results"]
    assert "| 2a | O | Is the card read in light of this position? | 80 | 4.00 |" in v["quantitative_results"]
    assert "- 2a by positions: Present 4.00, Challenge 4.00" in v["quantitative_results"]
    assert "- Felt seen (1-5;" in v["quantitative_results"] and "RF5 (major, caps the total at 60)" in v["quantitative_results"]
    assert "## By evaluating model" in v["breakdown"] and "| evaluator-model | 8 | 4 | 77.5 |" in v["breakdown"]
    assert "### 6.2 [" in v["question_feedback"] and "(top strength; " in v["question_feedback"]
    assert "; evaluator-model; generating model scripted-model) hard cards handled honestly" in v["question_feedback"]
    assert "## Persona reaction (8 of 8)" in v["overall_feedback"] and "(gave 80.0, grade B; " in v["overall_feedback"]
    assert "- three Swords form the spine (x8)" in v["overall_feedback"] and "- RF5 (" in v["overall_feedback"]
    assert v["focus"] == "tone" and template_variables(data, select(data, TASK), TASK)["breakdown"] is None

    few = template_variables(data, select(data, TASK), TASK, comments_per_question=1, comments_overall=2)
    assert "## Persona reaction (2 of 8)" in few["overall_feedback"]
    assert few == template_variables(data, select(data, TASK), TASK, comments_per_question=1, comments_overall=2)
    prompt = get_template("evaluation_summary_v1").render(v)
    assert "# Scores by group" in prompt.user and "Pay particular attention to this: tone" in prompt.user
    assert "# Scores by group" not in get_template("evaluation_summary_v1").render(few).user


def test_cli_writes_the_summary_and_how_it_was_made(tmp_path, registry, readings, summarizer, capsys):
    eval_root, store, _ = evaluated(tmp_path, registry, readings)
    common = ["--task", TASK, "--evaluations-root", str(eval_root), "--provider", "summary-scripted"]
    assert main([*common, "--dry-run", "--by", "persona"]) == 0
    printed = capsys.readouterr().out
    assert "----- system -----" in printed and "## By persona" in printed and summarizer.requests == []

    assert main([*common, "--evaluating-model", "evaluator-model", "--run", store.run_id, "--temperature", "0"]) == 0
    [request] = summarizer.requests
    assert request.model == DEFAULT_MODEL == "gpt-6-luna" and request.params == {"max_tokens": 4000, "temperature": 0.0}
    assert request.messages[0]["role"] == "system" and "# What the evaluators wrote overall" in request.messages[1]["content"]
    [page] = (eval_root / "summaries").glob("*.md")
    text = page.read_text(encoding="utf-8")
    assert text.startswith(f"# Evaluation summary: {TASK}") and "Solid but generic." in text
    assert "- Evaluating model (selected): evaluator-model (8)" in text and "summary-scripted/gpt-6-luna" in text
    record = json.loads(page.with_suffix(".json").read_text(encoding="utf-8"))
    assert record["filters"] == {"evaluating_model": ["evaluator-model"], "run": [store.run_id]}
    assert (record["n_evaluations"], record["template"], record["usage"]["total_tokens"]) == (8, "evaluation_summary_v1", 1000)
    assert record["messages"] == request.messages and record["summary"] in text

    out = tmp_path / "per_persona"
    assert main([*common, "--each", "persona", "--model", "other-model", "--out", str(out)]) == 0
    assert len(list(out.glob("*.md"))) == len(build_data(eval_root)["personas"]) == len(summarizer.requests) - 1
    assert summarizer.requests[-1].model == "other-model"

    assert main(["--task", "nope", "--evaluations-root", str(eval_root)]) == 1
    assert "Tasks with evaluations: tarot_celtic_cross" in capsys.readouterr().err

    # the texts' dataset ("demo") is not the evaluation runs' own label ("eval-demo")
    assert main([*common, "--dataset", "demo", "--dry-run"]) == 1
    assert "use --readings-dataset demo" in capsys.readouterr().err
    assert main([*common, "--readings-dataset", "demo", "--dry-run"]) == 0
    assert "- Readings dataset (selected): demo (8)" in capsys.readouterr().out
    assert main([*common, "--dataset", "eval-demo", "--dry-run"]) == 0
