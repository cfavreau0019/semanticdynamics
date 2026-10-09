"""
Subjects: the generated outputs an evaluation scores, read from data_generation runs.

A subject is the final response of a request, together with the structured input it was
generated from (e.g. the card draw), so the evaluator can be told what the output was
supposed to reflect.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

from data_generation.pipeline import final_responses, list_runs
from data_generation.store import DEFAULT_ROOT, RunStore


@dataclass(frozen=True)
class Subject:
    response_id: str
    request_id: str
    run_id: str
    text: str
    input: Any                       # the generation input payload, e.g. {position: card}
    template: Optional[str] = None   # prompt template that produced it
    model: Optional[str] = None
    persona_id: Optional[str] = None  # persona the output was written for, if any
    valid: Optional[bool] = None     # the generation run's validation verdict


def load_subjects(
    root: str | Path = DEFAULT_ROOT,
    runs: Sequence[str] = (),
    dataset: Optional[str] = None,
    templates: Sequence[str] = (),
    valid_only: bool = True,
    models: Sequence[str] = (),
) -> list[Subject]:
    """
    Final, successful responses of the given runs and/or of every run in a dataset.

    templates  — keep only outputs of these prompt templates (e.g. ["celtic_cross_v1"])
    models     — keep only outputs written by these models (as recorded with the response; case is ignored)
    valid_only — drop outputs that failed the generation run's validation
    Returned in a stable order (run, then response id), so seeded sampling is reproducible.
    """
    run_ids = list(runs)
    if dataset is not None:
        run_ids += [r["run_id"] for r in list_runs(root, dataset) if r["run_id"] not in run_ids]
    if not run_ids:
        raise ValueError("Give at least one run id or a dataset to take subjects from")

    wanted_models = {m.casefold() for m in models}
    subjects: list[Subject] = []
    for run_id in run_ids:
        store = RunStore.open(run_id, root)
        prompts = {p["prompt_id"]: p for p in store.read("prompts")}
        requests = {r["request_id"]: r for r in store.read("requests")}
        inputs = {i["input_id"]: i for i in store.read("inputs")}
        texts = {t["response_id"]: t["text"] for t in store.read("response_texts")}
        verdicts = {v["response_id"]: v["passed"] for v in store.read("validations")}
        for request_id, response in sorted(final_responses(store).items(), key=lambda kv: kv[1]["response_id"]):
            text = texts.get(response["response_id"])
            if response["error"] is not None or not text:
                continue
            valid = verdicts.get(response["response_id"])
            if valid_only and valid is False:
                continue
            prompt = prompts[requests[request_id]["prompt_id"]]
            if templates and prompt["template"] not in templates:
                continue
            model = response.get("model") or requests[request_id].get("model")
            if wanted_models and str(model).casefold() not in wanted_models:
                continue
            source = inputs[prompt["input_id"]]
            subjects.append(Subject(
                response_id=response["response_id"], request_id=request_id, run_id=run_id, text=text,
                input=source["payload"], template=prompt["template"], model=model,
                persona_id=source.get("persona_id"), valid=valid))
    return subjects
