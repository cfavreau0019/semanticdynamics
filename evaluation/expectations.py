"""
Stage 1 of a persona evaluation: what the persona expects *before* seeing the thing they
will evaluate, so hindsight can't colour it (Part A3 of the Celtic Cross form).

Expectations depend only on the persona and on what they are told they are about to
receive (the `context`), not on any particular output. They are therefore generated once
per persona, as their own small run, and reused for every evaluation by that persona.

What is generic here: one request per persona, the fields of the answer, the
persona_expectations table. What an application supplies, as a subclass of
PersonaExpectations: the task's name, its prompt template and the context (see
applications/tarot/code/persona_expectations.py).

    store = generate_expectations(TarotPersonaExpectations(), "tarot_personas", provider="openai", model="...")
    expectations = load_expectations(store)        # {persona_id: {...}}
"""
import random
from pathlib import Path
from typing import Any, Optional, Sequence

from inference.pipeline import GenerationConfig, final_responses, prepare_run, run_live
from inference.store import RunStore
from inference.task import InferenceTask
from evaluation.checks import JsonFieldsCheck
from llm import ChatResponse
from personas import PersonaSet
from validation import NotTruncated, Validator

FRAMING_PREFERENCES = ["predictive", "reflective", "creative", "no_preference"]
EXPECTATION_FIELDS = {
    "situation_summary": "text",
    "what_they_hoped_for": "text",
    "framing_preference": FRAMING_PREFERENCES,
    "must_haves": "list",
    "sensitivities": "list",
}


class PersonaExpectations(InferenceTask):
    """
    One request per persona. `context` is what the persona is told they are about to
    receive; its keys are the template's variables besides `persona`.

    A subclass sets `name`, the prompt template (`default_template` and `persona_template`,
    the same one: every request has a persona) and `default_context`.
    """
    input_type = "expectation_context"
    cli = None                              # started as stage 1 of an evaluation task: evaluate expectations <task>
    default_context: dict[str, Any] = {}

    def __init__(self, context: Optional[dict[str, Any]] = None):
        self.context = dict(self.default_context if context is None else context)
        self._check = JsonFieldsCheck(EXPECTATION_FIELDS, name="expectations")

    def options(self) -> dict:
        return {"context": self.context}

    @classmethod
    def from_options(cls, options: dict) -> "PersonaExpectations":
        return cls(options.get("context"))

    def sample_inputs(self, n: int, rng: random.Random) -> list[dict]:
        return [dict(self.context) for _ in range(n)]

    def template_variables(self, payload: dict, persona: Optional[dict]) -> dict[str, Any]:
        return {"persona": persona, **payload}

    def validator(self):
        return Validator([NotTruncated(), self._check], name=self.name)

    def derived_rows(self, response: ChatResponse, response_id: str) -> dict[str, list[dict]]:
        fields, _ = self._check.read(response.text)
        persona_id = response.metadata.get("persona_id")
        if fields is None or persona_id is None:
            return {}
        return {"persona_expectations": [{
            "response_id": response_id, "persona_id": persona_id,
            "context": response.metadata.get("input"), **fields,
        }]}


def generate_expectations(
    task: PersonaExpectations,
    persona_set: str,
    provider: str,
    model: Optional[str] = None,
    persona_ids: Sequence[str] = (),
    root: str | Path = "data/evaluations",
    params: Optional[dict] = None,
    max_workers: int = 4,
    name: Optional[str] = None,
    dataset: Optional[str] = None,
    dry_run: bool = False,
    progress: bool = True,
) -> RunStore:
    """Run stage 1 for every persona in the set (or the given ids), live. Returns the run's store."""
    people = PersonaSet.load(persona_set)
    if persona_ids:
        people = people.select(persona_ids)
    config = GenerationConfig(
        n=len(people), provider=provider, model=model, params=dict(params or {"max_tokens": 2000}),
        mode="dry_run" if dry_run else "live", max_workers=max_workers, name=name, dataset=dataset,
        persona_set=persona_set, persona_sampling="cycle", persona_ids=[str(i) for i in persona_ids])
    store, requests = prepare_run(task, config, root=root)
    if not dry_run:
        run_live(store, task, requests, progress=progress)
    return store


def load_expectations(store: RunStore) -> dict[str, dict[str, Any]]:
    """{persona_id: expectations} from a stage-1 run (each persona's final response)."""
    final = {r["response_id"] for r in final_responses(store).values()}
    out = {}
    for row in store.read("persona_expectations"):
        if row["response_id"] in final:
            out[row["persona_id"]] = {k: row[k] for k in EXPECTATION_FIELDS} | {"response_id": row["response_id"]}
    return out
