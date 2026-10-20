"""
InferenceTask: what the generation pipeline (pipeline.py) asks of a task.

A task describes one kind of LLM request that a run is made of: how its inputs are sampled,
which variables they give to the prompt templates, how a response is validated and which
extra table rows are computed from it. The pipeline does the rest: prompts, requests, live
or batch execution, validation retries, resume, and the tables.

Tasks live with the application they belong to, in applications/<application>/; see
applications/tarot/ for the reference implementations.
"""
import random
from abc import ABC, abstractmethod
from typing import Any, Optional

from llm import ChatResponse
from personas import PersonaSet

# metadata keys carried on each ChatRequest (never sent to the model)
INPUT_KEY = "input"


class InferenceTask(ABC):
    """What a task has to provide."""

    name: str                               # e.g. "tarot_celtic_cross"
    input_type: str                         # e.g. "tarot_draw"
    default_template: str                   # prompt library reference used when none is given
    persona_template: Optional[str] = None  # default when personas are supplied (must accept `persona`)

    @abstractmethod
    def sample_inputs(self, n: int, rng: random.Random) -> list[dict]:
        ...

    def template_variables(self, payload: dict, persona: Optional[dict]) -> dict[str, Any]:
        """
        Everything a prompt template may use for this input. Each template takes the
        variables it declares from this pool, so one task can serve templates with
        different needs (e.g. with and without `persona`).
        """
        return {"input": payload, "persona": persona}

    def assign_personas(self, payloads: list[dict], people: PersonaSet, rng: random.Random,
                        mode: str) -> list[Optional[dict]]:
        """One persona (or None) per payload. Override to control the pairing, e.g. distinct personas per subject."""
        return people.sample(rng, len(payloads), mode)

    def input_items(self, payload: dict) -> list[dict]:
        """Long-form rows (label, value, entity, qualifier) for input_items; optional."""
        return []

    def reference_rows(self) -> dict[str, list[dict]]:
        """Rows for reference tables this task uses (e.g. its rubric), written when a run is prepared."""
        return {}

    def validation_expected(self, metadata: dict) -> Any:
        """What the validator compares a response against (`expected`); default: the input payload."""
        return metadata.get(INPUT_KEY)

    def derived_rows(self, response: ChatResponse, response_id: str) -> dict[str, list[dict]]:
        """
        Extra table rows computed from a finished request's returned response, e.g. parsed
        scores: {table: [rows]}. Called once per request after its responses are written.
        """
        return {}

    def validator(self):
        """A validation.Validator whose `expected` is the input payload, or None."""
        return None

    def options(self) -> dict:
        """Task settings to record in runs.config."""
        return {}

    # ---- command line ------------------------------------------------------------------
    # What the general scripts (inference.generate, evaluation.evaluate) ask of a task,
    # so that they need no code of their own for it.
    cli: Optional[str] = "generate"         # the script whose `run` starts this task: "generate" or "evaluate";
                                            # None = it is only started as part of another task

    @classmethod
    def add_arguments(cls, parser) -> None:
        """
        Add this task's own options to its `run` command (an argparse parser that already has
        the general ones). parser.set_defaults(...) changes the default of a general option.
        """

    @classmethod
    def from_args(cls, args) -> "InferenceTask":
        """The task of a new run, from the parsed command line."""
        return cls()

    @classmethod
    def from_options(cls, options: dict) -> "InferenceTask":
        """The task of an existing run, rebuilt from the options() recorded with it (to resume or collect it)."""
        return cls(**options)

    def run_warnings(self, config, root) -> list[str]:
        """
        Anything to tell the user before a run with this GenerationConfig is prepared under
        `root`, e.g. that its seed would repeat the inputs of an earlier run.
        """
        return []
