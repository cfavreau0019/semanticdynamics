"""
InferenceTask: a task as the pipeline (pipeline.py) runs it, built from its InferenceTaskConfig.

    task = InferenceTask.from_config(config)     # the config's implemented_by class, or InferenceTask itself
    store, requests = prepare_run(task)

A task makes a run's inputs, gives each prompt template its variables and validates responses;
the pipeline does the rest: prompts, requests, live or batch execution, validation retries,
resume, and the tables. What a task does comes from its config:
  inputs      the instantiation's instants (n empty ones without an instantiation), each stored
              as its input's payload with personas by id; prompt.sample says how they meet the
              templates (crossed: every instant with every template)
  variables   each template takes the instant's entry of the same name (renamed by its `bind`),
              and its fixed `variables`
  validation  the generic checks the config names and its custom validator's; `expected` is the
              instant's entry a response is checked against

An application's own Python subclasses InferenceTask, and its config names the class as
implemented_by. It may add input_items, derived_rows and reference_rows. A task whose inputs a
config cannot declare (e.g. evaluation's subjects, the outputs of other runs) computes them in
Python instead: sample_inputs, assign_personas and template_variables, with the templates named by
default_template and persona_template (pipeline.inputs_from_hooks). Such a task is rebuilt to
resume a run from the options() the run recorded (from_options), and its own command-line options
come from add_arguments and from_args.

Tasks live with the application they belong to, in applications/<application>/; see
applications/tarot/ for the reference implementations.
"""
import random
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

from inference.task_config import InferenceTaskConfig
from instantiation import Instantiation, Personas
from llm import ChatResponse
from personas import PersonaSet
from prompts import PromptTemplate, RenderedPrompt
from spec_files import import_reference

INPUT_KEY = "input"                 # metadata key carried on each ChatRequest (never sent to the model)
INSTANT = "instant"                 # inputs.input_type when the payload is an instant, {entry: value}


# ---- what a task gives the pipeline ------------------------------------------------------
@dataclass
class RunTemplate:
    """A prompt template as a run uses it, with where its variables come from."""
    template: PromptTemplate
    status: str                               # the library's status (draft, locked, ...), or "inline"
    bind: Mapping[str, str] = field(default_factory=dict)        # {variable: the input value it takes}
    variables: Mapping[str, Any] = field(default_factory=dict)   # fixed values

    def render(self, values: Mapping[str, Any]) -> RenderedPrompt:
        """Render with the variables it declares, taken from an input's values (renamed by bind), and its fixed ones."""
        taken = {v: values[self.bind.get(v, v)] for v in self.template.variables if self.bind.get(v, v) in values}
        return self.template.render({**taken, **self.variables})


@dataclass
class RunInput:
    """One input of a run, as the pipeline writes it."""
    payload: Any                              # inputs.payload, and validation's metadata[INPUT_KEY]
    values: Mapping[str, Any]                 # what the templates take their variables from
    persona_id: Optional[str] = None          # inputs.persona_id
    personas: list[dict] = field(default_factory=list)          # personas table rows of the personas it uses
    templates: Optional[tuple[int, ...]] = None                  # the run's templates it is rendered with; None: all


# ---- the task ----------------------------------------------------------------------------
class InferenceTask:
    """A task of a run, as its config describes it (see the module's docstring)."""

    name: str                               # e.g. "tarot_celtic_cross"
    input_type: str = INSTANT               # stored as inputs.input_type
    config: Optional[InferenceTaskConfig] = None
    cli: Optional[str] = None               # "evaluate": evaluation.evaluate's `run` starts it from its hooks

    def __init__(self, config: Optional[InferenceTaskConfig] = None):
        self.config = config
        if config is not None:
            self.name = config.name

    @classmethod
    def from_config(cls, config: InferenceTaskConfig) -> "InferenceTask":
        """The task a config describes: an instance of its implemented_by class, else of InferenceTask."""
        target = import_reference(config.implemented_by) if cls is InferenceTask and config.implemented_by else cls
        return target(config)

    # ---- inputs ----------------------------------------------------------------------------
    @property
    def computes_inputs(self) -> bool:
        """Whether the task computes its inputs in Python (sample_inputs) rather than from its config's instantiation."""
        return type(self).sample_inputs is not InferenceTask.sample_inputs

    def run_inputs(self, seed: int, library) -> tuple[list[RunTemplate], list[RunInput], dict]:
        """A run's templates and inputs, and anything more to record in runs.config; sampled from `seed`."""
        if self.computes_inputs:
            from inference.pipeline import inputs_from_hooks
            return inputs_from_hooks(self, seed, library)
        c = self.config
        if any(ref.template is None for ref in c.prompt.templates):
            raise ValueError(f"{c.name}: check the config to run it (InferenceTaskConfig with check=True): "
                             f"its templates are not resolved")
        templates = [RunTemplate(ref.template, "inline" if ref.inline else library.status(ref.template),
                                 ref.bind, ref.variables) for ref in c.prompt.templates]
        rng, n = random.Random(seed), c.execution.n
        if c.instantiation is not None:
            run = Instantiation.sample(c.instantiation, n, rng=rng)
        elif n is None:
            raise ValueError(f"{c.name}: how many inputs? Give execution.n (on the command line, --n)")
        else:
            run = [Instantiation.of({}, index=k) for k in range(n)]

        state = c.instantiation.state if c.instantiation else {}
        sets = {name: PersonaSet.load(e.file or e.set) for name, e in state.items() if isinstance(e, Personas)}
        method, inputs = c.prompt.sample or "crossed", []
        for k, one in enumerate(run):
            instant = one.instant
            people = [name for name in sets if isinstance(instant.get(name), Mapping)]
            payload = {name: instant[name]["id"] if name in people else value for name, value in instant.items()}
            chosen = None if method == "crossed" else (k % len(templates),) if method == "cycle" else \
                (rng.randrange(len(templates)),)
            inputs.append(RunInput(payload, instant, payload[people[0]] if len(sets) == 1 and people else None,
                                   [sets[name].row(instant[name]) for name in people], chosen))
        return templates, inputs, {}

    def input_items(self, payload: dict) -> list[dict]:
        """Long-form rows (label, value, entity, qualifier) for input_items; optional."""
        return []

    def reference_rows(self) -> dict[str, list[dict]]:
        """Rows for reference tables this task uses (e.g. its rubric), written when a run is prepared."""
        return {}

    def derived_rows(self, response: ChatResponse, response_id: str) -> dict[str, list[dict]]:
        """
        Extra table rows computed from a finished request's returned response, e.g. parsed
        scores: {table: [rows]}. Called once per request after its responses are written.
        """
        return {}

    # ---- validation ------------------------------------------------------------------------
    def validator(self):
        """
        A validation.Validator, or None: the generic checks the config names, and its custom
        validator's checks. A generic check that the custom validator has too takes its place
        (the config tunes it); the others come first.
        """
        from validation import Validator, generic_checks

        v = self.config.validation if self.config is not None else None
        if v is None:
            return None
        named = generic_checks(v.checks)
        custom = import_reference(v.custom_validator)().checks if v.custom_validator else []
        if not named and not custom:
            return None
        by_name, theirs = {c.name: c for c in named}, {c.name for c in custom}
        return Validator([c for c in named if c.name not in theirs] + [by_name.get(c.name, c) for c in custom],
                         name=self.name)

    def validation_expected(self, metadata: dict) -> Any:
        """What the validator compares a response against: the input's `expected` entry, or the whole input."""
        payload = metadata.get(INPUT_KEY)
        expected = self.config.validation.expected if self.config is not None and self.config.validation else None
        return payload.get(expected) if expected is not None and isinstance(payload, Mapping) else payload

    # ---- before a run ----------------------------------------------------------------------
    def run_warnings(self, root) -> list[str]:
        """An earlier run in the dataset with this seed and the same instantiation has the same inputs."""
        from inference.pipeline import list_runs

        x = self.config.execution
        if x.dataset is None or x.seed is None:
            return []
        clashes = []
        for run in list_runs(root, x.dataset):
            stored = run["config"]
            recorded = stored.get("task_config")
            if run["task"] != self.name or stored.get("seed") != x.seed:
                continue
            if recorded is None or recorded.get("instantiation_sha256") == self.config.instantiation_sha256:
                clashes.append(run["run_id"])           # (a run before task configs: its instants may be the same)
        if not clashes:
            return []
        return [f"dataset {x.dataset!r} already has run(s) with --seed {x.seed} and the same instantiation "
                f"({', '.join(clashes)}); this run will repeat their inputs. Omit --seed (a fresh one is chosen and "
                f"recorded) or use a different one."]

    # ---- a task that computes its inputs in Python -----------------------------------------
    default_template: Optional[str] = None  # the template when its settings name none
    persona_template: Optional[str] = None  # the template when they name none but give personas (takes `persona`)

    def sample_inputs(self, n: int, rng: random.Random) -> list[dict]:
        """A run's n inputs, computed in Python. A task that does not override it is sampled from its instantiation."""
        raise NotImplementedError(f"{type(self).__name__} computes no inputs; its config's instantiation samples them")

    def assign_personas(self, payloads: list[dict], people: PersonaSet, rng: random.Random,
                        mode: str) -> list[Optional[dict]]:
        """One persona (or None) per payload. Override to control the pairing, e.g. distinct personas per subject."""
        return people.sample(rng, len(payloads), mode)

    def template_variables(self, payload: dict, persona: Optional[dict]) -> dict[str, Any]:
        """
        Everything a prompt template may use for this input. Each template takes the
        variables it declares from this pool, so one task can serve templates with
        different needs (e.g. with and without `persona`).
        """
        return {"input": payload, "persona": persona}

    def options(self) -> dict:
        """The task's own settings, recorded with its run (runs.config.task_options) to rebuild it."""
        return {}

    @classmethod
    def from_options(cls, options: dict) -> "InferenceTask":
        """The task of an existing run, rebuilt from the options() recorded with it (to resume or collect it)."""
        return cls(**options)

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
