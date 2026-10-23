"""
InferenceTaskConfig: everything a run of an inference task is, in one checked object: what
its inputs are (the instantiation), its prompt templates, the model, how the run executes, and
how responses are validated. An application's tasks.yaml writes it down; the command line and
Python edit it; a run records it, and resume and collect read it back.

    config = InferenceTaskConfig.from_file("applications/tarot/tasks.yaml", "tarot_celtic_cross",
                                           edits={"execution.n": 20})        # a task of a file, edited
    config = InferenceTaskConfig.from_entry("my_task", {...})                 # a task written in code
    config = InferenceTaskConfig.from_run(store.run())                       # what a run recorded
    config.instantiation.state["cards"], config.model.provider, config.execution.n

    load_task_file("applications/tarot/tasks.yaml")                         # {name: config}, every task
    python -m inference.task_config check                                    # every application's tasks.yaml
    python -m inference.task_config check applications/tarot/tasks.yaml

The format is described in applications/template/inference_task_reference.yaml. Loading
  1. reads the YAML strictly (spec_files.read_yaml): a duplicate key is an error, and
     dates stay as written;
  2. applies the edits ({dotted.path: value}, e.g. {"execution.n": 20}; the command line's
     settings are edits too), so an edited task is checked like a written one;
  3. checks the task against inference/schemas/task.schema.yaml (its `instantiation` against
     instantiation/instantiation.schema.yaml): unknown keys and wrong types are errors;
  4. builds the config; the instantiation is read by the `instantiation` package;
  5. checks what a schema cannot: the prompt templates, files, persona sets, classes and
     functions named exist, every variable a template needs is supplied, the state entries named
     exist, and every template renders: against a sample instant, and with every persona of a
     personas entry, so a persona field a template reads but a persona lacks is found here.
Every problem is reported at once, with where it is (task.key.path).

Settings the task leaves out take their defaults (execution: live, 4 workers, retry_mode live,
polling every 60 s; validation: 2 retries). `content` keeps the task as written, edits applied.
"""
import argparse
import copy
import hashlib
import json
import re
import sys
from dataclasses import asdict, dataclass, field, is_dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Optional

import yaml
from jsonschema import Draft202012Validator
from referencing import Registry, Resource
from referencing.jsonschema import DRAFT202012

from instantiation import Generate, InstantiationSpec, OneOf, Personas, Steps, load_instantiation, probes
from instantiation import spec as instantiation_spec
from spec_files import file_sha256, import_reference, read_yaml, relative_key, schema_problems

REPO_ROOT = Path(__file__).resolve().parents[1]
TASK_NAME = "^[a-z][a-z0-9_]*$"
TASK_SCHEMA = yaml.safe_load((Path(__file__).resolve().parent / "schemas" / "task.schema.yaml").read_text(encoding="utf-8"))
TASK_VALIDATOR = Draft202012Validator(TASK_SCHEMA, registry=Registry().with_resource(
    "instantiation.schema.yaml", Resource(instantiation_spec.SCHEMA, specification=DRAFT202012)))


class TaskConfigError(ValueError):
    """A task that cannot be used; `problems` lists every reason, each with its location."""

    def __init__(self, source: Any, problems: list[str]):
        self.source, self.problems = source, problems
        super().__init__(f"{source}:\n" + "\n".join(f"  {p}" for p in problems))


# ---- the parts of a config ------------------------------------------------------------------
@dataclass(frozen=True)
class TemplateRef:
    """A prompt template of the task: from the library (`id`), or written inline (`inline`, id <name>_v1)."""
    id: str
    inline: Optional[str] = None
    variables: Mapping[str, Any] = field(default_factory=dict)
    bind: Mapping[str, str] = field(default_factory=dict)
    template: Any = None                      # the PromptTemplate, once checked


@dataclass(frozen=True)
class PromptSpec:
    templates: tuple[TemplateRef, ...]
    sample: Optional[str] = None              # None: crossed, every instant with every template
    retired_templates: tuple[str, ...] = ()
    system: Optional[str] = None              # replaces every template's system prompt


@dataclass(frozen=True)
class ModelSpec:
    provider: str
    name: Optional[str] = None                # None: the provider's default model
    parameters: Mapping[str, Any] = field(default_factory=dict)
    extra_body: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExecutionSpec:
    mode: str = "live"                        # live | batch | dry_run
    workers: int = 4                          # live mode: concurrent requests
    n: Optional[int] = None                   # None: the instantiation's own number (given instants, exhaustive)
    seed: Optional[int] = None                # None: chosen when the run is prepared, and recorded
    retry_mode: str = "live"                  # batch runs: regenerate invalid responses live | batch
    poll_interval: float = 60.0
    dataset: Optional[str] = None
    name: Optional[str] = None


@dataclass(frozen=True)
class ValidationSpec:
    custom_validator: Optional[str] = None
    expected: Optional[str] = None            # None: the whole instant
    checks: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    retries: int = 2


@dataclass(frozen=True)
class InferenceTaskConfig:
    name: str
    description: str
    engine: str
    prompt: PromptSpec
    model: ModelSpec
    execution: ExecutionSpec
    validation: Optional[ValidationSpec]      # None: responses are not validated
    instantiation: Optional[InstantiationSpec] = None
    implemented_by: Optional[str] = None
    version: Optional[int] = None
    created_by: Optional[str] = None
    created_ts: Optional[str] = None
    tags: tuple[str, ...] = ()
    permissions: Any = None
    source: Optional[Path] = None             # the task file, if it came from one
    content: Mapping[str, Any] = field(default_factory=dict, repr=False)   # the task as written, edited; files inlined
    references: Mapping[str, Any] = field(default_factory=dict, repr=False)  # hashes of what it names
    sha256: str = ""                          # identity: content + references; not comments or formatting
    instantiation_sha256: str = ""            # the instantiation's own: the same, with the same seed, gives the same instants
    edits: Mapping[str, Any] = field(default_factory=dict)   # what was changed from the file, {dotted.path: value}

    # ---- where a config comes from -----------------------------------------------------------
    @classmethod
    def from_file(cls, path: str | Path, name: Optional[str] = None, edits: Optional[Mapping[str, Any]] = None,
                  check: bool = True) -> "InferenceTaskConfig":
        """
        A task of a task file: `name`, or the file's only task. `edits` change it before it is checked,
        {dotted.path: value}, e.g. {"execution.n": 20, "instantiation.state.persona": {"personas": "x"}}.
        Relative paths in it are relative to the file's folder. Raises TaskConfigError.
        """
        path = Path(path).resolve()
        data = _read(path)
        if name is None:
            if len(data) != 1:
                raise TaskConfigError(path, [f"the file has {len(data)} tasks; name one of them ({', '.join(data)})"])
            name = next(iter(data))
        if name not in data:
            raise TaskConfigError(path, [f"no task {name!r} in the file; it has {', '.join(map(str, data))}"])
        return cls._load(name, data[name], path.parent, path, edits, check)

    @classmethod
    def from_entry(cls, name: str, entry: Mapping[str, Any], here: Optional[Path] = None,
                   edits: Optional[Mapping[str, Any]] = None, check: bool = True) -> "InferenceTaskConfig":
        """A task written as a mapping, in code. Relative paths in it are relative to `here` (default: the working folder)."""
        return cls._load(name, entry, Path(here) if here is not None else Path.cwd(), None, edits, check)

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "InferenceTaskConfig":
        """
        The config a run recorded (record()), to resume or collect the run: everything but the
        instantiation, whose files are not read again (the run holds its instants and prompts).
        """
        source = Path(record["source"]) if record.get("source") else None
        if source is not None and not source.is_absolute():
            source = REPO_ROOT / source
        here = source.parent if source is not None else Path.cwd()
        config, _ = _build(record["name"], {**record["content"], "instantiation": None}, here, source, check=False)
        return replace(config, content=record["content"], references=record["references"], sha256=record["sha256"],
                       instantiation_sha256=record.get("instantiation_sha256", ""), edits=record.get("edits") or {})

    @classmethod
    def from_run(cls, run: Mapping[str, Any]) -> "InferenceTaskConfig":
        """
        The config of a run, from its runs row (in the current shape, RunStore.run()): the config it
        recorded, or, for a run that stored flat settings (runs before task configs, and runs of
        tasks that give their settings in Python), the config those settings make.
        """
        stored = run["config"]
        if "task_config" in stored:
            return cls.from_record(stored["task_config"])
        return cls.from_settings(run["task"], stored)

    @classmethod
    def from_settings(cls, name: str, settings: Any) -> "InferenceTaskConfig":
        """
        The config that flat settings make: a pipeline.GenerationConfig, or the runs.config it was
        stored as (any schema version). Its persona settings become the instantiation's `persona`
        entry. Checked for structure only: its templates take their variables from Python.
        """
        s = asdict(settings) if is_dataclass(settings) else dict(settings)
        templates = s.get("templates") or ([s["template"]] if s.get("template") else [])
        entry: dict[str, Any] = {
            "description": f"{name}, run with settings given in Python",
            "engine": "inference",
            "prompt": {"templates": [{"template": t} for t in templates], **({"system": s["system"]} if s.get("system") else {})},
            "model": {"provider": s["provider"], "name": s.get("model"), "parameters": dict(s.get("params") or {}),
                      "extra_body": dict(s.get("extra_body") or {})},
            "execution": {k: v for k, v in {
                "mode": s.get("mode"), "workers": s.get("max_workers"), "n": s.get("n"), "seed": s.get("seed"),
                "retry_mode": s.get("retry_mode"), "poll_interval": s.get("poll_interval"), "dataset": s.get("dataset"),
                "name": s.get("name")}.items() if v is not None},
            "validation": {"retries": s.get("validation_retries", 2)} if s.get("validate", True) else None,
        }
        if s.get("persona_set"):
            entry["instantiation"] = {"state": {"persona": {
                "personas": s["persona_set"], "sample": s.get("persona_sampling") or "random",
                **({"select": [str(i) for i in s["persona_ids"]]} if s.get("persona_ids") else {})}}}
        return cls.from_entry(name, entry, check=False)

    @classmethod
    def _load(cls, name: str, entry: Any, here: Path, source: Optional[Path], edits: Optional[Mapping[str, Any]],
              check: bool) -> "InferenceTaskConfig":
        where = source if source is not None else f"{name} (in code)"
        if edits:
            try:
                entry = edited(entry, edits, here)
            except (ValueError, yaml.YAMLError) as err:
                raise TaskConfigError(where, [f"{name}: {err}"]) from err
        config, problems = _load(name, entry, here, source, check)
        if problems:
            raise TaskConfigError(where, problems)
        return replace(config, edits=dict(edits or {}))

    # ---- what a run records -------------------------------------------------------------------
    def record(self) -> dict[str, Any]:
        """What a run stores of its config (runs.config.task_config); from_record reads it back."""
        source = Path(self.source) if self.source else None
        if source is not None and source.is_relative_to(REPO_ROOT):
            source = source.relative_to(REPO_ROOT)
        return {"name": self.name, "source": source.as_posix() if source else None, "sha256": self.sha256,
                "instantiation_sha256": self.instantiation_sha256, "edits": dict(self.edits),
                "content": self.content, "references": self.references}


# ---- reading -------------------------------------------------------------------------------
def load_task_file(path: str | Path, check: bool = True) -> dict[str, InferenceTaskConfig]:
    """
    {name: config} for every task in the file. check=False checks the structure only, not that
    what the tasks name exists (e.g. for the reference file, whose names are placeholders).
    Raises TaskConfigError listing every problem.
    """
    path = Path(path).resolve()
    data = _read(path)
    problems, configs = [], {}
    for name, entry in data.items():
        if not isinstance(name, str) or not re.match(TASK_NAME, name):
            problems.append(f"{name!r}: a task name is lowercase letters, digits and underscores")
            continue
        config, found = _load(name, entry, path.parent, path, check)
        problems += found
        if config is not None:
            configs[name] = config
    if problems:
        raise TaskConfigError(path, problems)
    return configs


def edited(entry: Mapping[str, Any], edits: Mapping[str, Any], here: Path) -> dict:
    """
    A task with edits applied, {dotted.path: value}: each sets the value at that path, making the
    mappings on the way as needed. An edit inside an instantiation kept in a file of its own reads
    the file in first (here: the task's folder). Raises ValueError for a path through a value
    that is not a mapping.
    """
    entry = copy.deepcopy(dict(entry))
    for dotted, value in edits.items():
        keys = dotted.split(".")
        if keys[0] == "instantiation" and len(keys) > 1 and isinstance(entry.get("instantiation"), str):
            entry["instantiation"] = read_yaml(here / entry["instantiation"])
        node = entry
        for k, key in enumerate(keys[:-1]):
            if node.get(key) is None:
                node[key] = {}
            elif not isinstance(node[key], dict):
                raise ValueError(f"cannot set {dotted}: {'.'.join(keys[:k + 1])} is {node[key]!r}, not a mapping")
            node = node[key]
        node[keys[-1]] = copy.deepcopy(value)
    return entry


def parse_edit(text: str) -> tuple[str, Any]:
    """A command-line edit, "dotted.path=value": the value is read as YAML (20, false, [a, b], {k: v})."""
    dotted, sep, value = text.partition("=")
    if not sep or not dotted.strip():
        raise ValueError(f"an edit is dotted.path=value, e.g. execution.n=20; not {text!r}")
    return dotted.strip(), yaml.safe_load(value) if value.strip() else None


def _read(path: Path) -> dict:
    try:
        data = read_yaml(path)
    except yaml.YAMLError as e:
        raise TaskConfigError(path, [f"not valid YAML: {e}"]) from e
    except FileNotFoundError as e:
        raise TaskConfigError(path, ["no such file"]) from e
    if not isinstance(data, dict) or not data:
        raise TaskConfigError(path, ["the file must map task names to tasks"])
    return data


def _load(name: str, entry: Any, here: Path, source: Optional[Path],
          check: bool) -> tuple[Optional[InferenceTaskConfig], list[str]]:
    found = schema_problems(TASK_VALIDATOR, entry, name)
    if found:
        return None, found
    return _build(name, entry, here, source, check)


def _sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()


def _given(section: Mapping[str, Any], spec: type) -> dict:
    """The settings of a section that are given (not left out, not null); the dataclass has the defaults."""
    return {k: section[k] for k in spec.__dataclass_fields__ if section.get(k) is not None}


def _build(name: str, entry: dict, here: Path, source: Optional[Path],
           check: bool) -> tuple[InferenceTaskConfig, list[str]]:
    problems: list[str] = []                   # relative paths are relative to `here`
    content = copy.deepcopy(entry)
    references: dict[str, dict[str, str]] = {"templates": {}, "files": {}, "persona_sets": {}}

    own: dict[str, dict[str, str]] = {}        # what the instantiation names, for its own hash
    instantiation, written = load_instantiation(entry.get("instantiation"), here, f"{name}.instantiation", problems,
                                                own, check)
    if written is not None:
        content["instantiation"] = written     # an instantiation file's content, read in
    for kind, found in own.items():
        references[kind].update(found)

    p = entry["prompt"]
    templates = []
    for k, t in enumerate(p["templates"], 1):
        variables = t.get("variables") or {}
        if isinstance(variables, str):
            file = (here / variables).resolve()
            if not file.is_file():
                problems.append(f"{name}.prompt.templates[{k - 1}].variables: no file {variables!r}")
                variables = {}
            else:
                variables = read_yaml(file) or {}
                references["files"][relative_key(file, here)] = file_sha256(file)
                if not isinstance(variables, dict):
                    problems.append(f"{name}.prompt.templates[{k - 1}].variables: {file.name} must map names to values")
                    variables = {}
        tid = t["template"] if "template" in t else f"{t.get('name', f'{name}_inline_{k}')}_v1"
        templates.append(TemplateRef(id=tid, inline=t.get("inline"), variables=variables, bind=t.get("bind") or {}))
    prompt = PromptSpec(tuple(templates), p.get("sample"), tuple(p.get("retired_templates") or ()), p.get("system"))

    m, v = entry["model"], entry.get("validation", {})
    config = InferenceTaskConfig(
        name=name, description=entry["description"], engine=entry["engine"], prompt=prompt,
        model=ModelSpec(m["provider"], m.get("name"), m.get("parameters") or {}, m.get("extra_body") or {}),
        execution=ExecutionSpec(**_given(entry.get("execution") or {}, ExecutionSpec)),
        validation=None if v is None else ValidationSpec(**_given(v, ValidationSpec)),
        instantiation=instantiation, implemented_by=entry.get("implemented_by"), version=entry.get("version"),
        created_by=entry.get("created_by"), created_ts=entry.get("created_ts"), tags=tuple(entry.get("tags") or ()),
        permissions=entry.get("permissions"), source=source)

    if check:
        config = _check(config, problems, references)
    references = {k: v for k, v in references.items() if v}
    sha = _sha256({"task": content, "references": references})
    own_sha = "" if instantiation is None else _sha256({"instantiation": content["instantiation"],
                                                        "references": {k: v for k, v in own.items() if v}})
    return replace(config, content=content, references=references, sha256=sha, instantiation_sha256=own_sha), problems


def _check(config: InferenceTaskConfig, problems: list[str], references: dict) -> InferenceTaskConfig:
    """The checks a schema cannot make. Returns the config with each template's PromptTemplate attached."""
    from llm.providers import available_providers
    from prompts import PromptError, PromptTemplate, default_library

    name = config.name
    state = set(config.instantiation.entries) if config.instantiation else set()   # what every instant has
    instantiation_read = not problems           # so far: problems reading the instantiation or variables files

    # templates, and the variables they need
    library, templates, seen, ready = default_library(), [], set(), []
    for k, ref in enumerate(config.prompt.templates):
        where = f"{name}.prompt.templates[{k}]"
        if ref.id in seen:
            problems.append(f"{where}: {ref.id} is listed twice")
        seen.add(ref.id)
        try:
            template = PromptTemplate.inline(ref.id[:-len("_v1")], ref.inline, source=config.source) if ref.inline \
                else library.get(ref.id)
        except (KeyError, PromptError) as err:
            problems.append(f"{where}: {err.args[0] if err.args else err}")
            templates.append(ref)
            continue
        references["templates"][ref.id] = template.sha256
        before = len(problems)
        declared = set(template.variables)
        for variable in sorted(set(ref.variables) - declared):
            problems.append(f"{where}.variables.{variable}: {ref.id} takes no variable {variable!r}")
        for variable, entry_name in ref.bind.items():
            if variable not in declared:
                problems.append(f"{where}.bind.{variable}: {ref.id} takes no variable {variable!r}")
            if entry_name not in state:
                problems.append(f"{where}.bind.{variable}: no state entry {entry_name!r}")
        supplied = set(ref.variables) | set(ref.bind) | state
        for variable in template.required:
            if variable not in supplied:
                problems.append(f"{where}: {ref.id} needs {variable!r}, which neither the instantiation nor "
                                f"`variables` supplies")
        templates.append(TemplateRef(**{**ref.__dict__, "template": template}))
        if len(problems) == before:
            ready.append((where, templates[-1]))
    for retired in config.prompt.retired_templates:
        try:
            library.resolve(retired)
        except KeyError:
            problems.append(f"{name}.prompt.retired_templates: no prompt template {retired!r}")

    if instantiation_read:                      # every template renders, e.g. with every persona of a set
        _check_rendering(config, ready, problems)

    # model, validation, implementation
    if config.model.provider not in available_providers():
        problems.append(f"{name}.model.provider: {config.model.provider!r} is not a registered provider "
                        f"({', '.join(available_providers())})")
    v = config.validation or ValidationSpec()
    if v.expected is not None and v.expected not in state:
        problems.append(f"{name}.validation.expected: no state entry {v.expected!r}")
    for kind, reference in (("validation.custom_validator", v.custom_validator), ("implemented_by", config.implemented_by)):
        if reference is None:
            continue
        try:
            target = import_reference(reference)
        except (ImportError, AttributeError) as err:
            problems.append(f"{name}.{kind}: cannot import {reference}: {err}")
            continue
        if kind == "implemented_by":
            from inference.task import InferenceTask
            if not (isinstance(target, type) and issubclass(target, InferenceTask)):
                problems.append(f"{name}.implemented_by: {reference} is not an InferenceTask subclass")
            elif getattr(target, "name", name) != name:
                problems.append(f"{name}.implemented_by: {reference} is the task {target.name!r}, not {name!r}")
        elif not callable(target):
            problems.append(f"{name}.validation.custom_validator: {reference} is not callable")
    return replace(config, prompt=replace(config.prompt, templates=tuple(templates)))


def _check_rendering(config: InferenceTaskConfig, ready: list, problems: list[str]) -> None:
    """Render each template against sample instants: once, and once per other candidate and persona."""
    from prompts import PromptError

    try:
        instants = list(probes(config.instantiation)) if config.instantiation else [("its fixed variables", {})]
    except Exception as err:                    # e.g. a generate function that fails: report it, don't crash
        problems.append(f"{config.name}.instantiation: could not make a sample instant: {type(err).__name__}: {err}")
        return
    for where, ref in ready:
        template = ref.template
        failures = []
        for label, instant in instants:
            values = {v: instant[ref.bind.get(v, v)] for v in template.variables if ref.bind.get(v, v) in instant}
            try:
                template.render({**values, **ref.variables})
            except PromptError as err:
                failures.append((label, str(err).removeprefix(f"{template.id}: ")))
        if failures:
            label, error = failures[0]
            share = "" if len(instants) == 1 else f" in {len(failures)} of {len(instants)} sample instants, e.g."
            problems.append(f"{where}: {ref.id} does not render{share} with {label}: {error}")


# ---- command line ------------------------------------------------------------------------
def _shown(path: Optional[Path]) -> str:
    if path is None:
        return "-"
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def describe(config: InferenceTaskConfig) -> list[str]:
    """What a config says, line by line, for `check`."""
    lines = [f"  {config.name}  OK  sha256 {config.sha256[:12]}",
             f"    engine          {config.engine}" + (f"  ({config.implemented_by})" if config.implemented_by else "")]
    for ref in config.prompt.templates:
        where = "inline" if ref.inline else _shown(getattr(ref.template, "source", None))
        lines.append(f"    template        {ref.id}  {where}")
    if config.prompt.retired_templates:
        lines.append(f"    retired         {', '.join(config.prompt.retired_templates)}")
    if config.prompt.system is not None:
        lines.append(f"    system          {json.dumps(config.prompt.system, ensure_ascii=False)[:60]}")
    inst = config.instantiation
    if inst and inst.instants is not None:
        source = _shown(inst.instants_file) if inst.instants_file else "inline"
        lines.append(f"    instants        {len(inst.instants)} given ({source})"
                     f"{', checked against the state' if inst.state else ''}")
    if inst:
        for entry_name, e in inst.state.items():
            if isinstance(e, Steps):
                what = (f"steps: {len(e.names)} steps x {len(e.states)} states"
                        f"{' (may be negative)' if e.negative_polarity else ''}, "
                        f"{'with' if e.replacement else 'without'} replacement, {len(e.fixed)} fixed"
                        f"{'; from ' + _shown(e.file) if e.file else ''}")
            elif isinstance(e, OneOf):
                what = f"one_of {len(e.candidates)} candidates, {inst.method(entry_name)}"
            elif isinstance(e, Personas):
                what = f"personas from {_shown(e.file) if e.file else e.set}, {inst.method(entry_name)}"
            elif isinstance(e, Generate):
                what = f"generate with {e.using}"
            else:
                what = f"literal {json.dumps(e.value, ensure_ascii=False)[:60]}"
            lines.append(f"    state.{entry_name:<10}{what}")
    else:
        lines.append("    instantiation   none")
    parameters = " ".join(f"{k}={v}" for k, v in config.model.parameters.items())
    lines.append(f"    model           {config.model.provider} / {config.model.name or '(provider default)'}  {parameters}")
    given = {k: v for k, v in config.execution.__dict__.items() if v is not None}
    lines.append(f"    execution       {' '.join(f'{k}={v}' for k, v in given.items())}")
    v = config.validation
    lines.append("    validation      none" if v is None else
                 f"    validation      {v.custom_validator or 'no custom validator'}; expected="
                 f"{v.expected or 'the whole instant'}; checks: {', '.join(v.checks) or 'none'}; retries={v.retries}")
    return lines


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m inference.task_config", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("check", help="check task files and say what they define")
    c.add_argument("files", nargs="*", type=Path, help="task files (default: every application's tasks.yaml)")
    c.add_argument("--structure-only", action="store_true",
                   help="check the structure, not that what the tasks name exists (e.g. the reference)")
    args = parser.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    if not args.files:
        from applications import task_files
        args.files = task_files()
    failed = 0
    for path in args.files:
        print(_shown(Path(path)))
        try:
            configs = load_task_file(path, check=not args.structure_only)
        except TaskConfigError as e:
            failed += 1
            for problem in e.problems:
                print(f"  PROBLEM  {problem}")
            continue
        for config in configs.values():
            print("\n".join(describe(config)))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
