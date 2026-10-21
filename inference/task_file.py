"""
Task files: an application's tasks.yaml read into checked, typed TaskSpec objects.

    specs = load_task_file("applications/tarot/tasks.yaml")      # {name: TaskSpec}
    spec = specs["tarot_celtic_cross"]
    spec.prompt.templates[0].id, spec.model.provider, spec.instantiation.state["cards"]

    python -m inference.task_file check                          # every application's tasks.yaml
    python -m inference.task_file check applications/tarot/tasks.yaml

A file maps task names to tasks; the format is described in
applications/template/inference_task_reference.yaml. Loading
  1. reads the YAML strictly (instantiation.files.read_yaml): a duplicate key is an error, and
     dates stay as written;
  2. checks each task against inference/schemas/task.schema.yaml (its `instantiation` against
     instantiation/instantiation.schema.yaml): unknown keys and wrong types are errors;
  3. builds the TaskSpec; the instantiation is read by the `instantiation` package;
  4. checks what a schema cannot: the prompt templates, files, persona sets, classes and
     functions named exist, every variable a template needs is supplied, the state entries named
     exist, and every template renders: against a sample instant, and with every persona of a
     personas entry, so a persona field a template reads but a persona lacks is found here.
Every problem is reported at once, with where it is (task.key.path).

A TaskSpec says what the file says. Settings the file leaves out are None, so that whoever
runs the task can tell "not given" from a given value. Nothing runs from a TaskSpec yet.
"""
import argparse
import copy
import hashlib
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

import yaml
from jsonschema import Draft202012Validator
from referencing import Registry, Resource
from referencing.jsonschema import DRAFT202012

from instantiation import Generate, InstantiationSpec, Literal, OneOf, Personas, Steps, load_instantiation, probes
from instantiation import spec as instantiation_spec
from instantiation.files import file_sha256, import_reference, read_yaml, relative_key, schema_problems

REPO_ROOT = Path(__file__).resolve().parents[1]
TASK_NAME = "^[a-z][a-z0-9_]*$"
TASK_SCHEMA = yaml.safe_load((Path(__file__).resolve().parent / "schemas" / "task.schema.yaml").read_text(encoding="utf-8"))
TASK_VALIDATOR = Draft202012Validator(TASK_SCHEMA, registry=Registry().with_resource(
    "instantiation.schema.yaml", Resource(instantiation_spec.SCHEMA, specification=DRAFT202012)))


class TaskFileError(ValueError):
    """A task file that cannot be used; `problems` lists every reason, each with its location."""

    def __init__(self, path: Path, problems: list[str]):
        self.path, self.problems = path, problems
        super().__init__(f"{path}:\n" + "\n".join(f"  {p}" for p in problems))


# ---- what a task file holds -------------------------------------------------------------
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
    sample: Optional[str] = None
    retired_templates: tuple[str, ...] = ()


@dataclass(frozen=True)
class ModelSpec:
    provider: str
    name: Optional[str] = None
    parameters: Mapping[str, Any] = field(default_factory=dict)
    extra_body: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExecutionSpec:
    mode: Optional[str] = None
    workers: Optional[int] = None
    n: Optional[int] = None
    seed: Optional[int] = None
    retry_mode: Optional[str] = None
    poll_interval: Optional[float] = None
    dataset: Optional[str] = None
    name: Optional[str] = None


@dataclass(frozen=True)
class ValidationSpec:
    custom_validator: Optional[str] = None
    expected: Optional[str] = None
    checks: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    retries: Optional[int] = None


@dataclass(frozen=True)
class TaskSpec:
    name: str
    description: str
    engine: str
    prompt: PromptSpec
    model: ModelSpec
    execution: ExecutionSpec
    validation: ValidationSpec
    instantiation: Optional[InstantiationSpec] = None
    implemented_by: Optional[str] = None
    version: Optional[int] = None
    created_by: Optional[str] = None
    created_ts: Optional[str] = None
    tags: tuple[str, ...] = ()
    permissions: Any = None
    source: Optional[Path] = None             # the task file
    content: Mapping[str, Any] = field(default_factory=dict, repr=False)   # the task as written, instantiation file inlined
    references: Mapping[str, Any] = field(default_factory=dict, repr=False)  # hashes of what it names
    sha256: str = ""                          # identity: content + references; not comments or formatting


# ---- loading -----------------------------------------------------------------------------
def load_task_file(path: str | Path, check: bool = True) -> dict[str, TaskSpec]:
    """
    {name: TaskSpec} for every task in the file. check=False checks the structure only, not
    that what the tasks name exists (e.g. for the reference file, whose names are placeholders).
    Raises TaskFileError listing every problem.
    """
    path = Path(path).resolve()
    try:
        data = read_yaml(path)
    except yaml.YAMLError as e:
        raise TaskFileError(path, [f"not valid YAML: {e}"]) from e
    if not isinstance(data, dict) or not data:
        raise TaskFileError(path, ["the file must map task names to tasks"])
    problems, specs = [], {}
    for name, entry in data.items():
        if not isinstance(name, str) or not re.match(TASK_NAME, name):
            problems.append(f"{name!r}: a task name is lowercase letters, digits and underscores")
            continue
        found = schema_problems(TASK_VALIDATOR, entry, name)
        if found:
            problems += found
            continue
        spec, found = _build(name, entry, path, check)
        problems += found
        specs[name] = spec
    if problems:
        raise TaskFileError(path, problems)
    return specs


def _build(name: str, entry: dict, path: Path, check: bool) -> tuple[TaskSpec, list[str]]:
    problems: list[str] = []
    here = path.parent                         # relative paths are relative to the task file's folder
    content = copy.deepcopy(entry)
    references: dict[str, dict[str, str]] = {"templates": {}, "files": {}, "persona_sets": {}}

    instantiation, written = load_instantiation(entry.get("instantiation"), here, f"{name}.instantiation", problems,
                                                references, check)
    if written is not None:
        content["instantiation"] = written     # an instantiation file's content, read in

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
    prompt = PromptSpec(tuple(templates), p.get("sample"), tuple(p.get("retired_templates") or ()))

    m, x, v = entry["model"], entry.get("execution") or {}, entry.get("validation") or {}
    spec = TaskSpec(
        name=name, description=entry["description"], engine=entry["engine"], prompt=prompt,
        model=ModelSpec(m["provider"], m.get("name"), m.get("parameters") or {}, m.get("extra_body") or {}),
        execution=ExecutionSpec(**{k: x.get(k) for k in ExecutionSpec.__dataclass_fields__}),
        validation=ValidationSpec(v.get("custom_validator"), v.get("expected"), v.get("checks") or {}, v.get("retries")),
        instantiation=instantiation, implemented_by=entry.get("implemented_by"), version=entry.get("version"),
        created_by=entry.get("created_by"), created_ts=entry.get("created_ts"), tags=tuple(entry.get("tags") or ()),
        permissions=entry.get("permissions"), source=path)

    if check:
        spec = _check(spec, here, problems, references)
    references = {k: v for k, v in references.items() if v}
    sha = hashlib.sha256(json.dumps({"task": content, "references": references}, sort_keys=True,
                                    ensure_ascii=False, default=str).encode()).hexdigest()
    return TaskSpec(**{**spec.__dict__, "content": content, "references": references, "sha256": sha}), problems


def _check(spec: TaskSpec, here: Path, problems: list[str], references: dict) -> TaskSpec:
    """The checks a schema cannot make. Returns the spec with each template's PromptTemplate attached."""
    from llm.providers import available_providers
    from prompts import PromptError, PromptTemplate, default_library

    name = spec.name
    state = dict(spec.instantiation.state) if spec.instantiation else {}
    instantiation_read = not problems           # so far: problems reading the instantiation or variables files

    # templates, and the variables they need
    library, templates, seen, ready = default_library(), [], set(), []
    for k, ref in enumerate(spec.prompt.templates):
        where = f"{name}.prompt.templates[{k}]"
        if ref.id in seen:
            problems.append(f"{where}: {ref.id} is listed twice")
        seen.add(ref.id)
        try:
            template = PromptTemplate.inline(ref.id[:-len("_v1")], ref.inline, source=spec.source) if ref.inline \
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
        supplied = set(ref.variables) | set(ref.bind) | set(state)
        for variable in template.required:
            if variable not in supplied:
                problems.append(f"{where}: {ref.id} needs {variable!r}, which neither the instantiation nor "
                                f"`variables` supplies")
        templates.append(TemplateRef(**{**ref.__dict__, "template": template}))
        if len(problems) == before:
            ready.append((where, templates[-1]))
    for retired in spec.prompt.retired_templates:
        try:
            library.resolve(retired)
        except KeyError:
            problems.append(f"{name}.prompt.retired_templates: no prompt template {retired!r}")

    if instantiation_read:                      # every template renders, e.g. with every persona of a set
        _check_rendering(spec, ready, problems)

    # model, validation, implementation
    if spec.model.provider not in available_providers():
        problems.append(f"{name}.model.provider: {spec.model.provider!r} is not a registered provider "
                        f"({', '.join(available_providers())})")
    if spec.validation.expected is not None and spec.validation.expected not in state:
        problems.append(f"{name}.validation.expected: no state entry {spec.validation.expected!r}")
    for kind, reference in (("validation.custom_validator", spec.validation.custom_validator),
                            ("implemented_by", spec.implemented_by)):
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
    json_fields = spec.validation.checks.get("json_fields")
    if json_fields:
        schemas = here / "schemas.yaml"
        known = (read_yaml(schemas) or {}).get("schemas", {}) if schemas.is_file() else {}
        if json_fields["schema"] not in known:
            problems.append(f"{name}.validation.checks.json_fields.schema: no schema "
                            f"{json_fields['schema']!r} in {schemas.name}")
    return TaskSpec(**{**spec.__dict__, "prompt": PromptSpec(tuple(templates), spec.prompt.sample,
                                                             spec.prompt.retired_templates)})


def _check_rendering(spec: TaskSpec, ready: list, problems: list[str]) -> None:
    """Render each template against sample instants: once, and once per other candidate and persona."""
    from prompts import PromptError

    try:
        instants = list(probes(spec.instantiation)) if spec.instantiation else [("its fixed variables", {})]
    except Exception as err:                    # e.g. a generate function that fails: report it, don't crash
        problems.append(f"{spec.name}.instantiation: could not make a sample instant: {type(err).__name__}: {err}")
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


def describe(spec: TaskSpec) -> list[str]:
    """What a task file says about a task, line by line, for `check`."""
    lines = [f"  {spec.name}  OK  sha256 {spec.sha256[:12]}",
             f"    engine          {spec.engine}" + (f"  ({spec.implemented_by})" if spec.implemented_by else "")]
    for ref in spec.prompt.templates:
        where = "inline" if ref.inline else _shown(getattr(ref.template, "source", None))
        lines.append(f"    template        {ref.id}  {where}")
    if spec.prompt.retired_templates:
        lines.append(f"    retired         {', '.join(spec.prompt.retired_templates)}")
    if spec.instantiation:
        for entry_name, e in spec.instantiation.state.items():
            if isinstance(e, Steps):
                what = (f"steps: {len(e.names)} steps x {len(e.states)} states"
                        f"{' (may be negative)' if e.negative_polarity else ''}, "
                        f"{'with' if e.replacement else 'without'} replacement, {len(e.fixed)} fixed"
                        f"{'; from ' + _shown(e.file) if e.file else ''}")
            elif isinstance(e, OneOf):
                what = f"one_of {len(e.candidates)} candidates, {spec.instantiation.method(entry_name)}"
            elif isinstance(e, Personas):
                what = f"personas from {e.set}, {spec.instantiation.method(entry_name)}"
            elif isinstance(e, Generate):
                what = f"generate with {e.using}"
            else:
                what = f"literal {json.dumps(e.value, ensure_ascii=False)[:60]}"
            lines.append(f"    state.{entry_name:<10}{what}")
    else:
        lines.append("    instantiation   none")
    parameters = " ".join(f"{k}={v}" for k, v in spec.model.parameters.items())
    lines.append(f"    model           {spec.model.provider} / {spec.model.name or '(provider default)'}  {parameters}")
    given = {k: v for k, v in spec.execution.__dict__.items() if v is not None}
    lines.append(f"    execution       {' '.join(f'{k}={v}' for k, v in given.items()) or '(defaults)'}")
    v = spec.validation
    lines.append(f"    validation      {v.custom_validator or 'no custom validator'}; expected="
                 f"{v.expected or 'the whole instant'}; checks: {', '.join(v.checks) or 'none'}; retries="
                 f"{v.retries if v.retries is not None else '(default)'}")
    return lines


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m inference.task_file", description=__doc__,
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
            specs = load_task_file(path, check=not args.structure_only)
        except TaskFileError as e:
            failed += 1
            for problem in e.problems:
                print(f"  PROBLEM  {problem}")
            continue
        for spec in specs.values():
            print("\n".join(describe(spec)))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
