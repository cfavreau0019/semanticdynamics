"""
Task files: an application's tasks.yaml read into checked, typed TaskSpec objects.

    specs = load_task_file("applications/tarot/tasks.yaml")      # {name: TaskSpec}
    spec = specs["tarot_celtic_cross"]
    spec.prompt.templates[0].id, spec.model.provider, spec.instantiation.state["cards"]

    python -m inference.task_file check                          # every application's tasks.yaml
    python -m inference.task_file check applications/tarot/tasks.yaml

A file maps task names to tasks; the format is described in
applications/template/inference_task_reference.yaml. Loading
  1. reads the YAML strictly: a duplicate key is an error (plain YAML keeps the last one
     silently), and dates stay as written rather than becoming date objects;
  2. checks each task against inference/schemas/task.schema.yaml (and the instantiation
     against instantiation.schema.yaml): unknown keys and wrong types are errors;
  3. builds the TaskSpec;
  4. checks what a schema cannot: the prompt templates, files, persona sets, classes and
     functions named exist, every variable a template needs is supplied, and the state
     entries named exist.
Every problem is reported at once, with where it is (task.key.path).

A TaskSpec says what the file says. Settings the file leaves out are None, so that whoever
runs the task can tell "not given" from a given value. Nothing runs from a TaskSpec yet.
"""
import argparse
import copy
import hashlib
import importlib
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

SCHEMAS_DIR = Path(__file__).resolve().parent / "schemas"
REPO_ROOT = Path(__file__).resolve().parents[1]
TASK_NAME = "^[a-z][a-z0-9_]*$"


class TaskFileError(ValueError):
    """A task file that cannot be used; `problems` lists every reason, each with its location."""

    def __init__(self, path: Path, problems: list[str]):
        self.path, self.problems = path, problems
        super().__init__(f"{path}:\n" + "\n".join(f"  {p}" for p in problems))


# ---- reading YAML strictly --------------------------------------------------------------
class _StrictLoader(yaml.SafeLoader):
    """SafeLoader that refuses duplicate keys and keeps timestamps as the strings written."""

    def construct_mapping(self, node, deep=False):
        if isinstance(node, yaml.MappingNode):
            self.flatten_mapping(node)
            seen = []
            for key_node, _ in node.value:
                key = self.construct_object(key_node, deep=True)
                if key in seen:
                    raise yaml.constructor.ConstructorError(
                        "while reading a mapping", node.start_mark, f"found duplicate key {key!r}", key_node.start_mark)
                seen.append(key)
        return super().construct_mapping(node, deep=deep)


_StrictLoader.yaml_implicit_resolvers = {
    first: [(tag, regexp) for tag, regexp in resolvers if tag != "tag:yaml.org,2002:timestamp"]
    for first, resolvers in yaml.SafeLoader.yaml_implicit_resolvers.items()}


def read_yaml(path: str | Path) -> Any:
    """A YAML (or JSON) file read strictly: see _StrictLoader."""
    with open(path, encoding="utf-8") as f:
        return yaml.load(f, Loader=_StrictLoader)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _key(path: Path, here: Path) -> str:
    """How a referenced file is named in a task's hash: relative to the task file, so that the same
    task hashes the same wherever the repository is."""
    try:
        return path.relative_to(here).as_posix()
    except ValueError:
        return path.as_posix()


# ---- schemas ----------------------------------------------------------------------------
def _schema(name: str) -> dict:
    return yaml.safe_load((SCHEMAS_DIR / name).read_text(encoding="utf-8"))


INSTANTIATION_SCHEMA = _schema("instantiation.schema.yaml")
TASK_SCHEMA = _schema("task.schema.yaml")
_REGISTRY = Registry().with_resource("instantiation.schema.yaml",
                                     Resource(INSTANTIATION_SCHEMA, specification=DRAFT202012))
TASK_VALIDATOR = Draft202012Validator(TASK_SCHEMA, registry=_REGISTRY)
INSTANTIATION_VALIDATOR = Draft202012Validator(INSTANTIATION_SCHEMA)


def _where(task: str, path) -> str:
    out = task
    for part in path:
        out += f"[{part}]" if isinstance(part, int) else f".{part}"
    return out


def _deepest(error):
    """
    The most specific error under one that has alternatives (oneOf). The alternative that went
    deepest into the value is the one meant: an entry with a stray key is reported at that key,
    not as "not of type null".
    """
    leaves = []

    def walk(e):
        if e.context:
            for c in e.context:
                walk(c)
        else:
            leaves.append(e)

    walk(error)
    telling = {"additionalProperties": 3, "not": 2, "required": 0, "type": 0}   # a stray key says most

    def stray(e) -> int:                       # the closest alternative finds the fewest stray keys
        allowed = e.schema.get("properties", {}) if isinstance(e.schema, dict) else {}
        return len(set(e.instance) - set(allowed)) if isinstance(e.instance, dict) else 0

    return max(leaves, key=lambda e: (len(e.absolute_path), telling.get(e.validator, 1), -stray(e)))


def _required_keys(schema: Any) -> set:
    """Every key a schema fragment requires, through anyOf/oneOf/allOf."""
    if not isinstance(schema, dict):
        return set()
    keys = set(schema.get("required", ()))
    for combinator in ("anyOf", "oneOf", "allOf"):
        for sub in schema.get(combinator, ()):
            keys |= _required_keys(sub)
    return keys


def _schema_problems(validator: Draft202012Validator, instance: Any, where: str) -> list[str]:
    problems = []
    for error in sorted(validator.iter_errors(instance), key=lambda e: list(e.absolute_path)):
        shown = _deepest(error)
        message = shown.message
        if shown.validator == "pattern" and "description" in shown.schema:   # say what is wanted, not the regex
            message = f"{shown.instance!r} is not {shown.schema['description']}"
        elif shown.validator == "not" and isinstance(shown.instance, dict):   # keys that exclude each other
            given = set(shown.schema.get("required", ())) & set(shown.instance)
            banned = _required_keys(shown.validator_value) & set(shown.instance)
            message = f"{', '.join(map(repr, sorted(given)))} cannot be combined with {', '.join(map(repr, sorted(banned)))}"
        problems.append(f"{_where(where, shown.absolute_path)}: {message}")
    return problems


# ---- what a task file holds -------------------------------------------------------------
@dataclass(frozen=True)
class Literal:
    """A state entry with the same value in every instant."""
    value: Any


@dataclass(frozen=True)
class OneOf:
    """One candidate per instant. `file` is where the candidates came from, if they are in a file."""
    candidates: tuple
    sample: Optional[str] = None
    file: Optional[Path] = None


@dataclass(frozen=True)
class Steps:
    """One state per step (coordinate_systems' Instantiation). `file` is the `from` file, if any."""
    names: tuple[str, ...]
    states: tuple[str, ...]
    negative_polarity: bool = False
    negative_suffix: Optional[str] = None
    replacement: bool = True
    fixed: Mapping[str, str] = field(default_factory=dict)
    file: Optional[Path] = None


@dataclass(frozen=True)
class Personas:
    """A persona from a persona set, stored by id."""
    set: str
    sample: Optional[str] = None
    select: tuple = ()


@dataclass(frozen=True)
class Generate:
    """A value returned by a Python callable, `module:function`."""
    using: str
    options: Mapping[str, Any] = field(default_factory=dict)


Entry = Literal | OneOf | Steps | Personas | Generate


@dataclass(frozen=True)
class InstantiationSpec:
    state: Mapping[str, Entry]
    sample: str = "random"
    version: int = 1
    file: Optional[Path] = None               # the instantiation file, when not written inline


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
    output: Mapping[str, Any] = field(default_factory=dict)
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
        found = _schema_problems(TASK_VALIDATOR, entry, name)
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

    instantiation = None
    raw = entry.get("instantiation")
    if isinstance(raw, str):                   # an instantiation file
        file, raw = (here / raw).resolve(), None
        if not file.is_file():
            problems.append(f"{name}.instantiation: no file {entry['instantiation']!r} (relative to {here})")
        else:
            try:
                raw = read_yaml(file)
            except yaml.YAMLError as e:
                problems.append(f"{name}.instantiation: {file.name} is not valid YAML: {e}")
            else:
                found = _schema_problems(INSTANTIATION_VALIDATOR, raw, f"{name}.instantiation ({file.name})")
                problems += found
                content["instantiation"] = raw
                references["files"][_key(file, here)] = _file_sha256(file)
                raw = None if found else raw
        if raw is not None:
            instantiation = _instantiation(name, raw, here, problems, references, file)
    elif isinstance(raw, dict):
        instantiation = _instantiation(name, raw, here, problems, references, None)

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
                references["files"][_key(file, here)] = _file_sha256(file)
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
        permissions=entry.get("permissions"), output=entry.get("output") or {}, source=path)

    if check:
        spec = _check(spec, here, problems, references)
    references = {k: v for k, v in references.items() if v}
    sha = hashlib.sha256(json.dumps({"task": content, "references": references}, sort_keys=True,
                                    ensure_ascii=False, default=str).encode()).hexdigest()
    return TaskSpec(**{**spec.__dict__, "content": content, "references": references, "sha256": sha}), problems


def _instantiation(task, raw, here, problems, references, file) -> InstantiationSpec:
    state = {}
    for entry_name, e in raw["state"].items():
        where = f"{task}.instantiation.state.{entry_name}"
        if not isinstance(e, dict):
            state[entry_name] = Literal(e)
        elif "value" in e:
            state[entry_name] = Literal(e["value"])
        elif "one_of" in e:
            candidates, source = e["one_of"], None
            if isinstance(candidates, str):
                source = (here / candidates).resolve()
                if not source.is_file():
                    problems.append(f"{where}.one_of: no file {candidates!r}")
                    candidates = []
                else:
                    candidates = read_yaml(source)
                    references["files"][_key(source, here)] = _file_sha256(source)
                    if not isinstance(candidates, list) or not candidates:
                        problems.append(f"{where}.one_of: {source.name} must hold a non-empty list")
                        candidates = []
            state[entry_name] = OneOf(tuple(candidates), e.get("sample"), source)
        elif "steps" in e:
            state[entry_name] = _steps(e["steps"], where, here, problems, references)
        elif "personas" in e:
            state[entry_name] = Personas(e["personas"], e.get("sample"), tuple(e.get("select") or ()))
        else:
            state[entry_name] = Generate(e["generate"]["using"], e["generate"].get("options") or {})
    return InstantiationSpec(state, raw.get("sample", "random"), raw.get("version", 1), file)


def _steps(s: dict, where: str, here: Path, problems: list, references: dict) -> Steps:
    names, states, polarity, source = s.get("names"), s.get("states"), s.get("negative_polarity", False), None
    if "from" in s:
        source = (here / s["from"]).resolve()
        if not source.is_file():
            problems.append(f"{where}.steps.from: no file {s['from']!r}")
            return Steps((), (), file=source)
        config = read_yaml(source)
        references["files"][_key(source, here)] = _file_sha256(source)
        missing = [k for k in ("step_aliases", "aliases") if k not in (config or {})]
        if missing:
            problems.append(f"{where}.steps.from: {source.name} has no {' or '.join(missing)}")
            return Steps((), (), file=source)
        names, states = config["step_aliases"], config["aliases"]
        polarity = bool(config.get("negative_polarity", False))
    steps = Steps(tuple(names), tuple(states), polarity, s.get("negative_suffix"), s.get("replacement", True),
                  s.get("fixed") or {}, source)
    if polarity and not steps.negative_suffix:
        problems.append(f"{where}.steps: states may be negative, so negative_suffix must say how one is written")
    for step, state in steps.fixed.items():
        base = state[:-len(steps.negative_suffix)] if polarity and steps.negative_suffix and \
            state.endswith(steps.negative_suffix) else state
        if step not in steps.names:
            problems.append(f"{where}.steps.fixed: {step!r} is not one of the steps {list(steps.names)}")
        elif base not in steps.states:
            problems.append(f"{where}.steps.fixed.{step}: {state!r} is not one of the states")
    if not steps.replacement:
        fixed = [s[:-len(steps.negative_suffix)] if steps.negative_suffix and s.endswith(steps.negative_suffix) else s
                 for s in steps.fixed.values()]
        if len(set(fixed)) != len(fixed):
            problems.append(f"{where}.steps.fixed: replacement is false, but a state is fixed for two steps")
        if len(steps.names) > len(steps.states):
            problems.append(f"{where}.steps: replacement is false, but there are more steps than states")
    return steps


def _import(reference: str):
    module, _, attribute = reference.partition(":")
    return getattr(importlib.import_module(module), attribute)


def _check(spec: TaskSpec, here: Path, problems: list[str], references: dict) -> TaskSpec:
    """The checks a schema cannot make. Returns the spec with each template's PromptTemplate attached."""
    from llm.providers import available_providers
    from personas import PersonaSet
    from prompts import PromptError, PromptTemplate, default_library

    name = spec.name
    state = dict(spec.instantiation.state) if spec.instantiation else {}

    # state entries that name things
    for entry_name, e in state.items():
        where = f"{name}.instantiation.state.{entry_name}"
        if isinstance(e, Personas):
            try:
                people = PersonaSet.load(e.set)
                references["persona_sets"][e.set] = people.sha256
                if e.select:
                    people.select(e.select)
            except (FileNotFoundError, KeyError, ValueError) as err:
                problems.append(f"{where}: {err}")
        elif isinstance(e, Generate):
            try:
                if not callable(_import(e.using)):
                    problems.append(f"{where}.generate.using: {e.using} is not callable")
            except (ImportError, AttributeError) as err:
                problems.append(f"{where}.generate.using: cannot import {e.using}: {err}")

    # templates, and the variables they need
    library, templates, seen = default_library(), [], set()
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
    for retired in spec.prompt.retired_templates:
        try:
            library.resolve(retired)
        except KeyError:
            problems.append(f"{name}.prompt.retired_templates: no prompt template {retired!r}")

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
            target = _import(reference)
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
                what = f"one_of {len(e.candidates)} candidates, {e.sample or spec.instantiation.sample}"
            elif isinstance(e, Personas):
                what = f"personas from {e.set}, {e.sample or spec.instantiation.sample}"
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
