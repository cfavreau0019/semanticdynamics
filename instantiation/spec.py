"""
Instantiation specifications: the possible underlying states of a task, as written in a task
file's `instantiation` or in an instantiation file of its own. The format is described in
applications/template/inference_task_reference.yaml and checked by instantiation.schema.yaml.

    spec = read_instantiation("instantiation.yaml", here=Path("applications/tarot"))
    spec.state["cards"]            # Steps(names=..., states=..., replacement=True, ...)

An InstantiationSpec only describes; instantiation.core.Instantiation realizes it.
"""
import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

import yaml
from jsonschema import Draft202012Validator

from spec_files import file_sha256, import_reference, read_yaml, relative_key, schema_problems

SCHEMA = yaml.safe_load((Path(__file__).resolve().parent / "instantiation.schema.yaml").read_text(encoding="utf-8"))
VALIDATOR = Draft202012Validator(SCHEMA)
METHODS = ("random", "cycle", "exhaustive")


class InstantiationError(ValueError):
    """An instantiation that cannot be used; `problems` lists every reason, each with where it is."""

    def __init__(self, problems: list[str]):
        self.problems = list(problems)
        super().__init__("\n".join(self.problems))


# ---- the kinds of state entry ----------------------------------------------------------------
@dataclass(frozen=True)
class Literal:
    """The same value in every instant."""
    value: Any


@dataclass(frozen=True)
class OneOf:
    """One candidate per instant. `file` is where the candidates came from, when they are in one."""
    candidates: tuple
    sample: Optional[str] = None               # None: the instantiation's default
    file: Optional[Path] = None


@dataclass(frozen=True)
class Steps:
    """One state per step, as coordinate_systems' Instantiation did. `file` is the `from` file, if any."""
    names: tuple[str, ...]
    states: tuple[str, ...]
    negative_polarity: bool = False
    negative_suffix: Optional[str] = None
    replacement: bool = True
    fixed: Mapping[str, str] = field(default_factory=dict)
    file: Optional[Path] = None


@dataclass(frozen=True)
class Personas:
    """
    A person from a persona set; stored with a run by id, with a snapshot. `set` is as written: a
    set name (personas/<name>.json), or a path, relative to the task's folder, which `file` resolves.
    """
    set: str
    sample: Optional[str] = None               # None: the instantiation's default
    select: tuple = ()
    file: Optional[Path] = None


@dataclass(frozen=True)
class Generate:
    """A value returned by a Python callable, `package.module:function`, called as function(rng, **options)."""
    using: str
    options: Mapping[str, Any] = field(default_factory=dict)


Entry = Literal | OneOf | Steps | Personas | Generate


@dataclass(frozen=True)
class InstantiationSpec:
    state: Mapping[str, Entry]                 # in the order written, which is the order entries are sampled
    sample: str = "random"                     # the default for one_of and personas entries
    version: int = 1
    file: Optional[Path] = None                # the instantiation file, when not written inline
    instants: Optional[tuple] = None           # instants given rather than sampled, as written
    instants_file: Optional[Path] = None       # where they came from, when from a file

    def method(self, name: str) -> Optional[str]:
        """How an entry chooses its value: its own `sample`, else the default; None for entries that do not choose."""
        entry = self.state[name]
        return (entry.sample or self.sample) if isinstance(entry, (OneOf, Personas)) else None

    @property
    def entries(self) -> tuple[str, ...]:
        """The names every instant has: the state's entries, or, with given instants only, the names they all share."""
        if self.state or not self.instants:
            return tuple(self.state)
        return tuple(k for k in self.instants[0] if all(k in i for i in self.instants))

    def problems(self, instant: Mapping[str, Any]) -> list[str]:
        """Why `instant` could not have come from this state (see instantiation.core.resolve); [] if it could."""
        from instantiation.core import resolve
        return resolve(self, instant)[1]


# ---- reading -----------------------------------------------------------------------------------
def load_instantiation(value: Any, here: Path, where: str, problems: list[str], references: dict,
                       check: bool = True) -> tuple[Optional[InstantiationSpec], Any]:
    """
    The InstantiationSpec a task file's `instantiation` value gives, and its content with an
    instantiation file read in (for recording). `value` is None, the path of an instantiation file
    relative to `here`, or the instantiation itself. Problems are added to `problems`, located by
    `where`; the hashes of files and persona sets it uses are added to `references`
    ({"files": {...}, "persona_sets": {...}}). check=False skips the checks that need other
    packages (persona sets, generate functions).
    """
    if value is None:
        return None, None
    file = None
    if isinstance(value, str):
        file = (here / value).resolve()
        if not file.is_file():
            problems.append(f"{where}: no file {value!r} (relative to {here})")
            return None, value
        try:
            value = read_yaml(file)
        except yaml.YAMLError as e:
            problems.append(f"{where}: {file.name} is not valid YAML: {e}")
            return None, None
        references.setdefault("files", {})[relative_key(file, here)] = file_sha256(file)
        where = f"{where} ({file.name})"
    found = schema_problems(VALIDATOR, value, where)
    if found:
        problems += found
        return None, value
    spec = _parse(value, here, where, problems, references, file)
    if check:
        _check(spec, where, problems, references)
    return spec, copy.deepcopy(value)


def read_instantiation(path: str | Path, here: Optional[Path] = None, check: bool = True) -> InstantiationSpec:
    """The InstantiationSpec of an instantiation file. Raises InstantiationError listing every problem."""
    path = Path(path)
    here = Path(here) if here is not None else path.resolve().parent
    problems: list[str] = []
    spec, _ = load_instantiation(str(path), here, "instantiation", problems, {}, check)
    if problems:
        raise InstantiationError(problems)
    return spec


def _parse(raw: dict, here: Path, where: str, problems: list, references: dict, file) -> InstantiationSpec:
    state = {}
    for name, e in (raw.get("state") or {}).items():
        at = f"{where}.state.{name}"
        if not isinstance(e, dict):
            state[name] = Literal(e)
        elif "value" in e:
            state[name] = Literal(e["value"])
        elif "one_of" in e:
            candidates, source = e["one_of"], None
            if isinstance(candidates, str):
                source = (here / candidates).resolve()
                if not source.is_file():
                    problems.append(f"{at}.one_of: no file {candidates!r}")
                    candidates = []
                else:
                    candidates = read_yaml(source)
                    references.setdefault("files", {})[relative_key(source, here)] = file_sha256(source)
                    if not isinstance(candidates, list) or not candidates:
                        problems.append(f"{at}.one_of: {source.name} must hold a non-empty list")
                        candidates = []
            state[name] = OneOf(tuple(candidates), e.get("sample"), source)
        elif "steps" in e:
            state[name] = _steps(e["steps"], at, here, problems, references)
        elif "personas" in e:
            people = (here / e["personas"]).resolve()
            if not people.is_file() and _is_path(e["personas"]):
                problems.append(f"{at}.personas: no file {e['personas']!r} (relative to {here})")
            state[name] = Personas(e["personas"], e.get("sample"), tuple(e.get("select") or ()),
                                   people if people.is_file() else None)
        else:
            state[name] = Generate(e["generate"]["using"], e["generate"].get("options") or {})
    instants, instants_file = raw.get("instants"), None
    if isinstance(instants, str):
        instants_file = (here / instants).resolve()
        instants = _read_instants(instants_file, f"{where}.instants", here, problems, references)
    return InstantiationSpec(state, raw.get("sample", "random"), raw.get("version", 1), file,
                             None if instants is None else tuple(instants), instants_file)


def _is_path(value: str) -> bool:
    """A persona set written as a path (a.json, people/a), rather than as a set name."""
    return bool(Path(value).suffix) or "/" in value or "\\" in value


def _read_instants(file: Path, where: str, here: Path, problems: list, references: dict) -> tuple:
    """Given instants from a YAML or JSON file holding a list, or a JSONL file with one per line."""
    if not file.is_file():
        problems.append(f"{where}: no file {relative_key(file, here)!r} (relative to {here})")
        return ()
    try:
        if file.suffix == ".jsonl":
            rows = [json.loads(line) for line in file.read_text(encoding="utf-8").splitlines() if line.strip()]
        else:
            rows = read_yaml(file)
    except (yaml.YAMLError, json.JSONDecodeError) as e:
        problems.append(f"{where}: {file.name} could not be read: {e}")
        return ()
    references.setdefault("files", {})[relative_key(file, here)] = file_sha256(file)
    if not isinstance(rows, list) or not rows or not all(isinstance(r, dict) and r for r in rows):
        problems.append(f"{where}: {file.name} must hold a non-empty list of instants, each mapping entry names "
                        f"to values")
        return ()
    return tuple(rows)


def _steps(s: dict, where: str, here: Path, problems: list, references: dict) -> Steps:
    names, states, polarity, source = s.get("names"), s.get("states"), s.get("negative_polarity", False), None
    if "from" in s:
        source = (here / s["from"]).resolve()
        if not source.is_file():
            problems.append(f"{where}.steps.from: no file {s['from']!r}")
            return Steps((), (), file=source)
        config = read_yaml(source)
        references.setdefault("files", {})[relative_key(source, here)] = file_sha256(source)
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
        if step not in steps.names:
            problems.append(f"{where}.steps.fixed: {step!r} is not one of the steps {list(steps.names)}")
        elif base_state(steps, state) not in steps.states:
            problems.append(f"{where}.steps.fixed.{step}: {state!r} is not one of the states")
    if not steps.replacement:
        fixed = [base_state(steps, s) for s in steps.fixed.values()]
        if len(set(fixed)) != len(fixed):
            problems.append(f"{where}.steps.fixed: replacement is false, but a state is fixed for two steps")
        if len(steps.names) > len(steps.states):
            problems.append(f"{where}.steps: replacement is false, but there are more steps than states")
    return steps


def base_state(steps: Steps, state: str) -> str:
    """A state without its negative suffix: "Death reversed" -> "Death"."""
    suffix = steps.negative_suffix
    return state[:-len(suffix)] if steps.negative_polarity and suffix and state.endswith(suffix) else state


def _check(spec: InstantiationSpec, where: str, problems: list[str], references: dict) -> None:
    """
    What the entries name must exist: persona sets (and the personas selected), generate functions.
    Given instants must be possible states, when there is a state to check them against.
    """
    from personas import PersonaSet

    before = len(problems)
    for name, e in spec.state.items():
        at = f"{where}.state.{name}"
        if isinstance(e, Personas):
            if e.file is None and _is_path(e.set):
                continue                       # no such file: said when it was read
            try:
                people = PersonaSet.load(e.file or e.set)
                references.setdefault("persona_sets", {})[e.set] = people.sha256
                if e.select:
                    people.select(e.select)
            except (FileNotFoundError, KeyError, ValueError) as err:
                problems.append(f"{at}: {err.args[0] if err.args else err}")
        elif isinstance(e, Generate):
            try:
                if not callable(import_reference(e.using)):
                    problems.append(f"{at}.generate.using: {e.using} is not callable")
            except (ImportError, AttributeError) as err:
                problems.append(f"{at}.generate.using: cannot import {e.using}: {err}")

    if spec.instants and spec.state and len(problems) == before:
        from instantiation.core import resolve
        at = f"{where}.instants" + (f" ({spec.instants_file.name})" if spec.instants_file else "")
        cache: dict = {}
        wrong = [found for k, given in enumerate(spec.instants) if (found := resolve(spec, given, f"{at}[{k}]", cache)[1])]
        for found in wrong[:SHOWN]:
            problems += found
        if len(wrong) > SHOWN:
            problems.append(f"{at}: and {len(wrong) - SHOWN} more instants that are not possible states")


SHOWN = 10                                     # given instants whose problems are listed; the rest are counted
