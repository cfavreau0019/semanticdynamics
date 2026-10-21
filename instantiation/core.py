"""
Instantiation: one realized state of a task, its `instant`.

    run = Instantiation.sample(spec, n=20, seed=42)        # a run's instants, from an InstantiationSpec
    run[0].instant                                          # {"cards": {"Present": "Death", ...}}
    Instantiation.from_file("instantiation.yaml", seed=42)  # one instant of an instantiation file
    Instantiation.of({"cards": {...}}, spec)                # values already chosen, checked against a spec

    Instantiation(possible_states, step_aliases, state_step_dict)   # the original use: one state per step
    .instant                                                         # {step: state}

How a run is sampled
  * Entries are sampled in the order written, each for the whole run before the next. So an
    entry's values depend only on the seed and the entries before it: adding an entry after
    the others never changes theirs.
  * literal     the same value in every instant (a copy each)
  * one_of, personas
                random: rng.choice per instant; cycle: candidates in order across the run;
                exhaustive: see below. Their `sample`, else the instantiation's default.
  * steps       one state per step, drawn for each instant (as draw_steps below)
  * generate    function(rng, **options), called once per instant
  * exhaustive  the entries sampled exhaustively together take every combination of their
                candidates exactly once, in order (the last entry varying fastest). The run's n
                must equal the number of combinations, or sample() refuses.
  * given       a spec's `instants` are not sampled: the run is those instants, in order, each
                checked against the state when there is one (see resolve). n must equal their number.

The tarot draws of every run before the task file existed are reproduced exactly from their seed:
draw_steps consumes the random generator as TarotReadings did.
"""
import copy
import itertools
import math
import random
from pathlib import Path
from typing import Any, Iterator, Mapping, Optional, Sequence

from spec_files import import_reference
from instantiation.spec import (
    Generate, InstantiationError, InstantiationSpec, Literal, OneOf, Personas, Steps, base_state, read_instantiation,
)


def draw_steps(rng, names: Sequence[str], states: Sequence[str], fixed: Optional[Mapping[str, str]] = None,
               replacement: bool = True, negative_polarity: bool = False,
               negative_suffix: Optional[str] = None) -> dict[str, str]:
    """
    {step: state} for every step, in step order. `fixed` steps keep their state. With replacement,
    every other step takes any state, uniformly among the states and, if they may be negative, their
    negative forms (a state may repeat). Without, each state is used at most once, and each drawn state
    is negative with probability 1/2.
    """
    fixed = dict(fixed or {})
    suffix = negative_suffix if negative_polarity else None
    if replacement:
        pool = [s + p for s in states for p in (("", suffix) if suffix else ("",))]
        return {name: fixed[name] if name in fixed else rng.choice(pool) for name in names}
    used = {s[:-len(suffix)] if suffix and s.endswith(suffix) else s for s in fixed.values()}
    picks = iter(rng.sample([s for s in states if s not in used], len(names) - len(fixed)))
    out = {}
    for name in names:
        if name in fixed:
            out[name] = fixed[name]
        else:
            state = next(picks)
            out[name] = state + suffix if suffix and rng.random() < 0.5 else state
    return out


def candidates(entry: OneOf | Personas) -> list:
    """What a one_of or personas entry chooses from: its candidates, or the (selected) persona records."""
    if isinstance(entry, OneOf):
        return list(entry.candidates)
    from personas import PersonaSet
    people = PersonaSet.load(entry.file or entry.set)
    return list(people.select(entry.select) if entry.select else people)


def fixed_size(spec: InstantiationSpec) -> Optional[int]:
    """
    The number of instants a run of this spec must have, when that is fixed: the number of given
    instants, or of combinations of the exhaustive entries. None: a run may have any number.
    """
    if spec.instants is not None:
        return len(spec.instants)
    exhaustive = [name for name in spec.state if spec.method(name) == "exhaustive"]
    return math.prod(len(candidates(spec.state[name])) for name in exhaustive) if exhaustive else None


# ---- values already chosen -----------------------------------------------------------------
def resolve(spec: InstantiationSpec, instant: Any, where: str = "instant",
            cache: Optional[dict] = None) -> tuple[dict, list[str]]:
    """
    A given instant checked against the spec's state, with what it refers to filled in: a persona
    given by id, number or full name becomes its record, and a literal entry left out takes its
    value. Returns the instant, its entries in the state's order, and every problem found, each
    located from `where`. Without a state there is nothing to check: the instant is taken as given.
    `cache` keeps persona sets loaded across calls.
    """
    if not isinstance(instant, Mapping):
        return {}, [f"{where}: an instant maps entry names to values, not {instant!r}"]
    if not spec.state:
        return copy.deepcopy(dict(instant)), []
    cache = {} if cache is None else cache
    problems = [f"{where}.{name}: not an entry of the state ({', '.join(spec.state)})"
                for name in instant if name not in spec.state]
    out = {}
    for name, entry in spec.state.items():
        at = f"{where}.{name}"
        if name not in instant:
            if isinstance(entry, Literal):
                out[name] = copy.deepcopy(entry.value)
            else:
                problems.append(f"{at}: missing")
            continue
        value = copy.deepcopy(instant[name])
        if isinstance(entry, Literal):
            if value != entry.value:
                problems.append(f"{at}: must be {entry.value!r}, the entry's only value")
        elif isinstance(entry, OneOf):
            if value not in entry.candidates:
                problems.append(f"{at}: {value!r} is not one of the candidates")
        elif isinstance(entry, Steps):
            value = _given_steps(entry, value, at, problems)
        elif isinstance(entry, Personas):
            value = _given_persona(name, entry, value, at, problems, cache)
        out[name] = value                       # generate: whatever its function could return
    return out, problems


def _given_steps(entry: Steps, value: Any, at: str, problems: list[str]) -> Any:
    if not isinstance(value, Mapping):
        problems.append(f"{at}: must map each step to its state")
        return value
    missing = [step for step in entry.names if step not in value]
    if missing:
        problems.append(f"{at}: no state for {missing}")
    stray = [step for step in value if step not in entry.names]
    if stray:
        problems.append(f"{at}: {stray} are not steps ({', '.join(entry.names)})")
    used = []
    for step in entry.names:
        if step not in value:
            continue
        state = value[step]
        if not isinstance(state, str) or base_state(entry, state) not in entry.states:
            problems.append(f"{at}.{step}: {state!r} is not one of the states")
        elif step in entry.fixed and state != entry.fixed[step]:
            problems.append(f"{at}.{step}: the step is fixed to {entry.fixed[step]!r}, not {state!r}")
        else:
            used.append(base_state(entry, state))
    repeated = sorted({state for state in used if used.count(state) > 1})
    if repeated and not entry.replacement:
        problems.append(f"{at}: replacement is false, but {repeated} appear more than once")
    return {step: value[step] for step in entry.names if step in value}


def _given_persona(name: str, entry: Personas, value: Any, at: str, problems: list[str], cache: dict) -> Any:
    """A persona given by id, number or full name, or as its record: the record, from the set."""
    from personas import PersonaSet
    if name not in cache:
        people = PersonaSet.load(entry.file or entry.set)
        cache[name] = (people, {p["id"] for p in candidates(entry)})
    people, allowed = cache[name]
    if isinstance(value, Mapping):
        record = next((p for p in people if p["id"] == value.get("id")), None)
        if record is None:
            problems.append(f"{at}: no persona with id {value.get('id')!r} in {entry.set}")
            return value
        if record != value:
            problems.append(f"{at}: {_who(record)} differs from its record in {entry.set}; "
                            f"give its id to use the record as it is")
            return value
    elif isinstance(value, (str, int)) and not isinstance(value, bool):
        try:
            record = people.get(value)
        except KeyError as err:
            problems.append(f"{at}: {err.args[0]}")
            return value
    else:
        problems.append(f"{at}: a persona is given by id, number or full name, or as its record")
        return value
    if record["id"] not in allowed:
        problems.append(f"{at}: {_who(record)} is not among those selected ({list(entry.select)})")
    return copy.deepcopy(record)


def _who(record: Mapping) -> str:
    """A persona as people know it: "persona 7 (Maya Delgado)"."""
    from personas import display_name
    name = display_name(record)
    return f"persona {record.get('persona_number', record['id'])}" + (f" ({name})" if name else "")


def _column(spec: InstantiationSpec, name: str, n: int, rng) -> list:
    """The values of one entry for a whole run (not exhaustive entries: sample() does those together)."""
    entry = spec.state[name]
    if isinstance(entry, Literal):
        return [copy.deepcopy(entry.value) for _ in range(n)]
    if isinstance(entry, Steps):
        return [draw_steps(rng, entry.names, entry.states, entry.fixed, entry.replacement, entry.negative_polarity,
                           entry.negative_suffix) for _ in range(n)]
    if isinstance(entry, Generate):
        function = import_reference(entry.using)
        return [function(rng, **entry.options) for _ in range(n)]
    pool = candidates(entry)
    if spec.method(name) == "cycle":
        return [copy.deepcopy(pool[i % len(pool)]) for i in range(n)]
    return [copy.deepcopy(rng.choice(pool)) for _ in range(n)]


class Instantiation:
    """One realized state, `.instant`; `.spec` and `.index` say where it came from, when it came from a spec."""

    def __init__(self, possible_states: Sequence[str], step_aliases: Sequence[str],
                 state_step_dict: Optional[Mapping[str, str]] = None, rng=None, replacement: bool = True):
        """
        The original use: one state per step. `possible_states` are every state a step may take,
        `state_step_dict` fixes some steps ("pins"). `rng` is a random.Random (default: the random
        module's own); replacement=False uses each state at most once.
        """
        self.possible_states = list(possible_states)
        self.step_aliases = list(step_aliases)
        self.state_step_dict = dict(state_step_dict or {})
        self.check_state_step_dict()
        self.spec: Optional[InstantiationSpec] = None
        self.index: Optional[int] = None
        self.instant = draw_steps(rng or random, self.step_aliases, self.possible_states, self.state_step_dict,
                                  replacement)

    def check_state_step_dict(self) -> None:
        """The fixed steps must be steps, and their states possible states. Raises ValueError."""
        unknown_steps = [s for s in self.state_step_dict if s not in self.step_aliases]
        unknown_states = [s for s in self.state_step_dict.values() if s not in self.possible_states]
        if unknown_steps or unknown_states:
            raise ValueError(f"state_step_dict fixes unknown steps {unknown_steps} or states {unknown_states}")

    @classmethod
    def _realized(cls, instant: dict, spec: Optional[InstantiationSpec], index: Optional[int]) -> "Instantiation":
        self = cls.__new__(cls)
        self.possible_states, self.step_aliases, self.state_step_dict = None, None, {}
        self.instant, self.spec, self.index = instant, spec, index
        return self

    def __repr__(self) -> str:
        return f"Instantiation(index={self.index}, instant={self.instant!r})"

    # ---- from a specification ------------------------------------------------------------
    @classmethod
    def of(cls, instant: Mapping[str, Any], spec: Optional[InstantiationSpec] = None,
           index: Optional[int] = None) -> "Instantiation":
        """
        Values already chosen, as an Instantiation. With a spec, they are checked against its state
        and what they refer to is filled in (see resolve); InstantiationError lists every problem.
        """
        if spec is None:
            return cls._realized(copy.deepcopy(dict(instant)), None, index)
        resolved, problems = resolve(spec, instant)
        if problems:
            raise InstantiationError(problems)
        return cls._realized(resolved, spec, index)

    @classmethod
    def sample(cls, spec: InstantiationSpec, n: Optional[int] = None, rng: Optional[random.Random] = None,
               seed: Optional[int] = None) -> list["Instantiation"]:
        """
        A run's n instants. The random generator is `rng`, or a new one from `seed` (None: unseeded).
        n may be left out when the spec fixes it (given instants, exhaustive entries). Raises
        InstantiationError if the run cannot be made, e.g. when exhaustive entries have more or
        fewer combinations than n, or a given instant is not a possible state.
        """
        if n is None:
            n = fixed_size(spec)
            if n is None:
                raise InstantiationError(["n is needed: this instantiation can make a run of any size"])
        if n < 1:
            raise InstantiationError([f"a run needs at least one instant, not n={n}"])
        if spec.instants is not None:
            return cls._given(spec, n)
        rng = rng if rng is not None else random.Random(seed)
        columns: dict[str, list] = {}
        exhaustive = [name for name in spec.state if spec.method(name) == "exhaustive"]
        if exhaustive:
            pools = [candidates(spec.state[name]) for name in exhaustive]
            count = math.prod(len(pool) for pool in pools)
            if n != count:
                raise InstantiationError([
                    f"exhaustive sampling of {', '.join(exhaustive)} gives {count} combinations "
                    f"({' x '.join(str(len(p)) for p in pools)}): n must be {count}, not {n}"])
            combinations = list(itertools.product(*pools))
            for k, name in enumerate(exhaustive):
                columns[name] = [copy.deepcopy(c[k]) for c in combinations]
        for name in spec.state:
            if name not in columns:
                columns[name] = _column(spec, name, n, rng)
        return [cls._realized({name: columns[name][i] for name in spec.state}, spec, i) for i in range(n)]

    @classmethod
    def _given(cls, spec: InstantiationSpec, n: int) -> list["Instantiation"]:
        if n != len(spec.instants):
            raise InstantiationError([f"the instantiation gives {len(spec.instants)} instants: "
                                      f"n must be {len(spec.instants)}, not {n}"])
        cache, run, problems = {}, [], []
        for k, given in enumerate(spec.instants):
            resolved, found = resolve(spec, given, f"instants[{k}]", cache)
            run.append(cls._realized(resolved, spec, k))
            problems += found
        if problems:
            raise InstantiationError(problems)
        return run

    @classmethod
    def from_file(cls, path: str | Path, seed: Optional[int] = None) -> "Instantiation":
        """One instant of an instantiation file."""
        return cls.sample(read_instantiation(path), 1, seed=seed)[0]


def probes(spec: InstantiationSpec, seed: int = 0, limit: int = 1000) -> Iterator[tuple[str, dict]]:
    """
    Instants to try templates against, each with a label: one instant, then the same instant with
    each other candidate of each one_of and personas entry in turn (up to `limit` per entry), so
    that a template is tried with every persona of a set. Literal, steps and generate entries keep
    their first value. Given instants are tried as they are (the first `limit` that are possible).
    """
    if spec.instants is not None:
        cache: dict = {}
        for k, given in enumerate(spec.instants[:limit]):
            resolved, problems = resolve(spec, given, cache=cache)
            if not problems:
                yield f"instants[{k}]", resolved
        return
    rng = random.Random(seed)
    pools = {name: candidates(e) for name, e in spec.state.items() if isinstance(e, (OneOf, Personas))}
    base = {name: copy.deepcopy(pools[name][0]) if name in pools else _column(spec, name, 1, rng)[0]
            for name in spec.state}
    yield "a sample instant", base
    for name, pool in pools.items():
        for k, candidate in enumerate(pool[1:limit], 2):
            yield _label(name, spec.state[name], candidate, k), {**base, name: copy.deepcopy(candidate)}


def _label(name: str, entry: Any, candidate: Any, k: int) -> str:
    if isinstance(entry, Personas) and isinstance(candidate, dict):
        return f"{name} = {_who(candidate)}"
    return f"{name} = candidate {k}"
