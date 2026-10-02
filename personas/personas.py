"""
Persona sets: JSON files of people (demographics, profile, description) that can be used
as optional inputs to prompts.

    from personas import PersonaSet

    people = PersonaSet.load("tarot_personas")          # personas/tarot_personas.json
    maya = people.get(1)                                 # by number, id, or full name
    beginners = people.filter(**{"tarot_profile.experience_level": "Beginner"})
    assigned = people.sample(random.Random(0), 500)      # seeded, one per prompt
    people.to_frame()                                    # flat DataFrame, one column per field

A persona is a plain dict (exactly the JSON record), so prompt templates read fields
directly: {{ persona.demographics.age }}. Which fields a prompt discloses is therefore
decided, and versioned, in the template.
"""
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Optional

PERSONAS_DIR = Path(__file__).resolve().parent
SAMPLING_MODES = ("random", "cycle")

Persona = dict[str, Any]


def content_sha256(persona: Persona) -> str:
    return hashlib.sha256(json.dumps(persona, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def get_path(persona: Persona, path: str, default: Any = None) -> Any:
    """Nested lookup with a dotted path: get_path(p, "demographics.location.city")."""
    value: Any = persona
    for key in path.split("."):
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def flatten(persona: Persona, prefix: str = "") -> dict[str, Any]:
    """{"demographics": {"age": 24}} -> {"demographics.age": 24}; lists are kept as lists."""
    out: dict[str, Any] = {}
    for key, value in persona.items():
        if isinstance(value, dict):
            out.update(flatten(value, f"{prefix}{key}."))
        else:
            out[f"{prefix}{key}"] = value
    return out


def display_name(persona: Persona) -> Optional[str]:
    name = persona.get("name")
    if isinstance(name, dict):
        return name.get("full") or " ".join(x for x in (name.get("first"), name.get("last")) if x) or None
    return name


@dataclass
class PersonaSet:
    name: str
    personas: list[Persona]
    path: Optional[Path] = None
    sha256: Optional[str] = None      # of the source file: identifies the exact set a run used

    def __post_init__(self):
        self.validate()
        self._by_id = {p["id"]: p for p in self.personas}

    # ---- loading / validation -----------------------------------------------------
    @classmethod
    def load(cls, name_or_path: str | Path) -> "PersonaSet":
        """A set name ("tarot_personas" -> personas/tarot_personas.json) or a path to a JSON file."""
        p = Path(name_or_path)
        candidates = [p, PERSONAS_DIR / p, PERSONAS_DIR / f"{name_or_path}.json"]
        path = next((c for c in candidates if c.is_file()), None)
        if path is None:
            available = sorted(f.stem for f in PERSONAS_DIR.glob("*.json"))
            raise FileNotFoundError(f"No persona set {str(name_or_path)!r}. Available: {available}")
        raw = path.read_bytes()
        data = json.loads(raw)
        if isinstance(data, dict):  # allow {"personas": [...]}
            data = data.get("personas")
        if not isinstance(data, list):
            raise ValueError(f"{path}: expected a JSON list of personas")
        return cls(path.stem, data, path.resolve(), hashlib.sha256(raw).hexdigest())

    def validate(self) -> None:
        """Every persona needs a unique string `id`; `persona_number`, when present, must be unique too."""
        problems = []
        ids = [p.get("id") for p in self.personas if isinstance(p, dict)]
        if len(ids) != len(self.personas) or not all(isinstance(i, str) and i for i in ids):
            problems.append("every persona must be an object with a non-empty string `id`")
        elif len(set(ids)) != len(ids):
            problems.append("duplicate persona ids")
        numbers = [p["persona_number"] for p in self.personas if isinstance(p, dict) and "persona_number" in p]
        if len(set(numbers)) != len(numbers):
            problems.append("duplicate persona_number values")
        if problems:
            raise ValueError(f"Persona set {self.name!r}: " + "; ".join(problems))

    # ---- access -------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.personas)

    def __iter__(self) -> Iterator[Persona]:
        return iter(self.personas)

    def get(self, ref: str | int) -> Persona:
        """Look up by id, persona_number (int or digit string), or full name (case-insensitive)."""
        if isinstance(ref, str) and ref in self._by_id:
            return self._by_id[ref]
        if isinstance(ref, int) or (isinstance(ref, str) and ref.isdigit()):
            for p in self.personas:
                if p.get("persona_number") == int(ref):
                    return p
        if isinstance(ref, str):
            for p in self.personas:
                if (display_name(p) or "").casefold() == ref.casefold():
                    return p
        raise KeyError(f"No persona {ref!r} in set {self.name!r}")

    def subset(self, personas: Iterable[Persona]) -> "PersonaSet":
        return PersonaSet(self.name, list(personas), self.path, self.sha256)

    def select(self, refs: Iterable[str | int]) -> "PersonaSet":
        """The personas with these ids / numbers / names, in the order given."""
        return self.subset(self.get(r) for r in refs)

    def filter(self, predicate: Optional[Callable[[Persona], bool]] = None, **equals) -> "PersonaSet":
        """
        Keep personas matching a predicate and/or field values given as dotted paths:
            people.filter(**{"demographics.generation": "Gen Z"})
            people.filter(lambda p: p["demographics"]["age"] >= 65)
        """
        def keep(p: Persona) -> bool:
            return (predicate is None or predicate(p)) and all(get_path(p, k) == v for k, v in equals.items())
        return self.subset(p for p in self.personas if keep(p))

    # ---- sampling -----------------------------------------------------------------
    def sample(self, rng: random.Random, n: int, mode: str = "random") -> list[Persona]:
        """
        n personas, one per prompt.
          random — independent draws with replacement (seeded by rng)
          cycle  — in set order, repeating: every persona is used equally often (±1)
        """
        if mode not in SAMPLING_MODES:
            raise ValueError(f"mode must be one of {SAMPLING_MODES}, got {mode!r}")
        if not self.personas:
            raise ValueError(f"Persona set {self.name!r} is empty")
        if mode == "cycle":
            return [self.personas[i % len(self.personas)] for i in range(n)]
        return [rng.choice(self.personas) for _ in range(n)]

    # ---- tabular views ------------------------------------------------------------
    def row(self, persona: Persona) -> dict[str, Any]:
        """The record stored in data_generation's `personas` table."""
        return {"persona_id": persona["id"], "persona_set": self.name,
                "persona_number": persona.get("persona_number"), "name": display_name(persona),
                "payload": persona, "content_sha256": content_sha256(persona)}

    def to_frame(self):
        """One row per persona, nested fields flattened to dotted column names."""
        import pandas as pd
        return pd.DataFrame([flatten(p) for p in self.personas])
