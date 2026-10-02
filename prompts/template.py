"""
Prompt templates: versioned, declarative, strictly rendered.

A PromptTemplate is text (Jinja2) plus the list of variables it takes. Rendering it with
structured inputs (a card draw, a persona, ...) gives a RenderedPrompt: the exact system
and user messages to send, plus the template id and content hash that produced them.

Strictness is deliberate - a prompt that silently renders a blank is a corrupted dataset:
  * every variable used in the text must be declared (required or optional), and vice versa
  * rendering fails on a missing required variable, an unknown variable, or a missing
    attribute (e.g. persona.demographics.agee)
  * optional variables default to None, so templates guard them with {% if persona %}
"""
import hashlib
import json
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

from jinja2 import StrictUndefined, TemplateError, meta
from jinja2.sandbox import SandboxedEnvironment


class PromptError(ValueError):
    """A template is malformed or was rendered with the wrong variables."""


# ---- filters: how structured values become text ---------------------------------------
def kv_lines(mapping: Mapping, sep: str = ": ") -> str:
    """{"Present": "Death"} -> "Present: Death" (one per line, in order)."""
    return "\n".join(f"{k}{sep}{v}" for k, v in mapping.items())


def bullets(items, marker: str = "- ") -> str:
    """["a", "b"] -> "- a\\n- b"; a mapping gives "- key: value" lines."""
    if isinstance(items, Mapping):
        items = [f"{k}: {v}" for k, v in items.items()]
    return "\n".join(f"{marker}{i}" for i in items)


def numbered(items) -> str:
    """["a", "b"] -> "1. a\\n2. b"."""
    if isinstance(items, Mapping):
        items = [f"{k}: {v}" for k, v in items.items()]
    return "\n".join(f"{n}. {i}" for n, i in enumerate(items, 1))


_env = SandboxedEnvironment(undefined=StrictUndefined, trim_blocks=True, lstrip_blocks=True,
                            keep_trailing_newline=False, autoescape=False)
_env.filters.update(kv_lines=kv_lines, bullets=bullets, numbered=numbered)


def register_filter(name: str, fn: Callable) -> None:
    """Add a Jinja filter available to every template (e.g. a domain-specific formatter)."""
    _env.filters[name] = fn


def _normalize(text: str) -> str:
    """Rendered text: no leading/trailing whitespace, no trailing spaces, at most one blank line in a row."""
    text = "\n".join(line.rstrip() for line in text.strip().splitlines())
    return re.sub(r"\n{3,}", "\n\n", text)


# ---- rendered prompt ------------------------------------------------------------------
@dataclass(frozen=True)
class RenderedPrompt:
    template_id: str
    template_sha256: str
    user: str
    system: Optional[str] = None
    variables: tuple[str, ...] = ()     # names of the variables that were supplied (non-None)

    @property
    def messages(self) -> list[dict[str, str]]:
        """OpenAI-style message list, ready for llm.ChatRequest(messages=...)."""
        return ([{"role": "system", "content": self.system}] if self.system else []) + \
               [{"role": "user", "content": self.user}]

    @property
    def sha256(self) -> str:
        """Hash of the rendered messages (what prompts.prompt_sha256 stores)."""
        return hashlib.sha256(json.dumps(self.messages, sort_keys=True).encode()).hexdigest()


# ---- template -------------------------------------------------------------------------
_ID = re.compile(r"^(?P<name>[a-z][a-z0-9_]*?)_v(?P<version>\d+)$")
_NAME = re.compile(r"^[a-z][a-z0-9_]*$")


def parse_id(template_id: str) -> Optional[tuple[str, int]]:
    """"celtic_cross_v2" -> ("celtic_cross", 2); None if it isn't a versioned id."""
    m = _ID.match(template_id)
    return (m["name"], int(m["version"])) if m else None


@dataclass(frozen=True)
class PromptTemplate:
    """
    name, version — identity; id is "<name>_v<version>" (e.g. celtic_cross_v1). A version is
                    meant to be immutable once used: change the wording -> new version.
    user, system  — Jinja2 template text for the messages (system is optional)
    required      — variables that must be supplied
    optional      — variables that may be omitted (rendered as None)
    description   — what the prompt is for
    changes       — what changed relative to the previous version, and why
    based_on      — id of the template this one was derived from (default: previous version)
    """
    name: str
    version: int
    user: str
    system: Optional[str] = None
    required: tuple[str, ...] = ()
    optional: tuple[str, ...] = ()
    description: str = ""
    changes: str = ""
    based_on: Optional[str] = None
    tags: tuple[str, ...] = ()
    source: Optional[Path] = field(default=None, compare=False)

    def __post_init__(self):
        if not _NAME.match(self.name) or parse_id(self.name):
            raise PromptError(f"Invalid template name {self.name!r}: use lowercase_with_underscores, "
                              f"and don't end it with _v<number>")
        if not isinstance(self.version, int) or self.version < 1:
            raise PromptError(f"{self.name}: version must be a positive integer")
        object.__setattr__(self, "user", _normalize(self.user))
        object.__setattr__(self, "system", _normalize(self.system) if self.system and self.system.strip() else None)
        object.__setattr__(self, "required", tuple(self.required))
        object.__setattr__(self, "optional", tuple(self.optional))
        object.__setattr__(self, "tags", tuple(self.tags))
        if not self.user:
            raise PromptError(f"{self.id}: user template is empty")
        overlap = set(self.required) & set(self.optional)
        if overlap:
            raise PromptError(f"{self.id}: variables both required and optional: {sorted(overlap)}")
        used = self.used_variables()
        declared = set(self.variables)
        if used - declared:
            raise PromptError(f"{self.id}: template uses undeclared variables {sorted(used - declared)}; "
                              f"add them to `required` or `optional`")
        if declared - used:
            raise PromptError(f"{self.id}: declared variables never used in the text: {sorted(declared - used)}")

    # ---- identity -------------------------------------------------------------------
    @property
    def id(self) -> str:
        return f"{self.name}_v{self.version}"

    @property
    def variables(self) -> tuple[str, ...]:
        return self.required + self.optional

    @property
    def sha256(self) -> str:
        """Hash of what determines the rendered text: name, version and the template text."""
        content = {"name": self.name, "version": self.version, "system": self.system, "user": self.user}
        return hashlib.sha256(json.dumps(content, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

    def used_variables(self) -> set[str]:
        try:
            used = set()
            for text in (self.user, self.system or ""):
                used |= meta.find_undeclared_variables(_env.parse(text))
            return used
        except TemplateError as e:
            raise PromptError(f"{self.id}: template syntax error: {e}") from e

    # ---- rendering ------------------------------------------------------------------
    def render(self, variables: Optional[Mapping[str, Any]] = None, /, **kwargs) -> RenderedPrompt:
        """Render with exactly the declared variables (optional ones may be omitted)."""
        values = {**(variables or {}), **kwargs}
        unknown = set(values) - set(self.variables)
        if unknown:
            raise PromptError(f"{self.id}: unknown variables {sorted(unknown)}; it takes {list(self.variables)}")
        missing = [v for v in self.required if values.get(v) is None]
        if missing:
            raise PromptError(f"{self.id}: missing required variables {missing}")
        context = {v: values.get(v) for v in self.variables}
        try:
            user = _normalize(_env.from_string(self.user).render(context))
            system = _normalize(_env.from_string(self.system).render(context)) if self.system else None
        except TemplateError as e:
            raise PromptError(f"{self.id}: could not render: {e}") from e
        if not user:
            raise PromptError(f"{self.id}: rendered to an empty user message")
        return RenderedPrompt(self.id, self.sha256, user, system or None,
                              tuple(v for v in self.variables if context[v] is not None))

    def render_from(self, available: Mapping[str, Any]) -> RenderedPrompt:
        """Render, taking only the variables this template declares from a larger pool."""
        return self.render({v: available[v] for v in self.variables if v in available})

    def uses(self, variable: str) -> bool:
        return variable in self.variables

    # ---- (de)serialisation ----------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {"template_id": self.id, "name": self.name, "version": self.version, "system": self.system,
                "user": self.user, "required": list(self.required), "optional": list(self.optional),
                "description": self.description, "changes": self.changes, "based_on": self.based_on,
                "tags": list(self.tags), "template_sha256": self.sha256}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any], source: Optional[Path] = None) -> "PromptTemplate":
        allowed = {"name", "version", "description", "changes", "based_on", "tags", "required", "optional",
                   "template"}
        unknown = set(d) - allowed
        if unknown:
            raise PromptError(f"{source or d.get('name')}: unknown keys {sorted(unknown)}")
        body = d.get("template") or {}
        if set(body) - {"user", "system"} or "user" not in body:
            raise PromptError(f"{source or d.get('name')}: [template] needs `user` and optionally `system`")
        try:
            return cls(name=d["name"], version=d["version"], user=body["user"], system=body.get("system"),
                       required=tuple(d.get("required", ())), optional=tuple(d.get("optional", ())),
                       description=d.get("description", ""), changes=d.get("changes", ""),
                       based_on=d.get("based_on"), tags=tuple(d.get("tags", ())), source=source)
        except KeyError as e:
            raise PromptError(f"{source}: missing key {e}") from e

    @classmethod
    def from_toml(cls, path: str | Path) -> "PromptTemplate":
        path = Path(path)
        with open(path, "rb") as f:
            try:
                data = tomllib.load(f)
            except tomllib.TOMLDecodeError as e:
                raise PromptError(f"{path}: invalid TOML: {e}") from e
        return cls.from_dict(data, source=path)
