"""
The prompt library: every template version as a TOML file under prompts/library/, plus a
lock file that pins the content of versions already in use.

    prompts/library/<area>/<name>_v<version>.toml
    prompts/library/prompts.lock.json          {template_id: content sha256}

Version workflow
    1. copy <name>_v1.toml to <name>_v2.toml, bump `version`, edit, fill in `changes`
    2. iterate freely: an unlocked version is a *draft* (usable, flagged as such)
    3. `python -m prompts lock` once it has produced data you keep
    4. from then on its text must not change; the library refuses to load a locked
       template whose content differs (make a new version instead)
"""
import difflib
import json
from pathlib import Path
from typing import Iterable, Optional

from prompts.template import PromptError, PromptTemplate, parse_id

LIBRARY_DIR = Path(__file__).resolve().parent / "library"
LOCK_NAME = "prompts.lock.json"

LOCKED, DRAFT, MODIFIED = "locked", "draft", "modified"


class PromptLockError(PromptError):
    """A locked template's content no longer matches the lock file."""


class PromptLibrary:
    def __init__(self, root: str | Path = LIBRARY_DIR, templates: Optional[Iterable[PromptTemplate]] = None):
        self.root = Path(root)
        self.lock_path = self.root / LOCK_NAME
        self._templates: dict[str, PromptTemplate] = {}
        for t in (templates if templates is not None else self._load_files()):
            self.add(t)

    def _load_files(self) -> list[PromptTemplate]:
        out = []
        for path in sorted(self.root.rglob("*.toml")):
            t = PromptTemplate.from_toml(path)
            if path.stem != t.id:
                raise PromptError(f"{path}: file name should be {t.id}.toml (it defines {t.id})")
            out.append(t)
        return out

    def add(self, template: PromptTemplate) -> None:
        if template.id in self._templates:
            raise PromptError(f"Duplicate template {template.id} ({template.source} and "
                              f"{self._templates[template.id].source})")
        self._templates[template.id] = template

    # ---- lookup ---------------------------------------------------------------------
    def names(self) -> list[str]:
        return sorted({t.name for t in self._templates.values()})

    def versions(self, name: str) -> list[PromptTemplate]:
        return sorted((t for t in self._templates.values() if t.name == name), key=lambda t: t.version)

    def all(self) -> list[PromptTemplate]:
        return sorted(self._templates.values(), key=lambda t: (t.name, t.version))

    def resolve(self, ref: str) -> PromptTemplate:
        """'celtic_cross_v2' -> that version; 'celtic_cross' -> its latest version. No lock check."""
        if ref in self._templates:
            return self._templates[ref]
        versions = self.versions(ref)
        if versions:
            return versions[-1]
        hint = f"; versions of {parse_id(ref)[0]!r}: {[t.id for t in self.versions(parse_id(ref)[0])]}" \
            if parse_id(ref) else ""
        raise KeyError(f"No prompt template {ref!r}. Available: {[t.id for t in self.all()]}{hint}")

    def get(self, ref: str) -> PromptTemplate:
        """Resolve a template and refuse it if it is locked but its content has changed."""
        t = self.resolve(ref)
        if self.status(t) == MODIFIED:
            raise PromptLockError(
                f"{t.id} is locked but its content has changed ({t.source}). Locked versions are immutable "
                f"so stored data stays reproducible: restore the file and put the new wording in "
                f"{t.name}_v{self.versions(t.name)[-1].version + 1}, or, if this version has produced no data "
                f"you keep, re-lock it with `python -m prompts lock --force {t.id}`.")
        return t

    # ---- locking --------------------------------------------------------------------
    def read_lock(self) -> dict[str, str]:
        if not self.lock_path.exists():
            return {}
        return json.loads(self.lock_path.read_text(encoding="utf-8"))

    def status(self, template: PromptTemplate) -> str:
        locked = self.read_lock().get(template.id)
        if locked is None:
            return DRAFT
        return LOCKED if locked == template.sha256 else MODIFIED

    def verify(self) -> list[str]:
        """Problems that would make stored data irreproducible; empty when all is well."""
        lock = self.read_lock()
        problems = [f"{t.id}: content differs from the lock" for t in self.all() if self.status(t) == MODIFIED]
        problems += [f"{tid}: locked but its file is missing" for tid in sorted(set(lock) - set(self._templates))]
        return problems

    def lock(self, ids: Optional[Iterable[str]] = None, force: bool = False) -> list[str]:
        """
        Lock drafts (all, or the given ids). Already-locked templates whose content changed
        are only re-locked with force=True. Returns the ids whose lock entry was written.
        """
        lock = self.read_lock()
        targets = [self.resolve(i) for i in ids] if ids else self.all()
        written = []
        for t in targets:
            state = self.status(t)
            if state == MODIFIED and not force:
                raise PromptLockError(f"{t.id} is locked with different content; pass force=True to overwrite")
            if state != LOCKED:
                lock[t.id] = t.sha256
                written.append(t.id)
        if written:
            self.lock_path.write_text(json.dumps(dict(sorted(lock.items())), indent=2) + "\n", encoding="utf-8")
        return written

    # ---- comparing versions ---------------------------------------------------------
    def diff(self, a: str, b: str) -> str:
        """Unified diff of two templates' text (system + user), for reviewing an iteration."""
        ta, tb = self.resolve(a), self.resolve(b)
        text = lambda t: (f"[system]\n{t.system}\n\n" if t.system else "") + f"[user]\n{t.user}\n"
        lines = difflib.unified_diff(text(ta).splitlines(), text(tb).splitlines(), ta.id, tb.id, lineterm="")
        return "\n".join(lines) or f"{ta.id} and {tb.id} have identical text"


_default: Optional[PromptLibrary] = None


def default_library(reload: bool = False) -> PromptLibrary:
    global _default
    if _default is None or reload:
        _default = PromptLibrary()
    return _default


def get_template(ref: str) -> PromptTemplate:
    """Template from the default library: 'name_vN' for a pinned version, 'name' for the latest."""
    return default_library().get(ref)
