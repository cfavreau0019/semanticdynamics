"""
Reading the declarative files of an application (task files, instantiation files, and the files
they name), and reporting what is wrong with them in plain words. Shared by the `instantiation`
package and the task config loader (inference.task_config); it depends on neither.

  read_yaml       strict: a duplicate key is an error, and dates stay as written
  file_sha256     a file's identity, independent of line endings
  relative_key    how a referenced file is named in a hash: relative, so the hash is the same anywhere
  schema_problems a JSON Schema's errors as "where: what" lines, worded for people
  import_reference  the object a "package.module:name" reference names
"""
import hashlib
import importlib
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft202012Validator


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
    """A YAML (or JSON) file, read strictly: a duplicate key is an error (plain YAML keeps the last
    one silently), and dates stay as the strings written rather than becoming date objects."""
    with open(path, encoding="utf-8") as f:
        return yaml.load(f, Loader=_StrictLoader)


def file_sha256(path: Path) -> str:
    """A file's identity. Line endings are normalised: a checkout may write the file with CRLF."""
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def relative_key(path: Path, here: Path) -> str:
    """How a referenced file is named in a hash: relative to the file that refers to it, so that
    the hash is the same wherever the repository is."""
    try:
        return Path(path).relative_to(here).as_posix()
    except ValueError:
        return Path(path).as_posix()


def import_reference(reference: str) -> Any:
    """The object named by "package.module:name"."""
    module, _, attribute = reference.partition(":")
    return getattr(importlib.import_module(module), attribute)


# ---- schema errors, in plain words ------------------------------------------------------------
def _where(where: str, path) -> str:
    for part in path:
        where += f"[{part}]" if isinstance(part, int) else f".{part}"
    return where


def _deepest(error):
    """
    The most specific error under one that has alternatives (oneOf). The alternative that went
    deepest into the value is the one meant: an entry with a stray key is reported at that key,
    not as "not of type null".
    """
    leaves = []

    def walk(e):
        if e.context and not _needs_one_of(e):
            for c in e.context:
                walk(c)
        else:
            leaves.append(e)

    walk(error)
    telling = {"additionalProperties": 3, "not": 2, "anyOf": 2, "required": 0, "type": 0}   # a stray key says most

    def stray(e) -> int:                       # the closest alternative finds the fewest stray keys
        allowed = e.schema.get("properties", {}) if isinstance(e.schema, dict) else {}
        return len(set(e.instance) - set(allowed)) if isinstance(e.instance, dict) else 0

    return max(leaves, key=lambda e: (len(e.absolute_path), telling.get(e.validator, 1), -stray(e)))


def _needs_one_of(error) -> bool:
    """An anyOf of alternatives that each require keys: "needs 'a' or 'b'" says it best."""
    return error.validator == "anyOf" and all(c.validator == "required" for c in error.context)


def _required_keys(schema: Any) -> set:
    """Every key a schema fragment requires, through anyOf/oneOf/allOf."""
    if not isinstance(schema, dict):
        return set()
    keys = set(schema.get("required", ()))
    for combinator in ("anyOf", "oneOf", "allOf"):
        for sub in schema.get(combinator, ()):
            keys |= _required_keys(sub)
    return keys


def schema_problems(validator: Draft202012Validator, instance: Any, where: str) -> list[str]:
    """Every way `instance` fails the schema, as "where.key.path: what is wrong" lines."""
    problems = []
    for error in sorted(validator.iter_errors(instance), key=lambda e: list(e.absolute_path)):
        shown = _deepest(error)
        message = shown.message
        if shown.validator == "pattern" and "description" in shown.schema:   # say what is wanted, not the regex
            message = f"{shown.instance!r} is not {shown.schema['description']}"
        elif shown.validator == "not" and isinstance(shown.instance, dict):   # keys that exclude each other
            banned = _required_keys(shown.validator_value) & set(shown.instance)
            given = set(shown.schema.get("required", ())) & set(shown.instance) - banned
            message = (f"{', '.join(map(repr, sorted(given)))} cannot be combined with {', '.join(map(repr, sorted(banned)))}"
                       if given else f"{' and '.join(map(repr, sorted(banned)))} cannot be combined")
        elif _needs_one_of(shown):
            message = "needs " + " or ".join(" and ".join(map(repr, c.validator_value)) for c in shown.context)
        problems.append(f"{_where(where, shown.absolute_path)}: {message}")
    return problems
