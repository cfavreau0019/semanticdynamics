"""
Validation checks for evaluator output, built on the `validation` package so they run
through the same validate= hook (and validation retries) as everything else.

Evaluators answer in JSON. extract_json() tolerates the usual wrappers (code fences, a
sentence before or after); RubricAnswerCheck then holds the answer to the rubric, and
JsonFieldsCheck covers simpler structured answers such as persona expectations.
"""
import json
import re
from typing import Any, Callable, Mapping, Optional, Sequence

from evaluation.rubric import Labels, Rubric
from validation.core import ERROR, WARNING, Check, CheckResult, ValidationContext


def extract_json(text: Optional[str]) -> Any:
    """
    The JSON object in a model's reply, or None. Accepts bare JSON, JSON in a ``` fence, and
    JSON with prose around it (the outermost {...} is taken).
    """
    if not text:
        return None
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.S | re.I)
    candidates = [text] + ([fenced.group(1).strip()] if fenced else [])
    start, end = text.find("{"), text.rfind("}")
    if 0 <= start < end:
        candidates.append(text[start:end + 1])
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


def normalize_for_quote(text: str) -> str:
    """Comparison form for quotes: straight quotes, collapsed whitespace, no markdown emphasis, lower case."""
    text = text.translate(str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"',
                                         "–": "-", "—": "-", "…": "..."}))
    text = re.sub(r"[*_`]", "", text)
    return re.sub(r"\s+", " ", text).strip().casefold()


def quote_in_text(quote: str, text: str) -> bool:
    """Is the quote a passage of the text? Surrounding quote marks and ellipses are ignored, and a
    quote elided with '...' matches if every part appears, in order."""
    haystack = normalize_for_quote(text)
    parts = [p.strip(" \"'.") for p in normalize_for_quote(quote).split("...")]
    position = 0
    for part in (p for p in parts if p):
        found = haystack.find(part, position)
        if found < 0:
            return False
        position = found + len(part)
    return any(parts)


class RubricAnswerCheck(Check):
    """
    Holds a JSON answer to a rubric:

      invalid_json, bad_structure, missing_item, wrong_length, not_a_whole_number,
      out_of_range, na_not_allowed, na_not_applicable, unknown_red_flag,
      red_flag_without_quote, quote_not_in_text, missing_outputs, bad_output      (errors)
      missing_justification, unknown_item, na_expected, unknown_item_reference      (warnings)

    labels      — (expected) -> {"positions": [...]} for the rubric's per-label items
    quoted_text — (expected) -> the text being evaluated; red-flag quotes must come from it
    na_required — (expected) -> {item_id: bool} where the facts decide whether N/A applies
    `expected` is ctx.expected (the input payload of the request).
    """
    name = "rubric_answer"

    def __init__(self, rubric: Rubric, labels: Callable[[Any], Labels],
                 quoted_text: Optional[Callable[[Any], Optional[str]]] = None,
                 na_required: Optional[Callable[[Any], Mapping[str, bool]]] = None):
        self.rubric = rubric
        self.labels = labels
        self.quoted_text = quoted_text
        self.na_required = na_required

    def run(self, ctx: ValidationContext) -> CheckResult:
        data = extract_json(ctx.text)
        if data is None:
            return self.result([self.issue("invalid_json", "the reply is not a JSON object")])
        answer, problems = self.rubric.read(
            data, self.labels(ctx.expected), self.na_required(ctx.expected) if self.na_required else None)
        issues = [self.issue(p.code, p.message, ERROR if p.severity == "error" else WARNING, p.item)
                  for p in problems]
        if answer is None:
            return self.result(issues)
        source = self.quoted_text(ctx.expected) if self.quoted_text else None
        if source is not None:
            for flag in answer.red_flags:
                if not quote_in_text(flag["quote"], source):
                    issues.append(self.issue("quote_not_in_text", f"{flag['id']}: the quoted passage does not "
                                             f"appear in the text: {flag['quote'][:80]!r}", label=flag["id"]))
        return self.result(issues, n_scored=sum(v is not None for v in answer.scores.values()),
                           n_na=sum(v is None for v in answer.scores.values()),
                           red_flags=[f["id"] for f in answer.red_flags])


class JsonFieldsCheck(Check):
    """
    A JSON object with the given fields. spec: {field: kind} where kind is
      "text"            non-empty string
      "list"            list of strings (may be empty)
      ["a", "b", ...]   one of these values
    """
    name = "json_fields"

    def __init__(self, spec: Mapping[str, Any], name: str = "json_fields"):
        self.spec = dict(spec)
        self.name = name

    def read(self, text: Optional[str]) -> tuple[Optional[dict], list[tuple[str, str, str]]]:
        """(normalised object, [(code, field, message)]); the object is None if unusable."""
        data = extract_json(text)
        if not isinstance(data, dict):
            return None, [("invalid_json", "", "the reply is not a JSON object")]
        out, problems = {}, []
        for name, kind in self.spec.items():
            value = data.get(name)
            if isinstance(kind, (list, tuple)):
                v = str(value or "").strip().lower().replace(" ", "_")
                if v in kind:
                    out[name] = v
                else:
                    problems.append(("bad_field", name, f"{name} must be one of {list(kind)}, got {value!r}"))
            elif kind == "list":
                if value is None:
                    value = []
                if isinstance(value, list):
                    out[name] = [str(x).strip() for x in value if str(x).strip()]
                else:
                    problems.append(("bad_field", name, f"{name} must be a list"))
            elif isinstance(value, str) and value.strip():
                out[name] = value.strip()
            else:
                problems.append(("bad_field", name, f"{name} must be non-empty text"))
        return (out if not problems else None), problems

    def run(self, ctx: ValidationContext) -> CheckResult:
        _, problems = self.read(ctx.text)
        return self.result([self.issue(code, msg, label=field or None) for code, field, msg in problems])
