"""Generic, LLM-free checks. Configure them for a task (see validation/tarot.py for an example)."""
import re
from typing import Callable, Mapping, Optional

from validation.core import ERROR, WARNING, Check, CheckResult, ValidationContext
from validation.extract import LabeledEntryExtractor, ValueMatcher

# (expected, got) -> None when they agree, else (issue_code, message)
Comparator = Callable[[str, str], Optional[tuple[str, str]]]


def exact_comparator(expected: str, got: str) -> Optional[tuple[str, str]]:
    if expected.casefold() == got.casefold():
        return None
    return "value_mismatch", f"expected {expected!r}, got {got!r}"


class NonEmpty(Check):
    name = "non_empty"

    def __init__(self, min_chars: int = 1):
        self.min_chars = min_chars

    def run(self, ctx: ValidationContext) -> CheckResult:
        n = len(ctx.text.strip())
        issues = [] if n >= self.min_chars else [
            self.issue("too_short", f"output has {n} characters (minimum {self.min_chars})")]
        return self.result(issues, n_chars=n)


class NotTruncated(Check):
    """Fails when generation stopped at the token limit (finish_reason == 'length')."""
    name = "not_truncated"

    def __init__(self, severity: str = ERROR):
        self.severity = severity

    def run(self, ctx: ValidationContext) -> CheckResult:
        issues = []
        if ctx.finish_reason == "length":
            issues.append(self.issue("truncated", "output was cut off at max_tokens", self.severity))
        return self.result(issues, finish_reason=ctx.finish_reason)


class RegexCheck(Check):
    """
    required  — {name: pattern} that must each match somewhere
    forbidden — {name: pattern} that must not match
    """

    def __init__(self, name: str, required: Optional[Mapping[str, str]] = None,
                 forbidden: Optional[Mapping[str, str]] = None, severity: str = ERROR, flags: int = re.IGNORECASE):
        self.name = name
        self.required = {k: re.compile(p, flags) for k, p in (required or {}).items()}
        self.forbidden = {k: re.compile(p, flags) for k, p in (forbidden or {}).items()}
        self.severity = severity

    def run(self, ctx: ValidationContext) -> CheckResult:
        issues, hits = [], {}
        for key, pat in self.required.items():
            if not pat.search(ctx.text):
                issues.append(self.issue("missing_pattern", f"required pattern {key!r} not found",
                                         self.severity, pattern=key))
        for key, pat in self.forbidden.items():
            m = pat.search(ctx.text)
            if m:
                hits[key] = m.group(0)
                issues.append(self.issue("forbidden_pattern", f"found {key!r}: {m.group(0)!r}",
                                         self.severity, pattern=key, match=m.group(0)))
        return self.result(issues, forbidden_hits=hits)


REFUSAL_PATTERNS = {
    "cannot_help": r"\bI(?:'m| am)? (?:sorry,? (?:but )?)?(?:can(?:no|')t|unable to|won't) (?:help|assist|provide|do that)",
    "as_an_ai": r"\bas an AI(?: language model)?\b",
}


def NoRefusal(severity: str = ERROR) -> RegexCheck:
    """Flags common refusal / disclaimer phrasings."""
    return RegexCheck("no_refusal", forbidden=REFUSAL_PATTERNS, severity=severity)


class LabeledValuesCheck(Check):
    """
    For outputs that should restate a {label: value} input (e.g. {position: card}):

      missing_label     (error)   label never appears as an entry
      missing_value     (error)   label appears but names no recognisable value
      <comparator code> (error)   value differs from the expected one (default "value_mismatch")
      conflicting_entries (warn)  label appears on several lines with different values
      out_of_order      (warn)    entries appear in a different order than expected (check_order)

    `ctx.expected` must be a mapping {label: expected_value}. Labels the extractor was not
    configured with are reported as unknown_label. Entries found only in prose (inline
    fallback) count as present; their mode is recorded in details.
    """

    def __init__(
        self,
        extractor: LabeledEntryExtractor,
        name: str = "labeled_values",
        compare: Comparator = exact_comparator,
        check_values: bool = True,
        check_order: bool = False,
        allow_inline: bool = True,
    ):
        self.name = name
        self.extractor = extractor
        self.compare = compare
        self.check_values = check_values
        self.check_order = check_order
        self.allow_inline = allow_inline

    def run(self, ctx: ValidationContext) -> CheckResult:
        expected = ctx.expected
        if not isinstance(expected, Mapping):
            return self.result([self.issue("no_expected", "ctx.expected must be a {label: value} mapping")])

        issues = []
        unknown = [l for l in expected if l not in self.extractor.labels]
        for label in unknown:
            issues.append(self.issue("unknown_label", f"extractor has no aliases for {label!r}", label=label))

        # first line entry with a value wins; remember all values seen per label
        chosen, seen = {}, {}
        for e in self.extractor.extract(ctx.text):
            if e.value is not None:
                seen.setdefault(e.label, []).append(e.value)
            if e.label not in chosen or (chosen[e.label].value is None and e.value is not None):
                chosen[e.label] = e

        missing_or_empty = [l for l in expected if l not in chosen or chosen[l].value is None]
        if self.allow_inline and missing_or_empty:
            chosen.update(self.extractor.extract_inline(ctx.text, missing_or_empty))

        for label, want in expected.items():
            if label in unknown:
                continue
            entry = chosen.get(label)
            if entry is None:
                issues.append(self.issue("missing_label", f"{label!r} not found", label=label))
                continue
            if entry.value is None:
                msg = (f"{label!r} is listed without a value" if not entry.segment
                       else f"{label!r} found but no recognisable value in {entry.segment!r}")
                issues.append(self.issue("missing_value", msg, label=label, segment=entry.segment))
                continue
            if self.check_values:
                problem = self.compare(want, entry.value)
                if problem:
                    code, msg = problem
                    issues.append(self.issue(code, f"{label}: {msg}", label=label, expected=want, got=entry.value))
            values = {self.extractor.matcher.identity(v) for v in seen.get(label, [])}
            if len(values) > 1:
                issues.append(self.issue("conflicting_entries", f"{label!r} appears with different values: "
                                                                f"{sorted(values)}", WARNING, label=label))

        if self.check_order:
            line_entries = sorted((e for e in chosen.values() if e.mode == "line"), key=lambda e: e.line)
            got_order = [e.label for e in line_entries if e.label in expected]
            want_order = [l for l in expected if l in got_order]
            if got_order != want_order:
                issues.append(self.issue("out_of_order", f"entries appear as {got_order}", WARNING,
                                         order=got_order))

        return self.result(
            issues,
            extracted={l: e.value for l, e in chosen.items()},
            modes={l: e.mode for l, e in chosen.items()},
            n_expected=len(expected),
            n_found=sum(1 for l in expected if l in chosen and chosen[l].value is not None),
        )


class ForeignValuesCheck(Check):
    """
    Flags vocabulary values mentioned anywhere in the text that were not part of the input
    (e.g. a card that wasn't drawn). Compares identities (matcher.identity), so orientation
    or other qualifiers are ignored.
    """

    def __init__(self, matcher: ValueMatcher, name: str = "foreign_values", severity: str = WARNING):
        self.name = name
        self.matcher = matcher
        self.severity = severity

    def run(self, ctx: ValidationContext) -> CheckResult:
        if not isinstance(ctx.expected, Mapping):
            return self.result([self.issue("no_expected", "ctx.expected must be a {label: value} mapping")])
        allowed = {self.matcher.identity(v) for v in ctx.expected.values()}
        foreign = sorted({ident for ident, _ in self.matcher.find_all(ctx.text)} - allowed)
        issues = [self.issue("foreign_value", f"mentions {v!r}, which was not in the input", self.severity, value=v)
                  for v in foreign]
        return self.result(issues, foreign=foreign)
