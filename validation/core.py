"""
Core validation types. Nothing here knows about tarot or about the llm package:
a Validator runs Checks over a piece of text (plus optional context such as the
input it was generated from) and returns a JSON-serialisable ValidationResult.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Any, Optional, Sequence

ERROR = "error"      # the output is wrong / unusable for its purpose
WARNING = "warning"  # suspicious, worth reviewing, but not disqualifying


@dataclass
class Issue:
    """
    One finding. `code` is machine-readable (group/filter on it); `message` is for humans.
    `label` names the field it concerns (e.g. a Celtic Cross position), if any.
    """
    check: str
    code: str
    message: str
    severity: str = ERROR
    label: Optional[str] = None
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationContext:
    """
    Everything a check may look at.

    text          — the output being validated
    expected      — what the output should reflect (e.g. {position: card} that was supplied)
    finish_reason — why generation stopped ("length" means truncated), when known
    metadata      — any other request/response data a custom check may need
    """
    text: str
    expected: Any = None
    finish_reason: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CheckResult:
    name: str
    issues: list[Issue] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)  # what the check extracted/measured

    @property
    def passed(self) -> bool:
        return not any(i.severity == ERROR for i in self.issues)


class Check(ABC):
    """A single rule. Subclasses implement run(); they should not raise on bad text."""

    name: str = "check"

    @abstractmethod
    def run(self, ctx: ValidationContext) -> CheckResult:
        ...

    def result(self, issues=None, **details) -> CheckResult:
        return CheckResult(self.name, list(issues or []), details)

    def issue(self, code: str, message: str, severity: str = ERROR, label: Optional[str] = None, **data) -> Issue:
        return Issue(self.name, code, message, severity, label, data)


@dataclass
class ValidationResult:
    passed: bool
    checks: dict[str, CheckResult]
    validator: Optional[str] = None

    @property
    def issues(self) -> list[Issue]:
        return [i for c in self.checks.values() for i in c.issues]

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == ERROR]

    @property
    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == WARNING]

    def summary(self) -> str:
        head = f"{'PASS' if self.passed else 'FAIL'} ({len(self.errors)} errors, {len(self.warnings)} warnings)"
        lines = [f"  [{i.severity}] {i.check}/{i.code}: {i.message}" for i in self.issues]
        return "\n".join([head, *lines])

    def to_dict(self) -> dict:
        """The form stored on ChatResponse.validation and written to JSONL."""
        return {
            "passed": self.passed,
            "validator": self.validator,
            "n_errors": len(self.errors),
            "n_warnings": len(self.warnings),
            "issues": [asdict(i) for i in self.issues],
            "checks": {name: {"passed": c.passed, "details": c.details} for name, c in self.checks.items()},
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ValidationResult":
        by_check: dict[str, list[Issue]] = {}
        for i in d.get("issues", []):
            by_check.setdefault(i["check"], []).append(Issue(**i))
        checks = {name: CheckResult(name, by_check.get(name, []), c.get("details", {}))
                  for name, c in d.get("checks", {}).items()}
        return cls(passed=d["passed"], checks=checks, validator=d.get("validator"))


class Validator:
    """
    Runs a list of checks. A check that raises is recorded as an error issue
    ("check_crashed") instead of propagating, so a buggy rule can't abort a run.
    """

    def __init__(self, checks: Sequence[Check], name: Optional[str] = None):
        names = [c.name for c in checks]
        if len(set(names)) != len(names):
            raise ValueError(f"Check names must be unique; got {names}")
        self.checks = list(checks)
        self.name = name

    def __repr__(self) -> str:
        return f"Validator({self.name!r}, checks={[c.name for c in self.checks]})"

    def run(self, ctx: ValidationContext) -> ValidationResult:
        results = {}
        for check in self.checks:
            try:
                results[check.name] = check.run(ctx)
            except Exception as e:
                results[check.name] = CheckResult(check.name, [Issue(
                    check.name, "check_crashed", f"{type(e).__name__}: {e}", ERROR)])
        return ValidationResult(all(r.passed for r in results.values()), results, self.name)

    def validate(
        self,
        text: Optional[str],
        expected: Any = None,
        finish_reason: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> ValidationResult:
        return self.run(ValidationContext(text or "", expected, finish_reason, metadata or {}))
