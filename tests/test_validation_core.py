import json

import pytest

from validation import (
    ERROR, WARNING, Check, NonEmpty, NoRefusal, NotTruncated, RegexCheck, ValidationContext, ValidationResult,
    Validator,
)


class AlwaysWarn(Check):
    name = "always_warn"

    def run(self, ctx):
        return self.result([self.issue("meh", "just a warning", WARNING)], seen=len(ctx.text))


class Crashes(Check):
    name = "crashes"

    def run(self, ctx):
        raise RuntimeError("bug in check")


def test_warnings_do_not_fail():
    res = Validator([AlwaysWarn()]).validate("text")
    assert res.passed and len(res.warnings) == 1 and not res.errors


def test_crashing_check_is_recorded_not_raised():
    res = Validator([Crashes(), AlwaysWarn()]).validate("text")
    assert not res.passed
    assert res.errors[0].code == "check_crashed" and "bug in check" in res.errors[0].message
    assert "always_warn" in res.checks  # later checks still ran


def test_duplicate_check_names_rejected():
    with pytest.raises(ValueError, match="unique"):
        Validator([AlwaysWarn(), AlwaysWarn()])


def test_none_text_treated_as_empty():
    res = Validator([NonEmpty()]).validate(None)
    assert not res.passed and res.errors[0].code == "too_short"


def test_to_dict_is_json_and_round_trips():
    res = Validator([NonEmpty(10), AlwaysWarn()], name="v").validate("short")
    d = res.to_dict()
    json.dumps(d)
    assert d["passed"] is False and d["validator"] == "v" and d["n_errors"] == 1 and d["n_warnings"] == 1
    assert d["checks"]["always_warn"]["details"] == {"seen": 5}
    back = ValidationResult.from_dict(d)
    assert back.passed == res.passed
    assert [i.code for i in back.issues] == [i.code for i in res.issues]
    assert back.checks["non_empty"].passed is False


def test_summary_lists_issues():
    s = Validator([NonEmpty(10)]).validate("x").summary()
    assert s.startswith("FAIL") and "non_empty/too_short" in s


def test_not_truncated():
    v = Validator([NotTruncated()])
    assert not v.validate("x", finish_reason="length").passed
    assert v.validate("x", finish_reason="stop").passed
    assert v.validate("x").passed  # unknown finish reason: no opinion
    assert Validator([NotTruncated(severity=WARNING)]).validate("x", finish_reason="length").passed


@pytest.mark.parametrize("text", [
    "I'm sorry, but I can't help with that.",
    "I cannot provide tarot readings.",
    "As an AI language model, I don't believe in tarot.",
])
def test_no_refusal_flags(text):
    assert not Validator([NoRefusal()]).validate(text).passed


def test_no_refusal_allows_normal_text():
    assert Validator([NoRefusal()]).validate("The Tower suggests sudden change. I can help you reflect.").passed


def test_regex_check_required_and_forbidden():
    check = RegexCheck("fmt", required={"overview": r"\boverall\b"}, forbidden={"url": r"https?://"})
    ok = Validator([check]).validate("Overall, a hopeful spread.")
    bad = Validator([check]).validate("See https://example.com")
    assert ok.passed
    assert {i.code for i in bad.errors} == {"missing_pattern", "forbidden_pattern"}
    assert bad.checks["fmt"].details["forbidden_hits"] == {"url": "https://"}


def test_context_passes_through_to_checks():
    class NeedsMeta(Check):
        name = "meta"

        def run(self, ctx: ValidationContext):
            return self.result(got=ctx.metadata.get("k"), expected=ctx.expected)

    res = Validator([NeedsMeta()]).validate("t", expected={"a": 1}, metadata={"k": "v"})
    assert res.checks["meta"].details == {"got": "v", "expected": {"a": 1}}
