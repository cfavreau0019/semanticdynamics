"""
Glue between validators and llm.ChatResponse objects. Kept here (not in llm) so the llm
package never depends on validation: llm only knows it may call a
`validate(response, request) -> dict` function and store the dict on response.validation.

    from llm import generate_many
    from validation import response_validator
    from validation.tarot import celtic_cross_validator

    validate = response_validator(celtic_cross_validator(), expected="instant")
    responses = generate_many(reqs, provider="featherless", validate=validate, on_result=checkpoint)
    [r.valid for r in responses]
"""
from typing import Any, Callable, Iterable, Optional

from validation.core import ValidationResult, Validator

ExpectedSpec = Optional[str | Callable[[Any, Any], Any]]


def _expected(spec: ExpectedSpec, response, request) -> Any:
    if spec is None:
        return None
    if callable(spec):
        return spec(response, request)
    # dotted path into metadata ("instant" or "sample.instant"); response metadata first
    for source in (getattr(response, "metadata", None), getattr(request, "metadata", None)):
        value = source
        for key in spec.split("."):
            value = value.get(key) if isinstance(value, dict) else None
        if value is not None:
            return value
    raise KeyError(f"expected={spec!r} not found in response/request metadata")


def response_validator(validator: Validator, expected: ExpectedSpec = None) -> Callable[..., dict]:
    """
    Wrap a Validator as the `validate=` hook accepted by llm.generate_many / run_batch /
    run_batch_results / apply_validation.

    expected — where the "what should the output say" data comes from:
               a dotted metadata key (e.g. "instant"), or a callable (response, request) -> value.
    """
    def validate(response, request=None) -> dict:
        return validator.validate(
            response.text,
            expected=_expected(expected, response, request),
            finish_reason=getattr(response, "finish_reason", None),
            metadata=getattr(response, "metadata", None) or {},
        ).to_dict()

    validate.validator = validator
    return validate


def result_of(response) -> Optional[ValidationResult]:
    """The ValidationResult stored on a response (None if not validated / validator crashed)."""
    v = getattr(response, "validation", None)
    if not v or v.get("passed") is None or "checks" not in v:
        return None
    return ValidationResult.from_dict(v)


def summary_frame(responses: Iterable):
    """One row per response: id, ok, valid, error/warning counts."""
    import pandas as pd
    rows = []
    for r in responses:
        v = r.validation or {}
        rows.append({"request_id": r.request_id, "ok": r.ok, "valid": v.get("passed"),
                     "n_errors": v.get("n_errors"), "n_warnings": v.get("n_warnings"),
                     "validator_error": v.get("error")})
    return pd.DataFrame(rows)


def issues_frame(responses: Iterable):
    """One row per issue across all responses — group by code/label to see what models get wrong."""
    import pandas as pd
    rows = [{"request_id": r.request_id, **{k: i.get(k) for k in ("check", "code", "severity", "label", "message")}}
            for r in responses for i in (r.validation or {}).get("issues", [])]
    return pd.DataFrame(rows, columns=["request_id", "check", "code", "severity", "label", "message"])
