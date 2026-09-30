"""
Post-processing hooks applied to responses before they are returned or written to disk.

A `validate` function has the signature (response, request) -> dict (or an object with
to_dict()). Its result is stored on response.validation. The llm package does not know
what the dict contains beyond an optional "passed" key; validators live in the separate
`validation` package (validation.response_validator builds one).
"""
from dataclasses import asdict
from typing import Any, Callable, Optional, Sequence

from llm.types import ChatRequest, ChatResponse

ValidateFn = Callable[[ChatResponse, Optional[ChatRequest]], Any]

# Regenerations allowed after a response fails validation (only used when a validator is given).
DEFAULT_VALIDATION_RETRIES = 2


def attempt_record(response: ChatResponse) -> dict[str, Any]:
    """What is kept of an attempt that is not returned: enough to analyse why it failed."""
    return {
        "text": response.text,
        "finish_reason": response.finish_reason,
        "usage": asdict(response.usage) if response.usage else None,
        "validation": response.validation,
        "error": response.error,
    }


def merge_retry(previous: ChatResponse, retry: ChatResponse) -> ChatResponse:
    """
    Combine a response with its regeneration. A successful retry becomes the returned
    response, inheriting the history; if the retry failed at the API level, the previous
    (invalid but usable) response is kept and the error is recorded. Either way `attempts`
    is the total and `failed_attempts` lists everything not returned, oldest first.
    """
    total = previous.attempts + retry.attempts
    if retry.ok:
        retry.failed_attempts = previous.failed_attempts + [attempt_record(previous)] + retry.failed_attempts
        retry.attempts = total
        return retry
    previous.failed_attempts = previous.failed_attempts + retry.failed_attempts + [attempt_record(retry)]
    previous.attempts = total
    return previous


def needs_retry(response: ChatResponse) -> bool:
    """Only content that was validated and failed; not API errors, not validator crashes."""
    return response.ok and response.valid is False


def attach_validation(
    response: ChatResponse, validate: Optional[ValidateFn], request: Optional[ChatRequest] = None,
) -> ChatResponse:
    """
    Run `validate` on a successful response and store the result on response.validation.
    Failed API calls (response.ok is False) are left unvalidated. A validator that raises
    is recorded as {"passed": None, "error": ...} rather than propagating, so a buggy
    validator can't abort a (paid) run.
    """
    if validate is None or not response.ok:
        return response
    try:
        result = validate(response, request)
        response.validation = result.to_dict() if hasattr(result, "to_dict") else dict(result)
    except Exception as e:
        response.validation = {"passed": None, "error": f"{type(e).__name__}: {e}"}
    return response


def apply_validation(
    responses: Sequence[ChatResponse],
    validate: ValidateFn,
    requests: Optional[Sequence[ChatRequest]] = None,
) -> list[ChatResponse]:
    """
    Validate responses in place (and return them), pairing each with its request by id
    when `requests` is given. Use it for any path without a built-in hook, e.g. to
    re-validate responses loaded from disk after improving a validator.
    """
    by_id = {r.id: r for r in requests or []}
    return [attach_validation(r, validate, by_id.get(r.request_id)) for r in responses]
