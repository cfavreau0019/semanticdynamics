"""Integration between llm (generate_many / batch / apply_validation) and the validation package."""
import pytest

from llm import (
    ChatRequest, ChatResponse, apply_validation, attach_validation, generate_many, load_responses, run_batch,
    run_batch_results, save_responses,
)
from validation import issues_frame, response_validator, result_of, summary_frame, Validator, RegexCheck


def has_echo(response, request):
    return {"passed": response.text.startswith("echo:"), "seen_request": request is not None}


def test_attach_validation_stores_dict_and_skips_failures():
    ok = attach_validation(ChatResponse("a", "echo:x"), has_echo)
    assert ok.validation == {"passed": True, "seen_request": False} and ok.valid is True
    failed = attach_validation(ChatResponse("b", None, error="boom"), has_echo)
    assert failed.validation is None and failed.valid is None
    assert attach_validation(ChatResponse("c", "x"), None).validation is None


def test_crashing_validator_is_recorded():
    def broken(response, request):
        raise ValueError("oops")

    r = attach_validation(ChatResponse("a", "x"), broken)
    assert r.validation == {"passed": None, "error": "ValueError: oops"} and r.valid is None


def test_object_with_to_dict_accepted():
    res = Validator([RegexCheck("r", required={"x": "echo"})]).validate("echo")
    r = attach_validation(ChatResponse("a", "echo"), lambda resp, req: res)
    assert r.validation["passed"] is True and "checks" in r.validation


def test_generate_many_validates_before_on_result(provider):
    seen_at_checkpoint = []
    reqs = [ChatRequest.from_prompt(p) for p in ("a", "FAIL", "b")]
    out = generate_many(reqs, provider=provider, progress=False, validate=has_echo,
                        on_result=lambda r: seen_at_checkpoint.append((r.request_id, r.validation)))
    assert [r.valid for r in out] == [True, None, True]          # failed call not validated
    assert all(v is None or v["seen_request"] for _, v in seen_at_checkpoint)
    assert sorted(rid for rid, v in seen_at_checkpoint if v) == sorted([reqs[0].id, reqs[2].id])


def test_validation_survives_save_and_load(provider, tmp_path):
    out = generate_many(["a"], provider=provider, progress=False, validate=has_echo)
    save_responses(out, tmp_path / "r.jsonl")
    loaded = load_responses(tmp_path / "r.jsonl")
    assert loaded[0].validation == out[0].validation and loaded[0].valid is True


def test_run_batch_and_results_validate(provider):
    reqs = [ChatRequest.from_prompt(f"q{i}", id=f"s{i}") for i in range(2)]
    starts_with_b = lambda resp, req: {"passed": resp.text.startswith("B:"), "id_match": req.id == resp.request_id}
    out = run_batch(reqs, provider=provider, poll_interval=0, validate=starts_with_b)
    assert all(r.valid and r.validation["id_match"] for r in out)
    again = run_batch_results("batch_1", provider=provider, requests=reqs, validate=starts_with_b)
    assert all(r.validation["id_match"] for r in again)
    assert run_batch_results("batch_1", provider=provider, requests=reqs)[0].validation is None


def test_apply_validation_pairs_requests_by_id():
    reqs = [ChatRequest.from_prompt("x", id="a"), ChatRequest.from_prompt("y", id="b")]
    resps = [ChatResponse("b", "echo:y"), ChatResponse("a", "nope")]
    apply_validation(resps, lambda r, q: {"passed": q.prompt in r.text}, reqs)
    assert [r.valid for r in resps] == [True, False]


# ---- response_validator (validation package side) -------------------------------------
INSTANT = {"Past": "The Fool", "Present": "Two of Cups reversed", "Future": "The Sun"}
READING = "Past: The Fool - beginnings.\nPresent: Two of Cups reversed - tension.\nFuture: The Sun - joy."


@pytest.fixture
def tarot_validate():
    from validation.tarot import spread_validator
    return response_validator(spread_validator(list(INSTANT), min_chars=10), expected="instant")


def test_response_validator_reads_expected_from_metadata(tarot_validate):
    good = ChatResponse("a", READING, finish_reason="stop", metadata={"instant": INSTANT})
    bad = ChatResponse("b", READING.replace("The Sun", "The Moon"), finish_reason="length",
                       metadata={"instant": INSTANT})
    apply_validation([good, bad], tarot_validate)
    assert good.valid is True and bad.valid is False
    assert {i["code"] for i in bad.validation["issues"]} >= {"wrong_card", "truncated"}
    assert result_of(bad).checks["positions"].details["extracted"]["Future"] == "The Moon"


def test_response_validator_falls_back_to_request_metadata(tarot_validate):
    resp = ChatResponse("a", READING, metadata={})
    req = ChatRequest.from_prompt("p", id="a", metadata={"sample": {"instant": INSTANT}})
    v = response_validator(tarot_validate.validator, expected="sample.instant")
    assert v(resp, req)["passed"] is True


def test_response_validator_missing_expected_is_recorded(tarot_validate):
    r = attach_validation(ChatResponse("a", READING, metadata={}), tarot_validate)
    assert r.valid is None and "KeyError" in r.validation["error"]


def test_callable_expected():
    from validation.tarot import spread_validator
    v = response_validator(spread_validator(list(INSTANT), min_chars=10), expected=lambda resp, req: INSTANT)
    assert v(ChatResponse("a", READING), None)["passed"] is True


def test_frames(tarot_validate):
    rs = [ChatResponse("a", READING, metadata={"instant": INSTANT}),
          ChatResponse("b", "Past: The Tower", metadata={"instant": INSTANT}),
          ChatResponse("c", None, error="api down")]
    apply_validation(rs, tarot_validate)
    summary = summary_frame(rs)
    assert summary["valid"].tolist() == [True, False, None]
    issues = issues_frame(rs)
    assert set(issues["request_id"]) == {"b"}
    assert set(issues["code"]) == {"wrong_card", "missing_label", "foreign_value"}
    assert set(issues.loc[issues.code == "missing_label", "label"]) == {"Present", "Future"}
    assert issues_frame([ChatResponse("z", "t")]).empty
