from llm.types import BatchJob, ChatRequest, ChatResponse, Usage, BATCH_COMPLETED, BATCH_RUNNING


def test_from_prompt_builds_messages_and_params():
    r = ChatRequest.from_prompt("hi", system="sys", model="m", id="a", metadata={"k": 1}, max_tokens=5)
    assert r.messages == [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
    assert (r.model, r.id, r.metadata, r.params) == ("m", "a", {"k": 1}, {"max_tokens": 5})


def test_from_prompt_without_system():
    r = ChatRequest.from_prompt("hi", system=None)
    assert r.messages == [{"role": "user", "content": "hi"}]


def test_ids_are_unique_by_default():
    assert ChatRequest.from_prompt("a").id != ChatRequest.from_prompt("a").id


def test_prompt_property_returns_last_user_message():
    r = ChatRequest(messages=[{"role": "user", "content": "first"},
                              {"role": "assistant", "content": "reply"},
                              {"role": "user", "content": "second"}])
    assert r.prompt == "second"
    assert ChatRequest(messages=[]).prompt == ""


def test_request_dict_round_trip():
    r = ChatRequest.from_prompt("hi", system="s", metadata={"x": [1, 2]}, temperature=0.3)
    assert ChatRequest.from_dict(r.to_dict()) == r


def test_usage_addition():
    assert Usage(1, 2, 3) + Usage(10, 20, 30) == Usage(11, 22, 33)


def test_response_ok_and_to_dict_drops_raw():
    ok = ChatResponse("id", "text", raw=object())
    bad = ChatResponse("id", None, error="nope")
    assert ok.ok and not bad.ok
    assert "raw" not in ok.to_dict()


def test_batch_job_done_and_to_dict():
    assert BatchJob("b", "p", BATCH_COMPLETED).done
    assert not BatchJob("b", "p", BATCH_RUNNING).done
    assert "raw" not in BatchJob("b", "p", BATCH_COMPLETED, raw=object()).to_dict()
