"""
WhiteBoxModel tests against GPT-2 run locally (remote=False). Skipped when nnsight is
missing or GPT-2 isn't in the local Hugging Face cache; nothing is downloaded.
"""
import os

import pytest

pytest.importorskip("nnsight")
torch = pytest.importorskip("torch")

MODEL = "openai-community/gpt2"
TEXT = "An insect usually has a small size."

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def wb():
    os.environ["HF_HUB_OFFLINE"] = "1"
    from llm.whitebox import WhiteBoxModel
    try:
        return WhiteBoxModel(MODEL, remote=False, device_map="cpu")
    except Exception as e:  # not cached / offline
        pytest.skip(f"{MODEL} unavailable locally: {e}")


def test_remote_requires_key(monkeypatch):
    from llm.whitebox import WhiteBoxModel
    monkeypatch.delenv("NDIF_API_TOKEN", raising=False)
    with pytest.raises(ValueError, match="NDIF"):
        WhiteBoxModel(MODEL, remote=True)


def test_lazy_import_from_package():
    import llm
    from llm.whitebox import WhiteBoxModel
    assert llm.WhiteBoxModel is WhiteBoxModel


def test_layer_autodetect(wb):
    assert wb.n_layers == 12  # found transformer.h


def test_bad_layer_path_raises(wb):
    with pytest.raises(AttributeError, match="dotted path"):
        wb._resolve("does.not.exist", (), "layers")


def test_hidden_states_shape_and_tokens(wb):
    hs = wb.hidden_states(TEXT, save_logits=True)
    n_tok = len(wb.tokenizer(TEXT)["input_ids"])
    assert hs.hidden_states.shape == (12, n_tok, 768)
    assert len(hs.tokens) == len(hs.token_ids) == n_tok
    assert hs.next_token_logits.shape == (wb.tokenizer.vocab_size,)
    assert hs.next_token_id == int(hs.next_token_logits.argmax())
    assert hs.mean_pool().shape == hs.last_token().shape == (12, 768)


def test_hidden_states_match_hf_output_hidden_states(wb):
    """Layer i's output should equal HF's hidden_states[i + 1] (index 0 is the embeddings)."""
    hs = wb.hidden_states(TEXT)
    ids = torch.tensor([wb.tokenizer(TEXT)["input_ids"]])
    with torch.no_grad():
        ref = wb.model._model(ids, output_hidden_states=True).hidden_states
    for i in (0, 5):
        torch.testing.assert_close(hs.hidden_states[i], ref[i + 1][0], atol=1e-4, rtol=1e-4)


def test_returned_tensors_are_detached(wb):
    hs = wb.hidden_states(TEXT, save_logits=True)
    many = wb.hidden_states_many([TEXT, "Hi"], save_logits=True, progress=False)
    g = wb.generate("Hello", max_new_tokens=2, do_sample=False)
    tensors = [hs.hidden_states, hs.next_token_logits, *(m.hidden_states for m in many),
               *(m.next_token_logits for m in many), *g.step_hidden_states]
    assert not any(t.requires_grad for t in tensors)
    hs.hidden_states.numpy()  # plotting / numpy interop must work directly


def test_layer_selection_sorted_and_negative(wb):
    hs = wb.hidden_states(TEXT, layers=[-1, 0, 5])
    assert hs.layers == [0, 5, 11] and hs.hidden_states.shape[0] == 3
    full = wb.hidden_states(TEXT)
    torch.testing.assert_close(hs.layer(11), full.hidden_states[11])


MIXED_TEXTS = [TEXT, "Hi", "The cat sat.", "A magnifying glass makes small things look bigger.",
               "A dog ran.", "Birds can fly."]


def test_hidden_states_many_matches_single_with_mixed_lengths(wb):
    """Padded (shorter) texts in a batch must equal their single, unpadded forward pass."""
    lengths = {len(wb.tokenizer(t)["input_ids"]) for t in MIXED_TEXTS}
    assert len(lengths) > 2, "test needs texts of several different lengths"
    many = wb.hidden_states_many(MIXED_TEXTS, batch_size=4, save_logits=True, progress=False)
    assert [m.text for m in many] == MIXED_TEXTS
    for t, m in zip(MIXED_TEXTS, many):
        single = wb.hidden_states(t, save_logits=True)
        assert m.hidden_states.shape == single.hidden_states.shape
        assert m.token_ids == single.token_ids
        torch.testing.assert_close(m.hidden_states, single.hidden_states, atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(m.next_token_logits, single.next_token_logits, atol=1e-3, rtol=1e-3)
        assert m.next_token_id == single.next_token_id


def test_hidden_states_many_uses_fixed_size_batches(wb, monkeypatch):
    traces = []
    real_trace = wb.model.trace

    def counting_trace(*args, **kwargs):
        traces.append(kwargs)
        return real_trace(*args, **kwargs)

    monkeypatch.setattr(wb.model, "trace", counting_trace, raising=False)
    wb.hidden_states_many(MIXED_TEXTS, batch_size=4, progress=False)
    assert len(traces) == 2  # ceil(6 / 4), regardless of token lengths


def test_hidden_states_many_restores_padding_side(wb, monkeypatch):
    assert wb.tokenizer.padding_side == "left"
    wb.hidden_states_many(["Hi", TEXT], progress=False)
    assert wb.tokenizer.padding_side == "left"

    def failing_trace(*args, **kwargs):
        assert wb.tokenizer.padding_side == "right"
        raise RuntimeError("remote failure")

    monkeypatch.setattr(wb.model, "trace", failing_trace, raising=False)
    with pytest.raises(RuntimeError, match="remote failure"):
        wb.hidden_states_many(["Hi"], progress=False)
    assert wb.tokenizer.padding_side == "left"


def test_generate_with_hidden_states(wb):
    prompt = "The lighthouse keeper found"
    g = wb.generate(prompt, max_new_tokens=6, do_sample=False)
    n_prompt = len(wb.tokenizer(prompt)["input_ids"])
    assert len(g.new_token_ids) == 6 and g.text == wb.tokenizer.decode(g.new_token_ids)
    assert len(g.step_hidden_states) == 6
    assert g.prompt_states().shape == (12, n_prompt, 768)
    assert g.step_hidden_states[1].shape == (12, 1, 768)
    assert g.new_token_states().shape == (12, 6, 768)


def test_generate_matches_plain_hf_greedy(wb):
    prompt = "The lighthouse keeper found"
    g = wb.generate(prompt, max_new_tokens=5, save_hidden_states=False, do_sample=False)
    ids = torch.tensor([wb.tokenizer(prompt)["input_ids"]])
    ref = wb.model._model.generate(ids, max_new_tokens=5, do_sample=False,
                                   pad_token_id=wb.tokenizer.eos_token_id)
    assert g.new_token_ids == ref[0, ids.shape[1]:].tolist()
    assert g.step_hidden_states == []


def test_generate_last_token_only_with_layer_subset(wb):
    g = wb.generate("Hello there", max_new_tokens=4, last_token_only=True, layers=[3], do_sample=False)
    assert [s.shape for s in g.step_hidden_states] == [(1, 1, 768)] * 4
    assert g.new_token_states().shape == (1, 4, 768)
