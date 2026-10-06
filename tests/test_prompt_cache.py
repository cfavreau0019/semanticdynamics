"""Prompt (KV) caching: cached tokens are recorded, and templates keep what varies at the end."""
from types import SimpleNamespace

from data_generation.pipeline import GenerationConfig, prepare_run, run_live, summarize
from data_generation.tarot import TarotReadings
from evaluation.rubric import get_rubric
from llm import ChatResponse, Usage
from llm.providers import register_provider
from llm.providers.openai_compatible import _usage
from personas import PersonaSet
from prompts import get_template
from test_data_generation import TarotScripted, isolated_registry, reading_from_prompt  # noqa: F401


def shared_prefix(a: str, b: str) -> int:
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))


def test_usage_reads_cached_tokens_however_the_provider_reports_them():
    assert _usage({"prompt_tokens": 1200, "completion_tokens": 50, "total_tokens": 1250,
                   "prompt_tokens_details": {"cached_tokens": 1024}}) == Usage(1200, 50, 1250, 1024)
    sdk = SimpleNamespace(prompt_tokens=1200, completion_tokens=50, total_tokens=1250,
                          prompt_tokens_details=SimpleNamespace(cached_tokens=1024))
    assert _usage(sdk).cached_tokens == 1024
    assert _usage({"prompt_tokens": 9, "completion_tokens": 1, "total_tokens": 10}).cached_tokens == 0   # e.g. vLLM
    assert _usage({"prompt_tokens": 9, "prompt_tokens_details": None}).cached_tokens == 0
    assert (Usage(10, 2, 12, 4) + Usage(10, 2, 12)).cached_tokens == 4


def test_cached_tokens_are_stored_and_summed(tmp_path, isolated_registry):
    class Cached(TarotScripted):
        def chat(self, request):
            return ChatResponse(request.id, reading_from_prompt(request.prompt), model="m", finish_reason="stop",
                                usage=Usage(1300, 700, 2000, 1024), metadata=request.metadata)

    provider = Cached()
    register_provider("cached", lambda **kw: provider)
    app = TarotReadings()
    store, requests = prepare_run(app, GenerationConfig(n=3, provider="cached"), root=tmp_path)
    summary = run_live(store, app, requests, progress=False)
    assert [r["cached_tokens"] for r in store.read("responses")] == [1024] * 3
    assert (summary["cached_tokens"], summary["prompt_tokens"]) == (3072, 3900)

    rows = store.read("responses")                      # a run written before the column existed
    store.path("responses").write_text("", encoding="utf-8")
    for r in rows:
        del r["cached_tokens"]
    with open(store.path("responses"), "a", encoding="utf-8") as f:
        import json
        f.writelines(json.dumps(r) + "\n" for r in rows)
    assert summarize(store)["cached_tokens"] == 0


def test_reading_template_v3_varies_only_at_the_end():
    t = get_template("celtic_cross_v3")
    a = t.render(cards={"Present": "Death", "Outcome": "The Sun reversed"}).user
    b = t.render(cards={"Present": "The Fool", "Outcome": "Ten of Cups"}).user
    assert a[:shared_prefix(a, b)].rstrip().endswith("Here is the spread to read:\n\nPresent:")


def test_evaluation_template_v2_puts_the_form_first_and_the_evaluator_last():
    rubric, people = get_rubric("celtic_cross_reading_v1"), list(PersonaSet.load("tarot_personas"))
    labels = {"positions": ["Present", "Outcome"]}
    common = {"context": {"question_asked": "none", "deck": "RWS", "reversals_used": "no"},
              "rubric_form": rubric.render_form(labels), "response_format": rubric.render_response_format(labels)}
    render = lambda cards, text, persona: get_template("celtic_cross_evaluation_v2").render(
        cards=cards, reading_text=text, persona=persona, **common).user
    one = render({"Present": "Death", "Outcome": "The Sun"}, "Reading one.", people[0])
    other_reading = render({"Present": "The Fool", "Outcome": "The Moon"}, "Reading two.", people[0])
    other_persona = render({"Present": "Death", "Outcome": "The Sun"}, "Reading one.", people[1])

    static = one[:shared_prefix(one, other_reading)]
    assert common["rubric_form"] in static and common["response_format"] in static     # all of the fixed part
    assert static.rstrip().endswith("- Cards drawn, by position:\nPresent:")
    assert len(static) > 0.8 * len(one)

    same_reading = one[:shared_prefix(one, other_persona)]                               # personas share the reading
    assert "<reading>\nReading one.\n</reading>" in same_reading and "# Who you are\n\nName:" in same_reading
    assert one.rstrip().endswith("return only the JSON object.")

    v1 = lambda cards, text: get_template("celtic_cross_evaluation_v1").render(
        cards=cards, reading_text=text, persona=people[0], **common).user
    a, b = v1({"Present": "Death", "Outcome": "The Sun"}, "Reading one."), v1({"Present": "The Fool"}, "Reading two.")
    assert common["rubric_form"] not in a[:shared_prefix(a, b)]                          # why v2 exists
    assert "No evaluator persona is given" in get_template("celtic_cross_evaluation_v2").render(
        cards={"Present": "Death"}, reading_text="x", **common).system
