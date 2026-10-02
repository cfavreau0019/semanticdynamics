"""data_generation: schema, store, tarot application, live/batch pipeline, resume, CLI."""
import json
import random
import re
import threading
from pathlib import Path

import pytest

import llm.providers as providers
from llm import ChatResponse, Usage
from llm.providers import LLMProvider, register_provider
from data_generation import schema
from data_generation.generate_tarot import main
from data_generation.pipeline import (
    GenerationConfig, collect_batch, final_responses, list_runs, load_config, prepare_run, rebuild_requests,
    run_live,
    submit_batch, summarize,
)
from data_generation.store import RunStore
from data_generation.tarot import TarotReadings
from prompts import get_template, parse_id as parse_template_id
from conftest import FakeOpenAIClient, batch_body
from llm.providers.openai_compatible import OpenAICompatibleProvider

FILLER = "This card carries a meaningful theme for you, worth careful reflection in the days ahead."


def reading_from_prompt(prompt: str, drop: str = None) -> str:
    """A well-formed reading restating the prompt's 'Position: Card' lines (optionally omitting one)."""
    lines = re.findall(r"^([^:\n]+): (.+)$", prompt, flags=re.M)
    body = "\n\n".join(f"{i}. {pos}: {card} - {FILLER}" for i, (pos, card) in enumerate(lines, 1) if pos != drop)
    return f"Here is your reading.\n\n{body}\n\nOverall, a thoughtful spread."


class TarotScripted(LLMProvider):
    """
    Answers with a valid reading, except: prompts whose index (per `plan`) says otherwise.
    plan[input_index] = list of behaviours per call: "ok" | "invalid" | "error" | "interrupt".
    """
    name = "tarot-scripted"

    def __init__(self, plan=None):
        super().__init__(default_model="scripted-model")
        self.plan, self.calls, self.lock = plan or {}, {}, threading.Lock()

    def chat(self, request):
        key = request.metadata["input_id"]
        with self.lock:
            n = self.calls.get(key, 0)
            self.calls[key] = n + 1
        steps = self.plan.get(request.metadata.get("plan_key", key), ["ok"])
        step = steps[min(n, len(steps) - 1)]
        if step == "error":
            raise RuntimeError("provider down")
        if step == "interrupt":
            raise KeyboardInterrupt
        text = reading_from_prompt(request.prompt, drop="Outcome" if step == "invalid" else None)
        return ChatResponse(request.id, text, model="scripted-model", finish_reason="stop",
                            usage=Usage(10, 100, 110), metadata=request.metadata)


@pytest.fixture
def isolated_registry(monkeypatch):
    monkeypatch.setattr(providers, "_CACHE", {})
    monkeypatch.setattr(providers, "_FACTORIES", dict(providers._FACTORIES))


@pytest.fixture
def scripted(isolated_registry):
    p = TarotScripted()
    register_provider("tarot-scripted", lambda **kw: p)
    return p


def config(**kw):
    kw.setdefault("n", 4)
    kw.setdefault("provider", "tarot-scripted")
    kw.setdefault("seed", 123)
    return GenerationConfig(**kw)


def check_foreign_keys(store: RunStore):
    """Every REFERENCES column value exists in the referenced table (what Postgres would enforce)."""
    for t in schema.TABLES.values():
        rows = store.read(t.name)
        for c in t.columns:
            if not c.references:
                continue
            ref_table, ref_col = re.match(r"(\w+)\((\w+)\)", c.references).groups()
            valid = {r[ref_col] for r in store.read(ref_table)}
            missing = {r[c.name] for r in rows if r[c.name] is not None} - valid
            assert not missing, f"{t.name}.{c.name} -> {c.references}: {missing}"


def check_primary_keys(store: RunStore):
    for t in schema.TABLES.values():
        keys = [tuple(r[k] for k in t.primary_key) for r in store.read(t.name)]
        assert len(keys) == len(set(keys)), f"duplicate primary keys in {t.name}"


# ---- schema -----------------------------------------------------------------------------
def test_committed_schema_sql_is_up_to_date():
    committed = (Path(schema.__file__).parent / "schema.sql").read_text(encoding="utf-8")
    assert committed.replace("\r\n", "\n").strip() == schema.ddl().strip(), \
        "regenerate: python -m data_generation.schema > data_generation/schema.sql"


def test_ddl_tables_in_dependency_order():
    order = list(schema.TABLES)
    for t in schema.TABLES.values():
        for c in t.columns:
            if c.references:
                ref = c.references.split("(")[0]
                assert order.index(ref) < order.index(t.name) or ref == t.name, f"{t.name} before {ref}"
    assert "CREATE OR REPLACE VIEW final_responses" in schema.ddl()


def test_check_row():
    row = schema.check_row("response_texts", {"response_id": "r", "run_id": "x"})
    assert row == {"response_id": "r", "run_id": "x", "text": None, "n_chars": None}
    with pytest.raises(ValueError, match="unknown columns"):
        schema.check_row("response_texts", {"response_id": "r", "run_id": "x", "bogus": 1})
    with pytest.raises(ValueError, match="required"):
        schema.check_row("response_texts", {"run_id": "x"})


# ---- store ------------------------------------------------------------------------------
def test_store_append_upsert_read(tmp_path):
    s = RunStore.create(tmp_path, run_id="run_test")
    s.append("response_texts", [{"response_id": "a", "run_id": "run_test", "text": "é"}])
    s.append("response_texts", [{"response_id": "b", "run_id": "run_test"}])
    assert [r["response_id"] for r in s.read("response_texts")] == ["a", "b"]
    row = {"batch_id": "b1", "run_id": "run_test", "provider": "p", "round": 0, "status": "pending",
           "n_requests": 2, "collected": False, "created_at": "t", "updated_at": "t"}
    s.upsert("batches", [row])
    s.upsert("batches", [{**row, "status": "completed"}])
    assert [b["status"] for b in s.read("batches")] == ["completed"]
    with pytest.raises(ValueError, match="mutable"):
        s.append("batches", [row])
    with pytest.raises(ValueError, match="append-only"):
        s.upsert("responses", [])
    with pytest.raises(FileNotFoundError):
        RunStore.open("nope", tmp_path)
    with pytest.raises(FileExistsError):
        RunStore.create(tmp_path, run_id="run_test")


# ---- tarot application ------------------------------------------------------------------
def test_tarot_app_reads_deck_config():
    app = TarotReadings()
    assert app.name == "tarot_celtic_cross" and app.default_template == "celtic_cross_v1"
    assert len(app.cards) == 78 and len(app.states) == 156 and len(app.positions) == 10


def test_draws_are_seeded_and_pinned():
    app = TarotReadings(pins={"Outcome": "Death reversed"})
    a = app.sample_inputs(5, random.Random(1))
    assert a == app.sample_inputs(5, random.Random(1))
    assert a != app.sample_inputs(5, random.Random(2))
    assert all(d["Outcome"] == "Death reversed" and list(d) == app.positions for d in a)


def test_unique_vs_replacement_draws():
    cards = lambda d: [s.removesuffix(" reversed") for s in d.values()]
    unique = TarotReadings(draw="unique", pins={"Present": "The Fool"}).sample_inputs(300, random.Random(0))
    assert all(len(set(cards(d))) == 10 for d in unique)
    assert all(d["Present"] == "The Fool" for d in unique)
    repl = TarotReadings(draw="replacement").sample_inputs(300, random.Random(0))
    assert any(len(set(cards(d))) < 10 for d in repl)  # repeats happen (~45% of draws)


@pytest.mark.parametrize("kwargs, match", [
    ({"pins": {"Nowhere": "Death"}}, "position"),
    ({"pins": {"Outcome": "Deth"}}, "valid state"),
    ({"draw": "unique", "pins": {"Outcome": "Death", "Present": "Death reversed"}}, "pinned twice"),
    ({"draw": "sideways"}, "draw must be"),
])
def test_bad_tarot_options(kwargs, match):
    with pytest.raises(ValueError, match=match):
        TarotReadings(**kwargs)


def test_input_items_and_template():
    app = TarotReadings()
    draw = app.sample_inputs(1, random.Random(3))[0]
    items = app.input_items(draw)
    assert [i["label"] for i in items] == app.positions
    assert all(i["value"] == (i["entity"] + (" reversed" if i["qualifier"] == "reversed" else "")) for i in items)
    variables = app.template_variables(draw, None)
    assert variables == {"cards": draw, "positions": app.positions, "persona": None}
    prompt = get_template(app.default_template).render_from(variables)
    assert prompt.system is None and prompt.user.startswith("You're a tarot reader.") and "Outcome: " in prompt.user


def test_tarot_validator_accepts_a_reading_of_the_draw():
    app = TarotReadings()
    draw = app.sample_inputs(1, random.Random(5))[0]
    text = reading_from_prompt(get_template("celtic_cross_v1").render(cards=draw).user)
    assert app.validator().validate(text, expected=draw, finish_reason="stop").passed


# ---- pipeline: live ---------------------------------------------------------------------
def test_dry_run_writes_preparation_tables_only(tmp_path, scripted):
    store, requests = prepare_run(TarotReadings(), config(mode="dry_run"), root=tmp_path)
    assert len(store.read("inputs")) == 4 and len(store.read("input_items")) == 40
    assert len(store.read("prompts")) == len(store.read("requests")) == 4
    assert store.read("responses") == [] and store.run()["status"] == "prepared"
    assert scripted.calls == {}
    check_foreign_keys(store)


def test_rebuild_requests_matches_prepared(tmp_path, scripted):
    store, requests = prepare_run(TarotReadings(), config(params={"max_tokens": 50},
                                                          extra_body={"top_k": 40}), root=tmp_path)
    rebuilt = rebuild_requests(store)
    assert [r.to_dict() for r in rebuilt] == [r.to_dict() for r in requests]


def test_live_run_writes_all_tables(tmp_path, scripted):
    app = TarotReadings()
    store, requests = prepare_run(app, config(), root=tmp_path)
    first_input = requests[0].metadata["input_id"]
    scripted.plan = {first_input: ["invalid", "invalid", "ok"]}   # needs both default retries
    summary = run_live(store, app, requests, progress=False)

    assert summary["n_requests"] == summary["n_finished"] == summary["n_valid"] == 4
    assert summary["n_regenerated"] == 1 and summary["n_attempts"] == 6
    assert summary["total_tokens"] == 6 * 110          # every attempt counts toward cost
    run = store.run()
    assert run["status"] == "completed" and run["summary"] == summary and run["schema_version"] == schema.SCHEMA_VERSION
    assert run["config"]["application_options"]["draw"] == "replacement"

    retried = [r for r in store.read("responses") if r["request_id"] == requests[0].id]
    assert [r["attempt"] for r in retried] == [1, 2, 3]
    verdicts = {v["response_id"]: v["passed"] for v in store.read("validations")}
    assert [verdicts[r["response_id"]] for r in retried] == [False, False, True]
    issues = store.read("validation_issues")
    assert {(i["code"], i["label"]) for i in issues} >= {("missing_label", "Outcome")}
    assert all(t["text"] for t in store.read("response_texts"))
    assert len(store.read("response_texts")) == len(store.read("validations")) == 6
    check_foreign_keys(store)
    check_primary_keys(store)


def test_resume_after_api_errors_continues_attempt_numbers(tmp_path, scripted):
    app = TarotReadings()
    store, requests = prepare_run(app, config(n=3), root=tmp_path)
    bad = requests[1].metadata["input_id"]
    scripted.plan = {bad: ["error", "ok"]}
    first = run_live(store, app, requests, progress=False)
    assert (first["n_finished"], first["n_api_errors"]) == (2, 1) and store.run()["status"] == "partial"

    second = run_live(store, app, progress=False)               # resume from the tables alone
    assert (second["n_finished"], second["n_api_errors"], second["n_valid"]) == (3, 0, 3)
    assert store.run()["status"] == "completed"
    rows = [r for r in store.read("responses") if r["request_id"] == requests[1].id]
    assert [(r["attempt"], r["error"] is None) for r in rows] == [(1, False), (2, True)]
    assert final_responses(store)[requests[1].id]["attempt"] == 2
    assert scripted.calls[requests[0].metadata["input_id"]] == 1   # finished requests not redone
    check_primary_keys(store)


def test_interrupted_run_is_marked_and_resumable(tmp_path, scripted):
    app = TarotReadings()
    store, requests = prepare_run(app, config(n=3, max_workers=1), root=tmp_path)
    scripted.plan = {requests[2].metadata["input_id"]: ["interrupt", "ok"]}
    with pytest.raises(KeyboardInterrupt):
        run_live(store, app, requests, progress=False)
    assert store.run()["status"] == "interrupted"
    assert run_live(store, app, progress=False)["n_finished"] == 3


def test_no_validate(tmp_path, scripted):
    app = TarotReadings()
    store, requests = prepare_run(app, config(n=2, validate=False), root=tmp_path)
    scripted.plan = {requests[0].metadata["input_id"]: ["invalid"]}
    summary = run_live(store, app, requests, progress=False)
    assert summary["n_unvalidated"] == 2 and summary["n_attempts"] == 2 and store.read("validations") == []


# ---- pipeline: batch --------------------------------------------------------------------
def tarot_batch_outputs(invalid_first_time=()):
    seen = {}

    def outputs(uploaded):
        out = []
        for line in uploaded:
            cid = line["custom_id"]
            seen[cid] = seen.get(cid, 0) + 1
            drop = "Outcome" if cid in invalid_first_time and seen[cid] == 1 else None
            text = reading_from_prompt(line["body"]["messages"][-1]["content"], drop=drop)
            out.append({"custom_id": cid, "response": {"status_code": 200, "body": batch_body(text)}, "error": None})
        return out, []
    return outputs


@pytest.fixture
def batch_provider(isolated_registry, tmp_path):
    holder = {}

    def make(statuses=("validating", "completed"), invalid_first_time=()):
        client = FakeOpenAIClient(statuses=statuses, batch_outputs=tarot_batch_outputs(invalid_first_time))
        p = OpenAICompatibleProvider(name="fake-batch", client=client, default_model="fake-model",
                                     supports_batch=True, batch_dir=tmp_path / "unused")
        register_provider("fake-batch", lambda **kw: p)
        holder["p"] = p
        return p
    return make


def test_batch_submit_then_collect(tmp_path, batch_provider):
    p = batch_provider(statuses=("validating", "in_progress", "completed"))
    app = TarotReadings()
    store, requests = prepare_run(app, config(n=3, provider="fake-batch", mode="batch"), root=tmp_path)
    job = submit_batch(store, requests)
    assert store.run()["status"] == "submitted"
    [b] = store.read("batches")
    assert (b["batch_id"], b["round"], b["status"], b["n_requests"], b["collected"]) == (job.id, 0, "pending", 3, False)
    assert Path(b["input_file"]).parent == store.batch_files_dir     # batch files live in the run dir

    not_yet = collect_batch(store, app, wait=False, progress=False)
    assert not_yet["status"] == "running" and store.read("responses") == []

    summary = collect_batch(store, app, wait=False, progress=False)
    assert summary["n_finished"] == summary["n_valid"] == 3
    assert {r["batch_id"] for r in store.read("responses")} == {job.id}
    assert all(b["collected"] for b in store.read("batches")) and store.run()["status"] == "completed"
    with pytest.raises(RuntimeError, match="no uncollected batch"):
        collect_batch(store, app)
    check_foreign_keys(store)


def test_batch_retry_rounds_recorded(tmp_path, batch_provider):
    app = TarotReadings()
    store, requests = prepare_run(app, config(n=2, provider="fake-batch", mode="batch", retry_mode="batch",
                                              poll_interval=0), root=tmp_path)
    batch_provider(invalid_first_time={requests[0].id})
    submit_batch(store, requests)
    summary = collect_batch(store, app, wait=True, progress=False)
    assert summary["n_valid"] == 2 and summary["n_regenerated"] == 1
    assert sorted(b["round"] for b in store.read("batches")) == [0, 1]
    rows = [r for r in store.read("responses") if r["request_id"] == requests[0].id]
    assert [r["attempt"] for r in rows] == [1, 2] and all(r["batch_id"] for r in rows)
    check_foreign_keys(store)


def test_batch_collect_retries_live_by_default(tmp_path, batch_provider):
    app = TarotReadings()
    store, requests = prepare_run(app, config(n=2, provider="fake-batch", mode="batch"), root=tmp_path)
    batch_provider(invalid_first_time={requests[1].id})
    submit_batch(store, requests)
    collect_batch(store, app, wait=True, progress=False)
    rows = [r for r in store.read("responses") if r["request_id"] == requests[1].id]
    assert [r["batch_id"] is None for r in rows] == [False, True]   # retry was a live call
    assert [b["round"] for b in store.read("batches")] == [0]


# ---- CLI --------------------------------------------------------------------------------
def test_cli_generate_status_list_resume(tmp_path, scripted, capsys):
    out = str(tmp_path)
    assert main(["--out", out, "generate", "--n", "2", "--provider", "tarot-scripted", "--seed", "5",
                 "--temperature", "0.7", "--max-tokens", "1234", "--extra-body", '{"top_k": 40}', "--pin", "Outcome=Death",
                 "--name", "cli-test"]) == 0
    [run_dir] = list(tmp_path.glob("run_*"))
    store = RunStore.open(run_dir.name, tmp_path)
    run = store.run()
    assert run["status"] == "completed" and run["name"] == "cli-test"
    assert run["config"]["params"] == {"max_tokens": 1234, "temperature": 0.7}
    assert run["config"]["extra_body"] == {"top_k": 40}
    assert all(i["payload"]["Outcome"] == "Death" for i in store.read("inputs"))
    assert main(["--out", out, "status", run_dir.name]) == 0
    assert main(["--out", out, "list"]) == 0
    assert "cli-test" in capsys.readouterr().out
    assert main(["--out", out, "resume", run_dir.name]) == 0            # nothing left to do


def test_dataset_groups_runs_across_sessions(tmp_path, scripted, capsys):
    out = str(tmp_path)
    gen = ["--out", out, "generate", "--provider", "tarot-scripted"]
    assert main([*gen, "--n", "3", "--dataset", "tarot-v1"]) == 0          # session 1
    assert main([*gen, "--n", "2", "--dataset", "tarot-v1"]) == 0          # session 2, later
    assert main([*gen, "--n", "1", "--dataset", "other"]) == 0
    assert main([*gen, "--n", "1"]) == 0                                    # no dataset
    capsys.readouterr()

    assert [r["dataset"] for r in list_runs(tmp_path)] == ["tarot-v1", "tarot-v1", "other", None]
    v1 = list_runs(tmp_path, "tarot-v1")
    assert [r["summary"]["n_finished"] for r in v1] == [3, 2]
    draws = [json.dumps(i["payload"], sort_keys=True)
             for r in v1 for i in RunStore.open(r["run_id"], tmp_path).read("inputs")]
    assert len(set(draws)) == 5                    # fresh seed per run -> no repeated draws

    assert main(["--out", out, "list", "--dataset", "tarot-v1"]) == 0
    listing = capsys.readouterr().out
    assert listing.count("dataset=tarot-v1") == 2 and "dataset=other" not in listing
    assert "dataset 'tarot-v1': 2 run(s), 5/5 finished, 5 valid" in listing
    assert main(["--out", out, "list", "--dataset", "missing"]) == 0
    assert "No runs" in capsys.readouterr().out


def test_reusing_a_seed_within_a_dataset_warns(tmp_path, scripted, capsys):
    gen = ["--out", str(tmp_path), "generate", "--provider", "tarot-scripted", "--n", "1", "--dry-run"]
    main([*gen, "--dataset", "d", "--seed", "7"])
    assert "WARNING" not in capsys.readouterr().err
    main([*gen, "--dataset", "d", "--seed", "8"])                           # different seed: fine
    main([*gen, "--dataset", "elsewhere", "--seed", "7"])                   # other dataset: fine
    main([*gen, "--dataset", "d", "--seed", "7", "--draw", "unique"])       # different draw settings: fine
    main([*gen, "--seed", "7"])                                             # no dataset: fine
    assert "WARNING" not in capsys.readouterr().err
    main([*gen, "--dataset", "d", "--seed", "7"])
    assert "already has run(s) with --seed 7" in capsys.readouterr().err


def test_cli_dry_run_and_argument_errors(tmp_path, scripted):
    assert main(["--out", str(tmp_path), "generate", "--n", "3", "--dry-run", "--provider", "tarot-scripted"]) == 0
    assert scripted.calls == {}
    with pytest.raises(SystemExit):
        main(["--out", str(tmp_path), "generate", "--n", "1", "--pin", "no-equals-sign"])
    with pytest.raises(SystemExit):
        main(["--out", str(tmp_path), "generate", "--n", "1", "--extra-body", "[1, 2]"])


def test_cli_batch_collect(tmp_path, batch_provider):
    batch_provider(statuses=("validating", "in_progress", "completed"))
    out = str(tmp_path)
    assert main(["--out", out, "generate", "--n", "2", "--provider", "fake-batch", "--mode", "batch"]) == 0
    [run_dir] = list(tmp_path.glob("run_*"))
    assert main(["--out", out, "collect", run_dir.name]) == 1           # still running
    assert main(["--out", out, "collect", run_dir.name]) == 0
    assert RunStore.open(run_dir.name, tmp_path).run()["status"] == "completed"
    assert main(["--out", out, "resume", run_dir.name]) == 2            # resume is live-only


# ---- prompts library + personas in the pipeline --------------------------------------------
def test_default_templates_are_pinned(tmp_path, scripted):
    app = TarotReadings()
    plain, _ = prepare_run(app, config(n=1, mode="dry_run"), root=tmp_path)
    with_people, _ = prepare_run(app, config(n=1, mode="dry_run", persona_set="tarot_personas"), root=tmp_path)
    assert plain.run()["config"]["templates"] == ["celtic_cross_v1"]          # existing datasets' prompt
    assert with_people.run()["config"]["templates"] == ["celtic_cross_v2"]    # the persona-aware version
    assert plain.read("personas") == [] and plain.read("inputs")[0]["persona_id"] is None


def test_template_reference_resolves_to_a_pinned_version(tmp_path, scripted):
    store, _ = prepare_run(TarotReadings(), config(n=1, mode="dry_run", templates=["celtic_cross"]), root=tmp_path)
    [t] = store.read("prompt_templates")
    assert store.run()["config"]["templates"] == [t["template_id"]] and parse_template_id(t["template_id"])
    assert t["template_sha256"] == get_template(t["template_id"]).sha256
    assert t["user_template"] == get_template(t["template_id"]).user
    with pytest.raises(KeyError, match="No prompt template"):
        prepare_run(TarotReadings(), config(n=1, templates=["nope"]), root=tmp_path)
    assert len(list(tmp_path.glob("run_*"))) == 1            # the failed run left nothing behind


def test_personas_and_two_templates_end_to_end(tmp_path, scripted):
    app = TarotReadings()
    cfg = config(n=4, persona_set="tarot_personas", persona_sampling="cycle",
                 templates=["celtic_cross_v1", "celtic_cross_v2"])
    with pytest.warns(UserWarning, match="celtic_cross_v1.*take no"):
        store, requests = prepare_run(app, cfg, root=tmp_path)
    assert len(requests) == 8 and len(store.read("inputs")) == 4 and len(store.read("prompts")) == 8

    inputs = store.read("inputs")
    personas = {p["persona_id"]: p for p in store.read("personas")}
    assert [personas[i["persona_id"]]["persona_number"] for i in inputs] == [1, 2, 3, 4]   # cycle order
    assert all(p["persona_set"] == "tarot_personas" and p["content_sha256"] for p in personas.values())

    by_input = {}
    for p in store.read("prompts"):
        by_input.setdefault(p["input_id"], {})[p["template"]] = p
    for inp in inputs:
        v1, v2 = by_input[inp["input_id"]]["celtic_cross_v1"], by_input[inp["input_id"]]["celtic_cross_v2"]
        first_name = personas[inp["persona_id"]]["payload"]["name"]["first"]
        assert f"- Name: {first_name}" in v2["user_text"] and "about me" not in v1["user_text"]
        cards = "\n".join(f"{k}: {v}" for k, v in inp["payload"].items())
        assert v1["user_text"].endswith(cards) and v2["user_text"].endswith(cards)     # same draw in both
        assert v1["template_sha256"] != v2["template_sha256"]

    assert [r.to_dict() for r in rebuild_requests(store)] == [r.to_dict() for r in requests]
    assert requests[1].metadata["persona_id"] == inputs[0]["persona_id"]
    assert requests[1].metadata["template"] == "celtic_cross_v2"

    summary = run_live(store, app, requests, progress=False)
    assert summary["n_finished"] == summary["n_valid"] == 8
    assert summary["by_template"] == {
        t: {"n_requests": 4, "n_finished": 4, "n_valid": 4, "n_invalid": 0, "n_valid_first_try": 4}
        for t in ("celtic_cross_v1", "celtic_cross_v2")}
    cfg_row = store.run()["config"]
    assert cfg_row["persona_set_sha256"] and cfg_row["persona_sampling"] == "cycle"
    check_foreign_keys(store)
    check_primary_keys(store)


def test_seed_gives_same_draws_with_or_without_personas(tmp_path, scripted):
    app = TarotReadings()
    a, _ = prepare_run(app, config(n=5, seed=9, mode="dry_run"), root=tmp_path)
    b, _ = prepare_run(app, config(n=5, seed=9, mode="dry_run", persona_set="tarot_personas"), root=tmp_path)
    assert [i["payload"] for i in a.read("inputs")] == [i["payload"] for i in b.read("inputs")]


def test_persona_restriction_and_errors(tmp_path, scripted):
    app = TarotReadings()
    store, _ = prepare_run(app, config(n=6, mode="dry_run", persona_set="tarot_personas",
                                       persona_ids=["3", "Maya Delgado"]), root=tmp_path)
    assert sorted(p["persona_number"] for p in store.read("personas")) == [1, 3]
    with pytest.raises(ValueError, match="without a persona_set"):
        prepare_run(app, config(n=1, persona_ids=["1"]), root=tmp_path)
    with pytest.raises(KeyError, match="No persona"):
        prepare_run(app, config(n=1, persona_set="tarot_personas", persona_ids=["nobody"]), root=tmp_path)
    with pytest.raises(FileNotFoundError):
        prepare_run(app, config(n=1, persona_set="no_such_set"), root=tmp_path)


def test_first_try_validity_per_template(tmp_path, scripted):
    app = TarotReadings()
    store, requests = prepare_run(app, config(n=3), root=tmp_path)
    scripted.plan = {requests[0].metadata["input_id"]: ["invalid", "ok"]}
    stats = run_live(store, app, requests, progress=False)["by_template"]["celtic_cross_v1"]
    assert stats == {"n_requests": 3, "n_finished": 3, "n_valid": 3, "n_invalid": 0, "n_valid_first_try": 2}


def test_runs_written_by_schema_version_1_still_work(tmp_path, scripted):
    """Old runs have config['template'] (a string) and no persona/template-hash columns."""
    app = TarotReadings()
    store, requests = prepare_run(app, config(n=2), root=tmp_path)
    run = store.run()
    old_config = {k: v for k, v in run["config"].items()
                  if k not in ("templates", "persona_set", "persona_sampling", "persona_ids", "persona_set_sha256")}
    old_config["template"] = "celtic_cross_v1"
    store.upsert("runs", [{**run, "config": old_config, "schema_version": 1}])
    for table, drop in (("inputs", "persona_id"), ("prompts", "template_sha256")):
        rows = [{k: v for k, v in r.items() if k != drop} for r in store.read(table)]
        store.path(table).write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    store.path("prompt_templates").unlink()

    assert load_config(store).templates == ["celtic_cross_v1"]
    rebuilt = rebuild_requests(store)
    assert [r.messages for r in rebuilt] == [r.messages for r in requests]
    assert all(r.metadata["persona_id"] is None for r in rebuilt)
    summary = run_live(store, app, progress=False)                     # resume an old run
    assert summary["n_valid"] == 2 and summary["by_template"]["celtic_cross_v1"]["n_finished"] == 2


def test_cli_personas_and_templates(tmp_path, scripted, capsys):
    out = str(tmp_path)
    assert main(["--out", out, "generate", "--n", "2", "--provider", "tarot-scripted", "--personas", "tarot_personas",
                 "--persona", "7", "--template", "celtic_cross_v1", "--template", "celtic_cross_v2"]) == 0
    captured = capsys.readouterr()
    assert "templates: celtic_cross_v1, celtic_cross_v2  personas: tarot_personas" in captured.out
    assert "celtic_cross_v2: 2/2 finished; 100% valid, 100% on the first try" in captured.out
    assert "note: Personas were requested but ['celtic_cross_v1']" in captured.err
    [run_dir] = list(tmp_path.glob("run_*"))
    store = RunStore.open(run_dir.name, tmp_path)
    assert [p["persona_number"] for p in store.read("personas")] == [7] and len(store.read("requests")) == 4
