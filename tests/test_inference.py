"""inference: schema, store, the tarot task, live/batch pipeline, resume, CLI."""
import json
import re
import threading
import warnings
from dataclasses import asdict
from pathlib import Path

import pytest
import yaml

import applications
import llm.providers as providers
from llm import ChatResponse, Usage
from llm.providers import LLMProvider, register_provider
from applications.tarot.code.readings import TarotReadings
from inference import schema
from inference.generate import main, run_task
from inference.pipeline import (
    GenerationConfig, collect_batch, final_responses, list_runs, load_config, prepare_run, rebuild_requests, run_live,
    submit_batch,
)
from inference.store import RunStore
from inference.task import INSTANT, InferenceTask
from inference.task_config import InferenceTaskConfig, TaskConfigError
from instantiation import Instantiation, InstantiationError
from spec_files import file_sha256
from prompts import get_template
from conftest import FakeOpenAIClient, batch_body
from llm.providers.openai_compatible import OpenAICompatibleProvider

ROOT = Path(__file__).parents[1]
TAROT_FILE = ROOT / "applications" / "tarot" / "tasks.yaml"
DECK = TAROT_FILE.parent / "vector_space_config.json"
POSITIONS = json.loads(DECK.read_text(encoding="utf-8"))["step_aliases"]
RUN = ["run", "tarot_celtic_cross", "--provider", "tarot-scripted"]   # a run of the tarot task, by the fake reader
FILLER = "This card carries a meaningful theme for you, worth careful reflection in the days ahead."
PERSONA = {"instantiation.state.persona": {"personas": "tarot_personas"}}


def reading_from_prompt(prompt: str, drop: str = None) -> str:
    """A well-formed reading restating the draw, i.e. the 'Position: Card' lines that end the prompt (drop: omit one)."""
    lines = re.findall(r"^([^:\n]+): (.+)$", prompt.strip().split("\n\n")[-1], flags=re.M)
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


def tarot(edits=None, provider="tarot-scripted", **execution) -> InferenceTask:
    """
    The tarot task as tasks.yaml says, answered by a fake provider: `edits` change it ({dotted.path: value}),
    keywords change its execution settings (n=4 and seed=123 unless given).
    """
    changes = {"model.provider": provider, "model.name": None, "execution.n": 4, "execution.seed": 123}
    changes.update({f"execution.{k}": v for k, v in execution.items()})
    changes.update(edits or {})
    return InferenceTask.from_config(InferenceTaskConfig.from_file(TAROT_FILE, "tarot_celtic_cross", changes))


def prepare(root, edits=None, validate=True, **execution):
    """A prepared run of the tarot task (see tarot): (task, store, requests)."""
    task = tarot({**(edits or {}), **({} if validate else {"validation": None})}, **execution)
    store, requests = prepare_run(task, root=root)
    return task, store, requests


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
        "regenerate: python -m inference.schema > inference/schema.sql"


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
def test_run_ids_name_the_task_then_the_time(tmp_path, scripted):
    from inference.store import new_run_id
    assert re.fullmatch(r"run_tarot_celtic_cross_\d{8}_\d{6}_[0-9a-f]{6}", new_run_id("tarot_celtic_cross"))
    assert re.fullmatch(r"run_my_task_2_\d{8}_\d{6}_[0-9a-f]{6}", new_run_id("My Task/2"))     # made folder-safe
    assert re.fullmatch(r"run_\d{8}_\d{6}_[0-9a-f]{6}", new_run_id())                          # the old form
    _, store, _ = prepare(tmp_path, n=1, mode="dry_run")
    assert store.run_id.startswith("run_tarot_celtic_cross_2") and store.run()["run_id"] == store.run_id
    assert store.dir.name == store.run_id and all(r["run_id"] == store.run_id for r in store.read("inputs"))
    # runs created under the old naming are still found, and listing is by start time, not by name
    old = RunStore.create(tmp_path, run_id="run_20200101_000000_abcdef")
    old.upsert("runs", [{**store.run(), "run_id": old.run_id, "created_at": "2020-01-01T00:00:00+00:00"}])
    assert [r["run_id"] for r in list_runs(tmp_path)] == [old.run_id, store.run_id]
    assert RunStore.open(old.run_id, tmp_path).run()["run_id"] == old.run_id


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


# ---- the tarot task -------------------------------------------------------------------------
def test_the_tarot_task_is_its_task_file_and_a_little_python(scripted):
    task = tarot()
    assert type(task) is TarotReadings and task.name == "tarot_celtic_cross" and task.entry == "cards"
    assert task.config.implemented_by == "applications.tarot.code.readings:TarotReadings"
    assert task.input_type == INSTANT and not task.computes_inputs and not task.before_task_file


def test_draws_are_seeded_and_pinned(scripted):
    pinned = tarot({"instantiation.state.cards.steps.fixed": {"Outcome": "Death reversed"}}).config.instantiation
    draws = lambda seed: [i.instant["cards"] for i in Instantiation.sample(pinned, 5, seed=seed)]
    assert draws(1) == draws(1) != draws(2)
    assert all(d["Outcome"] == "Death reversed" and list(d) == POSITIONS for d in draws(1))


def test_unique_vs_replacement_draws(scripted):
    cards = lambda d: [s.removesuffix(" reversed") for s in d.values()]
    unique = tarot({"instantiation.state.cards.steps.replacement": False,
                    "instantiation.state.cards.steps.fixed": {"Present": "The Fool"}}).config.instantiation
    unique = [i.instant["cards"] for i in Instantiation.sample(unique, 300, seed=0)]
    assert all(len(set(cards(d))) == 10 for d in unique) and all(d["Present"] == "The Fool" for d in unique)
    repeats = [i.instant["cards"] for i in Instantiation.sample(tarot().config.instantiation, 300, seed=0)]
    assert any(len(set(cards(d))) < 10 for d in repeats)  # repeats happen (~45% of draws)


def test_input_items_and_prompt(tmp_path, scripted):
    _, store, _ = prepare(tmp_path, n=1, mode="dry_run")
    [inp], [prompt] = store.read("inputs"), store.read("prompts")
    draw = inp["payload"]["cards"]
    assert inp["input_type"] == INSTANT and list(inp["payload"]) == ["cards"]
    items = store.read("input_items")
    assert [i["label"] for i in items] == POSITIONS == list(draw)
    assert all(i["value"] == (i["entity"] + (" reversed" if i["qualifier"] == "reversed" else "")) for i in items)
    assert prompt["template"] == "celtic_cross_v4" and prompt["system"] is None
    assert prompt["user_text"].startswith("You're an experienced tarot reader.")
    assert prompt["user_text"].rstrip().endswith("\n".join(f"{position}: {card}" for position, card in draw.items()))


def test_the_task_file_s_validator_is_the_one_runs_had(scripted):
    """The generic checks the file names take the place of the custom validator's own: the same checks, in order."""
    before, now = TarotReadings.from_options({}).validator(), tarot().validator()
    assert now.name == before.name == "tarot_celtic_cross"
    assert [c.name for c in now.checks] == [c.name for c in before.checks] == \
        ["non_empty", "not_truncated", "no_refusal", "positions", "foreign_cards"]
    assert now.checks[0].min_chars == 200
    tighter = tarot({"validation.checks": {"non_empty": {"min_chars": 5000}}}).validator()
    assert [c.name for c in tighter.checks] == [c.name for c in before.checks] and tighter.checks[0].min_chars == 5000
    only_generic = tarot({"validation.custom_validator": None, "validation.checks": {"not_truncated": {}}}).validator()
    assert [c.name for c in only_generic.checks] == ["not_truncated"]
    assert tarot({"validation.custom_validator": None, "validation.checks": {}}).validator() is None


def test_the_tarot_validator_accepts_a_reading_of_the_draw(tmp_path, scripted):
    task, store, _ = prepare(tmp_path, n=1, mode="dry_run")
    [prompt], [inp] = store.read("prompts"), store.read("inputs")
    expected = task.validation_expected({"input": inp["payload"]})
    assert expected == inp["payload"]["cards"]
    assert task.validator().validate(reading_from_prompt(prompt["user_text"]), expected=expected,
                                     finish_reason="stop").passed


# ---- pipeline: live ---------------------------------------------------------------------
def test_dry_run_writes_preparation_tables_only(tmp_path, scripted):
    _, store, _ = prepare(tmp_path, mode="dry_run")
    assert len(store.read("inputs")) == 4 and len(store.read("input_items")) == 40
    assert len(store.read("prompts")) == len(store.read("requests")) == 4
    assert store.read("responses") == [] and store.run()["status"] == "prepared"
    assert scripted.calls == {}
    check_foreign_keys(store)


def test_rebuild_requests_matches_prepared(tmp_path, scripted):
    _, store, requests = prepare(tmp_path, {"model.parameters": {"max_tokens": 50}, "model.extra_body": {"top_k": 40}})
    rebuilt = rebuild_requests(store)
    assert [r.to_dict() for r in rebuilt] == [r.to_dict() for r in requests]
    assert requests[0].params == {"max_tokens": 50} and requests[0].extra_body == {"top_k": 40}


def test_live_run_writes_all_tables(tmp_path, scripted):
    task, store, requests = prepare(tmp_path)
    first_input = requests[0].metadata["input_id"]
    scripted.plan = {first_input: ["invalid", "invalid", "ok"]}   # needs both retries of the task file
    summary = run_live(store, task, requests, progress=False)

    assert summary["n_requests"] == summary["n_finished"] == summary["n_valid"] == 4
    assert summary["n_regenerated"] == 1 and summary["n_attempts"] == 6
    assert summary["total_tokens"] == 6 * 110          # every attempt counts toward cost
    run = store.run()
    assert run["status"] == "completed" and run["summary"] == summary and run["schema_version"] == schema.SCHEMA_VERSION
    assert set(run["config"]) == {"task_config", "seed"} and run["config"]["seed"] == 123
    recorded = run["config"]["task_config"]
    assert recorded["name"] == "tarot_celtic_cross" and recorded["source"] == "applications/tarot/tasks.yaml"
    assert recorded["sha256"] == task.config.sha256 and recorded["edits"]["execution.n"] == 4
    assert recorded["content"]["instantiation"]["state"]["cards"]["steps"]["replacement"] is True

    retried = [r for r in store.read("responses") if r["request_id"] == requests[0].id]
    assert [r["attempt"] for r in retried] == [1, 2, 3]
    verdicts = {v["response_id"]: v["passed"] for v in store.read("validations")}
    assert [verdicts[r["response_id"]] for r in retried] == [False, False, True]
    assert {v["validator"] for v in store.read("validations")} == {"tarot_celtic_cross"}
    issues = store.read("validation_issues")
    assert {(i["code"], i["label"]) for i in issues} >= {("missing_label", "Outcome")}
    assert all(t["text"] for t in store.read("response_texts"))
    assert len(store.read("response_texts")) == len(store.read("validations")) == 6
    check_foreign_keys(store)
    check_primary_keys(store)


def test_resume_after_api_errors_continues_attempt_numbers(tmp_path, scripted):
    task, store, requests = prepare(tmp_path, n=3)
    bad = requests[1].metadata["input_id"]
    scripted.plan = {bad: ["error", "ok"]}
    first = run_live(store, task, requests, progress=False)
    assert (first["n_finished"], first["n_api_errors"]) == (2, 1) and store.run()["status"] == "partial"

    second = run_live(store, applications.task_of(store.run()), progress=False)   # resume from the tables alone
    assert (second["n_finished"], second["n_api_errors"], second["n_valid"]) == (3, 0, 3)
    assert store.run()["status"] == "completed"
    rows = [r for r in store.read("responses") if r["request_id"] == requests[1].id]
    assert [(r["attempt"], r["error"] is None) for r in rows] == [(1, False), (2, True)]
    assert final_responses(store)[requests[1].id]["attempt"] == 2
    assert scripted.calls[requests[0].metadata["input_id"]] == 1   # finished requests not redone
    check_primary_keys(store)


def test_interrupted_run_is_marked_and_resumable(tmp_path, scripted):
    task, store, requests = prepare(tmp_path, n=3, workers=1)
    scripted.plan = {requests[2].metadata["input_id"]: ["interrupt", "ok"]}
    with pytest.raises(KeyboardInterrupt):
        run_live(store, task, requests, progress=False)
    assert store.run()["status"] == "interrupted"
    assert run_live(store, task, progress=False)["n_finished"] == 3


def test_no_validate(tmp_path, scripted):
    task, store, requests = prepare(tmp_path, n=2, validate=False)
    scripted.plan = {requests[0].metadata["input_id"]: ["invalid"]}
    summary = run_live(store, task, requests, progress=False)
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


def prepare_batch(root, **execution):
    task = tarot(provider="fake-batch", mode="batch", **execution)
    store, requests = prepare_run(task, root=root)
    return task, store, requests


def test_batch_submit_then_collect(tmp_path, batch_provider):
    batch_provider(statuses=("validating", "in_progress", "completed"))
    task, store, requests = prepare_batch(tmp_path, n=3)
    job = submit_batch(store, requests)
    assert store.run()["status"] == "submitted"
    [b] = store.read("batches")
    assert (b["batch_id"], b["round"], b["status"], b["n_requests"], b["collected"]) == (job.id, 0, "pending", 3, False)
    assert Path(b["input_file"]).parent == store.batch_files_dir     # batch files live in the run dir

    not_yet = collect_batch(store, task, wait=False, progress=False)
    assert not_yet["status"] == "running" and store.read("responses") == []

    summary = collect_batch(store, task, wait=False, progress=False)
    assert summary["n_finished"] == summary["n_valid"] == 3
    assert {r["batch_id"] for r in store.read("responses")} == {job.id}
    assert all(b["collected"] for b in store.read("batches")) and store.run()["status"] == "completed"
    with pytest.raises(RuntimeError, match="no uncollected batch"):
        collect_batch(store, task)
    check_foreign_keys(store)


def test_batch_retry_rounds_recorded(tmp_path, batch_provider):
    batch_provider()
    task, store, requests = prepare_batch(tmp_path, n=2, retry_mode="batch", poll_interval=0.001)
    batch_provider(invalid_first_time={requests[0].id})
    submit_batch(store, requests)
    summary = collect_batch(store, task, wait=True, progress=False)
    assert summary["n_valid"] == 2 and summary["n_regenerated"] == 1
    assert sorted(b["round"] for b in store.read("batches")) == [0, 1]
    rows = [r for r in store.read("responses") if r["request_id"] == requests[0].id]
    assert [r["attempt"] for r in rows] == [1, 2] and all(r["batch_id"] for r in rows)
    check_foreign_keys(store)


def test_batch_collect_retries_live_by_default(tmp_path, batch_provider):
    batch_provider()
    task, store, requests = prepare_batch(tmp_path, n=2)
    batch_provider(invalid_first_time={requests[1].id})
    submit_batch(store, requests)
    collect_batch(store, task, wait=True, progress=False)
    rows = [r for r in store.read("responses") if r["request_id"] == requests[1].id]
    assert [r["batch_id"] is None for r in rows] == [False, True]   # retry was a live call
    assert [b["round"] for b in store.read("batches")] == [0]


# ---- settings: the task file, its edits, and the run's record -------------------------------
def test_the_run_s_settings_are_its_config(tmp_path, scripted):
    config = tarot().config
    assert (config.model.provider, config.model.name, dict(config.model.parameters)) == \
        ("tarot-scripted", None, {"max_tokens": 3600})                                   # edits, and the file
    x = config.execution
    assert (x.n, x.seed, x.mode, x.workers, config.validation.retries) == (4, 123, "live", 2, 2)
    assert (x.retry_mode, x.poll_interval, x.dataset, config.prompt.system) == ("live", 60.0, None, None)  # defaults
    edited = tarot({"execution.workers": 7, "execution.dataset": "d", "validation.retries": 0}).config
    assert (edited.execution.workers, edited.execution.dataset, edited.validation.retries) == (7, "d", 0)
    with pytest.raises(InstantiationError, match="n is needed"):
        prepare_run(tarot({"execution.n": None}), root=tmp_path)
    assert list(tmp_path.glob("run_*")) == []


def test_settings_given_in_python_become_a_config(tmp_path, scripted):
    """A task that computes its inputs in Python gives its settings as a GenerationConfig; they become its config."""
    settings = GenerationConfig(n=2, provider="tarot-scripted", templates=["spread_v1"], params={"max_tokens": 9},
                                system="Be brief.", validate=False, persona_set="tarot_personas",
                                persona_sampling="cycle", persona_ids=["2", "1"], max_workers=3, dataset="d")
    task = Draws()
    with pytest.warns(UserWarning, match="take no `persona` variable"):           # spread_v1 does not mention them
        store, requests = prepare_run(task, settings, root=tmp_path)
    c = task.config
    assert (c.model.parameters, c.prompt.system, c.validation, c.execution.workers, c.execution.dataset) == \
        ({"max_tokens": 9}, "Be brief.", None, 3, "d")
    persona = c.instantiation.state["persona"]
    assert (persona.set, persona.sample, persona.select) == ("tarot_personas", "cycle", ("2", "1"))
    assert [p["persona_number"] for p in store.read("personas")] == [2, 1]           # in that order, cycled
    assert requests[0].messages[0] == {"role": "system", "content": "Be brief."}
    stored = store.run()["config"]
    assert stored["persona_sampling"] == "cycle" and stored["task_options"] == {"deck": ["Death", "The Sun"]}
    assert load_config(store) == settings                                             # stored flat, as given
    rebuilt = InferenceTaskConfig.from_run(store.run())
    assert (rebuilt.execution.workers, rebuilt.prompt.system, rebuilt.validation) == (3, "Be brief.", None)
    with pytest.raises(ValueError, match="persona_ids given without a persona_set"):
        prepare_run(Draws(), GenerationConfig(n=1, provider="tarot-scripted", persona_ids=["1"]), root=tmp_path)


def test_an_edit_is_checked_like_the_file(scripted):
    with pytest.raises(TaskConfigError, match="'Deth' is not one of the states"):
        tarot({"instantiation.state.cards.steps.fixed.Outcome": "Deth"})
    with pytest.raises(TaskConfigError, match="cannot set model.name.x: model.name is 'gpt"):
        InferenceTaskConfig.from_file(TAROT_FILE, "tarot_celtic_cross", {"model.name.x": 1})


def test_resume_uses_the_task_the_run_recorded(tmp_path, scripted):
    """Not the task file as it is now: it may have changed since the run started."""
    copy = tmp_path / "app"
    copy.mkdir()
    (copy / "vector_space_config.json").write_bytes((TAROT_FILE.parent / "vector_space_config.json").read_bytes())
    (copy / "tasks.yaml").write_text(TAROT_FILE.read_text(encoding="utf-8"), encoding="utf-8")
    task = InferenceTask.from_config(InferenceTaskConfig.from_file(
        copy / "tasks.yaml", None, {"model.provider": "tarot-scripted", "execution.n": 2}))
    store, requests = prepare_run(task, root=tmp_path / "runs")
    run_live(store, task, requests[:1], progress=False)

    (copy / "tasks.yaml").write_text("not: [a task file", encoding="utf-8")         # the file is gone, in effect
    rebuilt = applications.task_of(store.run())
    assert type(rebuilt) is TarotReadings and rebuilt.config.sha256 == task.config.sha256
    assert rebuilt.config.execution == task.config.execution and not rebuilt.before_task_file
    assert [c.name for c in rebuilt.validator().checks] == [c.name for c in task.validator().checks]
    assert run_live(store, rebuilt, progress=False)["n_valid"] == 2


def test_run_task_and_the_repeated_seed_warning(tmp_path, scripted):
    edits = {"model.provider": "tarot-scripted", "execution.n": 1, "execution.dataset": "d", "execution.seed": 7}
    first = run_task("tarot_celtic_cross", edits, root=tmp_path, progress=False)
    assert first.run()["status"] == "completed" and first.run()["config"]["seed"] == 7
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run_task("tarot_celtic_cross", {**edits, "execution.seed": 8, "execution.mode": "dry_run"}, root=tmp_path)
        run_task("tarot_celtic_cross", {**edits, "execution.mode": "dry_run", "instantiation.state.cards.steps.replacement": False},
                 root=tmp_path)                                                       # another instantiation
        assert caught == []
        run_task(TAROT_FILE, {**edits, "execution.mode": "dry_run"}, root=tmp_path)    # by its file, the same
    assert len(caught) == 1 and "already has run(s) with --seed 7 and the same instantiation" in str(caught[0].message)


# ---- CLI --------------------------------------------------------------------------------
def test_cli_generate_status_list_resume(tmp_path, scripted, capsys):
    out = str(tmp_path)
    assert main(["--out", out, *RUN, "--n", "2", "--seed", "5", "--temperature", "0.7", "--max-tokens", "1234",
                 "--extra-body", '{"top_k": 40}', "--set", "instantiation.state.cards.steps.fixed={Outcome: Death}",
                 "--name", "cli-test"]) == 0
    [run_dir] = list(tmp_path.glob("run_*"))
    store = RunStore.open(run_dir.name, tmp_path)
    run = store.run()
    assert run["status"] == "completed" and run["name"] == "cli-test"
    config = InferenceTaskConfig.from_run(run)
    assert config.model.parameters == {"max_tokens": 1234, "temperature": 0.7}
    assert config.model.extra_body == {"top_k": 40} and run["config"]["seed"] == 5
    assert all(i["payload"]["cards"]["Outcome"] == "Death" for i in store.read("inputs"))
    assert config.edits["instantiation.state.cards.steps.fixed"] == {"Outcome": "Death"}
    assert main(["--out", out, "status", run_dir.name]) == 0
    assert main(["--out", out, "list"]) == 0
    assert "cli-test" in capsys.readouterr().out
    assert main(["--out", out, "resume", run_dir.name]) == 0            # nothing left to do


def test_dataset_groups_runs_across_sessions(tmp_path, scripted, capsys):
    gen = ["--out", str(tmp_path), *RUN]
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

    assert main(["--out", str(tmp_path), "list", "--dataset", "tarot-v1"]) == 0
    listing = capsys.readouterr().out
    assert listing.count("dataset=tarot-v1") == 2 and "dataset=other" not in listing
    assert "dataset 'tarot-v1': 2 run(s), 5/5 finished, 5 valid" in listing
    assert main(["--out", str(tmp_path), "list", "--dataset", "missing"]) == 0
    assert "No runs" in capsys.readouterr().out


def test_reusing_a_seed_within_a_dataset_warns(tmp_path, scripted, capsys):
    gen = ["--out", str(tmp_path), *RUN, "--n", "1", "--dry-run"]
    main([*gen, "--dataset", "d", "--seed", "7"])
    assert "WARNING" not in capsys.readouterr().err
    main([*gen, "--dataset", "d", "--seed", "8"])                           # different seed: fine
    main([*gen, "--dataset", "elsewhere", "--seed", "7"])                   # other dataset: fine
    main([*gen, "--dataset", "d", "--seed", "7", "--set", "instantiation.state.cards.steps.replacement=false"])
    main([*gen, "--seed", "7"])                                             # no dataset: fine
    assert "WARNING" not in capsys.readouterr().err                         # (another instantiation: fine)
    main([*gen, "--dataset", "d", "--seed", "7"])
    assert "already has run(s) with --seed 7" in capsys.readouterr().err


def test_cli_dry_run_and_argument_errors(tmp_path, scripted, capsys):
    assert main(["--out", str(tmp_path), *RUN, "--n", "3", "--dry-run"]) == 0
    assert scripted.calls == {}
    with pytest.raises(SystemExit):
        main(["--out", str(tmp_path), *RUN, "--n", "1", "--set", "no-equals-sign"])
    with pytest.raises(SystemExit):
        main(["--out", str(tmp_path), *RUN, "--n", "1", "--extra-body", "[1, 2]"])
    capsys.readouterr()
    assert main(["--out", str(tmp_path), *RUN, "--n", "1", "--template", "celtic_cross"]) == 2       # not pinned
    assert "'celtic_cross' is not a prompt template's id" in capsys.readouterr().err
    assert main(["--out", str(tmp_path), *RUN, "--n", "1", "--set", "execution.mode=nightly"]) == 2
    assert "'nightly' is not one of ['live', 'batch', 'dry_run']" in capsys.readouterr().err
    assert len(list(tmp_path.glob("run_*"))) == 1                            # only the dry run was written


def test_cli_batch_collect(tmp_path, batch_provider):
    batch_provider(statuses=("validating", "in_progress", "completed"))
    out = str(tmp_path)
    assert main(["--out", out, "run", "tarot_celtic_cross", "--n", "2", "--provider", "fake-batch", "--mode", "batch"]) == 0
    [run_dir] = list(tmp_path.glob("run_*"))
    assert main(["--out", out, "collect", run_dir.name]) == 1           # still running
    assert main(["--out", out, "collect", run_dir.name]) == 0
    assert RunStore.open(run_dir.name, tmp_path).run()["status"] == "completed"
    assert main(["--out", out, "resume", run_dir.name]) == 2            # resume is live-only


# ---- templates and personas in the pipeline --------------------------------------------------
def test_templates_are_the_task_file_s(tmp_path, scripted):
    _, plain, _ = prepare(tmp_path, n=1, mode="dry_run")
    _, with_people, _ = prepare(tmp_path, PERSONA, n=1, mode="dry_run")
    templates = lambda store: [t["template_id"] for t in store.read("prompt_templates")]
    assert templates(plain) == templates(with_people) == ["celtic_cross_v4"]
    assert plain.read("personas") == [] and plain.read("inputs")[0]["persona_id"] is None
    [t] = plain.read("prompt_templates")
    assert t["template_sha256"] == get_template("celtic_cross_v4").sha256 and t["user_template"] == get_template(
        "celtic_cross_v4").user


class Draws(InferenceTask):
    """A task computing its inputs in Python: a card per position from its own deck."""
    name, default_template = "draws", "spread_v1"

    def __init__(self, deck=("Death", "The Sun")):
        super().__init__()
        self.deck = list(deck)

    def options(self):
        return {"deck": self.deck}

    def sample_inputs(self, n, rng):
        return [{p: rng.choice(self.deck) for p in ("Past", "Future")} for _ in range(n)]

    def template_variables(self, payload, persona):
        return {"cards": payload, "persona": persona}


class Unrenderable(InferenceTask):
    """A task from hooks whose second input lacks what its template needs."""
    name, input_type, default_template = "unrenderable", "draw", "spread_v1"

    def sample_inputs(self, n, rng):
        return [{"Past": "Death"}, None]

    def template_variables(self, payload, persona):
        return {"cards": payload}


def test_a_template_that_cannot_render_leaves_no_run_behind(tmp_path, scripted):
    with pytest.raises(TaskConfigError, match="does not render with a sample instant: could not render: "
                                            "'dict object' has no attribute 'Nowhere'"):   # found when it is loaded
        tarot({"prompt.templates": [{"inline": "Read {{ cards.Nowhere }}"}]})
    with pytest.raises(ValueError, match="missing required variables"):                   # or before anything is written
        prepare_run(Unrenderable(), GenerationConfig(n=2, provider="tarot-scripted"), root=tmp_path)
    assert list(tmp_path.glob("run_*")) == []


def test_personas_and_two_templates_end_to_end(tmp_path, scripted):
    task, store, requests = prepare(tmp_path, {
        "instantiation.state.persona": {"personas": "tarot_personas", "sample": "cycle"},
        "prompt.templates": [{"template": "celtic_cross_v1"}, {"template": "celtic_cross_v2"}]})
    assert len(requests) == 8 and len(store.read("inputs")) == 4 and len(store.read("prompts")) == 8

    inputs = store.read("inputs")
    personas = {p["persona_id"]: p for p in store.read("personas")}
    assert [personas[i["persona_id"]]["persona_number"] for i in inputs] == [1, 2, 3, 4]   # cycle order
    assert all(i["payload"]["persona"] == i["persona_id"] for i in inputs)                # stored by id
    assert all(p["persona_set"] == "tarot_personas" and p["content_sha256"] for p in personas.values())

    by_input = {}
    for p in store.read("prompts"):
        by_input.setdefault(p["input_id"], {})[p["template"]] = p
    for inp in inputs:
        v1, v2 = by_input[inp["input_id"]]["celtic_cross_v1"], by_input[inp["input_id"]]["celtic_cross_v2"]
        first_name = personas[inp["persona_id"]]["payload"]["name"]["first"]
        assert f"- Name: {first_name}" in v2["user_text"] and "about me" not in v1["user_text"]
        cards = "\n".join(f"{k}: {v}" for k, v in inp["payload"]["cards"].items())
        assert v1["user_text"].endswith(cards) and v2["user_text"].endswith(cards)     # same draw in both
        assert v1["template_sha256"] != v2["template_sha256"]

    assert [r.to_dict() for r in rebuild_requests(store)] == [r.to_dict() for r in requests]
    assert requests[1].metadata["persona_id"] == inputs[0]["persona_id"]
    assert requests[1].metadata["template"] == "celtic_cross_v2"

    summary = run_live(store, task, requests, progress=False)
    assert summary["n_finished"] == summary["n_valid"] == 8
    assert summary["by_template"] == {
        t: {"n_requests": 4, "n_finished": 4, "n_valid": 4, "n_invalid": 0, "n_valid_first_try": 4}
        for t in ("celtic_cross_v1", "celtic_cross_v2")}
    assert store.run()["config"]["task_config"]["references"]["persona_sets"]["tarot_personas"]
    check_foreign_keys(store)
    check_primary_keys(store)


def test_a_seed_gives_the_same_draws_with_or_without_personas(tmp_path, scripted):
    _, a, _ = prepare(tmp_path, n=5, seed=9, mode="dry_run")
    _, b, _ = prepare(tmp_path, PERSONA, n=5, seed=9, mode="dry_run")
    assert [i["payload"]["cards"] for i in a.read("inputs")] == [i["payload"]["cards"] for i in b.read("inputs")]


def test_persona_selection_and_errors(tmp_path, scripted):
    _, store, _ = prepare(tmp_path, {"instantiation.state.persona": {"personas": "tarot_personas",
                                                                     "select": ["3", "Maya Delgado"]}},
                          n=6, mode="dry_run")
    assert sorted(p["persona_number"] for p in store.read("personas")) == [1, 3]
    with pytest.raises(TaskConfigError, match="No persona 'nobody'"):
        tarot({"instantiation.state.persona": {"personas": "tarot_personas", "select": ["nobody"]}})
    with pytest.raises(TaskConfigError, match="No persona set 'no_such_set'"):
        tarot({"instantiation.state.persona": {"personas": "no_such_set"}})


def test_prompt_sample_gives_each_instant_one_template(tmp_path, scripted):
    two = [{"template": "celtic_cross_v1"}, {"template": "celtic_cross_v4"}]
    _, cycled, _ = prepare(tmp_path, {"prompt.templates": two, "prompt.sample": "cycle"}, n=4, mode="dry_run")
    assert [p["template"] for p in cycled.read("prompts")] == ["celtic_cross_v1", "celtic_cross_v4"] * 2
    _, crossed, _ = prepare(tmp_path, {"prompt.templates": two}, n=4, mode="dry_run")
    assert len(crossed.read("prompts")) == 8
    _, drawn, _ = prepare(tmp_path, {"prompt.templates": two, "prompt.sample": "random"}, n=4, seed=9, mode="dry_run")
    assert len(drawn.read("prompts")) == 4
    _, again, _ = prepare(tmp_path, {"prompt.templates": two, "prompt.sample": "random"}, n=4, seed=9, mode="dry_run")
    assert [p["template"] for p in again.read("prompts")] == [p["template"] for p in drawn.read("prompts")]


def test_given_instants_and_an_inline_template(tmp_path, scripted):
    draws = [{"cards": {**{p: "The Sun" for p in POSITIONS}, "Outcome": card}, "persona": number}
             for card, number in (("Death", 2), ("The Moon reversed", "Maya Delgado"))]
    _, store, _ = prepare(tmp_path, {
        **PERSONA, "execution.n": None, "instantiation.instants": draws,
        "prompt.templates": [{"inline": "Read these for {{ who.name.first }}:\n\n{{ cards | kv_lines }}",
                              "name": "plain", "bind": {"who": "persona"}}]}, mode="dry_run")
    inputs = store.read("inputs")
    assert [i["payload"]["cards"]["Outcome"] for i in inputs] == ["Death", "The Moon reversed"]   # n is theirs
    assert [p["persona_number"] for p in store.read("personas")] == [2, 1]
    [template] = store.read("prompt_templates")
    assert (template["template_id"], template["status"]) == ("plain_v1", "inline")
    assert store.read("prompts")[1]["user_text"].startswith("Read these for Maya:")


def test_n_against_exhaustive_sampling_is_refused(tmp_path, scripted, capsys):
    exhaustive = ["--set", "instantiation.state.persona={personas: tarot_personas, sample: exhaustive, select: [1, 2]}"]
    assert main(["--out", str(tmp_path), *RUN, "--dry-run", *exhaustive, "--n", "3"]) == 2
    assert "gives 2 combinations (2): n must be 2, not 3" in capsys.readouterr().err
    assert list(tmp_path.glob("run_*")) == []
    assert main(["--out", str(tmp_path), *RUN, "--dry-run", *exhaustive, "--set", "execution.n=null"]) == 0  # n is theirs


def test_first_try_validity_per_template(tmp_path, scripted):
    task, store, requests = prepare(tmp_path, n=3)
    scripted.plan = {requests[0].metadata["input_id"]: ["invalid", "ok"]}
    stats = run_live(store, task, requests, progress=False)["by_template"]["celtic_cross_v4"]
    assert stats == {"n_requests": 3, "n_finished": 3, "n_valid": 3, "n_invalid": 0, "n_valid_first_try": 2}


def test_cli_personas_and_templates(tmp_path, scripted, capsys):
    out = str(tmp_path)
    assert main(["--out", out, *RUN, "--n", "2", "--personas", "tarot_personas",
                 "--set", "instantiation.state.persona.select=[7]",
                 "--template", "celtic_cross_v1", "--template", "celtic_cross_v2"]) == 0
    captured = capsys.readouterr()
    assert "templates: celtic_cross_v1, celtic_cross_v2  personas: tarot_personas" in captured.out
    assert "celtic_cross_v2: 2/2 finished; 100% valid, 100% on the first try" in captured.out
    assert "note: no `persona` in ['celtic_cross_v1']" in captured.err
    [run_dir] = list(tmp_path.glob("run_*"))
    store = RunStore.open(run_dir.name, tmp_path)
    assert [p["persona_number"] for p in store.read("personas")] == [7] and len(store.read("requests")) == 4


# ---- runs made before the task file ------------------------------------------------------------
def as_schema_version_4(store: RunStore) -> None:
    """Rewrite the run's row as version 4 wrote it: `application`, and config.application_options."""
    run = store.run()
    config_v4 = {("application_options" if k == "task_options" else k): v for k, v in run["config"].items()}
    row = {("application" if k == "task" else k): v for k, v in {**run, "config": config_v4, "schema_version": 4}.items()}
    store.path("runs").write_text(json.dumps(row) + "\n", encoding="utf-8")


def as_run_before_task_files(store: RunStore, version: int) -> None:
    """
    Rewrite a run as runs were written before task files: each input the draw itself, the deck and
    draw settings as the class's options; before version 5 those were `application`/application_options,
    and version 1 had one `template`, no persona or template-hash columns and no prompt_templates.
    """
    def rewrite(table, rows):
        store.path(table).write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")

    run = store.run()
    c = InferenceTaskConfig.from_run(run)
    config = asdict(GenerationConfig(
        n=c.execution.n, provider=c.model.provider, model=c.model.name, templates=[t.id for t in c.prompt.templates],
        params=dict(c.model.parameters), extra_body=dict(c.model.extra_body), mode=c.execution.mode,
        seed=run["config"]["seed"], validation_retries=c.validation.retries, max_workers=c.execution.workers,
        name=c.execution.name, dataset=c.execution.dataset))
    config["task_options"] = {"deck_config": str(DECK), "deck_sha256": file_sha256(DECK), "draw": "replacement",
                              "pins": {}, "positions": POSITIONS}
    inputs = [{**r, "payload": r["payload"]["cards"], "input_type": "tarot_draw"} for r in store.read("inputs")]
    if version == 1:
        config = {k: v for k, v in config.items()
                  if k not in ("templates", "persona_set", "persona_sampling", "persona_ids", "persona_set_sha256")}
        config["template"] = c.prompt.templates[0].id
        inputs = [{k: v for k, v in r.items() if k != "persona_id"} for r in inputs]
        rewrite("prompts", [{k: v for k, v in r.items() if k != "template_sha256"} for r in store.read("prompts")])
        store.path("prompt_templates").unlink()
    rewrite("inputs", inputs)
    rewrite("runs", [{**run, "config": config, "schema_version": version}])
    if version < 5:
        as_schema_version_4(store)
        rewrite("runs", [{**json.loads(store.path("runs").read_text(encoding="utf-8")), "schema_version": version}])


def test_runs_written_by_schema_version_1_still_work(tmp_path, scripted):
    task, store, requests = prepare(tmp_path, {"prompt.templates": [{"template": "celtic_cross_v1"}]}, n=2)
    as_run_before_task_files(store, 1)
    assert load_config(store).templates == ["celtic_cross_v1"]
    rebuilt = rebuild_requests(store)
    assert [r.messages for r in rebuilt] == [r.messages for r in requests]
    assert all(r.metadata["persona_id"] is None for r in rebuilt)
    old = applications.task_of(store.run())
    assert type(old) is TarotReadings and old.before_task_file and old.config.prompt.templates[0].id == "celtic_cross_v1"
    summary = run_live(store, old, progress=False)                       # resume an old run
    assert summary["n_valid"] == 2 and summary["by_template"]["celtic_cross_v1"]["n_finished"] == 2


def test_runs_written_before_schema_version_5_still_work(tmp_path, scripted, capsys):
    """Their row names the task `application`; everything reads it through RunStore.run(), in the current shape."""
    task, store, requests = prepare(tmp_path, n=2, dataset="d", seed=7)
    run_live(store, task, requests[:1], progress=False)
    as_run_before_task_files(store, 4)
    on_disk = lambda: json.loads(store.path("runs").read_text(encoding="utf-8"))
    assert on_disk()["application"] == "tarot_celtic_cross" and "task" not in on_disk()

    run = store.run()
    assert run["task"] == "tarot_celtic_cross" and "application" not in run and run["schema_version"] == 4
    assert run["config"]["task_options"]["draw"] == "replacement" and "application_options" not in run["config"]
    assert [r["task"] for r in list_runs(tmp_path)] == ["tarot_celtic_cross"]
    assert type(applications.task_of(run)) is TarotReadings
    out = str(tmp_path)
    assert main(["--out", out, "status", store.run_id]) == 0 and main(["--out", out, "list"]) == 0
    assert "tarot_celtic_cross" in capsys.readouterr().out
    assert on_disk()["application"] == "tarot_celtic_cross"                 # reading leaves the file as it was

    main(["--out", out, *RUN, "--n", "1", "--dry-run", "--dataset", "d", "--seed", "7"])
    assert "already has run(s) with --seed 7" in capsys.readouterr().err    # an older run with this seed may repeat
    assert main(["--out", out, "resume", store.run_id]) == 0                # the task is rebuilt from the old row
    assert store.run()["status"] == "completed" and len(final_responses(store)) == 2
    assert on_disk()["task"] == "tarot_celtic_cross" and on_disk()["schema_version"] == 4     # rewritten by the update


# ---- tasks are found by name; an application can be a task file alone ---------------------------
THREE_CARDS = """
three_cards:
  description: Three cards from a short deck, read with the general spread prompt.
  engine: inference
  instantiation:
    state:
      cards:
        steps: {names: [Past, Present, Future], states: [Death, The Sun, The Moon]}
  prompt:
    templates:
      - template: spread_v1
  model:
    provider: tarot-scripted
  execution:
    n: 3
"""


def test_tasks_are_found_in_the_application_folders():
    found = applications.tasks()
    assert found["tarot_celtic_cross"] is TarotReadings
    assert {name: task.cli for name, task in found.items()} == {
        "tarot_celtic_cross": None, "celtic_cross_evaluation": "evaluate", "persona_expectations": None}
    assert all(task.__module__.startswith("applications.tarot.") for task in found.values())
    assert applications.task_names() == {"tarot_celtic_cross": TAROT_FILE}
    assert applications.task_file_of("tarot_celtic_cross") == TAROT_FILE
    with pytest.raises(KeyError, match="No task named 'tarot_spread' in the task files; they define: tarot_celtic_cross"):
        applications.task_file_of("tarot_spread")


def test_the_script_runs_the_task_files_tasks(tmp_path, scripted, capsys):
    assert main(["--out", str(tmp_path), "run", "celtic_cross_evaluation", "--n", "1"]) == 2   # that is evaluate's
    assert "No task named 'celtic_cross_evaluation' in the task files" in capsys.readouterr().err
    assert main(["--out", str(tmp_path), "run", "--n", "1"]) == 2                              # a task must be named
    assert "Name a task, or give --task-file" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        main(["--out", str(tmp_path), "run", "--help"])
    help_text = " ".join(capsys.readouterr().out.split())                                     # undo the line wrapping
    assert "tarot_celtic_cross Generate a connected, position-aware Celtic Cross" in help_text
    assert "--personas SET" in help_text and "--set SETTING=VALUE" in help_text


def test_an_application_can_be_a_task_file_alone(tmp_path, scripted, capsys):
    """No Python of its own: its instantiation, prompt and model are all in the file; no validation either."""
    (tmp_path / "tasks.yaml").write_text(THREE_CARDS, encoding="utf-8")
    out = str(tmp_path / "runs")
    assert main(["--out", out, "run", "--task-file", str(tmp_path / "tasks.yaml"), "--seed", "1"]) == 0
    [run_dir] = list((tmp_path / "runs").glob("run_three_cards_*"))
    store = RunStore.open(run_dir.name, tmp_path / "runs")
    run = store.run()
    assert (run["task"], run["status"]) == ("three_cards", "completed")
    assert [t["template_id"] for t in store.read("prompt_templates")] == ["spread_v1"]
    assert run["summary"]["n_unvalidated"] == 3 and store.read("input_items") == []
    assert all(set(i["payload"]["cards"].values()) <= {"Death", "The Sun", "The Moon"} for i in store.read("inputs"))
    assert main(["--out", out, "resume", run_dir.name]) == 0
    assert main(["--out", out, "list"]) == 0 and "three_cards" in capsys.readouterr().out
    assert main(["--out", out, "run", "--task-file", str(tmp_path / "tasks.yaml"), "three_cards", "--dry-run"]) == 0
    assert run_task(f"{tmp_path / 'tasks.yaml'}::three_cards", {"execution.mode": "dry_run"},
                    root=tmp_path / "runs").run()["task"] == "three_cards"
    yaml.safe_load(THREE_CARDS)                                              # (the file is plain YAML)
