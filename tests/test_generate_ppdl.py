"""
inference.generate_ppdl: a run of a PDL file is stored as the run tables of a task file's run.

The PDL interpreter runs for real; only the model call is replaced, by LiteLLM's mock response
(a `mock_response` parameter), so nothing is sent.
"""
from pathlib import Path

import pytest
import yaml

pytest.importorskip("pdl")

from pdl.pdl import exec_dict  # noqa: E402

import applications  # noqa: E402
from inference import generate_ppdl  # noqa: E402
from inference.generate import run_task  # noqa: E402

PDL_FILE = Path(applications.__file__).parent / "tarot" / "tarot_ppdl.yaml"


@pytest.fixture
def mocked(monkeypatch):
    """Executions of the PDL file answer with a mock reading; {index: Exception} makes one fail."""
    monkeypatch.chdir(Path(applications.__file__).parent.parent)   # the PDL file's paths are from the repo root
    failing = {}

    def execute(pdl_file, scope):
        if scope["index"] in failing:
            raise failing[scope["index"]]
        program = yaml.safe_load(Path(pdl_file).read_text(encoding="utf-8"))
        program["defs"]["parameters"]["data"]["mock_response"] = "A short mock reading."
        return exec_dict(program, config={"yield_result": False, "yield_background": False, "batch": 1,
                                          "cwd": Path(pdl_file).parent}, scope=dict(scope), output="all")

    monkeypatch.setattr(generate_ppdl, "execute", execute)
    return failing


def columns(rows, names):
    return [{name: row[name] for name in names} for row in rows]


def test_a_run_has_the_inputs_and_prompts_of_the_task_file_run_with_its_seed(mocked, tmp_path):
    store = generate_ppdl.run_ppdl(PDL_FILE, n=3, seed=7, root=tmp_path, progress=False)
    engine = run_task("tarot_celtic_cross", {"execution.n": 3, "execution.seed": 7, "execution.mode": "dry_run"},
                      root=tmp_path)
    for table, names in [
        ("inputs", ["input_index", "input_type", "payload", "persona_id"]),
        ("input_items", ["item_index", "label", "value", "entity", "qualifier"]),
        ("prompt_templates", ["template_sha256", "template_id", "user_template", "status"]),
        ("prompts", ["template", "template_sha256", "system", "user_text", "messages", "prompt_sha256"]),
    ]:
        assert columns(store.read(table), names) == columns(engine.read(table), names), table
    assert columns(store.read("requests"), ["provider", "model", "mode"]) == \
        [{"provider": "openai", "model": "gpt-6-luna", "mode": "live"}] * 3


def test_every_attempt_is_stored_with_its_validation(mocked, tmp_path):
    store = generate_ppdl.run_ppdl(PDL_FILE, {"max_attempts": 2}, n=2, seed=7, root=tmp_path, progress=False)
    run = store.run()
    assert (run["task"], run["status"], run["provider"], run["model"]) == \
        ("tarot_ppdl", "completed", "openai", "gpt-6-luna")
    assert run["config"]["seed"] == 7 and run["config"]["ppdl"]["data"] == {"max_attempts": 2}

    responses = store.read("responses")                       # the mock reading is too short: both attempts fail
    assert [r["attempt"] for r in responses] == [1, 2, 1, 2]
    assert {r["finish_reason"] for r in responses} == {"stop"} and all(r["total_tokens"] for r in responses)
    assert [t["text"] for t in store.read("response_texts")] == ["A short mock reading."] * 4
    validations = store.read("validations")
    assert [v["passed"] for v in validations] == [False] * 4 and "positions" in validations[0]["details"]
    assert {i["code"] for i in store.read("validation_issues")} >= {"too_short"}
    assert run["summary"]["n_attempts"] == 4 and run["summary"]["n_invalid"] == 2
    assert sorted(p.name for p in (store.dir / "pdl_traces").iterdir()) == ["input_0000.json", "input_0001.json"]


def test_an_execution_that_fails_leaves_the_run_partial(mocked, tmp_path, capsys):
    mocked[1] = RuntimeError("no answer")
    store = generate_ppdl.run_ppdl(PDL_FILE, n=3, seed=7, root=tmp_path, progress=False)
    assert [i["input_index"] for i in store.read("inputs")] == [0, 2]
    assert store.run()["status"] == "partial" and store.run()["summary"]["n_inputs_failed"] == 1
    assert "input 1 failed: RuntimeError: no answer" in capsys.readouterr().err


def test_the_env_files_openai_token_is_the_key_pdl_reads(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_TOKEN", "test-key")
    generate_ppdl.use_env_key()
    assert generate_ppdl.os.environ["OPENAI_API_KEY"] == "test-key"
