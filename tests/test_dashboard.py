"""evaluation.dashboard: the data joined from evaluation and generation runs, and the page built from it."""
import json

from data_generation.pipeline import prepare_run, run_live
from evaluation.celtic_cross import EXPECTATION_CONTEXT
from evaluation.dashboard import DATA_MARKER, build_data, main, render
from evaluation.expectations import generate_expectations
from test_evaluation import POSITIONS, RUBRIC, eval_config, evaluation_app, readings, registry  # noqa: F401


def evaluated(tmp_path, registry, readings):
    """An expectations run and an evaluation run (4 readings x 2 personas); returns (eval_root, store, requests)."""
    eval_root = tmp_path / "evaluations"
    stage1 = generate_expectations("tarot_personas", provider="eval-scripted", context=EXPECTATION_CONTEXT,
                                   root=eval_root, progress=False)
    app = evaluation_app(readings, eval_root, personas_per_subject=2, expectations_run=stage1.run_id)
    store, requests = prepare_run(app, eval_config(dataset="eval-demo"), root=eval_root)
    registry.plan = {requests[1].metadata["input_id"]: ["real_flag"]}
    run_live(store, app, requests, progress=False)
    return eval_root, store, requests


def test_build_data_joins_evaluations_to_their_texts(tmp_path, registry, readings):
    eval_root, store, _ = evaluated(tmp_path, registry, readings)
    data = build_data(eval_root)
    assert [r["run_id"] for r in data["runs"]] == [store.run_id]             # the expectations run is not listed
    assert data["runs"][0]["n_evaluations"] == 8 and list(data["rubrics"]) == [RUBRIC.sha256]
    assert data["rubrics"][RUBRIC.sha256]["id"] == "celtic_cross_reading_v1"

    assert len(data["evaluations"]) == 8 and len(data["subjects"]) == 4
    texts = {t["response_id"]: t["text"] for t in readings[1].read("response_texts")}
    for sid, subject in data["subjects"].items():
        assert subject["text"] == texts[sid] and list(subject["input"]) == POSITIONS
        assert (subject["task"], subject["model"], subject["dataset"]) == ("tarot_celtic_cross", "scripted-model", "demo")
        assert subject["template"] == "celtic_cross_v1" and subject["labels"]["2a.1"] == "Present"
    for e in data["evaluations"]:
        assert e["ev"] == "evaluator-model" and e["sid"] in data["subjects"] and e["pid"] in data["personas"]
        assert len(e["s"]) == 47 and e["total"] == (60.0 if e["flags"] else 80.0)
        assert e["out"]["persona_reaction"] and set(e["sec"]) == {s.key for s in RUBRIC.sections}
    assert [e["flags"][0]["id"] for e in data["evaluations"] if e["flags"]] == ["RF5"]
    assert all(p["name"] for p in data["personas"].values())


def test_build_data_selection_and_missing_generation_run(tmp_path, registry, readings):
    eval_root, store, _ = evaluated(tmp_path, registry, readings)
    assert build_data(eval_root, runs=["nope"])["evaluations"] == []
    assert build_data(eval_root, dataset="other")["runs"] == []
    assert len(build_data(eval_root, runs=[store.run_id], dataset="eval-demo")["evaluations"]) == 8
    by_task = build_data(eval_root, tasks=["tarot_celtic_cross"])
    assert (len(by_task["evaluations"]), len(by_task["subjects"]), by_task["runs"][0]["n_evaluations"]) == (8, 4, 8)
    other = build_data(eval_root, tasks=["something_else"])
    assert (other["evaluations"], other["subjects"], other["runs"]) == ([], {}, [])
    assert main(["--evaluations-root", str(eval_root), "--task", "something_else",
                 "--out", str(tmp_path / "other.html")]) == 1
    lost = build_data(eval_root, readings_root=tmp_path / "elsewhere")
    assert len(lost["evaluations"]) == 8                                       # scores survive without the texts
    assert {(s["text"], s["task"], s["model"]) for s in lost["subjects"].values()} == \
        {(None, "tarot_reading", "scripted-model")}


def test_page_embeds_the_data(tmp_path, registry, readings, capsys):
    eval_root, _, _ = evaluated(tmp_path, registry, readings)
    out = tmp_path / "report.html"
    assert main(["--evaluations-root", str(eval_root), "--out", str(out)]) == 0
    assert "8 evaluations of 4 texts from 1 runs" in capsys.readouterr().out
    page = out.read_text(encoding="utf-8")
    assert DATA_MARKER not in page
    embedded = page.split('<script id="data" type="application/json">')[1].split("</script>")[0]
    assert len(json.loads(embedded)["evaluations"]) == 8

    hostile = render({"evaluations": [{"note": "</script><script>alert(1)</script>"}]})
    assert "</script><script>alert(1)" not in hostile                          # text cannot close the data block
    assert main(["--evaluations-root", str(tmp_path / "empty"), "--out", str(tmp_path / "none.html")]) == 1


def test_evaluation_commands_rebuild_the_dashboard(tmp_path, registry, readings, capsys):
    from evaluation.evaluate_tarot import main as evaluate
    root, store = readings
    out = tmp_path / "evaluations"
    page = out / "dashboard.html"
    common = ["evaluate", "--readings-root", str(root), "--readings-run", store.run_id, "--provider", "eval-scripted",
              "--no-personas", "--seed", "5"]
    assert evaluate(["--out", str(out), "--no-dashboard", *common, "--n", "1"]) == 0 and not page.exists()
    assert evaluate(["--out", str(out), *common, "--dry-run"]) == 0 and not page.exists()     # nothing new to show
    assert evaluate(["--out", str(out), *common, "--n", "2"]) == 0
    n_texts = len(build_data(out)["subjects"])                 # both finished runs; they may share a reading
    assert f"dashboard: {page}  (3 evaluations of {n_texts} texts)" in capsys.readouterr().out
    embedded = page.read_text(encoding="utf-8").split('<script id="data" type="application/json">')[1]
    assert len(json.loads(embedded.split("</script>")[0])["evaluations"]) == 3

    page.unlink()
    run_id = build_data(out)["runs"][0]["run_id"]
    assert evaluate(["--out", str(out), "resume", run_id]) == 0 and page.exists()
