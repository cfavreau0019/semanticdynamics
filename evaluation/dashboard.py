"""
Evaluation dashboard: one self-contained HTML file built from evaluation runs.

    python -m evaluation.dashboard                         # every evaluation run under data/evaluations
    python -m evaluation.dashboard --dataset eval-quick --open
    python -m evaluation.dashboard --task tarot_celtic_cross   # only evaluations of that task's texts
    python -m evaluation.dashboard --run <run_id> --run <run_id> --out report.html

The file embeds its data (scores, comments and the evaluated texts), needs no server and no
network, and is rebuilt by running the command again. In it, scores can be aggregated by
task, text, evaluating model, generating model, persona and rubric question; each text has
its score distributions and the evaluators' comments.

    task              the application of the generation run the text came from
    generating model  the model that wrote the text
    evaluating model  the model that filled in the form

`python evaluation/dashboard.py ...` works too.
"""
import argparse
import json
import sys
import webbrowser
from pathlib import Path
from typing import Any, Optional, Sequence

if __package__ in (None, ""):  # allow `python evaluation/dashboard.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_generation.pipeline import list_runs  # noqa: E402
from data_generation.store import DEFAULT_ROOT as READINGS_ROOT, REPO_ROOT, RunStore, utc_now  # noqa: E402

EVALUATIONS_ROOT = REPO_ROOT / "data" / "evaluations"
TEMPLATE = Path(__file__).resolve().parent / "dashboard.html"
DATA_MARKER = "__DASHBOARD_DATA__"


def _by(rows: list[dict], key: str) -> dict[Any, dict]:
    return {r[key]: r for r in rows}


def _generation_tables(root: Path, run_id: str) -> Optional[dict]:
    """What the dashboard needs from a generation run, or None if the run is not there."""
    try:
        store = RunStore.open(run_id, root)
    except FileNotFoundError:
        return None
    return {"run": store.run(), "texts": {t["response_id"]: t["text"] for t in store.read("response_texts")},
            "responses": _by(store.read("responses"), "response_id")}


def _persona(row: dict) -> dict:
    payload = row.get("payload") or {}
    name = row.get("name")
    if isinstance(name, dict):
        name = name.get("full")
    demographics, profile = payload.get("demographics") or {}, payload.get("tarot_profile") or {}
    facts = [demographics.get("age"), demographics.get("profession"), profile.get("experience_level")]
    return {"name": name or f"Persona {row.get('persona_number')}", "number": row.get("persona_number"),
            "facts": [str(f) for f in facts if f not in (None, "")], "description": payload.get("description")}


def build_data(evaluations_root: str | Path = EVALUATIONS_ROOT, runs: Sequence[str] = (),
               dataset: Optional[str] = None, readings_root: Optional[str | Path] = None,
               tasks: Sequence[str] = ()) -> dict:
    """
    The dashboard's data: every evaluation of the selected runs, joined to the text it scored.

    runs / dataset — restrict to these evaluation runs / to one evaluation dataset
    readings_root  — where the generation runs are; default: where each evaluation run found them
    tasks          — keep only evaluations of texts of these tasks; evaluation runs with none are left out
    A text whose generation run is missing is still listed, without its text or task.
    """
    data: dict[str, Any] = {"generated_at": utc_now(), "rubrics": {}, "runs": [], "personas": {}, "subjects": {},
                            "evaluations": []}
    generation: dict[tuple[str, str], Optional[dict]] = {}
    for run in list_runs(evaluations_root, dataset):
        if runs and run["run_id"] not in runs:
            continue
        store = RunStore.open(run["run_id"], evaluations_root)
        rubrics = store.read("rubrics")
        if not rubrics:                      # not an evaluation run (e.g. persona expectations)
            continue
        for r in rubrics:
            data["rubrics"][r["rubric_sha256"]] = {**r["definition"], "id": r["rubric_id"]}
        for p in store.read("personas"):
            data["personas"][p["persona_id"]] = _persona(p)

        evaluations = store.read("evaluations")
        summary = run.get("summary") or {}
        data["runs"].append({
            "run_id": run["run_id"], "name": run.get("name"), "dataset": run.get("dataset"), "status": run["status"],
            "provider": run["provider"], "model": run.get("model"), "created_at": run["created_at"],
            "n_evaluations": len(evaluations),
            **{k: summary.get(k) for k in ("n_requests", "n_finished", "n_valid", "n_invalid", "n_regenerated",
                                           "total_tokens")}})
        if not evaluations:
            continue

        options = run["config"].get("application_options") or {}
        root = Path(readings_root or options.get("subjects_root") or READINGS_ROOT)
        prompts, requests = _by(store.read("prompts"), "prompt_id"), _by(store.read("requests"), "request_id")
        inputs, responses = _by(store.read("inputs"), "input_id"), _by(store.read("responses"), "response_id")
        scores: dict[str, list[dict]] = {}
        for s in store.read("evaluation_item_scores"):
            scores.setdefault(s["evaluation_id"], []).append(s)
        flags: dict[str, list[dict]] = {}
        for f in store.read("evaluation_red_flags"):
            flags.setdefault(f["evaluation_id"], []).append(f)

        subjects: dict[str, dict] = {}       # every subject of this run, before the task selection
        n_kept = 0
        for e in evaluations:
            sid, items = e["subject_response_id"], scores.get(e["evaluation_id"], [])
            if sid not in subjects:
                payload = inputs[prompts[requests[e["request_id"]]["prompt_id"]]["input_id"]]["payload"]
                key = (str(root), e["subject_run_id"])
                if key not in generation:
                    generation[key] = _generation_tables(root, e["subject_run_id"])
                source = generation[key]
                source_run = source["run"] if source else {}
                subjects[sid] = {
                    "task": source_run.get("application") or e["subject_type"], "run_id": e["subject_run_id"],
                    "dataset": source_run.get("dataset"), "template": payload.get("subject_template"),
                    "model": payload.get("subject_model") or (source["responses"].get(sid, {}).get("model")
                                                              if source else None) or source_run.get("model"),
                    "persona_id": payload.get("subject_persona_id"), "input": payload.get("subject_input"),
                    "text": source["texts"].get(sid) if source else None, "words": e.get("subject_length_words"),
                    "labels": {s["item_key"]: s["label"] for s in items if s.get("label")}}
            if tasks and subjects[sid]["task"] not in tasks:
                continue
            data["subjects"][sid] = subjects[sid]
            n_kept += 1
            model = responses.get(e["response_id"], {}).get("model") or run.get("model")
            data["evaluations"].append({
                "id": e["evaluation_id"], "run": run["run_id"], "sid": sid, "pid": e.get("persona_id"),
                "ev": model or f"{run['provider']} (default model)", "r": e["rubric_sha256"],
                "total": e["total_score"], "pre": e["score_before_caps"], "cap": e["cap_applied"], "grade": e["grade"],
                "sec": e["section_scores"], "out": e.get("outputs") or {},
                "s": {s["item_key"]: (None if s["is_na"] else s["score"]) for s in items},
                "j": {s["item_key"]: s["justification"] for s in items if s.get("justification")},
                "flags": [{"id": f["flag_id"], "severity": f["severity"], "quote": f["quote"]}
                          for f in flags.get(e["evaluation_id"], [])]})
        data["runs"][-1]["n_evaluations"] = n_kept
    if tasks:
        data["runs"] = [r for r in data["runs"] if r["n_evaluations"]]
    return data


def render(data: dict) -> str:
    """The dashboard page with `data` embedded."""
    embedded = json.dumps(data, ensure_ascii=False, separators=(",", ":"), default=str).replace("</", "<\\/")
    return TEMPLATE.read_text(encoding="utf-8").replace(DATA_MARKER, embedded)


def write_dashboard(out: str | Path, **selection) -> tuple[Path, dict]:
    """Build the data (see build_data) and write the page; returns (path, data)."""
    data = build_data(**selection)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(data), encoding="utf-8")
    return out, data


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="dashboard", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--evaluations-root", default=str(EVALUATIONS_ROOT),
                        help="where the evaluation runs are (default: %(default)s)")
    parser.add_argument("--run", action="append", default=[], metavar="RUN_ID",
                        help="include only this evaluation run; repeatable (default: all)")
    parser.add_argument("--dataset", help="include only evaluation runs of this dataset")
    parser.add_argument("--task", action="append", default=[], metavar="NAME",
                        help="include only evaluations of texts of this task, i.e. the application of the generation "
                             "run they came from, e.g. tarot_celtic_cross; repeatable (default: all)")
    parser.add_argument("--readings-root", help="where the generation runs are (default: where each evaluation "
                                                "run found them)")
    parser.add_argument("--out", help="file to write (default: <evaluations-root>/dashboard.html)")
    parser.add_argument("--open", action="store_true", help="open the page in the default browser")
    args = parser.parse_args(argv)

    out, data = write_dashboard(args.out or Path(args.evaluations_root) / "dashboard.html",
                                evaluations_root=args.evaluations_root, runs=args.run, dataset=args.dataset,
                                readings_root=args.readings_root, tasks=args.task)
    if not data["evaluations"]:
        print(f"No evaluations found under {args.evaluations_root}"
              f"{' for task ' + ', '.join(args.task) if args.task else ''}; nothing to show.", file=sys.stderr)
        return 1
    missing = sum(s["text"] is None for s in data["subjects"].values())
    print(f"{len(data['evaluations'])} evaluations of {len(data['subjects'])} texts from {len(data['runs'])} runs "
          f"-> {out}  ({out.stat().st_size / 1e6:.1f} MB)")
    if missing:
        print(f"note: the text of {missing} subjects was not found; pass --readings-root", file=sys.stderr)
    if args.open:
        webbrowser.open(out.resolve().as_uri())
    return 0


if __name__ == "__main__":
    sys.exit(main())
