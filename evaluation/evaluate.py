"""
Evaluate generated outputs against a rubric from the command line, as judged by an LLM that
can take the role of a persona. It runs any evaluation task of any application
(applications/<application>/). Results are written as tables under
data/evaluations/<run_id>/ (see evaluation/README.md).

Two stages:
  1. expectations  each persona records what they expect before seeing any output
                    (one request per persona; reusable across evaluation runs)
  2. run           each (subject, persona) pair: the persona fills in the form for the
                    subject, a subject being one output of a generation run

    # evaluate 50 readings of dataset tarot-v1, one random persona each (runs stage 1 first)
    python -m evaluation.evaluate run celtic_cross_evaluation --subject-dataset tarot-v1 --n 50 \\
        --provider openai --model gpt-4o-mini

    # reuse expectations, 3 personas per reading, as an OpenAI batch
    python -m evaluation.evaluate expectations celtic_cross_evaluation --provider openai --model gpt-4o-mini
    python -m evaluation.evaluate run celtic_cross_evaluation --subject-dataset tarot-v1 --n 200 \\
        --personas-per-subject 3 --expectations <expectations_run_id> --provider openai --model gpt-4o-mini --mode batch
    python -m evaluation.evaluate collect <run_id> --wait

    python -m evaluation.evaluate status <run_id>
    python -m evaluation.evaluate resume <run_id>
    python -m evaluation.evaluate list

`run --help` lists the evaluation tasks; `run <task> --help` lists every option. collect,
resume and status take only a run id: the run recorded which task it belongs to. Run ids
name the task: run_celtic_cross_evaluation_<date>_<time>_<id> and
run_persona_expectations_<date>_<time>_<id>.

`python evaluation/evaluate.py ...` works too.
"""
import argparse
import inspect
import json
import statistics
import sys
import warnings
from collections import Counter
from pathlib import Path

if __package__ in (None, ""):  # allow `python evaluation/evaluate.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from applications import get_task, task_of, tasks  # noqa: E402
from inference.pipeline import (  # noqa: E402
    GenerationConfig, collect_batch, list_runs, load_config, prepare_run, run_live, submit_batch, summarize,
)
from inference.store import REPO_ROOT, RunStore  # noqa: E402
from evaluation.dashboard import write_dashboard  # noqa: E402
from evaluation.expectations import generate_expectations, load_expectations  # noqa: E402
from evaluation.subjects import load_subjects  # noqa: E402
from personas import SAMPLING_MODES  # noqa: E402

EVALUATIONS_ROOT = REPO_ROOT / "data" / "evaluations"
EXPECTATIONS_MAX_TOKENS = 1500


def _json_arg(text: str) -> dict:
    try:
        value = json.loads(text)
    except json.JSONDecodeError as e:
        raise argparse.ArgumentTypeError(f"not valid JSON: {e}")
    if not isinstance(value, dict):
        raise argparse.ArgumentTypeError("must be a JSON object")
    return value


def task_summary(cls) -> str:
    """The first line of a task's docstring, for --help."""
    return (inspect.getdoc(cls) or "").partition("\n")[0]


def _add_llm_args(p, task_cls, max_tokens: int) -> None:
    p.add_argument("--provider", default="openai", help="llm provider name (default: %(default)s)")
    p.add_argument("--model", help="model name (default: the provider's default from .env)")
    p.add_argument("--max-tokens", type=int, default=max_tokens, help="default: %(default)s")
    p.add_argument("--temperature", type=float)
    p.add_argument("--json-mode", action="store_true",
                   help='ask the provider for JSON output (response_format={"type": "json_object"}); OpenAI supports it')
    p.add_argument("--extra-body", type=_json_arg, default={}, metavar="JSON", help="provider-specific params")
    p.add_argument("--workers", type=int, default=4, help="live mode: concurrent requests (default: %(default)s)")
    p.add_argument("--personas", default=task_cls.default_personas, metavar="SET",
                   help="persona set name or JSON path (default: %(default)s)")
    p.add_argument("--persona", action="append", default=[], metavar="ID",
                   help="restrict to this persona (id, number or full name); repeatable")
    p.add_argument("--name", help="label stored with the run")
    p.add_argument("--dataset", help="dataset label for the evaluation run (groups evaluation runs)")
    p.add_argument("--dry-run", action="store_true", help="write inputs/prompts/requests only; no API calls")


def _add_run_args(e, task_cls) -> None:
    """The options every evaluation run takes. The task adds its own after these (EvaluationTask.add_arguments)."""
    e.add_argument("--n", type=int, help="number of subjects to evaluate (a seeded sample; default: all)")
    e.add_argument("--seed", type=int, help="seed for sampling subjects and personas (default: random, recorded)")
    e.add_argument("--no-personas", action="store_true", help="evaluate without personas (one neutral evaluation "
                                                              "per subject)")
    e.add_argument("--persona-sampling", choices=SAMPLING_MODES, default="random",
                   help="random: seeded draw per subject; cycle: walk through the set so every persona is used "
                        "equally often (default: %(default)s)")
    e.add_argument("--expectations", metavar="RUN_ID", help="reuse this expectations run (default: generate them "
                                                            "first for the personas in scope)")
    e.add_argument("--no-expectations", action="store_true", help="skip stage 1; evaluators get no expectations")
    e.add_argument("--template", help=f"evaluation prompt template from the prompts library "
                                      f"(default: {getattr(task_cls, 'default_template', None)})")
    e.add_argument("--mode", choices=["live", "batch"], default="live")
    e.add_argument("--wait", action="store_true", help="batch mode: wait for the batch and collect it")
    e.add_argument("--no-validate", action="store_true", help="skip answer validation (and retries)")
    e.add_argument("--validation-retries", type=int, default=2, help="default: %(default)s")
    e.add_argument("--retry-mode", choices=["live", "batch"], default="live")
    e.add_argument("--poll-interval", type=float, default=60.0)
    _add_llm_args(e, task_cls, max_tokens=4000)


def _params(args) -> dict:
    params = {"max_tokens": args.max_tokens}
    if args.temperature is not None:
        params["temperature"] = args.temperature
    if args.json_mode:
        params["response_format"] = {"type": "json_object"}
    return params


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="evaluate", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=str(EVALUATIONS_ROOT),
                        help="root directory for evaluation runs (default: %(default)s)")
    parser.add_argument("--no-dashboard", action="store_true",
                        help="do not rebuild <out>/dashboard.html after run, resume or collect")
    sub = parser.add_subparsers(dest="command", required=True)
    evaluation_tasks = {name: cls for name, cls in sorted(tasks().items()) if cls.cli == "evaluate"}

    x = sub.add_parser("expectations", help="stage 1: each persona's expectations before seeing an output")
    x_by_task = x.add_subparsers(dest="task", required=True, metavar="TASK")
    e = sub.add_parser("run", help="stage 2: evaluators fill in the evaluation form for subjects")
    e_by_task = e.add_subparsers(dest="task", required=True, metavar="TASK")
    for name, task_cls in evaluation_tasks.items():
        if task_cls.expectations_task is not None:
            _add_llm_args(x_by_task.add_parser(name, help=task_summary(task_cls.expectations_task)), task_cls,
                          max_tokens=EXPECTATIONS_MAX_TOKENS)
        p = e_by_task.add_parser(name, help=task_summary(task_cls), description=inspect.getdoc(task_cls),
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
        _add_run_args(p, task_cls)
        task_cls.add_arguments(p)

    for name, help_ in [("collect", "collect a batch evaluation run"), ("resume", "finish an interrupted live run"),
                        ("status", "show a run's status and scores")]:
        p = sub.add_parser(name, help=help_)
        p.add_argument("run_id")
        if name == "collect":
            p.add_argument("--wait", action="store_true")
        if name == "resume":
            p.add_argument("--workers", type=int, help="concurrent requests from now on (default: as the run was "
                                                       "started); lower it if the provider refused for concurrency")
    ls = sub.add_parser("list", help="list evaluation and expectations runs")
    ls.add_argument("--dataset")
    return parser


# ---- helpers ----------------------------------------------------------------------------
def score_report(store: RunStore) -> list[str]:
    """Lines summarising an evaluation run's scores, labelled from the rubric the run recorded."""
    evaluations = store.read("evaluations")
    if not evaluations:
        return []
    rubric = next((r["definition"] for r in store.read("rubrics")), {})
    totals = [e["total_score"] for e in evaluations]
    grades = Counter(e["grade"] for e in evaluations)
    lines = [f"  evaluations: {len(evaluations)}   mean total {statistics.fmean(totals):.1f}   "
             f"median {statistics.median(totals):.1f}   range {min(totals):.1f}-{max(totals):.1f}",
             "  grades: " + "  ".join(f"{g}: {grades[g]}" for g in sorted(grades))]
    keys = list(evaluations[0]["section_scores"])
    section_means = {k: statistics.fmean(e["section_scores"][k] for e in evaluations
                                         if e["section_scores"].get(k) is not None) for k in keys}
    scale = f" ({rubric['scale_min']}-{rubric['scale_max']})" if "scale_max" in rubric else ""
    lines.append(f"  section averages{scale}: " + "  ".join(f"{k} {v:.2f}" for k, v in section_means.items()))
    flags = Counter(f["flag_id"] for f in store.read("evaluation_red_flags"))
    lines.append("  red flags: " + ("  ".join(f"{k}: {v}" for k, v in sorted(flags.items())) if flags else "none"))
    ratings = []                                 # the evaluators' own numeric ratings, whatever the rubric asks for
    for output in rubric.get("outputs") or []:
        values = [e["outputs"][output["name"]] for e in evaluations
                  if (e.get("outputs") or {}).get(output["name"]) is not None]
        if output["type"] == "integer" and values:
            ratings.append(f"{output['name'].replace('_', ' ')} ({output['min']}-{output['max']}): "
                           f"{statistics.fmean(values):.2f}")
    if ratings:
        lines.append("  " + "   ".join(ratings))
    return lines


def print_summary(store: RunStore, summary: dict) -> None:
    run = store.run()
    print(f"\nrun {store.run_id}  [{run['status']}]  {run['task']}  {run['provider']}/"
          f"{run['model'] or '(default model)'}{'  dataset=' + run['dataset'] if run.get('dataset') else ''}")
    print(f"  tables: {store.dir}")
    for k in ("n_requests", "n_finished", "n_api_errors", "n_valid", "n_invalid", "n_regenerated", "total_tokens",
              "cached_tokens"):
        print(f"  {k:>14}: {summary.get(k)}")
    for line in score_report(store):
        print(line)
    n_expect = len(store.read("persona_expectations"))
    if n_expect:
        print(f"  persona expectations recorded: {n_expect}")


def _prepare(task, config, out):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        store, requests = prepare_run(task, config, root=out)
    for w in caught:
        print(f"note: {w.message}", file=sys.stderr)
    return store, requests


def refresh_dashboard(args, store: RunStore) -> None:
    """Rebuild <out>/dashboard.html (every evaluation run under <out>) once `store` has new evaluations."""
    if args.no_dashboard or not store.read("evaluations"):
        return
    try:
        out, data = write_dashboard(Path(args.out) / "dashboard.html", evaluations_root=args.out)
        print(f"  dashboard: {out}  ({len(data['evaluations'])} evaluations of {len(data['subjects'])} texts)")
    except Exception as e:      # the run itself is safe on disk; the page can be rebuilt by hand
        print(f"note: the dashboard was not rebuilt ({type(e).__name__}: {e}); "
              f"run `python -m evaluation.dashboard`", file=sys.stderr)


# ---- commands ---------------------------------------------------------------------------
def cmd_expectations(args) -> int:
    if not args.personas:
        print("error: give --personas", file=sys.stderr)
        return 2
    store = generate_expectations(
        get_task(args.task).expectations_task(), args.personas, provider=args.provider, model=args.model,
        persona_ids=args.persona, root=args.out, params={**_params(args)}, max_workers=args.workers,
        name=args.name, dataset=args.dataset, dry_run=args.dry_run)
    print_summary(store, summarize(store))
    print(f"\nUse with: python -m evaluation.evaluate run {args.task} --expectations {store.run_id} ...")
    return 0


def cmd_run(args) -> int:
    task_cls = get_task(args.task)
    if not args.subject_dataset and not args.subject_run:
        print("error: give --subject-dataset or --subject-run", file=sys.stderr)
        return 2
    if args.no_personas:
        args.personas = None
    if (args.personas and not args.expectations and not args.no_expectations and not args.dry_run
            and task_cls.expectations_task is not None):
        print("Stage 1: recording persona expectations (reuse them next time with --expectations) ...")
        stage1 = generate_expectations(
            task_cls.expectations_task(), args.personas, provider=args.provider, model=args.model,
            persona_ids=args.persona, root=args.out, params={"max_tokens": EXPECTATIONS_MAX_TOKENS},
            max_workers=args.workers, dataset=args.dataset, name=f"expectations for {args.name}" if args.name else None)
        args.expectations = stage1.run_id
        missing = len(stage1.read("personas")) - len(load_expectations(stage1))
        print(f"  expectations run {args.expectations}: {len(load_expectations(stage1))} personas"
              f"{f' ({missing} failed; rerun `resume {args.expectations}`)' if missing else ''}")
        if missing:
            return 1

    task = task_cls.from_args(args)
    if not task.subjects:
        available = Counter((s.template, s.model) for s in load_subjects(
            args.subjects_root, args.subject_run, args.subject_dataset, valid_only=not args.include_invalid))
        print("error: no subjects match" +
              (" --subject-template " + ", ".join(task.subject_templates) if task.subject_templates else "") +
              (" and" if task.subject_templates and task.subject_models else "") +
              (" --subject-model " + ", ".join(task.subject_models) if task.subject_models else "") +
              ". The selected runs hold: " + ("; ".join(f"{n} of {t} by {m}" for (t, m), n in available.most_common())
                                              or "no usable outputs"), file=sys.stderr)
        return 1
    n = args.n if args.n is not None else len(task.subjects)
    config = GenerationConfig(
        n=n, provider=args.provider, model=args.model, templates=[args.template] if args.template else [],
        params=_params(args), extra_body=args.extra_body, mode="dry_run" if args.dry_run else args.mode,
        seed=args.seed, validate=not args.no_validate, validation_retries=args.validation_retries,
        retry_mode=args.retry_mode, max_workers=args.workers, poll_interval=args.poll_interval, name=args.name,
        dataset=args.dataset, persona_set=args.personas, persona_sampling=args.persona_sampling,
        persona_ids=list(args.persona) if args.personas else [])
    store, requests = _prepare(task, config, args.out)
    print(f"Stage 2: prepared {len(requests)} evaluations of {min(n, len(task.subjects))} subjects "
          f"({len(task.subjects)} available) in {store.dir}\n  rubric: {task.rubric.id}   template: "
          f"{', '.join(config.templates)}   expectations: {task.expectations_run or 'none'}")
    if config.mode == "dry_run":
        print_summary(store, summarize(store))
        return 0
    if config.mode == "live":
        print_summary(store, run_live(store, task, requests))
        refresh_dashboard(args, store)
        return 0
    job = submit_batch(store, requests)
    print(f"Submitted batch {job.id}. Collect with:\n  python -m evaluation.evaluate collect {store.run_id}")
    if args.wait:
        print_summary(store, collect_batch(store, task, wait=True))
        refresh_dashboard(args, store)
    return 0


def cmd_collect(args) -> int:
    store = RunStore.open(args.run_id, args.out)
    result = collect_batch(store, task_of(store.run()), wait=args.wait)
    if "batch_id" in result:
        print(f"Batch {result['batch_id']} is {result['status']} {result['request_counts']}; "
              f"run again later or pass --wait.")
        return 1
    print_summary(store, result)
    refresh_dashboard(args, store)
    return 0


def cmd_resume(args) -> int:
    store = RunStore.open(args.run_id, args.out)
    if load_config(store).mode != "live":
        print("resume is for live runs; use `collect` for batch runs.")
        return 2
    if args.workers:
        store.update_run(config={**store.run()["config"], "max_workers": args.workers})
    print_summary(store, run_live(store, task_of(store.run())))
    refresh_dashboard(args, store)
    return 0


def cmd_status(args) -> int:
    store = RunStore.open(args.run_id, args.out)
    print_summary(store, summarize(store))
    for b in store.read("batches"):
        print(f"  batch {b['batch_id']} (round {b['round']}): {b['status']} {b['request_counts'] or ''}"
              f"{' collected' if b['collected'] else ''}")
    return 0


def cmd_list(args) -> int:
    runs = list_runs(args.out, args.dataset)
    if not runs:
        print(f"No runs under {args.out}")
    for r in runs:
        s = r["summary"] or {}
        n_eval = len(RunStore.open(r["run_id"], args.out).read("evaluations"))
        print(f"{r['run_id']}  {r['status']:<11} {r['task']:<24} {r['provider']}/{r['model'] or '-'}  "
              f"{s.get('n_finished', 0)}/{s.get('n_requests', '?')} finished"
              f"{f', {n_eval} evaluations' if n_eval else ''}  dataset={r.get('dataset') or '-'}"
              f"{'  ' + r['name'] if r['name'] else ''}")
    return 0


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return {"expectations": cmd_expectations, "run": cmd_run, "collect": cmd_collect,
            "resume": cmd_resume, "status": cmd_status, "list": cmd_list}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
