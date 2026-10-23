"""
Generate a dataset with an LLM from the command line: start, collect, resume and inspect
runs of the tasks of any application's task file (applications/<application>/tasks.yaml).
Results are written as tables under data/generations/<run_id>/ (see inference/README.md).

    python -m inference.generate run tarot_celtic_cross --n 20 --provider featherless --max-tokens 2000
    python -m inference.generate run tarot_celtic_cross --n 500 --provider openai --model gpt-4o-mini --mode batch
    python -m inference.generate run tarot_celtic_cross --n 20 --personas tarot_personas --template celtic_cross_v2
    python -m inference.generate run tarot_celtic_cross --n 5 --set "instantiation.state.cards.steps.fixed={Outcome: Death}"
    python -m inference.generate run --task-file drafts/tasks.yaml my_task --n 5 --dry-run
    python -m inference.generate collect run_tarot_celtic_cross_20261001_101500_ab12cd --wait
    python -m inference.generate resume run_tarot_celtic_cross_20261001_101500_ab12cd
    python -m inference.generate status run_tarot_celtic_cross_20261001_101500_ab12cd
    python -m inference.generate list [--dataset tarot-v1]

A run does what its task's config says. The options of `run` change that for the run: each is
an edit of the task (--n is execution.n, --model is model.name, ...), and --set makes any other.
The task is checked as edited, and the run records its config, with the edits. collect, resume
and status take only a run id: the run recorded its config.

From Python, run_task does the same:

    run_task("tarot_celtic_cross", {"execution.n": 5, "execution.mode": "dry_run"})

Grow one dataset over several sessions by giving each run the same --dataset label:

    python -m inference.generate run tarot_celtic_cross --n 500 --dataset tarot-v1 --provider featherless
    python -m inference.generate run tarot_celtic_cross --n 500 --dataset tarot-v1 --provider featherless   # later

`python inference/generate.py ...` works too.
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

if __package__ in (None, ""):  # allow `python inference/generate.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from typing import Any, Mapping, Optional  # noqa: E402

from applications import task_file_of, task_names, task_of  # noqa: E402
from inference.pipeline import collect_batch, list_runs, prepare_run, run_live, submit_batch, summarize  # noqa: E402
from inference.store import DEFAULT_ROOT, RunStore  # noqa: E402
from inference.task import InferenceTask  # noqa: E402
from inference.task_config import InferenceTaskConfig, TaskConfigError, parse_edit  # noqa: E402
from instantiation import InstantiationError, Personas  # noqa: E402
from spec_files import read_yaml  # noqa: E402


# ---- running a task, from Python -------------------------------------------------------
def find_config(task: str | Path | InferenceTaskConfig, edits: Optional[Mapping[str, Any]] = None) -> InferenceTaskConfig:
    """
    A task's config, edited: by name (from the applications' task files), by the path of a task
    file (its only task; or "path::name"), or a config (read again from its file, the edits added).
    """
    if isinstance(task, InferenceTaskConfig):
        return InferenceTaskConfig.from_file(task.source, task.name, {**task.edits, **(edits or {})})
    text = str(task)
    path, _, name = text.partition("::")
    if name or Path(path).suffix in (".yaml", ".yml") or Path(path).is_file():
        return InferenceTaskConfig.from_file(path, name or None, edits)
    return InferenceTaskConfig.from_file(task_file_of(text), text, edits)


def run_task(task: str | Path | InferenceTaskConfig, edits: Optional[Mapping[str, Any]] = None, *, root=DEFAULT_ROOT,
             wait: bool = False, progress: bool = True) -> RunStore:
    """
    Run a task (see find_config), with edits, e.g. {"execution.n": 5}. A live run runs to the end;
    a batch run is submitted, and collected too with wait=True; a dry run is prepared only.
    Returns the run's store.
    """
    runnable = InferenceTask.from_config(find_config(task, edits))
    for warning in runnable.run_warnings(root):
        warnings.warn(warning, stacklevel=2)
    store, requests = prepare_run(runnable, root=root)
    mode = runnable.config.execution.mode
    if mode == "live":
        run_live(store, runnable, requests, progress=progress)
    elif mode == "batch":
        submit_batch(store, requests)
        if wait:
            collect_batch(store, runnable, wait=True, progress=progress)
    return store


# ---- the command line ------------------------------------------------------------------


def _json_arg(text: str) -> dict:
    try:
        value = json.loads(text)
    except json.JSONDecodeError as e:
        raise argparse.ArgumentTypeError(f"not valid JSON: {e}")
    if not isinstance(value, dict):
        raise argparse.ArgumentTypeError("must be a JSON object, e.g. '{\"top_k\": 40}'")
    return value


def _edit_arg(text: str) -> tuple:
    try:
        return parse_edit(text)
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e))


# the options that edit one setting of the task: (option's dest, the setting)
EDITS = [
    ("n", "execution.n"), ("seed", "execution.seed"), ("mode", "execution.mode"), ("workers", "execution.workers"),
    ("retry_mode", "execution.retry_mode"), ("poll_interval", "execution.poll_interval"),
    ("name", "execution.name"), ("dataset", "execution.dataset"),
    ("provider", "model.provider"), ("model", "model.name"), ("max_tokens", "model.parameters.max_tokens"),
    ("temperature", "model.parameters.temperature"), ("top_p", "model.parameters.top_p"),
    ("extra_body", "model.extra_body"), ("validation_retries", "validation.retries"),
]


def edits_of(args) -> dict:
    """The task edits the command line asks for, {dotted.path: value}; --set last, so it has the last word."""
    edits = {setting: getattr(args, dest) for dest, setting in EDITS if getattr(args, dest) is not None}
    if args.dry_run:
        edits["execution.mode"] = "dry_run"
    if args.template:
        edits["prompt.templates"] = [{"template": t} for t in args.template]
    if args.personas:
        edits["instantiation.state.persona"] = {"personas": args.personas}
    if args.system is not None:
        edits["prompt.system"] = args.system
    if args.no_validate:
        edits["validation"] = None
    edits.update(dict(args.set))
    return edits


def add_run_arguments(g) -> None:
    """The options of `run`. Each changes a setting of the task file (default: as the file says)."""
    g.add_argument("task", nargs="?", help="a task of an application's tasks.yaml (listed below), or of --task-file")
    g.add_argument("--task-file", metavar="PATH", help="run a task of this task file (its only one, unless TASK names it)")
    g.add_argument("--set", type=_edit_arg, action="append", default=[], metavar="SETTING=VALUE",
                   help="change any setting of the task, by its dotted path; the value is read as YAML, e.g. "
                        "--set execution.n=20 --set 'model.parameters={max_tokens: 2000}'. Repeatable")
    g.add_argument("--n", type=int, help="number of inputs (instants); requests = n x templates (execution.n)")
    g.add_argument("--provider", help="llm provider name (model.provider)")
    g.add_argument("--model", help="model name (model.name; null = the provider's default from .env)")
    g.add_argument("--mode", choices=["live", "batch"],
                   help="live: concurrent calls now; batch: submit a provider batch, e.g. OpenAI (execution.mode)")
    g.add_argument("--wait", action="store_true", help="batch mode: wait for the batch and collect it")
    g.add_argument("--dry-run", action="store_true", help="write inputs/prompts/requests only; no API calls")
    g.add_argument("--name", help="label stored with the run (execution.name)")
    g.add_argument("--dataset", help="dataset this run belongs to; use the same label across sessions to grow "
                                     "one dataset from several runs (execution.dataset)")
    g.add_argument("--seed", type=int, help="seed for sampling the instants (execution.seed; default: chosen, recorded)")
    g.add_argument("--template", action="append", default=[], metavar="ID",
                   help="prompt template id (name_vN) instead of the task's; repeat to render every input with "
                        "each (a paired comparison). See `python -m prompts list`")
    g.add_argument("--personas", metavar="SET",
                   help="add a persona to every instant, from this persona set (name or path, e.g. tarot_personas): "
                        "the state entry `persona`, which templates taking `persona` receive")
    g.add_argument("--system", help="system prompt, replacing the templates' own (prompt.system)")
    g.add_argument("--max-tokens", type=int, help="model.parameters.max_tokens")
    g.add_argument("--temperature", type=float, help="model.parameters.temperature")
    g.add_argument("--top-p", type=float, help="model.parameters.top_p")
    g.add_argument("--extra-body", type=_json_arg, metavar="JSON",
                   help="provider-specific params, e.g. '{\"top_k\": 40, \"min_p\": 0.02}' (model.extra_body)")
    g.add_argument("--workers", type=int, help="live mode: concurrent requests (execution.workers)")
    g.add_argument("--no-validate", action="store_true",
                   help="skip validation, and therefore retries (validation: null)")
    g.add_argument("--validation-retries", type=int, help="regenerations per invalid output (validation.retries)")
    g.add_argument("--retry-mode", choices=["live", "batch"],
                   help="batch runs: regenerate invalid outputs live (fast) or as follow-up batches (cheaper, "
                        "slower) (execution.retry_mode)")
    g.add_argument("--poll-interval", type=float, help="batch status poll seconds (execution.poll_interval)")


def task_list() -> str:
    """The tasks of the applications' task files, with their descriptions, for `run --help`."""
    lines = ["tasks:"]
    for name, path in sorted(task_names().items()):
        description = " ".join(str((read_yaml(path)[name] or {}).get("description", "")).split())
        lines.append(f"  {name:<24} {description[:90]}")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="generate", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=str(DEFAULT_ROOT), help="root directory for runs (default: %(default)s)")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="start a new run of a task", epilog=task_list(),
                         description="Start a run of a task of a task file. Its options change the task's settings "
                                     "for this run.", formatter_class=argparse.RawDescriptionHelpFormatter)
    add_run_arguments(run)

    for name, help_ in [("collect", "collect a batch run's results"), ("resume", "finish an interrupted live run"),
                        ("status", "show a run's status")]:
        p = sub.add_parser(name, help=help_)
        p.add_argument("run_id")
        if name == "collect":
            p.add_argument("--wait", action="store_true", help="wait for the batch if it isn't finished")
    ls = sub.add_parser("list", help="list runs")
    ls.add_argument("--dataset", help="only runs of this dataset, with combined totals")
    return parser


def print_summary(store: RunStore, summary: dict) -> None:
    run = store.run()
    print(f"\nrun {store.run_id}  [{run['status']}]  {run['provider']}/{run['model'] or '(default model)'}"
          f"{'  dataset=' + run['dataset'] if run.get('dataset') else ''}")
    print(f"  tables: {store.dir}")
    for k, v in summary.items():
        if k != "by_template":
            print(f"  {k:>18}: {v}")
    for template, s in (summary.get("by_template") or {}).items():
        done = s["n_finished"]
        rate = f"{s['n_valid'] / done:.0%} valid, {s['n_valid_first_try'] / done:.0%} on the first try" if done else "-"
        print(f"  {template:>18}: {done}/{s['n_requests']} finished; {rate}")


def cmd_run(args) -> int:
    if args.task_file:
        which = f"{args.task_file}::{args.task}" if args.task else args.task_file
    elif args.task:
        which = args.task
    else:
        print("Name a task, or give --task-file.\n" + task_list(), file=sys.stderr)
        return 2
    try:
        task = InferenceTask.from_config(find_config(which, edits_of(args)))
    except (TaskConfigError, KeyError, ValueError) as e:
        print(f"error: {e.args[0] if isinstance(e, KeyError) else e}", file=sys.stderr)
        return 2
    config = task.config
    for warning in task.run_warnings(args.out):
        print(f"WARNING: {warning}", file=sys.stderr)
    state = config.instantiation.state if config.instantiation else {}
    people = {name: e.set for name, e in state.items() if isinstance(e, Personas)}
    for name in people:
        ignoring = [t.id for t in config.prompt.templates if name not in t.template.variables
                    and name not in t.bind.values()]
        if ignoring:
            print(f"note: no `{name}` in {ignoring}: those prompts will not mention the persona", file=sys.stderr)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            store, requests = prepare_run(task, root=args.out)
        except (InstantiationError, ValueError) as e:        # e.g. n against exhaustive sampling, a template
            print(f"error: {e}", file=sys.stderr)
            return 2
    for w in caught:
        print(f"note: {w.message}", file=sys.stderr)
    drafts = [t["template_id"] for t in store.read("prompt_templates") if t["status"] == "draft"]
    print(f"Prepared {len(requests)} requests in {store.dir}\n"
          f"  templates: {', '.join(t.id for t in config.prompt.templates)}"
          f"{'  personas: ' + ', '.join(people.values()) if people else ''}")
    if drafts:
        print(f"  note: {drafts} not locked yet; lock with `python -m prompts lock` once you keep data from them")
    mode = config.execution.mode
    if mode == "dry_run":
        print_summary(store, summarize(store))
        return 0
    if mode == "live":
        print_summary(store, run_live(store, task, requests))
        return 0
    job = submit_batch(store, requests)
    print(f"Submitted batch {job.id}. Collect with:\n  python -m inference.generate collect {store.run_id}")
    if args.wait:
        print_summary(store, collect_batch(store, task, wait=True))
    return 0


def cmd_collect(args) -> int:
    store = RunStore.open(args.run_id, args.out)
    result = collect_batch(store, task_of(store.run()), wait=args.wait)
    if "batch_id" in result:
        print(f"Batch {result['batch_id']} is {result['status']} {result['request_counts']}; "
              f"run again later or pass --wait.")
        return 1
    print_summary(store, result)
    return 0


def cmd_resume(args) -> int:
    store = RunStore.open(args.run_id, args.out)
    if store.run()["mode"] != "live":
        print("resume is for live runs; use `collect` for batch runs.")
        return 2
    print_summary(store, run_live(store, task_of(store.run())))
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
        print(f"No runs under {args.out}" + (f" for dataset {args.dataset!r}" if args.dataset else ""))
    totals = {"n_requests": 0, "n_finished": 0, "n_valid": 0, "n_invalid": 0, "total_tokens": 0}
    for r in runs:
        s = r["summary"] or {}
        for k in totals:
            totals[k] += s.get(k) or 0
        print(f"{r['run_id']}  {r['status']:<11} {r['task']:<19} {r['mode']:<7} "
              f"{r['provider']}/{r['model'] or '-'}  "
              f"{s.get('n_finished', 0)}/{s.get('n_requests', '?')} finished, {s.get('n_valid', 0)} valid"
              f"  dataset={r.get('dataset') or '-'}{'  ' + r['name'] if r['name'] else ''}")
    if args.dataset and runs:
        print(f"\ndataset {args.dataset!r}: {len(runs)} run(s), {totals['n_finished']}/{totals['n_requests']} "
              f"finished, {totals['n_valid']} valid, {totals['n_invalid']} invalid, "
              f"{totals['total_tokens']:,} tokens")
    return 0


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return {"run": cmd_run, "collect": cmd_collect, "resume": cmd_resume,
            "status": cmd_status, "list": cmd_list}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
