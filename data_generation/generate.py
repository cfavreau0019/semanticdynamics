"""
Generate a dataset with an LLM from the command line: start, collect, resume and inspect
runs of any generation task of any application (applications/<application>/). Results are
written as tables under data/generations/<run_id>/ (see data_generation/README.md).

    python -m data_generation.generate run tarot_celtic_cross --n 20 --provider featherless --max-tokens 2000
    python -m data_generation.generate run tarot_celtic_cross --n 500 --provider openai --model gpt-4o-mini --mode batch
    python -m data_generation.generate collect run_tarot_celtic_cross_20261001_101500_ab12cd --wait
    python -m data_generation.generate resume run_tarot_celtic_cross_20261001_101500_ab12cd
    python -m data_generation.generate status run_tarot_celtic_cross_20261001_101500_ab12cd
    python -m data_generation.generate list [--dataset tarot-v1]

`run --help` lists the tasks; `run <task> --help` lists the general options and the task's
own (for tarot_celtic_cross: --draw, --pin, --deck-config). collect, resume and status take
only a run id: the run recorded which task it belongs to.

Grow one dataset over several sessions by giving each run the same --dataset label:

    python -m data_generation.generate run tarot_celtic_cross --n 500 --dataset tarot-v1 --provider featherless
    python -m data_generation.generate run tarot_celtic_cross --n 500 --dataset tarot-v1 --provider featherless   # later

`python data_generation/generate.py ...` works too.
"""
import argparse
import inspect
import json
import sys
import warnings
from pathlib import Path

if __package__ in (None, ""):  # allow `python data_generation/generate.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from applications import get_task, task_of, tasks  # noqa: E402
from data_generation.pipeline import (  # noqa: E402
    GenerationConfig, collect_batch, list_runs, load_config, prepare_run, run_live, submit_batch, summarize,
)
from data_generation.store import DEFAULT_ROOT, RunStore  # noqa: E402
from personas import SAMPLING_MODES  # noqa: E402


def _json_arg(text: str) -> dict:
    try:
        value = json.loads(text)
    except json.JSONDecodeError as e:
        raise argparse.ArgumentTypeError(f"not valid JSON: {e}")
    if not isinstance(value, dict):
        raise argparse.ArgumentTypeError("must be a JSON object, e.g. '{\"top_k\": 40}'")
    return value


def task_summary(cls) -> str:
    """The first line of a task's docstring, for --help."""
    return (inspect.getdoc(cls) or "").partition("\n")[0]


def add_run_arguments(g, task_cls) -> None:
    """The options every generation run takes. The task adds its own after these (InferenceTask.add_arguments)."""
    default, with_personas = getattr(task_cls, "default_template", None), task_cls.persona_template
    templates = f"Default: {default}" + (f", or {with_personas} with --personas" if with_personas not in (None, default)
                                         else "") if default else "Default: the task's"
    g.add_argument("--n", type=int, required=True, help="number of inputs to sample (one request each, per template)")
    g.add_argument("--provider", default="featherless", help="llm provider name (default: %(default)s)")
    g.add_argument("--model", help="model name (default: the provider's default from .env)")
    g.add_argument("--mode", choices=["live", "batch"], default="live",
                   help="live: concurrent calls now; batch: submit a provider batch (OpenAI)")
    g.add_argument("--wait", action="store_true", help="batch mode: wait for the batch and collect it")
    g.add_argument("--dry-run", action="store_true", help="write inputs/prompts/requests only; no API calls")
    g.add_argument("--name", help="label stored with the run")
    g.add_argument("--dataset", help="dataset this run belongs to; use the same label across sessions to grow "
                                     "one dataset from several runs")
    # sampling / prompt
    g.add_argument("--seed", type=int, help="seed for sampling inputs and personas (default: random, recorded)")
    g.add_argument("--template", action="append", default=[], metavar="NAME",
                   help="prompt template from the prompts library: 'name_v2' pins a version, 'name' takes the "
                        "latest. Repeat to render every input with each template (n inputs x templates requests) "
                        f"for a paired comparison. {templates}. See `python -m prompts list`.")
    g.add_argument("--system", help="system prompt (overrides the template's)")
    g.add_argument("--personas", metavar="SET", help="persona set name or JSON path (e.g. tarot_personas); "
                                                     "each input is assigned one persona")
    g.add_argument("--persona-sampling", choices=SAMPLING_MODES, default="random",
                   help="random: seeded, with replacement; cycle: in order, every persona equally often "
                        "(default: %(default)s)")
    g.add_argument("--persona", action="append", default=[], metavar="ID",
                   help="restrict to this persona (id, number or full name); repeatable")
    # generation params
    g.add_argument("--max-tokens", type=int, default=2400, help="default: %(default)s")
    g.add_argument("--temperature", type=float)
    g.add_argument("--top-p", type=float)
    g.add_argument("--extra-body", type=_json_arg, default={}, metavar="JSON",
                   help="provider-specific params, e.g. '{\"top_k\": 40, \"min_p\": 0.02}'")
    g.add_argument("--workers", type=int, default=2, help="live mode: concurrent requests (default: %(default)s)")
    # validation
    g.add_argument("--no-validate", action="store_true", help="skip validation (and therefore retries)")
    g.add_argument("--validation-retries", type=int, default=2, help="regenerations per invalid output "
                                                                     "(default: %(default)s)")
    g.add_argument("--retry-mode", choices=["live", "batch"], default="live",
                   help="batch runs: regenerate invalid outputs live (fast) or as follow-up batches "
                        "(cheaper, slower) (default: %(default)s)")
    g.add_argument("--poll-interval", type=float, default=60.0, help="batch status poll seconds")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="generate", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=str(DEFAULT_ROOT), help="root directory for runs (default: %(default)s)")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="start a new run of a task")
    by_task = run.add_subparsers(dest="task", required=True, metavar="TASK")
    for name, task_cls in sorted(tasks().items()):
        if task_cls.cli != "generate":
            continue
        g = by_task.add_parser(name, help=task_summary(task_cls), description=inspect.getdoc(task_cls),
                               formatter_class=argparse.RawDescriptionHelpFormatter)
        add_run_arguments(g, task_cls)
        task_cls.add_arguments(g)

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
    task = get_task(args.task).from_args(args)
    params = {"max_tokens": args.max_tokens}
    if args.temperature is not None:
        params["temperature"] = args.temperature
    if args.top_p is not None:
        params["top_p"] = args.top_p
    config = GenerationConfig(
        n=args.n, provider=args.provider, model=args.model, templates=list(args.template), system=args.system,
        params=params, extra_body=args.extra_body, mode="dry_run" if args.dry_run else args.mode, seed=args.seed,
        validate=not args.no_validate, validation_retries=args.validation_retries, retry_mode=args.retry_mode,
        max_workers=args.workers, poll_interval=args.poll_interval, name=args.name, dataset=args.dataset,
        persona_set=args.personas, persona_sampling=args.persona_sampling, persona_ids=list(args.persona),
    )
    for warning in task.run_warnings(config, args.out):
        print(f"WARNING: {warning}", file=sys.stderr)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        store, requests = prepare_run(task, config, root=args.out)
    for w in caught:
        print(f"note: {w.message}", file=sys.stderr)
    drafts = [t["template_id"] for t in store.read("prompt_templates") if t["status"] == "draft"]
    print(f"Prepared {len(requests)} requests in {store.dir}\n"
          f"  templates: {', '.join(config.templates)}"
          f"{'  personas: ' + config.persona_set if config.persona_set else ''}")
    if drafts:
        print(f"  note: {drafts} not locked yet; lock with `python -m prompts lock` once you keep data from them")
    if config.mode == "dry_run":
        print_summary(store, summarize(store))
        return 0
    if config.mode == "live":
        print_summary(store, run_live(store, task, requests))
        return 0
    job = submit_batch(store, requests)
    print(f"Submitted batch {job.id}. Collect with:\n  python -m data_generation.generate collect {store.run_id}")
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
    if load_config(store).mode != "live":
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
