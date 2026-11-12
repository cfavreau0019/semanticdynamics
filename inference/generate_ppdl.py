"""
Generate a dataset by running a PDL program (IBM's Prompt Declaration Language) from the
command line, and store it as the same run tables as inference.generate, under
data/generations/<run_id>/ (see inference/README.md). inference.generate runs a task of a
tasks.yaml through the InferenceTask engine; this runs a PDL file, e.g.
applications/tarot/tarot_ppdl.yaml, through the PDL interpreter, which samples the input,
renders the prompt, calls the model, validates and retries. What is left here is to start one
execution of the file per input, in parallel, and to write what each returns.

    python -m inference.generate_ppdl run applications/tarot/tarot_ppdl.yaml --n 20 --workers 4
    python -m inference.generate_ppdl run applications/tarot/tarot_ppdl.yaml --n 5 --seed 7 --data max_attempts=1
    python -m inference.generate_ppdl status run_tarot_ppdl_20261112_101500_ab12cd
    python -m inference.generate_ppdl list [--dataset tarot-v1]

Run it from the repository root: the PDL file's Python blocks import the repo's packages.
The run's task is the PDL file's name (tarot_ppdl). There is no batch mode, dry run or resume:
an input whose execution fails writes nothing, and the run is then `partial`.

The model call is PDL's (through LiteLLM). For OpenAI it reads OPENAI_API_KEY; the repo's .env
names the key OPENAI_API_TOKEN, which is copied to it for this process (use_env_key).

What a PDL file must do to be run here. It is executed once per input, with the variables
`n` (the run's number of inputs), `seed` (the run's seed) and `index` (which input, from 0),
and any --data, and its result must be:

    seed        the run seed it used
    index       the input's position in the run
    input       the input itself: an instant, {entry: value}
    template    prompt template id, e.g. celtic_cross_v4 (looked up in the prompt library)
    messages    the exact message list sent to the model
    model       "<provider>/<model>" (LiteLLM's naming)
    parameters  the parameters sent with it
    attempts    every generation, in order, the last being the one kept; each
                {text, finish_reason, model, usage, validation}, with `usage` the provider's
                and `validation` a ValidationResult.to_dict() (or null)

From Python: run_ppdl("applications/tarot/tarot_ppdl.yaml", n=5).
"""
import argparse
import hashlib
import json
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

if __package__ in (None, ""):  # allow `python inference/generate_ppdl.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from typing import Any, Mapping, Optional  # noqa: E402

from dotenv import load_dotenv  # noqa: E402

from inference.generate import _edit_arg, cmd_list, cmd_status, print_summary  # noqa: E402
from inference.pipeline import _attempt_rows, summarize  # noqa: E402
from inference.schema import SCHEMA_VERSION  # noqa: E402
from inference.store import DEFAULT_ROOT, RunStore, code_version, new_id, utc_now  # noqa: E402
from inference.task import INSTANT, InferenceTask  # noqa: E402
from prompts import PromptLibrary, default_library  # noqa: E402


# ---- running the PDL file --------------------------------------------------------------
def use_env_key() -> None:
    """The repo's .env names the OpenAI key OPENAI_API_TOKEN; PDL's model call reads OPENAI_API_KEY."""
    load_dotenv()
    if "OPENAI_API_KEY" not in os.environ and "OPENAI_API_TOKEN" in os.environ:
        os.environ["OPENAI_API_KEY"] = os.environ["OPENAI_API_TOKEN"]


def execute(pdl_file: Path, scope: Mapping[str, Any]) -> dict:
    """One execution of a PDL file with these variables: {"result": ..., "trace": ..., ...}."""
    from pdl.pdl import exec_file

    # batch=1: no streaming, so a model block's modelResponse is the provider's whole response
    return exec_file(pdl_file, config={"yield_result": False, "yield_background": False, "batch": 1},
                     scope=dict(scope), output="all")


def write_trace(store: RunStore, index: int, trace) -> None:
    """The execution's PDL trace, kept beside the tables: <run>/pdl_traces/input_<index>.json."""
    from pdl.pdl_utils import write_trace as write

    folder = store.dir / "pdl_traces"
    folder.mkdir(exist_ok=True)
    write(folder / f"input_{index:04d}.json", trace)


# ---- from a PDL result to the run tables -----------------------------------------------
def _usage(usage: Optional[Mapping]) -> Optional[dict]:
    """A provider's usage as the responses table has it (cached tokens as llm/providers counts them)."""
    if not usage:
        return None
    details = usage.get("prompt_tokens_details") or {}
    return {"prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"), "cached_tokens": details.get("cached_tokens") or 0}


def reading_rows(store: RunStore, result: Mapping[str, Any], library: PromptLibrary) -> dict[str, list[dict]]:
    """
    The rows of one execution's result (see the module docstring), by table: its input, prompt,
    request and every attempt, as inference.pipeline writes them for a task file's run. The
    prompt template's row comes from the prompt library when it has the template.
    """
    now = utc_now()
    input_id, prompt_id, request_id = new_id("inp"), new_id("prm"), new_id("req")
    payload, messages = result["input"], result["messages"]
    provider, model = result["model"].split("/", 1) if "/" in result["model"] else ("litellm", result["model"])
    system = next((m["content"] for m in messages if m["role"] == "system"), None)
    user = next(m["content"] for m in reversed(messages) if m["role"] == "user")

    rows: dict[str, list[dict]] = {"prompt_templates": []}
    try:
        template = library.get(result["template"])
    except KeyError:
        template = None
    if template is not None:
        rows["prompt_templates"] = [{
            "template_sha256": template.sha256, "template_id": template.id, "name": template.name,
            "version": template.version, "system_template": template.system, "user_template": template.user,
            "required": list(template.required), "optional": list(template.optional),
            "description": template.description, "changes": template.changes, "based_on": template.based_on,
            "status": library.status(template),
        }]
        if template.render_from(payload).user != user:
            print(f"WARNING: input {result['index']}: the prompt sent is not {template.id} as the prompt library "
                  f"renders it", file=sys.stderr)

    rows["inputs"] = [{"input_id": input_id, "run_id": store.run_id, "input_index": result["index"],
                       "input_type": INSTANT, "payload": payload, "persona_id": None, "created_at": now}]
    rows["input_items"] = [{"input_id": input_id, "run_id": store.run_id, "item_index": k, **item}
                           for k, item in enumerate(InferenceTask().input_items(payload))]
    rows["prompts"] = [{
        "prompt_id": prompt_id, "input_id": input_id, "run_id": store.run_id, "template": result["template"],
        "template_sha256": template.sha256 if template is not None else None, "system": system,
        "user_text": user, "messages": messages,
        "prompt_sha256": hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest(),
        "created_at": now}]
    rows["requests"] = [{"request_id": request_id, "prompt_id": prompt_id, "run_id": store.run_id,
                         "provider": provider, "model": model, "params": dict(result["parameters"]),
                         "extra_body": {}, "mode": "live", "created_at": now}]
    for table in ("responses", "response_texts", "validations", "validation_issues"):
        rows[table] = []
    for k, attempt in enumerate(result["attempts"], start=1):
        made = _attempt_rows(store, request_id, k, text=attempt["text"], model=attempt["model"],
                             finish_reason=attempt["finish_reason"], error=None, usage=_usage(attempt["usage"]),
                             validation=attempt["validation"], batch_id=None)
        for table, r in made.items():
            rows[table] += r
    return rows


# ---- a run -----------------------------------------------------------------------------
def run_ppdl(pdl_file: str | Path, data: Optional[Mapping[str, Any]] = None, *, n: int = 1,
             seed: Optional[int] = None, workers: int = 2, name: Optional[str] = None,
             dataset: Optional[str] = None, root=DEFAULT_ROOT, progress: bool = True) -> RunStore:
    """
    Run a PDL file once per input, `workers` at a time, and write each result as it returns (in
    input order). `data` are more variables for the file. Returns the run's store; the run is
    `completed`, or `partial` when some executions failed (they are printed, and write nothing).
    """
    pdl_file = Path(pdl_file)
    use_env_key()
    seed = seed if seed is not None else random.SystemRandom().randrange(2**31)
    library = default_library()

    store = RunStore.create(root, task=pdl_file.stem)
    sha, dirty = code_version()
    now = utc_now()
    store.upsert("runs", [{
        "run_id": store.run_id, "name": name, "dataset": dataset, "task": pdl_file.stem, "mode": "live",
        "status": "running", "provider": None, "model": None,
        "config": {"ppdl": {"file": pdl_file.as_posix(),
                            "sha256": hashlib.sha256(pdl_file.read_bytes()).hexdigest(), "data": dict(data or {})},
                   "n": n, "seed": seed, "workers": workers},
        "code_version": sha, "code_dirty": dirty, "schema_version": SCHEMA_VERSION,
        "summary": None, "created_at": now, "updated_at": now,
    }])

    def one(index: int):
        try:
            return execute(pdl_file, {**(data or {}), "n": n, "seed": seed, "index": index})
        except Exception as e:  # one input's failure must not lose the others
            return e

    failed, templates = 0, set()
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for index, out in enumerate(pool.map(one, range(n))):
                if isinstance(out, Exception):
                    failed += 1
                    print(f"input {index} failed: {type(out).__name__}: {out}", file=sys.stderr)
                    continue
                rows = reading_rows(store, out["result"], library)
                for row in rows.pop("prompt_templates"):
                    if row["template_sha256"] not in templates:
                        templates.add(row["template_sha256"])
                        store.append("prompt_templates", [row])
                for table, table_rows in rows.items():
                    store.append(table, table_rows)
                write_trace(store, index, out["trace"])
                if store.run()["model"] is None:
                    request = rows["requests"][0]
                    store.update_run(provider=request["provider"], model=request["model"])
                if progress:
                    print(f"input {index + 1}/{n}: {len(rows['responses'])} attempt(s)")
    except BaseException:
        store.update_run(status="interrupted", summary=summarize(store))
        raise
    store.update_run(status="partial" if failed else "completed",
                     summary={**summarize(store), "n_inputs_failed": failed})
    return store


# ---- the command line ------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="generate_ppdl", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=str(DEFAULT_ROOT), help="root directory for runs (default: %(default)s)")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="start a new run of a PDL file")
    run.add_argument("pdl_file", help="the PDL program, e.g. applications/tarot/tarot_ppdl.yaml")
    run.add_argument("--n", type=int, default=1, help="number of inputs: one execution of the file each (default: 1)")
    run.add_argument("--seed", type=int, help="seed for sampling the inputs (default: chosen, recorded)")
    run.add_argument("--workers", type=int, default=2, help="executions at once (default: 2)")
    run.add_argument("--data", type=_edit_arg, action="append", default=[], metavar="NAME=VALUE",
                     help="a variable for the PDL file; the value is read as YAML, e.g. --data max_attempts=1. "
                          "Repeatable")
    run.add_argument("--name", help="label stored with the run")
    run.add_argument("--dataset", help="dataset this run belongs to; use the same label across sessions to grow "
                                       "one dataset from several runs")

    status = sub.add_parser("status", help="show a run's status")
    status.add_argument("run_id")
    ls = sub.add_parser("list", help="list runs")
    ls.add_argument("--dataset", help="only runs of this dataset, with combined totals")
    return parser


def cmd_run(args) -> int:
    if not Path(args.pdl_file).is_file():
        print(f"error: no PDL file {args.pdl_file}", file=sys.stderr)
        return 2
    store = run_ppdl(args.pdl_file, dict(args.data), n=args.n, seed=args.seed, workers=args.workers,
                     name=args.name, dataset=args.dataset, root=args.out)
    print_summary(store, store.run()["summary"])
    return 0


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return {"run": cmd_run, "status": cmd_status, "list": cmd_list}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
