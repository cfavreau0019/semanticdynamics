"""
Generic generation pipeline: inputs -> prompts -> requests -> responses (+ validation),
persisted as tables by RunStore. A run is a task (task.py) and its config (task_config.py):
the task makes the inputs and validates the responses, the config says everything else, and
every function here reads it as task.config. Prompt text comes from the `prompts` library and
optional personas from the `personas` package.

    task = InferenceTask.from_config(InferenceTaskConfig.from_file("applications/tarot/tasks.yaml"))
    store, requests = prepare_run(task)
    run_live(store, task, requests)

A task that gives its settings in Python rather than in a config (evaluation, until it has a
task file of its own) passes them to prepare_run as a GenerationConfig, which becomes its config.

Order of writes (so an interrupted run can be resumed or collected from the tables alone):
    runs -> prompt_templates, personas -> inputs, input_items -> prompts -> requests
                                                                (before any API call)
    responses, response_texts, validations, validation_issues   (as each request finishes,
                                                                 after validation + retries)
    batches                                                     (on submit / completion)
"""
import copy
import hashlib
import json
import random
import warnings
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Optional

from llm import BatchJob, ChatRequest, ChatResponse, generate_many, get_provider, retry_invalid, run_batch_results
from llm.batch import wait_for_batch
from inference.schema import SCHEMA_VERSION, TABLES
from inference.store import DEFAULT_ROOT, RunStore, code_version, new_id, utc_now
from inference.task import INPUT_KEY, InferenceTask, RunInput, RunTemplate
from inference.task_config import InferenceTaskConfig
from instantiation import Personas
from personas import SAMPLING_MODES, PersonaSet
from prompts import PromptLibrary, PromptTemplate, default_library


@dataclass
class GenerationConfig:
    """
    A run's settings, flat, as a task that gives them in Python states them (evaluation, until it
    has a task file of its own); prepare_run makes them the task's InferenceTaskConfig. Runs made
    before task configs stored their settings in this shape (InferenceTaskConfig.from_run reads them).
    """
    n: int                                    # number of inputs; requests = n x len(templates)
    provider: str
    model: Optional[str] = None
    # prompt library refs ("name_vN" pinned, "name" = latest); [] = the task's default.
    # With several, every input is rendered with each one (a paired comparison).
    templates: list[str] = field(default_factory=list)
    system: Optional[str] = None              # overrides the templates' system prompt
    params: dict[str, Any] = field(default_factory=dict)
    extra_body: dict[str, Any] = field(default_factory=dict)
    mode: str = "live"                        # live | batch | dry_run
    seed: Optional[int] = None
    validate: bool = True
    validation_retries: int = 2
    retry_mode: str = "live"                  # how batch runs regenerate invalid responses: live | batch
    max_workers: int = 4
    poll_interval: float = 60.0
    name: Optional[str] = None
    dataset: Optional[str] = None             # label grouping several runs into one dataset
    persona_set: Optional[str] = None         # persona set name or path; None = no personas
    persona_sampling: str = "random"          # random (seeded, with replacement) | cycle (balanced)
    persona_ids: list[str] = field(default_factory=list)   # restrict to these ids / numbers / names


# ---- preparing a run -------------------------------------------------------------------
def resolve_templates(task: InferenceTask, settings: GenerationConfig,
                      library: Optional[PromptLibrary] = None) -> list[PromptTemplate]:
    """The templates of a run given settings, pinned to exact versions (written back to settings.templates)."""
    library = library or default_library()
    refs = list(settings.templates)
    if not refs:
        refs = [task.persona_template if settings.persona_set and task.persona_template else task.default_template]
    templates = [library.get(r) for r in refs]
    if len({t.id for t in templates}) != len(templates):
        raise ValueError(f"The same template is listed twice: {[t.id for t in templates]}")
    if settings.persona_set:
        ignoring = [t.id for t in templates if not t.uses("persona")]
        if ignoring:
            warnings.warn(f"Personas were requested but {ignoring} take no `persona` variable; "
                          f"those prompts will not mention the persona.", stacklevel=3)
    settings.templates = [t.id for t in templates]
    return templates


def resolve_personas(config: InferenceTaskConfig) -> tuple[Optional[PersonaSet], str]:
    """The persona set of a task that computes its inputs in Python (its config's `persona` entry), and how to sample it."""
    entry = config.instantiation.state.get("persona") if config.instantiation else None
    if not isinstance(entry, Personas):
        return None, "random"
    mode = config.instantiation.method("persona")
    if mode not in SAMPLING_MODES:
        raise ValueError(f"persona sampling must be one of {SAMPLING_MODES}, not {mode!r}")
    people = PersonaSet.load(entry.file or entry.set)
    return (people.select(entry.select) if entry.select else people), mode


def inputs_from_hooks(task: InferenceTask, seed: int,
                      library: PromptLibrary) -> tuple[list[RunTemplate], list[RunInput], dict]:
    """
    A run's templates and inputs, for a task that computes its inputs in Python (sample_inputs,
    assign_personas, template_variables), with its config's templates and personas; and what to
    record of them in runs.config. InferenceTask.run_inputs uses it for such a task.
    """
    config = task.config
    templates = [RunTemplate(t, library.status(t)) for t in (library.get(ref.id) for ref in config.prompt.templates)]
    people, mode = resolve_personas(config)
    rng = random.Random(seed)
    payloads = task.sample_inputs(config.execution.n, rng)
    # personas are drawn after the inputs, so a seed gives the same inputs with or without them
    assigned = task.assign_personas(payloads, people, rng, mode) if people else [None] * len(payloads)
    if len(assigned) != len(payloads):
        raise ValueError("assign_personas must return one entry per payload")
    inputs = [RunInput(payload, task.template_variables(payload, persona), persona["id"] if persona else None,
                       [people.row(persona)] if persona else [])
              for payload, persona in zip(payloads, assigned)]
    return templates, inputs, {"persona_set_sha256": people.sha256 if people else None,
                               "task_options": task.options()}


def prepare_run(task: InferenceTask, settings: Optional[GenerationConfig] = None, root=DEFAULT_ROOT,
                library: Optional[PromptLibrary] = None) -> tuple[RunStore, list[ChatRequest]]:
    """
    Create the run and write runs, prompt_templates, personas, inputs, input_items, prompts and
    requests, as the task's config says (InferenceTask.from_config). A task that gives its settings
    in Python passes them as `settings`; they become its config. Each input is rendered with every
    template (or as prompt.sample says), and every prompt is rendered before anything is written.
    """
    library = library or default_library()
    if settings is not None:
        if settings.persona_ids and not settings.persona_set:
            raise ValueError("persona_ids given without a persona_set")
        resolve_templates(task, settings, library)
        if settings.seed is None:
            settings.seed = random.SystemRandom().randrange(2**31)
        task.config = InferenceTaskConfig.from_settings(task.name, settings)
    config = task.config
    if config is None:
        raise ValueError(f"{task.name} has no config: build it with InferenceTask.from_config, or give settings")
    x, m = config.execution, config.model
    seed = x.seed if x.seed is not None else random.SystemRandom().randrange(2**31)
    templates, run_inputs, extra = task.run_inputs(seed, library)
    # every prompt is rendered before anything is written: a template that cannot render leaves no run behind
    rendered = [[(templates[k], templates[k].render(inp.values))
                 for k in (inp.templates if inp.templates is not None else range(len(templates)))]
                for inp in run_inputs]

    store = RunStore.create(root, task=task.name)
    sha, dirty = code_version()
    now = utc_now()
    stored = {**asdict(settings), **extra} if settings is not None else \
        {"task_config": config.record(), "seed": seed, **extra}
    store.upsert("runs", [{
        "run_id": store.run_id, "name": x.name, "dataset": x.dataset, "task": task.name, "mode": x.mode,
        "status": "prepared", "provider": m.provider, "model": m.name, "config": stored,
        "code_version": sha, "code_dirty": dirty, "schema_version": SCHEMA_VERSION,
        "summary": None, "created_at": now, "updated_at": now,
    }])
    store.append("prompt_templates", [{
        "template_sha256": t.template.sha256, "template_id": t.template.id, "name": t.template.name,
        "version": t.template.version, "system_template": t.template.system, "user_template": t.template.user,
        "required": list(t.template.required), "optional": list(t.template.optional),
        "description": t.template.description, "changes": t.template.changes, "based_on": t.template.based_on,
        "status": t.status,
    } for t in templates])
    used = {row["persona_id"]: row for inp in run_inputs for row in inp.personas}
    if used:
        store.append("personas", list(used.values()))
    for table, rows in task.reference_rows().items():
        store.append(table, rows)

    request_mode = "batch" if x.mode == "batch" else "live"
    inputs, items, prompts, request_rows, requests = [], [], [], [], []
    for i, (inp, prompts_of_input) in enumerate(zip(run_inputs, rendered)):
        input_id = new_id("inp")
        payload, persona_id = inp.payload, inp.persona_id
        inputs.append({"input_id": input_id, "run_id": store.run_id, "input_index": i,
                       "input_type": task.input_type, "payload": payload, "persona_id": persona_id,
                       "created_at": now})
        items += [{"input_id": input_id, "run_id": store.run_id, "item_index": k, **item}
                  for k, item in enumerate(task.input_items(payload))]

        for run_template, rendered_prompt in prompts_of_input:
            template = run_template.template
            prompt_id, request_id = new_id("prm"), new_id("req")
            system = config.prompt.system if config.prompt.system is not None else rendered_prompt.system
            messages = ([{"role": "system", "content": system}] if system else []) + \
                       [{"role": "user", "content": rendered_prompt.user}]
            prompts.append({"prompt_id": prompt_id, "input_id": input_id, "run_id": store.run_id,
                            "template": template.id, "template_sha256": template.sha256, "system": system,
                            "user_text": rendered_prompt.user, "messages": messages,
                            "prompt_sha256": hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest(),
                            "created_at": now})
            request_rows.append({"request_id": request_id, "prompt_id": prompt_id, "run_id": store.run_id,
                                 "provider": m.provider, "model": m.name, "params": dict(m.parameters),
                                 "extra_body": dict(m.extra_body), "mode": request_mode, "created_at": now})
            requests.append(ChatRequest(
                messages=messages, model=m.name, params=dict(m.parameters), id=request_id,
                extra_body=dict(m.extra_body),
                metadata=_metadata(payload, input_id, prompt_id, persona_id, template.id)))

    store.append("inputs", inputs)
    store.append("input_items", items)
    store.append("prompts", prompts)
    store.append("requests", request_rows)
    return store, requests


def _metadata(payload, input_id, prompt_id, persona_id, template_id) -> dict[str, Any]:
    return {INPUT_KEY: payload, "input_id": input_id, "prompt_id": prompt_id, "persona_id": persona_id,
            "template": template_id}


def list_runs(root=DEFAULT_ROOT, dataset: Optional[str] = None) -> list[dict]:
    """The runs rows under `root` (oldest first), optionally only those of one dataset."""
    root = Path(root)
    if not root.exists():
        return []
    runs = [RunStore(p).run() for p in root.glob("run_*") if (p / "runs.jsonl").exists()]
    runs.sort(key=lambda r: (r["created_at"], r["run_id"]))
    return [r for r in runs if dataset is None or r.get("dataset") == dataset]


def load_config(store: RunStore) -> GenerationConfig:
    """
    The GenerationConfig of a run that stored its settings flat (a task that gives them in Python,
    or any run before task configs, of any schema version). A run's config, whatever its shape:
    InferenceTaskConfig.from_run(store.run()).
    """
    cfg = dict(store.run()["config"])
    if "task_config" in cfg:
        raise ValueError(f"{store.run_id} recorded an InferenceTaskConfig: read it with InferenceTaskConfig.from_run")
    if "template" in cfg:                       # schema version 1: a single template name
        template = cfg.pop("template")
        cfg.setdefault("templates", [template] if template else [])
    known = {f.name for f in fields(GenerationConfig)}
    return GenerationConfig(**{k: v for k, v in cfg.items() if k in known})


def rebuild_requests(store: RunStore) -> list[ChatRequest]:
    """Reconstruct the ChatRequests of a run from its requests/prompts/inputs tables."""
    prompts = {p["prompt_id"]: p for p in store.read("prompts")}
    inputs = {i["input_id"]: i for i in store.read("inputs")}
    out = []
    for r in store.read("requests"):
        p = prompts[r["prompt_id"]]
        inp = inputs[p["input_id"]]
        out.append(ChatRequest(
            messages=p["messages"], model=r["model"], params=dict(r["params"]), id=r["request_id"],
            extra_body=dict(r["extra_body"]),
            metadata=_metadata(inp["payload"], p["input_id"], p["prompt_id"], inp.get("persona_id"), p["template"])))
    return out


# ---- writing responses -----------------------------------------------------------------
def _attempt_rows(store: RunStore, request_id: str, attempt: int, *, text, model, finish_reason,
                  error, usage, validation, batch_id) -> dict[str, list[dict]]:
    response_id = f"{request_id}-a{attempt}"
    usage = usage or {}
    rows = {
        "responses": [{"response_id": response_id, "request_id": request_id, "run_id": store.run_id,
                       "attempt": attempt, "batch_id": batch_id, "model": model,
                       "finish_reason": finish_reason, "error": error,
                       "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens"),
                       "total_tokens": usage.get("total_tokens"), "cached_tokens": usage.get("cached_tokens"),
                       "received_at": utc_now()}],
        "response_texts": [{"response_id": response_id, "run_id": store.run_id, "text": text,
                            "n_chars": len(text) if text is not None else None}],
        "validations": [], "validation_issues": [],
    }
    if validation is not None:
        rows["validations"].append({
            "response_id": response_id, "run_id": store.run_id, "validator": validation.get("validator"),
            "passed": validation.get("passed"), "n_errors": validation.get("n_errors"),
            "n_warnings": validation.get("n_warnings"), "validator_error": validation.get("error"),
            "details": {name: c.get("details") for name, c in (validation.get("checks") or {}).items()} or None,
        })
        rows["validation_issues"] += [{
            "response_id": response_id, "run_id": store.run_id, "issue_index": k, "check_name": i.get("check"),
            "code": i.get("code"), "severity": i.get("severity"), "label": i.get("label"),
            "message": i.get("message"), "data": i.get("data"),
        } for k, i in enumerate(validation.get("issues") or [])]
    return rows


def write_response(store: RunStore, response: ChatResponse, previous_attempts: int = 0,
                   task: Optional[InferenceTask] = None) -> None:
    """
    Write every attempt of a finished request: failed attempts first, the returned response
    last, numbered previous_attempts+1 .. previous_attempts+response.attempts (so a request
    resumed after an API error continues its numbering, and its highest attempt stays final).
    """
    tables: dict[str, list[dict]] = {"responses": [], "response_texts": [], "validations": [], "validation_issues": []}
    returned = {"text": response.text, "model": response.model, "finish_reason": response.finish_reason,
                "error": response.error, "usage": asdict(response.usage) if response.usage else None,
                "validation": response.validation, "batch_id": response.batch_id}
    for k, a in enumerate([*response.failed_attempts, returned], start=previous_attempts + 1):
        rows = _attempt_rows(store, response.request_id, k, text=a.get("text"), model=a.get("model"),
                             finish_reason=a.get("finish_reason"), error=a.get("error"), usage=a.get("usage"),
                             validation=a.get("validation"), batch_id=a.get("batch_id"))
        for table, r in rows.items():
            tables[table] += r
    for table in ("responses", "response_texts", "validations", "validation_issues"):
        store.append(table, tables[table])
    if task is not None and response.ok:
        returned_id = f"{response.request_id}-a{previous_attempts + response.attempts}"
        try:
            derived = task.derived_rows(response, returned_id)
        except Exception as e:  # the response itself is already saved; don't lose the run over a parser bug
            warnings.warn(f"derived_rows failed for {returned_id}: {type(e).__name__}: {e}")
            derived = {}
        for table, rows in derived.items():  # the task needn't know the run id
            has_run_id = "run_id" in TABLES[table].column_names
            store.append(table, [{**r, "run_id": store.run_id} if has_run_id else r for r in rows])


def final_responses(store: RunStore) -> dict[str, dict]:
    """request_id -> its highest-attempt responses row (same as the final_responses SQL view)."""
    latest: dict[str, dict] = {}
    for r in store.read("responses"):
        if r["request_id"] not in latest or r["attempt"] > latest[r["request_id"]]["attempt"]:
            latest[r["request_id"]] = r
    return latest


def finished_request_ids(store: RunStore) -> set[str]:
    """Requests whose final response succeeded at the API level (errors are retried on resume)."""
    return {rid for rid, r in final_responses(store).items() if r["error"] is None}


def _attempt_offsets(store: RunStore) -> dict[str, int]:
    return {rid: r["attempt"] for rid, r in final_responses(store).items()}


# ---- running ---------------------------------------------------------------------------
def _config(store: RunStore, task: InferenceTask) -> InferenceTaskConfig:
    """The task's config; for a task rebuilt without one, what its run recorded."""
    return task.config if task.config is not None else InferenceTaskConfig.from_run(store.run())


def _provider_for(store: RunStore, provider: str):
    llm = get_provider(provider)
    if hasattr(llm, "batch_dir"):  # keep this run's batch files inside its directory
        llm = copy.copy(llm)
        llm.batch_dir = store.batch_files_dir
    return llm


def _validate_fn(task: InferenceTask, config: InferenceTaskConfig):
    if config.validation is None:
        return None
    validator = task.validator()
    if validator is None:
        return None
    from validation import response_validator

    def expected(response, request):
        return task.validation_expected(response.metadata or (request.metadata if request is not None else {}))

    return response_validator(validator, expected=expected)


def run_live(store: RunStore, task: InferenceTask, requests: Optional[list[ChatRequest]] = None,
             progress: bool = True) -> dict:
    """
    Generate every request without a successful final response, so it also resumes a run:
    requests never attempted, interrupted, or whose last attempt was an API error.
    """
    config = _config(store, task)
    requests = requests if requests is not None else rebuild_requests(store)
    done = finished_request_ids(store)
    offsets = _attempt_offsets(store)
    pending = [r for r in requests if r.id not in done]
    if progress and (done or offsets):
        n_err = sum(1 for r in pending if r.id in offsets)
        print(f"Resuming: {len(done)} finished; {len(pending)} to go ({n_err} retrying after API errors)")
    store.update_run(status="running")
    try:
        generate_many(pending, provider=_provider_for(store, config.model.provider),
                      max_workers=config.execution.workers, validate=_validate_fn(task, config),
                      validation_retries=config.validation.retries if config.validation else 0,
                      on_result=lambda r: write_response(store, r, offsets.get(r.request_id, 0), task),
                      progress=progress)
    except BaseException:
        store.update_run(status="interrupted", summary=summarize(store))
        raise
    return finalize(store)


def record_batch(store: RunStore, job: BatchJob, round_: int, n_requests: Optional[int] = None,
                 collected: Optional[bool] = None) -> None:
    prev = {b["batch_id"]: b for b in store.read("batches")}.get(job.id, {})
    total = (job.request_counts or {}).get("total") or 0
    now = utc_now()
    store.upsert("batches", [{
        "batch_id": job.id, "run_id": store.run_id, "provider": job.provider, "round": round_,
        "endpoint": job.endpoint, "status": job.status, "raw_status": job.raw_status,
        "n_requests": n_requests or prev.get("n_requests") or total, "request_counts": job.request_counts or None,
        "input_file": job.input_file or prev.get("input_file"),
        "collected": collected if collected is not None else prev.get("collected", False),
        "created_at": prev.get("created_at", now), "updated_at": now,
    }])


def submit_batch(store: RunStore, requests: Optional[list[ChatRequest]] = None) -> BatchJob:
    requests = requests if requests is not None else rebuild_requests(store)
    llm = _provider_for(store, store.run()["provider"])
    job = llm.submit_batch(requests, metadata={"run_id": store.run_id})
    record_batch(store, job, 0, n_requests=len(requests))
    store.update_run(status="submitted")
    return job


def collect_batch(store: RunStore, task: InferenceTask, wait: bool = False, progress: bool = True) -> dict:
    """
    Collect the run's initial batch: fetch results, validate, regenerate invalid ones
    (execution.retry_mode: live or follow-up batches), then write everything. Returns the run
    summary, or the batch's current status if it is still running and wait=False.
    """
    config = _config(store, task)
    x = config.execution
    pending = [b for b in store.read("batches") if b["round"] == 0 and not b["collected"]]
    if not pending:
        raise RuntimeError(f"{store.run_id} has no uncollected batch")
    llm = _provider_for(store, config.model.provider)
    job = llm.get_batch(pending[0]["batch_id"])
    job.input_file = pending[0]["input_file"]
    record_batch(store, job, 0)
    if not job.done:
        if not wait:
            return {"batch_id": job.id, "status": job.status, "request_counts": job.request_counts}
        job = wait_for_batch(job, llm, poll_interval=x.poll_interval, verbose=progress)
        record_batch(store, job, 0)

    requests = [r for r in rebuild_requests(store) if r.id not in finished_request_ids(store)]
    validate = _validate_fn(task, config)
    responses = run_batch_results(job, llm, requests, validate=validate)
    if validate is not None and config.validation.retries > 0:
        responses = retry_invalid(responses, requests, validate, llm, retries=config.validation.retries,
                                  mode=x.retry_mode, max_workers=x.workers,
                                  poll_interval=x.poll_interval, progress=progress,
                                  on_batch=lambda j, rnd: record_batch(store, j, rnd))
    offsets = _attempt_offsets(store)
    for r in responses:
        write_response(store, r, offsets.get(r.request_id, 0), task)
    for b in store.read("batches"):
        record_batch(store, BatchJob(b["batch_id"], b["provider"], b["status"], b["raw_status"], b["endpoint"],
                                     request_counts=b["request_counts"] or {}, input_file=b["input_file"]),
                     b["round"], collected=True)
    return finalize(store)


# ---- summaries -------------------------------------------------------------------------
def summarize(store: RunStore) -> dict:
    """Counts over each request's final response; token totals over every attempt (= cost)."""
    responses = store.read("responses")
    validations = {v["response_id"]: v for v in store.read("validations")}
    final = [r for r in final_responses(store).values() if r["error"] is None]
    passed = [validations.get(r["response_id"], {}).get("passed") for r in final]
    tokens = lambda key: sum(r.get(key) or 0 for r in responses)    # .get: columns added later
    requests = store.read("requests")
    n_requests = len(requests)

    # per prompt template: how the final responses fared, and how often the first attempt was already valid
    template_of_prompt = {p["prompt_id"]: p["template"] for p in store.read("prompts")}
    template_of_request = {r["request_id"]: template_of_prompt[r["prompt_id"]] for r in requests}
    by_template: dict[str, dict[str, int]] = {}
    for request_id, template in template_of_request.items():
        by_template.setdefault(template, {"n_requests": 0, "n_finished": 0, "n_valid": 0, "n_invalid": 0,
                                          "n_valid_first_try": 0})["n_requests"] += 1
    for r, ok in zip(final, passed):
        stats = by_template[template_of_request[r["request_id"]]]
        stats["n_finished"] += 1
        stats["n_valid"] += ok is True
        stats["n_invalid"] += ok is False
    for r in responses:
        if r["attempt"] == 1 and validations.get(r["response_id"], {}).get("passed") is True:
            by_template[template_of_request[r["request_id"]]]["n_valid_first_try"] += 1

    return {
        "by_template": by_template,
        "n_requests": n_requests,
        "n_finished": len(final),
        "n_api_errors": len(final_responses(store)) - len(final),
        "n_not_attempted": n_requests - len(final_responses(store)),
        "n_valid": sum(p is True for p in passed),
        "n_invalid": sum(p is False for p in passed),
        "n_unvalidated": sum(p is None for p in passed),
        "n_regenerated": sum(r["attempt"] > 1 for r in final),
        "n_attempts": len(responses),
        "prompt_tokens": tokens("prompt_tokens"),
        "completion_tokens": tokens("completion_tokens"),
        "total_tokens": tokens("total_tokens"),
        "cached_tokens": tokens("cached_tokens"),
    }


def finalize(store: RunStore) -> dict:
    """completed = every request has a successful final response; otherwise partial (resume/collect again)."""
    summary = summarize(store)
    status = "completed" if summary["n_finished"] == summary["n_requests"] else "partial"
    store.update_run(status=status, summary=summary)
    return summary
