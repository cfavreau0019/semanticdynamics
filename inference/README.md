# inference

Runs inference tasks: it samples a task's inputs, renders its prompts, calls the model (through the `llm` package), validates and retries, and stores each run as a set of relational tables, one JSONL file per table. The files are designed to load directly into Postgres (or DuckDB). Runs are kept under `data/generations/`.

```
inference/
  schema.py          table definitions: validate JSONL rows and generate schema.sql
  schema.sql         Postgres DDL (generated; don't edit by hand)
  store.py           RunStore: data/generations/<run_id>/<table>.jsonl
  task_config.py     InferenceTaskConfig: a task's whole config, from a task file (edited), code or a run;
                     `python -m inference.task_config check`
  task.py            InferenceTask: a task built from its config (inputs, template variables, validation)
  pipeline.py        prepare_run, run_live, submit_batch, collect_batch: each reads task.config
  generate.py        command-line script, and run_task(): runs the tasks of every application's task file
```

The tasks themselves are described by their application's task file, `applications/<application>/tasks.yaml` (the format: `applications/template/inference_task_reference.yaml`): what a run's inputs are (the instantiation), the prompt templates, the model, and validation. Python of the application's own is optional; `applications/tarot/code/readings.py` adds the tarot draw's long form.

## Generating from the command line

`generate.py` runs any task of a task file, named after `run`. `tarot_celtic_cross` is the tarot readings task; `run --help` lists the others. A run does what the task file says; the options change that for the run.

```bash
# live: concurrent calls, validated and retried, written as they finish
python -m inference.generate run tarot_celtic_cross --n 20 --provider featherless --max-tokens 2000

# options
python -m inference.generate run tarot_celtic_cross --n 50 --provider featherless --temperature 0.9 \
    --extra-body '{"top_k": 40, "min_p": 0.02}' --seed 42 --name death-outcome \
    --set instantiation.state.cards.steps.replacement=false \
    --set "instantiation.state.cards.steps.fixed={Outcome: Death reversed}"

# batch (OpenAI): submit now, collect later (--wait blocks until done)
python -m inference.generate run tarot_celtic_cross --n 500 --provider openai --model gpt-4o-mini --mode batch
python -m inference.generate collect <run_id> [--wait]

python -m inference.generate resume <run_id>    # live: finish interrupted / API-errored requests
python -m inference.generate status <run_id>
python -m inference.generate list
python -m inference.generate run tarot_celtic_cross --n 5 --dry-run   # write inputs/prompts/requests only
python -m inference.generate run --task-file drafts/tasks.yaml my_task --n 5 --dry-run   # a task of any file
```

From Python, `run_task` does the same, with the edits as a mapping:

```python
from inference.generate import run_task
store = run_task("tarot_celtic_cross", {"execution.n": 5, "execution.mode": "dry_run"})
```

### Growing a dataset over several sessions

Each `run` call is its own run, in its own folder, with its own recorded settings, seed and git commit. To build one dataset across sessions, give every run the same `--dataset` label:

```bash
python -m inference.generate run tarot_celtic_cross --n 500 --dataset tarot-v1 --provider featherless
# ... days later ...
python -m inference.generate run tarot_celtic_cross --n 500 --dataset tarot-v1 --provider featherless
python -m inference.generate list --dataset tarot-v1      # runs + combined totals
```

- The label is stored once, in `runs.dataset`. Every other table reaches it through `run_id`.
- Without `--seed`, each run gets a fresh seed (recorded in `runs.config`), so sessions draw different cards. Passing a `--seed` already used in the same dataset with the same instantiation would repeat those draws, so the script warns.
- A run stopped partway is finished with `resume <run_id>`; that adds to the same run's files.

`python -m inference.generate run --help` lists every option and the tasks. `collect`, `resume` and `status` take only a run id, because a run records its config (`runs.config.task_config`, with the chosen seed beside it): they rebuild the task from that record, not from the task file, which may have changed since.

**Every option of `run` is an edit of the task**: `--n` sets `execution.n`, `--model` sets `model.name`, `--max-tokens` sets `model.parameters.max_tokens`, and so on (`run --help` names each one's setting). `--set SETTING=VALUE` edits any other setting by its dotted path; the value is read as YAML, so `false`, `20`, `[a, b]` and `{k: v}` work. The task is checked as edited, exactly as a task file is, and the run records it with the list of edits. Options not given leave the file's settings alone. The main ones:

| Option | Edits | Notes |
|---|---|---|
| `--n` | `execution.n` | Number of inputs (instants). Requests = n x templates. With given instants, or exhaustive sampling, n is theirs |
| `--template ID` | `prompt.templates` | Templates from the [prompts library](../prompts/README.md), pinned (`name_vN`), instead of the task's. Repeat the flag to render every draw with each template |
| `--personas SET` | `instantiation.state.persona` | Adds a persona to every instant from a [persona set](../personas/README.md), e.g. `tarot_personas`. Templates that take `persona` receive it |
| `--set` | any | e.g. `instantiation.state.persona.sample=cycle`, `instantiation.state.persona.select=[7]`, `instantiation.state.cards.steps.replacement=false` (each card at most once, like a real deck), `validation.retries=0` |
| `--validation-retries` | `validation.retries` | Regenerations per reading that fails validation |
| `--retry-mode` | `execution.retry_mode` | Batch runs only: regenerate invalid readings live (fast) or as follow-up batches (cheaper, slower) |
| `--system` | `prompt.system` | A system prompt replacing the templates' own |
| `--no-validate` | `validation` | Sets it to null: responses are not validated (or retried) |

### Prompts, personas and comparing prompt versions

Prompt text lives in the `prompts` package as versioned template files. A template takes its variables from the instant: the entry of the same name (`cards`, `persona`), or another one named by the template's `bind` in the task file.

```bash
python -m prompts list                                                     # available templates and their status
python -m inference.generate run tarot_celtic_cross --n 200 --personas tarot_personas --template celtic_cross_v2
python -m inference.generate run tarot_celtic_cross --n 50 --template celtic_cross_v1 --template celtic_cross_v2
```

With several templates, `--n` is the number of card draws and each draw is sent once per template, so the comparison is paired. The summary and `status` then report, for each template, the valid rate and the first-try valid rate.

## Tables

```
personas ──< inputs                     prompt_templates ──< prompts          (reference tables)

runs ─┬─< inputs ─┬─< input_items
      │           └─< prompts ──< requests ──< responses ──┬── response_texts   (1:1)
      │                                          │          ├── validations      (1:1)
      └─< batches >──────────────────────────────┘          └─< validation_issues
```

| Table | One row per | Key columns |
|---|---|---|
| `runs` | generation run | task (the inference task's name), dataset (groups runs), config (JSONB: `task_config`, the task's config as run, with its hashes and edits, and `seed`; or, for a task giving its settings in Python (evaluation) and for runs before task configs, those flat settings: templates, params, seed, retries, and `task_options`, the class's own), code_version (git sha) + code_dirty, schema_version, status, summary |
| `personas` | persona used by the run | persona_set, persona_number, name, payload (JSONB), content_sha256 |
| `inputs` | structured input (an instant) | payload (JSONB `{cards: {position: card}, persona: <id>}`; before version 6, the draw itself), input_type (`instant`), input_index, persona_id |
| `input_items` | element of an input | label, value, entity, qualifier (position, `"Death reversed"`, `"Death"`, `"reversed"`) |
| `prompt_templates` | template version used by the run | template_id, name, version, system_template, user_template, required, optional, changes, status |
| `prompts` | rendered prompt | template, template_sha256, system, user_text, messages (JSONB), prompt_sha256 |
| `requests` | API call spec | provider, model, params, extra_body, mode |
| `responses` | **generation attempt** | attempt, batch_id, model, finish_reason, error, token counts |
| `response_texts` | attempt | text, n_chars |
| `validations` | attempt | validator, passed, n_errors, n_warnings, details (JSONB) |
| `validation_issues` | issue | check_name, code, severity, label (position), message, data |
| `batches` | batch job (incl. retry rounds) | round, status, request_counts, input_file, collected |

**Design rules**
- **Every attempt is a row.** That includes validation retries and requests resumed after API errors. Attempt numbers keep counting across resumes, and **a request's final response is its highest attempt.** In SQL that's the `final_responses` view; in Python, `pipeline.final_responses(store)`.
  - Filter to final responses to analyse outputs.
  - Use all rows for cost (token totals) and failure analysis.
- **Tables are written before any API call.** Runs, inputs, prompts and requests go to disk first, so an interrupted run can be resumed and a batch collected from the tables alone.
- **Two kinds of table:**
  - *Append-only* (most tables) are only ever appended to.
  - *Mutable* (`runs`, `batches`, both small) are rewritten atomically when a status changes. Load them with upsert-by-primary-key.
- **Reference tables** (`prompt_templates`, `personas`) describe things shared between runs, so they have no `run_id`. Each run's file holds the rows that run used, and the same row appears in every run that used it. Load them with `ON CONFLICT DO NOTHING`, or de-duplicate on the primary key.
- **Schema versions.** Version 2 added the two reference tables, `inputs.persona_id` and `prompts.template_sha256`. The new columns are nullable, so version 1 runs still load, resume and summarise. Version 3 added the evaluation tables (`rubrics`, `persona_expectations`, `evaluations`, `evaluation_item_scores`, `evaluation_red_flags`), which are written by the [evaluation](../evaluation/README.md) package. Version 4 added `responses.cached_tokens`. Version 5 renamed `runs.application` to `runs.task` and `config.application_options` to `config.task_options`, because "application" now means a set of tasks such as tarot. Version 6 runs record their config: each input's payload is the instant (`input_type` `instant`), and `config` holds the task's config (`task_config`) and its seed. There are no new columns; `InferenceTaskConfig.from_run` reads a run's config in either shape, and `evaluation.subjects` reads inputs of both shapes.
  - Earlier runs keep their files as they are. `RunStore.run()` returns every run's row in the current shape, so read rows through it; a run's row is rewritten in that shape the next time the run is resumed or collected. `schema_version` stays the version the run was created with.
  - When querying files of several versions together in DuckDB, pass `union_by_name=true` to `read_json`, and read the task as `coalesce(task, application)`.
  - A Postgres database created from an earlier `schema.sql` needs `ALTER TABLE runs RENAME COLUMN application TO task;`.
- **`schema.py` is the single source of truth.** Every row is checked against it before writing, and `schema.sql` is generated from it; a test fails if the two drift apart. After changing a table, bump `SCHEMA_VERSION` and regenerate: `python -m inference.schema > inference/schema.sql`.
- **Run ids name the task.** A run id is `run_<task>_<YYYYMMDD>_<HHMMSS>_<6 hex>` (UTC), e.g. `run_tarot_celtic_cross_20261005_142210_3fa9c1`, so run folders group by task and sort by start time within it. Runs created before this naming have no task part (`run_20261001_…`). `list` always sorts by each run's recorded start time.
- **Ids are prefixed text** (`run_…`, `inp_…`, `prm_…`, `req_…`, `<request_id>-a<attempt>`), and every table except the reference tables carries `run_id`, so runs can be combined, deleted or partitioned as units.
- **JSON columns (JSONB)** hold nested JSON, not strings. Timestamps are ISO-8601 UTC.
- **Batch files for a run** (provider input, output and error JSONL) are kept in `<run_dir>/batch_files/`.

## Querying and moving to a database

**DuckDB** queries the files directly with no server (`pip install duckdb`). This is the easiest option for analysis at small to medium scale:

```sql
SELECT i.value AS outcome_card, avg(v.passed::INT) AS valid_rate, count(*) AS n
FROM read_json('data/generations/*/responses.jsonl') r
JOIN read_json('data/generations/*/validations.jsonl') v USING (response_id)
JOIN read_json('data/generations/*/requests.jsonl') q USING (request_id)
JOIN read_json('data/generations/*/prompts.jsonl') p USING (prompt_id)
JOIN read_json('data/generations/*/input_items.jsonl') i ON i.input_id = p.input_id AND i.label = 'Outcome'
GROUP BY 1 ORDER BY valid_rate;
```

To restrict any query to one dataset, join `runs` and filter on the label:

```sql
SELECT count(*) AS readings
FROM read_json('data/generations/*/responses.jsonl') r
JOIN read_json('data/generations/*/runs.jsonl') ru USING (run_id)
WHERE ru.dataset = 'tarot-v1';
```

A table's file only exists once that table has rows. If *no* run has, say, `batches.jsonl`, the glob finds nothing and DuckDB raises an error instead of returning zero rows.

**Postgres** is the recommended destination once data outgrows files or needs sharing:
- JSONB for the flexible columns;
- enforced keys;
- pgvector, so embeddings can later sit next to the texts.

To set it up, `psql -f inference/schema.sql` creates the tables and the `final_responses` view. The simplest loader is DuckDB's Postgres extension:

```sql
ATTACH 'dbname=semanticdynamics' AS pg (TYPE postgres);
INSERT INTO pg.responses SELECT * FROM read_json('data/generations/<run_id>/responses.jsonl');
```

Load tables in the order they appear in `schema.sql` so foreign keys resolve.

**Large binary data** (hidden states, and embeddings before pgvector) should stay in files such as `.npy` or safetensors, with the tables storing a path to each file.

## Adding another task

Describe it in the application's `tasks.yaml` (see `applications/template/inference_task_reference.yaml`, and check it with `python -m inference.task_config check`). `generate.py` then offers it as `run <name>`, without being told about it, and `run_task("<name>")` runs it from Python.

A run's task is built from its config, `InferenceTask.from_config(config)`, and every pipeline function reads the config as `task.config`. Python is needed only for what a file cannot say. Name a `task.InferenceTask` subclass as the task's `implemented_by`; the pipeline asks it for
- `input_items(payload)`: long-form rows of an instant;
- `derived_rows(response, response_id)`: extra tables parsed from responses;
- `reference_rows()`: reference-table rows written when a run is prepared.

A validator of the application's own is a function (`validation.custom_validator: package.module:function`) returning a `validation.Validator`.

A task whose inputs a config cannot declare computes them in Python instead: it overrides `sample_inputs`, and `assign_personas`, `template_variables`, `default_template`, `options`/`from_options` and `add_arguments`/`from_args` as it needs. The evaluation tasks do (their subjects are the outputs of other runs); until they have task files, they give their settings as a `pipeline.GenerationConfig`, which `prepare_run(task, settings)` makes their config. `applications/tarot/code/celtic_cross_evaluation.py` is an example.
