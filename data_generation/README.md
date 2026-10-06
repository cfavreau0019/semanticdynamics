# data_generation

Generates datasets with LLMs and stores each run as a set of relational tables: one JSONL file per table. The files are designed to load directly into Postgres (or DuckDB).

```
data_generation/
  schema.py          table definitions: validate JSONL rows and generate schema.sql
  schema.sql         Postgres DDL (generated; don't edit by hand)
  store.py           RunStore: data/generations/<run_id>/<table>.jsonl
  pipeline.py        Application interface; prepare_run, run_live, submit_batch, collect_batch
  tarot.py           TarotReadings application: card draws, prompt variables, validator
  generate_tarot.py  command-line entry point
```

## Tarot readings from the command line

```bash
# live: concurrent calls, validated and retried, written as they finish
python -m data_generation.generate_tarot generate --n 20 --provider featherless --max-tokens 2000

# options
python -m data_generation.generate_tarot generate --n 50 --provider featherless --temperature 0.9 \
    --extra-body '{"top_k": 40, "min_p": 0.02}' --seed 42 --draw unique --pin "Outcome=Death reversed" --name death-outcome

# batch (OpenAI): submit now, collect later (--wait blocks until done)
python -m data_generation.generate_tarot generate --n 500 --provider openai --model gpt-4o-mini --mode batch
python -m data_generation.generate_tarot collect <run_id> [--wait]

python -m data_generation.generate_tarot resume <run_id>    # live: finish interrupted / API-errored requests
python -m data_generation.generate_tarot status <run_id>
python -m data_generation.generate_tarot list
python -m data_generation.generate_tarot generate --n 5 --dry-run   # write inputs/prompts/requests only
```

### Growing a dataset over several sessions

Each `generate` call is its own run, in its own folder, with its own recorded settings, seed and git commit. To build one dataset across sessions, give every run the same `--dataset` label:

```bash
python -m data_generation.generate_tarot generate --n 500 --dataset tarot-v1 --provider featherless
# ... days later ...
python -m data_generation.generate_tarot generate --n 500 --dataset tarot-v1 --provider featherless
python -m data_generation.generate_tarot list --dataset tarot-v1      # runs + combined totals
```

- The label is stored once, in `runs.dataset`. Every other table reaches it through `run_id`.
- Without `--seed`, each run gets a fresh seed (recorded in `runs.config`), so sessions draw different cards. Passing a `--seed` already used in the same dataset with the same draw settings would repeat those draws, so the script warns.
- A run stopped partway is finished with `resume <run_id>`; that adds to the same run's files.

`python -m data_generation.generate_tarot generate --help` lists every option. The main ones:

| Option | Default | Notes |
|---|---|---|
| `--draw` | `replacement` | Same behaviour as `Instantiation`: repeats are allowed, so one card can appear upright *and* reversed. `unique` draws each card at most once, like a real deck. |
| `--pin POSITION=CARD` | – | Fixes a card in every draw; repeatable |
| `--validation-retries` | `2` | Regenerations per reading that fails validation |
| `--retry-mode` | `live` | Batch runs only: regenerate invalid readings live (fast) or as follow-up batches (cheaper, slower) |
| `--deck-config` | `$VECTOR_SPACE_CONFIG`, else `sandbox/vector_space_config.json` | Supplies the positions, the cards, and whether cards can be reversed |
| `--template NAME` | `celtic_cross_v1` (`celtic_cross_v2` with `--personas`) | A template from the [prompts library](../prompts/README.md). `name_vN` pins a version; a bare name takes the latest. Repeat the flag to render every draw with each template. |
| `--personas SET` | – | Assign one persona per draw from a [persona set](../personas/README.md), e.g. `tarot_personas` |
| `--persona-sampling` | `random` | `random`: seeded, with replacement. `cycle`: in order, every persona equally often. |
| `--persona ID` | – | Restrict to specific personas (id, number or full name); repeatable |

### Prompts, personas and comparing prompt versions

Prompt text lives in the `prompts` package as versioned template files; the tarot application only supplies the variables (`cards`, `positions`, `persona`).

```bash
python -m prompts list                                                     # available templates and their status
python -m data_generation.generate_tarot generate --n 200 --personas tarot_personas
python -m data_generation.generate_tarot generate --n 50 --template celtic_cross_v1 --template celtic_cross_v2
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
| `runs` | generation run | dataset (groups runs), config (JSONB: templates, persona set, params, seed, retries, deck file and its hash), code_version (git sha) + code_dirty, schema_version, status, summary |
| `personas` | persona used by the run | persona_set, persona_number, name, payload (JSONB), content_sha256 |
| `inputs` | structured input (card draw) | payload (JSONB `{position: card}`), input_index, persona_id |
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
- **Schema versions.** Version 2 added the two reference tables, `inputs.persona_id` and `prompts.template_sha256`. The new columns are nullable, so version 1 runs still load, resume and summarise. Version 3 added the evaluation tables (`rubrics`, `persona_expectations`, `evaluations`, `evaluation_item_scores`, `evaluation_red_flags`), which are written by the [evaluation](../evaluation/README.md) package. When querying files from both versions together in DuckDB, pass `union_by_name=true` to `read_json`.
- **`schema.py` is the single source of truth.** Every row is checked against it before writing, and `schema.sql` is generated from it; a test fails if the two drift apart. After changing a table, bump `SCHEMA_VERSION` and regenerate: `python -m data_generation.schema > data_generation/schema.sql`.
- **Run ids name the task.** A run id is `run_<application>_<YYYYMMDD>_<HHMMSS>_<6 hex>` (UTC), e.g. `run_tarot_celtic_cross_20261005_142210_3fa9c1`, so run folders group by task and sort by start time within it. Runs created before this naming have no task part (`run_20261001_…`). `list` always sorts by each run's recorded start time.
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

To set it up, `psql -f data_generation/schema.sql` creates the tables and the `final_responses` view. The simplest loader is DuckDB's Postgres extension:

```sql
ATTACH 'dbname=semanticdynamics' AS pg (TYPE postgres);
INSERT INTO pg.responses SELECT * FROM read_json('data/generations/<run_id>/responses.jsonl');
```

Load tables in the order they appear in `schema.sql` so foreign keys resolve.

**Large binary data** (hidden states, and embeddings before pgvector) should stay in files such as `.npy` or safetensors, with the tables storing a path to each file.

## Adding another application

Subclass `pipeline.Application`:
- `sample_inputs(n, rng)` returns a list of payload dicts;
- `default_template` (and optionally `persona_template`) name templates in the prompts library;
- `template_variables(payload, persona)` returns the variables those templates use;
- optionally `input_items(payload)`, `validator()` and `options()`;
- optionally, for richer applications such as evaluation:
  - `assign_personas(...)` controls persona pairing;
  - `reference_rows()` supplies reference-table rows written at preparation;
  - `validation_expected(metadata)` sets what the validator compares against;
  - `derived_rows(response, response_id)` supplies extra tables parsed from responses.

`prepare_run`, `run_live`, `submit_batch` and `collect_batch` then work unchanged. `tarot.py` is the reference implementation.
