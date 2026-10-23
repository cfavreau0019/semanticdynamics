"""
Running inference tasks as recorded runs: sampling a task's inputs, rendering its prompts,
calling the model live or in batches, validating and retrying, and storing every step as
relational tables (JSONL per table per run, Postgres-ready; see README.md).

Talking to a model is the `llm` package's job; this package runs a task, many requests at a
time, through it and keeps the record.

  schema     table definitions -> JSONL row checks and Postgres DDL (schema.sql)
  store      RunStore: data/generations/<run_id>/<table>.jsonl
  task_config  InferenceTaskConfig: a task's whole config, from a task file (edited), code or a run
  task       InferenceTask: a task built from its config (inputs, prompt variables, validation, ...)
  pipeline   prepare_run, run_live, submit_batch, collect_batch: each reads task.config
             (prompt text comes from the `prompts` package, personas from `personas`)
  generate   command-line script, and run_task(): runs the tasks of every application's task file

The tasks themselves live with their application, in its tasks.yaml; Python of its own, if any,
next to it (applications/tarot/code/readings.py is TarotReadings).
"""
