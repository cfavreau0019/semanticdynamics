"""
Running inference tasks as recorded runs: sampling a task's inputs, rendering its prompts,
calling the model live or in batches, validating and retrying, and storing every step as
relational tables (JSONL per table per run, Postgres-ready; see README.md).

Talking to a model is the `llm` package's job; this package runs a task, many requests at a
time, through it and keeps the record.

  schema     table definitions -> JSONL row checks and Postgres DDL (schema.sql)
  store      RunStore: data/generations/<run_id>/<table>.jsonl
  task       InferenceTask: what a task supplies (inputs, prompt variables, validator, ...)
  task_file  load_task_file: an application's tasks.yaml as checked TaskSpecs (states from `instantiation`)
  pipeline   prepare_run, run_live, submit_batch, collect_batch
             (prompt text comes from the `prompts` package, personas from `personas`)
  generate   command-line script: runs the generation tasks of every application

The tasks themselves live with their application: applications/tarot/code/readings.py is
TarotReadings (card draws, prompt variables, validator).
"""
