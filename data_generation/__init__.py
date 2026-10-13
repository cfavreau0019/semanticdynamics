"""
Generating datasets with LLMs and storing them as relational tables (JSONL per table per
run, Postgres-ready; see README.md).

  schema     table definitions -> JSONL row checks and Postgres DDL (schema.sql)
  store      RunStore: data/generations/<run_id>/<table>.jsonl
  task       InferenceTask: what a task supplies (inputs, prompt variables, validator, ...)
  pipeline   prepare_run, run_live, submit_batch, collect_batch
             (prompt text comes from the `prompts` package, personas from `personas`)
  generate_tarot   command-line entry point

The tasks themselves live with their application: applications/tarot/readings.py is
TarotReadings (card draws, prompt variables, validator).
"""
