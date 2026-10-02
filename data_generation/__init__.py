"""
Generating datasets with LLMs and storing them as relational tables (JSONL per table per
run, Postgres-ready; see README.md).

  schema     table definitions -> JSONL row checks and Postgres DDL (schema.sql)
  store      RunStore: data/generations/<run_id>/<table>.jsonl
  pipeline   Application interface, prepare_run, run_live, submit_batch, collect_batch
             (prompt text comes from the `prompts` package, personas from `personas`)
  tarot      TarotReadings application (card draws, prompt variables, validator)
  generate_tarot   command-line entry point
"""
