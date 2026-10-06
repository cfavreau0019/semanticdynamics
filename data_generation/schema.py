"""
Table definitions for generated data: the single source of truth for both the JSONL files
written by RunStore and the Postgres DDL (schema.sql, regenerate with
`python -m data_generation.schema > data_generation/schema.sql`).

Conventions
- one JSONL file per table per run: data/generations/<run_id>/<table>.jsonl
- every row is a flat JSON object with exactly the table's columns; JSONB columns hold
  nested JSON (not strings); timestamps are ISO-8601 UTC strings
- ids are text; every table carries run_id so a run can be loaded, deleted or partitioned
  as a unit
- append-only tables are never rewritten; `mutable` tables (runs, batches) are small and
  rewritten in full when a row changes (load them with upsert-by-primary-key)
- `reference` tables (prompt_templates, personas) describe things shared between runs. They
  have no run_id: each run's file holds the rows that run used, so the same row appears in
  every run that used it (load with ON CONFLICT DO NOTHING / de-duplicate on the primary key)

Schema versions
  1  initial
  2  prompt_templates and personas reference tables; inputs.persona_id; prompts.template_sha256
     (all new columns are nullable, so version 1 runs load unchanged)
  3  evaluation tables: rubrics (reference), persona_expectations, evaluations,
     evaluation_item_scores, evaluation_red_flags (new tables only; earlier runs load unchanged)
  4  responses.cached_tokens (nullable; earlier runs load unchanged and count as 0 cached)
"""
from dataclasses import dataclass, field
from typing import Any, Optional

SCHEMA_VERSION = 4


@dataclass(frozen=True)
class Column:
    name: str
    type: str                       # Postgres type
    nullable: bool = True
    references: Optional[str] = None  # "table(column)"
    doc: str = ""


@dataclass(frozen=True)
class Table:
    name: str
    columns: tuple[Column, ...]
    primary_key: tuple[str, ...]
    doc: str = ""
    mutable: bool = False
    reference: bool = False

    @property
    def column_names(self) -> list[str]:
        return [c.name for c in self.columns]


def _t(name, pk, doc, *columns, mutable=False, reference=False) -> Table:
    return Table(name, tuple(columns), tuple(pk), doc, mutable, reference)


RUN = Column("run_id", "TEXT", False, "runs(run_id)", "run this row belongs to")

TABLES: dict[str, Table] = {t.name: t for t in [
    _t("runs", ["run_id"], "One generation run: configuration, provenance, status, summary.",
       Column("run_id", "TEXT", False),
       Column("name", "TEXT", doc="optional human label for the run"),
       Column("dataset", "TEXT", doc="groups runs that belong to one dataset (e.g. collected over several sessions)"),
       Column("application", "TEXT", False, doc="e.g. tarot_celtic_cross"),
       Column("mode", "TEXT", False, doc="live | batch | dry_run"),
       Column("status", "TEXT", False,
              doc="prepared | running | submitted | completed | partial (some requests unfinished) | interrupted"),
       Column("provider", "TEXT"),
       Column("model", "TEXT"),
       Column("config", "JSONB", False, doc="all generation settings (template, params, seed, retries, ...)"),
       Column("code_version", "TEXT", doc="git commit of the code that produced the run"),
       Column("code_dirty", "BOOLEAN", doc="uncommitted changes present when the run started"),
       Column("schema_version", "INTEGER", False),
       Column("summary", "JSONB", doc="counts / token totals, filled in when the run finishes"),
       Column("created_at", "TIMESTAMPTZ", False),
       Column("updated_at", "TIMESTAMPTZ", False),
       mutable=True),
    _t("personas", ["persona_id"], "People used as prompt inputs (a snapshot of each persona a run used).",
       Column("persona_id", "TEXT", False, doc="the persona's own id (stable across runs)"),
       Column("persona_set", "TEXT", False, doc="set it came from, e.g. tarot_personas"),
       Column("persona_number", "INTEGER"),
       Column("name", "TEXT"),
       Column("payload", "JSONB", False, doc="the full persona record"),
       Column("content_sha256", "TEXT", False, doc="hash of payload; differs if a persona was edited"),
       reference=True),
    _t("inputs", ["input_id"], "Structured inputs the prompts are built from (e.g. a card draw).",
       Column("input_id", "TEXT", False),
       RUN,
       Column("input_index", "INTEGER", False, doc="position within the run"),
       Column("input_type", "TEXT", False, doc="e.g. celtic_cross_draw"),
       Column("payload", "JSONB", False, doc="the input itself, e.g. {position: card state}"),
       Column("persona_id", "TEXT", True, "personas(persona_id)", "who the prompt is written for; NULL = none"),
       Column("created_at", "TIMESTAMPTZ", False)),
    _t("input_items", ["input_id", "item_index"], "Long form of inputs: one row per labelled element.",
       Column("input_id", "TEXT", False, "inputs(input_id)"),
       RUN,
       Column("item_index", "INTEGER", False),
       Column("label", "TEXT", False, doc="e.g. position 'Outcome'"),
       Column("value", "TEXT", False, doc="full value, e.g. 'Death reversed'"),
       Column("entity", "TEXT", doc="value without qualifiers, e.g. 'Death'"),
       Column("qualifier", "TEXT", doc="e.g. 'reversed' / 'upright'")),
    _t("prompt_templates", ["template_sha256"],
       "Prompt template versions used by runs: the exact text behind each rendered prompt.",
       Column("template_sha256", "TEXT", False, doc="hash of name, version and template text"),
       Column("template_id", "TEXT", False, doc="<name>_v<version>, e.g. celtic_cross_v2"),
       Column("name", "TEXT", False),
       Column("version", "INTEGER", False),
       Column("system_template", "TEXT"),
       Column("user_template", "TEXT", False),
       Column("required", "JSONB", False, doc="variables that must be supplied"),
       Column("optional", "JSONB", False, doc="variables that may be omitted, e.g. persona"),
       Column("description", "TEXT"),
       Column("changes", "TEXT", doc="what changed from the previous version, and why"),
       Column("based_on", "TEXT", doc="template_id this version was derived from"),
       Column("status", "TEXT", doc="locked | draft at the time of the run"),
       reference=True),
    _t("prompts", ["prompt_id"], "Rendered prompts (one input can have several templates).",
       Column("prompt_id", "TEXT", False),
       Column("input_id", "TEXT", False, "inputs(input_id)"),
       RUN,
       Column("template", "TEXT", False, doc="template id, e.g. celtic_cross_v1"),
       Column("template_sha256", "TEXT", True, "prompt_templates(template_sha256)",
              "exact template content used (NULL for schema version 1 runs)"),
       Column("system", "TEXT"),
       Column("user_text", "TEXT", False),
       Column("messages", "JSONB", False, doc="exact message list sent"),
       Column("prompt_sha256", "TEXT", False, doc="hash of messages, for dedup / grouping"),
       Column("created_at", "TIMESTAMPTZ", False)),
    _t("requests", ["request_id"], "One API call specification: prompt + provider + model + params.",
       Column("request_id", "TEXT", False),
       Column("prompt_id", "TEXT", False, "prompts(prompt_id)"),
       RUN,
       Column("provider", "TEXT", False),
       Column("model", "TEXT"),
       Column("params", "JSONB", False, doc="standard params (max_tokens, temperature, ...)"),
       Column("extra_body", "JSONB", False, doc="provider-specific params (top_k, min_p, ...)"),
       Column("mode", "TEXT", False, doc="live | batch"),
       Column("created_at", "TIMESTAMPTZ", False)),
    _t("batches", ["batch_id"], "Provider batch jobs, including validation-retry rounds.",
       Column("batch_id", "TEXT", False),
       RUN,
       Column("provider", "TEXT", False),
       Column("round", "INTEGER", False, doc="0 = initial submission, 1.. = retry rounds"),
       Column("endpoint", "TEXT"),
       Column("status", "TEXT", False, doc="pending | running | completed | failed | expired | cancelled"),
       Column("raw_status", "TEXT"),
       Column("n_requests", "INTEGER", False),
       Column("request_counts", "JSONB"),
       Column("input_file", "TEXT"),
       Column("collected", "BOOLEAN", False, doc="results written to responses"),
       Column("created_at", "TIMESTAMPTZ", False),
       Column("updated_at", "TIMESTAMPTZ", False),
       mutable=True),
    _t("responses", ["response_id"],
       "Every generation attempt (validation retries and resumed API failures included) and its metadata. "
       "A request's final response is its highest attempt (see the final_responses view).",
       Column("response_id", "TEXT", False, doc="<request_id>-a<attempt>"),
       Column("request_id", "TEXT", False, "requests(request_id)"),
       RUN,
       Column("attempt", "INTEGER", False,
              doc="1-based, continues across resumes; the highest attempt is the request's final response"),
       Column("batch_id", "TEXT", True, "batches(batch_id)", "batch that produced it (NULL for live)"),
       Column("model", "TEXT", doc="model reported by the provider"),
       Column("finish_reason", "TEXT"),
       Column("error", "TEXT", doc="API error; NULL on success"),
       Column("prompt_tokens", "INTEGER"),
       Column("completion_tokens", "INTEGER"),
       Column("total_tokens", "INTEGER"),
       Column("cached_tokens", "INTEGER", doc="part of prompt_tokens served from the provider's prompt cache"),
       Column("received_at", "TIMESTAMPTZ", False, doc="when the row was written")),
    _t("response_texts", ["response_id"], "Generated text, kept apart from the narrow metadata table.",
       Column("response_id", "TEXT", False, "responses(response_id)"),
       RUN,
       Column("text", "TEXT"),
       Column("n_chars", "INTEGER")),
    _t("validations", ["response_id"], "Validation verdict for each attempt.",
       Column("response_id", "TEXT", False, "responses(response_id)"),
       RUN,
       Column("validator", "TEXT"),
       Column("passed", "BOOLEAN", doc="NULL if the validator crashed"),
       Column("n_errors", "INTEGER"),
       Column("n_warnings", "INTEGER"),
       Column("validator_error", "TEXT"),
       Column("details", "JSONB", doc="per-check details, e.g. extracted {position: card}")),
    _t("validation_issues", ["response_id", "issue_index"], "One row per validation issue.",
       Column("response_id", "TEXT", False, "responses(response_id)"),
       RUN,
       Column("issue_index", "INTEGER", False),
       Column("check_name", "TEXT", False),
       Column("code", "TEXT", False),
       Column("severity", "TEXT", False),
       Column("label", "TEXT"),
       Column("message", "TEXT"),
       Column("data", "JSONB")),

    # ---- evaluation (written by the `evaluation` package; an evaluation run is a normal run whose
    #      responses are an evaluator's answers) -------------------------------------------------
    _t("rubrics", ["rubric_sha256"], "Rubric versions used by evaluation runs: the full form definition.",
       Column("rubric_sha256", "TEXT", False, doc="hash of the rubric file's content"),
       Column("rubric_id", "TEXT", False, doc="<name>_v<version>, e.g. celtic_cross_reading_v1"),
       Column("name", "TEXT", False),
       Column("version", "INTEGER", False),
       Column("title", "TEXT"),
       Column("definition", "JSONB", False, doc="scale, sections, items, red flags, grades, outputs"),
       reference=True),
    _t("persona_expectations", ["response_id"],
       "What a persona expected before seeing the thing evaluated (recorded first, so hindsight can't colour it).",
       Column("response_id", "TEXT", False, "responses(response_id)", "the response these were parsed from"),
       RUN,
       Column("persona_id", "TEXT", False, "personas(persona_id)"),
       Column("context", "JSONB", doc="what the persona was told they were about to receive, e.g. the question"),
       Column("situation_summary", "TEXT"),
       Column("what_they_hoped_for", "TEXT"),
       Column("framing_preference", "TEXT", doc="predictive | reflective | creative | no_preference"),
       Column("must_haves", "JSONB"),
       Column("sensitivities", "JSONB")),
    _t("evaluations", ["evaluation_id"],
       "One completed evaluation form: a subject (e.g. a reading) scored against a rubric by one evaluator.",
       Column("evaluation_id", "TEXT", False),
       RUN,
       Column("response_id", "TEXT", False, "responses(response_id)", "the evaluator's answer this was parsed from"),
       Column("request_id", "TEXT", False, "requests(request_id)"),
       Column("subject_type", "TEXT", False, doc="what was evaluated, e.g. tarot_reading"),
       Column("subject_response_id", "TEXT", False,
              doc="responses.response_id of the evaluated output, in the run that generated it (subject_run_id)"),
       Column("subject_run_id", "TEXT", doc="run that generated the subject"),
       Column("persona_id", "TEXT", True, "personas(persona_id)", "evaluator persona; NULL = no persona"),
       Column("evaluator_type", "TEXT", False, doc="llm | human | human_as_persona"),
       Column("rubric_id", "TEXT", False),
       Column("rubric_sha256", "TEXT", False, "rubrics(rubric_sha256)"),
       Column("expectations", "JSONB", doc="the persona's expectations the evaluator was given"),
       Column("section_scores", "JSONB", False, doc="{section key: average on the rubric scale}"),
       Column("score_before_caps", "DOUBLE PRECISION", False),
       Column("cap_applied", "DOUBLE PRECISION", doc="red-flag cap that lowered the total; NULL if none did"),
       Column("total_score", "DOUBLE PRECISION", False, doc="0-100 after caps, one decimal"),
       Column("grade", "TEXT", False),
       Column("n_red_flags", "INTEGER", False),
       Column("outputs", "JSONB", doc="the rubric's qualitative outputs, e.g. persona_reaction, felt_seen"),
       Column("subject_length_words", "INTEGER"),
       Column("evaluated_at", "TIMESTAMPTZ", False)),
    _t("evaluation_item_scores", ["evaluation_id", "item_key"], "One row per scored rubric item.",
       Column("evaluation_id", "TEXT", False, "evaluations(evaluation_id)"),
       RUN,
       Column("item_key", "TEXT", False, doc="item id, or <item id>.<n> for per-label items, e.g. 2a.3"),
       Column("item_id", "TEXT", False),
       Column("section_key", "TEXT", False),
       Column("item_type", "TEXT", False, doc="O = objective, S = subjective"),
       Column("label", "TEXT", doc="the label a per-label score refers to, e.g. the position"),
       Column("score", "INTEGER", doc="NULL when N/A"),
       Column("is_na", "BOOLEAN", False),
       Column("justification", "TEXT")),
    _t("evaluation_red_flags", ["evaluation_id", "flag_id"], "Red flags an evaluation raised.",
       Column("evaluation_id", "TEXT", False, "evaluations(evaluation_id)"),
       RUN,
       Column("flag_id", "TEXT", False),
       Column("severity", "TEXT"),
       Column("cap", "DOUBLE PRECISION"),
       Column("quote", "TEXT", doc="the offending passage, quoted from the subject")),
]}


FINAL_RESPONSES_VIEW = """-- The response returned for each request: its highest attempt.
CREATE OR REPLACE VIEW final_responses AS
SELECT DISTINCT ON (request_id) *
FROM responses
ORDER BY request_id, attempt DESC;
"""


def check_row(table: str, row: dict[str, Any]) -> dict[str, Any]:
    """Return the row with exactly the table's columns (missing nullable ones -> None)."""
    t = TABLES[table]
    unknown = set(row) - set(t.column_names)
    if unknown:
        raise ValueError(f"{table}: unknown columns {sorted(unknown)}")
    out = {}
    for c in t.columns:
        value = row.get(c.name)
        if value is None and not c.nullable:
            raise ValueError(f"{table}.{c.name} is required")
        out[c.name] = value
    return out


def ddl() -> str:
    """Postgres DDL for all tables, in dependency order."""
    parts = [f"-- Generated by data_generation/schema.py (schema version {SCHEMA_VERSION}). Do not edit by hand.\n"]
    for t in TABLES.values():
        lines = []
        for c in t.columns:
            line = f"    {c.name} {c.type}{'' if c.nullable else ' NOT NULL'}"
            if c.references:
                line += f" REFERENCES {c.references}"
            lines.append(line)
        lines.append(f"    PRIMARY KEY ({', '.join(t.primary_key)})")
        parts.append(f"-- {t.doc}\nCREATE TABLE IF NOT EXISTS {t.name} (\n" + ",\n".join(lines) + "\n);\n")
        for c in t.columns:
            if c.doc:
                parts.append(f"COMMENT ON COLUMN {t.name}.{c.name} IS '{c.doc.replace(chr(39), chr(39) * 2)}';")
        parts.append("")
    parts.append(FINAL_RESPONSES_VIEW)
    return "\n".join(parts)


if __name__ == "__main__":
    print(ddl())
