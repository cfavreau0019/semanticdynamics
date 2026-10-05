# evaluation

Scores generated outputs against a rubric, using an LLM as the evaluator, optionally in the role of a persona. An evaluation run is an ordinary `data_generation` run whose requests are evaluations. It uses the same tables, live or batch execution, resume, validation and validation retries.

```
evaluation/
  rubric.py          Rubric: the structure every evaluation follows; scoring; form and response-format text
  checks.py          extract_json, RubricAnswerCheck, JsonFieldsCheck (validation of evaluator answers)
  subjects.py        load_subjects: the generated outputs to evaluate, read from generation runs
  expectations.py    stage 1: what a persona expects before seeing the output
  application.py     EvaluationApplication: subjects x personas -> requests -> scored result rows
  celtic_cross.py    CelticCrossEvaluation: the tarot-specific configuration
  evaluate_tarot.py  command-line script
  rubrics/celtic_cross_reading_v1.toml    the Celtic Cross Reading Evaluation Form as data
```

Prompts are in the prompts library: `prompts/library/evaluation/celtic_cross_evaluation_v1.toml` and `reading_expectations_v1.toml`.

## Evaluating Celtic Cross readings

```bash
# 50 readings from dataset tarot-v1, one random persona each (runs stage 1 first, for all personas in scope)
python -m evaluation.evaluate_tarot evaluate --readings-dataset tarot-v1 --n 50 --provider openai --model <model>

# record expectations once, then reuse them; 3 personas per reading; as a batch
python -m evaluation.evaluate_tarot expectations --provider openai --model <model>
python -m evaluation.evaluate_tarot evaluate --readings-dataset tarot-v1 --n 200 --personas-per-reading 3 \
    --expectations <expectations_run_id> --provider openai --model <model> --mode batch
python -m evaluation.evaluate_tarot collect <run_id> --wait

python -m evaluation.evaluate_tarot status <run_id>     # counts, mean total, grades, section averages, red flags
python -m evaluation.evaluate_tarot resume <run_id>     # finish an interrupted live run
python -m evaluation.evaluate_tarot list
python -m evaluation.evaluate_tarot evaluate --readings-dataset tarot-v1 --n 3 --dry-run   # prepare only, no API calls
```

Results go to `data/evaluations/<run_id>/` (gitignored). Readings are taken from `data/generations/`.

| Option | Default | Notes |
|---|---|---|
| `--readings-dataset` / `--readings-run` | – | Which readings to evaluate (one is required) |
| `--readings-template` | `celtic_cross_v1` | Only readings generated with this prompt template |
| `--n` | all | Number of readings; a seeded sample when fewer than available |
| `--personas-per-reading` | 1 | Distinct personas evaluating each reading; requests = readings × this |
| `--persona-sampling` | `random` | `cycle` walks through the persona set so each is used equally often |
| `--persona ID` | – | Restrict to specific personas; repeatable |
| `--pairing` | `sample` | `subject`: the persona a reading was written for evaluates it (for readings generated with `--personas`) |
| `--expectations RUN_ID` | generate | Reuse a stage-1 run; `--no-expectations` skips stage 1 |
| `--no-personas` | – | One neutral evaluation per reading, with no persona |
| `--json-mode` | off | Ask the provider for JSON output (OpenAI supports it) |
| `--max-tokens` | 4000 | A complete form answer needs roughly 1,000–1,500 output tokens |

**Choose a capable evaluator model.** In a smoke test, a small model (Hermes-4-14B) scored nearly every item 4, skipped most justifications, and cited item ids that don't exist. The pipeline records those problems as validation warnings, but they mean the scores discriminate poorly.

## How an evaluation is structured

**1. A rubric is data.** `rubrics/<name>_v<N>.toml` holds:
- the scoring scale with anchors;
- weighted sections of items, each tagged **O** (objective: the answer should not depend on the evaluator) or **S** (subjective: it should);
- red flags with score caps;
- grade bands;
- the qualitative outputs.

An item with `per = "positions"` is scored once per label supplied at evaluation time, which is how section 2 scores every position.

**2. The evaluator fills in the form; code does the arithmetic.** The model returns item scores, justifications, red flags with quotes, and the qualitative outputs. Section averages, weights, caps, the total and the grade are computed by `Rubric.score`:

```
total = min(cap, Σ weight × section_average / scale_max)
```

N/A items drop out of their section's average, and the lowest cap among fired red flags wins. The form's worked example (70.87, grade C) is a test.

**3. Two stages, as the form requires.** The persona's expectations (Part A3) must be recorded before they see the reading.
- **Stage 1 (`expectations`)** is one request per persona, without any reading. Your readings were generated without a question, so expectations depend only on the persona. They are generated once and reused.
- **Stage 2 (`evaluate`)** is one request per (reading, persona) pair: the persona, their expectations, the cards, the reading, and the form.

**4. Answers are validated before they become scores.** `RubricAnswerCheck` runs through the usual `validate=` hook, so a bad answer is regenerated, up to 2 retries.

| Severity | What is checked |
|---|---|
| Error (retried) | Not JSON; an item missing; a score not a whole number in range; N/A where the form doesn't allow it or the facts rule it out; unknown red flag; **a red-flag quote that does not appear in the reading**; a missing or malformed output |
| Warning (recorded) | A missing justification for a 1, 2 or 5; an item scored when the facts say N/A; a strength or improvement citing a nonexistent item |

An answer that still fails after retries stays in `responses` and `validations` for inspection but produces no evaluation rows.

**5. Facts decide N/A, not the evaluator.**
- Item 1.2 (reversals) must be N/A exactly when no card in the draw is reversed.
- Item 7.4 (must-haves) must be N/A exactly when the persona listed none.

Deviations from the form, all in `celtic_cross_reading_v1.toml` or the prompt:
- **N/A on 7.4.** The form allows N/A only on 1.2; without must-haves, 7.4 has nothing to score.
- **"The person", not "the persona".** Item texts were reworded so the same form reads correctly with or without a persona.
- **A calibration line.** The evaluation prompt adds a sentence saying a competent but generic reading earns 3s.

### Running without expectations

Evaluators get no expectations with `--no-expectations`, with `--no-personas`, and in any `--dry-run` (a dry run skips stage 1, so its prepared prompts differ from a real run's). The evaluation still runs, with these differences:
- **The prompt** omits the "What you expected" block. With `--no-expectations` the persona is still described; with `--no-personas` subjective items are answered as a typical member of the public.
- **Item 7.4 should be N/A**, since no must-haves are on record. If the evaluator scores it anyway, that is the `na_expected` warning and the score still counts, so section 7 is averaged over three items in some evaluations and four in others.
- **Subjective items have no pre-recorded anchor.** The evaluator infers what the person wanted after seeing the reading, which is what Part A3 of the form is meant to prevent.
- **`evaluations.expectations` is null.**

Passing `--expectations <run_id>` for a run that lacks one of the evaluating personas stops preparation with a `KeyError`; generate expectations for those personas first.

To compare evaluator models, record expectations once and pass the same `--expectations` run to every evaluation run, so each model scores against identical expectations.

## Tables

An evaluation run has the standard run tables (`runs`, `inputs`, `prompts`, `requests`, `responses`, …; see `data_generation/README.md`) plus:

| Table | One row per | Key columns |
|---|---|---|
| `evaluations` | completed form | subject_response_id (the reading), persona_id, rubric_id, rubric_sha256, section_scores (JSONB), score_before_caps, cap_applied, total_score, grade, n_red_flags, outputs (JSONB: persona_reaction, felt_seen, would_use_again, …), expectations (JSONB) |
| `evaluation_item_scores` | scored item | item_key (`3.2`, `2a.7`), item_id, section_key, item_type (O/S), label (position), score, is_na, justification |
| `evaluation_red_flags` | fired flag | flag_id, severity, cap, quote |
| `persona_expectations` | persona (stage-1 runs) | situation_summary, what_they_hoped_for, framing_preference, must_haves, sensitivities |
| `rubrics` | rubric version used | rubric_id, definition (JSONB), rubric_sha256 |

- `inputs.payload` holds the reference to the reading (`subject_response_id`, `subject_run_id`, the cards).
- `inputs.persona_id` is the evaluator.
- `evaluations.subject_response_id` joins to `responses.response_id` of the generation run, so scores join to readings, card draws, prompt versions and persona demographics.

Rubric-specific outputs live in JSON columns, so these tables serve any rubric.

```sql
-- O items should not depend on the persona: where do evaluators disagree most?
SELECT s.item_key, stddev(s.score) AS spread, count(*) AS n
FROM read_json('data/evaluations/*/evaluation_item_scores.jsonl') s
JOIN read_json('data/evaluations/*/evaluations.jsonl') e USING (evaluation_id)
WHERE s.item_type = 'O' AND NOT s.is_na
GROUP BY e.subject_response_id, s.item_key HAVING count(*) > 1
ORDER BY spread DESC;
```

## Evaluating something else

What is generic and what a new application supplies:

| Generic (reused) | Supplied per application |
|---|---|
| Rubric engine, answer validation, scoring | A rubric file (`rubrics/<name>_v1.toml`) |
| Subject loading, persona pairing, two-stage flow | An evaluation prompt template in the prompts library; it receives `rubric_form` and `response_format` generated from the rubric |
| Result tables, CLI plumbing (live/batch/resume) | A subclass of `EvaluationApplication` setting `rubric_ref` and `default_template`, and overriding `labels`, `na_required`, `context` as needed |

`celtic_cross.py` is the whole tarot-specific part: about 40 lines.

**Limits of the generic design:**
- A rubric assumes one numeric scale, weighted section averages, and caps from yes/no flags. A scheme that isn't shaped like that, such as pairwise comparisons or rankings, needs a different scorer.
- Subjects are assumed to be final responses of generation runs.
- Rubric versions aren't locked the way prompt templates are. Each run records the rubric's hash, and a test pins `celtic_cross_reading_v1`, so change the form by adding a `_v2` file.
