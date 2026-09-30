# validation

Checks LLM outputs using only regex and vocabulary matching, with no further LLM calls. The core is generic; `tarot.py` configures it for tarot readings.

```
validation/
  core.py       Check, Validator, ValidationResult, Issue, ValidationContext
  extract.py    LabeledEntryExtractor, VocabularyMatcher, FreeTextMatcher  ("<label>: <value>" parsing)
  checks.py     NonEmpty, NotTruncated, RegexCheck, NoRefusal, LabeledValuesCheck, ForeignValuesCheck
  responses.py  response_validator (hook for llm), result_of, summary_frame, issues_frame
  tarot.py      celtic_cross_validator, spread_validator, TAROT_CARDS, CELTIC_CROSS_POSITIONS
```

## Tarot readings

```python
from llm import generate_many, save_responses
from validation import response_validator, summary_frame, issues_frame
from validation.tarot import celtic_cross_validator

validate = response_validator(celtic_cross_validator(), expected="instant")   # expected = metadata["instant"]

responses = generate_many(reqs, provider="featherless", validate=validate,
                          on_result=lambda r: save_responses([r], path, append=True))  # checkpoints include validation
[r.valid for r in responses]              # True / False / None (API failure or validator error)
summary_frame(responses)                  # one row per response
issues_frame(responses)                   # one row per issue; group by code / label
```

What `celtic_cross_validator()` checks:

| Check | Issue codes | Severity |
|---|---|---|
| `positions` | `missing_label` (position absent), `missing_value` (position listed without a card), `wrong_card`, `orientation_mismatch` (e.g. "reversed" dropped) | error |
| | `conflicting_entries` (position restated with a different card), `out_of_order` | warning |
| `foreign_cards` | `foreign_value`: a card was mentioned that wasn't drawn | warning |
| `not_truncated` | `truncated` (`finish_reason == "length"`) | error |
| `no_refusal` | `forbidden_pattern` ("I can't help…", "As an AI…") | error |
| `non_empty` | `too_short` (< 200 chars) | error |

**Formats it understands.** Tested on real Featherless and OpenAI output:
- `1. Present: Eight of Cups reversed - …`
- `1. **Present — Three of Pentacles:** …`
- short or lowercase position names (`Foundation`, `Conscious goal`)
- headings with the card on the next line
- prose ("In the Present, The Lovers reversed suggests…"), via an inline fallback that looks at most 40 characters past the position name

For orientation, `reversed`, `(Reversed)`, `inverted`, `(R)` and `not reversed` are all handled. Orientation is only read from the card's own phrase, not the rest of the paragraph.

**`result.passed`** is `True` when there are no *errors*. Warnings are recorded for review but don't fail a response.

## Where results go

`response.validation` is a plain JSON dict, `ValidationResult.to_dict()`:

```python
{"passed": False, "validator": "celtic_cross", "n_errors": 1, "n_warnings": 0,
 "issues": [{"check": "positions", "code": "orientation_mismatch", "severity": "error",
             "label": "Challenge", "message": "...", "data": {"expected": "Death reversed", "got": "Death"}}],
 "checks": {"positions": {"passed": False, "details": {"extracted": {...}, "modes": {...}, "n_found": 10}}, ...}}
```

It is saved and reloaded with the response by `save_responses` / `load_responses`. `result_of(response)` turns it back into a `ValidationResult`.

**Design notes**
- **Kept apart from `metadata` and `error`.**
  - `metadata` is the request's input, and the response shares that same dict object with the request. Writing into it would also change the request.
  - `error` means the API call failed. `ok=True, valid=False` means the call worked but the content is wrong.
- **One-way dependency.** `llm` never imports `validation`. It only calls a `validate(response, request) -> dict` function and stores the result.
- **A crashing validator can't lose results.**
  - A check that raises is recorded as a `check_crashed` error, and the other checks still run.
  - A validator that raises is recorded as `{"passed": None, "error": ...}`.
- **Failed API calls are not validated**, so `valid` stays `None` for them.

## Hooks in `llm`

| Path | How |
|---|---|
| Live, concurrent | `generate_many(..., validate=validate)`: runs in each worker thread *before* `on_result`, so checkpoints already contain it |
| Batch | `run_batch(..., validate=validate)` or `run_batch_results(job, requests=..., validate=validate)` |
| Anything else, or re-validating old files | `apply_validation(responses, validate, requests=None)` |

```python
# re-check saved results after improving the validator
responses = apply_validation(load_responses(path), validate)
save_responses(responses, path)
```

**Retries are on by default.** Responses that fail validation are regenerated up to 2 more times (`validation_retries=2`) by `generate_many` and blocking `run_batch`. For batches collected later, call `retry_invalid(responses, requests, validate, mode="live" | "batch")`.

Every failed attempt is kept on the returned response: `r.attempts` counts them, and `r.failed_attempts` holds each one's text and validation. That lets you measure first-try failure rates:

```python
first_try_valid = sum(r.attempts == 1 and r.valid for r in responses) / len(responses)
[f["validation"]["issues"] for r in responses for f in r.failed_attempts]   # why earlier attempts failed
```

## Other tasks

Other spreads: `spread_validator(["Past", "Present", "Future"])` or `spread_validator({position: [aliases]})`.

Other tasks entirely: build a `Validator` from the generic pieces.

```python
from validation import Validator, LabeledEntryExtractor, LabeledValuesCheck, VocabularyMatcher, FreeTextMatcher, NotTruncated

matcher = VocabularyMatcher({"positive": ["pos"], "negative": ["neg"], "neutral": []})
extractor = LabeledEntryExtractor(["Sentiment", "Topic"], matcher)
v = Validator([NotTruncated(), LabeledValuesCheck(extractor, check_values=False)], name="annotation")
v.validate(text, expected={"Sentiment": None, "Topic": None})     # presence-only
```

A custom rule is a `Check` subclass with `run(ctx) -> CheckResult`. Use `self.issue(code, message, severity, label)` to report findings.
