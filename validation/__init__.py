"""
LLM-output validation without further LLM calls (regex / vocabulary based).

  core        Check, Validator, ValidationResult, Issue, ValidationContext
  extract     LabeledEntryExtractor, VocabularyMatcher, FreeTextMatcher
  checks      NonEmpty, NotTruncated, RegexCheck, NoRefusal, LabeledValuesCheck, ForeignValuesCheck;
              GENERIC_CHECKS / generic_checks: the checks a task file names by name
  responses   response_validator (hook for llm), summary_frame, issues_frame

Task-specific configurations live with their application, e.g. applications/tarot/code/validation.py
(celtic_cross_validator, spread_validator, TAROT_CARDS).
"""
from validation.core import (
    ERROR, WARNING, Check, CheckResult, Issue, ValidationContext, ValidationResult, Validator,
)
from validation.extract import (
    Entry, FreeTextMatcher, LabeledEntryExtractor, ValueMatcher, VocabularyMatcher, default_aliases,
)
from validation.checks import (
    GENERIC_CHECKS, ForeignValuesCheck, LabeledValuesCheck, NonEmpty, NoRefusal, NotTruncated, RegexCheck,
    exact_comparator, generic_checks,
)
from validation.responses import issues_frame, response_validator, result_of, summary_frame

__all__ = [
    "ERROR", "WARNING", "Check", "CheckResult", "Issue", "ValidationContext", "ValidationResult", "Validator",
    "Entry", "FreeTextMatcher", "LabeledEntryExtractor", "ValueMatcher", "VocabularyMatcher", "default_aliases",
    "ForeignValuesCheck", "LabeledValuesCheck", "NonEmpty", "NoRefusal", "NotTruncated", "RegexCheck",
    "GENERIC_CHECKS", "exact_comparator", "generic_checks",
    "issues_frame", "response_validator", "result_of", "summary_frame",
]
