"""
Evaluating generated outputs against rubrics, with an LLM (optionally as a persona) as the
evaluator. An evaluation run is a data_generation run whose requests are evaluations.

  rubric         Rubric (scale, weighted sections of O/S items, red flags, grades), scoring
  checks         extract_json, RubricAnswerCheck, JsonFieldsCheck
  subjects       load_subjects: generated outputs to evaluate
  expectations   stage 1: a persona's expectations before seeing the output
  application    EvaluationApplication: the generic evaluation pipeline
  celtic_cross   CelticCrossEvaluation: tarot-specific configuration
  evaluate_tarot command-line script

See README.md.
"""
