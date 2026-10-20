"""
Evaluating generated outputs against rubrics, with an LLM (optionally as a persona) as the
evaluator. An evaluation run is an inference run whose requests are evaluations.

  rubric         Rubric (scale, weighted sections of O/S items, red flags, grades), scoring
  checks         extract_json, RubricAnswerCheck, JsonFieldsCheck
  subjects       load_subjects: generated outputs to evaluate
  expectations   PersonaExpectations, stage 1: a persona's expectations before seeing the output
  task           EvaluationTask, stage 2: the generic evaluation task
  evaluate       command-line script: runs the evaluation tasks of every application

What an application adds to the two tasks lives with it: applications/tarot/ has
TarotPersonaExpectations and CelticCrossEvaluation.

See README.md.
"""
