"""
The tarot application: Celtic Cross readings of sampled card draws, and their evaluation
by personas with the Celtic Cross Reading Evaluation Form.

The application's files: application.yaml (what it is), tasks.yaml (the tasks in the task file
format, so far tarot_celtic_cross, which runs from it; see
applications/template/inference_task_reference.yaml), tasks_draft.yaml (all three tasks in the
earlier, descriptive format), schemas.yaml, personas.yaml, workflows.yaml, prompts/, rubrics/,
and vector_space_config.json (the deck). The code does not read most of the others yet;
tests/test_tarot_application_files.py keeps them in step with it.

The Python is in code/, one module per task:
  readings                  TarotReadings: what tasks.yaml's tarot_celtic_cross cannot say (the
                            draw's long form), and the task of runs made before it
  persona_expectations      TarotPersonaExpectations: evaluation stage 1, what each persona
                            expects before seeing a reading
  celtic_cross_evaluation   CelticCrossEvaluation: evaluation stage 2, an evaluator fills in
                            the form for a reading

  validation                the card vocabulary, the Celtic Cross position aliases and the
                            validator for readings

The tasks are run by the general scripts, under the names their runs are stored with:
    python -m inference.generate run tarot_celtic_cross --n 20 ...
    python -m evaluation.evaluate expectations celtic_cross_evaluation ...
    python -m evaluation.evaluate run celtic_cross_evaluation --subject-dataset tarot-v1 ...

Still read from the general packages: the prompt templates (prompts/library/tarot/ and
prompts/library/evaluation/), the rubric (evaluation/rubrics/) and the personas (personas/).
"""
