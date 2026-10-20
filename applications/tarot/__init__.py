"""
The tarot application: Celtic Cross readings of sampled card draws, and their evaluation
by personas with the Celtic Cross Reading Evaluation Form.

The application's files: application.yaml (what it is), schemas.yaml, personas.yaml, tasks.yaml
(with tarot_celtic_cross_inference_task.yaml, a draft of one task in the new format),
workflows.yaml, prompts/, rubrics/, and vector_space_config.json (the deck). The code does not
read most of them yet; tests/test_tarot_application_files.py keeps them in step with it.

The Python is in code/, one module per task:
  readings                  TarotReadings: a card draw -> a reading
  persona_expectations      TarotPersonaExpectations: evaluation stage 1, what each persona
                            expects before seeing a reading
  celtic_cross_evaluation   CelticCrossEvaluation: evaluation stage 2, an evaluator fills in
                            the form for a reading

  validation                the card vocabulary, the Celtic Cross position aliases and the
                            validator for readings

The tasks are run by the general scripts, under the names their runs are stored with:
    python -m data_generation.generate run tarot_celtic_cross --n 20 ...
    python -m evaluation.evaluate expectations celtic_cross_evaluation ...
    python -m evaluation.evaluate run celtic_cross_evaluation --subject-dataset tarot-v1 ...

Still read from the general packages: the prompt templates (prompts/library/tarot/ and
prompts/library/evaluation/), the rubric (evaluation/rubrics/) and the personas (personas/).
"""
