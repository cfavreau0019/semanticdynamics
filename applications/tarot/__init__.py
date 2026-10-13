"""
The tarot application: Celtic Cross readings of sampled card draws, and their evaluation
by personas with the Celtic Cross Reading Evaluation Form.

Tasks, one file each:
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

Still kept with the general packages: the prompt templates (prompts/library/tarot/ and
prompts/library/evaluation/), the rubric (evaluation/rubrics/), the personas (personas/)
and the deck config (sandbox/vector_space_config.json).
"""
