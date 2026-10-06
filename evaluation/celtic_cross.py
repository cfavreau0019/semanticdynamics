"""
Celtic Cross reading evaluation: the tarot-specific configuration of EvaluationApplication.

Rubric:   evaluation/rubrics/celtic_cross_reading_v1.toml (the Celtic Cross Reading
          Evaluation Form)
Prompts:  prompts/library/evaluation/celtic_cross_evaluation_v2.toml (stage 2: scoring; v1 is the
          same wording in an order that a prompt cache cannot reuse)
          prompts/library/evaluation/reading_expectations_v1.toml    (stage 1: Part A3)

Tarot-specific facts supplied here:
  * the ten positions scored in section 2 are the positions of the draw the reading was for
  * item 1.2 (reversals) is N/A exactly when no card in the draw is reversed
  * item 7.4 (must-haves) is N/A exactly when the evaluator listed no must-haves
  * the readings were generated without a question, from standard Rider-Waite-Smith card names
"""
from typing import Any, Mapping, Optional

from evaluation.application import EvaluationApplication
from evaluation.rubric import Labels
from validation.tarot import parse_state

# What the persona is told they are about to receive, for stage 1 (persona expectations).
EXPECTATION_CONTEXT = {"reading_type": "Celtic Cross tarot reading"}
DECK = "Rider-Waite-Smith (standard card names; the reading did not state a deck)"


class CelticCrossEvaluation(EvaluationApplication):
    name = "celtic_cross_evaluation"
    subject_type = "tarot_reading"
    rubric_ref = "celtic_cross_reading_v1"
    default_template = persona_template = "celtic_cross_evaluation_v2"

    def labels(self, payload: dict) -> Labels:
        return {"positions": list(payload["subject_input"])}

    def na_required(self, payload: dict, expectations: Optional[dict]) -> Mapping[str, bool]:
        any_reversed = any(parse_state(state)[1] for state in payload["subject_input"].values())
        return {"1.2": not any_reversed,
                "7.4": not (expectations or {}).get("must_haves")}

    def context(self, payload: dict) -> dict[str, Any]:
        n_reversed = sum(parse_state(state)[1] for state in payload["subject_input"].values())
        return {"question_asked": "none", "deck": DECK, "reversals_used": "yes" if n_reversed else "no",
                "n_reversed": n_reversed}

    def template_variables(self, payload: dict, persona: Optional[dict]) -> dict[str, Any]:
        variables = super().template_variables(payload, persona)
        variables["cards"] = variables.pop("subject_input")
        variables["reading_text"] = variables.pop("subject_text")
        return variables
