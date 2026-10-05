"""
EvaluationApplication: a data_generation Application whose requests are evaluations.

    subjects (outputs of earlier generation runs)  x  evaluators (personas, optional)
        -> one request per pair: the rubric form, the subject, the evaluator's persona and
           expectations -> a JSON answer -> validated against the rubric -> scored in code
        -> rows in evaluations, evaluation_item_scores, evaluation_red_flags

Because it is an ordinary Application, an evaluation run gets everything a generation run
has: the same tables, live or batch execution, resume, validation and validation retries.

What is generic here: selecting subjects, pairing them with personas, supplying template
variables, validating and scoring answers, writing result rows. What a subclass supplies:
the rubric, the prompt template, the labels for per-label items, which items' N/A status is
decided by facts, and the context shown to the evaluator. See celtic_cross.py.
"""
import random
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from data_generation.pipeline import INPUT_KEY, Application
from data_generation.store import DEFAULT_ROOT, RunStore, new_id, utc_now
from evaluation.checks import RubricAnswerCheck, extract_json
from evaluation.expectations import load_expectations
from evaluation.rubric import Labels, Rubric, get_rubric
from evaluation.subjects import Subject, load_subjects
from llm import ChatResponse
from personas import PersonaSet
from validation import NonEmpty, NotTruncated, Validator

PAIRINGS = ("sample", "subject")


class EvaluationApplication(Application):
    """
    subjects_root, subject_runs, subject_dataset, subject_templates, valid_only
                         which generated outputs to evaluate (see subjects.load_subjects)
    personas_per_subject evaluators per subject when personas are used
    pairing              "sample": personas_per_subject distinct personas drawn per subject
                         "subject": the persona the subject was generated for evaluates it
    expectations_run     run id of a persona-expectations run (stage 1); None = none supplied
    rubric               rubric reference; None = the class's default
    """
    input_type = "evaluation"
    subject_type = "output"
    evaluator_type = "llm"
    rubric_ref: str = ""

    def __init__(
        self,
        subjects_root: str | Path = DEFAULT_ROOT,
        subject_runs: Sequence[str] = (),
        subject_dataset: Optional[str] = None,
        subject_templates: Sequence[str] = (),
        valid_only: bool = True,
        personas_per_subject: int = 1,
        pairing: str = "sample",
        expectations_run: Optional[str] = None,
        expectations_root: Optional[str | Path] = None,
        rubric: Optional[str] = None,
    ):
        if pairing not in PAIRINGS:
            raise ValueError(f"pairing must be one of {PAIRINGS}, got {pairing!r}")
        if personas_per_subject < 1:
            raise ValueError("personas_per_subject must be at least 1")
        self.subjects_root = Path(subjects_root)
        self.subject_runs = list(subject_runs)
        self.subject_dataset = subject_dataset
        self.subject_templates = list(subject_templates)
        self.valid_only = valid_only
        self.personas_per_subject = personas_per_subject
        self.pairing = pairing
        self.expectations_run = expectations_run
        self.expectations_root = Path(expectations_root) if expectations_root else None
        self.rubric: Rubric = get_rubric(rubric or self.rubric_ref)
        self._subjects: Optional[dict[str, Subject]] = None
        self._expectations: Optional[dict[str, dict]] = None

    # ---- to be supplied by a subclass ------------------------------------------------
    def labels(self, payload: dict) -> Labels:
        """Label lists for the rubric's per-label items, e.g. {"positions": [...]}."""
        return {}

    def na_required(self, payload: dict, expectations: Optional[dict]) -> Mapping[str, bool]:
        """{item_id: must be N/A?} for items whose N/A status the facts decide."""
        return {}

    def context(self, payload: dict) -> dict[str, Any]:
        """Facts about the subject shown to the evaluator (e.g. question asked, deck)."""
        return {}

    # ---- subjects and expectations ---------------------------------------------------
    @property
    def subjects(self) -> dict[str, Subject]:
        if self._subjects is None:
            found = load_subjects(self.subjects_root, self.subject_runs, self.subject_dataset,
                                  self.subject_templates, self.valid_only)
            self._subjects = {s.response_id: s for s in found}
        return self._subjects

    @property
    def expectations(self) -> dict[str, dict]:
        if self._expectations is None:
            self._expectations = {}
            if self.expectations_run:
                if self.expectations_root is None:
                    raise ValueError("expectations_root is needed to load expectations_run")
                self._expectations = load_expectations(RunStore.open(self.expectations_run, self.expectations_root))
        return self._expectations

    def expectations_for(self, persona_id: Optional[str]) -> Optional[dict]:
        if persona_id is None or not self.expectations_run:
            return None
        if persona_id not in self.expectations:
            raise KeyError(f"No expectations for persona {persona_id} in run {self.expectations_run}; "
                           f"generate them first (stage 1)")
        return self.expectations[persona_id]

    def options(self) -> dict:
        return {
            "subjects_root": str(self.subjects_root), "subject_runs": self.subject_runs,
            "subject_dataset": self.subject_dataset, "subject_templates": self.subject_templates,
            "valid_only": self.valid_only, "personas_per_subject": self.personas_per_subject,
            "pairing": self.pairing, "expectations_run": self.expectations_run,
            "expectations_root": str(self.expectations_root) if self.expectations_root else None,
            "rubric": self.rubric.id, "rubric_sha256": self.rubric.sha256,
        }

    # ---- Application interface -------------------------------------------------------
    def sample_inputs(self, n: int, rng: random.Random) -> list[dict]:
        """n subjects (a seeded sample if there are more), each repeated personas_per_subject times."""
        pool = list(self.subjects.values())
        if not pool:
            raise ValueError("No subjects matched the selection")
        chosen = pool if n >= len(pool) else rng.sample(pool, n)
        return [self._payload(s) for s in chosen for _ in range(self.personas_per_subject)]

    @staticmethod
    def _payload(s: Subject) -> dict:
        return {"subject_response_id": s.response_id, "subject_request_id": s.request_id,
                "subject_run_id": s.run_id, "subject_template": s.template, "subject_model": s.model,
                "subject_persona_id": s.persona_id, "subject_input": s.input}

    def assign_personas(self, payloads: list[dict], people: PersonaSet, rng: random.Random,
                        mode: str) -> list[Optional[dict]]:
        if self.pairing == "subject":
            missing = [p["subject_response_id"] for p in payloads if not p["subject_persona_id"]]
            if missing:
                raise ValueError(f"pairing='subject' but {len(missing)} subjects were generated without a persona")
            return [people.get(p["subject_persona_id"]) for p in payloads]
        k, everyone, assigned = self.personas_per_subject, list(people), []
        for start in range(0, len(payloads), k):
            if mode == "cycle":      # balanced: walk through the set, so every persona is used equally often
                group = [everyone[(start + j) % len(everyone)] for j in range(k)]
            elif k <= len(everyone):  # distinct evaluators for one subject
                group = rng.sample(everyone, k)
            else:
                group = [rng.choice(everyone) for _ in range(k)]
            assigned += group
        return assigned

    def template_variables(self, payload: dict, persona: Optional[dict]) -> dict[str, Any]:
        labels = self.labels(payload)
        subject = self.subjects[payload["subject_response_id"]]
        return {
            "persona": persona,
            "expectations": self.expectations_for(persona["id"] if persona else None),
            "subject_text": subject.text,
            "subject_input": payload["subject_input"],
            "context": self.context(payload),
            "rubric_form": self.rubric.render_form(labels),
            "response_format": self.rubric.render_response_format(labels),
        }

    def reference_rows(self) -> dict[str, list[dict]]:
        r = self.rubric
        return {"rubrics": [{"rubric_sha256": r.sha256, "rubric_id": r.id, "name": r.name, "version": r.version,
                             "title": r.title, "definition": r.definition()}]}

    def validation_expected(self, metadata: dict) -> Any:
        """The payload plus the evaluator's persona id (N/A rules can depend on their expectations)."""
        return {**metadata.get(INPUT_KEY, {}), "persona_id": metadata.get("persona_id")}

    def _na_required(self, expected: dict) -> Mapping[str, bool]:
        return self.na_required(expected, self.expectations_for(expected.get("persona_id")))

    def validator(self):
        check = RubricAnswerCheck(
            self.rubric, labels=self.labels,
            quoted_text=lambda expected: self.subjects[expected["subject_response_id"]].text,
            na_required=self._na_required)
        return Validator([NonEmpty(), NotTruncated(), check], name=f"{self.rubric.id}_answer")

    def derived_rows(self, response: ChatResponse, response_id: str) -> dict[str, list[dict]]:
        """
        Parse, check and score the evaluator's answer. An answer that failed validation (even
        after retries) or cannot be read produces no rows: it stays in `responses` and
        `validations` for inspection but never becomes a score.
        """
        if response.valid is False:
            return {}
        meta = response.metadata
        payload = meta[INPUT_KEY]
        persona_id = meta.get("persona_id")
        expectations = self.expectations_for(persona_id)
        labels = self.labels(payload)
        answer, problems = self.rubric.read(extract_json(response.text), labels,
                                            self.na_required(payload, expectations))
        if answer is None or any(p.severity == "error" for p in problems):
            return {}
        result = self.rubric.score(answer.scores, [f["id"] for f in answer.red_flags])
        subject = self.subjects[payload["subject_response_id"]]
        evaluation_id = new_id("evl")

        item_rows = []
        for item in self.rubric.items:
            names = labels.get(item.per, []) if item.per else []
            for n, key in enumerate(self.rubric.keys_in(item, answer.scores)):
                score = answer.scores[key]
                item_rows.append({
                    "evaluation_id": evaluation_id, "item_key": key, "item_id": item.id,
                    "section_key": self.rubric.section_of(item.id).key, "item_type": item.type,
                    "label": names[n] if n < len(names) else None, "score": score, "is_na": score is None,
                    "justification": answer.justifications.get(key) or answer.justifications.get(item.id),
                })
        flags = {f.id: f for f in self.rubric.red_flags}
        flag_rows = [{"evaluation_id": evaluation_id, "flag_id": f["id"], "severity": flags[f["id"]].severity,
                      "cap": flags[f["id"]].cap, "quote": f["quote"]} for f in answer.red_flags]
        return {
            "evaluations": [{
                "evaluation_id": evaluation_id, "response_id": response_id, "request_id": response.request_id,
                "subject_type": self.subject_type, "subject_response_id": subject.response_id,
                "subject_run_id": subject.run_id, "persona_id": persona_id,
                "evaluator_type": self.evaluator_type, "rubric_id": self.rubric.id,
                "rubric_sha256": self.rubric.sha256,
                "expectations": {k: v for k, v in expectations.items() if k != "response_id"} if expectations else None,
                "section_scores": {k: (round(v, 4) if v is not None else None)
                                   for k, v in result.section_scores.items()},
                "score_before_caps": result.score_before_caps, "cap_applied": result.cap_applied,
                "total_score": result.total_score, "grade": result.grade, "n_red_flags": len(flag_rows),
                "outputs": answer.outputs, "subject_length_words": len(subject.text.split()),
                "evaluated_at": utc_now(),
            }],
            "evaluation_item_scores": item_rows,
            "evaluation_red_flags": flag_rows,
        }
