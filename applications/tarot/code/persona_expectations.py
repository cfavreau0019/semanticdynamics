"""
Persona expectations for tarot: stage 1 of the Celtic Cross evaluation. Before seeing any
reading, each persona records what they expect from one (Part A3 of the Celtic Cross
Reading Evaluation Form). The mechanics are evaluation.expectations.PersonaExpectations;
this file is what tarot adds to it.

Prompt: prompts/library/evaluation/reading_expectations_v1.toml, which takes `reading_type`
and an optional `question` besides the persona.
"""
from evaluation.expectations import PersonaExpectations

# What the persona is told they are about to receive. The readings were generated without a
# question, so there is none here either.
EXPECTATION_CONTEXT = {"reading_type": "Celtic Cross tarot reading"}


class TarotPersonaExpectations(PersonaExpectations):
    """What each persona expects from a Celtic Cross reading, recorded before they see one."""
    name = "persona_expectations"
    default_template = persona_template = "reading_expectations_v1"
    default_context = EXPECTATION_CONTEXT
