"""
Tarot readings: the Python of the task tarot_celtic_cross, whose task file (../tasks.yaml)
says the rest: the card draws (its instantiation), the prompt template, the model and the
validation. TarotReadings adds the draw's long form (input_items), and rebuilds the task of
runs made before the task file, to resume or collect them.

A draw is {position: state}, state = "<card>" or "<card> reversed"; the instant holds it as
`cards`. The deck (../vector_space_config.json) supplies the positions, the cards and whether
cards can be reversed.
"""
from applications.tarot.code.validation import celtic_cross_validator, parse_state
from inference.task import InferenceTask


class TarotReadings(InferenceTask):
    """
    Celtic Cross readings of card draws: what tasks.yaml cannot say.

    For a run of the task file, input_items gives the draw (the instant's `cards`) in long form,
    a row per position. A run made before the task file stored the draw itself as its input, and
    its deck, draw mode and pins as options; from_options rebuilds its task, whose readings are
    checked as they were then.
    """
    name = "tarot_celtic_cross"
    entry = "cards"                             # the instant's entry holding the draw (tasks.yaml's state)
    before_task_file = False                    # rebuilt from the options of a run made before tasks.yaml

    def input_items(self, payload: dict) -> list[dict]:
        """A row per position of the draw, the instant's `cards`."""
        items = []
        for pos, state in payload[self.entry].items():
            card, reversed_ = parse_state(state)
            items.append({"label": pos, "value": state, "entity": card,
                          "qualifier": "reversed" if reversed_ else "upright"})
        return items

    def validator(self):
        """The config's validation; for a run made before tasks.yaml, the validator its readings had."""
        return celtic_cross_validator(name=self.name) if self.before_task_file else super().validator()

    @classmethod
    def from_options(cls, options: dict) -> "TarotReadings":
        """
        The task of a run made before tasks.yaml, which recorded its deck, draw mode and pins as
        options. Its draws and prompts are stored; only its validation is needed again.
        """
        task = cls()
        task.before_task_file = True
        return task
