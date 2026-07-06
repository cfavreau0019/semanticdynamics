from random import choice
from copy import deepcopy

class Instantiation:
    def __init__(self, possible_states, step_aliases,
                 state_step_dict=None
                 ):
        self.possible_states = possible_states
        self.step_aliases = step_aliases
        self.state_step_dict = state_step_dict or {}
        self.instant = self._instantiate_random()

    def check_state_step_dict(self):
        """
        Checks if the state step dictionary is valid
        Should be {'step_alias':'state'}
        """
        if self.state_step_dict is None:
            return

        assert len([i for i in self.state_step_dict.keys() if i not in self.step_aliases])==0
        assert len([i for i in self.state_step_dict.values() if i not in self.possible_states])==0

    def _instantiate_random(self):
        instant = deepcopy(self.state_step_dict)
        choose = choice
        steps = self.step_aliases
        for step in steps:
            if step not in instant.keys():
                instant[step] = choose(self.possible_states)
        return instant