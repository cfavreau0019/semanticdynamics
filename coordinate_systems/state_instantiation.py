"""
Instantiation now lives in the `instantiation` package, generalized from the class that was
here. It is re-exported so that existing code and notebooks keep importing it from here:

    from coordinate_systems.state_instantiation import Instantiation
    Instantiation(possible_states, step_aliases, state_step_dict).instant
"""
from instantiation.core import Instantiation

__all__ = ["Instantiation"]
