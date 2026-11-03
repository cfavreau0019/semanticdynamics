"""
Instantiation: the possible underlying states of a task, and their realization.

A task's state is broader than its prompt. Each input of a run is one resolved state, its
`instant`: prompt templates take their variables from it, validation checks responses against
it, and later analysis can use the rest. Nothing here knows about inference or any application;
applications write instantiations (in a task file, or a file of their own), and the inference
engine samples a run's instants from them.

  spec                  InstantiationSpec and its entries (Literal, OneOf, Steps, Personas, Generate),
                        read and checked from a task file's `instantiation` or an instantiation file
  core                  Instantiation: one realized state; Instantiation.sample(spec, n, seed) for a run,
                        Instantiation.of(values, spec) for values already chosen
  instantiation.schema.yaml   the format, as JSON Schema

    from instantiation import Instantiation, read_instantiation
    run = Instantiation.sample(read_instantiation("instantiation.yaml"), n=20, seed=42)
    [i.instant for i in run]
"""
from instantiation.core import Instantiation, candidates, draw_steps, fixed_size, probes, resolve
from instantiation.spec import (
    Entry, Generate, InstantiationError, InstantiationSpec, Literal, OneOf, Personas, Steps, load_instantiation,
    read_instantiation,
)

__all__ = [
    "Instantiation", "candidates", "draw_steps", "fixed_size", "probes", "resolve",
    "Entry", "Generate", "InstantiationError", "InstantiationSpec", "Literal", "OneOf", "Personas", "Steps",
    "load_instantiation", "read_instantiation",
]
