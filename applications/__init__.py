"""
Applications: what is built on the general packages (llm, prompts, personas, validation,
data_generation, evaluation). One folder per application, holding everything that is
specific to it; the general packages never import from here.

  tarot   Celtic Cross readings and their evaluation by personas

An application is a set of inference tasks (data_generation.task.InferenceTask), one file
per task, plus whatever only that application needs, such as its validators.
"""
