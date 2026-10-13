"""
Applications: what is built on the general packages (llm, prompts, personas, validation,
data_generation, evaluation). One folder per application, holding everything that is
specific to it; the general packages never import from here, only their command-line
scripts do, to look tasks up by name.

  tarot   Celtic Cross readings and their evaluation by personas

An application is a set of inference tasks (data_generation.task.InferenceTask), one file
per task, plus whatever only that application needs, such as its validators.

Adding an application: create applications/<name>/ with an __init__.py and a module per
task. Nothing else has to be registered. Every InferenceTask subclass with a `name` that is
defined in a module directly inside an application folder is found by tasks(), and the
general scripts then offer it:

    python -m data_generation.generate run <task> ...       # tasks with cli = "generate"
    python -m evaluation.evaluate run <task> ...            # tasks with cli = "evaluate"

What a task has to supply is described in data_generation/task.py; evaluation tasks start
from evaluation.task.EvaluationTask and evaluation.expectations.PersonaExpectations.
"""
import importlib
import inspect
import pkgutil
from functools import lru_cache


@lru_cache(maxsize=None)
def tasks() -> dict:
    """{name: class} of every task of every application. Imports each application's modules."""
    from data_generation.task import InferenceTask  # here, so that importing one module of an application stays light

    found: dict[str, type] = {}
    for application in pkgutil.iter_modules(__path__, prefix=f"{__name__}."):
        if not application.ispkg:
            continue
        package = importlib.import_module(application.name)
        for info in pkgutil.iter_modules(package.__path__, prefix=f"{application.name}."):
            module = importlib.import_module(info.name)
            for cls in vars(module).values():
                if not (inspect.isclass(cls) and issubclass(cls, InferenceTask) and cls.__module__ == module.__name__
                        and isinstance(getattr(cls, "name", None), str)):
                    continue
                if cls.name in found:
                    raise ValueError(f"Two tasks are named {cls.name!r}: {found[cls.name].__module__}."
                                     f"{found[cls.name].__name__} and {module.__name__}.{cls.__name__}")
                found[cls.name] = cls
    return found


def get_task(name: str) -> type:
    """The task class with this name (the name its runs are stored under, e.g. "tarot_celtic_cross")."""
    known = tasks()
    if name not in known:
        raise KeyError(f"No task named {name!r}; the applications define: {', '.join(sorted(known)) or 'none'}")
    return known[name]


def task_of(run: dict):
    """The task of an existing run, rebuilt from what its runs row recorded (to resume or collect the run)."""
    return get_task(run["task"]).from_options(run["config"]["task_options"])
