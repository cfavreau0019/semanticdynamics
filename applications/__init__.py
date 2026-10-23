"""
Applications: what is built on the general packages (llm, prompts, personas, validation,
inference, evaluation). One folder per application, holding everything that is
specific to it; the general packages never import from here, only their command-line
scripts do, to look tasks up by name.

  tarot      Celtic Cross readings and their evaluation by personas
  template   files to copy when starting an application; not an application itself

An application is a set of inference tasks, described in its tasks.yaml (the task file
format: template/inference_task_reference.yaml), plus whatever only that application needs,
such as its validators and any Python a task file names (implemented_by).

Adding an application: create applications/<name>/ with an __init__.py and a tasks.yaml.
Nothing else has to be registered: the general script runs every task of every task file,

    python -m inference.generate run <task> ...

and Python of the application's own, if it has any (e.g. in a code/ package), is found too.
Every InferenceTask subclass with a `name`, defined in any module of an application's package,
is found by tasks(): the evaluation tasks (cli = "evaluate", run by `python -m
evaluation.evaluate run <task>`), and the classes task files name. What a class may supply is
described in inference/task.py; evaluation tasks start from evaluation.task.EvaluationTask and
evaluation.expectations.PersonaExpectations.
"""
import importlib
import inspect
import pkgutil
from functools import lru_cache
from pathlib import Path


NOT_APPLICATIONS = {"template"}        # folders here that are not applications, so are never searched for tasks


@lru_cache(maxsize=None)
def tasks() -> dict:
    """{name: class} of every task of every application. Imports every module of each application."""
    from inference.task import InferenceTask  # here, so that importing one module of an application stays light

    found: dict[str, type] = {}
    for application in pkgutil.iter_modules(__path__, prefix=f"{__name__}."):
        if not application.ispkg or application.name.rpartition(".")[2] in NOT_APPLICATIONS:
            continue
        package = importlib.import_module(application.name)
        for info in pkgutil.walk_packages(package.__path__, prefix=f"{application.name}."):
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
    """
    The task of an existing run, rebuilt from what its runs row recorded, to resume or collect the
    run: its config (InferenceTaskConfig.from_run), and the task that config describes, or, for a
    run that recorded its task's options (a task computing its inputs in Python, a run before task
    configs), its class rebuilt from them.
    """
    from inference.task import InferenceTask
    from inference.task_config import InferenceTaskConfig

    config = InferenceTaskConfig.from_run(run)
    options = run["config"].get("task_options")
    if options is None:
        return InferenceTask.from_config(config)
    task = get_task(run["task"]).from_options(options)
    task.config = config
    return task


def task_files() -> list[Path]:
    """Every application's tasks.yaml."""
    here = Path(__file__).resolve().parent
    return sorted(p / "tasks.yaml" for p in here.iterdir()
                  if p.is_dir() and p.name not in NOT_APPLICATIONS and (p / "tasks.yaml").is_file())


def task_names() -> dict[str, Path]:
    """{task name: its task file} for every task of every application's tasks.yaml (read, not checked)."""
    from spec_files import read_yaml

    found: dict[str, Path] = {}
    for path in task_files():
        for name in read_yaml(path) or {}:
            if name in found:
                raise ValueError(f"Two task files define {name!r}: {found[name]} and {path}")
            found[name] = path
    return found


def task_file_of(name: str) -> Path:
    """The task file that defines the task `name`."""
    known = task_names()
    if name not in known:
        raise KeyError(f"No task named {name!r} in the task files; they define: {', '.join(sorted(known)) or 'none'}")
    return known[name]


@lru_cache(maxsize=None)
def task_configs() -> dict:
    """{name: InferenceTaskConfig} of every task in every application's tasks.yaml, checked (inference.task_config)."""
    from inference.task_config import load_task_file

    found: dict = {}
    for path in task_files():
        for name, config in load_task_file(path).items():
            if name in found:
                raise ValueError(f"Two task files define {name!r}: {found[name].source} and {path}")
            found[name] = config
    return found
