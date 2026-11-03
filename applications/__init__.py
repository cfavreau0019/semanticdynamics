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
    run: its config (InferenceTaskConfig.from_run), and the task that config describes. A run that
    recorded its task's options is a task computing its inputs in Python, rebuilt from them by its
    class; or, when no class has its name, a run made before task configs, validated as its task
    file says (see before_task_configs).
    """
    from inference.task import InferenceTask
    from inference.task_config import InferenceTaskConfig

    config = InferenceTaskConfig.from_run(run)
    options = run["config"].get("task_options")
    if options is None:
        return InferenceTask.from_config(config)
    if run["task"] in tasks():
        task = get_task(run["task"]).from_options(options)
        task.config = config
        return task
    return InferenceTask(before_task_configs(config))


def before_task_configs(config):
    """
    The config of a run made before task configs, completed from its task file. Its flat settings do
    not say how its responses were validated; its task file's validation does (custom validator and
    checks). The run stored each input as it was, not as an instant, so a response is checked against
    the whole input (no `expected` entry). Its own retries are kept.
    """
    from dataclasses import replace

    from inference.task_config import InferenceTaskConfig

    if config.validation is None:               # the run was not validated
        return config
    written = InferenceTaskConfig.from_file(task_file_of(config.name), config.name, check=False).validation
    if written is None:
        return config
    return replace(config, validation=replace(written, expected=None, retries=config.validation.retries))


def task_files() -> list[Path]:
    """Every application's tasks.yaml."""
    here = Path(__file__).resolve().parent
    return sorted(p / "tasks.yaml" for p in here.iterdir()
                  if p.is_dir() and p.name not in NOT_APPLICATIONS and (p / "tasks.yaml").is_file())


def _shown(path: Path) -> str:
    return f"applications/{path.parent.name}/{path.name}"


def _read_task_files() -> tuple[dict[str, list[Path]], list[str]]:
    """
    {task name: the files that define it}, read but not checked, and why any file could not be read.
    A bad file never raises here: it must not stand in the way of the other applications' tasks.
    """
    import yaml
    from spec_files import read_yaml

    defined: dict[str, list[Path]] = {}
    unreadable: list[str] = []
    for path in task_files():
        try:
            data = read_yaml(path)
        except (yaml.YAMLError, OSError, UnicodeDecodeError) as err:
            unreadable.append(f"{_shown(path)}: {' '.join(str(err).split())}")
            continue
        if not isinstance(data, dict) or not data:
            unreadable.append(f"{_shown(path)}: the file must map task names to tasks")
            continue
        for name in data:
            if isinstance(name, str):
                defined.setdefault(name, []).append(path)
            else:
                unreadable.append(f"{_shown(path)}: {name!r} is not a task name")
    return defined, unreadable


def _two_files_define(name: str, files: list[Path]) -> str:
    return f"Two task files define {name!r}: {' and '.join(_shown(f) for f in files)}"


def task_names() -> dict[str, Path]:
    """
    {task name: its task file} for every task of every application's tasks.yaml (read, not checked).
    A file that cannot be read and a name two files define are left out: see task_file_problems().
    """
    return {name: files[0] for name, files in _read_task_files()[0].items() if len(files) == 1}


def task_file_problems() -> list[str]:
    """What keeps tasks out of task_names(): files that cannot be read, and names two files define."""
    defined, unreadable = _read_task_files()
    return unreadable + [_two_files_define(name, files) for name, files in defined.items() if len(files) > 1]


def task_file_of(name: str) -> Path:
    """The task file that defines the task `name`. Raises KeyError saying why there isn't exactly one."""
    defined, unreadable = _read_task_files()
    files = defined.get(name, [])
    if len(files) > 1:
        raise KeyError(_two_files_define(name, files))
    if not files:
        known = sorted(n for n, f in defined.items() if len(f) == 1)
        raise KeyError(f"No task named {name!r} in the task files; they define: {', '.join(known) or 'none'}"
                       + (f". It may be in a file that could not be read: {'; '.join(unreadable)}" if unreadable else ""))
    return files[0]


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
