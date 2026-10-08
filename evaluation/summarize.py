"""
Evaluation summary: an LLM-written report of what the evaluations of one task say.

    python -m evaluation.summarize --task tarot_celtic_cross
    python -m evaluation.summarize --task tarot_celtic_cross --evaluating-model gpt-6-sol --by generating_model
    python -m evaluation.summarize --task tarot_celtic_cross --each evaluating_model     # one summary per model
    python -m evaluation.summarize --task tarot_celtic_cross --dry-run                   # print the prompt only

The evaluations of the task are selected with the same criteria as in the dashboard (prompt
template, generating model, evaluating model, persona, evaluation run), or by the generation
dataset their texts belong to (--readings-dataset). Their rubric results
are rendered as text blocks: scores as tables, and a sample of the evaluators' comments per
rubric question. Those blocks are the variables of a prompt template (prompts/library/
evaluation/evaluation_summary_v1.toml), which one LLM call turns into the summary.

Written to <evaluations-root>/summaries/: the summary as Markdown, and next to it a JSON file
with the exact prompt, the selection, the model and the token usage.

`python evaluation/summarize.py ...` works too.
"""
import argparse
import json
import random
import re
import statistics
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

if __package__ in (None, ""):  # allow `python evaluation/summarize.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_generation.store import utc_now  # noqa: E402
from evaluation.dashboard import EVALUATIONS_ROOT, build_data  # noqa: E402
from llm import ChatRequest  # noqa: E402
from llm.inference import chat  # noqa: E402
from prompts import get_template  # noqa: E402

DEFAULT_PROVIDER, DEFAULT_MODEL = "openai", "gpt-6-luna"
DEFAULT_TEMPLATE = "evaluation_summary_v1"

# The criteria evaluations are selected and grouped by: name -> (label, value of an evaluation).
DIMENSIONS: dict[str, tuple[str, Callable[[dict, dict], Any]]] = {
    "template": ("prompt template", lambda e, subject: subject.get("template")),
    "generating_model": ("generating model", lambda e, subject: subject.get("model")),
    "evaluating_model": ("evaluating model", lambda e, subject: e["ev"]),
    "persona": ("persona", lambda e, subject: e["pid"]),
    "run": ("evaluation run", lambda e, subject: e["run"]),
    "readings_dataset": ("readings dataset", lambda e, subject: subject.get("dataset")),
}


# ---- selection ------------------------------------------------------------------------------
def value_of(data: dict, e: dict, dimension: str) -> Any:
    return DIMENSIONS[dimension][1](e, data["subjects"][e["sid"]])


def label_of(data: dict, dimension: str, value: Any) -> str:
    if dimension == "persona":
        return data["personas"][value]["name"] if value else "no persona"
    return str(value) if value not in (None, "") else "(unknown)"


def _matches(data: dict, dimension: str, value: Any, wanted: Sequence[str]) -> bool:
    if dimension == "persona" and value:       # a persona can be named by id, number or name
        persona = data["personas"][value]
        names = {str(value), str(persona.get("number")), str(persona["name"]).casefold()}
        return any(w in names or w.casefold() in names for w in wanted)
    return str(value) in wanted


def select(data: dict, task: str, filters: Optional[Mapping[str, Sequence[str]]] = None) -> list[dict]:
    """The evaluations of `task` that match every filter ({dimension: accepted values}; empty = all)."""
    unknown = set(filters or {}) - set(DIMENSIONS)
    if unknown:
        raise ValueError(f"Unknown dimensions {sorted(unknown)}; choose from {list(DIMENSIONS)}")
    return [e for e in data["evaluations"]
            if data["subjects"][e["sid"]]["task"] == task
            and all(not wanted or _matches(data, d, value_of(data, e, d), wanted) for d, wanted in (filters or {}).items())]


# ---- rendering the results as text ----------------------------------------------------------
def _f(value: Optional[float], digits: int = 2) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def _mean(values: list) -> Optional[float]:
    return statistics.fmean(values) if values else None


def _sd(values: list) -> Optional[float]:
    return statistics.stdev(values) if len(values) > 1 else None


def _table(header: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return "\n".join(lines + ["| " + " | ".join(str(c).replace("|", "/").replace("\n", " ") for c in row) + " |"
                              for row in rows])


def _pretty(name: str) -> str:
    return name.replace("_", " ").capitalize()


def _one_line(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


class _Rubric:
    """Lookups over a rubric definition as stored with the evaluation runs."""

    def __init__(self, definition: dict):
        self.d = definition
        self.sections = definition["sections"]
        self.items = [dict(item, section=s) for s in self.sections for item in s["items"]]
        self.by_id = {i["id"]: i for i in self.items}
        self.scale = (definition["scale_min"], definition["scale_max"])

    def item_of(self, ref: Any) -> Optional[dict]:
        """The item a score key or a free-text reference names: '3.2', '2a.3', '2a (positions 1-2)'."""
        ref = str(ref or "").strip()
        return (self.by_id.get(ref) or self.by_id.get(ref.rpartition(".")[0])
                or self.by_id.get(re.split(r"[\s:(]", ref)[0]))

    def scores(self, e: dict, item_id: str) -> list[int]:
        return [v for k, v in e["s"].items() if v is not None and (self.item_of(k) or {}).get("id") == item_id]


def rubric_overview(rubric: _Rubric) -> str:
    d = rubric.d
    lines = [f"{d['title']} ({d['id']}). {_one_line(d.get('description') or '')}", "",
             f"Scale {rubric.scale[0]}-{rubric.scale[1]}: " +
             "; ".join(f"{a['score']} = {a['label']}" for a in sorted(d.get("anchors") or [], key=lambda a: -a["score"])) + ".",
             "Question types: O = objective (should not depend on who evaluates), S = subjective (should).",
             "The total (0-100) is the weighted sum of section averages; a red flag caps it.", "", "Sections:"]
    lines += [f"- {s['id']}. {s['title']} (weight {s['weight']:g}): {_one_line(s['question'])}" for s in rubric.sections]
    if d.get("grades"):
        lines += ["", "Grades: " + ", ".join(f"{g['grade']} from {g['min']:g}" for g in
                                              sorted(d["grades"], key=lambda g: -g["min"])) + "."]
    return "\n".join(lines)


def quantitative_results(data: dict, evaluations: list[dict], rubric: _Rubric) -> str:
    totals = [e["total"] for e in evaluations]
    grades = Counter(e["grade"] for e in evaluations)
    order = [g["grade"] for g in sorted(rubric.d.get("grades") or [], key=lambda g: -g["min"])] or sorted(grades)
    flagged = sum(bool(e["flags"]) for e in evaluations)
    out = ["## Overall",
           f"- Total score (0-100): mean {_f(_mean(totals), 1)}, median {_f(statistics.median(totals), 1)}, "
           f"SD {_f(_sd(totals), 1)}, range {min(totals):.1f}-{max(totals):.1f}, over {len(totals)} evaluations",
           "- Grades: " + ", ".join(f"{g} {grades.get(g, 0)}" for g in order),
           f"- Evaluations with a red flag: {flagged} ({100 * flagged / len(evaluations):.0f}%); "
           f"total capped in {sum(e['cap'] is not None for e in evaluations)}", "",
           f"## Sections (average on the {rubric.scale[0]}-{rubric.scale[1]} scale)"]
    rows = []
    for s in rubric.sections:
        values = [e["sec"][s["key"]] for e in evaluations if e["sec"].get(s["key"]) is not None]
        rows.append([f"{s['id']}. {s['title']}", f"{s['weight']:g}", _f(_mean(values)), _f(_sd(values))])
    out += [_table(["Section", "Weight", "Mean", "SD"], rows), "", "## Questions"]

    points = list(range(rubric.scale[0], rubric.scale[1] + 1))
    rows, by_label = [], []
    for item in rubric.items:
        values = [v for e in evaluations for v in rubric.scores(e, item["id"])]
        counts = Counter(values)
        n_na = sum(v is None for e in evaluations for k, v in e["s"].items()
                   if (rubric.item_of(k) or {}).get("id") == item["id"])
        rows.append([item["id"], item["type"], _one_line(item["text"]), len(values), _f(_mean(values)), _f(_sd(values)),
                     *[counts.get(p, 0) for p in points], n_na])
        if item.get("per"):                      # scored once per label, e.g. per spread position
            per_label: dict[str, list[int]] = {}
            for e in evaluations:
                labels = data["subjects"][e["sid"]].get("labels") or {}
                for k, v in e["s"].items():
                    if v is not None and k != item["id"] and (rubric.item_of(k) or {}).get("id") == item["id"]:
                        per_label.setdefault(labels.get(k, k), []).append(v)
            by_label.append(f"- {item['id']} by {item['per']}: " +
                            ", ".join(f"{label} {_f(_mean(v))}" for label, v in per_label.items()))
    out.append(_table(["Id", "Type", "Question", "n", "Mean", "SD", *[f"Scored {p}" for p in points], "N/A"], rows))
    out += ["", *by_label] if by_label else []

    ratings = []
    for o in rubric.d.get("outputs") or []:
        values = [e["out"][o["name"]] for e in evaluations if e["out"].get(o["name"]) is not None]
        if o["type"] == "integer" and values:
            ratings.append(f"- {_pretty(o['name'])} ({o['min']}-{o['max']}; {_one_line(o.get('description') or '')}): "
                           f"mean {_f(_mean(values))}, SD {_f(_sd(values))}, n {len(values)}")
        elif o["type"] == "enum" and values:
            counts = Counter(values)
            ratings.append(f"- {_pretty(o['name'])}: " + ", ".join(f"{c} {counts.get(c, 0)}" for c in o["choices"]))
    if ratings:
        out += ["", "## The evaluators' own ratings", *ratings]

    fired = Counter(f["id"] for e in evaluations for f in e["flags"])
    if fired:
        flags = {f["id"]: f for f in rubric.d.get("red_flags") or []}
        out += ["", "## Red flags fired"]
        out += [f"- {fid} ({flags[fid]['severity']}, caps the total at {flags[fid]['cap']:g}): "
                f"{_one_line(flags[fid]['text'])} Fired in {n} evaluations." for fid, n in fired.most_common()]
    return "\n".join(out)


def breakdown(data: dict, evaluations: list[dict], rubric: _Rubric, by: Sequence[str], max_groups: int = 15) -> str:
    """One table per dimension in `by`: n, total and section averages of each of its values."""
    out = []
    for dimension in by:
        groups: dict[Any, list[dict]] = {}
        for e in evaluations:
            groups.setdefault(value_of(data, e, dimension), []).append(e)
        ranked = sorted(groups.items(), key=lambda kv: -len(kv[1]))
        rows = []
        for value, group in ranked[:max_groups]:
            totals = [e["total"] for e in group]
            sections = [_f(_mean([e["sec"][s["key"]] for e in group if e["sec"].get(s["key"]) is not None]))
                        for s in rubric.sections]
            rows.append([label_of(data, dimension, value), len(group), len({e["sid"] for e in group}),
                         _f(_mean(totals), 1), _f(_sd(totals), 1), *sections])
        out += [f"## By {DIMENSIONS[dimension][0]}",
                _table([_pretty(DIMENSIONS[dimension][0]), "Evaluations", "Texts", "Mean total", "SD",
                        *[f"{s['id']}. {s['title']}" for s in rubric.sections]], rows)]
        if len(ranked) > max_groups:
            out.append(f"(The {max_groups} largest of {len(ranked)} groups.)")
        out.append("")
    return "\n".join(out).strip()


def _sample(comments: list[dict], k: int, rng: random.Random) -> list[dict]:
    """Up to k comments, spread over the scores they came with (the extremes first)."""
    if len(comments) <= k:
        return comments
    by_score: dict[Any, list[dict]] = {}
    for c in comments:
        by_score.setdefault(c.get("score"), []).append(c)
    for group in by_score.values():
        rng.shuffle(group)
    known = sorted(s for s in by_score if s is not None)
    order = [s for pair in zip(known, reversed(known)) for s in pair][:len(known)] + [s for s in by_score if s is None]
    picked: list[dict] = []
    while len(picked) < k:
        for score in order:
            if by_score[score] and len(picked) < k:
                picked.append(by_score[score].pop())
    return picked


def _who(data: dict, e: dict, by: Sequence[str]) -> str:
    extra = [f"{DIMENSIONS[d][0]} {label_of(data, d, value_of(data, e, d))}" for d in by
             if d not in ("persona", "evaluating_model")]
    return "; ".join([label_of(data, "persona", e["pid"]), e["ev"], *extra])


def feedback(data: dict, evaluations: list[dict], rubric: _Rubric, by: Sequence[str] = (),
             comments_per_question: int = 6, comments_overall: int = 20, seed: int = 0) -> tuple[str, str]:
    """
    (question_feedback, overall_feedback): the evaluators' comments as text.

    Question feedback is every comment tied to a rubric question: justifications of scores, and
    list outputs that name an item (strengths, improvements). Overall feedback is the rest: text
    outputs (reactions), plain list outputs (counted), and the quotes behind red flags.
    """
    rng = random.Random(seed)
    per_item: dict[str, list[dict]] = {}
    texts: dict[str, list[dict]] = {}
    counted: dict[str, Counter] = {}
    quotes: list[str] = []
    outputs = rubric.d.get("outputs") or []
    for e in evaluations:
        who, labels = _who(data, e, by), data["subjects"][e["sid"]].get("labels") or {}
        for key, text in e["j"].items():
            item = rubric.item_of(key)
            if item:
                score = e["s"].get(key)          # None for a comment on a whole per-label item
                about = f"scored {score}" + (f" on {labels[key]}" if key in labels else "") if score is not None else "comment"
                per_item.setdefault(item["id"], []).append({"score": score, "line": f"- ({about}; {who}) {_one_line(text)}"})
        for o in outputs:
            value = e["out"].get(o["name"])
            if o["type"] == "text" and value:
                texts.setdefault(o["name"], []).append(
                    {"score": e["grade"], "line": f"- (gave {e['total']:.1f}, grade {e['grade']}; {who}) {_one_line(value)}"})
            elif o["type"] == "list":
                for entry in value or []:
                    if isinstance(entry, dict):
                        item = rubric.item_of(entry.get("item"))
                        text = " - ".join(_one_line(v) for k, v in entry.items() if k != "item")
                        line = {"score": None, "line": f"- ({_pretty(o['name'])[:-1].lower()}; {who}) {text}"}
                        if item:
                            per_item.setdefault(item["id"], []).append(line)
                        else:
                            texts.setdefault(o["name"], []).append(line)
                    else:
                        counted.setdefault(o["name"], Counter())[_one_line(entry)] += 1
        quotes += [f'- {f["id"]} ({who}): "{_one_line(f["quote"])}"' for f in e["flags"]]

    by_question = []
    for s in rubric.sections:
        block = []
        for item in s["items"]:
            comments = per_item.get(item["id"], [])
            if not comments:
                continue
            values = [v for e in evaluations for v in rubric.scores(e, item["id"])]
            block += [f"### {item['id']} [{item['type']}] {_one_line(item['text'])} "
                      f"(mean {_f(_mean(values))}; {min(len(comments), comments_per_question)} of {len(comments)} comments)",
                      *[c["line"] for c in _sample(comments, comments_per_question, rng)], ""]
        if block:
            by_question += [f"## Section {s['id']}. {s['title']}", "", *block]

    overall = []
    for name, comments in texts.items():
        overall += [f"## {_pretty(name)} ({min(len(comments), comments_overall)} of {len(comments)})",
                    *[c["line"] for c in _sample(comments, comments_overall, rng)], ""]
    for name, counts in counted.items():
        overall += [f"## {_pretty(name)} (how many evaluators named each)",
                    *[f"- {text} (x{n})" for text, n in counts.most_common(comments_overall)], ""]
    if quotes:
        rng.shuffle(quotes)
        overall += [f"## Passages quoted for red flags ({min(len(quotes), comments_overall)} of {len(quotes)})",
                    *quotes[:comments_overall], ""]
    return ("\n".join(by_question).strip() or "No evaluator commented on a specific question.",
            "\n".join(overall).strip() or "No overall comments were recorded.")


def scope(data: dict, evaluations: list[dict], rubric: _Rubric, filters: Mapping[str, Sequence[str]]) -> str:
    lines = [f"- {len(evaluations)} evaluations of {len({e['sid'] for e in evaluations})} texts, "
             f"scored with the rubric {rubric.d['id']}"]
    for dimension, (label, _) in DIMENSIONS.items():
        values = Counter(label_of(data, dimension, value_of(data, e, dimension)) for e in evaluations)
        shown = ", ".join(f"{v} ({n})" for v, n in values.most_common(8)) + (", ..." if len(values) > 8 else "")
        only = " (selected)" if filters.get(dimension) else ""
        lines.append(f"- {_pretty(label)}{only}: {len(values)} - {shown}" if len(values) > 1 else
                     f"- {_pretty(label)}{only}: {shown}")
    return "\n".join(lines)


def template_variables(data: dict, evaluations: list[dict], task: str, filters: Optional[Mapping] = None,
                       by: Sequence[str] = (), focus: Optional[str] = None, comments_per_question: int = 6,
                       comments_overall: int = 20, seed: int = 0) -> dict[str, Any]:
    """The variables of the summary prompt for these evaluations (which must share one rubric)."""
    rubrics = {e["r"] for e in evaluations}
    if len(rubrics) != 1:
        raise ValueError(f"The selection has {len(rubrics)} rubrics; a summary covers exactly one")
    rubric = _Rubric(data["rubrics"][rubrics.pop()])
    by_question, overall = feedback(data, evaluations, rubric, by, comments_per_question, comments_overall, seed)
    return {"task": task, "scope": scope(data, evaluations, rubric, filters or {}),
            "rubric_overview": rubric_overview(rubric),
            "quantitative_results": quantitative_results(data, evaluations, rubric),
            "question_feedback": by_question, "overall_feedback": overall,
            "breakdown": breakdown(data, evaluations, rubric, by) if by else None, "focus": focus or None}


# ---- the LLM call ---------------------------------------------------------------------------
def summarize(variables: Mapping[str, Any], provider: str = DEFAULT_PROVIDER, model: Optional[str] = DEFAULT_MODEL,
              template: str = DEFAULT_TEMPLATE, **params) -> dict:
    """Render the prompt and ask the model. Returns the summary with the prompt and usage that made it."""
    prompt = get_template(template).render(variables)
    response = chat(ChatRequest(messages=prompt.messages, model=model, params=params), provider)
    if response.error or not response.text:
        raise RuntimeError(f"The model returned no summary: {response.error or response.finish_reason}")
    usage = response.usage
    return {"summary": response.text.strip(), "provider": provider, "model": response.model or model,
            "finish_reason": response.finish_reason, "template": prompt.template_id,
            "template_sha256": prompt.template_sha256, "params": params, "messages": prompt.messages,
            "usage": {"prompt_tokens": usage.prompt_tokens, "completion_tokens": usage.completion_tokens,
                      "total_tokens": usage.total_tokens, "cached_tokens": usage.cached_tokens} if usage else None}


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_")[:40]


def write_summary(out_dir: Path, result: dict, task: str, variables: Mapping[str, Any], selection: dict,
                  suffix: str = "") -> Path:
    """<out_dir>/summary_<task>[_<suffix>]_<time>.md, and the same name .json with how it was made."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"summary_{'_'.join(filter(None, [_slug(task), _slug(suffix)]))}_{stamp}.md"
    path.write_text(
        f"# Evaluation summary: {task}\n\n{variables['scope']}\n"
        f"- Written by {result['provider']}/{result['model']} with the prompt {result['template']}, {utc_now()[:10]}\n\n"
        f"---\n\n{result['summary']}\n", encoding="utf-8")
    path.with_suffix(".json").write_text(json.dumps(
        {"created_at": utc_now(), "task": task, **selection, **result}, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


# ---- command line ---------------------------------------------------------------------------
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="summarize", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", required=True, help="the task whose evaluations to summarise: the application of "
                                                      "the generation runs, e.g. tarot_celtic_cross")
    parser.add_argument("--template", action="append", default=[], metavar="NAME",
                        help="only texts generated with this prompt template; repeatable")
    parser.add_argument("--generating-model", action="append", default=[], metavar="MODEL", help="repeatable")
    parser.add_argument("--evaluating-model", action="append", default=[], metavar="MODEL", help="repeatable")
    parser.add_argument("--persona", action="append", default=[], metavar="ID",
                        help="only evaluations by this persona (id, number or full name); repeatable")
    parser.add_argument("--run", action="append", default=[], metavar="RUN_ID", help="only this evaluation run; repeatable")
    parser.add_argument("--readings-dataset", action="append", default=[], metavar="NAME",
                        help="only texts of this generation dataset, e.g. tarot-v4; repeatable")
    parser.add_argument("--dataset", help="only evaluation runs labelled with this dataset when they were run "
                                          "(evaluate --dataset); not the dataset of the texts, see --readings-dataset")
    parser.add_argument("--by", action="append", default=[], choices=list(DIMENSIONS), metavar="DIMENSION",
                        help=f"also compare the scores across this dimension in the summary; repeatable. "
                             f"One of: {', '.join(DIMENSIONS)}")
    parser.add_argument("--each", choices=list(DIMENSIONS), metavar="DIMENSION",
                        help="write one separate summary for each value of this dimension (one LLM call each)")
    parser.add_argument("--focus", help="an extra instruction for the summary, e.g. a question you want answered")
    parser.add_argument("--provider", default=DEFAULT_PROVIDER, help="llm provider (default: %(default)s)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="model that writes the summary (default: %(default)s)")
    parser.add_argument("--max-tokens", type=int, default=4000, help="default: %(default)s")
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--prompt", default=DEFAULT_TEMPLATE, metavar="TEMPLATE",
                        help="summary prompt template (default: %(default)s)")
    parser.add_argument("--comments-per-question", type=int, default=6,
                        help="comments shown to the model for each rubric question (default: %(default)s)")
    parser.add_argument("--comments-overall", type=int, default=20,
                        help="reactions and other overall comments shown, per kind (default: %(default)s)")
    parser.add_argument("--seed", type=int, default=0, help="seed for sampling comments (default: %(default)s)")
    parser.add_argument("--evaluations-root", default=str(EVALUATIONS_ROOT),
                        help="where the evaluation runs are (default: %(default)s)")
    parser.add_argument("--readings-root", help="where the generation runs are (default: where each evaluation "
                                                "run found them)")
    parser.add_argument("--out", help="directory for the summaries (default: <evaluations-root>/summaries)")
    parser.add_argument("--dry-run", action="store_true", help="print the prompt and its size; no API call")
    args = parser.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):      # comments and summaries hold any character; a legacy console doesn't
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")

    data = build_data(args.evaluations_root, runs=args.run, dataset=args.dataset, readings_root=args.readings_root,
                      tasks=[args.task])
    filters = {"template": args.template, "generating_model": args.generating_model,
               "evaluating_model": args.evaluating_model, "persona": args.persona, "run": args.run,
               "readings_dataset": args.readings_dataset}
    evaluations = select(data, args.task, filters)
    if not evaluations:
        everything = build_data(args.evaluations_root, readings_root=args.readings_root)
        print(f"error: no evaluations of task {args.task!r} match the selection. Tasks with evaluations: "
              f"{', '.join(sorted({s['task'] for s in everything['subjects'].values()})) or 'none'}", file=sys.stderr)
        if args.dataset:
            labels = sorted({r["dataset"] for r in everything["runs"] if r["dataset"]})
            of_texts = sorted({s["dataset"] for s in everything["subjects"].values() if s["dataset"]})
            print(f"note: --dataset selects evaluation runs by their own label ({', '.join(labels) or 'none are labelled'})."
                  + (f" {args.dataset!r} is a dataset of the evaluated texts: use --readings-dataset {args.dataset}"
                     if args.dataset in of_texts else ""), file=sys.stderr)
        return 1

    parts = [("", evaluations, filters)]
    if args.each:
        values = Counter(value_of(data, e, args.each) for e in evaluations)
        parts = [(label_of(data, args.each, v), [e for e in evaluations if value_of(data, e, args.each) == v],
                  {**filters, args.each: [label_of(data, args.each, v)]}) for v, _ in values.most_common()]
    params = {"max_tokens": args.max_tokens, **({"temperature": args.temperature} if args.temperature is not None else {})}
    out_dir = Path(args.out) if args.out else Path(args.evaluations_root) / "summaries"
    for suffix, part, part_filters in parts:
        heading = f"{args.task}{' / ' + suffix if suffix else ''}: {len(part)} evaluations"
        try:
            variables = template_variables(data, part, args.task, part_filters, args.by, args.focus,
                                           args.comments_per_question, args.comments_overall, args.seed)
        except ValueError as e:
            print(f"error: {heading}: {e}; narrow the selection (e.g. --run)", file=sys.stderr)
            return 1
        prompt = get_template(args.prompt).render(variables)
        size = sum(len(m["content"]) for m in prompt.messages)
        if args.dry_run:
            print(f"===== {heading} | prompt {prompt.template_id}, about {size // 4:,} tokens =====")
            for m in prompt.messages:
                print(f"\n----- {m['role']} -----\n{m['content']}")
            continue
        print(f"{heading}; asking {args.provider}/{args.model} (prompt of about {size // 4:,} tokens) ...")
        result = summarize(variables, args.provider, args.model, args.prompt, **params)
        selection = {"filters": {k: list(v) for k, v in part_filters.items() if v}, "by": args.by, "dataset": args.dataset,
                     "n_evaluations": len(part), "n_texts": len({e["sid"] for e in part}),
                     "comments_per_question": args.comments_per_question, "comments_overall": args.comments_overall,
                     "seed": args.seed}
        path = write_summary(out_dir, result, args.task, variables, selection, suffix)
        print(f"\n{result['summary']}\n\n-> {path}"
              f"{'  (' + str(result['usage']['total_tokens']) + ' tokens)' if result['usage'] else ''}")
        if result["finish_reason"] == "length":
            print("note: the summary was cut off at --max-tokens; raise it and run again", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
