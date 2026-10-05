"""
Rubrics: the structure every evaluation follows, independent of what is being evaluated.

A Rubric is a versioned data file (evaluation/rubrics/<name>_v<N>.toml) with
  * a scoring scale with anchors
  * weighted sections of items, each tagged O (objective: same answer whoever evaluates)
    or S (subjective: depends on the evaluator). An item with `per = "<labels>"` is scored
    once per label supplied at evaluation time (e.g. once per spread position)
  * red flags: yes/no checks that cap the total
  * grade bands
  * qualitative outputs the evaluator writes

The evaluator (an LLM, or a person) supplies only item scores, justifications, red flags
and the qualitative outputs. Everything numeric is computed here, so totals are exact:

    total = min(cap, sum over sections of  weight * section_average / scale.max)

Three things are generated from the same file, so they cannot drift apart:
    rubric.render_form(labels)             the form text put in the evaluation prompt
    rubric.render_response_format(labels)  the JSON the evaluator must return
    rubric.read(data, labels)              validation + normalisation of that JSON
"""
import hashlib
import json
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

RUBRICS_DIR = Path(__file__).resolve().parent / "rubrics"
NA = "N/A"
OBJECTIVE, SUBJECTIVE = "O", "S"

Labels = Mapping[str, Sequence[str]]


class RubricError(ValueError):
    pass


@dataclass(frozen=True)
class Item:
    id: str
    text: str
    type: str                        # "O" or "S"
    allow_na: bool = False
    na_when: Optional[str] = None    # human-readable condition under which N/A applies
    per: Optional[str] = None        # name of a label list: one score per label


@dataclass(frozen=True)
class Section:
    id: str
    key: str
    title: str
    weight: float
    question: str
    items: tuple[Item, ...]
    note: Optional[str] = None


@dataclass(frozen=True)
class RedFlag:
    id: str
    text: str
    severity: str
    cap: float


@dataclass(frozen=True)
class Grade:
    grade: str
    min: float
    meaning: str = ""


@dataclass(frozen=True)
class Output:
    name: str
    type: str                        # text | integer | enum | list
    description: str = ""
    min: Optional[int] = None
    max: Optional[int] = None
    choices: tuple[str, ...] = ()
    max_items: Optional[int] = None
    fields: tuple[str, ...] = ()     # for lists of objects; empty = list of strings


@dataclass
class Problem:
    """A defect in an evaluator's answer. severity: 'error' (unusable) or 'warning'."""
    code: str
    message: str
    severity: str = "error"
    item: Optional[str] = None


@dataclass
class Answer:
    """An evaluator's answer after normalisation."""
    scores: dict[str, Optional[int]]            # item key -> score, None = N/A. Per-label items: "2a.1", "2a.2", ...
    justifications: dict[str, str]
    red_flags: list[dict[str, str]]             # [{"id", "quote"}]
    outputs: dict[str, Any]


@dataclass
class ScoreResult:
    section_scores: dict[str, Optional[float]]  # section key -> average on the scale (None if nothing scored)
    section_points: dict[str, float]            # section key -> points contributed to the total
    score_before_caps: float
    cap_applied: Optional[float]
    total_score: float                          # rounded to one decimal
    grade: str


@dataclass(frozen=True)
class Rubric:
    name: str
    version: int
    title: str
    description: str
    scale_min: int
    scale_max: int
    anchors: tuple[dict, ...]
    justify: tuple[int, ...]
    na_note: str
    sections: tuple[Section, ...]
    red_flags: tuple[RedFlag, ...]
    grades: tuple[Grade, ...]
    outputs: tuple[Output, ...]
    source: Optional[str] = None
    path: Optional[Path] = field(default=None, compare=False)
    sha256: str = field(default="", compare=False)

    # ---- identity / lookup ----------------------------------------------------------
    @property
    def id(self) -> str:
        return f"{self.name}_v{self.version}"

    @property
    def items(self) -> list[Item]:
        return [i for s in self.sections for i in s.items]

    def item(self, item_id: str) -> Item:
        for i in self.items:
            if i.id == item_id:
                return i
        raise KeyError(item_id)

    def section_of(self, item_id: str) -> Section:
        return next(s for s in self.sections if any(i.id == item_id for i in s.items))

    def score_keys(self, item: Item, labels: Labels) -> list[str]:
        """Keys under which an item's scores are stored: ['3.1'] or ['2a.1', ..., '2a.10']."""
        if item.per is None:
            return [item.id]
        if item.per not in labels:
            raise RubricError(f"{self.id}: item {item.id} needs labels for {item.per!r}")
        return [f"{item.id}.{n}" for n in range(1, len(labels[item.per]) + 1)]

    def definition(self) -> dict[str, Any]:
        """The rubric as plain data (stored in the `rubrics` table)."""
        from dataclasses import asdict
        d = asdict(self)
        d.pop("path"), d.pop("sha256")
        return d

    # ---- rendering for prompts ------------------------------------------------------
    def render_form(self, labels: Labels) -> str:
        lines = [f"## Scoring scale ({self.scale_min}-{self.scale_max}, whole numbers only)", ""]
        for a in self.anchors:
            lines.append(f"- {a['score']} = {a['label']}: {a['anchor']}")
        if self.justify:
            lines.append(f"\nGive a one-line justification for any score of {_join_or(self.justify)}.")
        if self.na_note:
            lines.append(self.na_note)
        lines += ["", "Item types: O = objective (judge as a knowledgeable expert would; your answer should be the "
                      "same whoever you are). S = subjective (judge from your own point of view).", ""]
        for s in self.sections:
            lines += [f"## Section {s.id}. {s.title} (weight {s.weight:g})", s.question]
            if s.note:
                lines.append(s.note)
            for i in s.items:
                suffix = f" (N/A only if {i.na_when})" if i.allow_na and i.na_when else (" (N/A allowed)" if i.allow_na else "")
                per = f" Score once per item in the list below, in order." if i.per else ""
                lines.append(f"- {i.id} [{i.type}] {i.text}{suffix}{per}")
            for per in dict.fromkeys(i.per for i in s.items if i.per):
                lines.append(f"  {per.capitalize()}, in order: " +
                             "; ".join(f"{n}. {label}" for n, label in enumerate(labels[per], 1)))
            lines.append("")
        if self.red_flags:
            lines += ["## Red flags", "Yes/no checks that override the score. Report a flag only if the text clearly "
                      "does this, and quote the offending passage word for word.", ""]
            lines += [f"- {f.id} ({f.severity}): {f.text}" for f in self.red_flags]
        return "\n".join(lines).strip()

    def render_response_format(self, labels: Labels) -> str:
        scores = []
        for i in self.items:
            if i.per:
                n = len(labels[i.per])
                scores.append(f'    "{i.id}": [{n} whole numbers {self.scale_min}-{self.scale_max}, one per '
                              f'{i.per[:-1] if i.per.endswith("s") else i.per}, in order]')
            else:
                na = f' or "{NA}"' if i.allow_na else ""
                scores.append(f'    "{i.id}": <{self.scale_min}-{self.scale_max}{na}>')
        outs = []
        for o in self.outputs:
            if o.type == "integer":
                shape = f"<whole number {o.min}-{o.max}>"
            elif o.type == "enum":
                shape = '"' + " | ".join(o.choices) + '"'
            elif o.type == "list" and o.fields:
                shape = "[{" + ", ".join(f'"{f}": "..."' for f in o.fields) + "}]"
            elif o.type == "list":
                shape = '["..."]'
            else:
                shape = '"..."'
            outs.append(f'    "{o.name}": {shape}')
        per_items = [i for i in self.items if i.per]
        per_note = (f' For {_join_or([i.id for i in per_items])}, use the item and the list number, '
                    f'e.g. "{per_items[0].id}.3".') if per_items else ""
        parts = [
            "Return one JSON object with exactly these keys:",
            "{",
            '  "item_scores": {', ",\n".join(scores), "  },",
            '  "justifications": {"<item id>": "<one line>"},',
            '  "red_flags": [{"id": "<flag id>", "quote": "<exact passage from the text>"}],',
            '  "outputs": {', ",\n".join(outs), "  }",
            "}",
            "",
            "Rules:",
            f"- item_scores: every item must be present. Whole numbers only.",
            f"- justifications: one entry for each score of {_join_or(self.justify)}; omit the others.{per_note}"
            if self.justify else "- justifications: optional.",
            "- red_flags: [] if none fired. Each quote must be copied exactly from the text being evaluated.",
        ]
        for o in self.outputs:
            limit = f" (at most {o.max_items})" if o.max_items else ""
            parts.append(f"- outputs.{o.name}: {o.description}{limit}")
        parts.append("- Do not add section scores, a total or a grade: they are computed from your item scores.")
        return "\n".join(parts)

    # ---- reading an evaluator's answer ------------------------------------------------
    def read(self, data: Any, labels: Labels, na_required: Optional[Mapping[str, bool]] = None,
             ) -> tuple[Optional[Answer], list[Problem]]:
        """
        Validate and normalise an evaluator's JSON answer.

        na_required — {item_id: True/False} where the facts decide N/A (True: must be N/A,
                      False: must be scored), e.g. {"1.2": no card was reversed}.
        Returns (Answer, problems); Answer is None when the data is too malformed to use.
        Problems with severity 'error' mean the answer must not be scored.
        """
        problems: list[Problem] = []
        if not isinstance(data, dict) or not isinstance(data.get("item_scores"), dict):
            return None, [Problem("bad_structure", 'expected an object with an "item_scores" object')]
        raw_scores = data["item_scores"]
        na_required = dict(na_required or {})

        scores: dict[str, Optional[int]] = {}
        for item in self.items:
            keys = self.score_keys(item, labels)
            if item.id not in raw_scores:
                problems.append(Problem("missing_item", f"item {item.id} was not scored", item=item.id))
                continue
            raw = raw_scores[item.id]
            if item.per:
                if not isinstance(raw, list) or len(raw) != len(keys):
                    problems.append(Problem("wrong_length", f"item {item.id} needs a list of {len(keys)} scores, "
                                                            f"one per {item.per}", item=item.id))
                    continue
                values = raw
            else:
                values = [raw]
            for key, value in zip(keys, values):
                score, problem = self._score(value, item, key)
                if problem:
                    problems.append(problem)
                else:
                    scores[key] = score
            if item.id in na_required and not item.per and item.id in scores:
                is_na = scores[item.id] is None
                if na_required[item.id] and not is_na:
                    problems.append(Problem("na_expected", f"item {item.id} should be {NA}: {item.na_when}",
                                            "warning", item.id))
                elif not na_required[item.id] and is_na:
                    problems.append(Problem("na_not_applicable", f"item {item.id} was marked {NA} but "
                                            f"'{item.na_when}' is not the case", item=item.id))
        for extra in sorted(set(raw_scores) - {i.id for i in self.items}):
            problems.append(Problem("unknown_item", f"unknown item {extra!r}", "warning", extra))

        justifications = {str(k): str(v).strip() for k, v in (data.get("justifications") or {}).items()
                          if isinstance(v, (str, int, float)) and str(v).strip()} \
            if isinstance(data.get("justifications") or {}, dict) else {}
        for item in self.items:
            for key in self.keys_in(item, scores):
                if scores[key] in self.justify and key not in justifications and item.id not in justifications:
                    problems.append(Problem("missing_justification",
                                            f"score {scores[key]} on {key} has no justification", "warning", key))

        flags, flag_ids = [], {f.id for f in self.red_flags}
        raw_flags = data.get("red_flags") or []
        if not isinstance(raw_flags, list):
            problems.append(Problem("bad_red_flags", '"red_flags" must be a list'))
            raw_flags = []
        for f in raw_flags:
            fid = f.get("id") if isinstance(f, dict) else None
            if fid not in flag_ids:
                problems.append(Problem("unknown_red_flag", f"unknown red flag {fid!r}"))
                continue
            quote = str(f.get("quote") or "").strip()
            if not quote:
                problems.append(Problem("red_flag_without_quote", f"{fid} has no quoted passage", item=fid))
                continue
            if any(x["id"] == fid for x in flags):
                continue
            flags.append({"id": fid, "quote": quote})

        outputs, raw_out = {}, data.get("outputs")
        if not isinstance(raw_out, dict):
            problems.append(Problem("missing_outputs", '"outputs" object is missing'))
            raw_out = {}
        for o in self.outputs:
            value, problem = self._output(o, raw_out.get(o.name))
            if problem:
                problems.append(problem)
                continue
            outputs[o.name] = value
            if o.type == "list" and "item" in o.fields:      # entries "tied to an item ID" must name a real item
                for entry in value:
                    if not self.is_item_reference(entry["item"]):
                        problems.append(Problem("unknown_item_reference", f"outputs.{o.name} refers to "
                                                f"{entry['item']!r}, which is not an item of this rubric",
                                                "warning", o.name))
        return Answer(scores, justifications, flags, outputs), problems

    def is_item_reference(self, text: str) -> bool:
        """'3.2', '2a', '2a.3' or '2a (positions 1-2)': starts with an item id that is not part of a longer id."""
        text = text.strip()
        for item_id in sorted((i.id for i in self.items), key=len, reverse=True):
            if text.startswith(item_id):
                rest = text[len(item_id):]
                if not rest or not rest[0].isalnum():
                    return True
        return False

    @staticmethod
    def keys_in(item: Item, scores: Mapping[str, Any]) -> list[str]:
        """The keys in `scores` that belong to an item: its id, or '<id>.<n>' for per-label items."""
        if not item.per:
            return [item.id] if item.id in scores else []
        prefix = item.id + "."
        return sorted((k for k in scores if k.startswith(prefix) and k[len(prefix):].isdigit()),
                      key=lambda k: int(k[len(prefix):]))

    def _score(self, value: Any, item: Item, key: str) -> tuple[Optional[int], Optional[Problem]]:
        if value is None or (isinstance(value, str) and value.strip().upper().replace(".", "") in ("N/A", "NA")):
            if item.allow_na:
                return None, None
            return None, Problem("na_not_allowed", f"{NA} is not allowed on item {key}", item=key)
        if isinstance(value, str) and value.strip().lstrip("-").isdigit():
            value = int(value.strip())
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        if isinstance(value, bool) or not isinstance(value, int):
            return None, Problem("not_a_whole_number", f"score for {key} must be a whole number, got {value!r}",
                                 item=key)
        if not self.scale_min <= value <= self.scale_max:
            return None, Problem("out_of_range", f"score {value} for {key} is outside "
                                                 f"{self.scale_min}-{self.scale_max}", item=key)
        return value, None

    def _output(self, o: Output, value: Any) -> tuple[Any, Optional[Problem]]:
        bad = lambda msg: (None, Problem("bad_output", f"outputs.{o.name}: {msg}", item=o.name))
        if value is None:
            return ([], None) if o.type == "list" else bad("missing")
        if o.type == "text":
            return (value.strip(), None) if isinstance(value, str) and value.strip() else bad("must be non-empty text")
        if o.type == "integer":
            if isinstance(value, str) and value.strip().isdigit():
                value = int(value)
            if isinstance(value, float) and value.is_integer():
                value = int(value)
            if isinstance(value, bool) or not isinstance(value, int) or not o.min <= value <= o.max:
                return bad(f"must be a whole number {o.min}-{o.max}, got {value!r}")
            return value, None
        if o.type == "enum":
            v = str(value).strip().lower().replace(" ", "_")
            return (v, None) if v in o.choices else bad(f"must be one of {list(o.choices)}, got {value!r}")
        if o.type == "list":
            if not isinstance(value, list):
                return bad("must be a list")
            if o.fields:
                if not all(isinstance(x, dict) and all(str(x.get(f) or "").strip() for f in o.fields) for x in value):
                    return bad(f"each entry needs {list(o.fields)}")
                value = [{f: str(x[f]).strip() for f in o.fields} for x in value]
            else:
                value = [str(x).strip() for x in value if str(x).strip()]
            return (value[:o.max_items] if o.max_items else value), None
        return bad(f"unsupported output type {o.type!r}")

    # ---- scoring --------------------------------------------------------------------
    def score(self, scores: Mapping[str, Optional[int]], fired_flags: Sequence[str] = (),
              labels: Optional[Labels] = None) -> ScoreResult:
        """
        Section average = mean of the section's non-N/A scores (per-label items contribute one
        score per label). Each section contributes weight * average / scale_max. A section
        with nothing scored drops out and the remaining weights are rescaled to 100.
        The lowest cap among fired red flags limits the total.
        """
        section_scores: dict[str, Optional[float]] = {}
        for s in self.sections:
            values = [scores[k] for item in s.items for k in self.keys_in(item, scores) if scores[k] is not None]
            section_scores[s.key] = sum(values) / len(values) if values else None

        total_weight = sum(s.weight for s in self.sections)
        live_weight = sum(s.weight for s in self.sections if section_scores[s.key] is not None)
        rescale = total_weight / live_weight if live_weight else 0.0
        points = {s.key: (s.weight * rescale * section_scores[s.key] / self.scale_max
                          if section_scores[s.key] is not None else 0.0) for s in self.sections}
        raw = sum(points.values()) * (100.0 / total_weight if total_weight else 0.0)

        caps = [f.cap for f in self.red_flags if f.id in set(fired_flags)]
        cap = min(caps) if caps else None
        total = round(min(raw, cap) if cap is not None else raw, 1)
        grade = next(g.grade for g in sorted(self.grades, key=lambda g: -g.min) if total >= g.min)
        return ScoreResult(section_scores, points, round(raw, 2), cap if cap is not None and raw > cap else None,
                           total, grade)


def _join_or(values) -> str:
    values = [str(v) for v in values]
    return values[0] if len(values) == 1 else ", ".join(values[:-1]) + " or " + values[-1]


# ---- loading ----------------------------------------------------------------------------
def load_rubric(path: str | Path) -> Rubric:
    path = Path(path)
    raw = path.read_bytes()
    try:
        d = tomllib.loads(raw.decode("utf-8"))
    except tomllib.TOMLDecodeError as e:
        raise RubricError(f"{path}: invalid TOML: {e}") from e
    try:
        scale = d["scale"]
        sections = tuple(Section(
            id=str(s["id"]), key=s["key"], title=s["title"], weight=float(s["weight"]), question=s.get("question", ""),
            note=s.get("note"),
            items=tuple(Item(id=str(i["id"]), text=i["text"], type=i["type"], allow_na=i.get("allow_na", False),
                             na_when=i.get("na_when"), per=i.get("per")) for i in s["items"]),
        ) for s in d["sections"])
        rubric = Rubric(
            name=d["name"], version=int(d["version"]), title=d.get("title", d["name"]),
            description=d.get("description", ""), scale_min=int(scale["min"]), scale_max=int(scale["max"]),
            anchors=tuple(scale.get("anchors", ())), justify=tuple(scale.get("justify", ())),
            na_note=scale.get("na_note", ""), sections=sections,
            red_flags=tuple(RedFlag(f["id"], f["text"], f.get("severity", "major"), float(f["cap"]))
                            for f in d.get("red_flags", ())),
            grades=tuple(Grade(g["grade"], float(g["min"]), g.get("meaning", "")) for g in d.get("grades", ())),
            outputs=tuple(Output(name=o["name"], type=o["type"], description=o.get("description", ""),
                                 min=o.get("min"), max=o.get("max"), choices=tuple(o.get("choices", ())),
                                 max_items=o.get("max_items"), fields=tuple(o.get("fields", ())))
                          for o in d.get("outputs", ())),
            source=d.get("source"), path=path.resolve(),
            sha256=hashlib.sha256(json.dumps(d, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
        )
    except KeyError as e:
        raise RubricError(f"{path}: missing key {e}") from e
    _check(rubric, path)
    return rubric


def _check(r: Rubric, path: Path) -> None:
    ids = [i.id for i in r.items]
    problems = []
    if path.stem != r.id:
        problems.append(f"file name should be {r.id}.toml")
    if len(set(ids)) != len(ids):
        problems.append("duplicate item ids")
    if len({s.key for s in r.sections}) != len(r.sections):
        problems.append("duplicate section keys")
    if any(i.type not in (OBJECTIVE, SUBJECTIVE) for i in r.items):
        problems.append('item type must be "O" or "S"')
    if not r.grades or min(g.min for g in r.grades) > 0:
        problems.append("grade bands must reach down to 0")
    if any(o.type not in ("text", "integer", "enum", "list") for o in r.outputs):
        problems.append("output type must be text, integer, enum or list")
    if any(o.type == "integer" and (o.min is None or o.max is None) for o in r.outputs):
        problems.append("integer outputs need min and max")
    if problems:
        raise RubricError(f"{path}: " + "; ".join(problems))


def get_rubric(ref: str, root: str | Path = RUBRICS_DIR) -> Rubric:
    """'celtic_cross_reading_v1' for a pinned version, 'celtic_cross_reading' for the latest."""
    root = Path(root)
    exact = root / f"{ref}.toml"
    if exact.is_file():
        return load_rubric(exact)
    versions = sorted(root.glob(f"{ref}_v*.toml"), key=lambda p: int(p.stem.rsplit("_v", 1)[1]))
    if not versions:
        raise KeyError(f"No rubric {ref!r}. Available: {sorted(p.stem for p in root.glob('*.toml'))}")
    return load_rubric(versions[-1])
