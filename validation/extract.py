"""
Regex extraction of labelled entries ("<label>: <value>") from free-form LLM text.

LLMs format the same content many ways:
    1. Present: Eight of Cups reversed - Your current ...
    1. **Present — Three of Pentacles:** You're in a building phase ...
    **Conscious goal — Seven of Wands reversed:** ...
    ### Present (The Hanged Man)
    The near future (The Hanged Man) calls for ...          <- inline, in prose
so extraction (a) normalises each line (markdown, list markers), (b) matches a label
alias at the start of the line followed by a separator, (c) cuts the value segment at
the next separator, and (d) asks a ValueMatcher what value the segment names. Labels
never found on their own line can fall back to an inline search in prose.
"""
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import Iterable, Mapping, Optional

_MARKDOWN = re.compile(r"\*\*|__|[*`]|^\s*#{1,6}\s*|^\s*>\s*")
_LIST_MARKER = re.compile(r"^\s*(?:[-•+]|\(?\d{1,2}[.):])\s+")
_ENUM_PREFIX = r"(?:(?:position|card|step)\s*\d{1,2}\s*[:.)\-—–]?\s*)?"
_SEPARATOR = r"\s*(?:[:—–(]|\s-\s|=)\s*"
# where a value segment ends: description separators, closing brackets, sentence end
_SEGMENT_END = re.compile(r"\s[-—–]\s|[:—–)\]]|\.(?:\s|$)|\n")
_EMPHASIS = re.compile(r"\*\*|__|[*`]")
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+|\n+")


def normalize_line(line: str) -> str:
    line = _MARKDOWN.sub("", line)
    line = _LIST_MARKER.sub("", line)
    return line.strip()


# between the words of a label: spaces, '/', '&', 'and', '-' ("Hopes & Fears", "Root Cause / Foundation")
_WORD_SEP = r"(?:\s*(?:/|&|-|\band\b)\s*|\s+)"


def _alias_pattern(aliases: Iterable[str]) -> str:
    """Alternation of aliases (longest first), tolerant of the separators between words."""
    parts = []
    for a in sorted(set(aliases), key=len, reverse=True):
        words = [w for w in re.split(r"[^\w']+", a) if w and w.casefold() != "and"]
        parts.append(_WORD_SEP.join(re.escape(w) for w in words))
    return "|".join(parts)


def _label_key(s: str) -> str:
    """Spelling-insensitive key: 'Hopes & Fears', 'hopes and fears', 'Hopes/Fears' -> 'hopesfears'."""
    words = re.split(r"[^a-z0-9]+", s.casefold())
    return "".join(w for w in words if w and w != "and")


def default_aliases(label: str) -> list[str]:
    """The label itself plus each '/'-separated part ('Root Cause/Foundation' -> 'Root Cause', 'Foundation')."""
    parts = [p.strip() for p in label.split("/") if p.strip()]
    return [label] + (parts if len(parts) > 1 else [])


# ---- value matchers -----------------------------------------------------------------
class ValueMatcher(ABC):
    """Decides which value (if any) a text segment names."""

    @abstractmethod
    def find(self, segment: str) -> Optional[str]:
        """Canonical value named in `segment`, or None."""

    def find_all(self, text: str) -> list[tuple[str, int]]:
        """(identity, position) of every value mentioned anywhere in `text`. Optional."""
        raise NotImplementedError

    def mentions(self, text: str) -> list[str]:
        """Every value mentioned in `text`, in order, each as fully as find() would give it. Optional."""
        raise NotImplementedError

    def identity(self, value: str) -> str:
        """Key used to decide whether two values are 'the same thing' (e.g. ignoring orientation)."""
        return value


class VocabularyMatcher(ValueMatcher):
    """
    Values come from a closed vocabulary. `vocabulary` is a list of canonical values, or
    {canonical: [aliases...]}. Matching is word-bounded, longest alias first.
    """

    def __init__(self, vocabulary: Iterable[str] | Mapping[str, Iterable[str]], case_sensitive: bool = False):
        vocab = vocabulary if isinstance(vocabulary, Mapping) else {v: [] for v in vocabulary}
        self._canonical: dict[str, str] = {}
        for canon, aliases in vocab.items():
            for a in [canon, *aliases]:
                self._canonical[a.casefold()] = canon
        flags = 0 if case_sensitive else re.IGNORECASE
        alternation = "|".join(re.escape(a) for a in sorted(
            {a for canon, al in vocab.items() for a in [canon, *al]}, key=len, reverse=True))
        self._pattern = re.compile(rf"(?<![\w'])(?:{alternation})(?![\w'])", flags)
        self.vocabulary = list(vocab)

    def _match(self, m: re.Match) -> str:
        return self._canonical[m.group(0).casefold()]

    def find(self, segment: str) -> Optional[str]:
        m = self._pattern.search(segment)
        return self._match(m) if m else None

    def find_all(self, text: str) -> list[tuple[str, int]]:
        return [(self.identity(self._match(m)), m.start()) for m in self._pattern.finditer(text)]

    def mentions(self, text: str) -> list[str]:
        return [self._match(m) for m in self._pattern.finditer(text)]


class FreeTextMatcher(ValueMatcher):
    """Open-ended values: the (trimmed) segment itself is the value."""

    def __init__(self, max_chars: int = 200):
        self.max_chars = max_chars

    def find(self, segment: str) -> Optional[str]:
        s = segment.strip().strip("\"'")
        return s[:self.max_chars] or None


# ---- extraction -----------------------------------------------------------------------
@dataclass
class Entry:
    label: str               # canonical label
    value: Optional[str]     # canonical value, None if the segment named nothing recognisable
    segment: str             # the raw text the value was read from
    line: int                # 0-based line number in the original text
    mode: str                # "line" (label starts a line) or "inline" (found in prose)

    def to_dict(self) -> dict:
        return asdict(self)


class LabeledEntryExtractor:
    """
    labels   — canonical labels, or {canonical: [aliases]} (defaults via default_aliases)
    matcher  — ValueMatcher deciding what value a segment names
    inline_window — chars after an inline label mention to look for its value (0 disables)
    lookahead_lines — if a label line names no value (e.g. a heading), also try the next
                      N non-empty lines
    """

    def __init__(
        self,
        labels: Iterable[str] | Mapping[str, Iterable[str]],
        matcher: ValueMatcher,
        inline_window: int = 40,
        lookahead_lines: int = 1,
    ):
        label_map = labels if isinstance(labels, Mapping) else {l: default_aliases(l) for l in labels}
        self.labels = list(label_map)
        self.matcher = matcher
        self.inline_window = inline_window
        self.lookahead_lines = lookahead_lines
        self._alias_to_label = {_label_key(a): canon for canon, al in label_map.items() for a in {canon, *al}}
        alias_re = _alias_pattern(a for canon, al in label_map.items() for a in {canon, *al})
        self._line_re = re.compile(
            rf"^{_ENUM_PREFIX}(?:the\s+)?(?P<label>{alias_re})(?:\s+(?:position|card))?{_SEPARATOR}(?P<rest>.*)$",
            re.IGNORECASE)
        self._heading_re = re.compile(
            rf"^{_ENUM_PREFIX}(?:the\s+)?(?P<label>{alias_re})(?:\s+(?:position|card))?\s*[:.]?\s*$", re.IGNORECASE)
        self._inline_re = re.compile(rf"(?<![\w'])(?P<label>{alias_re})(?![\w'])", re.IGNORECASE)

    def canonical_label(self, matched: str) -> str:
        return self._alias_to_label.get(_label_key(matched), matched)

    @staticmethod
    def segment(rest: str) -> str:
        m = _SEGMENT_END.search(rest)
        return (rest[:m.start()] if m else rest)[:120].strip()

    def extract(self, text: str) -> list[Entry]:
        """All line-start entries, in document order (a label may appear more than once)."""
        lines = text.splitlines()
        normalized = [normalize_line(l) for l in lines]
        entries = []
        for i, line in enumerate(normalized):
            m = self._line_re.match(line) or self._heading_re.match(line)
            if not m:
                continue
            label = self.canonical_label(m.group("label"))
            seg = self.segment(m.groupdict().get("rest") or "")
            value = self.matcher.find(seg) if seg else None
            if value is None:  # e.g. "### Present" with the card on the next line
                following = [l for l in normalized[i + 1:i + 1 + 3] if l][:self.lookahead_lines]
                for nxt in following:
                    if self._line_re.match(nxt) or self._heading_re.match(nxt):
                        break
                    value = self.matcher.find(self.segment(nxt))
                    if value is not None:
                        seg = self.segment(nxt)
                        break
            entries.append(Entry(label, value, seg, i, "line"))
        return entries

    def co_mentions(self, text: str) -> Optional[tuple[dict[str, list[tuple[str, int]]], list[str]]]:
        """
        How labels and values meet in running prose, whichever comes first ("In the Present,
        Death ..." or "Death in the Present ..."):

            ({label: [(value, line), ...]}, [every value mentioned anywhere])

        A value belongs to a label when one sentence names both. None if the matcher cannot
        list mentions (free text).
        """
        try:
            self.matcher.mentions("")
        except NotImplementedError:
            return None
        clean = _EMPHASIS.sub("", text)
        together: dict[str, list[tuple[str, int]]] = {}
        everywhere: list[str] = []
        start = 0
        for end in [m.start() for m in _SENTENCE_BREAK.finditer(clean)] + [len(clean)]:
            sentence, line = clean[start:end], clean.count("\n", 0, start)
            start = end
            values = self.matcher.mentions(sentence)
            if not values:
                continue
            everywhere += values
            for label in {self.canonical_label(m.group("label")) for m in self._inline_re.finditer(sentence)}:
                together.setdefault(label, []).extend((v, line) for v in values)
        return together, everywhere

    def extract_inline(self, text: str, labels: Iterable[str]) -> dict[str, Entry]:
        """For the given labels, the first prose mention followed closely by a recognisable value."""
        wanted, found = set(labels), {}
        if not self.inline_window:
            return found
        for m in self._inline_re.finditer(text):
            label = self.canonical_label(m.group("label"))
            if label not in wanted or label in found:
                continue
            window = text[m.end():m.end() + self.inline_window]
            window = re.split(r"\.(?:\s|$)|\n", window)[0]
            value = self.matcher.find(self.segment(window.lstrip(" :—–-(")))
            if value is not None:
                found[label] = Entry(label, value, window.strip(), text.count("\n", 0, m.start()), "inline")
        return found
