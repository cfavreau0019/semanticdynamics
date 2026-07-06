from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import Optional


@dataclass
class Token:
    text: str
    start: int
    end: int
    id: Optional[int] = None        # subword tokenizers: vocab id
    label: Optional[str] = None     # spaCy-style: POS, entity type, etc.


@dataclass
class TokenizeResult:
    text: str
    tokens: list[Token]
    entities: list[Token] = field(default_factory=list)  # only some backends populate this


class Tokenizer(ABC):
    @abstractmethod
    def tokenize(self, text: str) -> TokenizeResult:
        ...

    def decode(self, tokens: list[Token]) -> str:
        # default: reconstruct via offsets — works for anything that preserves spans
        return "".join(t.text for t in tokens)
