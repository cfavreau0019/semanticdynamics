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

    @property
    @abstractmethod
    def vocab(self) -> dict[str, int]:
        """Return a token → id mapping for this tokenizer's vocabulary."""
        ...

    def decode(self, tokens: list[Token]) -> str:
        return "".join(t.text for t in tokens)
