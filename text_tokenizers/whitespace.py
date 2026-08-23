import re

from text_tokenizers.base import Token, TokenizeResult, Tokenizer


class WhitespaceTokenizer(Tokenizer):
    def __init__(self):
        self._vocab: dict[str, int] = {}

    @property
    def vocab(self) -> dict[str, int]:
        return self._vocab

    def tokenize(self, text: str) -> TokenizeResult:
        tokens = []
        for match in re.finditer(r"\S+", text):
            word = match.group()
            if word not in self._vocab:
                self._vocab[word] = len(self._vocab)
            tokens.append(Token(
                text=word,
                start=match.start(),
                end=match.end(),
                id=self._vocab[word],
            ))
        return TokenizeResult(text=text, tokens=tokens)
