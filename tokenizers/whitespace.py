import re

from tokenizers.base import Token, TokenizeResult, Tokenizer


class WhitespaceTokenizer(Tokenizer):
    def tokenize(self, text: str) -> TokenizeResult:
        tokens = []
        for match in re.finditer(r"\S+", text):
            tokens.append(Token(text=match.group(), start=match.start(), end=match.end()))
        return TokenizeResult(text=text, tokens=tokens)
