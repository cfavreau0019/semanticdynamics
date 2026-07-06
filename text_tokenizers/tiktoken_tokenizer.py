import tiktoken

from text_tokenizers.base import Token, TokenizeResult, Tokenizer


class TiktokenTokenizer(Tokenizer):
    def __init__(self, encoding_name: str = "cl100k_base"):
        self.enc = tiktoken.get_encoding(encoding_name)

    def tokenize(self, text: str) -> TokenizeResult:
        ids = self.enc.encode(text)
        # tiktoken doesn't give offsets natively — decode piece-by-piece to reconstruct them
        tokens, pos = [], 0
        for tid in ids:
            piece = self.enc.decode([tid])
            tokens.append(Token(piece, pos, pos + len(piece), id=tid))
            pos += len(piece)
        return TokenizeResult(text=text, tokens=tokens)

    def decode(self, tokens: list[Token]) -> str:
        return self.enc.decode([t.id for t in tokens if t.id is not None])
