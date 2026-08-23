import sentencepiece

from text_tokenizers.base import Token, TokenizeResult, Tokenizer


class SentencePieceTokenizer(Tokenizer):
    def __init__(self, model_path: str):
        self.sp = sentencepiece.SentencePieceProcessor(model_file=model_path)

    @property
    def vocab(self) -> dict[str, int]:
        return {self.sp.id_to_piece(i): i for i in range(self.sp.get_piece_size())}

    def tokenize(self, text: str) -> TokenizeResult:
        pieces = self.sp.encode(text, out_type=str)
        ids = self.sp.encode(text, out_type=int)
        tokens, pos = [], 0
        for piece, tid in zip(pieces, ids):
            clean = piece.replace("▁", " ").strip()
            idx = text.find(clean, pos) if clean else pos
            tokens.append(Token(piece, idx, idx + len(clean), id=tid))
            pos = idx + len(clean)
        return TokenizeResult(text=text, tokens=tokens)

    def decode(self, tokens: list[Token]) -> str:
        return self.sp.decode([t.id for t in tokens if t.id is not None])
