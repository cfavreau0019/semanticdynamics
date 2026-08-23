import spacy

from text_tokenizers.base import Token, TokenizeResult, Tokenizer


class SpacyTokenizer(Tokenizer):
    def __init__(self, model: str = "en_core_web_sm"):
        self.nlp = spacy.load(model)

    @property
    def vocab(self) -> dict[str, int]:
        # spaCy's StringStore maps every string it has seen to a stable integer hash
        return {s: self.nlp.vocab.strings[s] for s in self.nlp.vocab.strings}

    def tokenize(self, text: str) -> TokenizeResult:
        doc = self.nlp(text)
        tokens = [
            Token(t.text, t.idx, t.idx + len(t.text), id=t.orth_, label=t.pos_)
            for t in doc
        ]
        entities = [
            Token(e.text, e.start_char, e.end_char, label=e.label_)
            for e in doc.ents
        ]
        return TokenizeResult(text=text, tokens=tokens, entities=entities)
