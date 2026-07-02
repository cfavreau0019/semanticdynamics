import spacy

from tokenizers.base import Token, TokenizeResult, Tokenizer


class SpacyTokenizer(Tokenizer):
    def __init__(self, model: str = "en_core_web_sm"):
        self.nlp = spacy.load(model)

    def tokenize(self, text: str) -> TokenizeResult:
        doc = self.nlp(text)
        tokens = [Token(t.text, t.idx, t.idx + len(t.text), label=t.pos_) for t in doc]
        entities = [Token(e.text, e.start_char, e.end_char, label=e.label_) for e in doc.ents]
        return TokenizeResult(text=text, tokens=tokens, entities=entities)
