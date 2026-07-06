#from tokenizers_lib import Tokenizer as HFTokenizerLib
from transformers import AutoTokenizer

from text_tokenizers.base import Token, TokenizeResult, Tokenizer


class HFTokenizer(Tokenizer):
    """
    Wraps a Hugging Face fast tokenizer (transformers AutoTokenizer or
    tokenizers.Tokenizer directly) to provide exact character offsets
    via return_offsets_mapping=True.

    Any fast tokenizer works: GPT-2, RoBERTa, Llama, Mistral, etc.
    Slow tokenizers (is_fast=False) do not support offset mapping and
    will raise an error on init.

    Args:
        model_name_or_path: HF model ID or local path (e.g. "gpt2", "meta-llama/Llama-3-8B")
        add_special_tokens: whether to include [CLS]/[SEP]/BOS/EOS tokens (default False)
    """

    def __init__(self, model_name_or_path: str, add_special_tokens: bool = False):
        self.hf = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
        if not self.hf.is_fast:
            raise ValueError(
                f"{model_name_or_path} does not have a fast tokenizer. "
                "offset_mapping requires a fast (Rust-backed) tokenizer."
            )
        self.add_special_tokens = add_special_tokens

    def tokenize(self, text: str) -> TokenizeResult:
        encoding = self.hf(
            text,
            return_offsets_mapping=True,
            add_special_tokens=self.add_special_tokens,
        )
        tokens = []
        for token_id, (start, end) in zip(encoding["input_ids"], encoding["offset_mapping"]):
            # HF uses (0, 0) for special tokens — skip them if they sneak in
            if start == 0 and end == 0 and self.add_special_tokens:
                continue
            piece = text[start:end]
            tokens.append(Token(text=piece, start=start, end=end, id=token_id))
        return TokenizeResult(text=text, tokens=tokens)

    def decode(self, tokens: list[Token]) -> str:
        return self.hf.decode([t.id for t in tokens if t.id is not None])
