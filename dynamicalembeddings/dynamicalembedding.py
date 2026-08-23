import sys
sys.path.insert(0, '..')

from collections import defaultdict
from text_tokenizers import WhitespaceTokenizer, Tokenizer, TokenizeResult
from text_tokenizers.base import Token


def group_tokens(
    tokens: list[Token],
    kernel_size: int = None,
    char_len: int = None,
    stride: int = None,
    dilation: int = 1,
    drop_last: bool = True,
) -> list[list[Token]]:
    """
    Partition/slide a window over a flat list of tokens into groups.

    Exactly one of kernel_size or char_len must be provided.

    kernel_size — window width in tokens (same meaning as a CNN's kernel_size).
    stride      — step between window starts. Defaults to kernel_size (non-overlapping,
                  the original tok_len behavior).
    dilation    — spacing between the tokens sampled within one window (default 1 =
                  contiguous). Window i taps tokens[start], tokens[start+dilation], ...,
                  tokens[start+(kernel_size-1)*dilation] where start = i * stride —
                  identical index arithmetic to Conv1d/MaxPool1d.
    drop_last   — if True (default, matches PyTorch's no-padding "valid" conv), windows
                  that would run past the end of tokens are dropped. If False, leftover
                  tokens after the last full window are appended as one final (possibly
                  shorter) group.

    char_len — variable-size groups where the character span
               (last token end - first token start) stays <= char_len.
               A single token that already exceeds char_len gets its own group.
               stride/dilation/drop_last do not apply in this mode.
    """
    if (kernel_size is None) == (char_len is None):
        raise ValueError("Provide exactly one of kernel_size or char_len.")

    groups = []

    if kernel_size is not None:
        stride = stride or kernel_size
        n = len(tokens)
        span = dilation * (kernel_size - 1) + 1
        start = 0
        while start + span <= n:
            groups.append([tokens[start + j * dilation] for j in range(kernel_size)])
            start += stride

        if not drop_last and start < n:
            groups.append(tokens[start:])
    else:
        current = []
        for token in tokens:
            if not current:
                current.append(token)
            elif (token.end - current[0].start) <= char_len:
                current.append(token)
            else:
                groups.append(current)
                current = [token]
        if current:
            groups.append(current)

    return groups


def reconstruct_text(group: list[Token], original_text: str) -> str:
    """
    Reconstruct the original text span for a group of tokens by slicing
    original_text[first.start : last.end] — preserves original whitespace
    and punctuation exactly rather than joining token strings.
    """
    if not group:
        return ""
    return original_text[group[0].start : group[-1].end].strip()


class DynamicalEmbedding:
    """
    Broader design question: should it be one per text, even if the text is then split
    down into smaller texts?
    """

    DEFAULT_CONFIG = {
        'token_len_per_embedding': None,
        'token_split_style': 'split',
        'outputs': [
            'embedding',
            'velocity',
            'acceleration',
            'energy_functional',
        ],
        'basis_set': 'cartesian',
        'convert_to_walkers': False,
    }

    POSSIBLE_OUTPUTS = ['embedding', 'velocity', 'acceleration', 'energy_functional']
    POSSIBLE_BASIS_SETS = ['cartesian', 'probabilities', 'spherical_harmonics']

    def __init__(self, text: str, config=None, tokenizer: Tokenizer = None, spacing=None):
        self.text = text
        self.config = config or self.DEFAULT_CONFIG
        self.tokenizer = tokenizer or WhitespaceTokenizer()
        self.spacing = spacing
        self.outputs = defaultdict(dict)

        self.process_text()
        self.create_spaced_text()

    def process_text(self, tokenizer: Tokenizer = None) -> TokenizeResult:
        """
        Tokenizes text using the provided tokenizer, or falls back to the instance-level
        tokenizer (default: WhitespaceTokenizer).

        Planned pipelines (not yet implemented):
        1) base    - breaks text into tokens (this method)
        2) entities
        3) coreference
        4) key values
        """
        tok = tokenizer or self.tokenizer
        result = tok.tokenize(self.text)
        self.outputs['tokens'] = result.tokens
        if result.entities:
            self.outputs['entities'] = result.entities
        return result

    def create_spaced_text(self):
        """
        Uses self.spacing to partition tokens into sub-text windows and stores
        them in self.outputs['subtexts'].

        spacing accepts:
          - None                          → no-op
          - int or list[int]              → treated as tok_len(s)
          - dict with keys "tok_len" and/or "char_len",
            each mapping to int or list[int]

        Output structure:
          self.outputs['subtexts'] = {
              "tok_len_5":   ["group1 text", "group2 text", ...],
              "tok_len_10":  [...],
              "char_len_100": [...],
          }
        """
        if self.spacing is None:
            return

        tokens = self.outputs.get('tokens', [])
        if not tokens:
            return

        spec = {"tok_len": [], "char_len": []}

        if isinstance(self.spacing, int):
            spec["tok_len"] = [self.spacing]
        elif isinstance(self.spacing, list):
            spec["tok_len"] = self.spacing
        elif isinstance(self.spacing, dict):
            for key in ("tok_len", "char_len"):
                if key in self.spacing:
                    val = self.spacing[key]
                    spec[key] = [val] if isinstance(val, int) else val

        subtexts = {}

        for length in spec["tok_len"]:
            groups = group_tokens(tokens, kernel_size=length, drop_last=False)
            subtexts[f"tok_len_{length}"] = [
                reconstruct_text(g, self.text) for g in groups
            ]

        for length in spec["char_len"]:
            groups = group_tokens(tokens, char_len=length)
            subtexts[f"char_len_{length}"] = [
                reconstruct_text(g, self.text) for g in groups
            ]

        self.outputs['subtexts'] = subtexts


class EmbeddingInteraction:
    """
    !!! Is this the same as the Hamiltonian/Lagragian?
    """
