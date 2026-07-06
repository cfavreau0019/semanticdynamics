import sys
sys.path.insert(0, '..')

from text_tokenizers import WhitespaceTokenizer, Tokenizer, TokenizeResult

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
            groups = group_tokens(tokens, tok_len=length)
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
