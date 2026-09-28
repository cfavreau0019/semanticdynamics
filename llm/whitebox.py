"""
White-box model access through NNsight: hidden states from a forward pass, batched
forward passes, and generation with per-step hidden states. Runs remotely on NDIF
(remote=True, needs NDIF_API_TOKEN) or locally.

    from llm.whitebox import WhiteBoxModel

    wb = WhiteBoxModel("meta-llama/Llama-3.1-70B-Instruct")        # remote by default
    hs = wb.hidden_states("An insect usually has a small size.")
    hs.hidden_states.shape          # (n_layers, seq_len, hidden_dim)
    hs.mean_pool().shape            # (n_layers, hidden_dim)

    many = wb.hidden_states_many(sentences, batch_size=8)           # one NDIF job per 8 texts
    gen = wb.generate(wb.format_chat(prompt), max_new_tokens=50)
    gen.text, gen.new_token_states().shape                          # (n_layers, n_new, hidden_dim)
"""
import os
from dataclasses import dataclass, field
from typing import Optional, Sequence

import torch

# Where the list of decoder blocks / the unembedding lives, by architecture family.
LAYER_PATHS = ("model.layers", "transformer.h", "gpt_neox.layers", "model.decoder.layers", "transformer.layers")
LM_HEAD_PATHS = ("lm_head", "embed_out")


@dataclass
class HiddenStates:
    """Residual-stream outputs of each requested decoder layer for one text."""
    text: str
    token_ids: list[int]
    tokens: list[str]
    layers: list[int]                    # which layer indices are stacked in hidden_states
    hidden_states: torch.Tensor          # (n_layers, seq_len, hidden_dim), on CPU
    next_token_logits: Optional[torch.Tensor] = field(default=None, repr=False)  # (vocab,) at last position
    next_token_id: Optional[int] = None
    next_token: Optional[str] = None

    def mean_pool(self) -> torch.Tensor:
        """(n_layers, hidden_dim) — mean over token positions."""
        return self.hidden_states.mean(dim=1)

    def last_token(self) -> torch.Tensor:
        """(n_layers, hidden_dim) — state at the final position."""
        return self.hidden_states[:, -1, :]

    def layer(self, idx: int) -> torch.Tensor:
        """(seq_len, hidden_dim) for model layer `idx` (a model layer index, not a position in `layers`)."""
        return self.hidden_states[self.layers.index(idx)]


@dataclass
class GenerationTrace:
    """
    Output of WhiteBoxModel.generate. step_hidden_states[0] covers the whole prompt
    (n_layers, prompt_len, hidden); each later step covers the one new position
    (n_layers, 1, hidden) because of the KV cache.
    """
    prompt: str
    text: str                            # decoded continuation only
    prompt_token_ids: list[int]
    new_token_ids: list[int]
    layers: list[int]
    step_hidden_states: list[torch.Tensor] = field(default_factory=list, repr=False)

    def prompt_states(self) -> torch.Tensor:
        """(n_layers, prompt_len, hidden_dim)."""
        return self.step_hidden_states[0]

    def new_token_states(self) -> torch.Tensor:
        """
        (n_layers, n_new_tokens, hidden_dim): the state at the position that *predicted*
        each new token (last prompt position for token 0, then each generated position).
        """
        return torch.cat([s[:, -1:, :] for s in self.step_hidden_states], dim=1)


class WhiteBoxModel:
    """
    Args:
        model_name:   HF repo id (must be hosted on NDIF when remote=True)
        remote:       run on NDIF (True) or load weights locally (False)
        api_key:      NDIF key; defaults to $NDIF_API_TOKEN
        layers_path:  dotted path to the decoder block list; auto-detected if None
        lm_head_path: dotted path to the unembedding; auto-detected if None
        kwargs:       forwarded to nnsight.LanguageModel (device_map, dtype, ...). For
                      remote use, weights are never downloaded (meta tensors only).
    """

    def __init__(
        self,
        model_name: str,
        remote: bool = True,
        api_key: Optional[str] = None,
        layers_path: Optional[str] = None,
        lm_head_path: Optional[str] = None,
        **kwargs,
    ):
        from nnsight import CONFIG, LanguageModel

        if remote:
            key = api_key or os.getenv("NDIF_API_TOKEN")
            if not key:
                raise ValueError("remote=True needs an NDIF key (NDIF_API_TOKEN in .env or api_key=).")
            CONFIG.API.APIKEY = key
        kwargs.setdefault("device_map", "auto")
        self.model_name = model_name
        self.remote = remote
        self.model = LanguageModel(model_name, **kwargs)
        self._layers = self._resolve(layers_path, LAYER_PATHS, "decoder layers")
        self._lm_head = self._resolve(lm_head_path, LM_HEAD_PATHS, "lm_head")

    def __repr__(self) -> str:
        return f"WhiteBoxModel({self.model_name!r}, remote={self.remote}, n_layers={self.n_layers})"

    def _resolve(self, path: Optional[str], candidates: Sequence[str], what: str):
        for p in ([path] if path else candidates):
            obj = self.model
            try:
                for attr in p.split("."):
                    obj = getattr(obj, attr)
                return obj
            except AttributeError:
                continue
        raise AttributeError(f"Could not find {what} on {self.model_name}; pass its dotted path explicitly.")

    @property
    def tokenizer(self):
        return self.model.tokenizer

    @property
    def n_layers(self) -> int:
        return len(self._layers)

    def _layer_indices(self, layers: Optional[Sequence[int]]) -> list[int]:
        if layers is None:
            return list(range(self.n_layers))
        # nnsight requires modules to be accessed in execution order
        return sorted({l % self.n_layers for l in layers})

    def format_chat(self, prompt: str, system: Optional[str] = None, add_generation_prompt: bool = True) -> str:
        """Apply the model's chat template (instruct models expect this for sensible generations)."""
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)

    def _token_ids(self, text: str) -> list[int]:
        # same tokenisation LanguageModel applies to string inputs (incl. BOS)
        return self.tokenizer(text)["input_ids"]

    # ---- forward passes -----------------------------------------------------------
    def hidden_states(
        self,
        text: str,
        layers: Optional[Sequence[int]] = None,
        save_logits: bool = False,
        **trace_kwargs,
    ) -> HiddenStates:
        """One forward pass; returns each requested layer's output for every token."""
        import nnsight

        idx = self._layer_indices(layers)
        layer_modules = [self._layers[i] for i in idx]
        lm_head = self._lm_head

        with self.model.trace(text, remote=self.remote, **trace_kwargs):
            outs = []
            for layer in layer_modules:
                out = layer.output
                if isinstance(out, tuple):  # older transformers return (hidden, ...)
                    out = out[0]
                outs.append(out[0].cpu())
            states = nnsight.save(torch.stack(outs))
            logits = nnsight.save(lm_head.output[0, -1].cpu())

        return self._pack(text, idx, states, logits, save_logits)

    def hidden_states_many(
        self,
        texts: Sequence[str],
        layers: Optional[Sequence[int]] = None,
        batch_size: int = 8,
        save_logits: bool = False,
        progress: bool = True,
        **trace_kwargs,
    ) -> list[HiddenStates]:
        """
        Forward passes for many texts, `batch_size` texts per trace (one NDIF request each),
        using one invoke per text. Results are returned in input order and match
        hidden_states() on each text.

        Mixed-length batches are right-padded: under causal attention a real token never
        attends to padding that comes after it, and positions still start at 0, so every
        real position is unaffected (left padding would shift positions, which changes the
        states of absolute-position models such as GPT-2 entirely). Each text's states are
        sliced to its true length and its next-token logits are read at its last real token.
        """
        import nnsight
        from tqdm.auto import tqdm

        idx = self._layer_indices(layers)
        layer_modules = [self._layers[i] for i in idx]
        lm_head = self._lm_head
        results: list[HiddenStates] = []

        # nnsight pads batched invokes according to the tokenizer, not a trace kwarg
        original_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "right"
        try:
            for start in tqdm(range(0, len(texts), batch_size), desc="hidden states", disable=not progress):
                batch = list(texts[start:start + batch_size])
                lengths = [len(self._token_ids(t)) for t in batch]
                with self.model.trace(remote=self.remote, **trace_kwargs) as tracer:
                    saved_states = nnsight.save(list())
                    saved_logits = nnsight.save(list())
                    for t, n in zip(batch, lengths):
                        with tracer.invoke(t):
                            outs = []
                            for layer in layer_modules:
                                out = layer.output
                                if isinstance(out, tuple):
                                    out = out[0]
                                outs.append(out[0, :n].cpu())
                            saved_states.append(torch.stack(outs))
                            saved_logits.append(lm_head.output[0, n - 1].cpu())

                for t, states, logits in zip(batch, saved_states, saved_logits):
                    results.append(self._pack(t, idx, states, logits, save_logits))
        finally:
            self.tokenizer.padding_side = original_side
        return results

    def _pack(self, text, idx, states, logits, save_logits) -> HiddenStates:
        ids = self._token_ids(text)
        next_id = int(logits.argmax(dim=-1))
        return HiddenStates(
            text=text,
            token_ids=ids,
            tokens=self.tokenizer.convert_ids_to_tokens(ids),
            layers=idx,
            hidden_states=states,
            next_token_logits=logits if save_logits else None,
            next_token_id=next_id,
            next_token=self.tokenizer.decode([next_id]),
        )

    # ---- generation ---------------------------------------------------------------
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        save_hidden_states: bool = True,
        layers: Optional[Sequence[int]] = None,
        last_token_only: bool = False,
        **generate_kwargs,
    ) -> GenerationTrace:
        """
        Generate a continuation and (optionally) capture hidden states at every step.
        Pass the output of format_chat() as `prompt` for instruct models. generate_kwargs
        go to HF generate (do_sample, temperature, top_p, ...).

        last_token_only — keep only the final position of each step (shrinks the prompt
                          step from prompt_len positions to 1; the download for a 70B model
                          over a long prompt is otherwise large).
        """
        import nnsight

        idx = self._layer_indices(layers)
        layer_modules = [self._layers[i] for i in idx]
        generator = self.model.generator

        with self.model.generate(prompt, max_new_tokens=max_new_tokens, remote=self.remote,
                                 **generate_kwargs) as tracer:
            steps = nnsight.save(list())
            if save_hidden_states:
                for _ in tracer.iter[:]:
                    outs = []
                    for layer in layer_modules:
                        out = layer.output
                        if isinstance(out, tuple):
                            out = out[0]
                        out = out[0, -1:] if last_token_only else out[0]
                        outs.append(out.cpu())
                    steps.append(torch.stack(outs))
            output_ids = nnsight.save(generator.output[0].cpu())

        prompt_ids = self._token_ids(prompt)
        new_ids = output_ids[len(prompt_ids):].tolist()
        return GenerationTrace(
            prompt=prompt,
            text=self.tokenizer.decode(new_ids, skip_special_tokens=True),
            prompt_token_ids=prompt_ids,
            new_token_ids=new_ids,
            layers=idx,
            step_hidden_states=list(steps),
        )
