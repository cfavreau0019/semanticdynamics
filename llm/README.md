# llm

LLM access for the project: live inference, batch jobs, embeddings, and NNsight white-box access.

```
llm/
  types.py                  ChatRequest, ChatResponse, EmbeddingResponse, BatchJob, Usage
  providers/
    base.py                 LLMProvider: the interface every provider implements
    openai_compatible.py    OpenAI, Featherless, or any OpenAI-compatible server (chat, embeddings, Batch API)
    __init__.py             registry: get_provider("openai" | "featherless"), register_provider
  inference.py              generate, generate_many (thread pool)
  batch.py                  run_batch, wait_for_batch, run_batch_results, JSONL save/load
  embeddings.py             embed, embed_documents, embed_samples, SentenceTransformerEmbedder
  whitebox.py               WhiteBoxModel (NNsight / NDIF hidden states and generation)
```

## Configuration (`.env`)

| Provider      | Key                                        | Default chat model  | Default embedding model                                   |
|---------------|--------------------------------------------|---------------------|-----------------------------------------------------------|
| `openai`      | `OPENAI_API_KEY` or `OPENAI_API_TOKEN`     | `OPENAI_MODEL`      | `OPENAI_EMBEDDING_MODEL`, else `text-embedding-3-small`   |
| `featherless` | `FEATHERLESS_API_TOKEN`                    | `FEATHERLESS_MODEL` | `EMBEDDING_MODEL`, else `Qwen/Qwen3-Embedding-8B`         |
| NNsight       | `NDIF_API_TOKEN`                           |                     |                                                           |

`LLM_PROVIDER` sets the provider used when a call doesn't pass `provider=` (the fallback is `openai`).

## Inference

```python
from llm import generate, generate_many, ChatRequest

text = generate("Write an opening line.", provider="featherless", max_tokens=300, temperature=0.9)

reqs = [ChatRequest.from_prompt(celtic_cross_prompt(inst), max_tokens=6400, metadata={"instant": inst})
        for inst in instants]
responses = generate_many(reqs, provider="featherless", max_workers=4)   # same order as reqs
for r in responses:
    r.text, r.error, r.metadata["instant"], r.usage
```

A failed request comes back as a response with `r.error` set, and the rest of the run carries on. To checkpoint as results arrive, pass `on_result=lambda r: save_responses([r], path, append=True)`.

Params are written the same way for every provider. On OpenAI, `max_tokens` is renamed to `max_completion_tokens` automatically.

### Provider-specific parameters (`extra_body`)

Fields outside the standard OpenAI schema go in `extra_body`. Examples are vLLM sampling controls on Featherless such as `top_k`, `min_p` and `repetition_penalty`. The fields are sent verbatim at the top level of the request body.

```python
sampling = {"repetition_penalty": 1.1, "top_k": 40, "min_p": 0.02}

generate("...", provider="featherless", extra_body=sampling)                       # one call
ChatRequest.from_prompt("...", extra_body=sampling)                                 # per request
generate_many(prompts, provider="featherless", extra_body=sampling)                 # all string prompts
fl = get_provider("featherless", default_extra_body=sampling)                       # every request on this provider
```

Precedence, key by key, with later sources winning: `default_extra_body` < `default_params["extra_body"]` < `request.params["extra_body"]` < `request.extra_body`.

In batch files the fields are merged into each line's `body`. Call `provider.build_chat_body(req)` to see exactly what is sent. OpenAI rejects unknown fields with a 400 error. Featherless silently ignores fields it doesn't recognise, so a typo like `top_kk` fails without warning.

### Validating outputs

`generate_many`, `run_batch` and `run_batch_results` accept `validate=`, a `(response, request) -> dict` function. The result is stored on `response.validation`, with `response.valid` as a shortcut, before `on_result` runs or you save the responses. `apply_validation(responses, validate)` does the same for any other path, including re-validating loaded files. Validators, such as the regex-based tarot checks, live in the separate [`validation`](../validation/README.md) package.

**Retries (on by default).** With a validator, a response that fails validation is regenerated up to `validation_retries=2` more times. Set `validation_retries=0` to turn this off.

| Path | How it retries |
|---|---|
| `generate_many` | immediately, inside the worker; `on_result`/checkpoints only see the final response |
| `run_batch(wait=True)` | resubmits just the failures as a follow-up batch (another wait of up to 24h per round) |
| `run_batch_results` | never submits new work; pass its output to `retry_invalid(responses, requests, validate, mode="live" \| "batch")` |

Only content that was validated and failed is retried. API errors (the SDK retries transient ones itself) and validator crashes (`valid is None`) are not. If a retry hits an API error, the last invalid response is kept.

Nothing is thrown away:
- `response.attempts` counts every generation made for the request.
- `response.failed_attempts` keeps the text, validation, finish reason, usage and error of each attempt that wasn't returned. `usage` covers only the returned attempt, so add these for the total cost.

Retrying an identical request can reproduce the same failure, especially at `temperature=0`.

## Batch (OpenAI)

A batch costs about half the live price and finishes within 24h. Each `ChatRequest.id` becomes the line's `custom_id`. The input file looks like this:

```json
{"custom_id": "sample-0", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "...", "messages": [...], "max_completion_tokens": 2000}}
```

The code checks the API's rules before uploading: custom_ids must be unique, the file can target only one model, and the limits are 50k lines and 200 MB. Input, output and error files are written to `data/llm_batches/`, which is gitignored.

```python
from llm import get_provider, run_batch, run_batch_results, save_requests, load_requests

reqs = [ChatRequest.from_prompt(p, id=f"sample-{i}", max_tokens=2000) for i, p in enumerate(prompts)]
lines = get_provider("openai").build_batch_lines(reqs)           # inspect without submitting

responses = run_batch(reqs, provider="openai")                    # blocks, polls every 60s
# or: submit now and collect later
save_requests(reqs, "data/llm_batches/tarot_requests.jsonl")
job = run_batch(reqs, provider="openai", wait=False); print(job.id)
responses = run_batch_results(job.id, provider="openai", requests=load_requests("data/llm_batches/tarot_requests.jsonl"))
```

Embedding batches: `provider.submit_embedding_batch({id: text})`, then `provider.embedding_batch_results(job)`.

## Embeddings

```python
from llm import embed, embed_documents, embed_samples, SentenceTransformerEmbedder

vecs = embed(["a", "b"], provider="openai", normalize=True)        # (2, dim)
docs = embed_documents({i: s["reading"] for i, s in samples.items()}, provider="featherless")
samples = embed_samples(samples, provider="featherless")          # the notebook's in-place behaviour
local = embed(["a"], provider=SentenceTransformerEmbedder("all-MiniLM-L6-v2"))
```

`embed_documents` puts the chunks from every document into shared API calls, so it sends far fewer requests than one call per chunk. `splitter=` takes any `str -> list[str]` function; the default is a sentence splitter.

## White box (NNsight)

```python
from llm import WhiteBoxModel

wb = WhiteBoxModel("meta-llama/Llama-3.1-70B-Instruct")            # remote=True -> NDIF
hs = wb.hidden_states("An insect usually has a small size.")      # hs.hidden_states: (n_layers, seq, hidden)
hs.mean_pool(), hs.last_token(), hs.tokens, hs.next_token
many = wb.hidden_states_many(sentences, layers=range(0, 80, 4), batch_size=8)
gen = wb.generate(wb.format_chat(prompt), max_new_tokens=50, last_token_only=True)
gen.text, gen.new_token_states()                                  # (n_layers, n_new, hidden)
```

For local runs, use `WhiteBoxModel(name, remote=False, device_map="cpu")`.

## Adding a provider (e.g. Anthropic)

1. Subclass `LLMProvider` and implement `chat`: turn a `ChatRequest` into the native call, and the native reply into a `ChatResponse`. For Anthropic, that means moving the system message into `system=`.
2. Optionally implement `submit_batch` / `get_batch` / `batch_results`. Anthropic's Message Batches API takes the requests inline, with no file, and keys them by `custom_id`, which matches `ChatRequest.id`.
3. `register_provider("anthropic", lambda **kw: AnthropicProvider(**kw))` in `providers/__init__.py`.

`generate_many`, `run_batch`, `wait_for_batch` and the rest then work with `provider="anthropic"` without any changes.
