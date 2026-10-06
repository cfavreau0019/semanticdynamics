# prompts

Versioned prompt templates, independent of any one task. A template is a text file; rendering it with structured inputs (a card draw, a persona, …) produces the exact messages to send, tagged with the template version and content hash that made them.

```
prompts/
  template.py     PromptTemplate, RenderedPrompt, filters (kv_lines, bullets, numbered, to_json)
  library.py      PromptLibrary: loads the TOML files, resolves versions, enforces the lock
  __main__.py     python -m prompts list | show | diff | render | verify | lock
  library/
    tarot/celtic_cross_v1.toml, celtic_cross_v2.toml, spread_v1.toml
    evaluation/celtic_cross_evaluation_v1.toml, celtic_cross_evaluation_v2.toml, reading_expectations_v1.toml, evaluation_summary_v1.toml
    extraction/ner_coref_v1.toml
    prompts.lock.json          {template_id: content hash} for versions that must not change
```

## Using a template

```python
from prompts import get_template

t = get_template("celtic_cross_v2")            # a pinned version; "celtic_cross" gives the latest
p = t.render(cards=draw, persona=persona)      # persona is optional for this template
p.messages                                     # [{"role": "user", "content": ...}] -> llm.ChatRequest(messages=...)
p.template_id, p.template_sha256               # recorded with every stored prompt
```

Pin a version (`name_vN`) for anything that produces data you keep. The bare name is a convenience for interactive use.

## A template file

```toml
name = "celtic_cross"
version = 2
description = "Celtic Cross reading request that can include a short note about who is asking."
changes = "What changed from v1, and why."
based_on = "celtic_cross_v1"
required = ["cards"]            # must be supplied
optional = ["persona"]          # may be omitted; guard with {% if persona %}

[template]
user = '''
You're a tarot reader. ...
{% if persona %}

A little about me, in case it helps:
- Name: {{ persona.name.first }}
- Age: {{ persona.demographics.age }}
{% endif %}

I'll draw the cards:

{{ cards | kv_lines }}
'''
# system = '''...'''            # optional system message
```

The text is [Jinja2](https://jinja.palletsprojects.com/):
- `{{ variable }}` and `{{ a.b.c }}` insert values.
- `{{ value | filter }}` applies a filter.
- `{% if x %}…{% endif %}` is a conditional block.
- Wrap literal JSON examples in `{% raw %}…{% endraw %}` so their braces are left alone.

Filters turn structured values into text:
- `kv_lines`: mapping → `key: value` lines
- `bullets`: list → `- item`
- `numbered`: list → `1. item`
- `to_json`: any nested value → indented JSON
- your own, via `prompts.register_filter(name, fn)`

**Rendering is strict.** A prompt that silently renders a blank corrupts a dataset, so these all raise `PromptError`:
- a variable used in the text but not declared, or declared but never used (checked when the file loads);
- a missing required variable, or a variable the template doesn't take;
- a missing attribute, such as `persona.demographics.agee`.

Whitespace is normalised:
- leading and trailing whitespace is stripped;
- trailing spaces on each line are removed;
- runs of blank lines collapse to one.

## Versions and iterations

A version's text never changes once it has produced data you keep. To iterate:

1. Copy `celtic_cross_v2.toml` to `celtic_cross_v3.toml`, bump `version`, edit the text, and write what changed and why in `changes`.
2. Iterate freely. An unlocked version is a **draft**: usable, and flagged as such.
3. Review it against the previous one:
   ```bash
   python -m prompts diff celtic_cross_v2 celtic_cross_v3
   ```
4. Compare them on the *same* inputs. The summary reports, per template, the valid rate and the first-try valid rate:
   ```bash
   python -m data_generation.generate_tarot generate --n 50 --template celtic_cross_v2 --template celtic_cross_v3
   ```
5. Lock it once you keep its data:
   ```bash
   python -m prompts lock celtic_cross_v3
   ```

After locking, the library refuses to load that version if its text differs from the lock. Either put the new wording in the next version, or use `python -m prompts lock --force <id>` if the version produced nothing you keep. `python -m prompts verify` exits non-zero if any locked template was edited, and the test suite checks this too.

`celtic_cross_v1` is locked and reproduces the prompts already stored in `data/generations/` byte for byte.

## Command line

```bash
python -m prompts list
python -m prompts show celtic_cross_v2
python -m prompts diff celtic_cross_v1 celtic_cross_v2
python -m prompts render celtic_cross_v1 --vars '{"cards": {"Present": "Death"}}'
python -m prompts verify
python -m prompts lock [ID ...] [--force]
```

## How it connects to stored data

`data_generation` records, for every run:

| Table | What it holds |
|---|---|
| `prompt_templates` | The template text, variables, description, changes and hash of each version the run used |
| `prompts.template` | The template id of each rendered prompt |
| `prompts.template_sha256` | The exact content hash, linking to `prompt_templates` |
| `prompts.messages`, `prompts.prompt_sha256` | The rendered messages and their hash |

So a stored reading can always be traced to the precise wording that produced it, even for a draft that was later edited.

## Adding templates for another task

1. Add TOML files under `prompts/library/<area>/`.
2. In the `data_generation` application, return the variables your templates use from `template_variables(payload, persona)`. Each template takes only the variables it declares, so one application can serve templates with different needs.
