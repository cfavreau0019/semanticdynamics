# personas

Persona sets: JSON files of people (demographics, a profile, a short description) that can be supplied to prompts as an optional input.

```
personas/
  personas.py            PersonaSet (load, validate, look up, filter, sample), flatten, get_path
  tarot_personas.json    100 tarot-app users
```

## Using a set

```python
import random
from personas import PersonaSet

people = PersonaSet.load("tarot_personas")           # by name, or a path to any JSON file
maya = people.get(1)                                  # by persona_number, id, or full name
gen_z = people.filter(**{"demographics.generation": "Gen Z"})
seniors = people.filter(lambda p: p["demographics"]["age"] >= 65)
assigned = people.sample(random.Random(0), 500)                 # seeded, with replacement
balanced = people.sample(random.Random(0), 500, mode="cycle")   # every persona equally often
people.to_frame()                                     # flat DataFrame: demographics.age, tarot_profile.experience_level, ...
```

A persona is a plain dict, exactly the JSON record. Prompt templates read fields directly:

```jinja
{% if persona %}
- Name: {{ persona.name.first }}
- Age: {{ persona.demographics.age }}
{% endif %}
```

**The template decides what the model is told about a person**, and that choice is versioned with the template. The records include sensitive attributes: race/ethnicity, religion, political party, sexual orientation, income. `celtic_cross_v2` deliberately uses only first name, age, occupation, tarot experience and motivation. Create a new template version to disclose more, or different, fields.

## Generating readings with personas

```bash
python -m data_generation.generate_tarot generate --n 200 --personas tarot_personas
python -m data_generation.generate_tarot generate --n 200 --personas tarot_personas --persona-sampling cycle
python -m data_generation.generate_tarot generate --n 20 --personas tarot_personas --persona 7 --persona "Maya Delgado"
```

- **One persona per draw.** Each card draw is assigned one persona. With `--personas`, the default template becomes `celtic_cross_v2`.
- **Draws don't depend on personas.** Personas are sampled after the draws, so a given `--seed` produces the same draws with or without them.
- **What gets stored:**
  - the persona on each input (`inputs.persona_id`);
  - a snapshot of every persona used, with a content hash (the `personas` table);
  - the set's name and file hash (`runs.config`).

## File format

A persona set is a JSON list of objects (or `{"personas": [...]}`).
- **Required:** a unique string `id`.
- **Optional but must be unique:** `persona_number`.
- **Everything else is free-form.** A template fails with a clear error if it asks for a field a persona lacks.

`tarot_personas.json` records have:

| Field | Contents |
|---|---|
| `id`, `persona_number` | Identity |
| `name` | `first`, `last`, `full` |
| `demographics` | Age, generation, gender, pronouns, location, education, profession, income, religion, political party, … |
| `tarot_profile` | Experience level, usage frequency, device, motivation, desired features, accessibility needs |
| `description` | A short third-person sketch |
| `notes` | Occasional extra detail |

**Treat persona ids as permanent.** If you change who a persona is, give them a new id. Stored runs keep their own snapshot either way, and the `content_sha256` on each snapshot shows whether a persona was edited between runs.
