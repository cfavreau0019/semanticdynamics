"""personas: loading, validation, lookup, filtering, sampling, and the shipped tarot set."""
import json
import random

import pytest

from personas import PersonaSet, content_sha256, display_name, flatten, get_path
from prompts import get_template


def person(i, **demo):
    return {"id": f"id-{i}", "persona_number": i, "name": {"first": f"F{i}", "last": "L", "full": f"F{i} L"},
            "demographics": {"age": 20 + i, "generation": "Gen Z" if i % 2 else "Millennial", **demo}}


@pytest.fixture
def people():
    return PersonaSet("test", [person(i) for i in range(1, 6)])


def test_load_by_name_and_path(tmp_path):
    shipped = PersonaSet.load("tarot_personas")
    assert len(shipped) == 100 and shipped.name == "tarot_personas" and len(shipped.sha256) == 64
    f = tmp_path / "mine.json"
    f.write_text(json.dumps({"personas": [person(1)]}), encoding="utf-8")       # wrapped form also accepted
    assert len(PersonaSet.load(f)) == 1 and PersonaSet.load(f).name == "mine"
    with pytest.raises(FileNotFoundError, match="tarot_personas"):
        PersonaSet.load("nope")
    f.write_text('{"not": "a list"}', encoding="utf-8")
    with pytest.raises(ValueError, match="JSON list"):
        PersonaSet.load(f)


@pytest.mark.parametrize("personas, match", [
    ([person(1), person(1)], "duplicate persona ids"),
    ([{"name": "no id"}], "non-empty string `id`"),
    ([person(1), {**person(2), "persona_number": 1}], "duplicate persona_number"),
])
def test_validation(personas, match):
    with pytest.raises(ValueError, match=match):
        PersonaSet("bad", personas)


def test_get_by_id_number_or_name(people):
    assert people.get("id-3") is people.get(3) is people.get("3") is people.get("f3 l")
    with pytest.raises(KeyError):
        people.get("nobody")


def test_select_and_filter(people):
    assert [p["id"] for p in people.select([5, "id-1"])] == ["id-5", "id-1"]
    gen_z = people.filter(**{"demographics.generation": "Gen Z"})
    assert [p["persona_number"] for p in gen_z] == [1, 3, 5]
    older = people.filter(lambda p: p["demographics"]["age"] >= 24, **{"demographics.generation": "Millennial"})
    assert [p["persona_number"] for p in older] == [4]


def test_sampling(people):
    a = people.sample(random.Random(1), 50)
    assert a == people.sample(random.Random(1), 50) and a != people.sample(random.Random(2), 50)
    cycled = [p["persona_number"] for p in people.sample(random.Random(0), 12, mode="cycle")]
    assert cycled == [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2]
    with pytest.raises(ValueError, match="mode"):
        people.sample(random.Random(0), 1, mode="weird")
    with pytest.raises(ValueError, match="empty"):
        people.filter(lambda p: False).sample(random.Random(0), 1)


def test_helpers(people):
    p = people.get(1)
    assert get_path(p, "demographics.age") == 21 and get_path(p, "demographics.nope", "d") == "d"
    assert flatten(p)["name.first"] == "F1" and flatten(p)["demographics.age"] == 21
    assert display_name(p) == "F1 L" and display_name({"name": "Plain"}) == "Plain"
    assert content_sha256(p) != content_sha256({**p, "id": "other"})
    row = people.row(p)
    assert (row["persona_id"], row["persona_set"], row["name"], row["payload"]) == ("id-1", "test", "F1 L", p)


def test_to_frame(people):
    df = people.to_frame()
    assert len(df) == 5 and {"id", "name.full", "demographics.age"} <= set(df.columns)


def test_every_shipped_persona_renders_in_the_persona_template():
    """celtic_cross_v2 reads specific persona fields; all 100 personas must have them."""
    template = get_template("celtic_cross_v2")
    for persona in PersonaSet.load("tarot_personas"):
        text = template.render(cards={"Present": "Death"}, persona=persona).user
        assert f"- Name: {persona['name']['first']}" in text
