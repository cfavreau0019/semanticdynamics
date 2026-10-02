"""prompts: templates (strict rendering), the library (versions, lock), and the shipped templates."""
import json

import pytest

from prompts import (
    DRAFT, LOCKED, MODIFIED, PromptError, PromptLibrary, PromptLockError, PromptTemplate, default_library,
    get_template, parse_id,
)
from prompts.__main__ import main as prompts_cli

DRAW = {"Present": "Death reversed", "Challenge": "The Sun", "Outcome": "Two of Cups"}
PERSONA = {"name": {"first": "Maya"}, "demographics": {"age": 24, "profession": "Junior UX designer"},
           "tarot_profile": {"experience_level": "Beginner", "primary_motivation": "Daily ritual"}}


def make(user="Hello {{ who }}.", required=("who",), optional=(), name="greet", version=1, **kw):
    return PromptTemplate(name=name, version=version, user=user, required=required, optional=optional, **kw)


# ---- template -------------------------------------------------------------------------
def test_render_and_identity():
    t = make(system="Be brief.")
    p = t.render(who="Ann")
    assert (t.id, p.template_id, p.user, p.system) == ("greet_v1", "greet_v1", "Hello Ann.", "Be brief.")
    assert p.messages == [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hello Ann."}]
    assert p.template_sha256 == t.sha256 and p.variables == ("who",)
    assert make().render({"who": "Ann"}).messages == [{"role": "user", "content": "Hello Ann."}]


def test_hash_tracks_text_not_metadata():
    base = make()
    assert make(description="different words").sha256 == base.sha256
    assert make(user="Hi {{ who }}.").sha256 != base.sha256
    assert make(version=2).sha256 != base.sha256
    assert make(user="  Hello {{ who }}.  \n\n").sha256 == base.sha256      # whitespace is normalised


@pytest.mark.parametrize("kwargs, match", [
    ({"user": "Hello {{ who }} and {{ other }}."}, "undeclared variables"),
    ({"required": ("who", "unused")}, "never used"),
    ({"optional": ("who",)}, "both required and optional"),
    ({"user": "   "}, "empty"),
    ({"user": "{% if who %}unclosed"}, "syntax error"),
    ({"name": "Bad Name"}, "Invalid template name"),
    ({"name": "greet_v2"}, "Invalid template name"),
    ({"version": 0}, "positive integer"),
])
def test_malformed_templates_rejected(kwargs, match):
    with pytest.raises(PromptError, match=match):
        make(**kwargs)


def test_strict_rendering():
    t = make(user="Hi {{ who }}{% if extra %} ({{ extra.note }}){% endif %}.", optional=("extra",))
    assert t.render(who="Ann").user == "Hi Ann."
    assert t.render(who="Ann", extra={"note": "vip"}).user == "Hi Ann (vip)."
    with pytest.raises(PromptError, match="missing required"):
        t.render()
    with pytest.raises(PromptError, match="missing required"):
        t.render(who=None)
    with pytest.raises(PromptError, match="unknown variables"):
        t.render(who="Ann", whom="typo")
    with pytest.raises(PromptError, match="could not render"):
        t.render(who="Ann", extra={"wrong_key": 1})          # missing attribute is an error, not a blank


def test_render_from_takes_only_declared_variables():
    t = make()
    assert t.render_from({"who": "Ann", "persona": {"x": 1}, "cards": {}}).user == "Hello Ann."
    with pytest.raises(PromptError, match="missing required"):
        t.render_from({"persona": {}})


def test_filters_and_literal_json():
    t = make(user='{{ cards | kv_lines }}\n\n{{ items | bullets }}\n\n{{ items | numbered }}\n\n'
                  '{% raw %}{"a": {"b": 1}}{% endraw %}', required=("cards", "items"))
    out = t.render(cards={"A": 1, "B": 2}, items=["x", "y"]).user
    assert out == 'A: 1\nB: 2\n\n- x\n- y\n\n1. x\n2. y\n\n{"a": {"b": 1}}'


def test_parse_id():
    assert parse_id("celtic_cross_v12") == ("celtic_cross", 12)
    assert parse_id("celtic_cross") is None


# ---- library --------------------------------------------------------------------------
def write_toml(root, name, version, user, **extra):
    lines = [f'name = "{name}"', f"version = {version}", 'required = ["who"]']
    lines += [f'{k} = "{v}"' for k, v in extra.items()]
    lines += ["[template]", f"user = '''{user}'''"]
    path = root / f"{name}_v{version}.toml"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


@pytest.fixture
def lib_dir(tmp_path):
    write_toml(tmp_path, "greet", 1, "Hello {{ who }}.")
    write_toml(tmp_path, "greet", 2, "Good day, {{ who }}.", changes="more formal")
    (tmp_path / "sub").mkdir()
    write_toml(tmp_path / "sub", "bye", 1, "Bye {{ who }}.")
    return tmp_path


def test_library_loads_and_resolves_versions(lib_dir):
    lib = PromptLibrary(lib_dir)
    assert lib.names() == ["bye", "greet"]
    assert [t.id for t in lib.versions("greet")] == ["greet_v1", "greet_v2"]
    assert lib.get("greet_v1").render(who="A").user == "Hello A."
    assert lib.get("greet").id == "greet_v2"                 # bare name -> latest
    with pytest.raises(KeyError, match="greet_v1"):
        lib.get("greet_v9")
    with pytest.raises(KeyError, match="Available"):
        lib.get("nope")


def test_library_rejects_bad_files(tmp_path):
    write_toml(tmp_path, "greet", 1, "Hello {{ who }}.").rename(tmp_path / "wrong_name.toml")
    with pytest.raises(PromptError, match="file name should be greet_v1.toml"):
        PromptLibrary(tmp_path)
    (tmp_path / "wrong_name.toml").write_text('name = "x"\nversion = 1\nbogus = 1\n[template]\nuser = "hi"')
    with pytest.raises(PromptError, match="unknown keys"):
        PromptLibrary(tmp_path)


def test_lock_workflow(lib_dir):
    lib = PromptLibrary(lib_dir)
    assert {lib.status(t) for t in lib.all()} == {DRAFT} and lib.verify() == []
    assert lib.lock(["greet_v1"]) == ["greet_v1"]
    assert lib.status(lib.resolve("greet_v1")) == LOCKED and lib.status(lib.resolve("greet_v2")) == DRAFT
    assert lib.lock(["greet_v1"]) == []                                       # already locked

    write_toml(lib_dir, "greet", 1, "Hello there {{ who }}.")                 # edit a locked version
    edited = PromptLibrary(lib_dir)
    assert edited.status(edited.resolve("greet_v1")) == MODIFIED
    assert edited.verify() == ["greet_v1: content differs from the lock"]
    with pytest.raises(PromptLockError, match="greet_v3"):                    # tells you which version to create
        edited.get("greet_v1")
    assert edited.get("greet_v2").id == "greet_v2"                            # other versions unaffected
    with pytest.raises(PromptLockError):
        edited.lock()
    assert "greet_v1" in edited.lock(force=True)
    assert PromptLibrary(lib_dir).get("greet_v1").user == "Hello there {{ who }}."


def test_editing_a_draft_is_allowed(lib_dir):
    write_toml(lib_dir, "greet", 2, "Greetings, {{ who }}.")
    assert PromptLibrary(lib_dir).get("greet_v2").render(who="A").user == "Greetings, A."


def test_verify_reports_missing_locked_file(lib_dir):
    PromptLibrary(lib_dir).lock()
    (lib_dir / "greet_v2.toml").unlink()
    assert PromptLibrary(lib_dir).verify() == ["greet_v2: locked but its file is missing"]


def test_diff(lib_dir):
    lib = PromptLibrary(lib_dir)
    d = lib.diff("greet_v1", "greet_v2")
    assert "-Hello {{ who }}." in d and "+Good day, {{ who }}." in d
    assert "identical" in lib.diff("greet_v1", "greet_v1")


# ---- the shipped library --------------------------------------------------------------
def test_shipped_library_is_consistent():
    lib = default_library(reload=True)
    assert lib.verify() == [], "a locked template was edited; create a new version instead"
    assert {"celtic_cross_v1", "celtic_cross_v2", "spread_v1"} <= {t.id for t in lib.all()}
    assert lib.status(lib.resolve("celtic_cross_v1")) == LOCKED


def test_celtic_cross_v1_text_is_the_original_prompt():
    """v1 produced stored datasets: its rendering must never change."""
    expected = ("You're a tarot reader. Can you read my Celtic cross tarot? Give an overview of the reading "
                "after, any thoughts you have. I'll draw the cards:\n\n"
                "Present: Death reversed\nChallenge: The Sun\nOutcome: Two of Cups")
    p = get_template("celtic_cross_v1").render(cards=DRAW)
    assert p.user == expected and p.system is None
    assert get_template("celtic_cross_v1").sha256 == \
        "d89b71944144a779d31695c8fa49a32090f1004223b7dd333adfcd97c9c0af2d"


def test_celtic_cross_v2_persona_is_optional_and_limited():
    t = get_template("celtic_cross_v2")
    assert t.required == ("cards",) and t.optional == ("persona",) and t.based_on == "celtic_cross_v1"
    without = t.render(cards=DRAW).user
    assert "about me" not in without and without.endswith("Outcome: Two of Cups") and "\n\n\n" not in without
    with_p = t.render(cards=DRAW, persona=PERSONA).user
    for expected in ("- Name: Maya", "- Age: 24", "- Occupation: Junior UX designer",
                     "- Experience with tarot: Beginner", "- What I use tarot for: Daily ritual"):
        assert expected in with_p
    assert with_p.index("about me") < with_p.index("I'll draw the cards:") < with_p.index("Present: Death reversed")
    with pytest.raises(PromptError, match="could not render"):
        t.render(cards=DRAW, persona={"name": {"first": "X"}})               # incomplete persona fails loudly


def test_ner_coref_keeps_json_braces():
    out = get_template("ner_coref_v1").render(text="Ann met Bob.").user
    assert '{"text": "John Smith", "start": 0, "end": 10}' in out and "Ann met Bob." in out


# ---- CLI ------------------------------------------------------------------------------
def test_cli(capsys):
    assert prompts_cli(["list"]) == 0
    assert "celtic_cross_v1" in capsys.readouterr().out
    assert prompts_cli(["show", "celtic_cross_v2"]) == 0
    assert "based on:    celtic_cross_v1" in capsys.readouterr().out
    assert prompts_cli(["diff", "celtic_cross_v1", "celtic_cross_v2"]) == 0
    assert "+A little about me" in capsys.readouterr().out
    assert prompts_cli(["render", "celtic_cross_v1", "--vars", json.dumps({"cards": DRAW})]) == 0
    assert "Outcome: Two of Cups" in capsys.readouterr().out
    assert prompts_cli(["verify"]) == 0
    assert prompts_cli(["show", "nope"]) == 2
